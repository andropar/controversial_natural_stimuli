from __future__ import annotations

import csv
import json
import logging
import os
import pickle
import tarfile
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from cstims.paths import model_list_csv

_LOG = logging.getLogger(__name__)

__all__ = [
    "FeatureShardSlice",
    "build_selected_image_records",
    "load_natural_features_with_metadata",
    "max_images_for_ram",
]


DEFAULT_MODEL_CSV = model_list_csv()


def model_list(csv_path: Path = DEFAULT_MODEL_CSV) -> List[str]:
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        models, layers = [], []
        for row in reader:
            models.append(row["model"])
            layers.append(row["layer"])
        return models, layers
    return []

@dataclass(frozen=True)
class FeatureShardSlice:
    tar_path: Path
    start_index: int
    end_index: int
    failed_images: Tuple[str, ...] = ()
    summary_path: Optional[Path] = None

    @property
    def count(self) -> int:
        return self.end_index - self.start_index


def _sanitize_layer_tag(raw: str | int | None) -> str:
    if raw is None:
        raise KeyError("Missing layer identifier")
    value = str(raw).strip()
    if not value:
        raise KeyError("Empty layer identifier")
    value = value.replace("/", "_").replace(os.sep, "_")
    value = value.replace(".", "_").replace(" ", "_")
    return value


def _compute_shard_offsets(shard_slices: Sequence[FeatureShardSlice]) -> List[int]:
    offsets = [0]
    for shard in shard_slices:
        offsets.append(offsets[-1] + shard.count)
    return offsets


def _available_image_names(shard: FeatureShardSlice) -> List[str]:
    names = _iter_valid_image_members(shard.tar_path)
    if shard.failed_images:
        failed = set(shard.failed_images)
        names = [name for name in names if name not in failed]
    return names


def build_selected_image_records(
    global_indices: Sequence[int],
    shard_slices: Sequence[FeatureShardSlice],
) -> List[dict]:
    if not global_indices:
        return []
    records: List[dict] = []
    offsets = _compute_shard_offsets(shard_slices)
    global_list = [int(idx) for idx in global_indices]
    placements: Dict[int, Tuple[int, int]] = {}
    per_shard: Dict[int, List[Tuple[int, int]]] = {}

    for idx in global_list:
        shard_pos = bisect_right(offsets, idx) - 1
        if shard_pos < 0 or shard_pos >= len(shard_slices):
            raise IndexError(f"Selected index {idx} out of range for natural subset")
        shard = shard_slices[shard_pos]
        local = idx - shard.start_index
        if local < 0 or local >= shard.count:
            raise IndexError(f"Local index {local} invalid for shard {shard.tar_path}")
        placements[idx] = (shard_pos, local)
        per_shard.setdefault(shard_pos, []).append((idx, local))

    image_cache: Dict[int, List[str]] = {}
    for shard_pos, pairs in per_shard.items():
        shard = shard_slices[shard_pos]
        names = _available_image_names(shard)
        image_cache[shard_pos] = names
        for _, local in pairs:
            if local >= len(names):
                raise IndexError(
                    f"Shard {shard.tar_path} provides only {len(names)} images;"
                    f" requested local index {local}"
                )

    for idx in global_list:
        shard_pos, local = placements[idx]
        shard = shard_slices[shard_pos]
        image_name = image_cache[shard_pos][local]
        record = {
            "global_index": int(idx),
            "shard_index": shard_pos,
            "local_index": int(local),
            "shard_start_index": int(shard.start_index),
            "tar_path": str(shard.tar_path),
            "image_name": image_name,
        }
        if shard.summary_path is not None:
            record["summary_path"] = str(shard.summary_path)
        records.append(record)

    return records


def _list_subset_tars(subset_root: Path) -> List[Path]:
    return sorted(subset_root.rglob("*.tar"))


def _resolve_feature_dir(
    tar_path: Path, subset_root: Path, output_root: Optional[Path]
) -> Path:
    if output_root is None:
        return tar_path.parent
    try:
        rel = tar_path.relative_to(subset_root)
    except ValueError:
        rel = Path(tar_path.name)
    return output_root / rel.parent


def _candidate_feature_paths(
    stem: str, model_name: str, layer_tag: str, base_dir: Path
) -> Iterator[Path]:
    primary = base_dir / f"{stem}.{model_name}.{layer_tag}.npy"
    if primary.exists():
        yield primary
        return
    yield from sorted(base_dir.glob(f"{stem}.{model_name}.*.npy"))


def _load_model_array(feature_fp: Path) -> np.ndarray:
    arr = np.load(feature_fp)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.astype(np.float16, copy=False)


def _read_failed_images(summary_fp: Path) -> List[str]:
    if not summary_fp.exists():
        return []
    try:
        payload = json.loads(summary_fp.read_text())
    except Exception as exc:  # pragma: no cover
        _LOG.warning("Failed to parse %s: %s", summary_fp, exc)
        return []
    details = payload.get("failed_image_details", [])
    return [str(entry.get("image")) for entry in details if entry.get("image")]


def _iter_valid_image_members(tar_path: Path) -> List[str]:
    with tarfile.open(tar_path, "r:*", ignore_zeros=True) as tf:
        names = tf.getnames()
    name_set = set(names)
    images = [n for n in names if n.endswith(".jpg") and (n[:-4] + ".json") in name_set]
    return images


def _read_layer_map(
    model_csv: Path, model_names: Sequence[str]
) -> Dict[str, str | int]:
    with model_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = {row.get("model"): row for row in reader if row.get("model")}

    layer_map: Dict[str, str | int] = {}
    for name in model_names:
        if name not in rows:
            raise KeyError(f"Model '{name}' not found in {model_csv}")
        row = rows[name]
        layer = row.get("layer") or row.get("layer_uid")
        if layer is None or not str(layer).strip():
            raise KeyError(
                f"Layer identifier missing for model '{name}' in {model_csv}"
            )
        layer_map[name] = layer
    return layer_map


@dataclass(frozen=True)
class _ShardLoadSpec:
    tar_path: Path
    base_dir: Path
    global_start: int
    global_end: int
    shard_start_local: int
    shard_end_local: int


def _collect_load_specs(
    subset_root: Path,
    model_names: Sequence[str],
    layer_map: Dict[str, str | int],
    *,
    output_root: Optional[Path],
    max_images: Optional[int],
    offset: int,
    limit: Optional[int],
) -> List[_ShardLoadSpec]:
    load_specs: List[_ShardLoadSpec] = []
    global_idx = 0
    loaded_count = 0

    for tar_path in _list_subset_tars(subset_root):
        if max_images is not None and global_idx >= max_images:
            break
        if limit is not None and loaded_count >= limit:
            break

        base_dir = _resolve_feature_dir(tar_path, subset_root, output_root)
        if not base_dir.exists():
            _LOG.debug("Skipping %s: feature dir %s missing", tar_path, base_dir)
            continue

        stem = tar_path.stem
        has_features = False
        for model_name in model_names:
            layer_tag = _sanitize_layer_tag(layer_map.get(model_name))
            for candidate in _candidate_feature_paths(
                stem, model_name, layer_tag, base_dir
            ):
                if candidate.exists():
                    has_features = True
                    break
            if has_features:
                break

        if not has_features:
            continue

        expected = 0
        for model_name in model_names:
            layer_tag = _sanitize_layer_tag(layer_map.get(model_name))
            for candidate in _candidate_feature_paths(
                stem, model_name, layer_tag, base_dir
            ):
                if candidate.exists():
                    try:
                        arr = np.load(candidate)
                        expected = arr.shape[0]
                        break
                    except Exception:
                        pass
            if expected > 0:
                break

        if expected == 0:
            continue

        shard_end = global_idx + expected
        if shard_end <= offset:
            global_idx = shard_end
            continue

        shard_start_local = max(0, offset - global_idx)
        remaining_limit = None if limit is None else max(0, limit - loaded_count)
        remaining_max = None if max_images is None else max(0, max_images - global_idx)

        shard_take = expected - shard_start_local
        if remaining_limit is not None:
            shard_take = min(shard_take, remaining_limit)
        if remaining_max is not None:
            shard_take = min(shard_take, remaining_max)

        if shard_take <= 0:
            break

        shard_end_local = shard_start_local + shard_take
        start = offset + loaded_count
        end = start + shard_take

        load_specs.append(
            _ShardLoadSpec(
                tar_path=tar_path,
                base_dir=base_dir,
                global_start=start,
                global_end=end,
                shard_start_local=shard_start_local,
                shard_end_local=shard_end_local,
            )
        )

        loaded_count += shard_take
        global_idx += expected

    return load_specs


def _load_single_shard_features(
    spec: _ShardLoadSpec,
    model_names: Sequence[str],
    layer_map: Dict[str, str | int],
) -> Optional[Dict[str, np.ndarray]]:
    tar_path = spec.tar_path
    base_dir = spec.base_dir
    stem = tar_path.stem

    model_arrays: Dict[str, np.ndarray] = {}

    for model_name in model_names:
        layer_tag = _sanitize_layer_tag(layer_map.get(model_name))
        feature_fp = None
        for candidate in _candidate_feature_paths(
            stem, model_name, layer_tag, base_dir
        ):
            if candidate.exists():
                feature_fp = candidate
                break
        if feature_fp is None:
            _LOG.debug("Model %s missing features for %s", model_name, tar_path)
            return None
        model_arrays[model_name] = _load_model_array(feature_fp)

    if not model_arrays:
        return None

    lengths = {name: arr.shape[0] for name, arr in model_arrays.items()}
    expected = next(iter(lengths.values()))
    if any(length != expected for length in lengths.values()):
        _LOG.warning("Row mismatch for %s: %s", tar_path, lengths)
        return None

    return {
        name: arr[spec.shard_start_local : spec.shard_end_local]
        for name, arr in model_arrays.items()
    }


def load_natural_features_with_metadata(
    subset_root: Path,
    model_names: Sequence[str],
    *,
    output_root: Optional[Path] = None,
    max_images: Optional[int] = None,
    model_csv: Path = DEFAULT_MODEL_CSV,
    offset: int = 0,
    limit: Optional[int] = None,
    n_jobs: int = 64,
    preprocessed_dir: Optional[Path] = None,
    layer_names: Sequence[str] = None,
) -> Tuple[Dict[str, np.ndarray], List[FeatureShardSlice]]:
    """
    Load natural stimulus features and associated metadata for a selection of models.

    This function supports two primary modes:
      1. Fast path: If `preprocessed_dir` is provided and contains the appropriate
         pre-consolidated memmap files and metadata, the function loads all features
         directly from these files for each model.
      2. Shard-based loading: Otherwise, the function falls back to loading features
         shard-by-shard (potentially in parallel), reconstructing the feature arrays and
         assembling per-shard metadata.

    Args:
        subset_root (Path): Root directory containing feature shards or subset data.
        model_names (Sequence[str]): List of model names whose features to load.
        layer_names (Sequence[str]): List of layer names whose features to load.
        output_root (Optional[Path], optional): Directory where feature outputs are written.
            If not provided, defaults to None.
        max_images (Optional[int], optional): Maximum number of images to load (across all shards).
            If None, loads all images found.
        model_csv (Path, optional): Path to the model CSV reference file.
        offset (int, optional): Offset into the dataset to begin loading.
        limit (Optional[int], optional): Maximum number of images to load, in addition to any offset.
        n_jobs (int, optional): Number of parallel jobs for shard-based loading.
        preprocessed_dir (Optional[Path], optional): Directory containing pre-consolidated
            .mmap feature files and metadata. If present and valid, fast-path loading is used.

    Returns:
        Tuple[Dict[str, np.ndarray], List[FeatureShardSlice]]: 
            - features: A dictionary mapping from model name to a 2D numpy array of features 
              (shape: [num_images, feature_dim] for each model).
            - shard_slices: List of FeatureShardSlice objects, each providing metadata
              about the location, offset, and other information for each loaded feature shard.

    Notes:
        - The fast path (using `preprocessed_dir`) always loads all available features and
          does not currently support loading with offset, limit, or max_images constraints.
        - In all cases, features are returned as numpy arrays (or possibly memmaps in the fast path).
        - This function is parallelized for performance when n_jobs > 1.

    Raises:
        FileNotFoundError: If files required for fast-path loading are missing.
        Exception: For various file or data access problems during loading.
    """

    # --- START: FAST PATH LOGIC ---
    if preprocessed_dir is not None:
        if layer_names is None:
            all_model_names, all_layer_names = model_list(model_csv)
            layer_names = [all_layer_names[all_model_names.index(name)] for name in model_names]
            
        preprocessed_dir = preprocessed_dir.resolve()
        metadata_path = preprocessed_dir / "_metadata.pkl"

        # Check if the global metadata and all required model .mmap files exist
        can_use_fast_path = metadata_path.exists()
        if can_use_fast_path:
            for model_name, layer_name in zip(model_names, layer_names):
                if not (preprocessed_dir / f"{model_name}.{layer_name}.mmap").exists():
                    can_use_fast_path = False
                    break

        if can_use_fast_path:
            _LOG.info(
                f"FAST PATH: Loading pre-consolidated data from {preprocessed_dir}"
            )

            # Load the global shard metadata
            with open(metadata_path, "rb") as f:
                shard_slices_data = pickle.load(f)
            shard_slices = [FeatureShardSlice(**data) for data in shard_slices_data]

            # Load each model's memmap file
            features: Dict[str, np.ndarray] = {}
            for model_name, layer_name in zip(model_names, layer_names):
                mmap_path = preprocessed_dir / f"{model_name}.{layer_name}.mmap"
                # Load in read-only ('r') mode
                features[model_name] = np.load(mmap_path, mmap_mode="r")

            _LOG.info("Successfully loaded pre-consolidated features.")
            # NOTE: For simplicity, this fast path doesn't currently support
            # offset, limit, or max_images. It loads the entire dataset.
            # This is usually acceptable as memmap access is cheap.
            return features, shard_slices

        else:
            _LOG.warning(
                f"Preprocessed directory specified, but required files are missing. "
                f"Falling back to slow shard-based loading."
            )
    # --- END: FAST PATH LOGIC ---

    subset_root = subset_root.resolve()
    output_root = output_root.resolve() if output_root is not None else None
    model_csv = model_csv.resolve()
    layer_map = _read_layer_map(model_csv, model_names)

    load_specs = _collect_load_specs(
        subset_root,
        model_names,
        layer_map,
        output_root=output_root,
        max_images=max_images,
        offset=offset,
        limit=limit,
    )

    _LOG.info(
        "Loading %d feature shards in parallel (n_jobs=%d)", len(load_specs), n_jobs
    )

    arrays_by_model: Dict[str, List[np.ndarray]] = {name: [] for name in model_names}
    shard_slices: List[FeatureShardSlice] = []
    current_offset = 0

    def register_shard(
        spec: _ShardLoadSpec, shard_arrays: Dict[str, np.ndarray]
    ) -> None:
        nonlocal current_offset
        if not shard_arrays:
            return
        first = next(iter(shard_arrays.values()))
        rows = int(first.shape[0])
        start = current_offset
        end = start + rows
        summary_fp = None
        stem = spec.tar_path.stem
        failed: Tuple[str, ...] = ()
        base_dir = spec.base_dir
        if base_dir.exists():
            summary_candidates = list(base_dir.glob(f"{stem}.*.json"))
            if summary_candidates:
                summary_fp = summary_candidates[0]
                failed = tuple(_read_failed_images(summary_fp))
        shard_slices.append(
            FeatureShardSlice(
                tar_path=spec.tar_path,
                start_index=start,
                end_index=end,
                failed_images=failed,
                summary_path=summary_fp,
            )
        )
        current_offset = end
        for model_name, array in shard_arrays.items():
            arrays_by_model[model_name].append(array)

    if n_jobs == 1:
        for spec in tqdm(load_specs, desc="Loading shards", unit="shard"):
            shard_arrays = _load_single_shard_features(spec, model_names, layer_map)
            if shard_arrays is None:
                continue
            register_shard(spec, shard_arrays)
    else:
        try:
            with tqdm(
                total=len(load_specs), desc="Loading shards", unit="shard"
            ) as pbar:
                for idx, shard_arrays in enumerate(
                    Parallel(
                        n_jobs=n_jobs,
                        backend="loky",
                        batch_size=1,
                        verbose=0,
                        return_as="generator",
                    )(
                        delayed(_load_single_shard_features)(
                            spec, model_names, layer_map
                        )
                        for spec in load_specs
                    )
                ):
                    if shard_arrays is not None:
                        register_shard(load_specs[idx], shard_arrays)
                    pbar.update(1)
        except TypeError:
            with tqdm(
                total=len(load_specs), desc="Loading shards", unit="shard"
            ) as pbar:
                pbar.set_description("Loading shards (progress updates per batch)")
                results = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
                    delayed(_load_single_shard_features)(spec, model_names, layer_map)
                    for spec in load_specs
                )
                for spec, shard_arrays in zip(load_specs, results):
                    if shard_arrays is None:
                        continue
                    register_shard(spec, shard_arrays)
                pbar.update(len(load_specs))

    features: Dict[str, np.ndarray] = {}
    for name in tqdm(list(arrays_by_model.keys()), desc="Converting to numpy arrays", unit="model"):
        parts = arrays_by_model[name]
        if not parts:
            features[name] = np.empty((0, 0), dtype=np.float16)
            del arrays_by_model[name]
            continue
        total_rows = sum(p.shape[0] for p in parts)
        n_cols = parts[0].shape[1]
        result = np.empty((total_rows, n_cols), dtype=np.float16)
        offset = 0
        for p in parts:
            rows = p.shape[0]
            result[offset : offset + rows] = p
            offset += rows
        features[name] = result
        del arrays_by_model[name]

    return features, shard_slices


def _infer_feature_dims_for_models(
    subset_root: Path,
    model_names: Sequence[str],
    *,
    output_root: Optional[Path] = None,
    model_csv: Path = DEFAULT_MODEL_CSV,
) -> Dict[str, int]:
    """Infer per-model feature dimensionality by inspecting existing feature files.

    The loader casts features to float32 in-memory; this helper only determines
    dimensionality D for each model so RAM estimates can be computed as
    n_images * sum(D_model) * 4 bytes.
    """
    subset_root = subset_root.resolve()
    output_root = output_root.resolve() if output_root is not None else None
    model_csv = model_csv.resolve()

    layer_map = _read_layer_map(model_csv, model_names)

    remaining = set(model_names)
    dims_by_model: Dict[str, int] = {}

    for tar_path in _list_subset_tars(subset_root):
        if not remaining:
            break
        base_dir = _resolve_feature_dir(tar_path, subset_root, output_root)
        if not base_dir.exists():
            continue
        stem = tar_path.stem
        for model_name in list(remaining):
            layer_tag = _sanitize_layer_tag(layer_map.get(model_name))
            feature_fp = None
            for candidate in _candidate_feature_paths(
                stem, model_name, layer_tag, base_dir
            ):
                if candidate.exists():
                    feature_fp = candidate
                    break
            if feature_fp is None:
                continue
            try:
                arr = np.load(feature_fp, mmap_mode="r")
                if arr.ndim == 1:
                    dims = int(arr.size)
                else:
                    dims = int(arr.shape[-1])
                dims_by_model[model_name] = dims
                remaining.remove(model_name)
            except Exception:
                continue

    if remaining:
        missing = ", ".join(sorted(remaining))
        raise FileNotFoundError(
            f"Could not infer feature dimensions for models without any feature files: {missing}"
        )

    return dims_by_model


def max_images_for_ram(
    subset_root: Path,
    model_names: Sequence[str],
    max_ram_bytes: int,
    *,
    output_root: Optional[Path] = None,
    model_csv: Path = DEFAULT_MODEL_CSV,
    dtype_bytes: int = 4,
) -> int:
    """Return maximum number of images fitting into the RAM budget.

    Assumes features are represented as float32 in memory by default.
    Returns 0 if budget is insufficient for even a single image across models.
    """
    if max_ram_bytes <= 0:
        return 0
    dims_by_model = _infer_feature_dims_for_models(
        subset_root,
        model_names,
        output_root=output_root,
        model_csv=model_csv,
    )
    per_image_bytes = int(sum(dims_by_model.values()) * dtype_bytes)
    if per_image_bytes <= 0:
        return 0
    return int(max_ram_bytes // per_image_bytes)
