#!/usr/bin/env python3
"""Fill fresh selected-layer SRP5920 feature caches.

This cache is intentionally separate from the earlier compact selected-layer
cache. It mirrors the dense layer-sweep transfer feature protocol:

    selected dense layer -> flatten aggregation -> deterministic SRP to 5920

The exact selected layer is fixed from the dense layer-sweep best-on-shared
selection table. The historical source cache is not used here because the true
target-adaptation rerun should use fresh SRP5920 features with current metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import types
from pathlib import Path

import _paths  # noqa: F401
from _paths import CACHE_DIR, LAYER_SWEEP_ROOT, SHARE_ROOT

import numpy as np
import pandas as pd
from PIL import Image

from config import CSTIM_HDF5_ROOT, PAPER_ROOT
from cstims.datasets.deepvision import DeepVisionBenchmark
from layers_config import MODEL_SOURCE, get_layer_set
from multilayer_extractor import MultiLayerExtractor
from srp_utils import (
    FEATURE_PROTOCOL,
    SRPProjectorCache,
    SRP_SEED,
    SRP_TARGET_DIM,
    cached_layer_current,
    metadata_arrays,
)


CSTIM_SETS = ["all_models", "architecture", "dataset", "sota", "training_objective"]
BASELINE_SET = "vicco"
STIMULUS_SETS = [*CSTIM_SETS, BASELINE_SET]
SELECTION_CSV = LAYER_SWEEP_ROOT / "results" / "mrsa_dense_layer_selection_transfer.csv"
LOCAL_SELECTED_CACHE_DIR = CACHE_DIR / "selected_layer_features_srp5920"
LOCAL_FEATURE_DIR = LOCAL_SELECTED_CACHE_DIR / "features"
LOCAL_DV_FEATURE_DIR = LOCAL_SELECTED_CACHE_DIR / "dv_features"
DV_CACHE_ROOT = SHARE_ROOT / "01_brain_model_alignment" / "cache_or_heavy" / "brain_data"
LABSHARE_CSTIM_HDF5_ROOT = Path(
    "/data/labshare/_stachelschwein/SSD/jroth/final_cstims_hdf5_files"
)
DEFAULT_LAYERS_PER_CHUNK = 32


def install_torchvision_models_utils_compat() -> None:
    """Provide the old torchvision.models.utils import expected by robustness."""
    if "torchvision.models.utils" in sys.modules:
        return
    try:
        from torch.hub import load_state_dict_from_url
    except Exception:
        return
    module = types.ModuleType("torchvision.models.utils")
    module.load_state_dict_from_url = load_state_dict_from_url
    sys.modules["torchvision.models.utils"] = module


def stable_layer_seed(model: str, layer: str, *, base_seed: int = SRP_SEED) -> int:
    digest = hashlib.blake2b(f"{model}::{layer}".encode("utf-8"), digest_size=4).digest()
    return (int.from_bytes(digest, byteorder="little", signed=False) + int(base_seed)) % (
        2**31 - 1
    )


def load_best_shared_selections() -> pd.DataFrame:
    df = pd.read_csv(SELECTION_CSV)
    rows = df[
        df["selection_rule"].eq("best_on_shared")
        & df["selection_model_set"].eq("deepvision_shared")
        & df["eval_target"].eq("shared")
    ].copy()
    rows = rows[["subject", "model", "display_name", "selected_layer"]].drop_duplicates(
        ["subject", "model"]
    )
    return rows.rename(columns={"selected_layer": "layer"})


def dense_layer_specs_for_rows(rows: pd.DataFrame) -> dict[str, list[tuple[str, str]]]:
    dense = get_layer_set("dense")
    by_model: dict[str, list[tuple[str, str]]] = {}
    seen: dict[str, set[str]] = {}
    for row in rows.itertuples(index=False):
        specs = dict(dense[row.model])
        if row.layer not in specs:
            raise KeyError(f"{row.model}: selected layer {row.layer!r} not in dense layer set")
        by_model.setdefault(row.model, [])
        seen.setdefault(row.model, set())
        if row.layer in seen[row.model]:
            continue
        by_model[row.model].append((row.layer, specs[row.layer]))
        seen[row.model].add(row.layer)
    return by_model


def load_existing_payload(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=True) as z:
            return {key: z[key] for key in z.files}
    except Exception:
        return {}


def layer_current(path: Path, layer: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=True) as z:
            return cached_layer_current(z, layer, target_dim=SRP_TARGET_DIM)
    except Exception:
        return False


def atomic_savez_compressed(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def load_cstim_images(group: str) -> tuple[list[Image.Image], list[str]]:
    folder_group = group
    if group == "architecture":
        folder_group = "dataset"
    elif group == "dataset":
        folder_group = "architecture"

    if folder_group == "vicco":
        img_dir = CSTIM_HDF5_ROOT / "shared_vicco"
    else:
        img_dir = CSTIM_HDF5_ROOT / folder_group

    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not img_files and folder_group != "vicco":
        fallback_dirs = (
            SHARE_ROOT
            / "00_stimulus_selection"
            / "decision_checks"
            / "selection_evaluation"
            / "results"
            / folder_group
            / "images",
            PAPER_ROOT / "00_selection_evaluation" / "data" / folder_group / "images",
            PAPER_ROOT / "00_selection_evaluation" / "results" / folder_group / "images",
        )
        for fallback_dir in fallback_dirs:
            img_files = sorted(list(fallback_dir.glob("*.jpg")) + list(fallback_dir.glob("*.png")))
            if img_files:
                break
    if not img_files and folder_group == "vicco":
        img_dir = LABSHARE_CSTIM_HDF5_ROOT / "shared_vicco"
        img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not img_files:
        raise FileNotFoundError(f"No images found for stimulus group {group!r}")

    images = []
    for path in img_files:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images, [path.name for path in img_files]


def load_deepvision_paths(subject: str) -> tuple[list[Path], list[str]]:
    bench = DeepVisionBenchmark(
        cache_root=DV_CACHE_ROOT,
        subject=subject,
        voxel_set="visual",
        input_source="finalinterp",
        image_set="unique",
        n_jobs=1,
    )
    paths = [Path(p) for p in bench.stimulus_data["image_path"].tolist()]
    names = bench.stimulus_data["image_name"].astype(str).tolist()
    return paths, names


def load_batch_items(items: list[Image.Image | Path]) -> list[Image.Image]:
    images = []
    for item in items:
        if isinstance(item, Image.Image):
            images.append(item)
        else:
            with Image.open(item) as img:
                images.append(img.convert("RGB"))
    return images


def extract_items_srp5920(
    extractor: MultiLayerExtractor,
    items: list[Image.Image | Path],
    *,
    model: str,
    batch_size: int,
    progress_label: str = "",
    progress_every: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    layer_names = [name for name, _agg in extractor.layers]
    out = {name: [] for name in layer_names}
    meta_by_layer = {}
    srp_caches = {
        name: SRPProjectorCache(target_dim=SRP_TARGET_DIM, seed=stable_layer_seed(model, name))
        for name in layer_names
    }
    last_progress = 0
    for start in range(0, len(items), batch_size):
        batch_items = items[start : start + batch_size]
        batch = load_batch_items(batch_items)
        raw = extractor.extract(batch)
        for name in layer_names:
            reduced, meta = srp_caches[name].transform(raw[name])
            if name not in meta_by_layer:
                meta_by_layer[name] = meta
            elif int(meta_by_layer[name]["original_feature_dim"]) != int(
                meta["original_feature_dim"]
            ):
                raise RuntimeError(f"Inconsistent feature dim for {model}/{name}")
            out[name].append(np.ascontiguousarray(reduced, dtype=np.float32))
        done = min(start + batch_size, len(items))
        if progress_every > 0 and done - last_progress >= progress_every:
            label = f"{progress_label} " if progress_label else ""
            print(f"    {label}{done}/{len(items)} images", flush=True)
            last_progress = done
        del raw, batch
    features = {
        name: np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)
        for name, chunks in out.items()
    }
    return features, meta_by_layer


def needed_cstim_specs(
    model: str,
    stimulus_set: str,
    specs: list[tuple[str, str]],
    *,
    overwrite: bool,
) -> list[tuple[str, str]]:
    out = []
    path = LOCAL_FEATURE_DIR / model / f"{stimulus_set}.npz"
    for layer, agg in specs:
        if layer_current(path, layer) and not overwrite:
            continue
        out.append((layer, agg))
    return out


def needed_dv_specs(
    subject: str,
    model: str,
    specs: list[tuple[str, str]],
    *,
    overwrite: bool,
) -> list[tuple[str, str]]:
    out = []
    path = LOCAL_DV_FEATURE_DIR / subject / f"{model}.npz"
    for layer, agg in specs:
        if layer_current(path, layer) and not overwrite:
            continue
        out.append((layer, agg))
    return out


def merge_specs(spec_groups: list[list[tuple[str, str]]]) -> list[tuple[str, str]]:
    merged = []
    seen = set()
    for specs in spec_groups:
        for layer, agg in specs:
            if layer in seen:
                continue
            merged.append((layer, agg))
            seen.add(layer)
    return merged


def dense_chunk_context_specs(
    model: str,
    wanted_specs: list[tuple[str, str]],
    *,
    layers_per_chunk: int,
) -> list[tuple[str, str]]:
    """Return dense-layer chunk context matching the original stream run.

    The dense layer-sweep stream extracted layers in fixed dense-order chunks,
    not one selected layer at a time. Some networks have in-place operations, so
    exact reproduction requires the same extractor context. We still save only
    the requested selected layers.
    """
    if not wanted_specs:
        return []
    dense_specs = get_layer_set("dense")[model]
    dense_names = [name for name, _agg in dense_specs]
    wanted = {name for name, _agg in wanted_specs}
    include = []
    seen = set()
    for name in dense_names:
        if name not in wanted:
            continue
        idx = dense_names.index(name)
        start = (idx // int(layers_per_chunk)) * int(layers_per_chunk)
        stop = min(start + int(layers_per_chunk), len(dense_specs))
        for spec in dense_specs[start:stop]:
            if spec[0] in seen:
                continue
            include.append(spec)
            seen.add(spec[0])
    return include


def save_features(
    path: Path,
    features: dict[str, np.ndarray],
    meta_by_layer: dict[str, dict],
    filenames: list[str],
    specs: list[tuple[str, str]],
    *,
    overwrite: bool,
) -> None:
    payload = {} if overwrite else load_existing_payload(path)
    for layer, agg in specs:
        arr = features[layer]
        payload[layer] = arr
        payload.update(metadata_arrays(layer, meta_by_layer[layer]))
        safe = "".join(c if c.isalnum() else "_" for c in str(layer)).strip("_")
        payload[f"_aggregation__{safe}"] = np.array(agg)
    payload["image_filenames"] = np.array(filenames, dtype=object)
    payload["_feature_protocol"] = np.array(FEATURE_PROTOCOL)
    payload["_srp_target_dim"] = np.array(SRP_TARGET_DIM, dtype=np.int32)
    atomic_savez_compressed(path, payload)


def main() -> None:
    install_torchvision_models_utils_compat()
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="all")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--stim-sets", nargs="*", default=STIMULUS_SETS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--progress-every", type=int, default=1024)
    parser.add_argument("--layers-per-chunk", type=int, default=DEFAULT_LAYERS_PER_CHUNK)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    selections = load_best_shared_selections()
    if args.subject != "all":
        selections = selections[selections["subject"].eq(args.subject)].copy()
    if args.models:
        selections = selections[selections["model"].isin(args.models)].copy()
    if selections.empty:
        raise RuntimeError("No selected subject/model/layer rows after filtering")

    specs_by_model = dense_layer_specs_for_rows(selections)
    cstim_images: dict[str, tuple[list[Image.Image], list[str]]] = {}
    for stim_set in args.stim_sets:
        if stim_set in STIMULUS_SETS:
            images, filenames = load_cstim_images(stim_set)
            cstim_images[stim_set] = (images, filenames)
            print(f"Loaded {stim_set}: {len(images)} images", flush=True)

    total_missing = 0
    for model in sorted(specs_by_model):
        if model not in MODEL_SOURCE:
            print(f"[skip] unknown model source: {model}", flush=True)
            continue

        model_rows = selections[selections["model"].eq(model)].copy()
        base_specs = specs_by_model[model]
        missing_cstim = {
            stim_set: needed_cstim_specs(
                model, stim_set, base_specs, overwrite=args.overwrite
            )
            for stim_set in cstim_images
        }
        missing_dv = {
            subject: needed_dv_specs(
                subject,
                model,
                [(row.layer, dict(base_specs)[row.layer])],
                overwrite=args.overwrite,
            )
            for subject, row in model_rows.set_index("subject").iterrows()
        }
        cstim_specs_to_extract = merge_specs(list(missing_cstim.values()))
        dv_specs_to_extract = merge_specs(list(missing_dv.values()))
        specs_to_extract = merge_specs([cstim_specs_to_extract, dv_specs_to_extract])
        cstim_context_specs = dense_chunk_context_specs(
            model,
            cstim_specs_to_extract,
            layers_per_chunk=args.layers_per_chunk,
        )
        dv_context_specs = dense_chunk_context_specs(
            model,
            dv_specs_to_extract,
            layers_per_chunk=args.layers_per_chunk,
        )
        missing_count = sum(len(v) for v in missing_cstim.values()) + sum(
            len(v) for v in missing_dv.values()
        )
        total_missing += missing_count
        if not specs_to_extract:
            print(f"[cached] {model}", flush=True)
            continue

        print(
            f"\n=== {model} ({MODEL_SOURCE[model]}): "
            f"{len(specs_to_extract)} selected layers, {missing_count} missing SRP entries ===",
            flush=True,
        )
        print("  " + ", ".join(f"{name}:{agg}" for name, agg in specs_to_extract), flush=True)
        t_model = time.time()
        if cstim_specs_to_extract:
            extractor = MultiLayerExtractor(
                model,
                MODEL_SOURCE[model],
                cstim_context_specs,
                device=args.device,
            )
            try:
                print(
                    "  CSTIM extraction context: "
                    f"{len(cstim_context_specs)} dense-chunk layers",
                    flush=True,
                )
                for stim_set, specs in missing_cstim.items():
                    if not specs:
                        continue
                    images, filenames = cstim_images[stim_set]
                    t0 = time.time()
                    feats, meta = extract_items_srp5920(
                        extractor,
                        images,
                        model=model,
                        batch_size=args.batch_size,
                        progress_label=stim_set,
                        progress_every=0,
                    )
                    out_path = LOCAL_FEATURE_DIR / model / f"{stim_set}.npz"
                    save_features(
                        out_path, feats, meta, filenames, specs, overwrite=args.overwrite
                    )
                    dims = ", ".join(f"{layer}={feats[layer].shape[1]}" for layer, _agg in specs)
                    print(
                        f"  {stim_set:<22} {len(images):4d} imgs "
                        f"{time.time() - t0:6.1f}s [{dims}]",
                        flush=True,
                    )
            finally:
                extractor.free()

        dv_subjects = {subject: specs for subject, specs in missing_dv.items() if specs}
        if dv_subjects:
            extractor = MultiLayerExtractor(
                model,
                MODEL_SOURCE[model],
                dv_context_specs,
                device=args.device,
            )
            try:
                print(
                    "  DeepVision layers: "
                    f"{', '.join(layer for layer, _agg in dv_specs_to_extract)} "
                    f"({len(dv_subjects)} subjects); context={len(dv_context_specs)} dense-chunk layers",
                    flush=True,
                )
                for subject, specs in dv_subjects.items():
                    paths, filenames = load_deepvision_paths(subject)
                    t0 = time.time()
                    feats, meta = extract_items_srp5920(
                        extractor,
                        paths,
                        model=model,
                        batch_size=args.batch_size,
                        progress_label=f"{subject} unique",
                        progress_every=args.progress_every,
                    )
                    out_path = LOCAL_DV_FEATURE_DIR / subject / f"{model}.npz"
                    save_features(
                        out_path,
                        feats,
                        meta,
                        filenames,
                        specs,
                        overwrite=args.overwrite,
                    )
                    dims = ", ".join(
                        f"{layer}={feats[layer].shape[1]}" for layer, _agg in specs
                    )
                    print(
                        f"  {subject:<8} unique {len(paths):4d} imgs "
                        f"{time.time() - t0:6.1f}s [{dims}]",
                        flush=True,
                    )
            finally:
                extractor.free()
        print(f"  model done in {time.time() - t_model:.1f}s", flush=True)

    print(f"\nDone. Missing SRP entries considered: {total_missing}", flush=True)


if __name__ == "__main__":
    main()
