#!/usr/bin/env python3
"""Fill local feature caches for exact best-on-shared selected layers.

The source layer-sweep cache does not contain every exact FX node that appears
in the best-on-shared selection table. This script extracts just those missing
subject/model/layer features and writes them to the local follow-up cache:

    cache_or_heavy/selected_layer_features/features/{model}/{stimulus_set}.npz
    cache_or_heavy/selected_layer_features/dv_features/{subject}/{model}.npz

It never mutates the historical source layer-sweep cache.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import _paths  # noqa: F401
from _paths import CACHE_DIR, LAYER_SWEEP_ROOT, SHARE_ROOT, SOURCE_LAYER_SWEEP_ROOT

import numpy as np
import pandas as pd
from PIL import Image

from config import CSTIM_HDF5_ROOT, PAPER_ROOT
from cstims.datasets.deepvision import DeepVisionBenchmark
from layers_config import MODEL_SOURCE
from multilayer_extractor import MultiLayerExtractor


CSTIM_SETS = ["all_models", "architecture", "dataset", "sota", "training_objective"]
BASELINE_SET = "vicco"
STIMULUS_SETS = [*CSTIM_SETS, BASELINE_SET]
SELECTION_CSV = LAYER_SWEEP_ROOT / "results" / "mrsa_dense_layer_selection_transfer.csv"
LOCAL_SELECTED_CACHE_DIR = CACHE_DIR / "selected_layer_features"
LOCAL_FEATURE_DIR = LOCAL_SELECTED_CACHE_DIR / "features"
LOCAL_DV_FEATURE_DIR = LOCAL_SELECTED_CACHE_DIR / "dv_features"
SOURCE_FEATURE_DIR = SOURCE_LAYER_SWEEP_ROOT / "cache" / "features"
SOURCE_DV_FEATURE_DIR = SOURCE_LAYER_SWEEP_ROOT / "cache" / "dv_features"
DV_CACHE_ROOT = SHARE_ROOT / "01_brain_model_alignment" / "cache_or_heavy" / "brain_data"
LABSHARE_CSTIM_HDF5_ROOT = Path(
    "/data/labshare/_stachelschwein/SSD/jroth/final_cstims_hdf5_files"
)
FEATURE_PROTOCOL = "selected_layer_compact_v1"


def safe_layer_key(layer: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(layer)).strip("_")


def load_best_shared_selections() -> pd.DataFrame:
    df = pd.read_csv(SELECTION_CSV)
    rows = df[
        df["selection_rule"].eq("best_on_shared")
        & df["selection_model_set"].eq("deepvision_shared")
        & df["eval_target"].eq("shared")
    ].copy()
    rows = rows[
        ["subject", "model", "display_name", "selected_layer"]
    ].drop_duplicates(["subject", "model"])
    return rows.rename(columns={"selected_layer": "layer"})


def npz_has_key(path: Path, key: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=True) as z:
            return key in z.files
    except Exception:
        return False


def source_or_local_cstim_paths(model: str, stimulus_set: str) -> list[Path]:
    return [
        LOCAL_FEATURE_DIR / model / f"{stimulus_set}.npz",
        SOURCE_FEATURE_DIR / model / f"{stimulus_set}.npz",
    ]


def source_or_local_dv_paths(subject: str, model: str) -> list[Path]:
    return [
        LOCAL_DV_FEATURE_DIR / subject / f"{model}.npz",
        SOURCE_DV_FEATURE_DIR / subject / f"{model}.npz",
    ]


def has_key_any(paths: list[Path], key: str) -> bool:
    return any(npz_has_key(path, key) for path in paths)


def infer_aggregation(model: str, layer: str) -> str:
    """Infer compact aggregation matching the historical source cache style."""
    if layer.endswith("flatten") or layer.endswith("avgpool") or ".avgpool" in layer:
        return "squeeze"
    if "classifier" in layer:
        return "squeeze"
    if model == "openclip_vit_so400m_14_siglip_webli" and layer.startswith("trunk.blocks."):
        return "mean_patch"
    if any(
        token in model
        for token in (
            "resnet",
            "convnext",
            "vgg",
            "cornet",
            "robustness",
            "vissl",
            "vicreg",
        )
    ):
        return "gap"
    return "cls"


def layer_specs_for_rows(rows: pd.DataFrame) -> dict[str, list[tuple[str, str]]]:
    by_model: dict[str, list[tuple[str, str]]] = {}
    seen: dict[str, set[str]] = {}
    for row in rows.itertuples(index=False):
        by_model.setdefault(row.model, [])
        seen.setdefault(row.model, set())
        if row.layer in seen[row.model]:
            continue
        by_model[row.model].append((row.layer, infer_aggregation(row.model, row.layer)))
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


def metadata_arrays(layer: str, arr: np.ndarray, agg: str) -> dict[str, np.ndarray]:
    safe = safe_layer_key(layer)
    return {
        f"_aggregation__{safe}": np.array(agg),
        f"_feature_protocol__{safe}": np.array(FEATURE_PROTOCOL),
        f"_original_feature_dim__{safe}": np.array(int(arr.shape[1]), dtype=np.int32),
        f"_stored_feature_dim__{safe}": np.array(int(arr.shape[1]), dtype=np.int32),
    }


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


def extract_items(
    extractor: MultiLayerExtractor,
    items: list[Image.Image | Path],
    *,
    batch_size: int,
    progress_label: str = "",
    progress_every: int = 0,
) -> dict[str, np.ndarray]:
    out = {name: [] for name, _agg in extractor.layers}
    last_progress = 0
    for start in range(0, len(items), batch_size):
        batch_items = items[start : start + batch_size]
        batch = load_batch_items(batch_items)
        feats = extractor.extract(batch)
        for name, _agg in extractor.layers:
            arr = feats[name]
            if arr.ndim != 2:
                arr = arr.reshape(arr.shape[0], -1)
            out[name].append(np.ascontiguousarray(arr, dtype=np.float32))
        done = min(start + batch_size, len(items))
        if progress_every > 0 and done - last_progress >= progress_every:
            label = f"{progress_label} " if progress_label else ""
            print(f"    {label}{done}/{len(items)} images", flush=True)
            last_progress = done
    return {
        name: np.concatenate(blocks, axis=0).astype(np.float32)
        for name, blocks in out.items()
    }


def needed_cstim_specs(
    model: str,
    stimulus_set: str,
    specs: list[tuple[str, str]],
    *,
    overwrite: bool,
) -> list[tuple[str, str]]:
    out = []
    for layer, agg in specs:
        source_has = npz_has_key(SOURCE_FEATURE_DIR / model / f"{stimulus_set}.npz", layer)
        local_has = npz_has_key(LOCAL_FEATURE_DIR / model / f"{stimulus_set}.npz", layer)
        if source_has:
            continue
        if local_has and not overwrite:
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
    for layer, agg in specs:
        source_has = npz_has_key(SOURCE_DV_FEATURE_DIR / subject / f"{model}.npz", layer)
        local_has = npz_has_key(LOCAL_DV_FEATURE_DIR / subject / f"{model}.npz", layer)
        if source_has:
            continue
        if local_has and not overwrite:
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


def save_features(
    path: Path,
    features: dict[str, np.ndarray],
    filenames: list[str],
    specs: list[tuple[str, str]],
    *,
    overwrite: bool,
) -> None:
    payload = {} if overwrite else load_existing_payload(path)
    for layer, agg in specs:
        arr = features[layer]
        payload[layer] = arr
        payload.update(metadata_arrays(layer, arr, agg))
    payload["image_filenames"] = np.array(filenames, dtype=object)
    atomic_savez_compressed(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="all")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--stim-sets", nargs="*", default=STIMULUS_SETS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--progress-every", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    selections = load_best_shared_selections()
    if args.subject != "all":
        selections = selections[selections["subject"].eq(args.subject)].copy()
    if args.models:
        selections = selections[selections["model"].isin(args.models)].copy()
    if selections.empty:
        raise RuntimeError("No selected subject/model/layer rows after filtering")

    specs_by_model = layer_specs_for_rows(selections)
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
                model,
                stim_set,
                base_specs,
                overwrite=args.overwrite,
            )
            for stim_set in cstim_images
        }
        missing_dv = {
            subject: needed_dv_specs(
                subject,
                model,
                [(row.layer, infer_aggregation(model, row.layer))],
                overwrite=args.overwrite,
            )
            for subject, row in model_rows.set_index("subject").iterrows()
        }
        specs_to_extract = merge_specs([*missing_cstim.values(), *missing_dv.values()])
        missing_count = sum(len(v) for v in missing_cstim.values()) + sum(
            len(v) for v in missing_dv.values()
        )
        total_missing += missing_count
        if not specs_to_extract:
            print(f"[cached] {model}", flush=True)
            continue

        print(
            f"\n=== {model} ({MODEL_SOURCE[model]}): "
            f"{len(specs_to_extract)} layers, {missing_count} missing cache entries ===",
            flush=True,
        )
        print("  " + ", ".join(f"{name}:{agg}" for name, agg in specs_to_extract), flush=True)
        t_model = time.time()
        extractor = MultiLayerExtractor(
            model,
            MODEL_SOURCE[model],
            specs_to_extract,
            device=args.device,
        )
        try:
            for stim_set, specs in missing_cstim.items():
                if not specs:
                    continue
                images, filenames = cstim_images[stim_set]
                t0 = time.time()
                feats = extract_items(
                    extractor,
                    images,
                    batch_size=args.batch_size,
                    progress_label=f"{stim_set}",
                    progress_every=0,
                )
                out_path = LOCAL_FEATURE_DIR / model / f"{stim_set}.npz"
                save_features(out_path, feats, filenames, specs, overwrite=args.overwrite)
                dims = ", ".join(f"{layer}={feats[layer].shape[1]}" for layer, _agg in specs)
                print(
                    f"  {stim_set:<22} {len(images):4d} imgs "
                    f"{time.time() - t0:6.1f}s [{dims}]",
                    flush=True,
                )

            for subject, specs in missing_dv.items():
                if not specs:
                    continue
                paths, filenames = load_deepvision_paths(subject)
                t0 = time.time()
                feats = extract_items(
                    extractor,
                    paths,
                    batch_size=args.batch_size,
                    progress_label=f"{subject} unique",
                    progress_every=args.progress_every,
                )
                out_path = LOCAL_DV_FEATURE_DIR / subject / f"{model}.npz"
                save_features(out_path, feats, filenames, specs, overwrite=args.overwrite)
                dims = ", ".join(f"{layer}={feats[layer].shape[1]}" for layer, _agg in specs)
                print(
                    f"  {subject:<8} unique {len(paths):4d} imgs "
                    f"{time.time() - t0:6.1f}s [{dims}]",
                    flush=True,
                )
        finally:
            extractor.free()
        print(f"  model done in {time.time() - t_model:.1f}s", flush=True)

    print(f"\nDone. Missing entries considered: {total_missing}", flush=True)


if __name__ == "__main__":
    main()
