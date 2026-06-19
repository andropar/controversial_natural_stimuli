#!/usr/bin/env python3
"""Extract multi-layer features for the layer sweep.

For each model in MODEL_LAYERS and each stimulus_set in STIMULUS_SETS, runs ONE
forward pass per batch and writes a single .npz containing arrays for every
configured layer.

Cache layout:
    11_layer_sweep/cache/features/{model}/{stimulus_set}.npz
        keys: one per layer name
        each: float32 array (n_images, feature_dim)
        plus 'image_filenames' (object array, sorted glob order)

Idempotent: if the cache file exists and contains all configured layers, it
is left alone. Pass --overwrite to regenerate.

Image preprocessing matches the model's own transform from get_deepjuice_model
or get_custom_model, exactly as used in 02_rsa_scores/01_compute_crsa.py.
"""

import _paths  # noqa: F401  -- sets up sys.path
from _paths import LAYER_SWEEP_ROOT
import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from cstims import paths
from batch_tuning import parse_batch_candidates, parse_batch_size, tune_batch_size
from layers_config import MODEL_LAYERS, MODEL_SOURCE, STIMULUS_SETS, get_layer_set
from multilayer_extractor import MultiLayerExtractor
from srp_utils import cached_layer_current, metadata_arrays, SRPProjectorCache


CACHE_ROOT = LAYER_SWEEP_ROOT / "cache_or_heavy" / "features"


def load_images(group: str):
    """Load stimulus images for a group, applying the architecture/dataset
    folder swap that matches 02_compute_crsa.py.
    """
    img_files = paths.cstim_image_paths(group, apply_architecture_dataset_swap=True)
    images = [Image.open(f).convert("RGB") for f in img_files]
    filenames = [f.name for f in img_files]
    return images, filenames


def cache_path(model: str, stimulus_set: str) -> Path:
    return CACHE_ROOT / model / f"{stimulus_set}.npz"


def is_complete(model: str, stimulus_set: str) -> bool:
    p = cache_path(model, stimulus_set)
    if not p.exists():
        return False
    try:
        d = np.load(p, allow_pickle=True)
        for layer_name, _ in MODEL_LAYERS[model]:
            if not cached_layer_current(d, layer_name):
                return False
        return True
    except Exception:
        return False


def missing_layer_specs(model: str, stimulus_set: str):
    p = cache_path(model, stimulus_set)
    if not p.exists():
        return list(MODEL_LAYERS[model])
    try:
        d = np.load(p, allow_pickle=True)
        present = set(d.files)
        return [
            (name, agg)
            for name, agg in MODEL_LAYERS[model]
            if name not in present or not cached_layer_current(d, name)
        ]
    except Exception:
        return list(MODEL_LAYERS[model])


def load_existing_payload(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=True) as d:
            return {k: d[k] for k in d.files}
    except Exception:
        return {}


def extract_for_set(extractor: MultiLayerExtractor, images, filenames,
                    batch_size: int = 32) -> tuple[dict, dict]:
    """Extract flattened layer features and immediately SRP-project by batch."""
    n = len(images)
    out: dict = {name: [] for name, _ in extractor.layers}
    meta_by_layer: dict = {}
    srp_cache = SRPProjectorCache()
    for start in range(0, n, batch_size):
        batch = images[start:start + batch_size]
        feats = extractor.extract(batch)
        for name, _ in extractor.layers:
            reduced, meta = srp_cache.transform(feats[name])
            if name not in meta_by_layer:
                meta_by_layer[name] = meta
            elif int(meta_by_layer[name]["original_feature_dim"]) != int(meta["original_feature_dim"]):
                raise RuntimeError(f"Inconsistent feature dim for layer {name}")
            out[name].append(reduced)
    out = {name: np.concatenate(arr, axis=0).astype(np.float32) for name, arr in out.items()}
    out["image_filenames"] = np.array(filenames, dtype=object)
    return out, meta_by_layer


def main():
    global MODEL_LAYERS
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=list(MODEL_LAYERS.keys()),
                        help="Models to extract (default: all configured)")
    parser.add_argument("--layer-set", choices=["configured", "dense"], default="configured",
                        help="Layer inventory to extract")
    parser.add_argument("--stim-sets", nargs="*", default=STIMULUS_SETS,
                        help="Stimulus sets to extract (default: all)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract even if cache exists")
    parser.add_argument("--batch-size", default="auto",
                        help="Batch size or 'auto' to benchmark per model")
    parser.add_argument("--batch-candidates", default=None,
                        help="Comma-separated candidates for --batch-size auto")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    MODEL_LAYERS = get_layer_set(args.layer_set)
    args.batch_size = parse_batch_size(args.batch_size)
    batch_candidates = parse_batch_candidates(args.batch_candidates)

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    # Pre-load images once per stimulus_set (shared across models).
    print("Loading images for each stimulus set...")
    images_by_set = {}
    for s in args.stim_sets:
        imgs, fnames = load_images(s)
        images_by_set[s] = (imgs, fnames)
        print(f"  {s}: {len(imgs)} images")

    for model in args.models:
        if model not in MODEL_LAYERS:
            print(f"[skip] unknown model: {model}")
            continue

        missing_by_set = {
            s: (list(MODEL_LAYERS[model]) if args.overwrite else missing_layer_specs(model, s))
            for s in args.stim_sets
        }
        sets_to_do = [s for s, missing in missing_by_set.items() if missing]
        if not sets_to_do:
            print(f"[cached] {model} (all sets complete)")
            continue

        layers_to_extract = []
        seen = set()
        for layer_name, agg in MODEL_LAYERS[model]:
            if any(layer_name == n for missing in missing_by_set.values() for n, _ in missing):
                if layer_name not in seen:
                    layers_to_extract.append((layer_name, agg))
                    seen.add(layer_name)

        print(f"\n=== {model} ({MODEL_SOURCE[model]}, "
              f"{len(layers_to_extract)} missing / {len(MODEL_LAYERS[model])} dense layers) ===")
        t0 = time.time()
        extractor = MultiLayerExtractor(
            model, MODEL_SOURCE[model], layers_to_extract, device=args.device
        )
        print(f"  init done in {time.time()-t0:.1f}s, mode={extractor._mode}")

        batch_size = args.batch_size
        if batch_size == "auto":
            nonempty_sets = [s for s in sets_to_do if images_by_set[s][0]]
            if not nonempty_sets:
                print(f"  [skip] {model}: no images found for uncached sets", flush=True)
                extractor.free()
                continue
            probe_set = nonempty_sets[0]
            probe_images = images_by_set[probe_set][0][:max(batch_candidates)]
            print(f"  tuning batch size on {probe_set}...", flush=True)
            batch_size, _ = tune_batch_size(
                extractor.extract,
                probe_images,
                candidates=batch_candidates,
                verbose=True,
            )
            print(f"  selected batch_size={batch_size}", flush=True)

        try:
            for s in sets_to_do:
                imgs, fnames = images_by_set[s]
                if not imgs:
                    print(f"  [skip] {s:<22} no images found", flush=True)
                    continue
                missing = {name for name, _ in missing_by_set[s]}
                if not missing:
                    continue
                t0 = time.time()
                feats, meta_by_layer = extract_for_set(
                    extractor, imgs, fnames, batch_size=batch_size
                )
                out_path = cache_path(model, s)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {} if args.overwrite else load_existing_payload(out_path)
                for layer_name in sorted(missing):
                    payload[layer_name] = feats[layer_name]
                    payload.update(metadata_arrays(layer_name, meta_by_layer[layer_name]))
                payload["image_filenames"] = feats["image_filenames"]
                np.savez_compressed(out_path, **payload)
                dur = time.time() - t0
                shapes = ", ".join(
                    f"{n}={payload[n].shape[1]}" for n, _ in layers_to_extract
                    if n in missing
                )
                print(f"  {s:<22} {len(imgs):4d} imgs  {dur:5.1f}s  "
                      f"added={len(missing):3d} [{shapes}]")
        finally:
            extractor.free()

    print("\nDone.")


if __name__ == "__main__":
    main()
