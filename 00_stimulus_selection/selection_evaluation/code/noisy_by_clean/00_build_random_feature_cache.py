#!/usr/bin/env python3
"""Build a reusable random natural-pool feature cache for recovery evaluation.

Run this on Raven where the full candidate-pool memmaps are available. The
output format matches ``--random-feature-dir`` in the noisy-by-clean recovery
compute entrypoints:

    <output-dir>/_sampled_indices.npy
    <output-dir>/<model>.npz  # key: "features", shape [n_images, feature_dim]
    <output-dir>/manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[4]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cstims.data_loader import load_natural_features_with_metadata  # noqa: E402

ENV_CONFIG_ROOT = ROOT / "00_stimulus_selection" / "resources" / "configs" / "paths"
MODEL_SET_CONFIG_ROOT = (
    ROOT / "00_stimulus_selection" / "resources" / "configs" / "model_set"
)
REPO_MODEL_LIST_CSV = ROOT / "00_stimulus_selection" / "resources" / "model_list.csv"
DEFAULT_OUTPUT_PARENT = ROOT / "shared" / "cache_or_heavy"


def load_env_paths(env: str) -> dict[str, Any]:
    config_path = ENV_CONFIG_ROOT / f"{env}.yaml"
    with config_path.open() as f:
        payload = yaml.safe_load(f) or {}
    return payload.get("paths", payload)


def load_model_set(model_set: str) -> list[str]:
    path = MODEL_SET_CONFIG_ROOT / f"{model_set}.yaml"
    with path.open() as f:
        payload = yaml.safe_load(f) or {}
    names = payload.get("model_names")
    if not names:
        raise ValueError(f"No model_names found in {path}")
    return [str(name) for name in names]


def load_model_csv_names(model_csv: Path) -> list[str]:
    with model_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    names = [row["model"] for row in rows if row.get("model")]
    if not names:
        raise ValueError(f"No model names found in {model_csv}")
    return names


def default_output_dir(n_images: int, seed: int) -> Path:
    if n_images % 1000 == 0:
        size_tag = f"{n_images // 1000}k"
    else:
        size_tag = str(n_images)
    return DEFAULT_OUTPUT_PARENT / f"natural_pool_subset_{size_tag}_seed{seed}"


def save_npz_atomic(path: Path, features: np.ndarray, compress: bool) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        if compress:
            np.savez_compressed(f, features=features)
        else:
            np.savez(f, features=features)
    tmp.replace(path)


def existing_feature_shape(path: Path) -> tuple[int, ...] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        key = "features" if "features" in z.files else z.files[0]
        return tuple(z[key].shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="raven", choices=["raven", "iris"])
    parser.add_argument("--model-set", default="all_models")
    parser.add_argument(
        "--all-model-list",
        action="store_true",
        help="Use every model in model_list.csv instead of a model_set YAML.",
    )
    parser.add_argument("--model-list-csv", type=Path, default=None)
    parser.add_argument("--subset-root", type=Path, default=None)
    parser.add_argument("--preprocessed-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-images", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=["source", "float32", "float16"],
        default="source",
        help="Feature dtype to write. Default preserves the candidate-pool dtype.",
    )
    parser.add_argument("--compress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--overwrite-indices",
        action="store_true",
        help="Resample indices even if _sampled_indices.npy already exists.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_env_paths(args.env)

    model_csv = args.model_list_csv or Path(paths.get("model_list_csv", REPO_MODEL_LIST_CSV))
    if not model_csv.exists():
        model_csv = REPO_MODEL_LIST_CSV
    if args.all_model_list:
        model_names = load_model_csv_names(model_csv)
    else:
        model_names = load_model_set(args.model_set)

    subset_root = args.subset_root or Path(paths["subset_root"])
    preprocessed_dir = args.preprocessed_dir or Path(paths["preprocessed_dirs"]["raw"])
    output_dir = (args.output_dir or default_output_dir(args.n_images, args.seed)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Environment: {args.env}")
    print(f"Model count: {len(model_names)}")
    print(f"Model CSV: {model_csv}")
    print(f"Subset root: {subset_root}")
    print(f"Preprocessed dir: {preprocessed_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Target images: {args.n_images}")

    print("Opening candidate-pool memmaps...")
    features_by_model, shard_slices = load_natural_features_with_metadata(
        subset_root=subset_root,
        preprocessed_dir=preprocessed_dir,
        model_names=model_names,
        model_csv=model_csv,
    )
    n_available_by_model = {
        model: int(features_by_model[model].shape[0]) for model in model_names
    }
    n_available = min(n_available_by_model.values())
    if len(set(n_available_by_model.values())) != 1:
        print(f"[WARN] Model row counts differ; using min available={n_available}")
        print(n_available_by_model)
    if args.n_images > n_available:
        raise ValueError(
            f"Requested {args.n_images} images, but only {n_available} rows are available."
        )

    indices_path = output_dir / "_sampled_indices.npy"
    draw_order_path = output_dir / "_sampled_indices_draw_order.npy"
    if indices_path.exists() and not args.overwrite_indices:
        sampled_indices = np.load(indices_path)
        print(f"Reusing existing sampled indices: {indices_path}")
        if sampled_indices.shape != (args.n_images,):
            raise ValueError(
                f"Existing indices have shape {sampled_indices.shape}; "
                f"expected {(args.n_images,)}. Use --overwrite-indices to resample."
            )
        if int(sampled_indices.max()) >= n_available:
            raise ValueError(
                f"Existing indices exceed available rows: max={sampled_indices.max()}, "
                f"available={n_available}"
            )
    else:
        rng = np.random.default_rng(args.seed)
        draw_order = rng.choice(n_available, size=args.n_images, replace=False).astype(
            np.int64
        )
        sampled_indices = np.sort(draw_order)
        if not args.dry_run:
            np.save(indices_path, sampled_indices)
            np.save(draw_order_path, draw_order)
        print(f"Sampled {len(sampled_indices)} global indices with seed={args.seed}")

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(SCRIPT.relative_to(ROOT)),
        "env": args.env,
        "model_set": args.model_set,
        "all_model_list": bool(args.all_model_list),
        "model_names": model_names,
        "n_models": len(model_names),
        "n_images": int(args.n_images),
        "seed": int(args.seed),
        "indices_file": "_sampled_indices.npy",
        "draw_order_indices_file": "_sampled_indices_draw_order.npy",
        "sampled_indices_min": int(sampled_indices.min()),
        "sampled_indices_max": int(sampled_indices.max()),
        "n_available": int(n_available),
        "n_shards": len(shard_slices),
        "subset_root": str(subset_root),
        "preprocessed_dir": str(preprocessed_dir),
        "model_list_csv": str(model_csv),
        "compress": bool(args.compress),
        "features_key": "features",
        "dtype": args.dtype,
        "models": {},
    }

    if args.dry_run:
        print("Dry run: not writing model feature files.")
        print(json.dumps(manifest, indent=2))
        return

    for model in tqdm(model_names, desc="Writing model caches", unit="model"):
        out_path = output_dir / f"{model}.npz"
        existing_shape = existing_feature_shape(out_path)
        feature_dim = int(features_by_model[model].shape[1])
        expected_shape = (args.n_images, feature_dim)
        if existing_shape == expected_shape and not args.overwrite:
            print(f"  [skip] {model}: existing {existing_shape}")
            manifest["models"][model] = {
                "file": out_path.name,
                "shape": list(existing_shape),
                "skipped_existing": True,
            }
            continue
        if existing_shape is not None and existing_shape != expected_shape and not args.overwrite:
            raise ValueError(
                f"{out_path} exists with shape {existing_shape}, expected {expected_shape}. "
                "Use --overwrite to replace it."
            )

        sampled = np.asarray(features_by_model[model][sampled_indices])
        if args.dtype != "source":
            sampled = sampled.astype(args.dtype, copy=False)
        save_npz_atomic(out_path, sampled, compress=args.compress)
        manifest["models"][model] = {
            "file": out_path.name,
            "shape": list(sampled.shape),
            "dtype": str(sampled.dtype),
            "size_bytes": out_path.stat().st_size,
            "skipped_existing": False,
        }
        del sampled

    with (output_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    total_size = sum(p.stat().st_size for p in output_dir.glob("*.npz"))
    print(f"Done. Wrote {len(model_names)} model files.")
    print(f"Total .npz size: {total_size / 1024**3:.2f} GiB")
    print(f"Cache dir: {output_dir}")


if __name__ == "__main__":
    main()
