#!/usr/bin/env python3
"""Compute low-level image statistics for subject-specific unique images.

The held-out unique matching analysis needs covariates in the same image order
as the unique encoding features and beta columns. DeepVision image metadata CSVs
provide that order, so this script writes one row per subject x image_idx.

Output:
  experiments/cstim_paper/05_heldout_unique_baseline/data/unique_image_low_level_stats.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))

import config  # noqa: E402


OUT = _PAPER / "05_heldout_unique_baseline" / "results" / "unique_image_low_level_stats.csv"


def _load_compute_stats():
    path = _PAPER / "08_image_statistics" / "01_compute_image_stats.py"
    spec = importlib.util.spec_from_file_location("cstim_image_stats", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import compute_stats from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_stats


def subject_metadata(subject: str) -> pd.DataFrame:
    image_root = (
        config.SHARE_ROOT
        / "01_brain_model_alignment"
        / "cache_or_heavy"
        / "brain_data"
        / "image_sets"
    )
    path = image_root / f"deepvision_unique_{subject}.csv"
    df = pd.read_csv(path)
    df["image_idx"] = range(len(df))
    df["subject"] = subject
    df["image_path"] = (
        image_root
        / f"deepvision_unique_{subject}"
    ).as_posix() + "/" + df["image_name"]
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=config.SUBJECTS)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-side", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    compute_stats = _load_compute_stats()
    existing = pd.DataFrame()
    done = set()
    if OUT.exists() and not args.overwrite:
        existing = pd.read_csv(OUT)
        done = set(zip(existing["subject"], existing["image_idx"]))

    rows = []
    for subject in args.subjects:
        meta = subject_metadata(subject)
        if args.max_images is not None:
            meta = meta.head(args.max_images)
        for _, r in tqdm(meta.iterrows(), total=len(meta), desc=subject):
            key = (r["subject"], int(r["image_idx"]))
            if key in done:
                continue
            with Image.open(r["image_path"]) as img:
                img = img.convert("RGB")
                if args.max_side and max(img.size) > args.max_side:
                    img.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
                stats = compute_stats(img)
            stats.update(
                {
                    "subject": r["subject"],
                    "image_idx": int(r["image_idx"]),
                    "image": r["image_name"],
                    "dataset": r.get("dataset", ""),
                    "stat_max_side": args.max_side,
                }
            )
            rows.append(stats)

    out = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    if not out.empty:
        out = out.drop_duplicates(["subject", "image_idx"], keep="last").sort_values(
            ["subject", "image_idx"]
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Wrote {len(out)} rows -> {OUT}")


if __name__ == "__main__":
    main()
