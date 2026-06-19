#!/usr/bin/env python3
"""Build full-data DeepVision unique response caches aligned to feature order."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


THIS = Path(__file__).resolve()
RERUN_ROOT = THIS.parents[1]
SHARE_ROOT = THIS.parents[3]
sys.path.insert(0, str(SHARE_ROOT / "src"))

from cstims.cache import load_cstim_voxel_metadata  # noqa: E402

LAION_ROOT = Path("/data/home_roth/datasets/LAION-fMRI")
OUT_ROOT = RERUN_ROOT / "results" / "deepvision_unique_cache"
OLD_UNIQUE_ROOT = (
    SHARE_ROOT
    / "01_brain_model_alignment"
    / "cache_or_heavy"
    / "deepvision_benchmark_cache"
    / "voxel_sets"
)
CSTIM_CACHE = RERUN_ROOT / "results" / "brain_data_cache"
SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]


def zscore_by_voxel(betas: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.nanmean(betas, axis=1, keepdims=True)
            std = np.nanstd(betas, axis=1, keepdims=True)
            out = (betas - mean) / np.maximum(std, eps)
    return np.nan_to_num(out, copy=False).astype(np.float32)


def load_feature_order(subject: str) -> np.ndarray:
    p = (
        OLD_UNIQUE_ROOT
        / f"deepvision_unique_{subject}_visual_cve0p20"
        / "finalinterp"
        / subject
        / "voxel_betas_cols.npy"
    )
    if not p.exists():
        raise FileNotFoundError(p)
    return np.load(p, allow_pickle=True).astype(str)


def session_dirs(subject: str):
    root = LAION_ROOT / "derivatives" / "glmsingle-tedana" / subject
    return sorted(p for p in root.glob("ses-*") if p.name < "ses-31")


def process_subject(subject: str, overwrite: bool):
    out_dir = OUT_ROOT / subject
    out_path = out_dir / "unique_betas_averaged.npz"
    if out_path.exists() and not overwrite:
        print(f"{subject}: exists, skipping {out_path}", flush=True)
        return

    cstim_meta = load_cstim_voxel_metadata(subject, cache_root=CSTIM_CACHE)
    brain_flat_indices = cstim_meta["brain_flat_indices"].astype(int)
    volume_shape = tuple(int(x) for x in cstim_meta["volume_shape"])
    roi_names = [str(x) for x in cstim_meta["roi_names"]]
    roi_union = np.zeros_like(cstim_meta["visual_mask"], dtype=bool)
    for roi in roi_names:
        roi_union |= cstim_meta[f"roi_{roi}"].astype(bool)
    union_idx = np.where(roi_union)[0]
    union_flat_indices = brain_flat_indices[union_idx]
    order = load_feature_order(subject)
    wanted = {name: i for i, name in enumerate(order.tolist())}

    sums = np.zeros((len(union_flat_indices), len(order)), dtype=np.float32)
    reps = np.zeros(len(order), dtype=np.int16)
    collected = 0
    print(
        f"{subject}: collecting {len(order)} unique images over "
        f"{len(union_flat_indices)} ROI-union voxels",
        flush=True,
    )
    for ses_dir in session_dirs(subject):
        func = ses_dir / "func"
        trials_paths = sorted(func.glob("*_desc-SingletrialBetas_trials.tsv"))
        beta_paths = sorted(func.glob("*_stat-effect_desc-SingletrialBetas_statmap.nii.gz"))
        if not trials_paths or not beta_paths:
            continue
        trials = pd.read_csv(trials_paths[0], sep="\t")
        labels = trials["label"].astype(str).to_numpy()
        keep = [(i, wanted[label]) for i, label in enumerate(labels) if label in wanted]
        keep_idx = [i for i, _ in keep]
        if not keep_idx:
            continue

        img = nib.load(str(beta_paths[0]))
        if img.shape[:3] != volume_shape:
            raise RuntimeError(f"{subject} {ses_dir.name}: shape mismatch {img.shape[:3]} vs {volume_shape}")
        data = np.asarray(img.dataobj, dtype=np.float32).reshape((-1, img.shape[3]))
        betas = zscore_by_voxel(data[union_flat_indices, :])
        for trial_idx, col_idx in keep:
            sums[:, col_idx] += betas[:, trial_idx]
            reps[col_idx] += 1
        collected = int(np.count_nonzero(reps))
        print(f"  {ses_dir.name}: collected {collected} unique images", flush=True)

    missing = [name for name, n in zip(order, reps) if n == 0]
    if missing:
        raise RuntimeError(f"{subject}: missing {len(missing)} unique images, first={missing[:5]}")

    out = sums / reps[np.newaxis, :].astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_path,
        betas=out,
        image_names=order,
        n_reps=reps,
        voxel_space=np.asarray("roi_union"),
        union_mask=roi_union,
    )
    tmp_path.replace(out_path)
    print(f"{subject}: wrote {out.shape} -> {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    subjects = SUBJECTS if args.subject == "all" else [args.subject]
    for subject in subjects:
        process_subject(subject, args.overwrite)


if __name__ == "__main__":
    main()
