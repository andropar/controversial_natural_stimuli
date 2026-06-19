#!/usr/bin/env python3
"""Compute paper-layer classical RSA on the full-data cstim cache by ROI.

This uses cached paper-layer features from:

    shared/cache_or_heavy/cstim_paper_feature_cache/feature_cache/cstim

and full-preprocessed cstim betas from:

    05_controls_and_supplementary/full_data_rerun/results/brain_data_cache

No encoding model weights are used here, so this remains valid even though the
full-data voxel grid differs from the original paper cache.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


THIS = Path(__file__).resolve()
RERUN_ROOT = THIS.parents[1]
SHARE_ROOT = THIS.parents[3]
HELPERS = SHARE_ROOT / "src"
sys.path.insert(0, str(HELPERS))

from cstims import constants, paths
from cstims.cache import cstim_brain_cache_exists, load_cstim_brain_cache


SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
DEFAULT_BRAIN_CACHE = RERUN_ROOT / "results" / "brain_data_cache"
DEFAULT_FEATURE_CACHE = (
    SHARE_ROOT
    / "shared"
    / "cache_or_heavy"
    / "cstim_paper_feature_cache"
    / "feature_cache"
    / "cstim"
)
DEFAULT_OUT = RERUN_ROOT / "results" / "paper_layer_crsa_by_roi.csv"
DEFAULT_SUMMARY_OUT = RERUN_ROOT / "results" / "paper_layer_crsa_by_roi_summary.csv"


def bootstrap_sample_indices(n_total: int, n_sample: int, n_bootstrap: int, seed: int = 0):
    out = []
    for i in range(n_bootstrap):
        rng = np.random.default_rng(seed + i)
        out.append(np.sort(rng.choice(n_total, size=n_sample, replace=False)))
    return out


def rdm_rank_vec(x: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(x)
    rdm = 1.0 - corr
    vec = rdm[np.triu_indices(rdm.shape[0], k=1)]
    return rankdata(vec, method="average").astype(np.float32)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / den) if den > 0 else float("nan")


def load_subject(subject: str, brain_cache: Path, rois: list[str], n_vicco_boot: int):
    cache = load_cstim_brain_cache(subject, roi="all", cache_root=brain_cache)
    betas_all = cache.betas_roi
    group_brain_idx = cache.group_brain_indices()
    group_file_idx = cache.group_feature_indices()

    roi_masks = {}
    for roi in rois:
        key = f"roi_{roi}"
        if key not in cache.voxel_metadata:
            raise KeyError(f"{subject}: missing {key} in voxel_metadata.npz")
        mask = cache.voxel_metadata[key].astype(bool)
        if int(mask.sum()) == 0:
            print(f"warning: {subject} {roi} has zero voxels")
        roi_masks[roi] = mask

    n_vicco = len(group_brain_idx.get("vicco", []))
    n_vicco_sample = min(100, n_vicco)
    vicco_boot = (
        bootstrap_sample_indices(n_vicco, n_vicco_sample, n_vicco_boot, seed=0)
        if n_vicco > 0 else []
    )

    brain_rank_cache = {}
    for roi, roi_mask in roi_masks.items():
        if not roi_mask.any():
            continue
        betas_roi = betas_all[roi_mask, :]
        for group, brain_idx in group_brain_idx.items():
            if group == "vicco":
                continue
            brain_rank_cache[(roi, group, 0)] = rdm_rank_vec(betas_roi[:, brain_idx].T)
        if "vicco" in group_brain_idx:
            vicco_brain_idx = group_brain_idx["vicco"]
            for boot_idx, boot in enumerate(vicco_boot):
                brain_rank_cache[(roi, "vicco", boot_idx)] = rdm_rank_vec(
                    betas_roi[:, vicco_brain_idx[boot]].T
                )

    return {
        "betas": betas_all,
        "group_brain_idx": group_brain_idx,
        "group_file_idx": group_file_idx,
        "roi_masks": roi_masks,
        "vicco_boot": vicco_boot,
        "n_vicco_sample": n_vicco_sample,
        "brain_rank_cache": brain_rank_cache,
    }


def load_model_features(feature_cache: Path, model: str) -> dict[str, np.ndarray]:
    p = feature_cache / f"{model}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing paper-layer feature cache for {model}: {p}")
    z = np.load(p, allow_pickle=True)
    return {k: z[k].astype(np.float32) for k in z.files}


def score_subject_model(subject: str, sdata: dict, model: str, features: dict, rois: list[str]):
    rows = []
    display = constants.MODEL_DISPLAY_NAMES.get(model, model)
    model_rank_cache = {}
    for key, arr in features.items():
        if key == "vicco":
            continue
        if key in sdata["group_file_idx"]:
            model_rank_cache[(key, 0)] = rdm_rank_vec(arr[sdata["group_file_idx"][key]])
    if "vicco" in features and "vicco" in sdata["group_file_idx"]:
        vicco_features = features["vicco"][sdata["group_file_idx"]["vicco"]]
        for boot_idx, boot in enumerate(sdata["vicco_boot"]):
            model_rank_cache[("vicco", boot_idx)] = rdm_rank_vec(vicco_features[boot])

    for roi in rois:
        roi_mask = sdata["roi_masks"][roi]
        if not roi_mask.any():
            continue

        for model_set, models in constants.MODEL_SETS.items():
            if model not in models:
                continue
            if model_set not in features or model_set not in sdata["group_brain_idx"]:
                continue

            brain_idx = sdata["group_brain_idx"][model_set]
            model_ranks = model_rank_cache[(model_set, 0)]
            brain_ranks = sdata["brain_rank_cache"][(roi, model_set, 0)]
            rows.append({
                "subject": subject,
                "roi": roi,
                "model_set": model_set,
                "model": model,
                "display_name": display,
                "stimulus_type": "controversial",
                "bootstrap_idx": 0,
                "n_stimuli": int(len(brain_idx)),
                "crsa": pearson(model_ranks, brain_ranks),
            })

            if "vicco" in features and "vicco" in sdata["group_brain_idx"]:
                for boot_idx, boot in enumerate(sdata["vicco_boot"]):
                    model_ranks = model_rank_cache[("vicco", boot_idx)]
                    brain_ranks = sdata["brain_rank_cache"][(roi, "vicco", boot_idx)]
                    rows.append({
                        "subject": subject,
                        "roi": roi,
                        "model_set": model_set,
                        "model": model,
                        "display_name": display,
                        "stimulus_type": "vicco",
                        "bootstrap_idx": int(boot_idx),
                        "n_stimuli": int(sdata["n_vicco_sample"]),
                        "crsa": pearson(model_ranks, brain_ranks),
                    })

    return rows


def write_summary(df: pd.DataFrame, path: Path):
    summary = (
        df.groupby(["roi", "model_set", "model", "display_name", "stimulus_type"], as_index=False)
        .agg(
            n_rows=("crsa", "size"),
            n_subjects=("subject", "nunique"),
            crsa_mean=("crsa", "mean"),
            crsa_sem=("crsa", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
        )
    )
    summary.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brain-cache", type=Path, default=DEFAULT_BRAIN_CACHE)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--subject", default="all")
    parser.add_argument("--n-vicco-boot", type=int, default=1000)
    parser.add_argument(
        "--rois",
        default="EVC,ventral,lateral,dorsal,general,EBA,FFA,PPA,LOTC,floc_all,ventral_lateral_floc",
    )
    args = parser.parse_args()

    subjects = SUBJECTS if args.subject == "all" else [args.subject]
    rois = [r.strip() for r in args.rois.split(",") if r.strip()]
    models = sorted(constants.MODEL_SETS["all_models"])

    rows = []
    for subject in subjects:
        if not cstim_brain_cache_exists(subject, cache_root=args.brain_cache):
            print(f"{subject}: missing brain cache, skipping")
            continue
        print(f"{subject}: loading brain cache")
        sdata = load_subject(subject, args.brain_cache, rois, args.n_vicco_boot)
        for model in models:
            print(f"  {subject} {model}")
            feats = load_model_features(args.feature_cache, model)
            rows.extend(score_subject_model(subject, sdata, model, feats, rois))

    if not rows:
        raise SystemExit("No rows computed.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    write_summary(df, args.summary_out)
    print(f"Wrote {len(df)} rows -> {args.out}")
    print(f"Wrote summary -> {args.summary_out}")


if __name__ == "__main__":
    main()
