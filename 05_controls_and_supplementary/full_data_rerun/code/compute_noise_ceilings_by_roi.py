#!/usr/bin/env python3
"""Compute full-data cstim/VICCO noise ceilings by ROI.

This mirrors the paper noise-ceiling convention on the full-preprocessed
LAION-fMRI cstim cache:

- mRSA denominator: within-subject split-half RDM reliability, Spearman-Brown
  corrected. Downstream score normalization uses sqrt(reliability).
- fRSA denominator: between-subject Kriegeskorte RDM ceiling, using the midpoint
  between upper and lower bounds.

VICCO is subsampled to 100 images to match the cstim set size.
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
sys.path.insert(0, str(SHARE_ROOT / "src"))

from cstims.cache import (  # noqa: E402
    load_cstim_brain_cache,
    load_cstim_repetition_cache,
    load_cstim_stimulus_info,
    load_cstim_voxel_metadata,
)

SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
DEFAULT_BRAIN_CACHE = RERUN_ROOT / "results" / "brain_data_cache"
DEFAULT_RDM_OUT = RERUN_ROOT / "results" / "rdm_noise_ceilings_by_roi.csv"
DEFAULT_BETWEEN_OUT = RERUN_ROOT / "results" / "between_subject_noise_ceilings_by_roi.csv"
DEFAULT_ROIS = (
    "EVC",
    "ventral",
    "lateral",
    "dorsal",
    "general",
    "EBA",
    "FFA",
    "PPA",
    "LOTC",
    "floc_all",
    "ventral_lateral_floc",
)


def bootstrap_sample_indices(n_total: int, n_sample: int, n_bootstrap: int, seed: int = 0):
    out = []
    for i in range(n_bootstrap):
        rng = np.random.default_rng(seed + i)
        out.append(np.sort(rng.choice(n_total, size=n_sample, replace=False)))
    return out


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask].astype(np.float64, copy=False)
    y = y[mask].astype(np.float64, copy=False)
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / den) if den > 0 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    return pearson(rankdata(x[mask]), rankdata(y[mask]))


def spearman_brown(r: float) -> float:
    if not np.isfinite(r) or (1.0 + r) == 0:
        return float("nan")
    return float((2.0 * r) / (1.0 + r))


def rdm_vec_corr(stim_by_vox: np.ndarray) -> np.ndarray:
    """Correlation-distance RDM vector from an array shaped (n_stim, n_vox)."""
    x = stim_by_vox.astype(np.float32, copy=False)
    x = x - x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / (norms + 1e-12)
    corr = x @ x.T
    idx = np.triu_indices(corr.shape[0], k=1)
    return (1.0 - corr[idx]).astype(np.float32, copy=False)


def split_half_rdm_nc(even_betas: np.ndarray, odd_betas: np.ndarray) -> dict:
    vec_even = rdm_vec_corr(even_betas)
    vec_odd = rdm_vec_corr(odd_betas)
    r_spearman = spearman(vec_even, vec_odd)
    r_pearson = pearson(vec_even, vec_odd)
    return {
        "split_half_r_spearman": r_spearman,
        "noise_ceiling_spearman": spearman_brown(r_spearman),
        "split_half_r_pearson": r_pearson,
        "noise_ceiling_pearson": spearman_brown(r_pearson),
    }


def load_roi_masks(root: Path, rois: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    voxel_metadata = load_cstim_voxel_metadata(root.name, cache_root=root.parent)
    full_masks = {}
    union_mask = None
    for roi in rois:
        key = f"roi_{roi}"
        if key not in voxel_metadata:
            raise KeyError(f"{root.parent.name}: missing {key} in voxel_metadata.npz")
        mask = voxel_metadata[key].astype(bool)
        full_masks[roi] = mask
        union_mask = mask.copy() if union_mask is None else (union_mask | mask)
    roi_masks = {roi: mask[union_mask] for roi, mask in full_masks.items()}
    return roi_masks, union_mask


def load_stim_info(root: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    stim_info = load_cstim_stimulus_info(root.name, cache_root=root.parent)
    group_idx = {}
    for group in sorted(stim_info["group"].unique()):
        group_idx[group] = stim_info.index[stim_info["group"].eq(group)].to_numpy(dtype=int)
    return stim_info, group_idx


def load_split_halves(root: Path, stim_keys: list[str], union_mask: np.ndarray):
    n_vox = int(union_mask.sum())
    even = np.empty((len(stim_keys), n_vox), dtype=np.float32)
    odd = np.empty((len(stim_keys), n_vox), dtype=np.float32)
    reps = load_cstim_repetition_cache(root.name, roi="all", cache_root=root.parent)
    for i, key in enumerate(stim_keys):
        rep_betas = reps.betas_by_rep[str(key)][union_mask]
        n_reps = rep_betas.shape[1]
        if n_reps < 2:
            even[i] = rep_betas[:, 0]
            odd[i] = rep_betas[:, 0]
        elif n_reps == 2:
            even[i] = rep_betas[:, 0]
            odd[i] = rep_betas[:, 1]
        else:
            even[i] = rep_betas[:, np.arange(0, n_reps, 2)].mean(axis=1)
            odd[i] = rep_betas[:, np.arange(1, n_reps, 2)].mean(axis=1)
    return even, odd


def compute_rdm_noise_ceilings(
    brain_cache: Path,
    subjects: list[str],
    rois: list[str],
    n_vicco_boot: int,
    n_vicco_sample: int,
) -> pd.DataFrame:
    rows = []
    for subject in subjects:
        root = brain_cache / subject
        print(f"{subject}: loading split halves", flush=True)
        roi_masks, union_mask = load_roi_masks(root, rois)
        stim_info, group_idx = load_stim_info(root)
        stim_keys = stim_info["stim_key"].tolist()
        even_all, odd_all = load_split_halves(root, stim_keys, union_mask)

        vicco_idx = group_idx.get("vicco", np.array([], dtype=int))
        vicco_sample = min(n_vicco_sample, len(vicco_idx))
        vicco_boot = bootstrap_sample_indices(len(vicco_idx), vicco_sample, n_vicco_boot, seed=0)

        for roi in rois:
            roi_mask = roi_masks[roi]
            if not roi_mask.any():
                print(f"{subject} {roi}: empty ROI, skipping", flush=True)
                continue
            even_roi = even_all[:, roi_mask]
            odd_roi = odd_all[:, roi_mask]
            n_voxels = int(roi_mask.sum())

            for group, idx in group_idx.items():
                if group == "vicco":
                    continue
                nc = split_half_rdm_nc(even_roi[idx], odd_roi[idx])
                rows.append(
                    {
                        "roi": roi,
                        "subject": subject,
                        "group": group,
                        "stimulus_type": "controversial",
                        "bootstrap_idx": 0,
                        "n_stimuli": int(len(idx)),
                        "n_voxels": n_voxels,
                        **nc,
                    }
                )

            if len(vicco_idx):
                vicco_ncs = []
                for boot_idx, boot in enumerate(vicco_boot):
                    idx = vicco_idx[boot]
                    nc = split_half_rdm_nc(even_roi[idx], odd_roi[idx])
                    rows.append(
                        {
                            "roi": roi,
                            "subject": subject,
                            "group": "vicco",
                            "stimulus_type": "vicco",
                            "bootstrap_idx": int(boot_idx),
                            "n_stimuli": int(vicco_sample),
                            "n_voxels": n_voxels,
                            **nc,
                        }
                    )
                    vicco_ncs.append(nc["noise_ceiling_spearman"])

                print(
                    f"{subject} {roi}: n_vox={n_voxels} "
                    f"vicco_nc_mean={np.nanmean(vicco_ncs):.3f}",
                    flush=True,
                )

    return pd.DataFrame(rows)


def kriegeskorte_nc_per_subject(rdm_mat: np.ndarray):
    n = rdm_mat.shape[0]
    upper = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    mean_all = rdm_mat.mean(axis=0)
    rdm_sum = rdm_mat.sum(axis=0)
    for i in range(n):
        mean_loo = (rdm_sum - rdm_mat[i]) / (n - 1)
        upper[i] = spearman(rdm_mat[i], mean_all)
        lower[i] = spearman(rdm_mat[i], mean_loo)
    return upper, lower


def load_subject_average(root: Path, rois: list[str]):
    roi_masks, union_mask = load_roi_masks(root, rois)
    stim_info, group_idx = load_stim_info(root)
    cache = load_cstim_brain_cache(root.name, roi="all", cache_root=root.parent)
    betas = cache.betas_roi[union_mask, :].astype(np.float32, copy=False)
    return {"roi_masks": roi_masks, "group_idx": group_idx, "betas": betas}


def compute_between_subject_noise_ceilings(
    brain_cache: Path,
    subjects: list[str],
    rois: list[str],
    n_vicco_boot: int,
    n_vicco_sample: int,
) -> pd.DataFrame:
    print("Loading averaged betas for between-subject ceilings", flush=True)
    sdata = {
        subject: load_subject_average(brain_cache / subject, rois)
        for subject in subjects
    }
    rows = []
    first_groups = sdata[subjects[0]]["group_idx"]
    groups = [g for g in sorted(first_groups) if g != "vicco"] + ["vicco"]
    rng = np.random.default_rng(42)

    for roi in rois:
        for group in groups:
            stimulus_type = "vicco" if group == "vicco" else "controversial"
            if group != "vicco":
                rdm_mat = []
                for subject in subjects:
                    idx = sdata[subject]["group_idx"][group]
                    roi_mask = sdata[subject]["roi_masks"][roi]
                    betas = sdata[subject]["betas"][roi_mask][:, idx].T
                    rdm_mat.append(rdm_vec_corr(betas))
                upper, lower = kriegeskorte_nc_per_subject(np.stack(rdm_mat))
            else:
                n_vicco = len(first_groups["vicco"])
                vicco_sample = min(n_vicco_sample, n_vicco)
                boot_upper = np.zeros((n_vicco_boot, len(subjects)), dtype=np.float64)
                boot_lower = np.zeros((n_vicco_boot, len(subjects)), dtype=np.float64)
                for boot_idx in range(n_vicco_boot):
                    boot = np.sort(rng.choice(n_vicco, size=vicco_sample, replace=False))
                    rdm_mat = []
                    for subject in subjects:
                        idx = sdata[subject]["group_idx"]["vicco"][boot]
                        roi_mask = sdata[subject]["roi_masks"][roi]
                        betas = sdata[subject]["betas"][roi_mask][:, idx].T
                        rdm_mat.append(rdm_vec_corr(betas))
                    u, l = kriegeskorte_nc_per_subject(np.stack(rdm_mat))
                    boot_upper[boot_idx] = u
                    boot_lower[boot_idx] = l
                upper = boot_upper.mean(axis=0)
                lower = boot_lower.mean(axis=0)

            mid = (upper + lower) / 2.0
            for i, subject in enumerate(subjects):
                rows.append(
                    {
                        "roi": roi,
                        "group": group,
                        "stimulus_type": stimulus_type,
                        "subject": subject,
                        "nc_upper": upper[i],
                        "nc_lower": lower[i],
                        "nc_mid": mid[i],
                    }
                )
            print(
                f"{roi} {group}: between_nc_mid={np.nanmean(mid):.3f}",
                flush=True,
            )

    return pd.DataFrame(rows)


def parse_rois(value: str) -> list[str]:
    if value == "default":
        return list(DEFAULT_ROIS)
    return [r.strip() for r in value.split(",") if r.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brain-cache", type=Path, default=DEFAULT_BRAIN_CACHE)
    parser.add_argument("--rdm-out", type=Path, default=DEFAULT_RDM_OUT)
    parser.add_argument("--between-out", type=Path, default=DEFAULT_BETWEEN_OUT)
    parser.add_argument("--subject", default="all")
    parser.add_argument("--rois", default="default")
    parser.add_argument("--n-vicco-sample", type=int, default=100)
    parser.add_argument("--n-vicco-boot", type=int, default=1000)
    parser.add_argument("--n-between-vicco-boot", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    subjects = SUBJECTS if args.subject == "all" else [args.subject]
    rois = parse_rois(args.rois)

    if args.rdm_out.exists() and args.between_out.exists() and not args.force:
        print(f"Both outputs exist, skipping: {args.rdm_out}, {args.between_out}", flush=True)
        return

    args.rdm_out.parent.mkdir(parents=True, exist_ok=True)
    if args.force or not args.rdm_out.exists():
        rdm_df = compute_rdm_noise_ceilings(
            args.brain_cache,
            subjects,
            rois,
            args.n_vicco_boot,
            args.n_vicco_sample,
        )
        tmp = args.rdm_out.with_suffix(args.rdm_out.suffix + ".tmp")
        rdm_df.to_csv(tmp, index=False)
        tmp.replace(args.rdm_out)
        print(f"Wrote {len(rdm_df)} rows -> {args.rdm_out}", flush=True)
    else:
        print(f"Skipping existing {args.rdm_out}", flush=True)

    if args.force or not args.between_out.exists():
        between_df = compute_between_subject_noise_ceilings(
            args.brain_cache,
            subjects,
            rois,
            args.n_between_vicco_boot,
            args.n_vicco_sample,
        )
        tmp = args.between_out.with_suffix(args.between_out.suffix + ".tmp")
        between_df.to_csv(tmp, index=False)
        tmp.replace(args.between_out)
        print(f"Wrote {len(between_df)} rows -> {args.between_out}", flush=True)
    else:
        print(f"Skipping existing {args.between_out}", flush=True)


if __name__ == "__main__":
    main()
