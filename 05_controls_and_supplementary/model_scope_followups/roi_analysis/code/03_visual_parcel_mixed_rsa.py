#!/usr/bin/env python3
"""Mixed-RSA scores across full visual-cortex parcels.

This extends the primary hlvis analysis to HCP-MMP parcels inside each
subject's visual-response mask. It uses the already fitted unique-image
encoding models, whose weights cover the full visual mask, and evaluates the
same cstim and same-session baseline images used in the main paper.

Outputs
-------
data/visual_parcel_mixed_rsa.csv
    Long table of mixed-RSA scores by subject/model/set/ROI/stimulus type.
data/visual_parcel_endpoint_summary.csv
    Subject x model-set x ROI endpoint summaries, including cstim-baseline
    delta, relative delta, and model-spread ratio.
data/visual_parcel_metadata.csv
    ROI/parcel voxel counts and grouping metadata.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import rankdata

PAPER = Path(__file__).resolve().parents[1]
PROJECT = PAPER.parents[1]
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(PROJECT))

from cstims import constants, paths
from cstims.cache import load_cstim_brain_cache, load_cstim_feature_groups
from cstims.sampling import bootstrap_sample_indices  # noqa: E402


OUT = PAPER / "13_roi_analysis" / "results"
OUT.mkdir(parents=True, exist_ok=True)

ATLAS_ROOT = paths.deepvision_fmri_root() / "derivatives/functional/1sTR_1pt5mm/atlas"
GLASSER_NAMES = ATLAS_ROOT / "glasser_names.txt"

MODEL_SET_ORDER = ["all_models", "sota", "training_objective", "architecture", "dataset"]
N_BASELINE_BOOTSTRAPS = 10
N_BASELINE_SAMPLE = 100
MIN_VOXELS = 20

VISUAL_GROUPS: dict[str, set[str]] = {
    "Early visual": {"V1", "V2", "V3"},
    "Extended early/dorsal": {"V3A", "V3B", "V3CD", "V6", "V6A", "V7", "IPS1"},
    "Mid-level/lateral": {"V4", "V4t", "V8", "LO1", "LO2", "LO3", "PIT"},
    "Ventral object/scene": {
        "FFC",
        "VVC",
        "PHA1",
        "PHA2",
        "PHA3",
        "VMV1",
        "VMV2",
        "VMV3",
        "PH",
        "PHT",
        "TF",
        "TE2p",
    },
    "Motion": {"MT", "MST", "FST"},
    "Medial scene": {"RSC", "POS1", "POS2"},
}
GROUP_ORDER = list(VISUAL_GROUPS)


def load_glasser_names() -> list[str]:
    with open(GLASSER_NAMES) as f:
        return [line.strip() for line in f]


def parcel_core(parcel_name: str) -> str:
    return parcel_name.replace("L_", "").replace("R_", "").replace("_ROI", "")


def visual_group(parcel_name: str) -> str:
    core = parcel_core(parcel_name)
    for group, members in VISUAL_GROUPS.items():
        if core in members:
            return group
    return "Other"


def corr_rdm_vector(patterns: np.ndarray) -> np.ndarray:
    """Condensed correlation-distance RDM vector."""
    vec = pdist(np.asarray(patterns, dtype=np.float32), metric="correlation")
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def rank_standardize(vec: np.ndarray) -> np.ndarray | None:
    r = rankdata(vec, method="average").astype(np.float32)
    r -= float(r.mean())
    sd = float(r.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return None
    return r / sd


def spearman_from_ranked(pred_vec: np.ndarray, brain_rank: np.ndarray | None) -> float:
    pred_rank = rank_standardize(pred_vec)
    if pred_rank is None or brain_rank is None:
        return float("nan")
    return float(np.dot(pred_rank, brain_rank) / (len(pred_rank) - 1))


def median_abs_pairwise(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    diffs = np.abs(values[:, None] - values[None, :])
    return float(np.median(diffs[np.triu_indices(len(values), k=1)]))


def load_cstim_features(model: str) -> dict[str, np.ndarray]:
    return load_cstim_feature_groups(model, dtype=np.float32)


def load_encoding(model: str, subject: str) -> dict:
    path = paths.encoding_model_dir(subject, model) / "encoding_model.npz"
    with np.load(path, allow_pickle=True) as z:
        return {
            "weights": z["weights"].astype(np.float32),
            "intercept": z["intercept"].astype(np.float32),
            "feature_mean": z["feature_mean"].astype(np.float32),
            "feature_scale": z["feature_scale"].astype(np.float32),
        }


def predict_visual(features: np.ndarray, encoding: dict) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    mean = encoding["feature_mean"]
    scale = encoding["feature_scale"]
    if mean.size and np.any(mean != 0):
        x = x - mean
    if scale.size and np.any(scale != 1):
        x = x / (scale + 1e-8)
    return (x @ encoding["weights"] + encoding["intercept"]).astype(np.float32)


def load_subject_data(subject: str) -> dict:
    data_dir = paths.get_subject_data_dir(subject)
    cache = load_cstim_brain_cache(subject, roi="visual")
    visual_mask = cache.roi_mask.astype(bool)
    hlvis_over_visual = cache.voxel_metadata["hlvis_mask"].astype(bool)[visual_mask]
    betas_visual = cache.betas_roi.astype(np.float32, copy=False)

    groups = {}
    for group in cache.available_groups:
        brain_idx = cache.brain_indices(group)
        file_idx = cache.feature_indices(group)
        groups[group] = {"brain_idx": brain_idx, "file_idx": file_idx}

    return {
        "betas_visual": betas_visual,
        "visual_mask": visual_mask,
        "hlvis_over_visual": hlvis_over_visual,
        "brain_flat_indices": cache.voxel_metadata["brain_flat_indices"],
        "volume_shape": tuple(cache.voxel_metadata["volume_shape"]),
        "groups": groups,
    }


def build_rois(subject: str, sdata: dict, min_voxels: int) -> tuple[list[dict], pd.DataFrame]:
    names = load_glasser_names()
    atlas = nib.load(str(ATLAS_ROOT / subject / "HCPMMP1_func.nii.gz")).get_fdata(dtype=np.float32)

    visual_flat = sdata["brain_flat_indices"][sdata["visual_mask"]]
    coords = np.unravel_index(visual_flat, sdata["volume_shape"])
    probs = atlas[coords[0], coords[1], coords[2], :]
    assignment = np.argmax(probs, axis=1)
    max_prob = probs[np.arange(len(assignment)), assignment]
    del atlas, probs
    gc.collect()

    rois: list[dict] = []
    rows: list[dict] = []
    unique, counts = np.unique(assignment, return_counts=True)

    parcel_masks: dict[str, np.ndarray] = {}
    for parcel_idx, n_vox in zip(unique.astype(int), counts.astype(int)):
        if n_vox < min_voxels:
            continue
        name = names[parcel_idx]
        group = visual_group(name)
        if group == "Other":
            continue
        mask = assignment == parcel_idx
        parcel_masks[name] = mask
        rois.append(
            {
                "roi_level": "parcel",
                "roi": name,
                "roi_label": name,
                "visual_group": group,
                "mask": mask,
            }
        )
        rows.append(
            {
                "subject": subject,
                "roi_level": "parcel",
                "roi": name,
                "roi_label": name,
                "visual_group": group,
                "parcel_idx": parcel_idx,
                "n_voxels": int(mask.sum()),
                "mean_max_atlas_prob": float(max_prob[mask].mean()),
            }
        )

    for group in GROUP_ORDER:
        masks = [m for p, m in parcel_masks.items() if visual_group(p) == group]
        if not masks:
            continue
        mask = np.logical_or.reduce(masks)
        if int(mask.sum()) < min_voxels:
            continue
        rois.append(
            {
                "roi_level": "group",
                "roi": group.lower().replace("/", "_").replace(" ", "_"),
                "roi_label": group,
                "visual_group": group,
                "mask": mask,
            }
        )
        rows.append(
            {
                "subject": subject,
                "roi_level": "group",
                "roi": group.lower().replace("/", "_").replace(" ", "_"),
                "roi_label": group,
                "visual_group": group,
                "parcel_idx": np.nan,
                "n_voxels": int(mask.sum()),
                "mean_max_atlas_prob": float(max_prob[mask].mean()),
            }
        )

    aggregate_rois = {
        "full_visual": np.ones_like(assignment, dtype=bool),
        "hlvis": sdata["hlvis_over_visual"].astype(bool),
    }
    aggregate_labels = {"full_visual": "Full visual", "hlvis": "hlvis"}
    for roi, mask in aggregate_rois.items():
        if int(mask.sum()) < min_voxels:
            continue
        rois.append(
            {
                "roi_level": "aggregate",
                "roi": roi,
                "roi_label": aggregate_labels[roi],
                "visual_group": aggregate_labels[roi],
                "mask": mask,
            }
        )
        rows.append(
            {
                "subject": subject,
                "roi_level": "aggregate",
                "roi": roi,
                "roi_label": aggregate_labels[roi],
                "visual_group": aggregate_labels[roi],
                "parcel_idx": np.nan,
                "n_voxels": int(mask.sum()),
                "mean_max_atlas_prob": float(max_prob[mask].mean()),
            }
        )

    return rois, pd.DataFrame(rows)


def precompute_brain_ranks(sdata: dict, rois: list[dict], model_sets: list[str], n_boot: int) -> dict:
    ranks: dict[tuple, np.ndarray | None] = {}
    betas = sdata["betas_visual"]

    for model_set in model_sets:
        group = sdata["groups"].get(model_set)
        if group is None:
            continue
        brain = betas[:, group["brain_idx"]].T
        for roi in rois:
            vec = corr_rdm_vector(brain[:, roi["mask"]])
            ranks[(model_set, "controversial", 0, roi["roi_level"], roi["roi"])] = rank_standardize(vec)

    vicco = sdata["groups"]["vicco"]
    n_vicco = len(vicco["brain_idx"])
    bootstraps = bootstrap_sample_indices(n_vicco, N_BASELINE_SAMPLE, n_bootstrap=n_boot, seed=0)
    for bidx, subset in enumerate(bootstraps):
        brain = betas[:, vicco["brain_idx"][subset]].T
        for roi in rois:
            vec = corr_rdm_vector(brain[:, roi["mask"]])
            ranks[("vicco", "vicco", bidx, roi["roi_level"], roi["roi"])] = rank_standardize(vec)
    return ranks


def score_model_subject(
    subject: str,
    model: str,
    sdata: dict,
    rois: list[dict],
    brain_ranks: dict,
    model_sets: list[str],
    n_boot: int,
) -> list[dict]:
    rows: list[dict] = []
    encoding = load_encoding(model, subject)
    features = load_cstim_features(model)

    vicco_group = sdata["groups"]["vicco"]
    pred_vicco_all = predict_visual(features["vicco"][vicco_group["file_idx"]], encoding)
    n_vicco = len(vicco_group["brain_idx"])
    bootstraps = bootstrap_sample_indices(n_vicco, N_BASELINE_SAMPLE, n_bootstrap=n_boot, seed=0)

    display = constants.MODEL_DISPLAY_NAMES.get(model, model)

    for model_set in model_sets:
        if model not in constants.MODEL_SETS[model_set]:
            continue
        if model_set not in sdata["groups"]:
            continue

        cgroup = sdata["groups"][model_set]
        pred_cstim = predict_visual(features[model_set][cgroup["file_idx"]], encoding)
        for roi in rois:
            roi_key = (roi["roi_level"], roi["roi"])
            pred_vec = corr_rdm_vector(pred_cstim[:, roi["mask"]])
            score = spearman_from_ranked(
                pred_vec,
                brain_ranks[(model_set, "controversial", 0, *roi_key)],
            )
            rows.append(
                {
                    "subject": subject,
                    "model_set": model_set,
                    "model": model,
                    "display_name": display,
                    "roi_level": roi["roi_level"],
                    "roi": roi["roi"],
                    "roi_label": roi["roi_label"],
                    "visual_group": roi["visual_group"],
                    "stimulus_type": "controversial",
                    "bootstrap_idx": 0,
                    "n_stimuli": len(cgroup["brain_idx"]),
                    "n_voxels": int(roi["mask"].sum()),
                    "mixed_rsa": score,
                }
            )

            for bidx, subset in enumerate(bootstraps):
                pred_vec = corr_rdm_vector(pred_vicco_all[subset][:, roi["mask"]])
                score = spearman_from_ranked(
                    pred_vec,
                    brain_ranks[("vicco", "vicco", bidx, *roi_key)],
                )
                rows.append(
                    {
                        "subject": subject,
                        "model_set": model_set,
                        "model": model,
                        "display_name": display,
                        "roi_level": roi["roi_level"],
                        "roi": roi["roi"],
                        "roi_label": roi["roi_label"],
                        "visual_group": roi["visual_group"],
                        "stimulus_type": "vicco",
                        "bootstrap_idx": bidx,
                        "n_stimuli": N_BASELINE_SAMPLE,
                        "n_voxels": int(roi["mask"].sum()),
                        "mixed_rsa": score,
                    }
                )

    return rows


def summarize_endpoints(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = ["subject", "model_set", "roi_level", "roi", "roi_label", "visual_group"]
    for keys, grp in scores.groupby(group_cols, dropna=False):
        subject, model_set, roi_level, roi, roi_label, group = keys
        cstim = grp[grp["stimulus_type"] == "controversial"].groupby("model")["mixed_rsa"].mean()
        base = grp[grp["stimulus_type"] == "vicco"].groupby("model")["mixed_rsa"].mean()
        common = sorted(set(cstim.index) & set(base.index))
        if len(common) < 2:
            continue
        c = cstim.loc[common].to_numpy(dtype=float)
        b = base.loc[common].to_numpy(dtype=float)
        score_c = float(np.nanmean(c))
        score_b = float(np.nanmean(b))
        delta = score_c - score_b
        rel = delta / score_b if np.isfinite(score_b) and abs(score_b) > 1e-12 else np.nan
        spread_c = median_abs_pairwise(c)
        spread_b = median_abs_pairwise(b)
        rows.append(
            {
                "subject": subject,
                "model_set": model_set,
                "roi_level": roi_level,
                "roi": roi,
                "roi_label": roi_label,
                "visual_group": group,
                "n_models": len(common),
                "n_voxels": int(grp["n_voxels"].iloc[0]),
                "score_cstim": score_c,
                "score_baseline": score_b,
                "delta": delta,
                "relative_delta": rel,
                "spread_cstim": spread_c,
                "spread_baseline": spread_b,
                "spread_ratio": spread_c / spread_b
                if np.isfinite(spread_b) and spread_b > 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=constants.SUBJECTS)
    parser.add_argument("--model-sets", nargs="+", default=MODEL_SET_ORDER)
    parser.add_argument("--baseline-bootstraps", type=int, default=N_BASELINE_BOOTSTRAPS)
    parser.add_argument("--min-voxels", type=int, default=MIN_VOXELS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    score_path = OUT / "visual_parcel_mixed_rsa.csv"
    summary_path = OUT / "visual_parcel_endpoint_summary.csv"
    meta_path = OUT / "visual_parcel_metadata.csv"
    if not args.overwrite and score_path.exists() and summary_path.exists() and meta_path.exists():
        print(f"outputs already exist in {OUT}; pass --overwrite to recompute")
        return

    all_scores: list[dict] = []
    all_meta: list[pd.DataFrame] = []

    for subject in args.subjects:
        print(f"\n{subject}: loading brain data and visual parcels")
        sdata = load_subject_data(subject)
        rois, meta = build_rois(subject, sdata, args.min_voxels)
        all_meta.append(meta)
        print(
            f"  {int(sdata['visual_mask'].sum())} visual voxels; "
            f"{sum(r['roi_level'] == 'parcel' for r in rois)} parcels, "
            f"{sum(r['roi_level'] == 'group' for r in rois)} groups"
        )

        print("  precomputing brain RDM ranks")
        brain_ranks = precompute_brain_ranks(sdata, rois, args.model_sets, args.baseline_bootstraps)

        model_union = sorted(set(m for ms in args.model_sets for m in constants.MODEL_SETS[ms]))
        for i, model in enumerate(model_union, start=1):
            print(f"  scoring {i:02d}/{len(model_union)} {model}")
            rows = score_model_subject(
                subject,
                model,
                sdata,
                rois,
                brain_ranks,
                args.model_sets,
                args.baseline_bootstraps,
            )
            all_scores.extend(rows)
            gc.collect()

        del sdata, rois, brain_ranks
        gc.collect()

    scores = pd.DataFrame(all_scores)
    meta = pd.concat(all_meta, ignore_index=True)
    summary = summarize_endpoints(scores)

    scores.to_csv(score_path, index=False)
    meta.to_csv(meta_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"\nwrote {score_path} ({len(scores):,} rows)")
    print(f"wrote {summary_path} ({len(summary):,} rows)")
    print(f"wrote {meta_path} ({len(meta):,} rows)")


if __name__ == "__main__":
    main()
