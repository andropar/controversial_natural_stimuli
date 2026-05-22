#!/usr/bin/env python3
"""Image-blocked pair-level variance partitioning.

This is a mechanistic follow-up, not the primary causal-style control. The unit
of fitting is an RDM pair, but cross-validation is blocked by images: train
pairs contain no held-out image and test pairs contain only held-out images.
This avoids random pair splits that leak image identity.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler


PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(PAPER.parents[1]))

import config  # noqa: E402
from utils import compute_rdm_correlation, rdm_to_vector, load_cached_features, stimulus_cv_splits  # noqa: E402


DATA = PAPER / "18_explain_alignment_effect" / "results"
FIGURES = PAPER / "18_explain_alignment_effect" / "figures"
IMAGE_STATS = PAPER / "08_image_statistics" / "results" / "image_stats.csv"
OOD = PAPER / "06_ood" / "results" / "pca_loglik.csv"

DATA.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

GROUP_ORDER = ["all_models", "architecture", "dataset", "sota", "training_objective"]
STAT_COLS = [
    "lum_mean", "lum_rms", "colorfulness", "lab_chroma_mean", "hue_entropy",
    "sf_slope", "sf_high_low_ratio", "edge_mag_mean", "orient_anisotropy",
    "edge_com_x", "symmetry_lr", "entropy", "jpeg_ratio",
]
SEMANTIC_PROXY_MODELS = [
    "dinov2_vitl14",
    "openclip_vit_so400m_14_siglip_webli",
    "openclip_vit_l_14_quickgelu_metaclip_fullcc",
]
FAMILY_ORDER = ["low_level", "ood", "semantic_embedding", "model_disagreement", "model_space"]
ALPHAS = np.logspace(-2, 6, 30)
DEFAULT_RIDGE_ALPHA = 100.0


def _rank_columns(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    for j in range(x.shape[1]):
        out[:, j] = stats.rankdata(x[:, j]).astype(float)
    return out


def _rank_vec(y: np.ndarray) -> np.ndarray:
    return stats.rankdata(y).astype(float)


def _cv_r2(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    fixed_alpha: float | None = DEFAULT_RIDGE_ALPHA,
) -> tuple[float, float, int]:
    y_pred = np.full(y.shape, np.nan, dtype=float)
    alphas_seen: list[float] = []
    for train_idx, test_idx in splits:
        if len(test_idx) == 0:
            continue
        if X.shape[1] == 0:
            y_pred[test_idx] = float(np.mean(y[train_idx]))
            continue
        scaler = StandardScaler()
        x_train = scaler.fit_transform(X[train_idx])
        x_test = scaler.transform(X[test_idx])
        if fixed_alpha is None:
            model = RidgeCV(alphas=ALPHAS, scoring="neg_mean_squared_error")
            model.fit(x_train, y[train_idx])
            alphas_seen.append(float(model.alpha_))
        else:
            model = Ridge(alpha=fixed_alpha)
            model.fit(x_train, y[train_idx])
            alphas_seen.append(float(fixed_alpha))
        y_pred[test_idx] = model.predict(x_test)

    mask = np.isfinite(y_pred)
    if mask.sum() < 3:
        return float("nan"), float("nan"), int(mask.sum())
    ss_res = np.sum((y[mask] - y_pred[mask]) ** 2)
    ss_tot = np.sum((y[mask] - y[mask].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    alpha = float(np.median(alphas_seen)) if alphas_seen else np.nan
    return float(r2), alpha, int(mask.sum())


def _pair_summaries(x: np.ndarray, prefix: str) -> tuple[np.ndarray, list[str]]:
    """Pair abs-difference, mean, and max for each image-level column."""
    n = x.shape[0]
    i, j = np.triu_indices(n, k=1)
    cols = []
    names = []
    for k in range(x.shape[1]):
        a = x[i, k]
        b = x[j, k]
        cols.extend([np.abs(a - b), 0.5 * (a + b), np.maximum(a, b)])
        names.extend([f"{prefix}{k}_absdiff", f"{prefix}{k}_mean", f"{prefix}{k}_max"])
    return np.column_stack(cols), names


def _low_level_family(group: str, image_stats: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    sub = image_stats[image_stats["stimulus_set"] == group].sort_values("image").reset_index(drop=True)
    if len(sub) != 100:
        raise RuntimeError(f"Expected 100 image-stat rows for {group}, got {len(sub)}")
    x = sub[STAT_COLS].astype(float)
    x = x.fillna(x.median())
    xz = StandardScaler().fit_transform(x.to_numpy(dtype=float))
    pair_cols, pair_names = _pair_summaries(xz, "low_level_")
    euclid = pdist(xz, metric="euclidean")[:, None]
    return np.column_stack([euclid, pair_cols]), ["low_level_euclidean"] + pair_names


def _ood_image_scores() -> pd.DataFrame:
    df = pd.read_csv(OOD)
    df = df[df["stimulus_group"].isin(GROUP_ORDER)].copy()
    df["feature_ood"] = -pd.to_numeric(df["loglik_feature_z"], errors="coerce")
    df["pred_ood"] = -pd.to_numeric(df["loglik_pred_z"], errors="coerce")
    agg = (
        df.groupby(["stimulus_group", "stimulus_idx"], as_index=False)
        .agg(feature_ood=("feature_ood", "mean"), pred_ood=("pred_ood", "mean"))
        .sort_values(["stimulus_group", "stimulus_idx"])
    )
    return agg


def _ood_family(group: str, ood_scores: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    sub = ood_scores[ood_scores["stimulus_group"] == group].sort_values("stimulus_idx").reset_index(drop=True)
    if len(sub) != 100:
        raise RuntimeError(f"Expected 100 OOD rows for {group}, got {len(sub)}")
    x = sub[["feature_ood", "pred_ood"]].to_numpy(dtype=float)
    x = np.nan_to_num(x, nan=np.nanmedian(x))
    return _pair_summaries(x, "ood_")


def _rdm_vec_from_features(features: np.ndarray) -> np.ndarray:
    return rdm_to_vector(compute_rdm_correlation(features))


def _semantic_family(group: str) -> tuple[np.ndarray, list[str]]:
    cols = []
    names = []
    for model in SEMANTIC_PROXY_MODELS:
        try:
            feats = load_cached_features(model, group)
        except FileNotFoundError:
            continue
        cols.append(_rdm_vec_from_features(feats))
        names.append(f"semantic_proxy_rdm_{model}")
    if not cols:
        raise RuntimeError(f"No semantic proxy features found for {group}")
    return np.column_stack(cols), names


def _model_family(group: str) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    model_vecs = []
    model_names = []
    for model in config.MODEL_SETS[group]:
        try:
            feats = load_cached_features(model, group)
        except FileNotFoundError:
            continue
        model_vecs.append(_rdm_vec_from_features(feats))
        model_names.append(model)
    if not model_vecs:
        raise RuntimeError(f"No model-space features found for {group}")
    model_space = np.column_stack(model_vecs)

    z = StandardScaler().fit_transform(model_space)
    disagree = np.column_stack(
        [
            np.nanstd(z, axis=1),
            np.nanmax(z, axis=1) - np.nanmin(z, axis=1),
        ]
    )
    return (
        model_space,
        [f"model_rdm_{m}" for m in model_names],
        disagree,
        ["model_distance_sd_across_models", "model_distance_range_across_models"],
    )


def build_family_matrices(group: str, image_stats: pd.DataFrame, ood_scores: pd.DataFrame) -> dict[str, tuple[np.ndarray, list[str]]]:
    low, low_names = _low_level_family(group, image_stats)
    ood, ood_names = _ood_family(group, ood_scores)
    sem, sem_names = _semantic_family(group)
    model_space, model_names, disagree, disagree_names = _model_family(group)
    return {
        "low_level": (low, low_names),
        "ood": (ood, ood_names),
        "semantic_embedding": (sem, sem_names),
        "model_disagreement": (disagree, disagree_names),
        "model_space": (model_space, model_names),
    }


def _brain_vec(subject: str, group: str) -> np.ndarray:
    data_dir = config.get_brain_input_dir(subject)
    betas_npz = np.load(data_dir / "cstim_betas_averaged.npz", allow_pickle=True)
    voxel_npz = np.load(data_dir / "voxel_metadata.npz", allow_pickle=True)
    stim_info = pd.read_csv(data_dir / "cstim_stimulus_info.csv")

    hlvis_mask = voxel_npz["hlvis_mask"]
    stim_keys = betas_npz["stim_keys"]
    key_to_idx = {str(k): i for i, k in enumerate(stim_keys)}

    sub = stim_info[stim_info["group"] == group].sort_values("stim_idx").reset_index(drop=True)
    if len(sub) != 100:
        raise RuntimeError(f"Expected 100 stimuli for {subject}/{group}, got {len(sub)}")
    brain_idx = np.array([key_to_idx[str(k)] for k in sub["stim_key"]], dtype=int)
    betas = betas_npz["betas"][hlvis_mask, :][:, brain_idx].T
    return rdm_to_vector(compute_rdm_correlation(betas))


def fit_cell(
    subject: str,
    group: str,
    families: dict[str, tuple[np.ndarray, list[str]]],
    n_splits: int,
    ridge_alpha: float | None,
) -> list[dict]:
    y = _rank_vec(_brain_vec(subject, group))
    splits = stimulus_cv_splits(100, n_splits=n_splits, random_state=42)

    family_ranked = {name: _rank_columns(x) for name, (x, _) in families.items()}
    full_x = np.column_stack([family_ranked[f] for f in FAMILY_ORDER if f in family_ranked])
    r2_full, alpha_full, n_test_pairs = _cv_r2(full_x, y, splits, fixed_alpha=ridge_alpha)

    rows = []
    for family in FAMILY_ORDER:
        if family not in family_ranked:
            continue
        alone_x = family_ranked[family]
        r2_alone, alpha_alone, _ = _cv_r2(alone_x, y, splits, fixed_alpha=ridge_alpha)
        reduced_parts = [family_ranked[f] for f in FAMILY_ORDER if f in family_ranked and f != family]
        reduced_x = np.column_stack(reduced_parts) if reduced_parts else np.zeros((len(y), 0))
        r2_reduced, alpha_reduced, _ = _cv_r2(reduced_x, y, splits, fixed_alpha=ridge_alpha)
        rows.append(
            {
                "subject": subject,
                "model_set": group,
                "target": "brain_rdm_distance",
                "family": family,
                "n_stimuli": 100,
                "n_pairs_total": int(len(y)),
                "n_pairs_tested_image_blocked": n_test_pairs,
                "n_predictors_family": int(alone_x.shape[1]),
                "n_predictors_full": int(full_x.shape[1]),
                "cv_folds_image_blocked": n_splits,
                "r2_full": r2_full,
                "alpha_full_median": alpha_full,
                "r2_family_alone": r2_alone,
                "alpha_family_alone_median": alpha_alone,
                "r2_reduced_without_family": r2_reduced,
                "alpha_reduced_median": alpha_reduced,
                "unique_r2_drop_from_full": r2_full - r2_reduced,
            }
        )
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("pooled", "all_sets", rows)]
    for model_set, sub in rows.groupby("model_set", sort=False):
        scopes.append(("model_set", model_set, sub))
    for scope, model_set, sub in scopes:
        for family, grp in sub.groupby("family", sort=False):
            vals = grp["unique_r2_drop_from_full"].to_numpy(dtype=float)
            alone = grp["r2_family_alone"].to_numpy(dtype=float)
            out.append(
                {
                    "scope": scope,
                    "model_set": model_set,
                    "family": family,
                    "n_cells": int(len(grp)),
                    "n_subjects": int(grp["subject"].nunique()),
                    "mean_r2_full": float(grp["r2_full"].mean()),
                    "mean_r2_family_alone": float(np.nanmean(alone)),
                    "sem_r2_family_alone": float(np.nanstd(alone, ddof=1) / np.sqrt(len(alone))) if len(alone) > 1 else np.nan,
                    "mean_unique_r2_drop": float(np.nanmean(vals)),
                    "sem_unique_r2_drop": float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan,
                    "n_positive_unique_drop": int(np.sum(vals > 0)),
                    "n_negative_unique_drop": int(np.sum(vals < 0)),
                }
            )
    return pd.DataFrame(out)


def plot_summary(summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    pooled = summary[summary["scope"] == "pooled"].set_index("family").reindex(FAMILY_ORDER)
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    x = np.arange(len(pooled))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.bar(x, pooled["mean_unique_r2_drop"], color="#0072B2", width=0.55)
    ax.errorbar(
        x,
        pooled["mean_unique_r2_drop"],
        yerr=1.96 * pooled["sem_unique_r2_drop"],
        fmt="none",
        color="black",
        linewidth=0.8,
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["low", "OOD", "embed", "disagree", "models"], rotation=0)
    ax.set_ylabel("Unique blocked-CV R2 drop")
    ax.set_title("Pair-level variance partitioning", loc="left", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.45)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES / f"pair_variance_partition.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default="all", help="'all' or comma-separated model sets")
    parser.add_argument("--subjects", default="all", help="'all' or comma-separated subjects")
    parser.add_argument("--cv-folds", type=int, default=5, help="Image-blocked CV folds")
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=DEFAULT_RIDGE_ALPHA,
        help="Fixed ridge alpha for all blocked-CV fits. Use --tune-alpha to run RidgeCV instead.",
    )
    parser.add_argument("--tune-alpha", action="store_true", help="Tune ridge alpha inside each training fold.")
    args = parser.parse_args()

    groups = GROUP_ORDER if args.groups == "all" else [g.strip() for g in args.groups.split(",") if g.strip()]
    subjects = config.SUBJECTS if args.subjects == "all" else [s.strip() for s in args.subjects.split(",") if s.strip()]
    bad_groups = set(groups) - set(GROUP_ORDER)
    if bad_groups:
        raise ValueError(f"Unknown groups: {sorted(bad_groups)}")
    ridge_alpha = None if args.tune_alpha else args.ridge_alpha

    image_stats = pd.read_csv(IMAGE_STATS)
    ood_scores = _ood_image_scores()

    rows: list[dict] = []
    for group in groups:
        print(f"[{group}] building predictor families", flush=True)
        families = build_family_matrices(group, image_stats, ood_scores)
        for subject in subjects:
            print(f"  fitting {subject}", flush=True)
            rows.extend(fit_cell(subject, group, families, n_splits=args.cv_folds, ridge_alpha=ridge_alpha))

    out = pd.DataFrame(rows)
    out.to_csv(DATA / "pair_variance_partition_by_cell.csv", index=False)
    summary = summarize(out)
    summary.to_csv(DATA / "pair_variance_partition_summary.csv", index=False)
    plot_summary(summary)

    print(f"wrote {DATA / 'pair_variance_partition_by_cell.csv'}")
    print(f"wrote {DATA / 'pair_variance_partition_summary.csv'}")
    print(summary[summary["scope"] == "pooled"].to_string(index=False))


if __name__ == "__main__":
    main()
