#!/usr/bin/env python3
"""Fold-specific held-out unique-image baseline for mixed RSA.

The full analysis is intentionally explicit and resumable because it refits
encoding models. For each subject and split, a ridge model is fit on the unique
training fold and evaluated on controversial stimuli, the same-session baseline,
held-out unique baseline subsets, matched held-out subsets, and high-OOD
held-out controls.

This script does not use shared-image responses for baseline evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

STAGE = Path(__file__).resolve().parents[1]
SHARE_ROOT = STAGE.parents[1]
sys.path.insert(0, str(SHARE_ROOT / "src"))

from cstims import constants, paths
from cstims.cache import load_cstim_brain_cache, load_cstim_feature_groups
from cstims.rdm import compute_rdm_correlation, compute_rsa_score  # noqa: E402
from cstims.paper.utils import load_encoding_model  # noqa: E402


OUT_DIR = STAGE / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SETS = ["all_models", "sota", "training_objective", "architecture", "dataset"]
UNIQUE_LOW_LEVEL_PATH = OUT_DIR / "unique_image_low_level_stats.csv"
CSTIM_LOW_LEVEL_PATH = (
    SHARE_ROOT
    / "05_controls_and_supplementary"
    / "low_level_and_ood"
    / "image_statistics"
    / "results"
    / "image_stats.csv"
)
LOW_LEVEL_COLS = [
    "lum_mean",
    "lum_rms",
    "colorfulness",
    "lab_chroma_mean",
    "hue_entropy",
    "sf_slope",
    "sf_high_low_ratio",
    "edge_mag_mean",
    "orient_anisotropy",
    "edge_com_x",
    "symmetry_lr",
    "entropy",
    "jpeg_ratio",
]


def _unique_voxel_dir(subject: str) -> Path:
    return (
        paths.project_root()
        / "results"
        / "cache"
        / "voxel_sets"
        / f"deepvision_unique_{subject}_visual_cve0p20"
        / "finalinterp"
        / subject
    )


def load_unique_betas(subject: str, roi_hlvis: np.ndarray) -> np.ndarray:
    path = _unique_voxel_dir(subject) / "voxel_betas.npy"
    betas = np.load(path, mmap_mode="r")
    return np.asarray(betas[roi_hlvis, :], dtype=np.float32)


def load_cstim_brain(subject: str) -> dict:
    cache = load_cstim_brain_cache(subject)
    return {
        "betas_hlvis": cache.betas_roi.astype(np.float32, copy=False),
        "group_indices": cache.group_brain_indices(),
        "group_file_idx": cache.group_feature_indices(),
    }


def load_unique_features(subject: str, model: str) -> np.ndarray:
    folder = paths.encoding_model_dir(subject, model)
    path = folder / "features.npz"
    with np.load(path) as z:
        return z["features"].astype(np.float32)


def load_median_hlvis_alpha(subject: str, model: str, roi_hlvis: np.ndarray) -> float:
    path = paths.encoding_model_dir(subject, model) / "encoding_model.npz"
    try:
        with np.load(path, allow_pickle=True) as z:
            if "alphas" in z.files and z["alphas"].shape[0] == roi_hlvis.shape[0]:
                return float(np.nanmedian(z["alphas"][roi_hlvis]))
    except Exception:
        pass
    return 1.0


def load_cstim_features(model: str) -> dict[str, np.ndarray]:
    return load_cstim_feature_groups(model, dtype=np.float32)


def make_splits(n_images: int, n_splits: int, train_fraction: float, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    n_train = int(round(n_images * train_fraction))
    splits = []
    for split_id in range(n_splits):
        perm = rng.permutation(n_images)
        train = np.sort(perm[:n_train])
        heldout = np.sort(perm[n_train:])
        splits.append((train, heldout))
    return splits


def fit_fold_encoding(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval_blocks: list[np.ndarray],
    alpha: float,
) -> list[np.ndarray]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    x_train_z = scaler.fit_transform(x_train).astype(np.float32)
    eval_z = [scaler.transform(x).astype(np.float32) for x in x_eval_blocks]
    ridge = Ridge(alpha=float(alpha), fit_intercept=True)
    ridge.fit(x_train_z, y_train)
    return [ridge.predict(x).astype(np.float32) for x in eval_z]


def rsa_score(brain_patterns: np.ndarray, pred_patterns: np.ndarray) -> float:
    return compute_rsa_score(
        compute_rdm_correlation(pred_patterns),
        compute_rdm_correlation(brain_patterns),
        method="spearman",
    )


def nc_value(subject: str, group: str, stimulus_type: str) -> float:
    path = SHARE_ROOT / "02_alignment_reliability" / "results" / "rdm_noise_ceilings.csv"
    if not path.exists() or stimulus_type == "heldout_unique":
        return np.nan
    nc = pd.read_csv(path)
    if stimulus_type == "same_session":
        sub = nc[(nc["subject"] == subject) & (nc["stimulus_type"] == "vicco")]
        return float(sub["noise_ceiling_spearman"].mean()) if not sub.empty else np.nan
    sub = nc[(nc["subject"] == subject) & (nc["group"] == group) & (nc["stimulus_type"] == "controversial")]
    return float(sub["noise_ceiling_spearman"].iloc[0]) if not sub.empty else np.nan


def nc_norm(score: float, nc: float) -> float:
    if not np.isfinite(nc) or nc <= 0:
        return np.nan
    return float(score / np.sqrt(nc))


def embedding_pc_covariates(
    cstim_features: np.ndarray,
    heldout_features: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    n_components = int(min(n_components, cstim_features.shape[0] - 1, heldout_features.shape[0] - 1, cstim_features.shape[1]))
    scaler = StandardScaler()
    combined = scaler.fit_transform(np.vstack([cstim_features, heldout_features]))
    pca = PCA(n_components=n_components, random_state=0)
    pcs = pca.fit_transform(combined)
    cstim_pc = pcs[: len(cstim_features)]
    held_pc = pcs[len(cstim_features) :]
    pre = float(np.linalg.norm(cstim_pc.mean(axis=0) - held_pc.mean(axis=0)))
    diag = {
        "n_components": n_components,
        "pre_match_centroid_distance": pre,
        "common_support_metric": float(cdist(cstim_pc, held_pc, metric="euclidean").min(axis=1).mean()),
    }
    return cstim_pc, held_pc, diag


def greedy_match_indices(
    cstim_covariates: np.ndarray,
    heldout_covariates: np.ndarray,
) -> tuple[np.ndarray, dict]:
    cstim_covariates = np.asarray(cstim_covariates, dtype=np.float64)
    heldout_covariates = np.asarray(heldout_covariates, dtype=np.float64)
    keep = np.isfinite(cstim_covariates).all(axis=0) & np.isfinite(heldout_covariates).all(axis=0)
    cstim_covariates = cstim_covariates[:, keep]
    heldout_covariates = heldout_covariates[:, keep]
    if cstim_covariates.shape[1] == 0:
        raise ValueError("no finite covariate columns available for matching")
    scaler = StandardScaler()
    combined = scaler.fit_transform(np.vstack([cstim_covariates, heldout_covariates]))
    cstim_z = combined[: len(cstim_covariates)]
    heldout_z = combined[len(cstim_covariates) :]
    dist = cdist(cstim_z, heldout_z, metric="euclidean")
    selected = []
    used = set()
    for i in np.argsort(dist.min(axis=1)):
        for j in np.argsort(dist[i]):
            if int(j) not in used:
                selected.append(int(j))
                used.add(int(j))
                break
    selected = np.array(selected[: len(cstim_covariates)], dtype=int)
    pre = float(np.linalg.norm(cstim_z.mean(axis=0) - heldout_z.mean(axis=0)))
    post = float(np.linalg.norm(cstim_z.mean(axis=0) - heldout_z[selected].mean(axis=0)))
    diag = {
        "n_covariates": int(cstim_z.shape[1]),
        "pre_match_centroid_distance": pre,
        "post_match_centroid_distance": post,
        "common_support_metric": float(dist.min(axis=1).mean()),
    }
    return selected, diag


def ppca_loglik(train: np.ndarray, blocks: list[np.ndarray], n_components: int) -> tuple[list[np.ndarray], dict]:
    train = np.asarray(train, dtype=np.float32)
    n_components = int(min(n_components, train.shape[0] - 2, train.shape[1] - 1))
    if n_components < 2:
        raise ValueError("not enough samples/features for PPCA OOD scoring")
    scaler = StandardScaler()
    train_z = scaler.fit_transform(train)
    blocks_z = [scaler.transform(np.asarray(block, dtype=np.float32)) for block in blocks]
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    pca.fit(train_z)
    train_ll = pca.score_samples(train_z)
    mu = float(np.nanmean(train_ll))
    sd = float(np.nanstd(train_ll, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0
    out = [((pca.score_samples(block) - mu) / sd).astype(np.float32) for block in blocks_z]
    diag = {
        "n_components": n_components,
        "train_loglik_mean": mu,
        "train_loglik_sd": sd,
        "explained_variance": float(np.nansum(pca.explained_variance_ratio_)),
    }
    return out, diag


def load_unique_low_level(subject: str) -> pd.DataFrame | None:
    if not UNIQUE_LOW_LEVEL_PATH.exists():
        return None
    df = pd.read_csv(UNIQUE_LOW_LEVEL_PATH)
    sub = df[df["subject"] == subject].sort_values("image_idx").reset_index(drop=True)
    if sub.empty:
        return None
    return sub


def load_cstim_low_level() -> pd.DataFrame | None:
    if not CSTIM_LOW_LEVEL_PATH.exists():
        return None
    return pd.read_csv(CSTIM_LOW_LEVEL_PATH)


def cstim_low_level_block(stats: pd.DataFrame | None, model_set: str, file_idx: np.ndarray) -> np.ndarray | None:
    if stats is None:
        return None
    sub = stats[stats["stimulus_set"] == model_set].reset_index(drop=True)
    if len(sub) <= int(np.max(file_idx)):
        return None
    return sub.iloc[file_idx][LOW_LEVEL_COLS].to_numpy(dtype=np.float32)


def unique_low_level_block(stats: pd.DataFrame | None, indices: np.ndarray) -> np.ndarray | None:
    if stats is None or len(stats) <= int(np.max(indices)):
        return None
    return stats.iloc[indices][LOW_LEVEL_COLS].to_numpy(dtype=np.float32)


def score_heldout_subset(
    unique_betas: np.ndarray,
    pred_heldout_all: np.ndarray,
    heldout_idx: np.ndarray,
    rel_idx: np.ndarray,
) -> float:
    absolute_idx = heldout_idx[rel_idx]
    brain = unique_betas[:, absolute_idx].T
    return rsa_score(brain, pred_heldout_all[rel_idx])


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def summarize_endpoints() -> None:
    path = OUT_DIR / "heldout_unique_baseline_results.csv"
    if not path.exists():
        return
    frames = [pd.read_csv(path)]
    matched_path = OUT_DIR / "matched_baseline_results.csv"
    if matched_path.exists():
        frames.append(pd.read_csv(matched_path))
    df = pd.concat(frames, ignore_index=True, sort=False)
    columns = [
        "subject", "split_id", "train_fraction", "model_set", "metric",
        "method_source", "baseline_type", "n_models", "score_cstim",
        "score_baseline", "delta", "score_cstim_NCnorm",
        "score_baseline_NCnorm", "delta_NCnorm", "NC_cstim_mean",
        "NC_baseline_mean", "spread_ratio", "primary_alignment_endpoint",
        "primary_spread_endpoint",
    ]
    rows = []
    for (subject, split_id, train_fraction, model_set), grp in df.groupby(
        ["subject", "split_id", "train_fraction", "model_set"]
    ):
        cstim = grp[grp["baseline_type"] == "controversial"]
        if cstim.empty:
            continue
        c_model = cstim.groupby("model")["score"].mean()
        for baseline_type in sorted(set(grp["baseline_type"]) - {"controversial"}):
            base = grp[grp["baseline_type"] == baseline_type]
            b_model = base.groupby("model")["score"].mean()
            common = sorted(set(c_model.index) & set(b_model.index))
            if len(common) < 2:
                continue
            rows.append(
                {
                    "subject": subject,
                    "split_id": split_id,
                    "train_fraction": train_fraction,
                    "model_set": model_set,
                    "metric": "mixed_RSA",
                    "method_source": "fold_specific_ridge_mixed_rsa",
                    "baseline_type": baseline_type,
                    "n_models": len(common),
                    "score_cstim": float(c_model.loc[common].mean()),
                    "score_baseline": float(b_model.loc[common].mean()),
                    "delta": float(c_model.loc[common].mean() - b_model.loc[common].mean()),
                    "score_cstim_NCnorm": np.nan,
                    "score_baseline_NCnorm": np.nan,
                    "delta_NCnorm": np.nan,
                    "NC_cstim_mean": np.nan,
                    "NC_baseline_mean": np.nan,
                    "spread_ratio": np.nan,
                    "primary_alignment_endpoint": baseline_type in {
                        "heldout_unique",
                        "heldout_unique_matched_low_level",
                        "heldout_unique_matched_embedding_pc",
                        "heldout_unique_matched_ppca_ood",
                        "heldout_unique_matched_combined",
                        "heldout_unique_high_ppca_ood",
                    },
                    "primary_spread_endpoint": False,
                }
            )
    by_split = pd.DataFrame(rows, columns=columns)
    by_split.to_csv(OUT_DIR / "heldout_unique_endpoint_by_split.csv", index=False)
    if by_split.empty:
        by_split.to_csv(OUT_DIR / "heldout_unique_endpoint_summary.csv", index=False)
        return

    group_cols = [
        "subject",
        "train_fraction",
        "model_set",
        "metric",
        "method_source",
        "baseline_type",
        "primary_alignment_endpoint",
        "primary_spread_endpoint",
    ]
    value_cols = [
        "score_cstim",
        "score_baseline",
        "delta",
        "score_cstim_NCnorm",
        "score_baseline_NCnorm",
        "delta_NCnorm",
        "NC_cstim_mean",
        "NC_baseline_mean",
        "spread_ratio",
    ]
    summary = (
        by_split.groupby(group_cols, as_index=False, dropna=False)
        .agg(
            n_splits=("split_id", "nunique"),
            n_models=("n_models", "median"),
            **{col: (col, "mean") for col in value_cols},
        )
        .sort_values(["model_set", "baseline_type", "subject"])
    )
    summary["n_models"] = summary["n_models"].astype(int)
    summary.to_csv(OUT_DIR / "heldout_unique_endpoint_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=constants.SUBJECTS)
    parser.add_argument("--model-sets", nargs="+", default=MODEL_SETS)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--baseline-samples", type=int, default=10)
    parser.add_argument("--embedding-components", type=int, default=20)
    parser.add_argument("--ppca-components", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outputs = [
        OUT_DIR / "heldout_unique_splits.csv",
        OUT_DIR / "heldout_unique_baseline_results.csv",
        OUT_DIR / "baseline_matching_diagnostics.csv",
        OUT_DIR / "matched_baseline_results.csv",
        OUT_DIR / "heldout_unique_endpoint_summary.csv",
        OUT_DIR / "heldout_unique_endpoint_by_split.csv",
    ]
    if args.overwrite:
        for p in outputs:
            if p.exists():
                p.unlink()

    split_rows = []
    result_rows = []
    diag_rows = []
    matched_rows = []
    cstim_low_level_stats = load_cstim_low_level()

    for subject in args.subjects:
        cstim_brain = load_cstim_brain(subject)
        unique_low_level_stats = load_unique_low_level(subject)
        model_union = sorted(set(m for ms in args.model_sets for m in constants.MODEL_SETS[ms]))
        if args.max_models is not None:
            model_union = model_union[: args.max_models]

        for model in tqdm(model_union, desc=subject):
            try:
                encoding = load_encoding_model(model, subject)
                roi = encoding["roi_hlvis"].astype(bool)
                unique_features = load_unique_features(subject, model)
                unique_betas = load_unique_betas(subject, roi)
                cstim_features = load_cstim_features(model)
                alpha = load_median_hlvis_alpha(subject, model, roi)
            except Exception as exc:
                print(f"skip {subject} {model}: {exc}")
                continue

            splits = make_splits(unique_features.shape[0], args.n_splits, args.train_fraction, args.seed)
            rng = np.random.default_rng(args.seed)
            for split_id, (train_idx, heldout_idx) in enumerate(splits):
                split_rows.append(
                    {
                        "subject": subject,
                        "split_id": split_id,
                        "train_fraction": args.train_fraction,
                        "n_train": len(train_idx),
                        "n_heldout": len(heldout_idx),
                        "seed": args.seed,
                    }
                )
                y_train = unique_betas[:, train_idx].T
                heldout_x_all = unique_features[heldout_idx]
                vicco_x_all = cstim_features["vicco"][cstim_brain["group_file_idx"]["vicco"]]

                cstim_x_by_set = {}
                file_idx_by_set = {}
                for model_set in args.model_sets:
                    if model not in constants.MODEL_SETS[model_set] or model_set not in cstim_brain["group_indices"]:
                        continue
                    file_idx = cstim_brain["group_file_idx"][model_set]
                    file_idx_by_set[model_set] = file_idx
                    cstim_x_by_set[model_set] = cstim_features[model_set][file_idx]

                if not cstim_x_by_set:
                    append_csv(OUT_DIR / "heldout_unique_splits.csv", split_rows)
                    split_rows.clear()
                    continue

                # Fit once per split/model and reuse predictions across model-set endpoints.
                eval_blocks = [vicco_x_all, heldout_x_all] + [cstim_x_by_set[ms] for ms in cstim_x_by_set]
                preds = fit_fold_encoding(
                    unique_features[train_idx],
                    y_train,
                    eval_blocks,
                    alpha=alpha,
                )
                pred_vicco_all = preds[0]
                pred_heldout_all = preds[1]
                pred_cstim_by_set = dict(zip(cstim_x_by_set.keys(), preds[2:], strict=True))

                cstim_ppca_by_set = {}
                heldout_ppca = None
                ppca_diag = {}
                try:
                    ppca_blocks, ppca_diag = ppca_loglik(
                        unique_features[train_idx],
                        [heldout_x_all] + [cstim_x_by_set[ms] for ms in cstim_x_by_set],
                        args.ppca_components,
                    )
                    heldout_ppca = (-ppca_blocks[0])[:, None]
                    cstim_ppca_by_set = {
                        model_set: (-block)[:, None]
                        for model_set, block in zip(cstim_x_by_set.keys(), ppca_blocks[1:], strict=True)
                    }
                except Exception as exc:
                    ppca_diag = {"status_detail": f"ppca_failed: {exc}"}

                for model_set, cstim_x in cstim_x_by_set.items():
                    file_idx = file_idx_by_set[model_set]
                    heldout_rel_random = np.sort(rng.choice(len(heldout_idx), size=100, replace=False))
                    pred_cstim = pred_cstim_by_set[model_set]

                    brain_cstim = cstim_brain["betas_hlvis"][:, cstim_brain["group_indices"][model_set]].T
                    score_c = rsa_score(brain_cstim, pred_cstim)
                    nc_c = nc_value(subject, model_set, "controversial")
                    result_rows.append(
                        {
                            "subject": subject,
                            "split_id": split_id,
                            "train_fraction": args.train_fraction,
                            "model_set": model_set,
                            "model": model,
                            "metric": "mixed_RSA",
                            "baseline_type": "controversial",
                            "score": score_c,
                            "noise_ceiling": nc_c,
                            "score_NCnorm": nc_norm(score_c, nc_c),
                            "n_stimuli": 100,
                            "encoding_fit_mode": "fold_specific_ridge_median_alpha",
                        }
                    )

                    # Same-session baseline subsets.
                    n_vicco = pred_vicco_all.shape[0]
                    for b in range(args.baseline_samples):
                        subset = np.sort(rng.choice(n_vicco, size=100, replace=False))
                        brain_v = cstim_brain["betas_hlvis"][:, cstim_brain["group_indices"]["vicco"][subset]].T
                        score_v = rsa_score(brain_v, pred_vicco_all[subset])
                        nc_v = nc_value(subject, model_set, "same_session")
                        result_rows.append(
                            {
                                "subject": subject,
                                "split_id": split_id,
                                "train_fraction": args.train_fraction,
                                "model_set": model_set,
                                "model": model,
                                "metric": "mixed_RSA",
                                "baseline_type": "same_session_unselected",
                                "score": score_v,
                                "noise_ceiling": nc_v,
                                "score_NCnorm": nc_norm(score_v, nc_v),
                                "n_stimuli": 100,
                                "baseline_sample": b,
                                "encoding_fit_mode": "fold_specific_ridge_median_alpha",
                            }
                        )

                    score_h = score_heldout_subset(unique_betas, pred_heldout_all, heldout_idx, heldout_rel_random)
                    result_rows.append(
                        {
                            "subject": subject,
                            "split_id": split_id,
                            "train_fraction": args.train_fraction,
                            "model_set": model_set,
                            "model": model,
                            "metric": "mixed_RSA",
                            "baseline_type": "heldout_unique",
                            "score": score_h,
                            "noise_ceiling": np.nan,
                            "score_NCnorm": np.nan,
                            "n_stimuli": 100,
                            "baseline_sample": 0,
                            "encoding_fit_mode": "fold_specific_ridge_median_alpha",
                        }
                    )

                    cstim_pc, heldout_pc, embedding_diag = embedding_pc_covariates(
                        cstim_x, heldout_x_all, args.embedding_components
                    )
                    rel_match, diag = greedy_match_indices(cstim_pc, heldout_pc)
                    diag = {**embedding_diag, **diag}
                    score_m = score_heldout_subset(unique_betas, pred_heldout_all, heldout_idx, rel_match)
                    diag_rows.append(
                        {
                            "subject": subject,
                            "split_id": split_id,
                            "model_set": model_set,
                            "model": model,
                            "baseline_pool": "heldout_unique",
                            "match_type": "embedding_pc",
                            "covariate_family": "model_feature_pcs",
                            **diag,
                            "status": "ok",
                        }
                    )
                    matched_rows.append(
                        {
                            "subject": subject,
                            "split_id": split_id,
                            "train_fraction": args.train_fraction,
                            "model_set": model_set,
                            "model": model,
                            "metric": "mixed_RSA",
                            "baseline_type": "heldout_unique_matched_embedding_pc",
                            "score": score_m,
                            "noise_ceiling": np.nan,
                            "score_NCnorm": np.nan,
                            "n_stimuli": len(rel_match),
                            "encoding_fit_mode": "fold_specific_ridge_median_alpha",
                        }
                    )

                    cstim_low = cstim_low_level_block(cstim_low_level_stats, model_set, file_idx)
                    heldout_low = unique_low_level_block(unique_low_level_stats, heldout_idx)
                    cstim_ppca = cstim_ppca_by_set.get(model_set)

                    match_specs = []
                    if cstim_low is not None and heldout_low is not None:
                        match_specs.append(("low_level", "low_level_stats", cstim_low, heldout_low))
                    else:
                        diag_rows.append(
                            {
                                "subject": subject,
                                "split_id": split_id,
                                "model_set": model_set,
                                "model": model,
                                "baseline_pool": "heldout_unique",
                                "match_type": "low_level",
                                "covariate_family": "low_level_stats",
                                "status": "not_run_missing_unique_low_level_stats",
                            }
                        )
                    if cstim_ppca is not None and heldout_ppca is not None:
                        match_specs.append(("ppca_ood", "feature_ppca_ood", cstim_ppca, heldout_ppca))
                        high_rel = np.argsort(heldout_ppca[:, 0])[::-1][:100]
                        score_high = score_heldout_subset(unique_betas, pred_heldout_all, heldout_idx, high_rel)
                        matched_rows.append(
                            {
                                "subject": subject,
                                "split_id": split_id,
                                "train_fraction": args.train_fraction,
                                "model_set": model_set,
                                "model": model,
                                "metric": "mixed_RSA",
                                "baseline_type": "heldout_unique_high_ppca_ood",
                                "score": score_high,
                                "noise_ceiling": np.nan,
                                "score_NCnorm": np.nan,
                                "n_stimuli": len(high_rel),
                                "encoding_fit_mode": "fold_specific_ridge_median_alpha",
                            }
                        )
                        diag_rows.append(
                            {
                                "subject": subject,
                                "split_id": split_id,
                                "model_set": model_set,
                                "model": model,
                                "baseline_pool": "heldout_unique",
                                "match_type": "high_ppca_ood",
                                "covariate_family": "feature_ppca_ood",
                                "n_components": ppca_diag.get("n_components"),
                                "common_support_metric": float(np.nanmean(heldout_ppca[high_rel, 0])),
                                "status": "ok",
                            }
                        )
                    else:
                        diag_rows.append(
                            {
                                "subject": subject,
                                "split_id": split_id,
                                "model_set": model_set,
                                "model": model,
                                "baseline_pool": "heldout_unique",
                                "match_type": "ppca_ood",
                                "covariate_family": "feature_ppca_ood",
                                "status": "not_run_ppca_failed",
                                **ppca_diag,
                            }
                        )
                    if (
                        cstim_low is not None
                        and heldout_low is not None
                        and cstim_ppca is not None
                        and heldout_ppca is not None
                    ):
                        match_specs.append(
                            (
                                "combined",
                                "low_level_embedding_pc_feature_ppca_ood",
                                np.hstack([cstim_low, cstim_pc, cstim_ppca]),
                                np.hstack([heldout_low, heldout_pc, heldout_ppca]),
                            )
                        )
                    else:
                        diag_rows.append(
                            {
                                "subject": subject,
                                "split_id": split_id,
                                "model_set": model_set,
                                "model": model,
                                "baseline_pool": "heldout_unique",
                                "match_type": "combined",
                                "covariate_family": "low_level_embedding_pc_feature_ppca_ood",
                                "status": "not_run_missing_component_covariates",
                            }
                        )

                    for match_type, cov_family, c_cov, h_cov in match_specs:
                        rel_idx, match_diag = greedy_match_indices(c_cov, h_cov)
                        score_match = score_heldout_subset(unique_betas, pred_heldout_all, heldout_idx, rel_idx)
                        diag_rows.append(
                            {
                                "subject": subject,
                                "split_id": split_id,
                                "model_set": model_set,
                                "model": model,
                                "baseline_pool": "heldout_unique",
                                "match_type": match_type,
                                "covariate_family": cov_family,
                                **match_diag,
                                "status": "ok",
                            }
                        )
                        matched_rows.append(
                            {
                                "subject": subject,
                                "split_id": split_id,
                                "train_fraction": args.train_fraction,
                                "model_set": model_set,
                                "model": model,
                                "metric": "mixed_RSA",
                                "baseline_type": f"heldout_unique_matched_{match_type}",
                                "score": score_match,
                                "noise_ceiling": np.nan,
                                "score_NCnorm": np.nan,
                                "n_stimuli": len(rel_idx),
                                "encoding_fit_mode": "fold_specific_ridge_median_alpha",
                            }
                        )

                append_csv(OUT_DIR / "heldout_unique_splits.csv", split_rows)
                append_csv(OUT_DIR / "heldout_unique_baseline_results.csv", result_rows)
                append_csv(OUT_DIR / "baseline_matching_diagnostics.csv", diag_rows)
                append_csv(OUT_DIR / "matched_baseline_results.csv", matched_rows)
                split_rows.clear()
                result_rows.clear()
                diag_rows.clear()
                matched_rows.clear()

    summarize_endpoints()
    splits_path = OUT_DIR / "heldout_unique_splits.csv"
    if splits_path.exists():
        splits = pd.read_csv(splits_path).drop_duplicates(
            ["subject", "split_id", "train_fraction", "seed"]
        )
        splits.to_csv(splits_path, index=False)
    print(f"saved outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
