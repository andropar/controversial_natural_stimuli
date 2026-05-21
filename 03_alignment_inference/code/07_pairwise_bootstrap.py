#!/usr/bin/env python3
"""
Pairwise model comparison via stimulus bootstrap.

NOT USED IN THE CURRENT PAPER (no figure or main-text number in
writing/cstims_paper/ consumes the output CSVs as of this audit).
If this analysis is promoted, two bugs must be fixed first:
  (a) L~181 computes mRSA from the FULL predicted voxel vector without
      masking to encoding["roi_hlvis"] as every other mRSA pipeline does.
  (b) L~244 labels its bootstrap quantity a "p_value"; this is a
      bootstrap-interval-style quantity, not a permutation null test.

For each pair of models within a model set, test whether their brain
alignment scores differ significantly. Bootstrap over stimuli (resample
with replacement), recompute RDMs and RSA scores, compute pairwise
differences. Apply Benjamini-Hochberg FDR correction.

Runs for both fRSA and mRSA, for all model sets + all_models.

Outputs (in data/):
    pairwise_bootstrap_results.csv  - All pairwise comparisons with p-values
    pairwise_bootstrap_summary.csv  - FDR-corrected summary

Usage:
    python 13_pairwise_bootstrap_tests.py [--n-bootstrap 10000]
"""

import argparse
import sys
from pathlib import Path
from itertools import combinations

# Setup imports from cstim_paper root
_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))  # project root for cstims

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

import config
from utils import (
    get_encoding_folder,
    compute_rdm_correlation,
    rdm_to_vector,
    load_encoding_model,
    predict_voxel_responses,
)

# Cached features
FEATURE_CACHE = config.CONSENSUS_DATA_DIR / "features"

# VICReg excluded from mRSA
VERSA_EXCLUDED = {"vicreg_resnet50"}


# ===========================================================================
# Data loading
# ===========================================================================

def load_cached_features(model_set: str, model: str) -> np.ndarray:
    """Load pre-extracted features from consensus_mystery cache."""
    path = FEATURE_CACHE / model_set / f"{model}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Cached features not found: {path}")
    return np.load(path)["features"]


def load_brain_data(subject: str, model_set: str):
    """Load brain betas for a subject and model set, return brain RDM indices."""
    data_dir = config.get_subject_data_dir(subject)
    betas_data = np.load(data_dir / "cstim_betas_averaged.npz", allow_pickle=True)
    voxel_data = np.load(data_dir / "voxel_metadata.npz", allow_pickle=True)
    stim_info = pd.read_csv(data_dir / "cstim_stimulus_info.csv")

    hlvis_mask = voxel_data["hlvis_mask"]
    betas_hlvis = betas_data["betas"][hlvis_mask, :]
    stim_keys = betas_data["stim_keys"]
    stim_key_to_idx = {k: i for i, k in enumerate(stim_keys)}

    # Get indices for this model set
    mask = stim_info["group"] == model_set
    if mask.sum() == 0:
        return None, None, None

    keys = stim_info.loc[mask, "stim_key"].values
    brain_idx = np.array([stim_key_to_idx[k] for k in keys])
    stim_idx = stim_info.loc[mask, "stim_idx"].values

    # Brain betas for this stimulus set (n_voxels, n_stimuli)
    brain_betas = betas_hlvis[:, brain_idx]

    return brain_betas, stim_idx, len(brain_idx)


# ===========================================================================
# RDM computation
# ===========================================================================

def compute_rdm_upper_tri(data: np.ndarray) -> np.ndarray:
    """Compute RDM (correlation distance) and return upper triangle vector."""
    rdm = compute_rdm_correlation(data)
    return rdm_to_vector(rdm)


# ===========================================================================
# Main bootstrap
# ===========================================================================

def run_bootstrap_for_model_set(model_set: str, method: str, n_bootstrap: int,
                                 rng: np.random.Generator) -> list:
    """
    Run stimulus bootstrap for all model pairs in a model set.

    Returns list of dicts with pairwise results.
    """
    models = config.MODEL_SETS[model_set]
    if method == "mRSA":
        models = [m for m in models if m not in VERSA_EXCLUDED]

    if len(models) < 2:
        return []

    results = []

    # For each subject: compute full RDM vectors, then bootstrap
    for subject in config.SUBJECTS:
        brain_betas, stim_idx, n_stim = load_brain_data(subject, model_set)
        if brain_betas is None:
            continue

        # Compute brain RDM vector
        brain_rdm_vec = compute_rdm_upper_tri(brain_betas.T)  # T: (n_stim, n_vox)

        # Compute model RDM vectors
        model_rdm_vecs = {}
        for model in models:
            try:
                features = load_cached_features(model_set, model)
                # Subset to this subject's available stimuli
                features_sub = features[stim_idx]
            except FileNotFoundError:
                continue

            if method == "mRSA":
                try:
                    encoding = load_encoding_model(model, subject)
                except FileNotFoundError:
                    continue
                predicted = predict_voxel_responses(features_sub, encoding)
                model_rdm_vecs[model] = compute_rdm_upper_tri(predicted)
            else:  # fRSA
                model_rdm_vecs[model] = compute_rdm_upper_tri(features_sub)

        available_models = [m for m in models if m in model_rdm_vecs]
        if len(available_models) < 2:
            continue

        # Observed scores
        observed_scores = {}
        for model in available_models:
            r, _ = stats.spearmanr(model_rdm_vecs[model], brain_rdm_vec)
            observed_scores[model] = r

        # Bootstrap: resample stimulus indices
        boot_scores = {m: np.zeros(n_bootstrap) for m in available_models}
        n = n_stim

        # Pre-reconstruct full RDMs for efficiency
        full_rdms = {}
        tri = np.triu_indices(n, k=1)
        for model in available_models:
            rdm = np.zeros((n, n))
            rdm[tri] = model_rdm_vecs[model]
            rdm += rdm.T
            full_rdms[model] = rdm

        brain_rdm_full = np.zeros((n, n))
        brain_rdm_full[tri] = brain_rdm_vec
        brain_rdm_full += brain_rdm_full.T

        for b in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            brain_sub = brain_rdm_full[np.ix_(idx, idx)]
            sub_tri = np.triu_indices(len(idx), k=1)
            brain_vec = brain_sub[sub_tri]

            for model in available_models:
                model_sub = full_rdms[model][np.ix_(idx, idx)]
                model_vec = model_sub[sub_tri]
                r, _ = stats.spearmanr(model_vec, brain_vec)
                boot_scores[model][b] = r

        # Store per-subject pairwise results
        for m1, m2 in combinations(available_models, 2):
            obs_diff = observed_scores[m1] - observed_scores[m2]
            boot_diffs = boot_scores[m1] - boot_scores[m2]

            results.append({
                "model_set": model_set,
                "method": method,
                "subject": subject,
                "model_1": m1,
                "model_2": m2,
                "display_1": config.MODEL_DISPLAY_NAMES.get(m1, m1),
                "display_2": config.MODEL_DISPLAY_NAMES.get(m2, m2),
                "score_1": observed_scores[m1],
                "score_2": observed_scores[m2],
                "observed_diff": obs_diff,
                "boot_mean_diff": boot_diffs.mean(),
                "boot_std_diff": boot_diffs.std(),
                "boot_ci_lo": np.percentile(boot_diffs, 2.5),
                "boot_ci_hi": np.percentile(boot_diffs, 97.5),
                # Two-sided p-value: fraction of bootstrap diffs on wrong side of 0
                "p_value": 2 * min(np.mean(boot_diffs <= 0),
                                   np.mean(boot_diffs >= 0)),
                "n_stimuli": n,
                "n_bootstrap": n_bootstrap,
            })

    return results


def combine_subjects_and_correct(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine per-subject p-values using Fisher's method.
    Apply Benjamini-Hochberg FDR correction within each model_set × method.
    """
    if results_df.empty:
        return pd.DataFrame()

    # Group by model_set, method, model pair
    group_cols = ["model_set", "method", "model_1", "model_2",
                  "display_1", "display_2"]
    summary = []

    for keys, grp in results_df.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["n_subjects"] = len(grp)
        row["mean_diff"] = grp["observed_diff"].mean()
        row["sem_diff"] = grp["observed_diff"].sem()
        row["mean_score_1"] = grp["score_1"].mean()
        row["mean_score_2"] = grp["score_2"].mean()

        # Fisher's method to combine per-subject p-values
        p_vals = grp["p_value"].values
        # Clamp p-values away from 0 to avoid log(0)
        p_vals = np.clip(p_vals, 1e-10, 1.0)
        chi2_stat = -2 * np.sum(np.log(p_vals))
        fisher_p = 1 - stats.chi2.cdf(chi2_stat, df=2 * len(p_vals))
        row["fisher_p"] = fisher_p

        # Also store per-subject bootstrap CI overlap
        ci_los = grp["boot_ci_lo"].values
        ci_his = grp["boot_ci_hi"].values
        row["all_ci_exclude_zero"] = all(
            (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
            for lo, hi in zip(ci_los, ci_his)
        )

        summary.append(row)

    summary_df = pd.DataFrame(summary)

    # Benjamini-Hochberg FDR correction within each model_set × method
    summary_df["fdr_p"] = np.nan
    summary_df["significant_fdr05"] = False

    for (ms, method), grp in summary_df.groupby(["model_set", "method"]):
        idx = grp.index
        p_values = grp["fisher_p"].values
        n_tests = len(p_values)
        if n_tests == 0:
            continue

        # BH procedure
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        rank = np.arange(1, n_tests + 1)
        bh_threshold = rank / n_tests * 0.05

        # Find largest k where p_(k) <= k/m * alpha
        significant = sorted_p <= bh_threshold
        if significant.any():
            max_k = np.max(np.where(significant)[0])
            sig_mask = np.zeros(n_tests, dtype=bool)
            sig_mask[sorted_idx[:max_k + 1]] = True
        else:
            sig_mask = np.zeros(n_tests, dtype=bool)

        # Adjusted p-values (BH)
        adjusted_p = np.zeros(n_tests)
        adjusted_p[sorted_idx[-1]] = sorted_p[-1]
        for i in range(n_tests - 2, -1, -1):
            adjusted_p[sorted_idx[i]] = min(
                adjusted_p[sorted_idx[i + 1]],
                sorted_p[i] * n_tests / (i + 1)
            )
        adjusted_p = np.clip(adjusted_p, 0, 1)

        summary_df.loc[idx, "fdr_p"] = adjusted_p
        summary_df.loc[idx, "significant_fdr05"] = sig_mask

    return summary_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=10000,
                        help="Number of bootstrap iterations (default: 10000)")
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    all_model_sets = ["sota", "training_objective", "architecture", "dataset", "all_models"]
    all_methods = ["fRSA", "mRSA"]

    all_results = []

    for model_set in all_model_sets:
        for method in all_methods:
            n_models = len(config.MODEL_SETS[model_set])
            if method == "mRSA":
                n_models -= sum(1 for m in config.MODEL_SETS[model_set] if m in VERSA_EXCLUDED)
            n_pairs = n_models * (n_models - 1) // 2

            print(f"\n{'='*60}")
            print(f"{model_set} / {method}: {n_models} models, {n_pairs} pairs")
            print(f"{'='*60}")

            results = run_bootstrap_for_model_set(
                model_set, method, args.n_bootstrap, rng
            )
            all_results.extend(results)
            print(f"  Got {len(results)} per-subject pairwise results")

    # Save per-subject results
    results_df = pd.DataFrame(all_results)
    out_path = config.STATS_DATA_DIR / "pairwise_bootstrap_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved per-subject results: {out_path} ({len(results_df)} rows)")

    # Combine across subjects and apply FDR
    summary_df = combine_subjects_and_correct(results_df)
    summary_path = config.STATS_DATA_DIR / "pairwise_bootstrap_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path} ({len(summary_df)} rows)")

    # Print summary
    for (ms, method), grp in summary_df.groupby(["model_set", "method"]):
        n_sig = grp["significant_fdr05"].sum()
        n_total = len(grp)
        print(f"\n{ms} / {method}: {n_sig}/{n_total} significant (FDR < 0.05)")
        if n_sig > 0:
            sig = grp[grp["significant_fdr05"]].sort_values("mean_diff", ascending=False)
            for _, row in sig.head(5).iterrows():
                print(f"  {row['display_1']} > {row['display_2']}: "
                      f"Δ = {row['mean_diff']:.4f}, FDR p = {row['fdr_p']:.4f}")


if __name__ == "__main__":
    main()
