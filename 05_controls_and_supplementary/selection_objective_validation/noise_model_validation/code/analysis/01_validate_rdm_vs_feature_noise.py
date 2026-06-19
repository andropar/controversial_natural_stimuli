#!/usr/bin/env python3
"""Validate that RDM-space noise injection approximates feature-space noise injection.

The stimulus selection pipeline adds Gaussian noise directly to RDM vectors
and uses an analytical attenuation formula. This script validates that this
approximation produces comparable results to a feature-space Monte Carlo
reference that adds noise to feature vectors and recomputes RDMs.

Four validation tests:
1. Noise ceiling equivalence
2. Correlation matrix agreement
3. Model discriminability agreement
4. Candidate ranking preservation
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from cstims.data_loader import load_natural_features_with_metadata
from cstims.evaluation.model_discrimination import (
    calibrate_feature_noise,
    model_discriminability,
)
from cstims.noise_estimation import rdm_noise_by_model
from cstims.rdm import get_rdm_vector
from cstims.selection.primitives import (
    aggregate_across_models,
    compute_correlation_matrix,
    compute_model_utilities,
)
from cstims.selection.utility import compute_analytical_utility

SCRIPT_DIR = Path(__file__).resolve().parent
SECTION_DIR = SCRIPT_DIR.parent
DATA_DIR = SECTION_DIR / "results"
FIGURES_DIR = SECTION_DIR / "figures"

MODEL_NAMES = [
    "vissl_resnet50_supervised",
    "vissl_resnet50_barlowtwins",
    "vissl_resnet50_mocov2",
    "vicreg_resnet50",
    "robustness_imagenet_l2_eps3",
]

NC_TARGETS = [0.3, 0.46, 0.7, 0.9]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metric", default="correlation", choices=["correlation", "cosine", "euclidean"])
    p.add_argument("--corr-type", default="correlation", choices=["correlation", "spearman"])
    p.add_argument("--n-mc", type=int, default=100, help="MC repeats for tests 1-3")
    p.add_argument("--n-mc-ranking", type=int, default=32, help="MC repeats for test 4")
    p.add_argument("--n-selected", type=int, default=100)
    p.add_argument("--n-candidates", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_features(n_calib: int, n_selected: int, n_candidates: int, seed: int):
    """Load features and split into calibration, selected, candidate sets."""
    subset_root = Path("/SSD/datasets/cstims_laion_natural_subset")
    memmap_dir = Path("/SSD/datasets/cstims_laion_natural_subset_memmaps")
    model_csv = Path(__file__).resolve().parents[5] / "00_stimulus_selection" / "resources" / "model_list.csv"

    total_needed = n_calib + n_selected + n_candidates
    print(f"Loading {total_needed} images for {len(MODEL_NAMES)} models...")

    features_np, _ = load_natural_features_with_metadata(
        subset_root=subset_root,
        model_names=MODEL_NAMES,
        model_csv=model_csv,
        max_images=total_needed + 500,  # extra buffer
        preprocessed_dir=memmap_dir,
    )

    rng = np.random.default_rng(seed)
    n_available = features_np[MODEL_NAMES[0]].shape[0]
    all_idx = rng.choice(n_available, size=total_needed, replace=False)

    calib_idx = all_idx[:n_calib]
    selected_idx = all_idx[n_calib : n_calib + n_selected]
    candidate_idx = all_idx[n_calib + n_selected :]

    def subset(idx):
        return {m: features_np[m][idx] for m in MODEL_NAMES}

    return subset(calib_idx), subset(selected_idx), subset(candidate_idx)


# ---------------------------------------------------------------------------
# Noise calibration
# ---------------------------------------------------------------------------
def calibrate_noise(calib_features, metric, corr_type, device):
    """Calibrate both RDM-space and feature-space noise for all NC targets."""
    var_rdm = {}  # nc_target -> {model: variance}
    sigma_feat = {}  # nc_target -> {model: sigma}

    for nc in NC_TARGETS:
        print(f"\n--- Calibrating noise for NC target = {nc} ---")

        # RDM-space (numeric bisection)
        print("  RDM-space calibration...")
        var_rdm[nc] = rdm_noise_by_model(
            calib_features, MODEL_NAMES, torch.device(device),
            metric=metric, target_nc=nc, mode="numeric",
            corr_type=corr_type, seed=42,
        )

        # Feature-space (bisection on features)
        print("  Feature-space calibration...")
        sigma_feat[nc] = {}
        for m in tqdm(MODEL_NAMES, desc="  Feature-space"):
            feats = torch.from_numpy(calib_features[m][:1000].astype(np.float32)).to(device)
            sigma_feat[nc][m] = calibrate_feature_noise(
                features=feats,
                target_self_correlation=nc,
                rdm_metric=metric,
                n_samples=20,
                max_iterations=20,
                tolerance=0.01,
                device=torch.device(device),
            )
            if device == "cuda":
                torch.cuda.empty_cache()

    return var_rdm, sigma_feat


# ---------------------------------------------------------------------------
# Test 1: Noise Ceiling Equivalence
# ---------------------------------------------------------------------------
def test_noise_ceiling(selected_np, var_rdm, sigma_feat, metric, corr_type, n_mc, device):
    """Verify both approaches achieve the target NC on held-out data."""
    print("\n=== Test 1: Noise Ceiling Equivalence ===")
    rows = []

    for nc in NC_TARGETS:
        for m in MODEL_NAMES:
            feats = torch.from_numpy(selected_np[m].astype(np.float32)).to(device)
            clean_rdm = get_rdm_vector(feats, metric=metric)

            # RDM-space: add noise to RDM
            noise_std_rdm = float(np.sqrt(var_rdm[nc][m]))
            corrs_rdm = []
            for _ in range(n_mc):
                noise = torch.randn_like(clean_rdm) * noise_std_rdm
                noisy_rdm = clean_rdm + noise
                # Pearson correlation
                c = clean_rdm - clean_rdm.mean()
                n = noisy_rdm - noisy_rdm.mean()
                r = (c * n).sum() / (c.norm() * n.norm() + 1e-10)
                corrs_rdm.append(r.item())

            # Feature-space: add noise to features, recompute RDM
            sig = sigma_feat[nc][m]
            corrs_feat = []
            for _ in range(n_mc):
                noisy_feats = feats + torch.randn_like(feats) * sig
                noisy_rdm = get_rdm_vector(noisy_feats, metric=metric)
                c = clean_rdm - clean_rdm.mean()
                n = noisy_rdm - noisy_rdm.mean()
                r = (c * n).sum() / (c.norm() * n.norm() + 1e-10)
                corrs_feat.append(r.item())

            nc_rdm = float(np.mean(corrs_rdm))
            nc_feat = float(np.mean(corrs_feat))
            rows.append({
                "nc_target": nc,
                "model": m,
                "achieved_nc_rdm": nc_rdm,
                "achieved_nc_feat": nc_feat,
                "abs_error_rdm": abs(nc_rdm - nc),
                "abs_error_feat": abs(nc_feat - nc),
                "rdm_feat_diff": abs(nc_rdm - nc_feat),
            })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Test 2: Correlation Matrix Agreement
# ---------------------------------------------------------------------------
def test_correlation_matrices(selected_np, var_rdm, sigma_feat, metric, corr_type, n_mc, device):
    """Compare M x M correlation matrices under both noise types."""
    print("\n=== Test 2: Correlation Matrix Agreement ===")

    M = len(MODEL_NAMES)
    results_rows = []
    matrices = {}  # nc -> (rdm_matrix, feat_matrix)

    for nc in NC_TARGETS:
        # Compute clean RDMs
        feats_by_model = {
            m: torch.from_numpy(selected_np[m].astype(np.float32)).to(device)
            for m in MODEL_NAMES
        }
        clean_rdms = torch.stack([
            get_rdm_vector(feats_by_model[m], metric=metric) for m in MODEL_NAMES
        ])  # [M, P]

        # RDM-space: add noise to RDM vectors
        noise_stds_rdm = torch.tensor(
            [np.sqrt(var_rdm[nc][m]) for m in MODEL_NAMES],
            device=device, dtype=torch.float32,
        ).unsqueeze(1)  # [M, 1]

        corr_sims_rdm = []
        for _ in range(n_mc):
            noise = torch.randn_like(clean_rdms) * noise_stds_rdm
            noisy = clean_rdms + noise
            c = compute_correlation_matrix(
                noisy.unsqueeze(0), clean_rdms.unsqueeze(0), corr_type
            )[0]  # [M, M]
            corr_sims_rdm.append(c)
        corr_sims_rdm = torch.stack(corr_sims_rdm)  # [n_mc, M, M]
        mean_rdm = corr_sims_rdm.mean(dim=0)

        # Feature-space: add noise to features, recompute RDMs
        corr_sims_feat = []
        for _ in tqdm(range(n_mc), desc=f"  Feat-space corr (NC={nc})", leave=False):
            noisy_rdms = []
            for m in MODEL_NAMES:
                noisy_feats = feats_by_model[m] + torch.randn_like(feats_by_model[m]) * sigma_feat[nc][m]
                noisy_rdms.append(get_rdm_vector(noisy_feats, metric=metric))
            noisy_stack = torch.stack(noisy_rdms)
            c = compute_correlation_matrix(
                noisy_stack.unsqueeze(0), clean_rdms.unsqueeze(0), corr_type
            )[0]
            corr_sims_feat.append(c)
        corr_sims_feat = torch.stack(corr_sims_feat)
        mean_feat = corr_sims_feat.mean(dim=0)

        # Compare
        diff = (mean_rdm - mean_feat).cpu().numpy()
        mask = np.triu_indices(M, k=1)
        upper_rdm = mean_rdm.cpu().numpy()[mask]
        upper_feat = mean_feat.cpu().numpy()[mask]

        frob = float(np.linalg.norm(diff, "fro"))
        max_err = float(np.max(np.abs(diff)))
        upper_corr = float(np.corrcoef(upper_rdm, upper_feat)[0, 1])

        results_rows.append({
            "nc_target": nc,
            "frobenius_error": frob,
            "max_element_error": max_err,
            "upper_tri_pearson_r": upper_corr,
        })

        matrices[nc] = (
            mean_rdm.cpu().numpy(),
            mean_feat.cpu().numpy(),
            corr_sims_rdm,
            corr_sims_feat,
        )

        print(f"  NC={nc}: Frob={frob:.4f}, Max={max_err:.4f}, r={upper_corr:.4f}")

    df = pd.DataFrame(results_rows)
    return df, matrices


# ---------------------------------------------------------------------------
# Test 3: Model Discriminability
# ---------------------------------------------------------------------------
def test_discriminability(matrices, device):
    """Compare model discriminability under both noise types."""
    print("\n=== Test 3: Model Discriminability ===")
    rows = []

    for nc in NC_TARGETS:
        _, _, corr_sims_rdm, corr_sims_feat = matrices[nc]

        disc_rdm = model_discriminability(corr_sims_rdm.to(device))
        disc_feat = model_discriminability(corr_sims_feat.to(device))

        err_rdm = float(disc_rdm["non_parametric_multiclass_error_prob"])
        err_feat = float(disc_feat["non_parametric_multiclass_error_prob"])
        avg_pw_rdm = float(disc_rdm["average_parametric_pairwise_error_probability"])
        avg_pw_feat = float(disc_feat["average_parametric_pairwise_error_probability"])

        rows.append({
            "nc_target": nc,
            "error_prob_rdm": err_rdm,
            "error_prob_feat": err_feat,
            "error_prob_diff": abs(err_rdm - err_feat),
            "avg_pairwise_err_rdm": avg_pw_rdm,
            "avg_pairwise_err_feat": avg_pw_feat,
            "avg_pairwise_err_diff": abs(avg_pw_rdm - avg_pw_feat),
        })
        print(f"  NC={nc}: err_rdm={err_rdm:.4f}, err_feat={err_feat:.4f}, diff={abs(err_rdm-err_feat):.4f}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 4: Candidate Ranking Preservation
# ---------------------------------------------------------------------------
def test_candidate_ranking(selected_np, candidate_np, var_rdm, sigma_feat,
                           metric, corr_type, n_mc, device):
    """Compare candidate rankings under both noise types."""
    print("\n=== Test 4: Candidate Ranking Preservation ===")

    M = len(MODEL_NAMES)
    n_candidates = candidate_np[MODEL_NAMES[0]].shape[0]
    rows = []

    for nc in NC_TARGETS:
        selected_feats = {
            m: torch.from_numpy(selected_np[m].astype(np.float32)).to(device)
            for m in MODEL_NAMES
        }
        cand_feats = {
            m: torch.from_numpy(candidate_np[m].astype(np.float32)).to(device)
            for m in MODEL_NAMES
        }

        # Precompute selected RDMs
        selected_rdms = {
            m: get_rdm_vector(selected_feats[m], metric=metric)
            for m in MODEL_NAMES
        }

        noise_vars_tensor = torch.tensor(
            [var_rdm[nc][m] for m in MODEL_NAMES], device=device, dtype=torch.float32
        )

        utilities_rdm = []
        utilities_feat = []

        for c in tqdm(range(n_candidates), desc=f"  Ranking (NC={nc})"):
            # Build augmented features
            aug_feats = {
                m: torch.cat([selected_feats[m], cand_feats[m][c:c+1]])
                for m in MODEL_NAMES
            }

            # Augmented RDMs
            aug_rdms = torch.stack([
                get_rdm_vector(aug_feats[m], metric=metric)
                for m in MODEL_NAMES
            ]).unsqueeze(0)  # [1, M, P']

            # RDM-space: analytical utility
            u_rdm = compute_analytical_utility(
                aug_rdms, noise_vars_tensor,
                aggregation_within="mean", aggregation_across="min",
            ).item()
            utilities_rdm.append(u_rdm)

            # Feature-space: MC utility
            clean_rdms_aug = aug_rdms[0]  # [M, P']
            mc_utils = []
            for _ in range(n_mc):
                noisy_rdms = []
                for mi, m in enumerate(MODEL_NAMES):
                    noisy = aug_feats[m] + torch.randn_like(aug_feats[m]) * sigma_feat[nc][m]
                    noisy_rdms.append(get_rdm_vector(noisy, metric=metric))
                noisy_stack = torch.stack(noisy_rdms).unsqueeze(0)  # [1, M, P']
                corr = compute_correlation_matrix(
                    noisy_stack, clean_rdms_aug.unsqueeze(0), corr_type
                )  # [1, M, M]
                u_per_model = compute_model_utilities(corr, "mean")  # [1, M]
                u = aggregate_across_models(u_per_model, "min").item()  # scalar
                mc_utils.append(u)
            utilities_feat.append(float(np.mean(mc_utils)))

        u_rdm_arr = np.array(utilities_rdm)
        u_feat_arr = np.array(utilities_feat)

        rho, rho_p = stats.spearmanr(u_rdm_arr, u_feat_arr)
        tau, tau_p = stats.kendalltau(u_rdm_arr, u_feat_arr)

        # Top-K overlap
        rank_rdm = np.argsort(-u_rdm_arr)
        rank_feat = np.argsort(-u_feat_arr)
        top10_overlap = len(set(rank_rdm[:10]) & set(rank_feat[:10])) / 10
        top20_overlap = len(set(rank_rdm[:20]) & set(rank_feat[:20])) / 20

        rows.append({
            "nc_target": nc,
            "spearman_rho": rho,
            "spearman_p": rho_p,
            "kendall_tau": tau,
            "kendall_p": tau_p,
            "top10_overlap": top10_overlap,
            "top20_overlap": top20_overlap,
        })
        print(f"  NC={nc}: rho={rho:.4f}, tau={tau:.4f}, top10={top10_overlap:.1%}, top20={top20_overlap:.1%}")

        # Save per-candidate utilities for scatter plots
        cand_df = pd.DataFrame({
            "candidate_idx": range(n_candidates),
            "utility_rdm": u_rdm_arr,
            "utility_feat": u_feat_arr,
            "nc_target": nc,
        })
        cand_df.to_csv(DATA_DIR / f"candidate_utilities_nc{nc}.csv", index=False)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def create_summary_figure(nc_df, corr_df, disc_df, rank_df, matrices):
    """Create 4-panel summary figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: Noise ceiling scatter
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(NC_TARGETS)))
    for i, nc in enumerate(NC_TARGETS):
        sub = nc_df[nc_df["nc_target"] == nc]
        ax.scatter(sub["achieved_nc_rdm"], sub["achieved_nc_feat"],
                   c=[colors[i]], label=f"NC={nc}", s=60, zorder=3)
    lims = [nc_df[["achieved_nc_rdm", "achieved_nc_feat"]].min().min() - 0.05,
            nc_df[["achieved_nc_rdm", "achieved_nc_feat"]].max().max() + 0.05]
    ax.plot(lims, lims, "k--", alpha=0.5, lw=1)
    ax.set_xlabel("Achieved NC (RDM-space noise)")
    ax.set_ylabel("Achieved NC (Feature-space noise)")
    ax.set_title("A) Noise Ceiling Equivalence")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")

    # Panel B: Correlation matrix heatmaps at NC=0.46
    ax = axes[0, 1]
    nc_key = 0.46
    if nc_key in matrices:
        mat_rdm, mat_feat, _, _ = matrices[nc_key]
        diff = mat_rdm - mat_feat
        M = len(MODEL_NAMES)
        short_names = [m.split("_")[1] if "_" in m else m for m in MODEL_NAMES]

        im = ax.imshow(diff, cmap="RdBu_r", vmin=-0.05, vmax=0.05, aspect="auto")
        ax.set_xticks(range(M))
        ax.set_xticklabels(short_names, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(M))
        ax.set_yticklabels(short_names, fontsize=8)
        for i in range(M):
            for j in range(M):
                ax.text(j, i, f"{diff[i,j]:.3f}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"B) Correlation Matrix Difference (NC={nc_key})\n(RDM-space minus Feature-space)")

    # Panel C: Discriminability comparison
    ax = axes[1, 0]
    x = np.arange(len(NC_TARGETS))
    w = 0.35
    ax.bar(x - w/2, disc_df["error_prob_rdm"], w, label="RDM-space noise", color="#4C72B0")
    ax.bar(x + w/2, disc_df["error_prob_feat"], w, label="Feature-space noise", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([str(nc) for nc in NC_TARGETS])
    ax.set_xlabel("Noise Ceiling Target")
    ax.set_ylabel("Multiclass Error Probability")
    ax.set_title("C) Model Discriminability")
    ax.legend(fontsize=9)

    # Panel D: Ranking preservation
    ax = axes[1, 1]
    ax.plot(NC_TARGETS, rank_df["spearman_rho"], "o-", color="#4C72B0", label="Spearman rho", lw=2)
    ax.plot(NC_TARGETS, rank_df["kendall_tau"], "s--", color="#DD8452", label="Kendall tau", lw=2)
    ax.plot(NC_TARGETS, rank_df["top10_overlap"], "^:", color="#55A868", label="Top-10 overlap", lw=2)
    ax.set_xlabel("Noise Ceiling Target")
    ax.set_ylabel("Agreement")
    ax.set_title("D) Candidate Ranking Preservation")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "noise_validation_summary.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "noise_validation_summary.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved to {FIGURES_DIR / 'noise_validation_summary.png'}")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(nc_df, corr_df, disc_df, rank_df, elapsed_sec):
    """Generate markdown report with results."""
    report = f"""# Validation: RDM-Space vs Feature-Space Noise Injection

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Runtime**: {elapsed_sec/60:.1f} minutes

## Summary

This validation compares two approaches for simulating measurement noise in RSA-based
stimulus selection:

1. **RDM-space noise** (the approximation used in the selection pipeline): Add Gaussian
   noise directly to RDM vectors and use an analytical attenuation formula.
2. **Feature-space noise** (Monte Carlo reference): Add Gaussian noise to feature vectors,
   recompute RDMs, then measure correlations.

Both approaches are calibrated to achieve the same target noise ceiling (the correlation
between clean and noisy RDMs). The question is whether they produce equivalent downstream
results for model discriminability and candidate ranking.

## Models

{', '.join(MODEL_NAMES)}

## Test 1: Noise Ceiling Equivalence

Both calibration procedures should achieve the target noise ceiling on held-out data.

| NC Target | Mean Achieved (RDM) | Mean Achieved (Feat) | Mean Abs Diff |
|-----------|--------------------:|---------------------:|--------------:|
"""
    for nc in NC_TARGETS:
        sub = nc_df[nc_df["nc_target"] == nc]
        report += f"| {nc} | {sub['achieved_nc_rdm'].mean():.4f} | {sub['achieved_nc_feat'].mean():.4f} | {sub['rdm_feat_diff'].mean():.4f} |\n"

    report += """
**Interpretation**: Both approaches should achieve the target noise ceiling within
calibration tolerance (~0.02). Differences here indicate that the noise propagation
is well-matched between the two approaches.

## Test 2: Correlation Matrix Agreement

The M x M model correlation matrices under both noise types should be similar.

| NC Target | Frobenius Error | Max Element Error | Upper-Tri Pearson r |
|-----------|----------------:|------------------:|--------------------:|
"""
    for _, row in corr_df.iterrows():
        report += f"| {row['nc_target']} | {row['frobenius_error']:.4f} | {row['max_element_error']:.4f} | {row['upper_tri_pearson_r']:.4f} |\n"

    report += """
![Correlation Matrix Difference](figures/noise_validation_summary.png)

**Interpretation**: Small Frobenius errors and high upper-triangle correlations indicate
that the relative structure of model-to-model similarity is preserved under both noise types.

## Test 3: Model Discriminability

The multiclass error probability should be similar under both noise types.

| NC Target | Error Prob (RDM) | Error Prob (Feat) | Abs Diff |
|-----------|------------------:|------------------:|---------:|
"""
    for _, row in disc_df.iterrows():
        report += f"| {row['nc_target']} | {row['error_prob_rdm']:.4f} | {row['error_prob_feat']:.4f} | {row['error_prob_diff']:.4f} |\n"

    report += """
**Interpretation**: If the error probabilities match within ~0.02, the approximation
preserves the key downstream metric used to evaluate stimulus selection quality.

## Test 4: Candidate Ranking Preservation

The relative ordering of candidates by utility score should be similar.

| NC Target | Spearman rho | Kendall tau | Top-10 Overlap | Top-20 Overlap |
|-----------|-------------:|------------:|---------------:|---------------:|
"""
    for _, row in rank_df.iterrows():
        report += f"| {row['nc_target']} | {row['spearman_rho']:.4f} | {row['kendall_tau']:.4f} | {row['top10_overlap']:.1%} | {row['top20_overlap']:.1%} |\n"

    report += """
**Interpretation**: High Spearman correlations (> 0.9) indicate that the selection
algorithm would make similar choices under both noise models. Top-K overlap measures
whether the same candidates would be selected as the "best" options.

## Conclusion

"""

    # Auto-generate conclusion based on results
    mean_rho = rank_df["spearman_rho"].mean()
    mean_disc_diff = disc_df["error_prob_diff"].mean()
    mean_nc_diff = nc_df["rdm_feat_diff"].mean()

    if mean_rho > 0.9 and mean_disc_diff < 0.03 and mean_nc_diff < 0.03:
        report += """The RDM-space noise approximation produces results that are highly
consistent with the feature-space Monte Carlo reference across all four validation tests.
The analytical attenuation formula provides a valid and computationally efficient
alternative to Monte Carlo feature-space noise injection for the purpose of
controversial stimulus selection."""
    elif mean_rho > 0.7:
        report += """The RDM-space noise approximation shows moderate agreement with the
feature-space Monte Carlo reference. While the overall ranking structure is preserved, there
are non-trivial differences in the exact utility values. The approximation remains
useful for efficient selection but should be interpreted with appropriate caution."""
    else:
        report += """The RDM-space noise approximation shows substantial divergence from
the feature-space Monte Carlo reference. The results suggest that the linear noise model
on RDM vectors does not adequately capture the nonlinear effect of feature-level
noise on pairwise distances. Further investigation is recommended."""

    report_path = SECTION_DIR / "REPORT.md"
    report_path.write_text(report)
    print(f"Report saved to {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    t0 = time.time()

    # Load data
    calib_np, selected_np, candidate_np = load_features(
        n_calib=1000, n_selected=args.n_selected,
        n_candidates=args.n_candidates, seed=args.seed,
    )

    # Calibrate noise
    var_rdm, sigma_feat = calibrate_noise(
        calib_np, args.metric, args.corr_type, args.device,
    )

    # Test 1
    nc_df = test_noise_ceiling(
        selected_np, var_rdm, sigma_feat,
        args.metric, args.corr_type, args.n_mc, args.device,
    )
    nc_df.to_csv(DATA_DIR / "noise_ceiling_results.csv", index=False)

    # Test 2
    corr_df, matrices = test_correlation_matrices(
        selected_np, var_rdm, sigma_feat,
        args.metric, args.corr_type, args.n_mc, args.device,
    )
    corr_df.to_csv(DATA_DIR / "correlation_matrix_results.csv", index=False)

    # Test 3
    disc_df = test_discriminability(matrices, args.device)
    disc_df.to_csv(DATA_DIR / "discriminability_results.csv", index=False)

    # Test 4
    rank_df = test_candidate_ranking(
        selected_np, candidate_np, var_rdm, sigma_feat,
        args.metric, args.corr_type, args.n_mc_ranking, args.device,
    )
    rank_df.to_csv(DATA_DIR / "ranking_results.csv", index=False)

    elapsed = time.time() - t0

    # Generate outputs
    create_summary_figure(nc_df, corr_df, disc_df, rank_df, matrices)
    generate_report(nc_df, corr_df, disc_df, rank_df, elapsed)

    print(f"\nDone in {elapsed/60:.1f} minutes.")


if __name__ == "__main__":
    main()
