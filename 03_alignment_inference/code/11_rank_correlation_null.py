"""Permutation null for cross-stimulus-set model-rank correlations.

Motivation: Fig. rank_shift reports rho_b<->c (cross-set rank correlation between baseline
and controversial orderings) at small M (5-6 models per set). At small M, |rho| has
substantial mass near 0.4 by chance, so the qualitative claim "ranks are largely preserved"
needs to be read against a chance distribution. This script:

1. Reads the best-on-independent-shared transfer scores used by the main
   brain-alignment figure.
2. For each (model_set, method) computes the observed mean-across-subjects rho_b<->c
3. Applies synchronized random model-label permutations across participants
   to preserve cross-participant dependence within each condition.
4. Outputs rank_null.csv  with observed rho, null distribution percentiles, and p-value
5. Plots rank_null.pdf / rank_null.png with one panel per model set, two metrics overlaid

Outputs:
  03_alignment_inference/results/rank_null.csv
  03_alignment_inference/figures/supplementary/rank_null.{pdf,png}
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata, spearmanr

STAGE = Path(__file__).resolve().parents[1]
SHARE_ROOT = STAGE.parent
RANK_FIGURE_CODE = (
    SHARE_ROOT
    / "01_brain_model_alignment"
    / "code"
    / "rsa_scoring"
    / "figures"
)
sys.path.insert(0, str(RANK_FIGURE_CODE))
from plot_rank_shift import (  # noqa: E402
    MODEL_SETS as RANK_MODEL_SETS,
    SUBJECTS,
    load_scores,
)

OUT_DATA = STAGE / "results"
OUT_FIG = STAGE / "figures" / "supplementary"
OUT_PNG = OUT_FIG / "png"
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_PNG.mkdir(parents=True, exist_ok=True)

N_PERMUTATIONS = 100_000

MODEL_SETS = ["all_models", "sota", "training_objective", "architecture", "dataset"]
SET_LABEL = {
    "all_models": "All models",
    "sota": "SOTA",
    "training_objective": "Training Objective",
    "architecture": "Architecture",
    "dataset": "Dataset",
}
METHODS = [("crsa", "Fixed RSA"), ("wrsa_transfer", "Mixed RSA")]


def _load_scores(method):
    return load_scores(method)


def _per_subject_rho(scores_long, method, model_set):
    """For one (method, model_set), compute per-subject cross-set rho between
    baseline (vicco) and controversial orderings."""
    score_col = "crsa" if method == "crsa" else "wrsa_transfer"
    sub_df = scores_long[
        (scores_long["model_set"] == model_set)
    ]
    subject_pairs = []
    for sub in SUBJECTS:
        a = sub_df[(sub_df["subject"] == sub) & (sub_df["stimulus_type"] == "vicco")]
        b = sub_df[(sub_df["subject"] == sub) & (sub_df["stimulus_type"] == "controversial")]
        if a.empty or b.empty:
            continue
        a_mean = a.groupby("model")[score_col].mean()
        b_mean = b.groupby("model")[score_col].mean()
        subject_pairs.append((a_mean, b_mean))
    if not subject_pairs:
        return np.array([]), None, np.empty((0, 0)), np.empty((0, 0))

    # A synchronized model-label permutation needs the same ordered model
    # columns for every subject.  Preserve only the complete intersection.
    common = set(subject_pairs[0][0].index) & set(subject_pairs[0][1].index)
    for a_mean, b_mean in subject_pairs[1:]:
        common &= set(a_mean.index) & set(b_mean.index)
    common = sorted(common)
    if len(common) < 3:
        return np.array([]), None, np.empty((0, 0)), np.empty((0, 0))
    expected = set(RANK_MODEL_SETS[model_set])
    if set(common) != expected:
        missing = sorted(expected - set(common))
        extra = sorted(set(common) - expected)
        raise ValueError(
            f"{method}/{model_set}: complete-case roster mismatch; "
            f"missing={missing}, extra={extra}"
        )

    base_values = np.stack([a.loc[common].to_numpy(dtype=float) for a, _ in subject_pairs])
    cstim_values = np.stack([b.loc[common].to_numpy(dtype=float) for _, b in subject_pairs])
    rhos = np.asarray([
        spearmanr(base_values[i], cstim_values[i])[0]
        for i in range(len(subject_pairs))
    ])
    return rhos, len(common), base_values, cstim_values


def _null_distribution(base_values, cstim_values, rng, n_perm=N_PERMUTATIONS):
    """Dependency-preserving null for the mean subject-level Spearman rho.

    Each draw applies one synchronized permutation of controversial-condition
    model labels to every subject.  This preserves cross-subject model
    structure within each condition while breaking model identity across
    conditions.
    """
    n_models = base_values.shape[1]
    base_rank = np.apply_along_axis(rankdata, 1, base_values)
    cstim_rank = np.apply_along_axis(rankdata, 1, cstim_values)
    base_rank -= base_rank.mean(axis=1, keepdims=True)
    cstim_rank -= cstim_rank.mean(axis=1, keepdims=True)
    base_rank /= np.linalg.norm(base_rank, axis=1, keepdims=True)
    cstim_rank /= np.linalg.norm(cstim_rank, axis=1, keepdims=True)
    rhos = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n_models)
        subject_rhos = np.sum(base_rank * cstim_rank[:, perm], axis=1)
        rhos[i] = float(np.mean(subject_rhos))
    return rhos


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["crsa", "wrsa_transfer"],
        default=["crsa", "wrsa_transfer"],
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run and report the analysis without writing result or figure files.",
    )
    args = parser.parse_args()
    if args.n_permutations < 1:
        parser.error("--n-permutations must be positive")
    return args


def main():
    args = parse_args()
    start = time.perf_counter()
    methods = [item for item in METHODS if item[0] in args.methods]
    rows = []
    fig, axes = plt.subplots(1, len(MODEL_SETS), figsize=(3.0 * len(MODEL_SETS), 2.6),
                             sharey=True)
    for ax, ms in zip(axes, MODEL_SETS):
        for method, method_label in methods:
            scores = _load_scores(method)
            if scores.empty:
                continue
            rhos, n_models, base_values, cstim_values = _per_subject_rho(scores, method, ms)
            if n_models is None or len(rhos) == 0:
                continue
            obs_mean = float(np.nanmean(rhos))
            obs_sem = float(np.nanstd(rhos, ddof=1) / max(np.sqrt(len(rhos)), 1))

            # Reinitialize a deterministic stream per model set.  Both RSA
            # metrics therefore use the same model-label permutations, and a
            # metric's result is invariant to CLI method inclusion or order.
            cell_rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, MODEL_SETS.index(ms)])
            )
            null_rhos = _null_distribution(
                base_values,
                cstim_values,
                cell_rng,
                n_perm=args.n_permutations,
            )
            # Include algebraically equal values in this discrete tail; the
            # tolerance avoids floating-point splits at the observed boundary.
            p_two = float((1 + np.sum(np.abs(null_rhos) >= abs(obs_mean) - 1e-12))
                          / (len(null_rhos) + 1))
            null_p95 = float(np.percentile(null_rhos, 97.5))
            null_p05 = float(np.percentile(null_rhos, 2.5))

            rows.append({
                "model_set": ms, "method": method,
                "n_models": n_models, "n_subjects_used": int(len(rhos)),
                "rho_obs_mean": obs_mean, "rho_obs_sem": obs_sem,
                "null_p2.5": null_p05, "null_p97.5": null_p95,
                "p_two_tailed": p_two,
            })

            color = "#555555" if method == "crsa" else "#0072B2"
            ax.hist(
                null_rhos,
                bins=40,
                density=True,
                histtype="step",
                color=color,
                linewidth=0.9,
                alpha=0.9,
                label=f"{method_label} null",
            )
            ax.axvline(
                obs_mean,
                color=color,
                linewidth=1.5,
                label=f"{method_label} observed",
                zorder=10,
            )
            ax.axvspan(
                obs_mean - obs_sem,
                obs_mean + obs_sem,
                color=color,
                alpha=0.10,
                linewidth=0,
                zorder=1,
            )

        ax.set_xlim(-1.05, 1.05)
        ax.set_xlabel(r"Spearman $\rho_{b\leftrightarrow c}$")
        ax.set_title(SET_LABEL.get(ms, ms), fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_yticks([])
    axes[0].set_ylabel("permutation density")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        fontsize=8,
        frameon=False,
        ncol=max(1, len(methods)),
    )

    # Panel titles plus the manuscript caption carry the figure-level
    # description. Reserve the top band for the legend so it cannot collide
    # with titles, and let the shared y-axis autoscale for discrete small-M
    # permutation distributions rather than clipping high-density bins.
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    if not args.validate_only:
        fig.savefig(OUT_FIG / "rank_null.pdf", bbox_inches="tight")
        fig.savefig(OUT_PNG / "rank_null.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame(rows)
    if not args.validate_only:
        df.to_csv(OUT_DATA / "rank_null.csv", index=False)
    print(df.to_string(index=False))
    elapsed = time.perf_counter() - start
    print(f"\nElapsed: {elapsed:.3f} s for {args.n_permutations:,} permutations per cell")
    if not args.validate_only:
        print(f"Saved {OUT_FIG / 'rank_null.pdf'}")
        print(f"Saved {OUT_DATA / 'rank_null.csv'}")


if __name__ == "__main__":
    main()
