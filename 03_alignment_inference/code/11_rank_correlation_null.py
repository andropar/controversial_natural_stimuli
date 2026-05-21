"""
11_rank_correlation_null.py

Permutation null for cross-stimulus-set Spearman rank correlations of model orderings.

Motivation: Fig. rank_shift reports rho_b<->c (cross-set rank correlation between baseline
and controversial orderings) at small M (5-6 models per set). At small M, |rho| has
substantial mass near 0.4 by chance, so the qualitative claim "ranks are largely preserved"
needs to be read against a chance distribution. This script:

1. Reads per-subject baseline + controversial scores from
   experiments/cstim_paper/02_rsa_scores/data/sub-XX/{crsa,wrsa_transfer}_scores.csv
2. For each (model_set, method) computes the observed mean-across-subjects rho_b<->c
3. Generates 10_000 random re-orderings of M models and computes the null Spearman rho
4. Outputs rank_null.csv  with observed rho, null distribution percentiles, and p-value
5. Plots rank_null.pdf / rank_null.png with one panel per model set, two metrics overlaid

Outputs:
  experiments/cstim_paper/03_statistics/data/rank_null.csv
  experiments/cstim_paper/03_statistics/figures/rank_null.{pdf,png}
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

STAGE = Path(__file__).resolve().parents[1]
SHARE_ROOT = STAGE.parent
RSA = SHARE_ROOT / "01_brain_model_alignment" / "rsa_scores"
OUT_DATA = STAGE / "data"
OUT_FIG = STAGE / "figures" / "supplementary"
OUT_PNG = OUT_FIG / "png"
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_PNG.mkdir(parents=True, exist_ok=True)

SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
N_PERMUTATIONS = 10_000
RNG = np.random.default_rng(42)

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
    frames = []
    for sub in SUBJECTS:
        f = RSA / sub / f"{method}_scores.csv"
        if not f.exists():
            print(f"  missing: {f}", file=sys.stderr)
            continue
        df = pd.read_csv(f)
        df["subject"] = sub
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _per_subject_rho(scores_long, method, model_set):
    """For one (method, model_set), compute per-subject cross-set rho between
    baseline (vicco) and controversial orderings."""
    score_col = "crsa" if method == "crsa" else "wrsa_transfer"
    sub_df = scores_long[
        (scores_long["model_set"] == model_set)
    ]
    rhos = []
    n_models = None
    for sub in SUBJECTS:
        a = sub_df[(sub_df["subject"] == sub) & (sub_df["stimulus_type"] == "vicco")]
        b = sub_df[(sub_df["subject"] == sub) & (sub_df["stimulus_type"] == "controversial")]
        if a.empty or b.empty:
            continue
        a_mean = a.groupby("model")[score_col].mean()
        b_mean = b.groupby("model")[score_col].mean()
        common = sorted(set(a_mean.index) & set(b_mean.index))
        if len(common) < 3:
            continue
        rho, _ = spearmanr(a_mean.loc[common], b_mean.loc[common])
        rhos.append(rho)
        n_models = len(common) if n_models is None else n_models
    return np.array(rhos), n_models


def _null_distribution(n_models, n_perm=N_PERMUTATIONS):
    """Distribution of Spearman rho between two random orderings of n_models items."""
    base = np.arange(n_models)
    rhos = np.empty(n_perm)
    for i in range(n_perm):
        perm = RNG.permutation(n_models)
        rho, _ = spearmanr(base, perm)
        rhos[i] = rho
    return rhos


def main():
    rows = []
    null_cache = {}

    fig, axes = plt.subplots(1, len(MODEL_SETS), figsize=(3.0 * len(MODEL_SETS), 2.6),
                             sharey=True)
    for ax, ms in zip(axes, MODEL_SETS):
        for method, method_label in METHODS:
            scores = _load_scores(method)
            if scores.empty:
                continue
            rhos, n_models = _per_subject_rho(scores, method, ms)
            if n_models is None or len(rhos) == 0:
                continue
            obs_mean = float(np.nanmean(rhos))
            obs_sem = float(np.nanstd(rhos, ddof=1) / max(np.sqrt(len(rhos)), 1))

            if n_models not in null_cache:
                null_cache[n_models] = _null_distribution(n_models)
            null_rhos = null_cache[n_models]
            p_two = float((np.abs(null_rhos) >= abs(obs_mean)).mean())
            null_p95 = float(np.percentile(null_rhos, 97.5))
            null_p05 = float(np.percentile(null_rhos, 2.5))

            rows.append({
                "model_set": ms, "method": method,
                "n_models": n_models, "n_subjects_used": int(len(rhos)),
                "rho_obs_mean": obs_mean, "rho_obs_sem": obs_sem,
                "null_p2.5": null_p05, "null_p97.5": null_p95,
                "p_two_tailed": p_two,
            })

            color = "#444" if method == "crsa" else "#c44"
            label = method_label
            if method == "crsa":
                ax.hist(null_rhos, bins=40, color="0.85", edgecolor="none", density=True)
                ax.axvline(null_p95, color="0.5", ls="--", lw=0.8)
                ax.axvline(null_p05, color="0.5", ls="--", lw=0.8)
            ax.scatter([obs_mean], [0.4 if method == "crsa" else 0.7],
                       color=color, s=40, zorder=10, label=label,
                       edgecolor="white", lw=0.5)
            ax.errorbar([obs_mean], [0.4 if method == "crsa" else 0.7],
                        xerr=[[obs_sem], [obs_sem]],
                        color=color, lw=1.0, capsize=2, zorder=9)

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel(r"Spearman $\rho_{b\leftrightarrow c}$")
        ax.set_title(SET_LABEL.get(ms, ms), fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_yticks([])
    axes[0].set_ylabel("density / observed")
    axes[-1].legend(loc="upper left", fontsize=8, frameon=False)

    fig.suptitle("Cross-set rank correlation against permutation null", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "rank_null.pdf", bbox_inches="tight")
    fig.savefig(OUT_PNG / "rank_null.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DATA / "rank_null.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nSaved {OUT_FIG / 'rank_null.pdf'}")
    print(f"Saved {OUT_DATA / 'rank_null.csv'}")


if __name__ == "__main__":
    main()
