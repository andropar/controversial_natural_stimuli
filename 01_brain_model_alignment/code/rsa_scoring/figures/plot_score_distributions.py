#!/usr/bin/env python3
"""
Score distribution figure: KDE histograms of brain alignment scores.

Layout: 2 rows (mRSA, fRSA) x 5 cols (model sets).
Each panel: overlapping KDEs for controversial (red) vs baseline (blue).
Annotated with SD ratio (controversial / baseline).
Y-axis hidden (KDE density is arbitrary).

Usage:
    python plot_score_distributions.py
"""

import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
_SHARE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))  # project root
sys.path.insert(0, str(_PAPER / "figures"))  # for shared figure style
sys.path.insert(0, str(_SHARE_ROOT / "shared" / "code" / "paper_helpers"))
sys.path.insert(0, str(_SHARE_ROOT / "shared" / "code" / "paper_helpers" / "figures"))
import config

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from style_improved import apply_style, FONT, DPI, W_DOUBLE

apply_style()

SUBJECTS = config.SUBJECTS
STAGE_DIR = Path(__file__).resolve().parents[3]
SHARE_ROOT = STAGE_DIR.parent
RSA_DATA_DIR = STAGE_DIR / "results" / "rsa_scores"
STATS_DATA_DIR = SHARE_ROOT / "03_alignment_inference" / "results"
FIGURES_DIR = STAGE_DIR / "figures" / "rsa_scores" / "supplementary"
PNG_DIR = FIGURES_DIR / "png"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SET_ORDER = ["sota", "architecture", "training_objective", "dataset", "all_models"]

MODEL_SET_LABELS = {
    "sota": "SOTA",
    "architecture": "Architecture",
    "training_objective": "Train. Obj.",
    "dataset": "Dataset",
    "all_models": "All Models",
}

METHOD_ORDER = ["wrsa_transfer", "crsa"]
METHOD_LABELS = {"wrsa_transfer": "mRSA", "crsa": "fRSA"}

CSTIM_COLOR = "#E74C3C"
BASELINE_COLOR = "#3498DB"

# Bonferroni threshold: 50 tests (5 sets x 2 methods x 5 metrics)
BONFERRONI_ALPHA = 0.05 / 50


def _sig_stars(p):
    """Return significance stars string from p-value."""
    if p <= 0.001:
        return "***"
    elif p <= 0.01:
        return "**"
    elif p <= 0.05:
        return "*"
    return ""


def main():
    # Load scores
    wrsa_parts, crsa_parts = [], []
    for subj in SUBJECTS:
        wrsa_path = RSA_DATA_DIR / subj / "wrsa_transfer_scores.csv"
        crsa_path = RSA_DATA_DIR / subj / "crsa_scores.csv"
        if wrsa_path.exists():
            wrsa_parts.append(pd.read_csv(wrsa_path))
        if crsa_path.exists():
            crsa_parts.append(pd.read_csv(crsa_path))

    wrsa = pd.concat(wrsa_parts, ignore_index=True)
    crsa = pd.concat(crsa_parts, ignore_index=True)

    score_dfs = {"wrsa_transfer": wrsa, "crsa": crsa}
    score_cols = {"wrsa_transfer": "wrsa_transfer", "crsa": "crsa"}

    # Load permutation test results for significance annotation
    perm_path = STATS_DATA_DIR / "permutation_test_results.csv"
    perm_df = pd.read_csv(perm_path) if perm_path.exists() else None

    n_rows = len(METHOD_ORDER)
    n_cols = len(MODEL_SET_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(W_DOUBLE, 5.0), squeeze=False)

    for row, method in enumerate(METHOD_ORDER):
        for col, model_set in enumerate(MODEL_SET_ORDER):
            ax = axes[row, col]
            df = score_dfs[method]
            score_col = score_cols[method]

            sub = df[(df["model_set"] == model_set) & (df["model"] != "vicreg_resnet50")]

            # Controversial: mean across subjects per model
            cstim = sub[sub["stimulus_type"] == "controversial"]
            cstim_scores = cstim.groupby("model")[score_col].mean().values

            # Baseline: mean over bootstraps and subjects per model
            vicco = sub[sub["stimulus_type"] == "vicco"]
            vicco_scores = vicco.groupby("model")[score_col].mean().values

            # Per-subject scores for thin background lines
            per_subject_cstim = {}
            per_subject_vicco = {}
            for subj in SUBJECTS:
                subj_df = sub[sub["subject"] == subj]
                cs = subj_df[subj_df["stimulus_type"] == "controversial"]
                if not cs.empty:
                    per_subject_cstim[subj] = cs.groupby("model")[score_col].mean().values
                vc = subj_df[subj_df["stimulus_type"] == "vicco"]
                if not vc.empty:
                    per_subject_vicco[subj] = vc.groupby("model")[score_col].mean().values

            # KDE range
            all_vals = np.concatenate([cstim_scores, vicco_scores])
            lo, hi = all_vals.min(), all_vals.max()
            margin = (hi - lo) * 0.2 if hi > lo else 0.05
            x_grid = np.linspace(lo - margin, hi + margin, 200)

            max_density = 0
            for scores, color, label, per_subj in [
                (vicco_scores, BASELINE_COLOR, "Baseline", per_subject_vicco),
                (cstim_scores, CSTIM_COLOR, "Controversial", per_subject_cstim),
            ]:
                if len(scores) < 2:
                    ax.plot(scores, np.zeros_like(scores), "|",
                            color=color, ms=10, mew=1.5, label=label)
                    continue

                # Thin per-subject KDE lines (behind the main fill)
                for subj_scores in per_subj.values():
                    if len(subj_scores) >= 2:
                        try:
                            kde_s = gaussian_kde(subj_scores, bw_method="silverman")
                            ax.plot(x_grid, kde_s(x_grid), color=color,
                                    linewidth=0.4, alpha=0.3, zorder=1)
                        except np.linalg.LinAlgError:
                            pass

                # Main KDE (subject-averaged scores)
                kde = gaussian_kde(scores, bw_method="silverman")
                density = kde(x_grid)
                max_density = max(max_density, density.max())
                ax.fill_between(x_grid, density, alpha=0.25, color=color,
                                label=label, zorder=2)
                ax.plot(x_grid, density, color=color, linewidth=1.2, zorder=3)

                # Rug plot at bottom
                ax.plot(scores, np.full_like(scores, -0.02 * max(density.max(), 1)),
                        "|", color=color, ms=6, mew=1.0, clip_on=False, zorder=4)

            ax.set_ylim(bottom=0)
            ax.set_yticks([])


            # Labels
            if row == 0:
                ax.set_title(MODEL_SET_LABELS[model_set], fontsize=FONT["title"], fontweight="bold")
            if col == 0:
                ax.set_ylabel(METHOD_LABELS[method], fontsize=FONT["axis_label"], fontweight="bold",
                              rotation=0, labelpad=25, va="center")
            if row == n_rows - 1:
                ax.set_xlabel("Score")

            # Legend only first panel
            if row == 0 and col == 0:
                ax.legend(loc="upper left", frameon=True, framealpha=0.9,
                          handlelength=1.0, borderpad=0.3)

    plt.tight_layout(w_pad=1.0, h_pad=1.2)

    out_pdf = FIGURES_DIR / "score_distributions.pdf"
    out_png = PNG_DIR / "score_distributions.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=DPI)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    plt.close()


if __name__ == "__main__":
    main()
