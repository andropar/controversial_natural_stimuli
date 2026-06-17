#!/usr/bin/env python3
"""
Score distribution figure (strip-plot variant).

Layout: 2 rows (mRSA, fRSA) x 5 cols (model sets).
Each panel: paired strip plot — one dot per model for controversial (red)
and baseline (blue), with thin lines connecting the same model across
conditions. Per-subject dots shown with lower opacity.

Annotated with mean pairwise absolute difference ratio from the permutation
test CSV (cross-subject mean of per-subject ratios).

Usage:
    python plot_score_distributions_strip.py
"""

import sys
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))  # project root
from cstims.paper import config

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cstims.paper.style_improved import apply_style, FONT, DPI, W_DOUBLE

apply_style()

SUBJECTS = config.SUBJECTS
RSA_DATA_DIR = config.RSA_DATA_DIR
STATS_DATA_DIR = config.STATS_DATA_DIR
FIGURES_DIR = Path(__file__).resolve().parent
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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

COLOR_CSTIM = "#D64541"
COLOR_BASE = "#2980B9"

JITTER = 0.12  # horizontal jitter for strip dots


def _sig_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def load_all_scores():
    """Load fRSA and mRSA scores for all subjects."""
    frames = {"wrsa_transfer": [], "crsa": []}
    for subject in SUBJECTS:
        for method, filename in [("wrsa_transfer", "wrsa_transfer_scores.csv"),
                                 ("crsa", "crsa_scores.csv")]:
            path = RSA_DATA_DIR / subject / filename
            if path.exists():
                df = pd.read_csv(path)
                df["subject"] = subject
                frames[method].append(df)
    return {m: pd.concat(fs) for m, fs in frames.items() if fs}


def get_model_scores(df, model_set, score_col):
    """Get per-model scores: subject-averaged controversial and baseline.

    Returns:
        cstim_scores: dict model -> subject-averaged controversial score
        vicco_scores: dict model -> subject-averaged baseline score (mean across bootstraps, then subjects)
        per_subject_cstim: dict model -> dict subject -> score
        per_subject_vicco: dict model -> dict subject -> score
    """
    ms_df = df[df["model_set"] == model_set]

    # Controversial
    cstim = ms_df[ms_df["stimulus_type"] == "controversial"]
    cstim_subj_avg = cstim.groupby("model")[score_col].mean()

    # Vicco: average across bootstraps, then subjects
    vicco = ms_df[ms_df["stimulus_type"] == "vicco"]
    vicco_per_subj = vicco.groupby(["subject", "model"])[score_col].mean().reset_index()
    vicco_subj_avg = vicco_per_subj.groupby("model")[score_col].mean()

    # Per-subject scores
    per_subject_cstim = {}
    per_subject_vicco = {}
    for model in cstim_subj_avg.index:
        per_subject_cstim[model] = {}
        per_subject_vicco[model] = {}
        for subject in SUBJECTS:
            c = cstim[(cstim["model"] == model) & (cstim["subject"] == subject)]
            if not c.empty:
                per_subject_cstim[model][subject] = c[score_col].values[0]
            v = vicco_per_subj[(vicco_per_subj["model"] == model) &
                               (vicco_per_subj["subject"] == subject)]
            if not v.empty:
                per_subject_vicco[model][subject] = v[score_col].values[0]

    return cstim_subj_avg, vicco_subj_avg, per_subject_cstim, per_subject_vicco


def main():
    all_scores = load_all_scores()

    # Load permutation test results for annotations
    perm_path = STATS_DATA_DIR / "permutation_test_results.csv"
    perm_df = pd.read_csv(perm_path) if perm_path.exists() else None

    n_cols = len(MODEL_SET_ORDER)
    n_rows = len(METHOD_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(W_DOUBLE, 6.0),
                             sharex=False, sharey=False)

    for row, method in enumerate(METHOD_ORDER):
        score_col = method
        df = all_scores[method]

        for col, model_set in enumerate(MODEL_SET_ORDER):
            ax = axes[row, col]

            cstim_avg, vicco_avg, ps_cstim, ps_vicco = get_model_scores(
                df, model_set, score_col)

            models = sorted(cstim_avg.index)
            n = len(models)
            if n == 0:
                ax.set_visible(False)
                continue

            # x positions: 0 = baseline, 1 = controversial
            x_base = 0
            x_cstim = 1

            # Per-subject dots (low opacity)
            for model in models:
                for subject in SUBJECTS:
                    if subject in ps_vicco.get(model, {}) and subject in ps_cstim.get(model, {}):
                        yb = ps_vicco[model][subject]
                        yc = ps_cstim[model][subject]
                        jit = np.random.uniform(-JITTER, JITTER)
                        ax.plot([x_base + jit, x_cstim + jit], [yb, yc],
                                color="#999999", linewidth=0.3, alpha=0.25, zorder=1)
                        ax.scatter(x_base + jit, yb, s=6, color=COLOR_BASE,
                                   alpha=0.2, edgecolors="none", zorder=2)
                        ax.scatter(x_cstim + jit, yc, s=6, color=COLOR_CSTIM,
                                   alpha=0.2, edgecolors="none", zorder=2)

            # Subject-averaged dots (bold) with connecting lines
            y_base = np.array([vicco_avg[m] for m in models])
            y_cstim = np.array([cstim_avg[m] for m in models])

            for i in range(n):
                ax.plot([x_base, x_cstim], [y_base[i], y_cstim[i]],
                        color="#555555", linewidth=0.6, alpha=0.5, zorder=3)

            ax.scatter(np.full(n, x_base), y_base, s=18, color=COLOR_BASE,
                       edgecolors="white", linewidths=0.3, zorder=4)
            ax.scatter(np.full(n, x_cstim), y_cstim, s=18, color=COLOR_CSTIM,
                       edgecolors="white", linewidths=0.3, zorder=4)

            # Annotation: mean pairwise diff ratio from permutation test
            if perm_df is not None:
                match = perm_df[
                    (perm_df["model_set"] == model_set) &
                    (perm_df["method"] == method) &
                    (perm_df["metric"] == "mean_pairwise_diff")
                ]
                if not match.empty:
                    ratio = match.iloc[0]["observed_ratio"]
                    stars = _sig_stars(match.iloc[0]["p_perm"])
                    ax.text(0.5, 0.97, f"{ratio:.1f}×{stars}",
                            transform=ax.transAxes, fontsize=FONT["annotation"],
                            ha="center", va="top", color="#555555",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                      ec="none", alpha=0.8))

            # Axes
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Base", "Cstim"], fontsize=FONT["tick"])
            ax.set_xlim(-0.5, 1.5)

            if row == 0:
                ax.set_title(MODEL_SET_LABELS[model_set], fontsize=FONT["title"],
                             fontweight="bold")
            if col == 0:
                ax.set_ylabel(METHOD_LABELS[method], fontsize=FONT["axis_label"],
                              fontweight="bold")

    plt.tight_layout(h_pad=1.0, w_pad=0.5)

    for ext in ("pdf", "png"):
        out = FIGURES_DIR / f"score_distributions_strip.{ext}"
        fig.savefig(out, dpi=DPI)
        print(f"Saved {out}")

    plt.close(fig)


if __name__ == "__main__":
    np.random.seed(42)
    main()
