#!/usr/bin/env python3
"""
Combined figure: subject consistency + top unexplained image pairs.

Layout (2 rows):
  Row 1: [A] Inter-subject consistency bars (4 conditions)
         [B] NC-normalized scores: vicco vs controversial (strip)
  Row 2: [C] Top 4 pairs — brain less similar than models predict (positive residual)
         [D] Top 4 pairs — brain more similar than models predict (negative residual)

Requires (run first):
  01_subject_consistency.py  -> results/subject_consistency.csv, results/subject_residuals.npz
  00c_compute_vicco_residuals.py -> results/vicco_consistency.csv
  00_nc_comparison.py        -> results/nc_comparison.csv

Usage:
    python figures/plot_combined.py
"""

import sys
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

STAGE = Path(__file__).resolve().parents[2]
SHARE_ROOT = STAGE.parents[2]

DATA_DIR = STAGE / "results"
IMAGE_DIR = (
    SHARE_ROOT
    / "00_stimulus_selection"
    / "decision_checks"
    / "selection_evaluation"
    / "results"
    / "all_models"
    / "images"
)
FIG_DIR = STAGE / "figures"
SUBJECTS_CSTIM = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]

try:
    from cstims.paper.style_improved import apply_style, DPI, W_DOUBLE
    apply_style()
except ImportError:
    DPI = 150
    W_DOUBLE = 14

COLOR_CSTIM = "#D64541"
COLOR_VICCO = "#2980B9"
COLOR_RES   = "#888888"


def load_image(idx):
    path = IMAGE_DIR / f"image_{idx:04d}.png"
    if not path.exists():
        path = IMAGE_DIR / f"image_{idx:04d}.jpg"
    return Image.open(path).convert("RGB")


def deduplicate_pairs(pairs_df, max_appearances=1):
    from collections import defaultdict
    counts = defaultdict(int)
    selected = []
    for _, row in pairs_df.iterrows():
        i, j = int(row["img_i"]), int(row["img_j"])
        if counts[i] < max_appearances and counts[j] < max_appearances:
            selected.append(row)
            counts[i] += 1
            counts[j] += 1
    return pd.DataFrame(selected)


def panel_consistency(ax, cstim_df, vicco_df, subjects_cstim):
    """4-bar aggregated inter-subject consistency with individual dots."""
    data = {
        "Vicco\nbrain":       vicco_df["brain_rho"],
        "Vicco\nresidual":    vicco_df["residual_rho"],
        "Cstim\nbrain":       cstim_df["brain_rho"],
        "Cstim\nresidual":    cstim_df["residual_rho"],
    }
    colors = [COLOR_VICCO, COLOR_VICCO, COLOR_CSTIM, COLOR_CSTIM]
    alphas = [0.85, 0.45, 0.85, 0.45]

    x = np.arange(len(data))
    for i, (label, vals) in enumerate(data.items()):
        ax.bar(i, vals.mean(), yerr=vals.sem(), capsize=3,
               color=colors[i], alpha=alphas[i], width=0.55,
               error_kw={"linewidth": 1.2})
        jitter = np.random.default_rng(i).uniform(-0.1, 0.1, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color="black", s=14, zorder=3, alpha=0.65)

    ax.set_xticks(x)
    ax.set_xticklabels(list(data.keys()), fontsize=8)
    ax.set_ylabel("Spearman ρ (inter-subject)", fontsize=8)
    ax.set_title("A   Inter-subject consistency", fontsize=9, loc="left", fontweight="bold")
    ax.set_ylim(0, 0.55)
    ax.axhline(0, color="black", linewidth=0.6)


def panel_nc_scores(ax, nc_df):
    """Strip plot of NC-normalized scores: vicco vs controversial, all models."""
    cstim_scores = nc_df["nc_normalized_mean_cstim"].values
    vicco_scores = nc_df["nc_normalized_mean_vicco"].values
    n = len(cstim_scores)

    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.12, 0.12, n)

    ax.scatter(np.zeros(n) + jitter, vicco_scores, color=COLOR_VICCO, s=20, alpha=0.75, zorder=3)
    ax.scatter(np.ones(n) + jitter,  cstim_scores, color=COLOR_CSTIM, s=20, alpha=0.75, zorder=3)

    # Connect matched models
    for v, c, j in zip(vicco_scores, cstim_scores, jitter):
        ax.plot([j, 1 + j], [v, c], color="gray", linewidth=0.5, alpha=0.4, zorder=2)

    # Means
    ax.hlines(vicco_scores.mean(), -0.3, 0.3, color=COLOR_VICCO, linewidth=2, zorder=4)
    ax.hlines(cstim_scores.mean(),  0.7, 1.3, color=COLOR_CSTIM,  linewidth=2, zorder=4)

    # NC ceiling line
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="Noise ceiling")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Vicco\nbaseline", "Controversial\n(all_models)"], fontsize=8)
    ax.set_ylabel("NC-normalized mRSA", fontsize=8)
    ax.set_title("B   NC-normalized brain alignment", fontsize=9, loc="left", fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7, loc="upper right")


def panel_pairs(axes_row, pairs_df, direction, n_pairs=4):
    """
    Plot n_pairs image pairs in axes_row (list of 4 axes: img_i, img_j, bar, spacer pattern).
    direction: 'positive' or 'negative'
    Each pair uses 3 axes: image_i, image_j, bar chart.
    axes_row has n_pairs * 3 axes.
    """
    sign = 1 if direction == "positive" else -1
    subset = pairs_df[np.sign(pairs_df["mean_signed_residual"]) == sign]
    subset = deduplicate_pairs(subset.sort_values("mean_abs_residual", ascending=False))
    subset = subset.head(n_pairs)

    title_dir = "less similar" if direction == "positive" else "more similar"
    color_dir = COLOR_CSTIM if direction == "positive" else COLOR_VICCO

    panel_label = "C" if direction == "positive" else "D"
    axes_row[0].set_title(
        f"{panel_label}   Brain treats pairs as {title_dir} than models predict",
        fontsize=9, loc="left", fontweight="bold"
    )

    subjects = [c.replace("residual_", "") for c in pairs_df.columns if c.startswith("residual_sub")]

    for col, (_, pair) in enumerate(subset.iterrows()):
        ax_i   = axes_row[col * 3]
        ax_j   = axes_row[col * 3 + 1]
        ax_bar = axes_row[col * 3 + 2]

        img_i = load_image(int(pair["img_i"]))
        img_j = load_image(int(pair["img_j"]))

        ax_i.imshow(img_i)
        ax_i.axis("off")
        ax_i.set_title(f"#{int(pair['img_i'])}", fontsize=6, pad=1)

        ax_j.imshow(img_j)
        ax_j.axis("off")
        ax_j.set_title(f"#{int(pair['img_j'])}", fontsize=6, pad=1)

        subj_vals = [pair[f"residual_{s}"] for s in subjects]
        bar_colors = [COLOR_CSTIM if v > 0 else COLOR_VICCO for v in subj_vals]
        short_labels = [s.replace("sub-", "S") for s in subjects]
        ax_bar.barh(short_labels, subj_vals, color=bar_colors, alpha=0.75, height=0.5)
        ax_bar.axvline(0, color="black", linewidth=0.8)
        ax_bar.set_xlim(-3.5, 3.5)
        ax_bar.tick_params(axis="y", labelsize=5.5)
        ax_bar.tick_params(axis="x", labelsize=5.5)
        ax_bar.set_title(
            f"|r|={pair['mean_abs_residual']:.2f}\ncons={pair['sign_consistency']:.2f}",
            fontsize=6, pad=1
        )


def main():
    # Load data
    cstim_cons = pd.read_csv(DATA_DIR / "subject_consistency.csv")
    vicco_cons  = pd.read_csv(DATA_DIR / "vicco_consistency.csv")
    nc_df       = pd.read_csv(DATA_DIR / "nc_comparison.csv")
    pairs       = pd.read_csv(DATA_DIR / "consistent_top_pairs.csv")

    # -------------------------------------------------------------------------
    # Build figure
    # n_pairs=4, so row 2 needs 4 * 3 = 12 axes per half-row
    # Layout: 2 main rows
    #   Row 1: [consistency bar | NC strip] — 2 panels
    #   Row 2a: top positive pairs (4 pairs × 3 axes each = 12 axes)
    #   Row 2b: top negative pairs (same)
    # -------------------------------------------------------------------------
    N_PAIRS = 4
    fig = plt.figure(figsize=(W_DOUBLE, 11))

    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[2.5, 2.5, 2.5],
        hspace=0.45,
    )

    # Row 1: consistency + NC scores
    row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.35)
    ax_cons = fig.add_subplot(row1[0])
    ax_nc   = fig.add_subplot(row1[1])

    # Row 2: positive pairs
    row2 = gridspec.GridSpecFromSubplotSpec(
        1, N_PAIRS * 3,
        subplot_spec=outer[1],
        wspace=0.05,
        width_ratios=[2, 2, 1.5] * N_PAIRS,
    )
    axes_pos = [fig.add_subplot(row2[i]) for i in range(N_PAIRS * 3)]

    # Row 3: negative pairs
    row3 = gridspec.GridSpecFromSubplotSpec(
        1, N_PAIRS * 3,
        subplot_spec=outer[2],
        wspace=0.05,
        width_ratios=[2, 2, 1.5] * N_PAIRS,
    )
    axes_neg = [fig.add_subplot(row3[i]) for i in range(N_PAIRS * 3)]

    # Fill panels
    panel_consistency(ax_cons, cstim_cons, vicco_cons, SUBJECTS_CSTIM)
    panel_nc_scores(ax_nc, nc_df)
    panel_pairs(axes_pos, pairs, direction="positive", n_pairs=N_PAIRS)
    panel_pairs(axes_neg, pairs, direction="negative", n_pairs=N_PAIRS)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "combined.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "combined.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/combined.pdf/png")


if __name__ == "__main__":
    main()
