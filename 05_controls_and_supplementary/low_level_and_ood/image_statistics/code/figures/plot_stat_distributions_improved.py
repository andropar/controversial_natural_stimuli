#!/usr/bin/env python3
"""
Improved low-level image-statistics figure.

Fixes vs. original:
- Effect-size heatmap (set × statistic) added as panel a, giving the broad
  view at a glance. Cell colour = median z-score of the set's distribution
  vs the training distribution (red = elevated, blue = suppressed).
- Boxplot panels (panel b) retained but reduced to 6 of the most
  informative statistics (those with the largest training-vs-baseline OR
  training-vs-controversial median z-shifts).
- Train scatter sub-sampled to 200 points so its visual density matches
  the other groups (otherwise it dominates by sheer count).
- Okabe-Ito palette via style_improved.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))
sys.path.insert(0, str(_PAPER / "figures"))

from style_improved import (
    apply_style, FONT, DPI, W_DOUBLE,
    COLOR_CSTIM, COLOR_BASELINE, COLOR_TRAIN, add_panel_label,
)

apply_style()

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "results" / "image_stats.csv"
OUT_BASE = HERE / "stat_distributions_improved"

STAT_ORDER = [
    ("lum_mean", "Luminance"),
    ("lum_rms", "Contrast"),
    ("colorfulness", "Colorfulness"),
    ("lab_chroma_mean", "LAB chroma"),
    ("hue_entropy", "Hue entropy"),
    ("sf_slope", "1/f slope"),
    ("sf_high_low_ratio", "High/low SF"),
    ("edge_mag_mean", "Edge magnitude"),
    ("orient_anisotropy", "Orient. anisotropy"),
    ("edge_com_x", "Edge horiz. CoM"),
    ("symmetry_lr", "L–R symmetry"),
    ("entropy", "Shannon entropy"),
    ("jpeg_ratio", "JPEG ratio"),
]
SET_ORDER = ["deepvision_train", "vicco",
              "all_models", "sota", "training_objective", "architecture", "dataset"]
SET_LABELS = {
    "deepvision_train": "Train",
    "vicco": "Baseline",
    "all_models": "All",
    "architecture": "Arch.",
    "dataset": "Dataset",
    "sota": "SOTA",
    "training_objective": "Train. Obj.",
}


def color_for(s):
    if s == "deepvision_train":
        return COLOR_TRAIN
    if s == "vicco":
        return COLOR_BASELINE
    return COLOR_CSTIM


def main():
    df = pd.read_csv(CSV)
    sets = [s for s in SET_ORDER if s in df["stimulus_set"].unique()]

    # ---- Compute median z-scores vs training distribution ----
    z_table = pd.DataFrame(index=[s for s in sets if s != "deepvision_train"],
                           columns=[lab for _, lab in STAT_ORDER], dtype=float)
    for col, lab in STAT_ORDER:
        train_vals = df.loc[df["stimulus_set"] == "deepvision_train", col].dropna().values
        if len(train_vals) < 5:
            continue
        mu, sigma = train_vals.mean(), train_vals.std() + 1e-9
        for s in sets:
            if s == "deepvision_train":
                continue
            vals = df.loc[df["stimulus_set"] == s, col].dropna().values
            if len(vals):
                z_table.loc[s, lab] = (np.median(vals) - mu) / sigma

    # ---- Pick top 6 statistics by maximum |z| across non-train sets ----
    max_abs_z = z_table.abs().max(axis=0).sort_values(ascending=False)
    top6 = max_abs_z.head(6).index.tolist()
    top6_cols = [(c, l) for (c, l) in STAT_ORDER if l in top6]

    # ---- Build the figure: 2 rows ----
    fig = plt.figure(figsize=(W_DOUBLE, 8.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.3], hspace=0.35,
                            left=0.08, right=0.98, top=0.96, bottom=0.10)

    # ---- Panel a: heatmap ----
    ax_h = fig.add_subplot(gs[0])
    z = z_table.values.astype(float)
    vmax = max(np.nanmax(np.abs(z)), 0.2)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax_h.imshow(z, aspect="auto", cmap="RdBu_r", norm=norm)
    ax_h.set_xticks(range(len(z_table.columns)))
    ax_h.set_xticklabels(z_table.columns, rotation=30, ha="right")
    ax_h.set_yticks(range(len(z_table.index)))
    ax_h.set_yticklabels([SET_LABELS[s] for s in z_table.index])
    ax_h.set_title("Median image-statistic z-score vs training distribution "
                    "(controversial drift mirrors baseline drift)", pad=4)

    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            v = z[i, j]
            if not np.isnan(v):
                ax_h.text(j, i, f"{v:+.1f}", ha="center", va="center",
                            fontsize=FONT["small"] - 1,
                            color="white" if abs(v) > vmax * 0.6 else "#222")

    cax = ax_h.inset_axes([1.01, 0.0, 0.012, 1.0])
    fig.colorbar(im, cax=cax, label="median z")
    add_panel_label(ax_h, "a", x=-0.04, y=1.05)

    # ---- Panel b: boxplots for top 6 statistics ----
    gs_b = gs[1].subgridspec(2, 3, wspace=0.30, hspace=0.55)
    rng = np.random.default_rng(7)
    xs = np.arange(len(sets))

    panel_b_axes = []
    for idx, (col, label) in enumerate(top6_cols):
        ax = fig.add_subplot(gs_b[idx // 3, idx % 3])
        panel_b_axes.append(ax)
        data = []
        for s in sets:
            vals = df.loc[df["stimulus_set"] == s, col].dropna().values
            # Subsample train to match other groups visually
            if s == "deepvision_train" and len(vals) > 200:
                vals = rng.choice(vals, 200, replace=False)
            data.append(vals)

        # Reference lines
        if "deepvision_train" in sets:
            t_med = np.median(data[sets.index("deepvision_train")])
            ax.axhline(t_med, color=COLOR_TRAIN, lw=0.6, ls="--", alpha=0.55, zorder=0)
        if "vicco" in sets:
            v_med = np.median(data[sets.index("vicco")])
            ax.axhline(v_med, color=COLOR_BASELINE, lw=0.6, ls="--", alpha=0.55, zorder=0)

        for xi, vals, s in zip(xs, data, sets):
            c = color_for(s)
            jitter = rng.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(xi + jitter, vals, s=2.5, color=c,
                        alpha=0.35, linewidths=0, zorder=1)

        bp = ax.boxplot(data, positions=xs, widths=0.55, showfliers=False,
                          patch_artist=True, zorder=2,
                          medianprops=dict(color="white", lw=1.0),
                          whiskerprops=dict(lw=0.6),
                          capprops=dict(lw=0.6),
                          boxprops=dict(lw=0.6))
        for patch, s in zip(bp["boxes"], sets):
            patch.set_facecolor(color_for(s)); patch.set_alpha(0.75)
            patch.set_edgecolor(color_for(s))

        ax.set_title(label, fontsize=FONT["title"], pad=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([SET_LABELS[s] for s in sets], rotation=30,
                            ha="right", fontsize=FONT["small"])

    if panel_b_axes:
        add_panel_label(panel_b_axes[0], "b", x=-0.18, y=1.18)

    for ext in (".pdf", ".png"):
        fig.savefig(OUT_BASE.with_suffix(ext), dpi=DPI)
        print(f"Saved {OUT_BASE.with_suffix(ext)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
