#!/usr/bin/env python3
"""
Improved log-likelihood distribution figure.

Fixes vs. original:
- Per-model loglik values are z-scored against the training distribution so
  that all 20 panels share the same x-range — cross-panel comparison is now
  possible at a glance.
- Group means shown as bold ticks at the bottom rather than thin lines.
- Okabe-Ito palette via style_improved.
"""
from __future__ import annotations

import sys
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from cstims import constants, paths
from cstims.paper.style_improved import (
    apply_style, FONT, DPI, W_DOUBLE,
    COLOR_CSTIM, COLOR_BASELINE, COLOR_TRAIN,
)

apply_style()

OOD_DATA = paths.ood_data_dir() / "pca_loglik.csv"
FIGURES  = Path(__file__).resolve().parent
ALL_MODELS = constants.MODEL_SETS["all_models"]

GROUPS = ["training", "vicco", "all_models"]
GROUP_LABELS = {
    "training":   "Training (n=4712)",
    "vicco":      "Baseline (n=292)",
    "all_models": "Controversial — All Models (n=100)",
}
COLORS = {
    "training":   COLOR_TRAIN,
    "vicco":      COLOR_BASELINE,
    "all_models": COLOR_CSTIM,
}

NROWS, NCOLS = 4, 5


def kde_curve(values, x_grid):
    values = values[np.isfinite(values)]
    if len(values) < 5:
        return None
    kde = gaussian_kde(values, bw_method="scott")
    return kde(x_grid)


def main():
    df = pd.read_csv(OOD_DATA)
    df_avg = (df.groupby(["model", "stimulus_group", "stimulus_idx"])
                ["loglik_pred_raw"].mean().reset_index())

    fig, axes = plt.subplots(NROWS, NCOLS,
                              figsize=(W_DOUBLE, W_DOUBLE * 0.70))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.08,
                          hspace=0.40, wspace=0.18)
    axes_flat = axes.flatten()

    # Compute global z-scored x-range
    x_grid = np.linspace(-6, 4, 400)

    for idx, model in enumerate(ALL_MODELS):
        ax = axes_flat[idx]
        mdf = df_avg[df_avg["model"] == model]
        train_vals = mdf[mdf["stimulus_group"] == "training"]["loglik_pred_raw"].values
        if len(train_vals) < 5:
            continue
        mu, sigma = train_vals.mean(), train_vals.std() + 1e-9

        for group in GROUPS:
            vals = mdf[mdf["stimulus_group"] == group]["loglik_pred_raw"].values
            if len(vals) == 0:
                continue
            z = (vals - mu) / sigma
            y = kde_curve(z, x_grid)
            if y is None:
                continue
            color = COLORS[group]
            ax.plot(x_grid, y, color=color, lw=1.5,
                    label=GROUP_LABELS[group])
            ax.fill_between(x_grid, y, alpha=0.18, color=color, linewidth=0)
            # Group mean as filled triangle at the bottom
            ax.scatter(z.mean(), 0, marker="v", color=color, s=22,
                        edgecolor="white", linewidth=0.6, zorder=5,
                        clip_on=False)

        display = constants.MODEL_DISPLAY_NAMES.get(model, model)
        ax.set_title(display, fontsize=FONT["small"] + 1, fontweight="bold",
                       pad=2)
        ax.tick_params(labelsize=FONT["small"] - 1)
        ax.set_xlim(-6, 4)
        ax.set_yticks([])
        ax.set_ylim(bottom=0)

    for ax in axes_flat[len(ALL_MODELS):]:
        ax.set_visible(False)

    # Shared x-label and legend (positioned in figure margins, no overlap)
    fig.text(0.5, 0.025,
             "log p(x | PPCA) — z-scored vs training distribution",
             ha="center", va="center", fontsize=FONT["axis_label"])
    handles = [mpatches.Patch(color=COLORS[g], label=GROUP_LABELS[g])
                for g in GROUPS]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               fontsize=FONT["small"], bbox_to_anchor=(0.5, 0.99),
               frameon=False)

    for ext in ("pdf", "png"):
        out = FIGURES / f"loglik_distributions_raw_improved.{ext}"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
