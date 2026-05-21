#!/usr/bin/env python3
"""
Plot distance metric robustness: scatter of correlation-distance fRSA vs
cosine-distance fRSA, with Spearman rho annotation.

Outputs:
    figures/distance_metric_robustness.pdf/png

Usage:
    python plot_distance_metric_robustness.py
"""

import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))  # project root
sys.path.insert(0, str(_PAPER / "figures"))  # for style.py
import config

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

DATA_DIR = config.STATS_DATA_DIR
FIGURES_DIR = Path(__file__).resolve().parent

from style import apply_style, FONT, DPI, W_SINGLE

apply_style()


def main():
    df = pd.read_csv(DATA_DIR / "distance_metric_robustness.csv")

    # Aggregate: mean across subjects per (model, model_set, stimulus_type, bootstrap_idx)
    agg = df.groupby(["model", "model_set", "stimulus_type", "bootstrap_idx"]).agg(
        crsa_correlation=("crsa_correlation", "mean"),
        crsa_cosine=("crsa_cosine", "mean"),
    ).reset_index()

    # Separate controversial and vicco for coloring
    cstim = agg[agg["stimulus_type"] == "controversial"]
    vicco = agg[agg["stimulus_type"] == "vicco"]

    # Overall stats
    rho_all, p_all = spearmanr(agg["crsa_correlation"], agg["crsa_cosine"])
    rho_cstim, _ = spearmanr(cstim["crsa_correlation"], cstim["crsa_cosine"])

    fig, ax = plt.subplots(1, 1, figsize=(W_SINGLE, W_SINGLE))

    # Plot vicco (gray) and controversial (colored)
    ax.scatter(vicco["crsa_correlation"], vicco["crsa_cosine"],
               s=8, alpha=0.3, c="gray", linewidths=0, label="Baseline (Vicco)", zorder=2)
    ax.scatter(cstim["crsa_correlation"], cstim["crsa_cosine"],
               s=12, alpha=0.7, c="#d62728", linewidths=0, label="Controversial", zorder=3)

    # Identity line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, ls="--", c="k", lw=0.5, alpha=0.4, zorder=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Regression line
    coeffs = np.polyfit(agg["crsa_correlation"], agg["crsa_cosine"], 1)
    x_line = np.array(lims)
    ax.plot(x_line, np.polyval(coeffs, x_line), c="#1f77b4", lw=1, alpha=0.7, zorder=2)

    ax.set_xlabel("fRSA (correlation distance)")
    ax.set_ylabel("fRSA (cosine distance)")
    ax.set_aspect("equal")

    # Annotation
    ax.text(0.05, 0.95, f"Spearman $\\rho$ = {rho_all:.3f}",
            transform=ax.transAxes, fontsize=FONT["annotation"], va="top")
    ax.text(0.05, 0.88, f"mean |$\\Delta$| = {(agg['crsa_correlation'] - agg['crsa_cosine']).abs().mean():.4f}",
            transform=ax.transAxes, fontsize=FONT["annotation"], va="top", color="0.4")

    ax.legend(fontsize=FONT["small"], loc="lower right", framealpha=0.8, handletextpad=0.3)

    fig.tight_layout()

    for ext in [".pdf", ".png"]:
        out = FIGURES_DIR / f"distance_metric_robustness{ext}"
        fig.savefig(out, dpi=DPI)
        print(f"Saved {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
