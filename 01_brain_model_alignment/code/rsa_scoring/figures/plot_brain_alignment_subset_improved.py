#!/usr/bin/env python3
"""
Improved brain_alignment_subset_scatter figure.

Fixes vs. original:
- Okabe-Ito palette (consistent with all paper figures, no red+green together).
- Panel labels (a/b).
- Slightly larger markers and clearer identity-line label.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))
sys.path.insert(0, str(_PAPER / "figures"))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from style_improved import (
    apply_style, FONT, DPI, W_DOUBLE,
    MODEL_SET_DISPLAY, MODEL_SET_COLORS,
    add_panel_label,
)
import config

apply_style()

# Reuse logic
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_brain_alignment_subset import (
    load_full_scores, load_subset_scores, prepare_paired_data,
)

FIGURES_DIR = Path(__file__).resolve().parent


def plot_comparison(df_full, df_subset, max_stim):
    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE * 0.65, 4.0))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.14, wspace=0.22)

    panel_letters = ["a", "b"]
    for ax, (method, label), letter in zip(
        axes,
        [("wrsa_transfer", "mixed RSA"), ("crsa", "fixed RSA")],
        panel_letters,
    ):
        all_full, all_sub = [], []

        for ms in ["sota", "training_objective", "architecture", "dataset", "all_models"]:
            full_data = prepare_paired_data(df_full, ms, method)
            sub_data  = prepare_paired_data(df_subset, ms, method)
            if full_data is None or sub_data is None:
                continue
            common = [m for m in full_data["models"] if m in sub_data["models"]]
            full_scores = [full_data["scores"][full_data["models"].index(m)] for m in common]
            sub_scores  = [sub_data["scores"][sub_data["models"].index(m)] for m in common]
            full_sem    = [full_data["sem"][full_data["models"].index(m)] for m in common]
            sub_sem     = [sub_data["sem"][sub_data["models"].index(m)] for m in common]

            ax.errorbar(
                full_scores, sub_scores, xerr=full_sem, yerr=sub_sem,
                fmt="o", color=MODEL_SET_COLORS[ms],
                markersize=5, alpha=0.85, capsize=1.5,
                elinewidth=0.6, linewidth=0,
                label=MODEL_SET_DISPLAY[ms],
            )
            all_full.extend(full_scores); all_sub.extend(sub_scores)

        if all_full:
            lo = min(min(all_full), min(all_sub))
            hi = max(max(all_full), max(all_sub))
            margin = (hi - lo) * 0.10
            lims = [lo - margin, hi + margin]
            ax.plot(lims, lims, color="#777", linestyle="--",
                    linewidth=0.7, alpha=0.7, zorder=0, label="Identity")
            ax.set_xlim(lims); ax.set_ylim(lims)

            r, _ = stats.pearsonr(all_full, all_sub)
            ax.text(0.04, 0.96, f"$r$ = {r:.3f}", transform=ax.transAxes,
                    fontsize=FONT["annotation"], va="top",
                    color="#222", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="none", alpha=0.85))

        ax.set_xlabel(f"{label} (N=100)")
        ax.set_ylabel(f"{label} (N={max_stim})")
        # Claim-first title; "r ≈ ..." is shown via inset.
        ax.set_title(f"{label}: first-{max_stim} subset tracks full set",
                      fontweight="bold", pad=4)
        ax.set_aspect("equal")
        add_panel_label(ax, letter, x=-0.13, y=1.04)

    axes[0].legend(frameon=True, framealpha=0.9, edgecolor="none",
                   fontsize=FONT["small"], loc="lower right",
                   handletextpad=0.4, handlelength=1.0)
    return fig


def main():
    df_full = load_full_scores()
    df_subset = load_subset_scores(20)
    fig = plot_comparison(df_full, df_subset, 20)
    for fmt in ("pdf", "png"):
        out = FIGURES_DIR / f"brain_alignment_subset_scatter_improved.{fmt}"
        fig.savefig(out, dpi=DPI)
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
