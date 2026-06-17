#!/usr/bin/env python3
"""
NC-normalized comparison: controversial vs vicco baseline.

Shows that the NC-normalized gap to ceiling is 2.5x larger on controversial
stimuli (0.38) than on vicco baseline (0.15), establishing that the unexplained
variance is reliable brain structure, not noise.

Outputs:
    results/nc_comparison.csv
    figures/nc_comparison.pdf/png

Usage:
    python 00_nc_comparison.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

STAGE = Path(__file__).resolve().parents[1]
SHARE_ROOT = STAGE.parents[2]
sys.path.insert(0, str(SHARE_ROOT / "src"))
from cstims.paper import config

STATS_DIR = SHARE_ROOT / "02_alignment_reliability" / "results"
DATA_DIR = STAGE / "results"
FIG_DIR = STAGE / "figures"

try:
    from cstims.paper.style_improved import apply_style, DPI, W_SINGLE, W_DOUBLE
    apply_style()
except ImportError:
    DPI = 150
    W_SINGLE = 6
    W_DOUBLE = 12


def main():
    nc_norm = pd.read_csv(STATS_DIR / "nc_normalized_summary.csv")

    # Focus on all_models model set, wrsa_transfer method
    df = nc_norm[
        (nc_norm["model_set"] == "all_models") &
        (nc_norm["method"] == "wrsa_transfer")
    ].copy()

    cstim = df[df["stimulus_type"] == "controversial"].sort_values("nc_normalized_mean", ascending=False)
    vicco = df[df["stimulus_type"] == "vicco"].sort_values("nc_normalized_mean", ascending=False)

    nc_cstim = cstim["noise_ceiling_mean"].iloc[0]
    nc_vicco = vicco["noise_ceiling_mean"].iloc[0]

    print("=== NC-normalized scores: all_models model set, wrsa_transfer ===")
    print(f"\nNoise ceiling (Spearman): controversial={nc_cstim:.3f} (sqrt={np.sqrt(nc_cstim):.3f})")
    print(f"                          vicco={nc_vicco:.3f}         (sqrt={np.sqrt(nc_vicco):.3f})")

    print(f"\n{'Model':<20} {'cstim score':>12} {'cstim NC-norm':>14} {'vicco score':>12} {'vicco NC-norm':>14}")
    print("-" * 76)
    models = cstim["model"].values
    for m in models:
        crow = cstim[cstim["model"] == m]
        vrow = vicco[vicco["model"] == m]
        if crow.empty or vrow.empty:
            continue
        print(f"{crow['display_name'].iloc[0]:<20} "
              f"{crow['score_mean'].iloc[0]:>12.3f} "
              f"{crow['nc_normalized_mean'].iloc[0]:>14.3f} "
              f"{vrow['score_mean'].iloc[0]:>12.3f} "
              f"{vrow['nc_normalized_mean'].iloc[0]:>14.3f}")

    best_cstim = cstim["nc_normalized_mean"].max()
    best_vicco = vicco["nc_normalized_mean"].max()
    print(f"\nBest NC-normalized: controversial={best_cstim:.3f}  vicco={best_vicco:.3f}")
    print(f"Gap to ceiling:     controversial={1-best_cstim:.3f}  vicco={1-best_vicco:.3f}")
    print(f"Gap ratio (cstim/vicco): {(1-best_cstim)/(1-best_vicco):.2f}x")

    # Save comparison CSV
    merged = pd.merge(
        cstim[["model", "display_name", "score_mean", "score_sem", "nc_normalized_mean", "nc_normalized_sem", "noise_ceiling_mean"]],
        vicco[["model", "score_mean", "score_sem", "nc_normalized_mean", "nc_normalized_sem", "noise_ceiling_mean"]],
        on="model", suffixes=("_cstim", "_vicco")
    )
    merged["gap_cstim"] = 1 - merged["nc_normalized_mean_cstim"]
    merged["gap_vicco"] = 1 - merged["nc_normalized_mean_vicco"]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(DATA_DIR / "nc_comparison.csv", index=False)
    print(f"\nSaved: results/nc_comparison.csv")

    # Figure: NC-normalized scores for both stimulus types, all models
    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 4.5), sharey=False)

    for ax, (rows, label, color) in zip(axes, [
        (cstim, "Controversial (all_models)", "#D64541"),
        (vicco, "Vicco baseline", "#2980B9"),
    ]):
        names = rows["display_name"].values[::-1]
        scores = rows["nc_normalized_mean"].values[::-1]
        sems = rows["nc_normalized_sem"].values[::-1]
        nc_val = rows["noise_ceiling_mean"].iloc[0]

        y = np.arange(len(names))
        ax.barh(y, scores, xerr=sems, color=color, alpha=0.8, height=0.6, capsize=2)
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="Noise ceiling")
        ax.axvline(np.sqrt(nc_val), color="gray", linestyle=":", linewidth=1,
                   label=f"NC (sqrt={np.sqrt(nc_val):.2f})")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("NC-normalized mRSA")
        ax.set_title(f"{label}\nNC={nc_val:.3f}, best gap={1-scores.max():.3f}", fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.legend(fontsize=7)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "nc_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "nc_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/nc_comparison.pdf/png")
    print("Done.")


if __name__ == "__main__":
    main()
