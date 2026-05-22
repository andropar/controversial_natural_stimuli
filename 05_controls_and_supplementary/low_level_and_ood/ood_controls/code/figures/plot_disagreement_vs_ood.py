#!/usr/bin/env python3
"""
Plot model-disagreement vs OOD-ness from 06_ood/04 (corrected design with
per-set vicco references).

The CSVs now have a `ref_for_set` column: vicco rows appear once per cstim
set (because each cstim set has its own vicco reference using the same model
roster). When showing "vicco" we pick `ref_for_set == 'all_models'` for the
canonical 20-model view.

Outputs:
  disagreement_vs_ood_pairs.{pdf,png}     pair-level scatter, ood_mean + ood_max
  disagreement_vs_ood_images.{pdf,png}    image-level overlay
  disagreement_vs_ood_residual.{pdf,png}  residual disagreement with matched ref
  disagreement_vs_ood_vicreg.{pdf,png}    VICReg sensitivity (all_models)
"""

import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
_SHARE_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_SHARE_ROOT / "shared" / "code" / "paper_helpers"))
sys.path.insert(0, str(_SHARE_ROOT / "shared" / "code" / "paper_helpers" / "figures"))

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import config

OOD_DIR = _PAPER / "results"
PAIRS   = OOD_DIR / "disagreement_vs_ood_pairs.csv"
IMGS    = OOD_DIR / "disagreement_vs_ood_images.csv"
SUMMARY = OOD_DIR / "disagreement_vs_ood_summary.csv"
SUPP_FIGS = Path(__file__).resolve().parents[2] / "figures" / "supplementary"
SUPP_PNG_DIR = SUPP_FIGS / "png"

GROUPS = ["all_models", "architecture", "training_objective", "sota", "dataset"]
LABELS = {
    "all_models": "All models", "architecture": "Architecture",
    "training_objective": "Training obj.", "sota": "SOTA",
    "dataset": "Dataset", "vicco": "Baseline (vicco)",
}
COLORS = {
    "all_models": "#d6604d", "architecture": "#fb6a4a",
    "training_objective": "#fdae6b", "sota": "#bf812d",
    "dataset": "#c994c7", "vicco": "#2166ac",
}


def vicco_canonical(df):
    """For figures showing vicco overall, use the 20-model reference (all_models)."""
    return df[(df["stim_set"] == "vicco") & (df["ref_for_set"] == "all_models")
              & (df["variant"] == "all")]


def cstim_main(df):
    return df[(df["stim_set"] != "vicco") & (df["variant"] == "all")]


# --------------------------------------------------------------------------------
# (1) Pair-level scatter — one panel per stim_set, both ood_mean and ood_max
# --------------------------------------------------------------------------------

def fig_pairs(pairs):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    axes = axes.flatten()

    panels = list(GROUPS) + ["vicco"]
    for ax, stim_set in zip(axes, panels):
        if stim_set == "vicco":
            sub = vicco_canonical(pairs)
            ttl = LABELS["vicco"] + " (ref for all_models)"
        else:
            sub = pairs[(pairs["stim_set"] == stim_set) & (pairs["variant"] == "all")]
            ttl = LABELS[stim_set]
        if sub.empty:
            ax.set_visible(False)
            continue

        ax.scatter(sub["pair_ood_mean"], sub["pair_disagreement"],
                   s=4, alpha=0.18, color=COLORS[stim_set], edgecolor="none",
                   label="ood_mean")
        ax.scatter(sub["pair_ood_max"], sub["pair_disagreement"],
                   s=4, alpha=0.18, color="black", edgecolor="none",
                   label="ood_max")
        r_m, _ = stats.spearmanr(sub["pair_ood_mean"], sub["pair_disagreement"])
        r_x, _ = stats.spearmanr(sub["pair_ood_max"],  sub["pair_disagreement"])
        ax.text(0.97, 0.97,
                f"r(mean)={r_m:+.2f}\nr(max)={r_x:+.2f}\nn={len(sub):,}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=COLORS[stim_set], alpha=0.85))
        ax.set_title(ttl, fontsize=10, fontweight="bold")
        ax.set_xlabel("Pair OOD (loglik z, pred)", fontsize=8)
        ax.set_ylabel("Pair model-disagreement (var across models)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        "Pair-level disagreement vs OOD (per-set model rosters; both ood_mean and ood_max)",
        fontsize=11)
    return fig


# --------------------------------------------------------------------------------
# Residual disagreement using each set's matched vicco reference
# --------------------------------------------------------------------------------

def fig_residual(imgs):
    """Box / strip per stim_set of `residual` against its matched vicco ref."""
    fig, ax = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)
    positions, data, colors, labels = [], [], [], []

    # Plot the cstim sets first
    cs = cstim_main(imgs)
    for k, stim_set in enumerate(GROUPS):
        sub = cs[cs["stim_set"] == stim_set]
        if sub.empty:
            continue
        positions.append(k)
        data.append(sub["residual"].values)
        colors.append(COLORS[stim_set])
        labels.append(LABELS[stim_set])

    # Plot vicco at the end (its residual under the canonical (all_models) ref —
    # zero by construction)
    vsub = vicco_canonical(imgs)
    if not vsub.empty:
        positions.append(len(GROUPS))
        data.append(vsub["residual"].values)
        colors.append(COLORS["vicco"])
        labels.append(LABELS["vicco"] + "\n(ref=all_models)")

    bp = ax.boxplot(data, positions=positions, widths=0.55,
                    patch_artist=True, showfliers=False, medianprops=dict(color="black"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    rng = np.random.default_rng(0)
    for pos, vals, c in zip(positions, data, colors):
        x_jit = pos + rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(x_jit, vals, s=8, color=c, alpha=0.45, edgecolor="none")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(
        "Residual disagreement\n(disagreement − vicco-fit prediction at same OOD)",
        fontsize=9)
    ax.set_title(
        "Residual model-disagreement at matched OOD\n"
        "Each set's residual uses a vicco reference fit on the SAME model roster",
        fontsize=10)
    ax.tick_params(labelsize=8)
    return fig


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

def save(fig, name):
    SUPP_FIGS.mkdir(parents=True, exist_ok=True)
    SUPP_PNG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = SUPP_FIGS / f"{name}.pdf"
    png_path = SUPP_PNG_DIR / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {pdf_path}")
    print(f"Saved → {png_path}")
    plt.close(fig)


def main():
    pairs = pd.read_csv(PAIRS)
    imgs  = pd.read_csv(IMGS)

    save(fig_pairs(pairs),    "disagreement_vs_ood_pairs")
    save(fig_residual(imgs),  "disagreement_vs_ood_residual")

    print("\nSummary:")
    print(pd.read_csv(SUMMARY).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
