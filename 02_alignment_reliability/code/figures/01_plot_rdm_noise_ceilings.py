#!/usr/bin/env python3
"""
Plot RDM noise ceilings: cross-subject mean bars with individual subject markers,
plus paired t-test significance brackets comparing each cstim group to vicco.

Inputs:
    02_alignment_reliability/results/rdm_noise_ceilings.csv

Outputs:
    02_alignment_reliability/figures/supplementary/rdm_noise_ceilings.pdf
    02_alignment_reliability/figures/supplementary/png/rdm_noise_ceilings.png

Usage:
    python 02_alignment_reliability/code/figures/01_plot_rdm_noise_ceilings.py
"""

import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parents[2]
SHARE_ROOT = STAGE.parent
PAPER_HELPERS = SHARE_ROOT / "src"
sys.path.insert(0, str(PAPER_HELPERS))
from cstims import constants, paths

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from cstims.paper.style_improved import apply_style, FONT, DPI, W_SINGLE

apply_style()

DATA_DIR = STAGE / "results"
FIGURES_DIR = STAGE / "figures" / "supplementary"
PNG_DIR = FIGURES_DIR / "png"

COLOR_CSTIM = "#D64541"
COLOR_BASE = "#2980B9"

GROUP_ORDER = ["all_models", "sota", "training_objective", "architecture", "dataset", "vicco"]
GROUP_LABELS = {
    "all_models": "All Models",
    "sota": "SotA",
    "training_objective": "Train. Obj.",
    "architecture": "Architecture",
    "dataset": "Dataset",
    "vicco": "Vicco\n(baseline)",
}


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_DIR / "rdm_noise_ceilings.csv")

    # For vicco: average across bootstraps per subject first
    vicco = df[df["stimulus_type"] == "vicco"].groupby("subject").agg(
        noise_ceiling_spearman=("noise_ceiling_spearman", "mean")
    ).reset_index()
    vicco["group"] = "vicco"

    cstim = df[df["stimulus_type"] == "controversial"][
        ["subject", "group", "noise_ceiling_spearman"]
    ].copy()

    combined = pd.concat([cstim, vicco], ignore_index=True)

    # Per-subject vicco mean for pairing
    vicco_per_subject = vicco.set_index("subject")["noise_ceiling_spearman"]

    return combined, vicco_per_subject


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    df, vicco_per_subject = load_data()

    fig, ax = plt.subplots(figsize=(W_SINGLE, 3.5))

    x = np.arange(len(GROUP_ORDER))
    bar_w = 0.55

    bar_tops = {}

    for i, group in enumerate(GROUP_ORDER):
        g = df[df["group"] == group]
        if g.empty:
            continue

        vals = g["noise_ceiling_spearman"].values
        mean = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0

        color = COLOR_BASE if group == "vicco" else COLOR_CSTIM

        ax.bar(x[i], mean, bar_w, color=color, alpha=0.75, zorder=2,
               yerr=sem, capsize=3,
               error_kw=dict(linewidth=0.8, color="black", zorder=4))

        bar_tops[group] = vals.max()

        # Individual subject markers
        jitter = np.linspace(-0.12, 0.12, len(vals))
        for j, v in enumerate(vals):
            ax.scatter(x[i] + jitter[j], v, s=14, color=color,
                       edgecolors="white", linewidths=0.4, zorder=5, alpha=0.9)

    # Paired t-tests: each cstim group vs vicco — annotate directly above bar
    for group in [g for g in GROUP_ORDER if g != "vicco"]:
        g = df[df["group"] == group].set_index("subject")
        common = g.index.intersection(vicco_per_subject.index)
        cstim_vals = g.loc[common, "noise_ceiling_spearman"].values
        vicco_vals = vicco_per_subject.loc[common].values

        diffs = cstim_vals - vicco_vals
        if len(diffs) >= 2:
            t, p_two = stats.ttest_1samp(diffs, 0)
            p_one = p_two / 2 if t < 0 else 1.0  # one-sided: cstim < vicco
        else:
            p_one = np.nan

        stars = sig_stars(p_one) if not np.isnan(p_one) else ""
        if not stars or stars == "ns":
            continue

        gi = GROUP_ORDER.index(group)
        top = bar_tops[group] + 0.02
        label = f"{stars}, p={p_one:.3f}"
        ax.text(x[gi], top, label, ha="center", va="bottom",
                fontsize=6, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER],
                       fontsize=FONT["tick"])
    ax.set_ylabel("RDM noise ceiling ($r_{SB}$)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    ax.axhline(0, color="black", linewidth=0.5)

    # Divider between cstim groups and vicco
    ax.axvline(len(GROUP_ORDER) - 1.5, color="#CCCCCC", linewidth=0.8,
               linestyle="--", zorder=0)

    fig.tight_layout()

    out_pdf = FIGURES_DIR / "rdm_noise_ceilings.pdf"
    out_png = PNG_DIR / "rdm_noise_ceilings.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=DPI)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")

    plt.close(fig)


if __name__ == "__main__":
    main()
