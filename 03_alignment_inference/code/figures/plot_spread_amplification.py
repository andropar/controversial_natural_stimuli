#!/usr/bin/env python3
"""
Plot spread amplification summary.

Four-panel figure:
  (a) Max–min range ratio (controversial / baseline)
  (b) IQR ratio
  (c) Coefficient of variation ratio
  (d) Mean pairwise difference ratio

Outputs:
    figures/spread_amplification.pdf/png

Usage:
    python plot_spread_amplification.py
"""

import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
_SHARE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))  # project root
sys.path.insert(0, str(_PAPER / "figures"))  # for shared figure style
sys.path.insert(0, str(_SHARE_ROOT / "shared" / "code" / "paper_helpers"))
sys.path.insert(0, str(_SHARE_ROOT / "shared" / "code" / "paper_helpers" / "figures"))
import config

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures" / "supplementary"
PNG_DIR = FIGURES_DIR / "png"

from style_improved import apply_style, FONT, DPI, W_DOUBLE

apply_style()

MODEL_SET_ORDER = ["sota", "architecture", "training_objective", "dataset", "all_models"]

MODEL_SET_LABELS = {
    "sota": "SOTA",
    "architecture": "Arch.",
    "training_objective": "Train.\nObj.",
    "dataset": "Dataset",
    "all_models": "All\nModels",
}

METHOD_COLORS = {
    "wrsa_transfer": "#E74C3C",
    "crsa": "#3498DB",
}

METHOD_LABELS = {
    "wrsa_transfer": "mRSA",
    "crsa": "fRSA",
}


# Map from ratio column to the corresponding p-value column in summary
RATIO_TO_P_COL = {
    "range_ratio": "ttest_range_p",
    "iqr_ratio": "ttest_iqr_p",
    "cv_ratio": "ttest_cv_p",
    "pairwise_diff_ratio": "ttest_pairwise_p",
}


def plot_ratio_panel(ax, detailed, summary, ratio_col, ylabel, title,
                     show_legend=False, show_xticklabels=True):
    """Plot a single ratio panel."""
    methods = ["wrsa_transfer", "crsa"]
    n_methods = len(methods)
    n_sets = len(MODEL_SET_ORDER)
    x = np.arange(n_sets)
    width = 0.35

    p_col = RATIO_TO_P_COL[ratio_col]

    for j, method in enumerate(methods):
        means = []
        sems = []
        sigs = []

        for model_set in MODEL_SET_ORDER:
            ms_data = detailed[
                (detailed["model_set"] == model_set) &
                (detailed["method"] == method)
            ][ratio_col]

            if len(ms_data) == 0:
                means.append(0)
                sems.append(0)
                sigs.append("")
                continue

            means.append(ms_data.mean())
            sems.append(ms_data.sem())

            # Get metric-specific significance from summary
            if summary is not None and p_col in summary.columns:
                row = summary[
                    (summary["model_set"] == model_set) &
                    (summary["method"] == method)
                ]
                if not row.empty:
                    p = row.iloc[0][p_col]
                    if p < 0.001:
                        sigs.append("***")
                    elif p < 0.01:
                        sigs.append("**")
                    elif p < 0.05:
                        sigs.append("*")
                    else:
                        sigs.append("")
                else:
                    sigs.append("")
            else:
                sigs.append("")

        offset = (j - (n_methods - 1) / 2) * width
        bars = ax.bar(
            x + offset, means, width,
            yerr=sems, capsize=2,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            alpha=0.85,
            error_kw=dict(linewidth=0.8),
        )

        # Add significance stars
        for i, (bar, sig) in enumerate(zip(bars, sigs)):
            if sig:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + sems[i] + 0.05,
                    sig, ha="center", va="bottom", fontsize=6,
                    fontweight="bold",
                )

    # Reference line at ratio = 1 (no amplification)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, zorder=0)

    ax.set_xticks(x)
    if show_xticklabels:
        ax.set_xticklabels([MODEL_SET_LABELS[ms] for ms in MODEL_SET_ORDER],
                           fontsize=FONT["tick"])
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    detailed_path = DATA_DIR / "spread_statistics.csv"
    if not detailed_path.exists():
        print(f"Missing {detailed_path}. Run 06_compute_spread_statistics.py first.")
        return

    detailed = pd.read_csv(detailed_path)

    summary_path = DATA_DIR / "spread_statistics_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else None

    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE, 10.8))

    panels = [
        {
            "ax": axes[0, 0], "label": "a",
            "ratio_col": "range_ratio",
            "ylabel": "Ratio\n(cstim / baseline)",
            "title": "Max–min range",
            "show_xticklabels": False,
            "show_legend": True,
        },
        {
            "ax": axes[0, 1], "label": "b",
            "ratio_col": "iqr_ratio",
            "ylabel": "",
            "title": "Interquartile range",
            "show_xticklabels": False,
            "show_legend": False,
        },
        {
            "ax": axes[1, 0], "label": "c",
            "ratio_col": "cv_ratio",
            "ylabel": "Ratio\n(cstim / baseline)",
            "title": "Coefficient of variation",
            "show_xticklabels": True,
            "show_legend": False,
        },
        {
            "ax": axes[1, 1], "label": "d",
            "ratio_col": "pairwise_diff_ratio",
            "ylabel": "",
            "title": "Mean pairwise |diff|",
            "show_xticklabels": True,
            "show_legend": False,
        },
    ]

    # Find a shared y-max across all panels
    y_maxes = []
    for p in panels:
        col = p["ratio_col"]
        for ms in MODEL_SET_ORDER:
            for method in ["wrsa_transfer", "crsa"]:
                vals = detailed[
                    (detailed["model_set"] == ms) &
                    (detailed["method"] == method)
                ][col]
                if len(vals) > 0:
                    y_maxes.append(vals.mean() + vals.sem())
    y_max = max(max(y_maxes) * 1.2, 3.5)

    for p in panels:
        plot_ratio_panel(
            p["ax"], detailed, summary,
            ratio_col=p["ratio_col"],
            ylabel=p["ylabel"],
            title=p["title"],
            show_legend=p["show_legend"],
            show_xticklabels=p["show_xticklabels"],
        )
        p["ax"].set_ylim(0, y_max)
        p["ax"].text(-0.12, 1.08, p["label"], transform=p["ax"].transAxes,
                     fontsize=FONT["panel_label"], fontweight="bold", va="top")

    plt.tight_layout(h_pad=2.0, w_pad=2.5)

    out_pdf = FIGURES_DIR / "spread_amplification.pdf"
    out_png = PNG_DIR / "spread_amplification.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=DPI)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")

    plt.close()


if __name__ == "__main__":
    main()
