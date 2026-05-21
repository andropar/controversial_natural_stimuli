#!/usr/bin/env python3
"""
Plot in silico evaluation of stimulus selection quality.

Three-panel figure:
  (a) Error probability curves in raw feature space
  (b) Error probability curves in encoding (predicted brain) space
  (c) AUC improvement bar chart comparing raw vs encoding

Outputs:
    figures/insilico_evaluation.pdf/png

Usage:
    python plot_insilico_evaluation.py
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
from style import apply_style, FONT, DPI, W_DOUBLE

apply_style()

FIGURES_DIR = Path(__file__).resolve().parent

MODEL_SETS = ["sota", "architecture", "training_objective", "dataset", "all_models"]

MODEL_SET_LABELS = {
    "sota": "SOTA",
    "architecture": "Arch.",
    "training_objective": "Train.\nObj.",
    "dataset": "Dataset",
    "all_models": "All\nModels",
}

MODEL_SET_COLORS = {
    "sota": "#2ECC71",
    "architecture": "#E74C3C",
    "training_objective": "#3498DB",
    "dataset": "#F39C12",
    "all_models": "#9B59B6",
}

NC_BASE = 0.46  # Target noise ceiling (at noise_mult = 1.0)


def load_discriminability(model_set: str) -> pd.DataFrame:
    """Load discriminability data for a model set."""
    eval_dir = config.get_eval_pipeline_dir(model_set)
    path = eval_dir / "discriminability.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["model_set"] = model_set
    return df


def load_statistics(model_set: str) -> pd.DataFrame:
    """Load statistics for a model set."""
    eval_dir = config.get_eval_pipeline_dir(model_set)
    path = eval_dir / "statistics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["model_set"] = model_set
    return df


def plot_accuracy_curves(ax, track_filter, title, show_legend=True):
    """Plot model recovery accuracy curves (1 - error) for a given track type filter."""
    for ms in MODEL_SETS:
        disc = load_discriminability(ms)
        if disc.empty:
            continue

        subset_df = disc[track_filter(disc)]

        for subset_type, ls in [("selected", "-"), ("random", "--")]:
            sub = subset_df[subset_df["subset_type"] == subset_type]
            avg = sub.groupby("noise_ceiling")["error_prob"].mean().reset_index()
            avg = avg.sort_values("noise_ceiling")

            label = MODEL_SET_LABELS[ms].replace("\n", " ") if subset_type == "selected" else None
            alpha = 0.9 if subset_type == "selected" else 0.4
            lw = 1.8 if subset_type == "selected" else 1.2

            ax.plot(
                avg["noise_ceiling"], 1 - avg["error_prob"],
                linestyle=ls, color=MODEL_SET_COLORS[ms],
                linewidth=lw, alpha=alpha, label=label,
            )

    # Target NC line
    ax.axvline(x=NC_BASE, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    # Dummy entries for line style legend
    ax.plot([], [], "k-", linewidth=1.5, label="Selected")
    ax.plot([], [], "k--", linewidth=1.0, alpha=0.5, label="Random")

    ax.set_xlabel("Equivalent noise ceiling")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, 1.0)
    ax.invert_xaxis()
    ax.set_title(title)
    if show_legend:
        ax.legend(loc="lower right", ncol=2, frameon=True, framealpha=0.9,
                  columnspacing=0.8, handletextpad=0.4, fontsize=6)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, 4.8))

    # --- Panel A: Raw feature space ---
    plot_accuracy_curves(
        axes[0],
        track_filter=lambda df: df["track_type"] == "identity",
        title="Raw feature space",
        show_legend=True,
    )
    axes[0].set_ylabel("Model recovery accuracy")
    axes[0].text(-0.12, 1.08, "a", transform=axes[0].transAxes,
                 fontsize=FONT["panel_label"], fontweight="bold", va="top")

    # --- Panel B: Encoding (predicted brain) space ---
    plot_accuracy_curves(
        axes[1],
        track_filter=lambda df: df["track_type"] == "encoding",
        title="Encoding (predicted brain) space",
        show_legend=False,
    )
    axes[1].text(-0.12, 1.08, "b", transform=axes[1].transAxes,
                 fontsize=FONT["panel_label"], fontweight="bold", va="top")

    # --- Panel C: AUC bar chart (raw vs encoding) ---
    ax = axes[2]

    ms_labels = []
    raw_rand = []
    raw_sel = []
    enc_rand = []
    enc_sel = []

    for ms in MODEL_SETS:
        stats = load_statistics(ms)
        if stats.empty:
            continue

        raw_row = stats[stats["track"] == "raw"]
        agg_row = stats[stats["track"] == "AGGREGATE"]
        if raw_row.empty or agg_row.empty:
            continue

        ms_labels.append(MODEL_SET_LABELS[ms])
        raw_rand.append(1 - raw_row.iloc[0]["random_auc"])
        raw_sel.append(1 - raw_row.iloc[0]["selected_auc"])
        enc_rand.append(1 - agg_row.iloc[0]["random_auc"])
        enc_sel.append(1 - agg_row.iloc[0]["selected_auc"])

    x = np.arange(len(ms_labels))
    width = 0.19

    # Four bars: raw random, raw selected, encoding random, encoding selected
    ax.bar(
        x - 1.5 * width, raw_rand, width,
        color="#DDDDDD", alpha=0.85, edgecolor="white", linewidth=0.5,
        hatch="//",
        label="Random (raw)",
    )
    ax.bar(
        x - 0.5 * width, raw_sel, width,
        color=[MODEL_SET_COLORS[ms] for ms in MODEL_SETS[:len(ms_labels)]],
        alpha=0.5, edgecolor="white", linewidth=0.5,
        hatch="//",
        label="Selected (raw)",
    )
    ax.bar(
        x + 0.5 * width, enc_rand, width,
        color="#AAAAAA", alpha=0.85, edgecolor="white", linewidth=0.5,
        label="Random (enc.)",
    )
    ax.bar(
        x + 1.5 * width, enc_sel, width,
        color=[MODEL_SET_COLORS[ms] for ms in MODEL_SETS[:len(ms_labels)]],
        alpha=0.85, edgecolor="white", linewidth=0.5,
        label="Selected (enc.)",
    )

    # Add improvement % annotations (selected accuracy is higher = better)
    for i in range(len(ms_labels)):
        ms = MODEL_SETS[i]
        # Raw improvement: how much higher selected accuracy is vs random
        raw_pct = (raw_sel[i] - raw_rand[i]) / (1 - raw_rand[i]) * 100 if raw_rand[i] < 1 else 0
        raw_mid = x[i] - width
        ax.text(raw_mid, max(raw_rand[i], raw_sel[i]) + 0.015,
                f"↑{raw_pct:.0f}%", fontsize=FONT["small"], fontweight="bold",
                ha="center", va="bottom", color=MODEL_SET_COLORS[ms], alpha=0.7)
        # Encoding improvement
        enc_pct = (enc_sel[i] - enc_rand[i]) / (1 - enc_rand[i]) * 100 if enc_rand[i] < 1 else 0
        enc_mid = x[i] + width
        ax.text(enc_mid, max(enc_rand[i], enc_sel[i]) + 0.015,
                f"↑{enc_pct:.0f}%", fontsize=FONT["small"], fontweight="bold",
                ha="center", va="bottom", color=MODEL_SET_COLORS[ms])

    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=FONT["small"],
              ncol=2, columnspacing=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(ms_labels, fontsize=FONT["tick"])
    ax.set_ylabel("Accuracy AUC (higher = better)")
    ax.set_ylim(0.35, 1.02)
    ax.set_title("AUC comparison")
    ax.text(-0.12, 1.08, "c", transform=ax.transAxes,
            fontsize=FONT["panel_label"], fontweight="bold", va="top")

    plt.tight_layout(w_pad=2.5)

    for ext in ["pdf", "png"]:
        out = FIGURES_DIR / f"insilico_evaluation.{ext}"
        fig.savefig(out)
        print(f"Saved {out}")

    plt.close()


if __name__ == "__main__":
    main()
