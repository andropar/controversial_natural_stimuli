#!/usr/bin/env python3
"""Downstream claim robustness under correlation vs cosine RDM distance."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

STAGE = Path(__file__).resolve().parents[2]
SHARE_ROOT = STAGE.parent
PAPER_HELPERS = SHARE_ROOT / "src"
sys.path.insert(0, str(PAPER_HELPERS))

from cstims import constants, paths
from cstims.paper.style_improved import (  # noqa: E402
    DPI,
    FONT,
    MODEL_SET_COLORS,
    MODEL_SET_DISPLAY_SHORT,
    MODEL_SET_ORDER,
    W_DOUBLE,
    add_panel_label,
    apply_style,
)

MODEL_SET_MARKERS = {
    "all_models": "o",
    "sota": "s",
    "training_objective": "^",
    "architecture": "D",
    "dataset": "P",
}


DATA_DIR = paths.robustness_data_dir()
FIGURES_DIR = STAGE / "figures"
PNG_DIR = FIGURES_DIR / "png"


def plot_metric(ax, data: pd.DataFrame, method: str, value_col: str, title: str, panel: str) -> None:
    wide = data[data["method"] == method].pivot_table(
        index=["subject", "model_set"],
        columns="distance_metric",
        values=value_col,
    ).reset_index()
    for model_set in MODEL_SET_ORDER:
        sub = wide[wide["model_set"] == model_set]
        ax.scatter(
            sub["correlation"],
            sub["cosine"],
            s=28,
            color=MODEL_SET_COLORS[model_set],
            marker=MODEL_SET_MARKERS[model_set],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.88,
            label=MODEL_SET_DISPLAY_SHORT[model_set] if panel == "a" else None,
            zorder=3,
        )
    vals = np.concatenate([wide["correlation"].to_numpy(), wide["cosine"].to_numpy()])
    vals = vals[np.isfinite(vals)]
    lo, hi = float(vals.min()), float(vals.max())
    pad = 0.10 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#333333", lw=0.8, ls="--", zorder=1)
    ax.axhline(0, color="#BBBBBB", lw=0.6, zorder=0)
    ax.axvline(0, color="#BBBBBB", lw=0.6, zorder=0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    rho = spearmanr(wide["correlation"], wide["cosine"]).statistic
    ax.text(0.04, 0.94, f"$\\rho$ = {rho:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=FONT["tick"])
    ax.set_title(f"{method}: {title}", fontsize=FONT["title"], pad=4)
    ax.set_xlabel("correlation distance", fontsize=FONT["axis_label"])
    ax.set_ylabel("cosine distance", fontsize=FONT["axis_label"])
    ax.tick_params(axis="both", labelsize=FONT["tick"])
    ax.grid(True, color="#E6E6E6", linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_panel_label(ax, panel, x=-0.12, y=1.05)


def draw() -> None:
    apply_style()
    data = pd.read_csv(DATA_DIR / "distance_metric_claim_robustness.csv")
    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE * 0.86, 6.35))
    plot_metric(axes[0, 0], data, "mixed RSA", "mean_delta", "mean alignment delta", "a")
    plot_metric(axes[0, 1], data, "mixed RSA", "log2_spread_ratio", "spread ratio", "b")
    plot_metric(axes[1, 0], data, "fixed RSA", "mean_delta", "mean alignment delta", "c")
    plot_metric(axes[1, 1], data, "fixed RSA", "log2_spread_ratio", "spread ratio", "d")
    axes[0, 0].legend(
        frameon=False,
        loc="lower right",
        fontsize=FONT["legend"] - 1,
        handletextpad=0.3,
        borderaxespad=0.2,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.09, top=0.94, wspace=0.28, hspace=0.48)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = FIGURES_DIR / "distance_metric_claim_robustness.pdf"
    out_png = PNG_DIR / "distance_metric_claim_robustness.png"
    fig.savefig(out_pdf, dpi=DPI)
    fig.savefig(out_png, dpi=DPI)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    plt.close(fig)
    print(pd.read_csv(DATA_DIR / "distance_metric_claim_robustness_summary.csv").to_string(index=False))


if __name__ == "__main__":
    draw()
