#!/usr/bin/env python3
"""Subject-level spread-ratio companion figure for the brain-alignment result."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))
sys.path.insert(0, str(_PAPER / "figures"))

from style_improved import (  # noqa: E402
    apply_style,
    FONT,
    DPI,
    W_DOUBLE,
    OKABE_ITO,
    MODEL_SET_ORDER,
    MODEL_SET_DISPLAY_SHORT,
    add_panel_label,
)

DATA = _PAPER / "03_statistics" / "data" / "primary_endpoint_summary.csv"
FIG = _PAPER / "03_statistics" / "figures"

METHODS = [
    ("mixed_RSA", "mixed RSA", OKABE_ITO["blue"], "o"),
    ("fixed_RSA", "fixed RSA", OKABE_ITO["sky_blue"], "D"),
]


def _mean_sem(values: pd.Series) -> tuple[float, float]:
    arr = values.to_numpy(dtype=float)
    mean = float(np.nanmean(arr))
    sem = float(np.nanstd(arr, ddof=1) / np.sqrt(np.isfinite(arr).sum()))
    return mean, sem


def main() -> None:
    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA)
    df = df[
        (df["baseline_type"] == "same_session_unselected")
        & df["metric"].isin(["mixed_RSA", "fixed_RSA"])
        & df["spread_ratio"].notna()
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.7), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.25, wspace=0.08)

    rng = np.random.default_rng(7)
    for ax, (metric, title, color, marker), panel in zip(axes, METHODS, ["a", "b"]):
        sub = df[df["metric"] == metric].copy()
        xs = np.arange(len(MODEL_SET_ORDER))
        for i, model_set in enumerate(MODEL_SET_ORDER):
            vals = sub[sub["model_set"] == model_set]["spread_ratio"].dropna()
            if vals.empty:
                continue
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(
                np.full(len(vals), xs[i]) + jitter,
                vals,
                s=22,
                facecolors=color if metric == "mixed_RSA" else "white",
                edgecolors=color,
                linewidths=0.8,
                alpha=0.9,
                marker=marker,
                zorder=3,
            )
            mean, sem = _mean_sem(vals)
            ax.errorbar(
                xs[i],
                mean,
                yerr=1.96 * sem,
                fmt="_",
                markersize=18,
                color="#222222",
                elinewidth=1.0,
                capsize=3,
                zorder=4,
            )
            ax.text(
                xs[i],
                mean + 1.96 * sem + 0.16,
                f"{mean:.2f}x",
                ha="center",
                va="bottom",
                fontsize=FONT["small"],
                color="#222222",
            )

        ax.axhline(1.0, color="#555555", lw=0.8, ls="--", zorder=0)
        ax.set_xticks(xs)
        ax.set_xticklabels([MODEL_SET_DISPLAY_SHORT[m] for m in MODEL_SET_ORDER])
        ax.set_title(title, fontweight="bold", pad=6)
        ax.grid(axis="y", alpha=0.22)
        add_panel_label(ax, panel, x=-0.07, y=1.04)

    axes[0].set_ylabel("spread ratio\n(controversial / baseline)")
    axes[0].set_ylim(0, 8.8)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=OKABE_ITO["blue"],
               markeredgecolor=OKABE_ITO["blue"], markersize=5, label="subject"),
        Line2D([0], [0], marker="_", color="#222222", markersize=12,
               label="mean +/- 1.96 SEM"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=2, frameon=False, fontsize=FONT["legend"])

    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"spread_ratio_subjects.{ext}", dpi=DPI if ext == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
