#!/usr/bin/env python3
"""Plot selected-vs-random values of the selection objective."""

from __future__ import annotations

import sys
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

PAPER_ROOT = Path(__file__).resolve().parents[2]
SHARE_ROOT = PAPER_ROOT.parents[1]
HELPERS = SHARE_ROOT / "shared" / "code" / "paper_helpers"
sys.path.insert(0, str(HELPERS))
sys.path.insert(0, str(HELPERS / "figures"))

import config
from style_improved import (
    apply_style,
    DPI,
    FONT,
    W_1_5COL,
    MODEL_SET_ORDER,
    MODEL_SET_DISPLAY_SHORT,
    model_set_color,
    add_panel_label,
)

apply_style()

FIGURES_DIR = PAPER_ROOT / "figures" / "insilico_curve"
PNG_DIR = FIGURES_DIR / "png"
DATA_PATH = config.EVAL_DATA_DIR / "selection_objective_combined.csv"
TRACK = "combined_raw_plus_encoding"


def _load_data() -> list[dict]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run analysis/11_compute_selection_objective.py first."
        )
    with DATA_PATH.open(newline="") as f:
        return [row for row in csv.DictReader(f) if row["track"] == TRACK]


def _plot_panel(ax, rows: list[dict], noise_condition: str, title: str) -> None:
    sub = [row for row in rows if row["noise_condition"] == noise_condition]
    x_positions = np.arange(len(MODEL_SET_ORDER))
    width = 0.34

    for x, model_set in zip(x_positions, MODEL_SET_ORDER):
        vals = {
            row["subset_type"]: row
            for row in sub
            if row["model_set"] == model_set
        }
        if not {"random", "selected"} <= set(vals):
            continue

        random_val = float(vals["random"]["objective_min"])
        selected_val = float(vals["selected"]["objective_min"])
        color = model_set_color(model_set)

        random_err = _asymmetric_yerr(vals["random"])
        selected_err = _asymmetric_yerr(vals["selected"])
        ax.bar(
            x - width / 2,
            random_val,
            width=width,
            color="#D9D9D9",
            edgecolor="#666666",
            linewidth=0.7,
            yerr=random_err,
            capsize=2.5,
            error_kw=dict(ecolor="#444444", lw=0.8),
            zorder=3,
        )
        ax.bar(
            x + width / 2,
            selected_val,
            width=width,
            color=color,
            edgecolor=color,
            linewidth=0.7,
            yerr=selected_err,
            capsize=2.5,
            error_kw=dict(ecolor=shade_for_error(color), lw=0.8),
            zorder=3,
        )

    ax.axhline(0, color="#999999", lw=0.7, ls=":", zorder=0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [MODEL_SET_DISPLAY_SHORT[m] for m in MODEL_SET_ORDER],
        rotation=25,
        ha="right",
    )
    ax.grid(axis="y", alpha=0.35)
    ax.set_title(title, pad=4)
    ax.set_ylabel("Selection objective U")
    ax.set_ylim(0, 0.92)


def _asymmetric_yerr(row: dict) -> np.ndarray | None:
    if not row.get("objective_ci95_low") or not row.get("objective_ci95_high"):
        return None
    value = float(row["objective_min"])
    low = float(row["objective_ci95_low"])
    high = float(row["objective_ci95_high"])
    return np.array([[max(0.0, value - low)], [max(0.0, high - value)]])


def shade_for_error(color: str) -> str:
    return "#333333" if color.lower() == "#f0e442" else color


def main() -> None:
    df = _load_data()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(W_1_5COL, 3.2),
        sharey=True,
        constrained_layout=False,
    )

    _plot_panel(axes[0], df, "clean", "Clean RDM geometry")
    _plot_panel(axes[1], df, "noised", "Noise-aware objective")

    add_panel_label(axes[0], "a", x=-0.10, y=1.04)
    add_panel_label(axes[1], "b", x=-0.07, y=1.04)

    handles = [
        Patch(facecolor="#D9D9D9", edgecolor="#666666", label="Random"),
        Patch(facecolor="#444444", edgecolor="#444444", label="Selected"),
    ]
    axes[1].legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        edgecolor="none",
        fontsize=FONT["small"],
    )

    fig.suptitle(
        "Selected sets score higher under the selection objective",
        y=0.98,
        fontsize=FONT["title"],
    )
    fig.text(
        0.5,
        0.04,
        "Error bars: 95% model-bootstrap interval for the worst-case objective",
        ha="center",
        fontsize=FONT["small"],
        color="#444444",
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.82, bottom=0.28, wspace=0.24)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    for out in (
        FIGURES_DIR / "selection_objective_comparison.pdf",
        PNG_DIR / "selection_objective_comparison.png",
    ):
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
