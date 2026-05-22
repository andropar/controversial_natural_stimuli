#!/usr/bin/env python3
"""Broad-benchmark cstim-minus-baseline deltas by model family."""

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

DATA = _PAPER / "03_statistics" / "results" / "posthoc_model_family_summary.csv"
FIG = _PAPER / "02_rsa_scores" / "figures"

FAMILY_ORDER = ["supervised_classification", "image_text", "self_supervised"]
FAMILY_LABEL = {
    "supervised_classification": "Supervised",
    "image_text": "Image-text",
    "self_supervised": "Self-supervised",
}
FAMILY_COLOR = {
    "supervised_classification": OKABE_ITO["blue"],
    "image_text": OKABE_ITO["orange"],
    "self_supervised": OKABE_ITO["bluish_green"],
}
FAMILY_MARKER = {
    "supervised_classification": "o",
    "image_text": "s",
    "self_supervised": "^",
}
METRICS = [("mixed_RSA", "mixed RSA"), ("fixed_RSA", "fixed RSA")]


def main() -> None:
    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    df = df[df["family"].isin(FAMILY_ORDER)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.9), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.25, wspace=0.07)
    rng = np.random.default_rng(12)
    offsets = np.linspace(-0.22, 0.22, len(FAMILY_ORDER))

    for ax, (metric, title), panel in zip(axes, METRICS, ["a", "b"]):
        sub = df[df["metric"] == metric]
        xs = np.arange(len(MODEL_SET_ORDER))
        for off, family in zip(offsets, FAMILY_ORDER):
            fam = sub[sub["family"] == family]
            means, sems = [], []
            for model_set in MODEL_SET_ORDER:
                vals = fam[fam["model_set"] == model_set]["delta"].dropna()
                means.append(float(vals.mean()) if len(vals) else np.nan)
                sems.append(float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan)
                if len(vals):
                    jitter = rng.uniform(-0.025, 0.025, len(vals))
                    ax.scatter(
                        np.full(len(vals), xs[MODEL_SET_ORDER.index(model_set)] + off) + jitter,
                        vals,
                        s=13,
                        color=FAMILY_COLOR[family],
                        marker=FAMILY_MARKER[family],
                        alpha=0.38,
                        linewidths=0,
                        zorder=2,
                    )
            ax.errorbar(
                xs + off,
                means,
                yerr=1.96 * np.asarray(sems, dtype=float),
                fmt=FAMILY_MARKER[family],
                ms=4,
                color=FAMILY_COLOR[family],
                elinewidth=0.9,
                capsize=2,
                label=FAMILY_LABEL[family],
                zorder=4,
            )

        ax.axhline(0, color="#333333", lw=0.8, zorder=0)
        ax.set_xticks(xs)
        ax.set_xticklabels([MODEL_SET_DISPLAY_SHORT[m] for m in MODEL_SET_ORDER])
        ax.set_title(title, fontweight="bold", pad=6)
        ax.grid(axis="y", alpha=0.22)
        add_panel_label(ax, panel, x=-0.07, y=1.04)

    axes[0].set_ylabel("delta alignment\n(controversial - baseline)")
    axes[0].set_ylim(-0.32, 0.08)
    handles = [
        Line2D([0], [0], marker=FAMILY_MARKER[f], color=FAMILY_COLOR[f], markersize=5,
               linestyle="-", label=FAMILY_LABEL[f])
        for f in FAMILY_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=3, frameon=False, fontsize=FONT["legend"])

    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"benchmark_family_deltas.{ext}", dpi=DPI if ext == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
