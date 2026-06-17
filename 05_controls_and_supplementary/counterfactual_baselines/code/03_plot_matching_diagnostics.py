#!/usr/bin/env python3
"""Pre/post covariate balance diagnostics for held-out unique matching."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

STAGE = Path(__file__).resolve().parents[1]
SHARE_ROOT = STAGE.parents[2]
sys.path.insert(0, str(SHARE_ROOT / "src"))

from cstims.paper.style_improved import (  # noqa: E402
    apply_style,
    FONT,
    DPI,
    W_DOUBLE,
    OKABE_ITO,
    MODEL_SET_ORDER,
    MODEL_SET_DISPLAY_SHORT,
    add_panel_label,
)

DATA = STAGE / "results" / "baseline_matching_diagnostics.csv"
FIG = STAGE / "figures" / "supplementary"
PNG_DIR = FIG / "png"

MATCH_ORDER = ["low_level", "embedding_pc", "ppca_ood", "combined"]
MATCH_LABEL = {
    "low_level": "Low-level",
    "embedding_pc": "Embedding PCs",
    "ppca_ood": "Feature PPCA OOD",
    "combined": "Combined",
}


def main() -> None:
    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    df = df[
        (df["status"] == "ok")
        & df["match_type"].isin(MATCH_ORDER)
        & df["pre_match_centroid_distance"].notna()
        & df["post_match_centroid_distance"].notna()
    ].copy()

    summary = (
        df.groupby(["match_type", "model_set"], as_index=False)
        .agg(
            pre=("pre_match_centroid_distance", "mean"),
            post=("post_match_centroid_distance", "mean"),
            n=("post_match_centroid_distance", "size"),
        )
    )
    summary["reduction"] = 100 * (summary["pre"] - summary["post"]) / summary["pre"]

    fig, axes = plt.subplots(1, 4, figsize=(W_DOUBLE, 3.8), sharey=True)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.82, bottom=0.24, wspace=0.10)
    y = np.arange(len(MODEL_SET_ORDER))
    colors = {"pre": "#888888", "post": OKABE_ITO["blue"]}

    for ax, match, panel in zip(axes, MATCH_ORDER, list("abcd")):
        sub = summary[summary["match_type"] == match].set_index("model_set")
        xmax = 0.0
        for yi, model_set in enumerate(MODEL_SET_ORDER):
            if model_set not in sub.index:
                continue
            row = sub.loc[model_set]
            ax.plot([row["pre"], row["post"]], [yi, yi], color="#999999", lw=1.0, zorder=1)
            ax.scatter(row["pre"], yi, s=24, marker="s", facecolor="white",
                       edgecolor=colors["pre"], linewidth=0.9, zorder=2)
            ax.scatter(row["post"], yi, s=30, marker="o", color=colors["post"], zorder=3)
            ax.text(
                max(row["pre"], row["post"]) * 1.03,
                yi,
                f"{row['reduction']:.0f}%",
                va="center",
                ha="left",
                fontsize=FONT["small"],
                color="#333333",
            )
            xmax = max(xmax, row["pre"], row["post"])
        ax.set_title(MATCH_LABEL[match], fontweight="bold", pad=6)
        ax.set_xlabel("centroid gap", fontsize=FONT["small"])
        ax.set_xlim(0, xmax * 1.35 if xmax > 0 else 1)
        ax.set_yticks(y)
        ax.set_yticklabels([MODEL_SET_DISPLAY_SHORT[m] for m in MODEL_SET_ORDER])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.22)
        add_panel_label(ax, panel, x=-0.10, y=1.05)

    axes[0].set_ylabel("model set", fontsize=FONT["axis_label"])
    fig.text(0.5, 0.925, "gray = before matching; blue = after matching; labels = mean percent reduction",
             ha="center", fontsize=FONT["small"], color="#444444")

    fig.savefig(FIG / "heldout_matching_diagnostics.pdf")
    fig.savefig(PNG_DIR / "heldout_matching_diagnostics.png", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    main()
