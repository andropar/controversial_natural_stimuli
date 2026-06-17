#!/usr/bin/env python3
"""Residualized mixed-RSA alignment-drop summary after PPCA-OOD controls."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

STAGE = Path(__file__).resolve().parents[2]
SHARE_ROOT = STAGE.parents[2]
sys.path.insert(0, str(SHARE_ROOT / "src"))

from cstims.paper.style_improved import apply_style, FONT, DPI, W_SINGLE, OKABE_ITO, add_panel_label  # noqa: E402

DATA = STAGE / "results" / "ood_residualization_results.csv"
FIG = STAGE / "figures" / "supplementary"

LABELS = {
    "delta_alignment ~ delta_ood_feature + delta_ood_pred": "PPCA OOD axes",
    "delta_alignment ~ delta_ood_feature + delta_ood_pred + model_set_fixed_effects": "PPCA OOD + set effects",
}


def main() -> None:
    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    df["label"] = df["model_spec"].map(LABELS).fillna(df["model_spec"])

    fig, ax = plt.subplots(figsize=(W_SINGLE, 2.9))
    fig.subplots_adjust(left=0.38, right=0.97, top=0.87, bottom=0.22)
    y = range(len(df))
    ax.axvline(0, color="#333333", lw=0.8, zorder=0)
    ax.axvline(float(df["mean_observed_delta"].iloc[0]), color="#777777", lw=0.8,
               ls="--", zorder=0, label="observed mean delta")
    ax.errorbar(
        df["intercept"],
        list(y),
        xerr=[
            df["intercept"] - df["intercept_ci95_lo_boot_rows"],
            df["intercept_ci95_hi_boot_rows"] - df["intercept"],
        ],
        fmt="o",
        color=OKABE_ITO["blue"],
        ecolor=OKABE_ITO["blue"],
        elinewidth=1.0,
        capsize=3,
        zorder=3,
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["label"])
    ax.invert_yaxis()
    ax.set_xlabel("residualized delta alignment")
    ax.set_title("Residualized alignment drop", fontweight="bold", pad=6)
    ax.grid(axis="x", alpha=0.22)
    add_panel_label(ax, "a", x=-0.55, y=1.04)
    ax.legend(frameon=False, loc="lower right", fontsize=FONT["legend"])

    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"residualized_alignment.{ext}", dpi=DPI if ext == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
