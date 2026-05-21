#!/usr/bin/env python3
"""Descriptive scatter: set-level rescue vs set-level low-level Mahalanobis.

5 cstim sets, one point each. Annotated with set name.
"""

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from config import PAPER_ROOT
from style import apply_style, FONT

apply_style()

DATA_DIR = LAYER_SWEEP_ROOT / "data"
FIG_DIR = LAYER_SWEEP_ROOT / "figures"


def main():
    df = pd.read_csv(DATA_DIR / "rescue_vs_low_level.csv")
    if df["mean_mahal"].isna().any():
        print("[skip] low-level shift not joined; figure not produced")
        return

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.errorbar(df["mean_mahal"], df["rescue_mean"], yerr=df["rescue_sem"],
                fmt="o", color="tab:blue", ms=5, capsize=2, lw=0.6)
    for _, row in df.iterrows():
        ax.annotate(row["model_set"], (row["mean_mahal"], row["rescue_mean"]),
                    fontsize=FONT["annotation"], xytext=(4, 4), textcoords="offset points")

    rho, p = spearmanr(df["mean_mahal"], df["rescue_mean"])
    ax.set_title(f"Set-level rescue vs low-level shift  (Spearman ρ = {rho:.2f})",
                 fontsize=FONT["title"])
    ax.set_xlabel("Mean Mahalanobis distance from baseline\n(low-level image stats)",
                  fontsize=FONT["axis_label"])
    ax.set_ylabel("Rescue (Δ_best − Δ_paper-layer)", fontsize=FONT["axis_label"])
    ax.axhline(0, color="black", lw=0.4, alpha=0.5)
    fig.tight_layout()
    out_pdf = FIG_DIR / "rescue_vs_low_level.pdf"
    out_png = FIG_DIR / "rescue_vs_low_level.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Wrote {out_pdf}\nWrote {out_png}")


if __name__ == "__main__":
    main()
