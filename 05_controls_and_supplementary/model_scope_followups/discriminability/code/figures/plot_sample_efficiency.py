#!/usr/bin/env python3
"""Sample-efficiency curves: # separated pairs and rank stability vs n,
comparing cstim_all_models to vicco baseline, mRSA + fRSA.
"""

import sys
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[4]
PAPER = PROJECT / "experiments" / "cstim_paper"
sys.path.insert(0, str(PAPER))
from cstims.paper.style_improved import apply_style, FONT, DPI  # noqa

apply_style()

DATA_DIR = Path(__file__).resolve().parents[1] / "results"
FIG_DIR = Path(__file__).resolve().parent

COLOR_CSTIM = "#D64541"
COLOR_VICCO = "#2980B9"


def main():
    df = pd.read_csv(DATA_DIR / "sample_efficiency_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5), sharex=True)
    metrics = ["mRSA", "fRSA"]

    for col, metric in enumerate(metrics):
        d = df[df["metric"] == metric]
        cstim = d[d["set"] == "all_models"].sort_values("n")
        vicco = d[d["set"] == "vicco"].sort_values("n")

        # Top: # separated pairs
        ax = axes[0, col]
        ax.errorbar(cstim["n"], cstim["n_separated_mean"],
                    yerr=cstim["n_separated_sem"],
                    fmt="-o", color=COLOR_CSTIM, lw=1.5, ms=5, capsize=3,
                    label="cstim (all_models)")
        ax.errorbar(vicco["n"], vicco["n_separated_mean"],
                    yerr=vicco["n_separated_sem"],
                    fmt="-o", color=COLOR_VICCO, lw=1.5, ms=5, capsize=3,
                    label="vicco baseline")
        ax.axhline(190, color="black", lw=0.5, ls=":", alpha=0.6)
        ax.text(105, 190, "max=190", fontsize=FONT["annotation"],
                ha="left", va="center", color="#666")
        ax.set_ylabel("# pairs separated\n(out of 190, FDR q<0.05)",
                      fontsize=FONT["axis_label"])
        ax.set_title(metric, fontsize=FONT["title"])
        ax.legend(loc="upper left", fontsize=FONT["legend"], frameon=False)
        ax.set_ylim(0, 200)

        # Bottom: rank stability
        ax = axes[1, col]
        ax.errorbar(cstim["n"], cstim["rank_rho_mean"],
                    yerr=cstim["rank_rho_sem"],
                    fmt="-o", color=COLOR_CSTIM, lw=1.5, ms=5, capsize=3)
        ax.errorbar(vicco["n"], vicco["rank_rho_mean"],
                    yerr=vicco["rank_rho_sem"],
                    fmt="-o", color=COLOR_VICCO, lw=1.5, ms=5, capsize=3)
        ax.axhline(1.0, color="black", lw=0.5, ls=":", alpha=0.6)
        ax.set_ylabel("Rank stability\n(Spearman ρ vs full-100 ranking)",
                      fontsize=FONT["axis_label"])
        ax.set_xlabel("n images", fontsize=FONT["axis_label"])
        ax.set_ylim(0, 1.05)

    fig.suptitle("Sample efficiency: how many model pairs become separable as n increases?",
                 fontsize=FONT["title"] + 1, y=1.00)
    fig.tight_layout()
    out_pdf = FIG_DIR / "sample_efficiency.pdf"
    out_png = FIG_DIR / "sample_efficiency.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=DPI)
    print(f"Wrote {out_pdf}\nWrote {out_png}")


if __name__ == "__main__":
    main()
