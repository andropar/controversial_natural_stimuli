#!/usr/bin/env python3
"""Plot mixed-RSA robustness to correlation vs cosine RDM distance."""

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

from cstims.paper import config  # noqa: E402
from cstims.paper.style_improved import DPI, apply_style  # noqa: E402


MODEL_SETS = ["all_models", "sota", "training_objective", "architecture", "dataset"]
SET_LABELS = ["All", "SOTA", "Train", "Arch", "Data"]
DATA_DIR = config.ROBUSTNESS_DATA_DIR
FIGURES_DIR = STAGE / "figures"
PNG_DIR = FIGURES_DIR / "png"


def main() -> None:
    apply_style()
    df = pd.read_csv(DATA_DIR / "mixed_distance_metric_robustness.csv")
    rank_df = pd.read_csv(DATA_DIR / "mixed_distance_metric_rank_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    ax = axes[0]
    ax.scatter(df["mixed_rsa_correlation"], df["mixed_rsa_cosine"], s=10, alpha=0.35)
    lo = float(np.nanmin(df[["mixed_rsa_correlation", "mixed_rsa_cosine"]].values))
    hi = float(np.nanmax(df[["mixed_rsa_correlation", "mixed_rsa_cosine"]].values))
    ax.plot([lo, hi], [lo, hi], color="0.25", lw=0.8, ls="--")
    rho, _ = spearmanr(df["mixed_rsa_correlation"], df["mixed_rsa_cosine"])
    mad = (df["mixed_rsa_correlation"] - df["mixed_rsa_cosine"]).abs().mean()
    ax.set_title(f"score agreement\nrho={rho:.3f}, mean |delta|={mad:.3f}")
    ax.set_xlabel("mixed RSA, correlation distance")
    ax.set_ylabel("mixed RSA, cosine distance")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    vals = [
        rank_df[(rank_df["model_set"] == ms) & (rank_df["stimulus_type"] == "controversial")][
            "rank_rho"
        ].mean()
        for ms in MODEL_SETS
    ]
    sems = [
        rank_df[(rank_df["model_set"] == ms) & (rank_df["stimulus_type"] == "controversial")][
            "rank_rho"
        ].sem()
        for ms in MODEL_SETS
    ]
    ax.bar(np.arange(len(MODEL_SETS)), vals, yerr=sems, color="#0072B2", alpha=0.85, capsize=2)
    ax.set_xticks(np.arange(len(MODEL_SETS)))
    ax.set_xticklabels(SET_LABELS, rotation=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rank rho")
    ax.set_title("per-set model-rank agreement")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = FIGURES_DIR / "mixed_distance_metric_robustness.pdf"
    out_png = PNG_DIR / "mixed_distance_metric_robustness.png"
    fig.savefig(out_pdf, dpi=DPI)
    fig.savefig(out_png, dpi=DPI)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
