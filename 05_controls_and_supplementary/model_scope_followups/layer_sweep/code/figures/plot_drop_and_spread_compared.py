#!/usr/bin/env python3
"""Two-panel summary: (1) cstim alignment drop and (2) between-model spread
ratio at three layer-choice levels — paper layer (existing), best-on-cstim
layer (post-hoc, biased), and held-out best layer (cross-validated).

Shows that:
    1. The drop is real at the paper layer.
    2. The post-hoc rescue is partly selection bias.
    3. Spread on `all_models` survives best-layer choice; controlled sets
       partially collapse.
"""

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import PAPER_ROOT
from style import apply_style, FONT
from layers_config import MAIN_LAYER

apply_style()

DATA_DIR = LAYER_SWEEP_ROOT / "data"
FIG_DIR = LAYER_SWEEP_ROOT / "figures"
CSTIM_SETS = ["all_models", "architecture", "dataset", "sota", "training_objective"]


def main():
    # Drops at paper layer (mRSA), and rescue from biased / held-out
    bl_mr = pd.read_csv(DATA_DIR / "best_layer_wrsa_scores.csv")
    ho = pd.read_csv(DATA_DIR / "held_out_rescue.csv")
    spread = pd.read_csv(DATA_DIR / "spread_summary.csv")
    drop_summary = pd.read_csv(DATA_DIR / "mrsa_layer_rescue_summary_subject_avg.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    x = np.arange(len(CSTIM_SETS))

    # --- Panel A: mRSA drop, three layer-choice levels ---
    ax = axes[0]
    width = 0.27
    drop_paper, drop_biased, drop_holdout = [], [], []
    for s in CSTIM_SETS:
        row_drop = drop_summary[drop_summary["model_set"] == s]
        # Average drop_paper and drop_best across models in this set
        drop_paper.append(row_drop["delta_paper_mean"].mean())
        drop_biased.append(row_drop["delta_best_mean"].mean())
        # held-out: paper-mean already includes the drop relative to vicco, but ho
        # only stored cstim_RSA at paper/best (not delta). Reconstruct from
        # held_out_rescue: drop_held_out = drop_paper + held_out_rescue_mean
        ho_sub = ho[ho["model_set"] == s]
        ho_rescue = ho_sub["mrsa_rescue_mean"].mean()
        drop_holdout.append(row_drop["delta_paper_mean"].mean() + ho_rescue)

    ax.bar(x - width, drop_paper, width=width, color="#D64541",
           label="Paper layer", edgecolor="black", linewidth=0.4)
    ax.bar(x, drop_holdout, width=width, color="#F39C12",
           label="Held-out best layer (unbiased)", edgecolor="black", linewidth=0.4)
    ax.bar(x + width, drop_biased, width=width, color="#27AE60",
           label="Best-on-cstim layer (biased)", edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(CSTIM_SETS, rotation=20, ha="right", fontsize=FONT["tick"])
    ax.set_ylabel("mRSA cstim − vicco baseline\n(subject-averaged Δ across 20 models)",
                  fontsize=FONT["axis_label"])
    ax.set_title("Cstim alignment drop, three layer-choice levels",
                 fontsize=FONT["title"])
    ax.legend(loc="lower left", fontsize=FONT["legend"], frameon=False)

    # --- Panel B: spread ratio ---
    ax = axes[1]
    sp_mrsa = spread[spread["metric"] == "mRSA"].set_index("model_set")
    paper_ratio, ho_ratio = [], []
    v_paper = sp_mrsa.loc["vicco", "spread_paper"]
    for s in CSTIM_SETS:
        paper_ratio.append(sp_mrsa.loc[s, "spread_paper"] / v_paper)
        ho_ratio.append(sp_mrsa.loc[s, "spread_held_out"] / v_paper)

    width2 = 0.35
    ax.bar(x - width2 / 2, paper_ratio, width=width2, color="#D64541",
           label="Paper layer", edgecolor="black", linewidth=0.4)
    ax.bar(x + width2 / 2, ho_ratio, width=width2, color="#F39C12",
           label="Held-out best layer", edgecolor="black", linewidth=0.4)
    ax.axhline(1.0, color="black", lw=0.6, ls="--", label="Vicco spread")
    ax.set_xticks(x)
    ax.set_xticklabels(CSTIM_SETS, rotation=20, ha="right", fontsize=FONT["tick"])
    ax.set_ylabel("Spread ratio (cstim / vicco)\n(median pairwise diff of model means)",
                  fontsize=FONT["axis_label"])
    ax.set_title("Between-model spread, mRSA",
                 fontsize=FONT["title"])
    ax.legend(loc="upper left", fontsize=FONT["legend"], frameon=False)

    fig.tight_layout()
    out_pdf = FIG_DIR / "drop_and_spread_compared.pdf"
    out_png = FIG_DIR / "drop_and_spread_compared.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Wrote {out_pdf}\nWrote {out_png}")


if __name__ == "__main__":
    main()
