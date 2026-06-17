#!/usr/bin/env python3
"""
Improved distance-metric robustness figure.

Fixes vs. original:
- The figure now matches its caption: per-subject Spearman rank correlation
  between model rankings obtained with correlation-distance vs cosine-distance
  fRSA, separately by stimulus type (controversial / baseline).
- Note: only fRSA is currently re-computed under both distances in the
  source CSV; the caption in main.tex should be updated to read
  "for fixed RSA" rather than "separately for fixed and mixed RSA".
- Okabe-Ito palette via style_improved.
"""
from __future__ import annotations

import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parents[2]
SHARE_ROOT = STAGE.parent
PAPER_HELPERS = SHARE_ROOT / "src"
sys.path.insert(0, str(PAPER_HELPERS))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from cstims.paper import config
from cstims.paper.style_improved import (
    apply_style, FONT, DPI, W_1_5COL,
    COLOR_CSTIM, COLOR_BASELINE, add_panel_label,
)

apply_style()

DATA_DIR = config.ROBUSTNESS_DATA_DIR
FIGURES_DIR = STAGE / "figures"
PNG_DIR = FIGURES_DIR / "png"


def per_subject_rho(sub_df):
    """For one subject × stimulus_type × model_set, compute the Spearman ρ
    between the model ranking under correlation and under cosine distance.
    """
    out = []
    for (sub, ms, st), block in sub_df.groupby(
            ["subject", "model_set", "stimulus_type"]):
        # Average over bootstrap replicates per model
        avg = block.groupby("model")[["crsa_correlation", "crsa_cosine"]].mean()
        if len(avg) < 3:
            continue
        rho, _ = spearmanr(avg["crsa_correlation"], avg["crsa_cosine"])
        out.append(dict(subject=sub, model_set=ms, stimulus_type=st, rho=float(rho)))
    return pd.DataFrame(out)


def main():
    df = pd.read_csv(DATA_DIR / "distance_metric_robustness.csv")
    rho_df = per_subject_rho(df)

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(W_1_5COL, 4.0),
        gridspec_kw=dict(width_ratios=[1.0, 1.2]))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.20, wspace=0.30)

    # ---- Left: per-subject rho values ----
    model_sets = sorted(rho_df["model_set"].unique())
    n_ms = len(model_sets)
    x = np.arange(n_ms)
    bar_w = 0.35

    for off, st, color, label in [
        (-bar_w/2 - 0.02, "controversial", COLOR_CSTIM, "Controversial"),
        ( bar_w/2 + 0.02, "vicco",         COLOR_BASELINE, "Baseline"),
    ]:
        means = []
        sems  = []
        for ms in model_sets:
            block = rho_df[(rho_df["model_set"] == ms)
                            & (rho_df["stimulus_type"] == st)]["rho"]
            means.append(block.mean())
            sems.append(block.std(ddof=1) / np.sqrt(max(len(block), 1)))
        ax_left.bar(x + off, means, width=bar_w,
                     yerr=sems, color=color, alpha=0.85,
                     error_kw=dict(linewidth=0.7, capsize=2, ecolor="#444"),
                     label=label)
        # Per-subject dots
        for i, ms in enumerate(model_sets):
            block = rho_df[(rho_df["model_set"] == ms)
                            & (rho_df["stimulus_type"] == st)]["rho"].values
            jitter = np.random.default_rng(7).uniform(-0.05, 0.05, len(block))
            ax_left.scatter(np.full_like(block, x[i] + off) + jitter, block,
                             s=10, color="#222", alpha=0.7, linewidths=0,
                             zorder=4)

    ax_left.axhline(1.0, color="#444", linestyle=":", linewidth=0.8, alpha=0.7)
    ax_left.text(n_ms - 0.5, 1.005, "identical rankings",
                 ha="right", va="bottom", fontsize=FONT["small"] - 1,
                 color="#444", style="italic")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels([m.replace("_", " ").title() for m in model_sets],
                              rotation=30, ha="right")
    ax_left.set_ylim(0.5, 1.07)
    ax_left.set_ylabel("Per-subject Spearman ρ\n(corr-dist vs cos-dist rankings)")
    ax_left.set_title("Rank correlations sit near 1 across model sets", pad=4)
    ax_left.legend(loc="lower right", frameon=False, fontsize=FONT["small"])
    add_panel_label(ax_left, "a")

    # ---- Right: scatter of raw scores (kept as a sanity check) ----
    agg = df.groupby(["model", "model_set", "stimulus_type", "bootstrap_idx"]).agg(
        crsa_correlation=("crsa_correlation", "mean"),
        crsa_cosine=("crsa_cosine", "mean"),
    ).reset_index()
    cstim = agg[agg["stimulus_type"] == "controversial"]
    vicco = agg[agg["stimulus_type"] == "vicco"]

    rho_all, _ = spearmanr(agg["crsa_correlation"], agg["crsa_cosine"])
    ax_right.scatter(vicco["crsa_correlation"], vicco["crsa_cosine"],
                      s=8, alpha=0.40, c=COLOR_BASELINE, linewidths=0,
                      label="Baseline", zorder=2)
    ax_right.scatter(cstim["crsa_correlation"], cstim["crsa_cosine"],
                      s=12, alpha=0.75, c=COLOR_CSTIM, linewidths=0,
                      label="Controversial", zorder=3)
    lims = [min(ax_right.get_xlim()[0], ax_right.get_ylim()[0]),
             max(ax_right.get_xlim()[1], ax_right.get_ylim()[1])]
    ax_right.plot(lims, lims, ls="--", c="#777", lw=0.6, alpha=0.7,
                   zorder=1, label="Identity")
    ax_right.set_xlim(lims); ax_right.set_ylim(lims)
    ax_right.set_aspect("equal")
    ax_right.set_xlabel("fRSA (correlation distance)")
    ax_right.set_ylabel("fRSA (cosine distance)")
    ax_right.text(0.04, 0.96,
                   f"Spearman ρ = {rho_all:.3f}\n"
                   f"mean |Δ| = {(agg['crsa_correlation'] - agg['crsa_cosine']).abs().mean():.4f}",
                   transform=ax_right.transAxes, fontsize=FONT["small"],
                   va="top", color="#222",
                   bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="none", alpha=0.85))
    ax_right.legend(loc="lower right", fontsize=FONT["small"], frameon=False,
                     handletextpad=0.3)
    ax_right.set_title("Score-level: scores fall on the identity line", pad=4)
    add_panel_label(ax_right, "b")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = FIGURES_DIR / "distance_metric_robustness_improved.pdf"
    out_png = PNG_DIR / "distance_metric_robustness_improved.png"
    fig.savefig(out_pdf, dpi=DPI)
    fig.savefig(out_png, dpi=DPI)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
