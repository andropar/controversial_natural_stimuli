#!/usr/bin/env python3
"""Pair separation visualization.

Two parts:
 (A) 2x2 contingency: tied-on-baseline (yes/no) × separated-on-cstim (yes/no)
     for the headline cstim set (all_models), one panel per metric.
 (B) Conversion rate bar chart: per (metric, cstim_set), what fraction of
     baseline-tied (near-ceiling) pairs become separated on cstim?
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
CSTIM_SETS = ["all_models", "architecture", "dataset", "sota", "training_objective"]


def load():
    mrsa = pd.read_csv(DATA_DIR / "pair_separation_mrsa.csv")
    frsa = pd.read_csv(DATA_DIR / "pair_separation_frsa.csv")
    return mrsa, frsa


def panel_contingency(ax, df_subset, title):
    """2x2 table: rows = tied_on_baseline yes/no, cols = separated_on_cstim yes/no."""
    # Manually build 2x2 table to handle missing categories cleanly
    counts = np.zeros((2, 2), dtype=int)
    for tied_val, ti in [(True, 0), (False, 1)]:
        for sep_val, si in [(False, 0), (True, 1)]:
            counts[ti, si] = int(((df_subset["tied_baseline"] == tied_val)
                                 & (df_subset["separated_cstim"] == sep_val)).sum())
    colors = np.array([["#FFC857", "#2A9D8F"],   # tied + (sep_no = stayed_tied, sep_yes = converted)
                       ["#E76F51", "#264653"]])  # not_tied row
    ax.imshow(counts, cmap="Blues", aspect="auto", vmin=0, vmax=counts.max() * 1.2)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(counts[i, j]),
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["NO", "YES"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["YES", "NO"])
    ax.set_xlabel("Separated on cstim (q < 0.05)", fontsize=FONT["axis_label"])
    ax.set_ylabel("Tied on baseline (q > 0.05)", fontsize=FONT["axis_label"])
    ax.set_title(title, fontsize=FONT["title"])
    n_tied = int(counts[0].sum())
    n_conv = int(counts[0, 1])
    pct = 100 * n_conv / max(n_tied, 1)
    ax.text(0.5, -0.30,
            f"Of {n_tied} baseline-tied pairs, {n_conv} ({pct:.0f}%) become separated on cstim",
            transform=ax.transAxes, fontsize=FONT["annotation"], ha="center",
            color="#222")


def panel_conversion_rates(ax, mrsa, frsa):
    sets = CSTIM_SETS
    width = 0.35
    x = np.arange(len(sets))

    def rate(df, st):
        sub = df[df["model_set"] == st]
        n_tied_nc = int(sub["tied_and_near_ceiling"].sum())
        n_conv = int(sub["converted"].sum())
        return (100 * n_conv / max(n_tied_nc, 1)), n_conv, n_tied_nc

    mrsa_rates = [rate(mrsa, s) for s in sets]
    frsa_rates = [rate(frsa, s) for s in sets]
    mrsa_pct = [r[0] for r in mrsa_rates]
    frsa_pct = [r[0] for r in frsa_rates]

    bars_m = ax.bar(x - width / 2, mrsa_pct, width=width,
                    color="#264653", edgecolor="black", linewidth=0.4,
                    label="mRSA-transfer")
    bars_f = ax.bar(x + width / 2, frsa_pct, width=width,
                    color="#E9C46A", edgecolor="black", linewidth=0.4,
                    label="fRSA")
    for i, (rm, rf) in enumerate(zip(mrsa_rates, frsa_rates)):
        ax.text(x[i] - width / 2, rm[0] + 2, f"{rm[1]}/{rm[2]}",
                ha="center", fontsize=FONT["annotation"], color="#264653")
        ax.text(x[i] + width / 2, rf[0] + 2, f"{rf[1]}/{rf[2]}",
                ha="center", fontsize=FONT["annotation"], color="#8a7a3c")

    ax.set_xticks(x)
    ax.set_xticklabels(sets, rotation=15, ha="right", fontsize=FONT["tick"])
    ax.set_ylabel("Conversion rate (%)", fontsize=FONT["axis_label"])
    ax.set_title("Of baseline-tied near-ceiling pairs,\nfraction separated on cstim (FDR q<0.05)",
                 fontsize=FONT["title"])
    ax.set_ylim(0, 110)
    ax.axhline(0, color="black", lw=0.6)
    ax.legend(loc="upper right", fontsize=FONT["legend"], frameon=False)


def main():
    mrsa, frsa = load()

    fig = plt.figure(figsize=(12, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 2.0], wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    panel_contingency(ax0, mrsa[mrsa["model_set"] == "all_models"],
                      "mRSA, all_models")
    ax1 = fig.add_subplot(gs[0, 1])
    panel_contingency(ax1, frsa[frsa["model_set"] == "all_models"],
                      "fRSA, all_models")
    ax2 = fig.add_subplot(gs[0, 2])
    panel_conversion_rates(ax2, mrsa, frsa)

    fig.suptitle(
        "Cstim images render baseline-indistinguishable models distinguishable",
        fontsize=FONT["title"] + 1, y=1.02)

    out_pdf = FIG_DIR / "pair_separation.pdf"
    out_png = FIG_DIR / "pair_separation.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=DPI)
    print(f"Wrote {out_pdf}\nWrote {out_png}")


if __name__ == "__main__":
    main()
