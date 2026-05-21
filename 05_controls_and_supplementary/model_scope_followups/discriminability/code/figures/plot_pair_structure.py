#!/usr/bin/env python3
"""Pair-structure figure — redesigned for glanceability.

Single-narrative layout:
    a) Top:    schematic + 2×2 contingency with the headline number.
    b) Middle: per-model gain/loss diverging bar chart, sorted by net gain.
    c) Bottom: within-vs-across architecture family conversion rates.

Color encoding (Okabe-Ito):
    GAIN  (tied → separated)        : green     (#009E73)
    STAY  (tied → tied)             : yellow    (#F0E442)
    KEEP  (separated → separated)   : sky blue  (#56B4E9)
    LOSS  (separated → tied)        : vermillion(#D55E00)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT = Path(__file__).resolve().parents[4]
PAPER = PROJECT / "experiments" / "cstim_paper"
sys.path.insert(0, str(PAPER))
from style_improved import apply_style, FONT, DPI  # noqa

apply_style()

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parent

# Okabe-Ito
GAIN  = "#009E73"
STAY  = "#F0E442"
KEEP  = "#56B4E9"
LOSS  = "#D55E00"
INK   = "#222222"


# ---------------------------------------------------------------------------
# Panel A: schematic + 2x2 contingency
# ---------------------------------------------------------------------------
def panel_a(ax, df):
    """Top panel: explain the analysis and show the 2x2 outcome table."""
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")

    # ---- Headline text (top) ----
    ax.text(0.0, 9.5,
            "Of 190 model pairs, 57 are statistically indistinguishable on baseline natural images. "
            "On cstim images, 37 (65%) become distinguishable.",
            fontsize=FONT["title"], color=INK, va="top", ha="left",
            wrap=True, fontweight="bold")
    ax.text(0.0, 8.4,
            "We test each of $\\binom{20}{2}\\!=\\!190$ pairs of models for whether subject-level "
            "RSA scores differ (paired t, FDR q<0.05) and label each pair by its outcome:",
            fontsize=FONT["annotation"], color=INK, va="top", ha="left")

    # ---- 2×2 contingency table ----
    # Counts from the data (mRSA, all_models)
    sub = df[df["model_set"] == "all_models"].copy()
    n_total = len(sub)
    n_tied_base = int(sub["tied_baseline"].sum())
    n_sep_base = n_total - n_tied_base
    n_converted = int((sub["tied_baseline"] & sub["separated_cstim"]).sum())
    n_stayed_tied = int((sub["tied_baseline"] & ~sub["separated_cstim"]).sum())
    n_remained = int((~sub["tied_baseline"] & sub["separated_cstim"]).sum())
    n_lost = int((~sub["tied_baseline"] & ~sub["separated_cstim"]).sum())

    table_x0, table_y0 = 0.5, 0.5
    cell_w, cell_h = 3.6, 2.3

    # Cell positions (col, row): col 0 = "tied on cstim", col 1 = "sep on cstim"
    #                           row 0 (top) = "sep on baseline", row 1 = "tied on baseline"
    cells = [
        # (col, row, color, count, label)
        (0, 0, KEEP,  n_lost,        "LOST\n(sep → tied)"),
        (1, 0, KEEP,  n_remained,    "remained sep."),
        (0, 1, GAIN,  n_stayed_tied, "stayed tied"),
        (1, 1, GAIN,  n_converted,   "GAINED\n(tied → sep)"),
    ]
    color_map = {(0, 0): LOSS, (1, 0): KEEP, (0, 1): STAY, (1, 1): GAIN}
    for (col, row, _, count, label) in cells:
        x = table_x0 + col * cell_w
        # Lay rows top-to-bottom: row=0 (sep_baseline) at top, row=1 (tied) at bottom
        y = table_y0 + (1 - row) * cell_h
        c = color_map[(col, row)]
        ax.add_patch(mpatches.FancyBboxPatch((x, y), cell_w, cell_h,
                                              boxstyle="round,pad=0.04",
                                              facecolor=c, edgecolor="black",
                                              linewidth=0.6, alpha=0.55))
        ax.text(x + cell_w / 2, y + cell_h * 0.65, str(count),
                ha="center", va="center", fontsize=14, fontweight="bold",
                color="black")
        ax.text(x + cell_w / 2, y + cell_h * 0.22, label,
                ha="center", va="center", fontsize=FONT["tick"], color="black")

    # Column header (cstim status)
    ax.text(table_x0 + cell_w / 2, table_y0 + 2 * cell_h + 0.35,
            "tied on cstim\n(q > 0.05)", ha="center", va="bottom",
            fontsize=FONT["axis_label"], color=INK)
    ax.text(table_x0 + cell_w + cell_w / 2, table_y0 + 2 * cell_h + 0.35,
            "separated on cstim\n(q < 0.05)", ha="center", va="bottom",
            fontsize=FONT["axis_label"], color=INK)
    # Row header (baseline status)
    ax.text(table_x0 - 0.25, table_y0 + cell_h * 1.5,
            "separated on\nbaseline (q < 0.05)\n      ↑\n190 pairs total\n      ↓\ntied on baseline\n(q > 0.05)",
            ha="right", va="center", fontsize=FONT["tick"], color=INK)

    # Annotate "n=57 baseline-tied" and "n=133 baseline-sep" with arrows
    ax.text(table_x0 + 2 * cell_w + 0.1,
            table_y0 + cell_h * 1.5,
            f"57 pairs\ntied on\nbaseline",
            ha="left", va="center", fontsize=FONT["annotation"],
            color="#444", fontweight="bold")
    ax.text(table_x0 + 2 * cell_w + 0.1,
            table_y0 + cell_h * 0.5,
            f"133 pairs\nseparated on\nbaseline",
            ha="left", va="center", fontsize=FONT["annotation"],
            color="#444")


# ---------------------------------------------------------------------------
# Panel B: per-model diverging bar (gain vs loss)
# ---------------------------------------------------------------------------
def panel_b(ax, per_model):
    sub = per_model.copy()
    # net gain = converted_near_ceiling - lost
    sub["net_gain"] = sub["n_converted_near_ceiling"] - sub["n_lost"]
    sub = sub.sort_values("net_gain", ascending=True).reset_index(drop=True)
    y = np.arange(len(sub))

    # Lost: negative-direction bar (red, leftward)
    ax.barh(y, -sub["n_lost"], color=LOSS, edgecolor="black", linewidth=0.4,
            label="lost (sep → tied)")
    # Gained: positive-direction bar (green, rightward)
    ax.barh(y, sub["n_converted_near_ceiling"],
            color=GAIN, edgecolor="black", linewidth=0.4,
            label="gained (tied → sep)")

    # Annotate count text at end of each bar
    for i, row in sub.iterrows():
        if row["n_lost"] > 0:
            ax.text(-row["n_lost"] - 0.15, i, str(int(row["n_lost"])),
                    ha="right", va="center", fontsize=FONT["annotation"],
                    color=INK)
        if row["n_converted_near_ceiling"] > 0:
            ax.text(row["n_converted_near_ceiling"] + 0.15, i,
                    str(int(row["n_converted_near_ceiling"])),
                    ha="left", va="center", fontsize=FONT["annotation"],
                    color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(sub["display"].values, fontsize=FONT["tick"])
    ax.set_xlabel("# pair-distinctions   (lost)  ←  →   (gained on cstim)",
                  fontsize=FONT["axis_label"])
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlim(-7.5, 11)
    ax.set_ylim(-0.6, len(sub) - 0.4)
    ax.set_title("Each model's contribution to discriminability gain (cstim vs baseline)",
                 fontsize=FONT["title"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", fontsize=FONT["legend"], frameon=False)


# ---------------------------------------------------------------------------
# Panel C: family conversion rates
# ---------------------------------------------------------------------------
def panel_c(ax, family_df):
    groups = ["ResNet-50", "ViT-L", "across"]
    labels = ["Within ResNet-50",
              "Within ViT-L",
              "Across architectures"]
    rates = [family_df[family_df["group"] == g]["conversion_rate_pct"].iloc[0]
             for g in groups]
    n_tied = [int(family_df[family_df["group"] == g]["n_tied_near_ceiling"].iloc[0])
              for g in groups]
    y = np.arange(len(groups))
    ax.barh(y, rates, color=GAIN, edgecolor="black", linewidth=0.5,
            alpha=0.75)
    for i, (r, n) in enumerate(zip(rates, n_tied)):
        ax.text(r + 1.5, i, f"{r:.0f}%  ({int(round(r/100*n))}/{n})",
                ha="left", va="center", fontsize=FONT["tick"], color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FONT["tick"])
    ax.set_xlabel("% baseline-tied near-ceiling pairs separated on cstim",
                  fontsize=FONT["axis_label"])
    ax.set_xlim(0, 100)
    ax.axvline(50, color="#888", lw=0.5, ls=":", alpha=0.7)
    ax.set_title("Conversion rate by architecture-family relationship",
                 fontsize=FONT["title"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pair_df = pd.read_csv(DATA_DIR / "which_models_pair_outcomes.csv")
    per_model = pd.read_csv(DATA_DIR / "which_models_per_model_profile.csv")
    family_df = pd.read_csv(DATA_DIR / "which_models_family_conversion.csv")

    # Nature double-column: 183 mm = 7.2"
    fig = plt.figure(figsize=(7.2, 9.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.6, 0.55],
                          hspace=0.42,
                          left=0.10, right=0.97, top=0.97, bottom=0.05)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_a(ax_a, pair_df)
    ax_a.text(-0.05, 1.00, "a", transform=ax_a.transAxes,
              fontsize=FONT["panel_label"], fontweight="bold")

    ax_b = fig.add_subplot(gs[1, 0])
    panel_b(ax_b, per_model)
    ax_b.text(-0.13, 1.04, "b", transform=ax_b.transAxes,
              fontsize=FONT["panel_label"], fontweight="bold")

    ax_c = fig.add_subplot(gs[2, 0])
    panel_c(ax_c, family_df)
    ax_c.text(-0.13, 1.10, "c", transform=ax_c.transAxes,
              fontsize=FONT["panel_label"], fontweight="bold")

    out_pdf = FIG_DIR / "pair_structure.pdf"
    out_png = FIG_DIR / "pair_structure.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=DPI)
    print(f"Wrote {out_pdf}\nWrote {out_png}")


if __name__ == "__main__":
    main()
