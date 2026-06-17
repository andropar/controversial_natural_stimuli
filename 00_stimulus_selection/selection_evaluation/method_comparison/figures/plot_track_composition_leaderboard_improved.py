#!/usr/bin/env python3
"""
Improved selection-track composition leaderboard.

Fixes vs. original:
- AUC convention flipped to match Fig. 1 of the paper:
  x-axis is now "Accuracy AUC (higher = better)" rather than the original
  error-probability AUC. This eliminates the inconsistency between Fig. 1c
  and the methods-supplementary leaderboard.
- Method labels humanised ("Raw + sub-03 encoding (correlation)" instead
  of "raw_plus_sub-03_hlvis (correlation)").
- Okabe-Ito palette via style_improved (blue + vermillion).
"""
from __future__ import annotations

import sys
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Path to style_improved
_PAPER_FIGURES = Path(__file__).resolve().parents[4] / "experiments" / "cstim_paper" / "figures"
sys.path.insert(0, str(_PAPER_FIGURES))
from cstims.paper.style_improved import (
    apply_style, FONT, DPI, OKABE_ITO,
)

apply_style()

REPO_ROOT = Path(__file__).resolve().parents[4]
CSV_PATH = REPO_ROOT / "scripts/cursor/outputs/final_aggregate_plot/leaderboard_summary.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "track_composition_leaderboard_improved"

COLOR_CORR = OKABE_ITO["blue"]
COLOR_EUCL = OKABE_ITO["vermillion"]


def humanize(name: str) -> str:
    """Convert e.g. 'raw_plus_sub-03_hlvis (correlation)' →
    'Raw + sub-03 encoding (correlation)'."""
    if name.lower() == "random":
        return "Random"
    base, _, parens = name.partition(" (")
    parens = "(" + parens if parens else ""
    parts = base.split("_")
    if parts[0] == "raw":
        if len(parts) == 1:
            human = "Raw"
        elif parts[1] == "plus":
            tail = "_".join(parts[2:])
            tail = (tail
                    .replace("all_encodings", "all subj. encodings")
                    .replace("group", "group encoding")
                    .replace("hlvis", "encoding"))
            human = f"Raw + {tail}"
        else:
            human = " ".join(parts).title()
    elif parts[0] == "group":
        human = "Group encoding only"
    else:
        human = " ".join(parts)
    return f"{human} {parens}".strip()


def main():
    df = pd.read_csv(CSV_PATH)
    wide = df.pivot_table(
        index="Full Method", columns="metric",
        values=["score_mean", "score_se"]
    )
    wide.columns = [f"{stat}_{metric}" for stat, metric in wide.columns]
    # Convert error-AUC to accuracy-AUC: 1 - score
    wide["acc_corr"] = 1.0 - wide["score_mean_correlation"]
    wide["acc_eucl"] = 1.0 - wide["score_mean_euclidean"]
    wide["se_corr"] = wide["score_se_correlation"]
    wide["se_eucl"] = wide["score_se_euclidean"]
    wide["avg"] = wide[["acc_corr", "acc_eucl"]].mean(axis=1)
    # Best (highest accuracy) on top
    wide = wide.sort_values("avg", ascending=True)

    methods = list(wide.index)
    n = len(methods)
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(10, 0.42 * n + 1.5))
    offset = 0.18

    ax.errorbar(
        wide["acc_corr"].values, y - offset,
        xerr=wide["se_corr"].values,
        fmt="o", color=COLOR_CORR, markersize=7,
        markeredgecolor="white", markeredgewidth=0.8,
        capsize=3, linewidth=1.2, label="eval: correlation", zorder=3,
    )
    ax.errorbar(
        wide["acc_eucl"].values, y + offset,
        xerr=wide["se_eucl"].values,
        fmt="s", color=COLOR_EUCL, markersize=6,
        markeredgecolor="white", markeredgewidth=0.8,
        capsize=3, linewidth=1.2, label="eval: euclidean", zorder=3,
    )

    if "Random" in wide.index:
        ax.axvline(wide.loc["Random", "acc_corr"], color=COLOR_CORR,
                    linestyle="--", linewidth=1.0, alpha=0.45, zorder=1)
        ax.axvline(wide.loc["Random", "acc_eucl"], color=COLOR_EUCL,
                    linestyle="--", linewidth=1.0, alpha=0.45, zorder=1)
        rand_idx = methods.index("Random")
        ax.axhspan(rand_idx - 0.5, rand_idx + 0.5, color="0.93", zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels([humanize(m) for m in methods],
                        fontsize=FONT["small"])
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy AUC (higher = better)")
    # Identify the best composition for a claim-first subtitle.
    if len(wide):
        best_idx = wide["avg"].idxmax()
        best_label = humanize(best_idx)
    else:
        best_label = "(none)"
    ax.set_title(
        f"Selection-track composition leaderboard — best: {best_label}\n"
        "aggregated across model sets; x = in-silico model-recovery AUC",
        fontsize=FONT["title"],
    )
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    legend_handles = [
        Line2D([0], [0], marker="o", color=COLOR_CORR, markersize=7,
                markeredgecolor="white", linestyle="none",
                label="eval: correlation"),
        Line2D([0], [0], marker="s", color=COLOR_EUCL, markersize=6,
                markeredgecolor="white", linestyle="none",
                label="eval: euclidean"),
        Line2D([0], [0], color="0.4", linestyle="--", linewidth=1.0,
                label="Random baseline"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
                frameon=False, fontsize=FONT["small"])

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / (OUTPUT_STEM + ".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / (OUTPUT_STEM + ".png"),
                  bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"Wrote {OUTPUT_DIR / (OUTPUT_STEM + '.pdf')}")


if __name__ == "__main__":
    main()
