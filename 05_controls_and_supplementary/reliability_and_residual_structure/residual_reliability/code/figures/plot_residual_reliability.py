#!/usr/bin/env python3
"""
Plot the corrected ensemble-gap and residual-LOSO analyses.

Top row:
  - bar = cross-validated 20-model ensemble correlation with the brain RDM;
  - grey tick = best single model;
  - colored cap = sqrt(split-half reliability), i.e. the model-correlation
    ceiling.

Bottom row:
  - bar = subject-to-LOSO residual RSA after removing each subject's full
    model-RDM space;
  - colored cap = subject-to-LOSO brain RSA before residualization.

Error bars: SEM across subjects after averaging vicco bootstraps per subject.

Output: figures/residual_reliability.{pdf,png}
"""

from __future__ import annotations

import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
_SHARE_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))
sys.path.insert(0, str(_SHARE_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cstims.paper.style_improved import apply_style, FONT, DPI, W_DOUBLE

apply_style()

GROUPS = ["vicco", "all_models", "sota", "architecture", "dataset", "training_objective"]
GROUP_LABELS = {
    "vicco": "Baseline",
    "all_models": "All\nModels",
    "sota": "SOTA",
    "architecture": "Arch.",
    "dataset": "Dataset",
    "training_objective": "Train.\nObj.",
}

COLOR_BASELINE = "#2980B9"
COLOR_CSTIM = "#D64541"

STAGE = Path(__file__).resolve().parents[2]
DATA_CSV = STAGE / "results" / "residual_rsa.csv"
OUT_DIR = STAGE / "figures" / "supplementary"
PNG_DIR = OUT_DIR / "png"


def agg_across_subjects(df: pd.DataFrame, rsa_type: str, group: str) -> dict:
    sub = df[(df.rsa_type == rsa_type) & (df.stimulus_group == group)]
    if sub.empty:
        return {}
    per_subject = sub.groupby("subject").agg(
        r_single=("r_single_best", "mean"),
        r_ens=("r_ensemble_cv", "mean"),
        corr_ceil=("correlation_ceiling", "mean"),
        loso_brain=("r_loso_brain", "mean"),
        loso_resid=("r_loso_residual", "mean"),
    )
    n = len(per_subject)
    if n == 0:
        return {}
    stat = {}
    for col in ["r_single", "r_ens", "corr_ceil", "loso_brain", "loso_resid"]:
        stat[col] = float(per_subject[col].mean())
        stat[f"{col}_sem"] = (
            float(per_subject[col].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        )
    stat["n"] = n
    return stat


def _draw_ensemble_panel(ax, df, rsa_type, title):
    xs = np.arange(len(GROUPS))
    width = 0.58

    for i, g in enumerate(GROUPS):
        v = agg_across_subjects(df, rsa_type, g)
        if not v:
            continue
        base_c = COLOR_BASELINE if g == "vicco" else COLOR_CSTIM
        x0 = xs[i]

        # Solid fill = cross-validated ensemble correlation.
        ax.bar(x0, v["r_ens"], width, color=base_c, alpha=0.55,
               edgecolor=base_c, linewidth=0.8, zorder=2,
               yerr=v["r_ens_sem"],
               error_kw=dict(ecolor="0.3", capsize=2, elinewidth=0.7, zorder=5))
        # Correct model-correlation ceiling: sqrt(split-half reliability).
        ax.hlines(v["corr_ceil"], x0 - width/2, x0 + width/2,
                  colors=base_c, linewidth=1.4, linestyles="-", zorder=4)
        ax.fill_between([x0 - width/2, x0 + width/2],
                        v["corr_ceil"] - v["corr_ceil_sem"],
                        v["corr_ceil"] + v["corr_ceil_sem"],
                        color=base_c, alpha=0.18, linewidth=0, zorder=3)

        # Best-single-model tick (thin grey solid line).
        ax.hlines(v["r_single"], x0 - width/2 - 0.02, x0 + width/2 + 0.02,
                  colors="#333", linewidth=0.7, zorder=5)

        if np.isfinite(v["corr_ceil"]) and np.isfinite(v["r_ens"]):
            gap = v["corr_ceil"] - v["r_ens"]
            y_text = max(v["corr_ceil"], v["r_ens"], v["r_single"]) + 0.035
            ax.text(x0, y_text, f"gap\n{gap:.2f}",
                    ha="center", va="bottom", fontsize=FONT["small"] - 1,
                    color=base_c, linespacing=0.95)

    ax.set_xticks(xs)
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUPS], fontsize=FONT["tick"])
    ax.set_title(title, fontsize=FONT["title"])
    ax.axhline(0, color="#444", linewidth=0.6, zorder=0)
    ax.set_ylim(0, 1.0)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.20, linewidth=0.5)


def _draw_loso_panel(ax, df, rsa_type, title):
    xs = np.arange(len(GROUPS))
    width = 0.58

    ymin = 0.0
    ymax = 0.0
    for i, g in enumerate(GROUPS):
        v = agg_across_subjects(df, rsa_type, g)
        if not v:
            continue
        base_c = COLOR_BASELINE if g == "vicco" else COLOR_CSTIM
        x0 = xs[i]

        ax.bar(x0, v["loso_resid"], width, color=base_c, alpha=0.55,
               edgecolor=base_c, linewidth=0.8, zorder=2,
               yerr=v["loso_resid_sem"],
               error_kw=dict(ecolor="0.3", capsize=2, elinewidth=0.7, zorder=5))
        ax.hlines(v["loso_brain"], x0 - width/2, x0 + width/2,
                  colors=base_c, linewidth=1.4, linestyles="-", zorder=4)
        ax.fill_between([x0 - width/2, x0 + width/2],
                        v["loso_brain"] - v["loso_brain_sem"],
                        v["loso_brain"] + v["loso_brain_sem"],
                        color=base_c, alpha=0.18, linewidth=0, zorder=3)

        if np.isfinite(v["loso_resid"]):
            y_text = max(v["loso_brain"], v["loso_resid"]) + 0.03
            ax.text(x0, y_text, f"r={v['loso_resid']:.2f}",
                    ha="center", va="bottom", fontsize=FONT["small"] - 1,
                    color=base_c, linespacing=0.95)

        ymin = min(ymin, v["loso_resid"] - v["loso_resid_sem"])
        ymax = max(ymax, v["loso_brain"] + v["loso_brain_sem"], v["loso_resid"])

    ax.set_xticks(xs)
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUPS], fontsize=FONT["tick"])
    ax.set_title(title, fontsize=FONT["title"])
    ax.axhline(0, color="#444", linewidth=0.6, zorder=0)
    ax.set_ylim(min(-0.05, ymin - 0.04), max(0.35, ymax + 0.12))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.20, linewidth=0.5)


def make_figure(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE, 6.4), sharey=False)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.10,
                        hspace=0.45, wspace=0.18)

    _draw_ensemble_panel(axes[0, 0], df, "fixed", "Fixed RSA: ensemble gap")
    _draw_ensemble_panel(axes[0, 1], df, "mixed", "Mixed RSA: ensemble gap")
    _draw_loso_panel(axes[1, 0], df, "fixed", "Fixed RSA: LOSO residual")
    _draw_loso_panel(axes[1, 1], df, "mixed", "Mixed RSA: LOSO residual")
    axes[0, 0].set_ylabel(r"Spearman $r_s$")
    axes[1, 0].set_ylabel(r"Spearman $r_s$")

    # Legend
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    handles = [
        mpatches.Patch(facecolor=COLOR_CSTIM, alpha=0.55, edgecolor=COLOR_CSTIM,
                       label="20-model ensemble / residual RSA"),
        Line2D([0], [0], color=COLOR_CSTIM, linewidth=1.4, linestyle="-",
               label="Correlation ceiling or total LOSO brain RSA"),
        Line2D([0], [0], color="#333", linewidth=0.7, linestyle="-",
               label="Best single model, top row"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               frameon=False, fontsize=FONT["legend"],
               bbox_to_anchor=(0.5, 1.00), handlelength=1.6, columnspacing=1.2)
    return fig


def main():
    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df)} rows from {DATA_CSV}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_figure(df)
    out_pdf = OUT_DIR / "residual_reliability.pdf"
    out_png = PNG_DIR / "residual_reliability.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
