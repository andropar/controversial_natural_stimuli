#!/usr/bin/env python3
"""
Plot mRSA + fRSA score distributions for 125 large-benchmark models vs noise ceiling.

Two rows (mRSA / fRSA) × two output variants:
  large_benchmark_gap_raw.pdf/png   — raw scores
  large_benchmark_gap_norm.pdf/png  — NC-normalized (score / sqrt(NC))

Each dot = one model averaged across subjects.
Set-model overlay uses cross_set_wrsa_scores.csv (mRSA) and crsa_scores.csv (fRSA).
"""

import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
_FIGURES = _PAPER / "figures"
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))
sys.path.insert(0, str(_FIGURES))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from style import apply_style, FONT, DPI, W_DOUBLE
from config import RSA_DATA_DIR, STATS_DATA_DIR, SUBJECTS

apply_style()

GROUPS = ["vicco", "all_models", "architecture", "dataset", "sota", "training_objective"]
LABELS = {
    "all_models": "All Models",
    "architecture": "Architecture",
    "dataset": "Dataset",
    "sota": "SOTA",
    "training_objective": "Train. Obj.",
    "vicco": "Baseline",
}
COLOR_CSTIM = "#D64541"
COLOR_VICCO = "#2980B9"
COLOR_SET   = "#2C3E50"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_large_scores():
    return pd.read_csv(RSA_DATA_DIR / "rsa_large_benchmark_scores.csv")


def load_set_scores():
    """
    Returns (wrsa_cstim, wrsa_vicco, crsa_all):
      wrsa_cstim  — cross_set_wrsa_scores (all groups, controversial)
      wrsa_vicco  — wrsa_transfer_scores  (vicco only)
      crsa_all    — crsa_scores           (in-set controversial + vicco)
    """
    w_cstim, w_vicco, crsa = [], [], []
    for sub in SUBJECTS:
        for fname, lst in [
            ("cross_set_wrsa_scores.csv", w_cstim),
            ("wrsa_transfer_scores.csv",  w_vicco),
            ("crsa_scores.csv",           crsa),
        ]:
            f = RSA_DATA_DIR / sub / fname
            if f.exists():
                lst.append(pd.read_csv(f))
    return (
        pd.concat(w_cstim, ignore_index=True) if w_cstim else None,
        pd.concat(w_vicco, ignore_index=True) if w_vicco else None,
        pd.concat(crsa,    ignore_index=True) if crsa    else None,
    )


def load_nc():
    return pd.read_csv(STATS_DATA_DIR / "rdm_noise_ceilings.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_nc_vals(nc_df, group, stim_type):
    rows = nc_df[(nc_df["group"] == group) & (nc_df["stimulus_type"] == stim_type)]
    if rows.empty:
        return np.array([])
    return np.sqrt(rows.groupby("subject")["noise_ceiling_spearman"].mean().values)


def get_large_scores(rsa, group, score_col):
    """Per-model mean across subjects from the large benchmark CSV."""
    is_vicco = group == "vicco"
    if is_vicco:
        sub = rsa[rsa["stimulus_type"] == "vicco"]
    else:
        sub = rsa[(rsa["stimulus_type"] == "controversial") & (rsa["group"] == group)]
    s = sub.groupby("model")[score_col].mean()
    return s[np.isfinite(s.values)]


def get_set_scores(wrsa_cstim, wrsa_vicco, crsa_all, group, score_col):
    """Per-model mean for the original set models."""
    is_vicco = group == "vicco"
    if score_col == "wrsa_transfer":
        if is_vicco:
            if wrsa_vicco is None:
                return pd.Series(dtype=float)
            s = wrsa_vicco[wrsa_vicco["stimulus_type"] == "vicco"].groupby("model")["wrsa_transfer"].mean()
        else:
            if wrsa_cstim is None:
                return pd.Series(dtype=float)
            s = wrsa_cstim[wrsa_cstim["stimulus_group"] == group].groupby("model")["wrsa_transfer"].mean()
    else:  # crsa
        if crsa_all is None:
            return pd.Series(dtype=float)
        stim = "vicco" if is_vicco else "controversial"
        s = crsa_all[crsa_all["stimulus_type"] == stim].groupby("model")["crsa"].mean()
    return s[np.isfinite(s.values)]


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------

def draw_panel(ax, rsa, nc_df, wrsa_cstim, wrsa_vicco, crsa_all,
               score_col, row_label, normalize, show_xticks):
    n = len(GROUPS)
    xs = np.arange(n)

    if normalize:
        vicco_nc = get_nc_vals(nc_df, "vicco", "vicco").mean()
        vicco_scores = get_large_scores(rsa, "vicco", score_col)
        vicco_models = vicco_scores / vicco_nc

        # Vicco spread (median absolute pairwise difference) for ratio
        vicco_norm = vicco_scores.values / vicco_nc
        def _median_pair_diff(x):
            x = np.asarray(x)
            diffs = np.abs(x[:, None] - x[None, :])
            i, j = np.triu_indices_from(diffs, k=1)
            return float(np.median(diffs[i, j]))
        mpd_vicco = _median_pair_diff(vicco_norm)

    # Top-10 models by all_models score — used for connecting lines
    TOP_N = 10
    am_scores = get_large_scores(rsa, "all_models", score_col)
    am_nc     = get_nc_vals(nc_df, "all_models", "controversial").mean()
    top_models = set(am_scores.nlargest(TOP_N).index)

    # Collect dot positions for top models per group so we can draw lines later
    top_positions = {}  # group -> {model: (x, y)}

    for i, group in enumerate(GROUPS):
        is_vicco = group == "vicco"
        color = COLOR_VICCO if is_vicco else COLOR_CSTIM
        stim_type = "vicco" if is_vicco else "controversial"

        scores_raw = get_large_scores(rsa, group, score_col)
        nc_vals = get_nc_vals(nc_df, group, stim_type)
        nc_mean = nc_vals.mean() if len(nc_vals) > 0 else 1.0
        nc_sem  = nc_vals.std(ddof=1) / np.sqrt(len(nc_vals)) if len(nc_vals) > 1 else 0

        scores = scores_raw / nc_mean if normalize else scores_raw
        nc_line = 1.0 if normalize else nc_mean
        nc_lo   = (nc_mean - nc_sem) / nc_mean if normalize else nc_mean - nc_sem
        nc_hi   = (nc_mean + nc_sem) / nc_mean if normalize else nc_mean + nc_sem

        # Violin
        if len(scores) > 3:
            vp = ax.violinplot(scores.values, positions=[i], widths=0.6,
                               showmedians=False, showextrema=False)
            for pc in vp["bodies"]:
                pc.set_facecolor(color); pc.set_alpha(0.22)
                pc.set_edgecolor(color); pc.set_linewidth(0.5)

        # Split into top and non-top models
        is_top = np.array([m in top_models for m in scores.index])
        jitter_all = np.random.default_rng(42).uniform(-0.15, 0.15, len(scores))

        # Non-top strip dots
        ax.scatter(xs[i] + jitter_all[~is_top], scores.values[~is_top],
                   s=4, color=color, alpha=0.35, linewidths=0, zorder=3)

        # Top model dots — highlighted, use rank-based consistent jitter
        top_in_group = [m for m in scores.index[is_top]]
        if top_in_group:
            n_top = len(top_in_group)
            jitter_top = np.linspace(-0.10, 0.10, n_top)
            top_y = scores[top_in_group].values
            ax.scatter(xs[i] + jitter_top, top_y, s=8,
                       color=color, alpha=0.8, linewidths=0, zorder=5)
            top_positions[group] = {m: (xs[i] + jitter_top[k], top_y[k])
                                    for k, m in enumerate(top_in_group)}

        # Median bar
        ax.hlines(np.median(scores.values), xs[i] - 0.22, xs[i] + 0.22,
                  colors=color, linewidth=1.8, zorder=6)

        # NC ribbon (raw only)
        if not normalize:
            ax.fill_between([xs[i] - 0.35, xs[i] + 0.35], nc_lo, nc_hi,
                            color=color, alpha=0.10, linewidth=0, zorder=0)
            ax.hlines(nc_line, xs[i] - 0.35, xs[i] + 0.35,
                      colors=color, linewidth=0.8, alpha=0.55, zorder=1, linestyle="--")

        # Set-model overlay
        set_sc = get_set_scores(wrsa_cstim, wrsa_vicco, crsa_all, group, score_col)
        if len(set_sc) > 0:
            sv = set_sc.values / nc_mean if normalize else set_sc.values
            jitter_m = np.random.default_rng(7).uniform(-0.06, 0.06, len(sv))
            ax.scatter(xs[i] + 0.28 + jitter_m, sv, s=14,
                       facecolors="white", edgecolors=COLOR_SET,
                       linewidths=0.8, zorder=7, marker="o")

    # Connecting lines for top models: vicco → all_models
    if "vicco" in top_positions and "all_models" in top_positions:
        for model in top_models:
            if model in top_positions["vicco"] and model in top_positions["all_models"]:
                x0, y0 = top_positions["vicco"]["all_models" if False else model]  # vicco col
                x1, y1 = top_positions["all_models"][model]
                ax.plot([x0, x1], [y0, y1], color=COLOR_SET,
                        linewidth=0.6, alpha=0.4, zorder=4)

    # Thin separator between vicco (x=0) and all_models (x=1)
    ax.axvline(0.5, color="#cccccc", linewidth=0.8, linestyle="-", zorder=0)

    # Row label
    ax.set_ylabel(f"{row_label} / NC" if normalize else f"{row_label} score ($r_s$)")

    # X-ticks
    ax.set_xticks(xs)
    if normalize and show_xticks:
        # Bottom row: group name + median % of NC + CV ratio
        tick_labels = []
        for group in GROUPS:
            is_vicco = group == "vicco"
            stim_type = "vicco" if is_vicco else "controversial"
            nc_mean_t = get_nc_vals(nc_df, group, stim_type).mean()
            s_norm = get_large_scores(rsa, group, score_col).values / nc_mean_t
            pct = int(round(np.median(s_norm) * 100))
            if is_vicco:
                tick_labels.append(f"{LABELS[group]}\nmed. {pct}% of NC")
            else:
                spread_ratio = _median_pair_diff(s_norm) / mpd_vicco if mpd_vicco > 0 else np.nan
                tick_labels.append(
                    f"{LABELS[group]}\nmed. {pct}% of NC\nspread ratio: {spread_ratio:.2f}×")
        ax.set_xticklabels(tick_labels)
    else:
        # Top row or raw: just group names
        ax.set_xticklabels([LABELS[g] for g in GROUPS])

    ax.set_xlim(-0.6, n - 0.4)
    if normalize:
        ax.set_ylim(0, 1.15)
        ax.axhline(1.0, color="#444444", linewidth=0.8, linestyle="--", zorder=0, alpha=0.5)
        ax.text(-0.45, 1.02, "NC (max)", va="bottom",
                fontsize=FONT["annotation"], color="#444444", alpha=0.8)
    else:
        ax.set_ylim(bottom=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = Path(__file__).resolve().parent
    rsa = load_large_scores()
    nc_df = load_nc()
    wrsa_cstim, wrsa_vicco, crsa_all = load_set_scores()
    has_set = wrsa_cstim is not None or wrsa_vicco is not None

    rows = [("wrsa_transfer", "mixed RSA"), ("crsa", "fixed RSA")]

    for normalize, suffix in [(False, "raw"), (True, "norm")]:
        fig, axes = plt.subplots(2, 1, figsize=(W_DOUBLE, 7.5))
        fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.14, hspace=0.28)

        for ax, (score_col, row_label) in zip(axes, rows):
            draw_panel(ax, rsa, nc_df, wrsa_cstim, wrsa_vicco, crsa_all,
                       score_col=score_col, row_label=row_label,
                       normalize=normalize, show_xticks=True)

        # Shared legend on bottom panel
        handles = []
        if not normalize:
            handles.append(Line2D([0], [0], linestyle="--", color="gray",
                                  linewidth=0.8, label="Noise ceiling (±SEM)"))
        if has_set:
            handles.append(Line2D([0], [0], marker="o", color="none",
                                  markerfacecolor="white", markeredgecolor=COLOR_SET,
                                  markeredgewidth=0.8, markersize=5, label="Set models"))
        if handles:
            axes[1].legend(handles=handles, loc="lower right", frameon=True,
                           framealpha=0.9, edgecolor="none", fontsize=FONT["legend"],
                           handletextpad=0.4, handlelength=1.5)

        for ext in ("pdf", "png"):
            p = out_dir / f"large_benchmark_gap_{suffix}.{ext}"
            fig.savefig(p, dpi=DPI if ext == "png" else None, bbox_inches="tight")
        print(f"Saved: large_benchmark_gap_{suffix}.pdf/png")
        plt.close(fig)


if __name__ == "__main__":
    main()
