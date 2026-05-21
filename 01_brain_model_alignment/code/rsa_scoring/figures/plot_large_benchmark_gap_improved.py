#!/usr/bin/env python3
"""
Improved large-benchmark gap figure.

Fixes vs. original:
- Half-violin + raw-point strip instead of full violins (less ink, same info).
- "Set models" foregrounded: bigger black dots, drawn on top of the strip.
- Connector lines for set models drawn for ALL controversial panels
  (not only the all-models column).
- Group-summary text moved out of x-tick labels into a per-panel annotation
  box; x-ticks just have the group name.
- Okabe-Ito palette via style_improved (red→vermillion, blue→blue).
- Panel labels (a, b).
"""
from __future__ import annotations

import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
_FIGURES = _PAPER / "figures"
_SHARE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))
sys.path.insert(0, str(_FIGURES))
sys.path.insert(0, str(_SHARE_ROOT / "shared" / "code" / "paper_helpers"))
sys.path.insert(0, str(_SHARE_ROOT / "shared" / "code" / "paper_helpers" / "figures"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon

from style_improved import (
    apply_style, FONT, DPI, W_DOUBLE,
    COLOR_CSTIM, COLOR_BASELINE,
    add_panel_label,
)
from config import SUBJECTS

apply_style()

GROUPS = ["vicco", "all_models", "sota", "training_objective", "architecture", "dataset"]
STAGE_DIR = Path(__file__).resolve().parents[3]
SHARE_ROOT = STAGE_DIR.parent
RSA_DATA_DIR = STAGE_DIR / "rsa_scores"
STATS_DATA_DIR = SHARE_ROOT / "02_alignment_reliability" / "data"
FIGURES_DIR = STAGE_DIR / "figures" / "rsa_scores"
PNG_DIR = FIGURES_DIR / "png"
SUPPLEMENTARY_DIR = FIGURES_DIR / "supplementary"
SUPPLEMENTARY_PNG_DIR = SUPPLEMENTARY_DIR / "png"
LABELS = {
    "all_models":         "All Models",
    "architecture":       "Architecture",
    "dataset":            "Dataset",
    "sota":               "SOTA",
    "training_objective": "Train. Obj.",
    "vicco":              "Baseline",
}
COLOR_SET = "#222222"


def load_large_scores():
    return pd.read_csv(RSA_DATA_DIR / "rsa_large_benchmark_scores.csv")


def load_set_scores():
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


def get_nc_vals(nc_df, group, stim_type):
    rows = nc_df[(nc_df["group"] == group) & (nc_df["stimulus_type"] == stim_type)]
    if rows.empty:
        return np.array([])
    return np.sqrt(rows.groupby("subject")["noise_ceiling_spearman"].mean().values)


def get_large_scores(rsa, group, score_col):
    is_vicco = group == "vicco"
    if is_vicco:
        sub = rsa[rsa["stimulus_type"] == "vicco"]
    else:
        sub = rsa[(rsa["stimulus_type"] == "controversial") & (rsa["group"] == group)]
    s = sub.groupby("model")[score_col].mean()
    return s[np.isfinite(s.values)]


def get_set_scores(wrsa_cstim, wrsa_vicco, crsa_all, group, score_col):
    is_vicco = group == "vicco"
    if score_col == "wrsa_transfer":
        if is_vicco:
            if wrsa_vicco is None:
                return pd.Series(dtype=float)
            return wrsa_vicco[wrsa_vicco["stimulus_type"] == "vicco"].groupby("model")["wrsa_transfer"].mean()
        if wrsa_cstim is None:
            return pd.Series(dtype=float)
        return wrsa_cstim[wrsa_cstim["stimulus_group"] == group].groupby("model")["wrsa_transfer"].mean()
    if crsa_all is None:
        return pd.Series(dtype=float)
    stim = "vicco" if is_vicco else "controversial"
    return crsa_all[crsa_all["stimulus_type"] == stim].groupby("model")["crsa"].mean()


def _median_pair_diff(x):
    x = np.asarray(x)
    diffs = np.abs(x[:, None] - x[None, :])
    i, j = np.triu_indices_from(diffs, k=1)
    return float(np.median(diffs[i, j]))


def _bh_fdr(pvals):
    """Benjamini–Hochberg FDR correction. Returns adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    mask = np.isfinite(p)
    out = np.full_like(p, np.nan)
    if mask.sum() == 0:
        return out
    pm = p[mask]
    order = np.argsort(pm)
    ranked = pm[order]
    m = len(ranked)
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    res = np.empty(m)
    res[order] = adj
    out[mask] = res
    return out


def paired_wilcoxon_vs_baseline(rsa, score_col, normalize, nc_by_group):
    """For each controversial group, test (group - baseline) per benchmark
    model, paired by model. Returns dict: group -> dict(median_delta, p, q)."""
    base_scores = get_large_scores(rsa, "vicco", score_col)
    if normalize:
        nc_b = nc_by_group["vicco"] or 1.0
        base_scores = base_scores / nc_b
    results = {}
    pvals = []
    keys = []
    for g in GROUPS:
        if g == "vicco":
            continue
        gs = get_large_scores(rsa, g, score_col)
        if normalize:
            nc_g = nc_by_group[g] or 1.0
            gs = gs / nc_g
        common = base_scores.index.intersection(gs.index)
        if len(common) < 5:
            results[g] = dict(median_delta=np.nan, p=np.nan, n=len(common))
            continue
        b = base_scores.loc[common].values
        c = gs.loc[common].values
        delta = c - b
        try:
            stat, p = wilcoxon(c, b, alternative="two-sided", zero_method="wilcox")
        except ValueError:
            p = np.nan
        results[g] = dict(median_delta=float(np.median(delta)), p=float(p),
                            n=int(len(common)))
        pvals.append(p)
        keys.append(g)
    qvals = _bh_fdr(np.array(pvals))
    for k, q in zip(keys, qvals):
        results[k]["q"] = float(q)
    return results


def draw_panel(ax, rsa, nc_df, sets, score_col, row_label, normalize, panel_label):
    wrsa_cstim, wrsa_vicco, crsa_all = sets
    n = len(GROUPS)
    xs = np.arange(n)

    if normalize:
        vicco_nc = get_nc_vals(nc_df, "vicco", "vicco").mean()
        vicco_norm = get_large_scores(rsa, "vicco", score_col).values / vicco_nc
        mpd_vicco = _median_pair_diff(vicco_norm)

    set_scores_by_group = {
        g: get_set_scores(wrsa_cstim, wrsa_vicco, crsa_all, g, score_col)
        for g in GROUPS
    }
    nc_by_group = {
        g: get_nc_vals(nc_df, g, "vicco" if g == "vicco" else "controversial").mean()
        for g in GROUPS
    }

    rng = np.random.default_rng(42)

    # Set y-limits early so the mask rectangle can use the right height.
    if normalize:
        ax.set_ylim(0, 1.18)
    else:
        # Compute a sane raw upper bound from data
        all_vals = []
        for g in GROUPS:
            all_vals.extend(get_large_scores(rsa, g, score_col).values)
        ax.set_ylim(0, max(all_vals) * 1.15 if all_vals else 1.0)

    # ---- Mirrored layout ----
    # Baseline column: strip on LEFT, selection-set dots on RIGHT (+0.30).
    # Controversial columns: strip on RIGHT, selection-set dots on LEFT (-0.30).
    # → black dots in Baseline (right side) sit DIRECTLY ACROSS from black
    #   dots in All Models (left side), separated only by the inter-column
    #   gap. Reader can pair same-model scores by tracing horizontally with
    #   no need for arc connectors.

    def is_baseline(g):
        return g == "vicco"

    # ---- 1. Compute set-model positions ----
    set_positions = {}
    for i, group in enumerate(GROUPS):
        x_center = xs[i]
        nc_mean = nc_by_group[group] if nc_by_group[group] > 0 else 1.0
        sset = set_scores_by_group[group]
        if len(sset) == 0:
            continue
        sv = sset.values / nc_mean if normalize else sset.values
        x_set = x_center + (0.30 if is_baseline(group) else -0.30)
        set_positions[group] = dict(zip(sset.index, zip(np.full_like(sv, x_set), sv)))

    # ---- 2. Population of large-benchmark models ----
    for i, group in enumerate(GROUPS):
        is_b = is_baseline(group)
        color = COLOR_BASELINE if is_b else COLOR_CSTIM
        x_center = xs[i]

        scores = get_large_scores(rsa, group, score_col)
        nc_mean = nc_by_group[group] if nc_by_group[group] > 0 else 1.0
        s_plot = scores.values / nc_mean if normalize else scores.values

        # IQR box
        q1, med, q3 = np.percentile(s_plot, [25, 50, 75])
        ax.add_patch(plt.Rectangle(
            (x_center - 0.16, q1), 0.32, q3 - q1,
            facecolor=color, alpha=0.20, edgecolor="none", zorder=2,
        ))

        # Jittered strip overlaid ON the box plot (centred on column),
        # so the box+median+points read as one unit and the selection-set
        # dots on the side don't compete for the same column-half.
        jitter = rng.uniform(-0.14, 0.14, len(s_plot))
        ax.scatter(x_center + jitter, s_plot, s=5, color=color,
                   alpha=0.55, linewidths=0, zorder=3)

        # Median
        ax.hlines(med, x_center - 0.18, x_center + 0.18,
                   colors=color, linewidth=2.0, zorder=4)

    # ---- 3. Light direct connectors (baseline ↔ all-models only) ----
    # With dots already adjacent, a thin straight line is enough to make the
    # pairing explicit without dominating the figure.
    if "vicco" in set_positions and "all_models" in set_positions:
        for m in set_positions["vicco"]:
            if m not in set_positions["all_models"]:
                continue
            x0, y0 = set_positions["vicco"][m]
            x1, y1 = set_positions["all_models"][m]
            ax.plot([x0, x1], [y0, y1], color=COLOR_SET,
                    linewidth=0.5, alpha=0.40, zorder=2.5)

    # ---- 4. Set-model dots (drawn last, on top of everything) ----
    for group, positions in set_positions.items():
        if not positions:
            continue
        xy = np.array(list(positions.values()))
        ax.scatter(xy[:, 0], xy[:, 1], s=24,
                   facecolors=COLOR_SET, edgecolors="white",
                   linewidths=0.9, zorder=6, marker="o")

    # --- Per-panel summary annotations (above each x-tick) ---
    if normalize:
        ax.set_ylim(0, 1.28)
    ymin, ymax = ax.get_ylim()
    # Three-line annotation per controversial column:
    #   line 1 (top, bold): spread ratio
    #   line 2 (mid, smaller): median Δ vs baseline + Wilcoxon stars (BH-FDR)
    #   line 3 (small, gray): median % of NC
    spread_y = ymax * 0.99
    test_y   = ymax * 0.93
    pct_y    = ymax * 0.87

    test_results = paired_wilcoxon_vs_baseline(rsa, score_col, normalize, nc_by_group)

    for i, group in enumerate(GROUPS):
        nc_mean_t = nc_by_group[group] if nc_by_group[group] > 0 else 1.0
        s_norm = get_large_scores(rsa, group, score_col).values / nc_mean_t
        pct = int(round(np.median(s_norm) * 100))

        if normalize and group != "vicco":
            spread = _median_pair_diff(s_norm) / mpd_vicco if mpd_vicco > 0 else np.nan
            spread_color = "#9A4500" if spread >= 1.0 else "#666666"
            ax.text(xs[i], spread_y, f"{spread:.2f}× spread",
                    ha="center", va="top",
                    fontsize=FONT["annotation"], fontweight="bold",
                    color=spread_color,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                               edgecolor="none", alpha=0.9))

            # Descriptive median Δ vs baseline (no inferential stars,
            # consistent with the paper's descriptive-only reporting stance).
            tr = test_results.get(group, {})
            mdelta = tr.get("median_delta", np.nan)
            if np.isfinite(mdelta):
                ax.text(xs[i], test_y,
                        f"Δ̃={mdelta:+.2f}",
                        ha="center", va="top",
                        fontsize=FONT["small"], color="#444")

            ax.text(xs[i], pct_y, f"{pct}% NC",
                    ha="center", va="top",
                    fontsize=FONT["small"] - 1, color="#888")
        else:
            ax.text(xs[i], spread_y, f"{pct}% NC",
                    ha="center", va="top",
                    fontsize=FONT["annotation"], fontweight="bold",
                    color="#444444",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                               edgecolor="none", alpha=0.9))

    # --- X-axis ---
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[g] for g in GROUPS])
    ax.set_xlim(-0.55, n - 0.4)

    # Vertical separator between baseline and controversial sets
    ax.axvline(0.5, color="#cccccc", linewidth=0.6, zorder=0)

    # --- Y-axis ---
    ax.set_ylabel(f"{row_label} / NC" if normalize else f"{row_label} ($r_s$)")
    if normalize:
        ax.axhline(1.0, color="#444444", linewidth=0.6, linestyle="--",
                   zorder=0, alpha=0.5)
        ax.text(-0.45, 1.04, "NC (max)", va="bottom",
                fontsize=FONT["small"], color="#444444", alpha=0.8)
    else:
        ax.set_ylim(bottom=0)

    add_panel_label(ax, panel_label, x=-0.05)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    SUPPLEMENTARY_DIR.mkdir(parents=True, exist_ok=True)
    SUPPLEMENTARY_PNG_DIR.mkdir(parents=True, exist_ok=True)
    rsa = load_large_scores()
    nc_df = load_nc()
    sets = load_set_scores()

    rows = [("wrsa_transfer", "mixed RSA", "a"), ("crsa", "fixed RSA", "b")]

    for normalize, suffix in [(True, "norm"), (False, "raw")]:
        fig, axes = plt.subplots(2, 1, figsize=(W_DOUBLE, 7.8))
        fig.subplots_adjust(left=0.07, right=0.98, top=0.96, bottom=0.13, hspace=0.30)

        for ax, (score_col, row_label, panel) in zip(axes, rows):
            draw_panel(ax, rsa, nc_df, sets,
                        score_col=score_col, row_label=row_label,
                        normalize=normalize, panel_label=panel)

        # Shared legend
        handles = [
            Line2D([0], [0], marker="o", linestyle="none",
                   markerfacecolor=COLOR_BASELINE, markeredgecolor="none",
                   markersize=4,
                   label="benchmark model on baseline stimuli"),
            Line2D([0], [0], marker="o", linestyle="none",
                   markerfacecolor=COLOR_CSTIM, markeredgecolor="none",
                   markersize=4,
                   label="benchmark model on controversial stimuli"),
            Line2D([0], [0], marker="o", linestyle="none",
                   markerfacecolor=COLOR_SET, markeredgecolor="white",
                   markeredgewidth=0.8, markersize=5,
                   label="model used during selection (highlighted)"),
            Line2D([0], [0], color=COLOR_SET, linewidth=0.6, alpha=0.7,
                   label="line connects same selection model: baseline ↔ all-models"),
        ]
        if not normalize:
            handles.append(Line2D([0], [0], linestyle="--", color="gray",
                                  linewidth=0.6, label="NC ± SEM"))
        # Place legend in the figure margin (below both panels), not inside data
        fig.legend(handles=handles, loc="lower center",
                    bbox_to_anchor=(0.5, 0.01), ncol=len(handles),
                    frameon=False, edgecolor="none",
                    fontsize=FONT["small"],
                    handletextpad=0.3, handlelength=1.2,
                    columnspacing=1.4)
        # Caption below the legend (descriptive)
        fig.text(
            0.5, -0.012,
            "Δ̃ = median paired difference (set − baseline) across benchmark "
            "models. Spread = median absolute pairwise model-score difference, "
            "normalised to the baseline column.",
            ha="center", va="bottom",
            fontsize=FONT["small"], color="#444", fontstyle="italic",
        )

        if normalize:
            pdf_path = FIGURES_DIR / "large_benchmark_gap_norm_improved.pdf"
            png_path = PNG_DIR / "large_benchmark_gap_norm_improved.png"
        else:
            pdf_path = SUPPLEMENTARY_DIR / "large_benchmark_gap_raw_improved.pdf"
            png_path = SUPPLEMENTARY_PNG_DIR / "large_benchmark_gap_raw_improved.png"
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved: large_benchmark_gap_{suffix}_improved.pdf/png")
        plt.close(fig)


if __name__ == "__main__":
    main()
