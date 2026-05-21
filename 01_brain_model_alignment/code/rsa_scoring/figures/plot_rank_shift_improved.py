#!/usr/bin/env python3
"""
Improved rank-shift figure (and rank_rho_summary).

Fixes vs. original:
- Okabe-Ito blue + sky_blue palette (mixed RSA = blue, fixed RSA = sky_blue
  with hatching).
- The 3 ρ values per panel are visualised as a small inline horizontal-bar
  inset instead of a packed text block.
- Uses panel labels via add_panel_label.
- rank_rho_summary y-limit trimmed to [-0.05, 1.05].
"""
from __future__ import annotations

import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
_SHARE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))
sys.path.insert(0, str(_PAPER / "figures"))
sys.path.insert(0, str(_SHARE_ROOT / "shared" / "code" / "paper_helpers"))
sys.path.insert(0, str(_SHARE_ROOT / "shared" / "code" / "paper_helpers" / "figures"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import config
from style_improved import (
    apply_style, FONT, DPI, W_DOUBLE,
    OKABE_ITO, add_panel_label, MODEL_SET_DISPLAY,
)

apply_style()

# Reuse logic from existing script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_rank_shift import (
    load_scores, compute_deltas, cross_subject_rho, base_vs_cstim_rho,
    SHORT_DISPLAY_NAMES, TITLE_MAP, MODEL_SETS, DN, _MS_SHORT, compute_rho_summary,
    per_subject_ranks, pairwise_rhos,
)

STAGE_DIR = Path(__file__).resolve().parents[3]
SHARE_ROOT = STAGE_DIR.parent
FIGURES_DIR = STAGE_DIR / "figures" / "rsa_scores"
PNG_DIR = FIGURES_DIR / "png"
SUPPLEMENTARY_DIR = FIGURES_DIR / "supplementary"
SUPPLEMENTARY_PNG_DIR = SUPPLEMENTARY_DIR / "png"
RANK_CORR_PATH = SHARE_ROOT / "03_alignment_inference" / "data" / "rank_correlations.csv"

COLOR_MRSA = OKABE_ITO["blue"]      # #0072B2
COLOR_FRSA = OKABE_ITO["sky_blue"]  # #56B4E9


def _canonical_cross_set_rhos(rank_corr: pd.DataFrame, model_set: str):
    """Return source-of-truth mean cross-set rhos as (mixed, fixed)."""
    if rank_corr is None or rank_corr.empty:
        return np.nan, np.nan
    rows = rank_corr[
        (rank_corr["aggregation"] == "mean_across_subjects")
        & (rank_corr["model_set"] == model_set)
    ]
    mixed = rows[rows["metric"] == "mixed_RSA"]["rho_base_to_controversial"]
    fixed = rows[rows["metric"] == "fixed_RSA"]["rho_base_to_controversial"]
    return (
        float(mixed.iloc[0]) if len(mixed) else np.nan,
        float(fixed.iloc[0]) if len(fixed) else np.nan,
    )


def _format_rho_annotation(ax, model_set, rank_corr, base_m, cstim_m, base_f, cstim_f):
    """Render the cross-set Spearman ρ (ρ_{b↔c}) as a compact takeaway label
    in the panel header. This is the rank-stability number — close to 1 means
    the controversial-vs-baseline reordering is local, not a hierarchy reshuffle.
    Within-set ρ_base / ρ_cstim live in the companion rank_rho_summary figure.
    """
    def _safe(fn, *a):
        try:
            return float(fn(*a))
        except Exception:
            return np.nan

    rbc_m, rbc_f = _canonical_cross_set_rhos(rank_corr, model_set)
    if np.isnan(rbc_m) and not base_m.empty and not cstim_m.empty:
        rbc_m = _safe(base_vs_cstim_rho, base_m, cstim_m)
    if np.isnan(rbc_f) and not base_f.empty and not cstim_f.empty:
        rbc_f = _safe(base_vs_cstim_rho, base_f, cstim_f)

    pieces = []
    if not np.isnan(rbc_m):
        pieces.append(f"{rbc_m:.2f}")
    if not np.isnan(rbc_f):
        pieces.append(f"{rbc_f:.2f}")
    if not pieces:
        return
    # Compact: "ρ_{b↔c} = 0.66 / 0.43  (m / f)"
    label = (r"$\rho_{b\leftrightarrow c}=\,$" + " / ".join(pieces)
             + r"  $\;$(m / f)")
    ax.text(0.5, 1.005, label,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=FONT["annotation"], color="#222", fontweight="bold",
            zorder=10)


def draw_panel_improved(ax, df_mrsa, df_frsa, model_set, rank_corr, title=None,
                        panel_label=None, show_ylabel=True, show_legend=False,
                        use_short_names=False):
    mean_m, std_m, base_m, cstim_m = compute_deltas(df_mrsa, model_set, "wrsa_transfer")
    mean_f, std_f, base_f, cstim_f = compute_deltas(df_frsa, model_set, "crsa")

    models = sorted(
        set(mean_m) | set(mean_f),
        key=lambda m: mean_m.get(m, mean_f.get(m, 0)),
        reverse=True,
    )
    if not models:
        ax.set_visible(False); return

    n = len(models)
    x = np.arange(n)
    bar_w = 0.35
    offset = bar_w / 2 + 0.02

    for i, m in enumerate(models):
        if m in mean_m:
            ax.bar(x[i] - offset, mean_m[m], width=bar_w,
                   yerr=std_m.get(m, 0),
                   color=COLOR_MRSA, alpha=0.85,
                   error_kw=dict(linewidth=0.7, capsize=2, ecolor="#444"),
                   zorder=3)
        if m in mean_f:
            ax.bar(x[i] + offset, mean_f[m], width=bar_w,
                   yerr=std_f.get(m, 0),
                   color=COLOR_FRSA, alpha=0.85, hatch="////",
                   edgecolor=COLOR_FRSA,
                   error_kw=dict(linewidth=0.7, capsize=2, ecolor="#444"),
                   zorder=3)

    ax.axhline(0, color="#333", linewidth=0.7, zorder=4)
    ax.set_xticks(x)
    name_lookup = SHORT_DISPLAY_NAMES if use_short_names else DN
    ax.set_xticklabels([name_lookup.get(m, DN.get(m, m)) for m in models],
                       rotation=45, ha="right")
    ax.set_xlim(-0.7, n - 0.3)
    if show_ylabel:
        ax.set_ylabel(r"$\Delta$ rank  (+ = improved on cstim)")
    ax.grid(axis="y", alpha=0.22, linewidth=0.4, zorder=0)

    if title:
        # Push title above the rank-stability strip.
        ax.set_title(title, fontweight="bold", y=1.07)
    if panel_label:
        add_panel_label(ax, panel_label, x=-0.06, y=1.12)

    _format_rho_annotation(ax, model_set, rank_corr, base_m, cstim_m, base_f, cstim_f)

    if show_legend:
        handles = [
            mpatches.Patch(facecolor=COLOR_MRSA, alpha=0.85, label="mixed RSA"),
            mpatches.Patch(facecolor=COLOR_FRSA, alpha=0.85, hatch="////",
                            edgecolor=COLOR_FRSA, label="fixed RSA"),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=False,
                  fontsize=FONT["legend"], ncol=2, handlelength=1.0)


def _subject_pair_values(df, score_col, model_set, stimulus_type):
    ranks = per_subject_ranks(df, model_set, score_col, stimulus_type)
    return pairwise_rhos(ranks)


def draw_rho_summary_improved(df_mrsa, df_frsa):
    """Companion 2-panel ρ_base / ρ_cstim figure."""
    rho_m, _ = compute_rho_summary(df_mrsa, "wrsa_transfer")
    rho_f, _ = compute_rho_summary(df_frsa, "crsa")

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE * 0.70, 3.6), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.78, bottom=0.18, wspace=0.08)

    color_base  = OKABE_ITO["orange"]
    color_cstim = OKABE_ITO["bluish_green"]

    for ax, rho_df, df_source, score_col, method_label, panel_label in zip(
        axes, [rho_m, rho_f], [df_mrsa, df_frsa], ["wrsa_transfer", "crsa"],
        ["mixed RSA", "fixed RSA"], ["a", "b"]
    ):
        if rho_df.empty:
            ax.set_visible(False); continue

        n_sets = len(rho_df)
        x = np.arange(n_sets)
        bar_w = 0.32

        for off, key, color, label in [
            (-0.5 * (bar_w + 0.04), "base",  color_base,  r"$\rho_\mathrm{base}$"),
            ( 0.5 * (bar_w + 0.04), "cstim", color_cstim, r"$\rho_\mathrm{cstim}$"),
        ]:
            vals = rho_df[f"rho_{key}"].values
            ax.bar(x + off, vals, width=bar_w, color=color, alpha=0.85,
                   label=label, zorder=3)
            stim = "vicco" if key == "base" else "controversial"
            for xi, v, ms in zip(x + off, vals, rho_df["model_set"]):
                if not np.isnan(v):
                    pw = _subject_pair_values(df_source, score_col, ms, stim)
                    if len(pw):
                        jitter = np.linspace(-0.07, 0.07, len(pw))
                        sem = np.std(pw, ddof=1) / np.sqrt(len(pw)) if len(pw) > 1 else 0.0
                        ax.errorbar(
                            xi, v, yerr=1.96 * sem, fmt="none",
                            ecolor="#222", elinewidth=0.7, capsize=2,
                            zorder=5,
                        )
                        ax.scatter(
                            xi + jitter, pw, s=10,
                            facecolor="white", edgecolor="#444",
                            linewidth=0.45, alpha=0.9, zorder=6,
                        )
                    ax.text(xi, v + 0.025, f"{v:.2f}",
                            ha="center", va="bottom",
                            fontsize=FONT["small"], color="#222")

        ax.axhline(0, color="#444", linewidth=0.6, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([_MS_SHORT.get(ms, ms) for ms in rho_df["model_set"]])
        ax.set_ylim(-0.75, 1.15)
        ax.set_title(method_label, fontweight="bold", pad=4)
        ax.grid(axis="y", alpha=0.22, linewidth=0.4, zorder=0)
        add_panel_label(ax, panel_label, x=-0.07, y=1.05)
        if panel_label == "a":
            ax.set_ylabel("cross-subject Spearman ρ")

    handles = [
        mpatches.Patch(facecolor=color_base, alpha=0.85, label=r"$\rho_\mathrm{base}$"),
        mpatches.Patch(facecolor=color_cstim, alpha=0.85, label=r"$\rho_\mathrm{cstim}$"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.98),
               frameon=False, fontsize=FONT["legend"], ncol=2,
               handlelength=1.0, columnspacing=1.0)

    return fig


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    SUPPLEMENTARY_DIR.mkdir(parents=True, exist_ok=True)
    SUPPLEMENTARY_PNG_DIR.mkdir(parents=True, exist_ok=True)

    df_mrsa = load_scores("wrsa_transfer")
    df_frsa = load_scores("crsa")
    rank_corr = pd.read_csv(RANK_CORR_PATH) if RANK_CORR_PATH.exists() else pd.DataFrame()
    if df_mrsa.empty or df_frsa.empty:
        print("No data found"); return

    # ---- rank_shift figure ----
    fig = plt.figure(figsize=(W_DOUBLE, 7.6))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.85],
                              wspace=0.12, left=0.06, right=0.97,
                              top=0.88, bottom=0.18)

    ax_all = fig.add_subplot(outer[0])
    draw_panel_improved(ax_all, df_mrsa, df_frsa, "all_models", rank_corr,
                        title=TITLE_MAP["all_models"], panel_label="a",
                        show_ylabel=True, show_legend=True, use_short_names=True)

    gs_right = outer[1].subgridspec(2, 2, wspace=0.42, hspace=0.95)
    small_sets = ["sota", "training_objective", "architecture", "dataset"]
    for idx, ms in enumerate(small_sets):
        r, c = divmod(idx, 2)
        ax = fig.add_subplot(gs_right[r, c])
        draw_panel_improved(ax, df_mrsa, df_frsa, ms, rank_corr,
                            title=TITLE_MAP[ms],
                            panel_label="bcde"[idx],
                            show_ylabel=(c == 0))

    out_pdf = FIGURES_DIR / "rank_shift_improved.pdf"
    out_png = PNG_DIR / "rank_shift_improved.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=DPI)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    plt.close(fig)

    # ---- rank_rho_summary figure ----
    fig2 = draw_rho_summary_improved(df_mrsa, df_frsa)
    out_pdf = SUPPLEMENTARY_DIR / "rank_rho_summary_improved.pdf"
    out_png = SUPPLEMENTARY_PNG_DIR / "rank_rho_summary_improved.png"
    fig2.savefig(out_pdf)
    fig2.savefig(out_png, dpi=DPI)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
