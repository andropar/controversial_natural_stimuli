#!/usr/bin/env python3
"""Summarize ROI-wise model spread and cstim/VICCO alignment drops."""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


THIS = Path(__file__).resolve()
RERUN_ROOT = THIS.parents[1]
SHARE_ROOT = THIS.parents[3]
HELPERS = SHARE_ROOT / "src"
sys.path.insert(0, str(HELPERS))

from cstims.paper.style_improved import (  # noqa: E402
    DPI,
    FONT,
    MODEL_SET_DISPLAY,
    W_DOUBLE,
    add_panel_label,
    apply_style,
    model_set_color,
)

from plot_brain_alignment_improved_with_shared import (  # noqa: E402
    DEFAULT_BETWEEN_NC,
    DEFAULT_RDM_NC,
    FIGURES_DIR,
    PANEL_ORDER,
    RESULTS_DIR,
    load_noise_ceilings,
    load_scores,
    normalize_scores,
)


apply_style()

DEFAULT_OUT = FIGURES_DIR / "roi_spread_alignment_drop_summary.pdf"
DEFAULT_PNG = FIGURES_DIR / "roi_spread_alignment_drop_summary.png"
DEFAULT_OUT_DIR = FIGURES_DIR / "roi_spread_alignment_drop_by_model_set"
DEFAULT_SUMMARY = RESULTS_DIR / "roi_spread_alignment_drop_summary.csv"

ROI_ORDER = [
    "EVC",
    "ventral",
    "lateral",
    "dorsal",
    "general",
    "EBA",
    "FFA",
    "PPA",
    "LOTC",
    "floc_all",
    "ventral_lateral_floc",
]

ROI_LABELS = {
    "EVC": "EVC",
    "ventral": "Ventral",
    "lateral": "Lateral",
    "dorsal": "Dorsal",
    "general": "General",
    "EBA": "EBA",
    "FFA": "FFA",
    "PPA": "PPA",
    "LOTC": "LOTC",
    "floc_all": "fLOC all",
    "ventral_lateral_floc": "V+L+fLOC",
}

METHOD_LABELS = {
    "mRSA": "Mixed RSA",
    "fRSA": "Fixed RSA",
}


def median_pairwise_distance(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    return float(np.median([abs(a - b) for a, b in combinations(values, 2)]))


def subject_scores(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "roi",
        "method",
        "model_set",
        "model",
        "display_name",
        "subject",
        "stimulus_type",
    ]
    return df.groupby(group_cols, as_index=False).agg(score=("score", "mean"))


def compute_summary(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scores = subject_scores(scores)
    for (roi, method, model_set), block in scores.groupby(["roi", "method", "model_set"]):
        means = (
            block.groupby(["model", "stimulus_type"], as_index=False)
            .agg(score=("score", "mean"))
        )
        cstim = means[means["stimulus_type"].eq("controversial")]["score"].to_numpy()
        vicco = means[means["stimulus_type"].eq("vicco")]["score"].to_numpy()
        spread_cstim = median_pairwise_distance(cstim)
        spread_vicco = median_pairwise_distance(vicco)

        paired = (
            block.pivot_table(
                index=["subject", "model"],
                columns="stimulus_type",
                values="score",
                aggfunc="mean",
            )
            .dropna(subset=["controversial", "vicco"])
            .reset_index()
        )
        drop = paired["vicco"].to_numpy() - paired["controversial"].to_numpy()
        rows.append(
            {
                "roi": roi,
                "method": method,
                "model_set": model_set,
                "n_models": int(block["model"].nunique()),
                "n_subject_model_pairs": int(len(paired)),
                "spread_cstim": spread_cstim,
                "spread_vicco": spread_vicco,
                "spread_ratio_cstim_over_vicco": (
                    spread_cstim / spread_vicco
                    if np.isfinite(spread_cstim) and np.isfinite(spread_vicco) and spread_vicco > 0
                    else np.nan
                ),
                "median_alignment_drop_vicco_minus_cstim": float(np.nanmedian(drop)),
                "mean_alignment_drop_vicco_minus_cstim": float(np.nanmean(drop)),
            }
        )
    return pd.DataFrame(rows)


def load_normalized_scores(rdm_nc: Path, between_nc: Path) -> pd.DataFrame:
    scores = load_scores()
    rdm, between = load_noise_ceilings(rdm_nc, between_nc)
    return normalize_scores(scores, rdm, between)


def ordered_rois(summary: pd.DataFrame) -> list[str]:
    present = list(summary["roi"].unique())
    ordered = [r for r in ROI_ORDER if r in present]
    ordered.extend(sorted(set(present) - set(ordered)))
    return ordered


def log_limits(values: pd.Series | np.ndarray) -> tuple[float, float, np.ndarray]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    ticks = np.array([0.25, 0.5, 1, 2, 4, 8, 16, 32], dtype=float)
    if len(vals) == 0:
        return 0.5, 2.0, np.array([0.5, 1, 2], dtype=float)
    lower_candidates = ticks[ticks <= vals.min()]
    upper_candidates = ticks[ticks >= vals.max()]
    lo = float(lower_candidates[-1]) if len(lower_candidates) else float(vals.min() * 0.8)
    hi = float(upper_candidates[0]) if len(upper_candidates) else float(vals.max() * 1.2)
    visible_ticks = ticks[(ticks >= lo) & (ticks <= hi)]
    return lo, hi, visible_ticks


def linear_limits(values: pd.Series | np.ndarray, refline: float) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return refline - 0.1, refline + 0.1
    lo = min(float(vals.min()), refline)
    hi = max(float(vals.max()), refline)
    pad = max((hi - lo) * 0.12, 0.02)
    return lo - pad, hi + pad


def draw_bar_panel(
    ax,
    summary: pd.DataFrame,
    rois: list[str],
    model_set: str,
    method: str,
    metric: str,
    ylabel: str,
    refline: float,
    panel_label: str,
    log_scale: bool = False,
    ylim: tuple[float, float] | None = None,
    yticks: np.ndarray | None = None,
):
    block = (
        summary[(summary["method"].eq(method)) & (summary["model_set"].eq(model_set))]
        .set_index("roi")
        .reindex(rois)
    )
    values = block[metric].to_numpy(dtype=float)
    color = model_set_color(model_set)
    x = np.arange(len(rois))

    if log_scale:
        for xi, yi in zip(x, values):
            if not np.isfinite(yi) or yi <= 0:
                continue
            bottom = min(refline, yi)
            height = abs(yi - refline)
            ax.bar(
                xi,
                height,
                bottom=bottom,
                width=0.72,
                color=color,
                edgecolor=color,
                alpha=0.82,
                linewidth=0.8,
                zorder=3,
            )
        ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(*ylim)
        if yticks is not None and len(yticks):
            ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f"{y:g}"))
    else:
        ax.bar(
            x,
            values,
            width=0.72,
            color=color,
            edgecolor=color,
            alpha=0.82,
            linewidth=0.8,
            zorder=3,
        )
        if ylim is not None:
            ax.set_ylim(*ylim)

    ax.axhline(refline, color="#777777", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels([ROI_LABELS.get(r, r) for r in rois], rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(METHOD_LABELS[method], pad=8)
    ax.grid(axis="y", alpha=0.25)
    add_panel_label(ax, panel_label, x=-0.05, y=1.04)


def panel_limits(summary: pd.DataFrame) -> dict[tuple[str, str], tuple[tuple[float, float], np.ndarray | None]]:
    limits = {}
    ratio = "spread_ratio_cstim_over_vicco"
    drop = "median_alignment_drop_vicco_minus_cstim"
    for method in ("mRSA", "fRSA"):
        method_df = summary[summary["method"].eq(method)]
        lo, hi, ticks = log_limits(method_df[ratio])
        limits[(method, ratio)] = ((lo, hi), ticks)
        limits[(method, drop)] = (linear_limits(method_df[drop], refline=0.0), None)
    return limits


def plot_model_set_summary(
    summary: pd.DataFrame,
    model_set: str,
    out_pdf: Path,
    out_png: Path,
    limits: dict[tuple[str, str], tuple[tuple[float, float], np.ndarray | None]],
):
    rois = ordered_rois(summary)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(W_DOUBLE, 8.2),
        sharex=True,
        constrained_layout=True,
    )
    ratio = "spread_ratio_cstim_over_vicco"
    drop = "median_alignment_drop_vicco_minus_cstim"

    draw_bar_panel(
        axes[0, 0],
        summary,
        rois,
        model_set,
        "mRSA",
        ratio,
        "Spread ratio\ncstim / VICCO",
        1.0,
        "a",
        log_scale=True,
        ylim=limits[("mRSA", ratio)][0],
        yticks=limits[("mRSA", ratio)][1],
    )
    draw_bar_panel(
        axes[0, 1],
        summary,
        rois,
        model_set,
        "fRSA",
        ratio,
        "Spread ratio\ncstim / VICCO",
        1.0,
        "b",
        log_scale=True,
        ylim=limits[("fRSA", ratio)][0],
        yticks=limits[("fRSA", ratio)][1],
    )
    draw_bar_panel(
        axes[1, 0],
        summary,
        rois,
        model_set,
        "mRSA",
        drop,
        "Median alignment drop\nVICCO - cstim",
        0.0,
        "c",
        ylim=limits[("mRSA", drop)][0],
    )
    draw_bar_panel(
        axes[1, 1],
        summary,
        rois,
        model_set,
        "fRSA",
        drop,
        "Median alignment drop\nVICCO - cstim",
        0.0,
        "d",
        ylim=limits[("fRSA", drop)][0],
    )

    fig.suptitle(
        MODEL_SET_DISPLAY.get(model_set, model_set),
        fontsize=FONT["title"] + 2,
        fontweight="bold",
        color=model_set_color(model_set),
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


def plot_summary(summary: pd.DataFrame, out_pdf: Path, out_png: Path, out_dir: Path):
    limits = panel_limits(summary)
    # Preserve the previous top-level output path as the all-models bar summary.
    plot_model_set_summary(summary, "all_models", out_pdf, out_png, limits)
    for model_set in PANEL_ORDER:
        plot_model_set_summary(
            summary,
            model_set,
            out_dir / f"roi_spread_alignment_drop_{model_set}.pdf",
            out_dir / f"roi_spread_alignment_drop_{model_set}.png",
            limits,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdm-noise-ceilings", type=Path, default=DEFAULT_RDM_NC)
    parser.add_argument("--between-noise-ceilings", type=Path, default=DEFAULT_BETWEEN_NC)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    scores = load_normalized_scores(args.rdm_noise_ceilings, args.between_noise_ceilings)
    summary = compute_summary(scores)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_out, index=False)
    print(f"Wrote {len(summary)} rows -> {args.summary_out}")
    plot_summary(summary, args.out, args.png, args.out_dir)


if __name__ == "__main__":
    main()
