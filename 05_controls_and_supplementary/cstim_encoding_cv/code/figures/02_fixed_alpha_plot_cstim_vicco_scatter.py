#!/usr/bin/env python3
"""Plot 1:1 target-adaptation mixed-RSA comparisons."""

from __future__ import annotations

import argparse

import _paths  # noqa: F401
from _paths import FIGURES_DIR, RESULTS_DIR

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from cstims import constants, paths
from cstims.target_adaptation import SHORT_MODEL_NAMES
from cstims.paper.style_improved import (
    COLOR_CSTIM,
    DPI,
    FONT,
    W_DOUBLE,
    apply_style,
)


apply_style()

FIXED_RESULTS_DIR = RESULTS_DIR / "02_fixed_alpha"
FIGURE_DATA_DIR = FIXED_RESULTS_DIR / "figure_data"
OUTPUT_FIGURES_DIR = FIGURES_DIR / "02_fixed_alpha"
OUTPUT_PNG_DIR = OUTPUT_FIGURES_DIR / "png"
SCORE_CSV = FIXED_RESULTS_DIR / "scores.csv"
PANEL_SPECS = [
    (
        "original_gap",
        "Original encoding",
        "original_vicco",
        "original_cstim",
        "Original Vicco mRSA",
        "Original CSTIM mRSA",
    ),
    (
        "cstim_adapted_gap",
        "CSTIM target-adapted",
        "adapted_vicco_heldout",
        "adapted_cstim_loso",
        "Held-out Vicco mRSA",
        "CSTIM LOSO mRSA",
    ),
    (
        "cstim_shift",
        "CSTIM shift",
        "original_cstim",
        "adapted_cstim_loso",
        "Original CSTIM mRSA",
        "CSTIM LOSO mRSA",
    ),
    (
        "vicco_loso_shift",
        "Vicco LOSO shift",
        "original_vicco",
        "adapted_vicco_loso",
        "Original Vicco mRSA",
        "Vicco LOSO mRSA",
    ),
]
FONT_TINY = FONT.get("tiny", FONT.get("small", FONT["tick"]))


def load_wide(weight: float, model_set: str) -> pd.DataFrame:
    df = pd.read_csv(SCORE_CSV)
    df["target_weight"] = df["target_weight"].astype(float)
    endpoint = df[np.isclose(df["target_weight"], weight)].copy()
    keys = ["subject", "model", "display_name", "selected_layer", "model_set"]
    cstim = endpoint[
        endpoint["model_set"].eq(model_set) & endpoint["eval_target"].eq("cstim_loso")
    ][keys + ["mrsa_loso", "original_best_shared_mrsa"]].rename(
        columns={
            "mrsa_loso": "adapted_cstim_loso",
            "original_best_shared_mrsa": "original_cstim",
        }
    )
    heldout = endpoint[
        endpoint["model_set"].eq(model_set) & endpoint["eval_target"].eq("vicco_heldout")
    ][keys + ["mrsa_loso", "original_best_shared_mrsa"]].rename(
        columns={
            "mrsa_loso": "adapted_vicco_heldout",
            "original_best_shared_mrsa": "original_vicco",
        }
    )
    vicco_keys = ["subject", "model", "display_name", "selected_layer"]
    vicco_loso = endpoint[endpoint["eval_target"].eq("vicco_loso")][
        vicco_keys + ["mrsa_loso"]
    ].rename(columns={"mrsa_loso": "adapted_vicco_loso"})

    wide = cstim.merge(heldout, on=keys, how="inner", validate="one_to_one")
    wide = wide.merge(vicco_loso, on=vicco_keys, how="inner", validate="many_to_one")
    if wide.empty:
        raise RuntimeError(f"No matched target-adaptation rows for {model_set}, w={weight:g}")
    wide = wide.sort_values(["model", "subject"]).reset_index(drop=True)
    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = FIGURE_DATA_DIR / f"cstim_vicco_scatter_{model_set}_w{weight:g}.csv"
    wide.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    return wide


def finite_limits(df: pd.DataFrame) -> tuple[float, float]:
    vals = []
    for _name, _title, xcol, ycol, _xlabel, _ylabel in PANEL_SPECS:
        vals.extend(df[xcol].to_numpy(dtype=float))
        vals.extend(df[ycol].to_numpy(dtype=float))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    lo = float(vals.min())
    hi = float(vals.max())
    pad = max((hi - lo) * 0.07, 0.015)
    return lo - pad, hi + pad


def draw_panel(
    ax,
    df: pd.DataFrame,
    spec: tuple[str, str, str, str, str, str],
    *,
    label_models: bool,
) -> None:
    _name, title, xcol, ycol, xlabel, ylabel = spec
    points = df[[xcol, ycol]].to_numpy(dtype=float)
    finite = np.isfinite(points).all(axis=1)
    sub = df.loc[finite].copy()
    ax.scatter(
        sub[xcol],
        sub[ycol],
        s=16,
        color="0.35",
        alpha=0.22,
        linewidths=0,
        zorder=2,
        label="Subject-model",
    )
    means = (
        sub.groupby(["model", "display_name"], as_index=False)[[xcol, ycol]]
        .mean()
        .sort_values(ycol, ascending=False)
    )
    ax.scatter(
        means[xcol],
        means[ycol],
        s=42,
        facecolor=COLOR_CSTIM,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.92,
        zorder=4,
        label="Model mean",
    )
    if label_models:
        for row in means.itertuples(index=False):
            label = SHORT_MODEL_NAMES.get(
                row.model,
                constants.MODEL_DISPLAY_NAMES.get(row.model, row.model),
            )
            ax.text(
                getattr(row, xcol),
                getattr(row, ycol),
                f" {label}",
                fontsize=FONT_TINY,
                color="0.18",
                alpha=0.76,
                ha="left",
                va="center",
                zorder=5,
            )
    rho = stats.spearmanr(sub[xcol], sub[ycol]).statistic if len(sub) > 2 else np.nan
    delta = float(np.nanmean(sub[ycol] - sub[xcol])) if len(sub) else np.nan
    ax.text(
        0.03,
        0.97,
        f"rho={rho:.2f}\nmean delta={delta:+.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_TINY,
        color="0.25",
    )
    ax.set_title(title, fontsize=FONT["small"], fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22, linewidth=0.45)


def plot_comparison(
    df: pd.DataFrame,
    *,
    model_set: str,
    weight: float,
    label_models: bool,
) -> None:
    lo, hi = finite_limits(df)
    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE * 0.72, 7.4), constrained_layout=True)
    for ax, spec in zip(axes.ravel(), PANEL_SPECS):
        draw_panel(ax, df, spec, label_models=label_models)
        ax.plot([lo, hi], [lo, hi], color="0.55", linestyle="--", linewidth=0.8, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle(
        f"Target-adaptation mixed RSA comparisons, weight {weight:g}",
        fontsize=FONT["title"],
        fontweight="bold",
    )
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=FONT["small"])
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PNG_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_labeled" if label_models else ""
    stem = f"target_adaptation_fixed_weight_cstim_vicco_scatter_{model_set}_w{weight:g}{suffix}_cached"
    pdf = OUTPUT_FIGURES_DIR / f"{stem}.pdf"
    png = OUTPUT_PNG_DIR / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=DPI)
    print(f"Saved {pdf}")
    print(f"Saved {png}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=float, default=2.0)
    parser.add_argument("--model-set", default="all_models")
    parser.add_argument("--label-models", action="store_true")
    args = parser.parse_args()

    wide = load_wide(args.weight, args.model_set)
    plot_comparison(
        wide,
        model_set=args.model_set,
        weight=args.weight,
        label_models=args.label_models,
    )


if __name__ == "__main__":
    main()
