#!/usr/bin/env python3
"""Plot weighted CSTIM encoding-CV trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import _paths  # noqa: F401
from _paths import FIGURES_DIR, PNG_DIR, RESULTS_DIR, SHARE_ROOT

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cstims.paper import config
from cstims.paper.style_improved import (
    COLOR_BASELINE,
    COLOR_CSTIM,
    DPI,
    FONT,
    W_DOUBLE,
    apply_style,
)


apply_style()

SCORE_CSV = RESULTS_DIR / "cstim_loso_weighted_scores.csv"
AUDIT_CSV = RESULTS_DIR / "cached_selection_audit.csv"
PANEL_ORDER = ["all_models", "sota", "training_objective", "architecture", "dataset"]
TITLE = {
    "training_objective": "Train. Objective",
    "sota": "State of the Art",
    "architecture": "Architecture",
    "dataset": "Dataset",
    "all_models": "All Models",
}
MODEL_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "*"]
FONT_TINY = FONT.get("tiny", FONT.get("small", FONT["tick"]))


def load_scores() -> pd.DataFrame:
    df = pd.read_csv(SCORE_CSV)
    df["cstim_weight"] = df["cstim_weight"].astype(float)
    if "eval_target" not in df.columns:
        df["eval_target"] = "cstim_loso"
    return df


def mean_ci(vals: np.ndarray) -> tuple[float, float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    mean = float(vals.mean())
    if len(vals) <= 1:
        return mean, mean, mean
    sem = float(vals.std(ddof=1) / np.sqrt(len(vals)))
    return mean, mean - 1.96 * sem, mean + 1.96 * sem


def set_summary(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    for (model_set, weight), block in df.groupby(["model_set", "cstim_weight"]):
        mean, lo, hi = mean_ci(block[value_col].to_numpy())
        rows.append(
            {
                "model_set": model_set,
                "cstim_weight": weight,
                "mean": mean,
                "lo": lo,
                "hi": hi,
                "n": len(block),
                "n_models": block["model"].nunique(),
                "n_subjects": block["subject"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def x_positions(weights: list[float]) -> dict[float, float]:
    ordered = sorted(weights)
    return {w: i for i, w in enumerate(ordered)}


def draw_trajectory_panel(
    ax,
    df: pd.DataFrame,
    model_set: str,
    *,
    value_col: str,
    ylabel: str | None,
    show_legend: bool = False,
) -> None:
    sub = df[df["model_set"].eq(model_set)].copy()
    if sub.empty:
        ax.set_visible(False)
        return
    weights = sorted(sub["cstim_weight"].unique())
    xpos = x_positions(weights)
    sub["x"] = sub["cstim_weight"].map(xpos)

    for i, ((subject, model), block) in enumerate(sub.groupby(["subject", "model"])):
        block = block.sort_values("cstim_weight")
        ax.plot(
            block["x"],
            block[value_col],
            color="0.55",
            linewidth=0.45,
            alpha=0.20,
            zorder=1,
        )

    for i, (model, block) in enumerate(sub.groupby("model")):
        means = (
            block.groupby("cstim_weight", as_index=False)[value_col]
            .mean()
            .sort_values("cstim_weight")
        )
        ax.plot(
            means["cstim_weight"].map(xpos),
            means[value_col],
            marker=MODEL_MARKERS[i % len(MODEL_MARKERS)],
            markersize=2.8,
            linewidth=0.7,
            alpha=0.52,
            label="_nolegend_",
            zorder=2,
        )

    summ = set_summary(sub, value_col).sort_values("cstim_weight")
    x = summ["cstim_weight"].map(xpos).to_numpy(dtype=float)
    ax.fill_between(x, summ["lo"], summ["hi"], color=COLOR_CSTIM, alpha=0.13, linewidth=0)
    ax.plot(
        x,
        summ["mean"],
        color=COLOR_CSTIM,
        marker="o",
        markersize=4,
        linewidth=1.5,
        label="Mean",
        zorder=5,
    )

    if value_col == "mrsa_loso":
        ref = sub.dropna(subset=["original_best_shared_mrsa"])
        if not ref.empty:
            ref_vals = (
                ref.drop_duplicates(["subject", "model", "model_set"])
                ["original_best_shared_mrsa"]
                .to_numpy(dtype=float)
            )
            ref_mean = float(np.nanmean(ref_vals))
            ax.axhline(
                ref_mean,
                color=COLOR_BASELINE,
                linestyle="--",
                linewidth=1.0,
                alpha=0.85,
                label="Original best-shared",
                zorder=0,
            )
    else:
        ax.axhline(0, color="0.35", linestyle="--", linewidth=0.8, alpha=0.7, zorder=0)

    ax.set_title(TITLE[model_set], fontsize=FONT["small"], fontweight="bold")
    ax.set_xticks([xpos[w] for w in weights])
    ax.set_xticklabels([f"{w:g}" for w in weights], rotation=0)
    ax.set_xlabel("CSTIM sample weight")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.22, linewidth=0.45)
    ax.text(
        0.02,
        0.96,
        f"n={sub['model'].nunique()} models, {sub['subject'].nunique()} subjects",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_TINY,
        color="0.25",
    )
    if show_legend:
        ax.legend(
            loc="lower right",
            fontsize=FONT_TINY,
            frameon=True,
            framealpha=0.92,
            edgecolor="none",
            ncol=1,
        )


def plot_grid(df: pd.DataFrame, *, value_col: str, out_stem: str, ylabel: str) -> None:
    fig, axes = plt.subplots(
        1,
        len(PANEL_ORDER),
        figsize=(W_DOUBLE, 3.35),
        sharey=False,
        constrained_layout=True,
    )
    for i, (ax, model_set) in enumerate(zip(axes, PANEL_ORDER)):
        draw_trajectory_panel(
            ax,
            df,
            model_set,
            value_col=value_col,
            ylabel=ylabel if i == 0 else None,
            show_legend=(i == 0),
        )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES_DIR / f"{out_stem}.pdf"
    png = PNG_DIR / f"{out_stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=DPI)
    print(f"Saved {pdf}")
    print(f"Saved {png}")
    plt.close(fig)


def plot_endpoint(df: pd.DataFrame, endpoint_weight: float) -> None:
    rows = []
    for model_set, block in df.groupby("model_set"):
        endpoint = block[np.isclose(block["cstim_weight"], endpoint_weight)]
        if endpoint.empty:
            continue
        original = endpoint.drop_duplicates(["subject", "model", "model_set"])[
            "original_best_shared_mrsa"
        ].to_numpy(dtype=float)
        loso = endpoint["mrsa_loso"].to_numpy(dtype=float)
        for label, vals in [
            ("Original best-shared", original),
            (f"CSTIM LOSO w={endpoint_weight:g}", loso),
        ]:
            mean, lo, hi = mean_ci(vals)
            rows.append(
                {
                    "model_set": model_set,
                    "condition": label,
                    "mean": mean,
                    "lo": lo,
                    "hi": hi,
                    "n": len(vals),
                }
            )
    summ = pd.DataFrame(rows)
    x = np.arange(len(PANEL_ORDER))
    width = 0.34
    fig, ax = plt.subplots(figsize=(W_DOUBLE * 0.62, 3.4), constrained_layout=True)
    for j, (condition, color) in enumerate(
        [("Original best-shared", COLOR_BASELINE), (f"CSTIM LOSO w={endpoint_weight:g}", COLOR_CSTIM)]
    ):
        vals = [
            summ[(summ["model_set"].eq(ms)) & (summ["condition"].eq(condition))]
            for ms in PANEL_ORDER
        ]
        means = [float(v["mean"].iloc[0]) if len(v) else np.nan for v in vals]
        los = [float(v["lo"].iloc[0]) if len(v) else np.nan for v in vals]
        his = [float(v["hi"].iloc[0]) if len(v) else np.nan for v in vals]
        yerr = np.array([
            np.asarray(means) - np.asarray(los),
            np.asarray(his) - np.asarray(means),
        ])
        ax.bar(
            x + (j - 0.5) * width,
            means,
            width,
            yerr=yerr,
            color=color,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.5,
            label=condition,
            capsize=2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([TITLE[ms] for ms in PANEL_ORDER], rotation=25, ha="right")
    ax.set_ylabel("Mixed RSA ($r_s$)")
    ax.grid(axis="y", alpha=0.22, linewidth=0.45)
    ax.legend(frameon=True, framealpha=0.92, edgecolor="none", fontsize=FONT["small"])
    pdf = FIGURES_DIR / "cstim_encoding_endpoint_cached.pdf"
    png = PNG_DIR / "cstim_encoding_endpoint_cached.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=DPI)
    print(f"Saved {pdf}")
    print(f"Saved {png}")
    plt.close(fig)


def write_plot_summary(df: pd.DataFrame) -> None:
    summary = set_summary(df, "mrsa_loso")
    summary.to_csv(RESULTS_DIR / "cstim_loso_weighted_summary.csv", index=False)
    if AUDIT_CSV.exists():
        audit = pd.read_csv(AUDIT_CSV)
        audit_summary = (
            audit.groupby("cache_ready", as_index=False)
            .agg(n=("model", "size"), n_models=("model", "nunique"))
        )
        audit_summary.to_csv(RESULTS_DIR / "cached_selection_audit_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-weight", type=float, default=2.0)
    args = parser.parse_args()

    df = load_scores()
    cstim_df = df[df["eval_target"].eq("cstim_loso")].copy()
    write_plot_summary(cstim_df)
    plot_grid(
        cstim_df,
        value_col="mrsa_loso",
        out_stem="cstim_weighting_trajectory_cached",
        ylabel="CSTIM LOSO mixed RSA ($r_s$)",
    )
    plot_grid(
        cstim_df,
        value_col="delta_vs_original",
        out_stem="cstim_weighting_delta_trajectory_cached",
        ylabel=r"$\Delta$ vs original best-shared",
    )
    plot_endpoint(cstim_df, args.endpoint_weight)


if __name__ == "__main__":
    main()
