#!/usr/bin/env python3
"""Visualize sensitivity-fit scores across SOTA models."""

from __future__ import annotations

import argparse
from pathlib import Path

import _paths  # noqa: F401
from _paths import FIGURES_DIR, PNG_DIR, RESULTS_DIR

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from cstims.target_adaptation import SHORT_MODEL_NAMES, sem
from cstims.paper.style_improved import (
    COLOR_BASELINE,
    COLOR_CSTIM,
    DPI,
    FONT,
    W_DOUBLE,
    apply_style,
)


apply_style()

FIT_SCOPE_CONFIG = {
    "deepvision-plus-cstim": {
        "score_csv": "target_adaptation_full_refit_sensitivity_scores.csv",
        "summary_csv": "target_adaptation_full_refit_sensitivity_by_model.csv",
        "figure_stem": "target_adaptation_full_refit_sensitivity_by_model",
        "fit_label": "DeepVision+CSTIM full refit",
    },
    "cstim-only": {
        "score_csv": "target_adaptation_cstim_only_sensitivity_scores.csv",
        "summary_csv": "target_adaptation_cstim_only_sensitivity_by_model.csv",
        "figure_stem": "target_adaptation_cstim_only_sensitivity_by_model",
        "fit_label": "CSTIM-only fit",
    },
}

EVAL_SPECS = [
    ("cstim_loso", "CSTIM", COLOR_CSTIM),
    ("vicco_heldout", "Baseline", COLOR_BASELINE),
]

SUBJECT_OFFSETS = {
    "sub-01": -0.10,
    "sub-03": -0.05,
    "sub-05": 0.00,
    "sub-06": 0.05,
    "sub-07": 0.10,
}
EVAL_CENTER_OFFSETS = {"cstim_loso": -0.20, "vicco_heldout": 0.20}
CANONICAL_OFFSET = -0.065
FIT_OFFSET = 0.065
POINT_SIZE = 17
MEAN_SIZE = 44
FONT_TICK = 8
FONT_SMALL = FONT.get("small", 10)
FONT_AXIS = 10
FONT_SUPTITLE = 12


def short_model_label(model: str, display_name: str) -> str:
    return SHORT_MODEL_NAMES.get(model, display_name)


def paths_for_fit_scope(fit_scope: str) -> tuple[Path, Path, Path, Path]:
    cfg = FIT_SCOPE_CONFIG[fit_scope]
    return (
        RESULTS_DIR / cfg["score_csv"],
        RESULTS_DIR / cfg["summary_csv"],
        FIGURES_DIR / f"{cfg['figure_stem']}.pdf",
        PNG_DIR / f"{cfg['figure_stem']}.png",
    )


def load_scores(score_csv: Path, fit_scope: str) -> pd.DataFrame:
    if not score_csv.exists():
        raise FileNotFoundError(f"Missing sensitivity scores: {score_csv}")
    df = pd.read_csv(score_csv)
    required = {
        "subject",
        "model",
        "display_name",
        "eval_target",
        "target_weight",
        "mrsa_loso",
        "canonical_fixed_dv_stats_mrsa",
        "delta_vs_canonical_fixed_dv_stats",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{score_csv} missing columns: {sorted(missing)}")
    df = df[df["eval_target"].isin([spec[0] for spec in EVAL_SPECS])].copy()
    if df.empty:
        raise RuntimeError("No sensitivity rows to plot.")
    df["target_weight"] = df["target_weight"].astype(float)
    df["fit_scope"] = df.get("fit_scope", fit_scope)
    df["fit_scope_label"] = df.get("fit_scope_label", FIT_SCOPE_CONFIG[fit_scope]["fit_label"])
    df["model_label"] = [
        short_model_label(row.model, row.display_name) for row in df.itertuples(index=False)
    ]
    return df


def model_order(df: pd.DataFrame) -> list[str]:
    cstim = df[df["eval_target"].eq("cstim_loso")]
    return (
        cstim.groupby("model")["delta_vs_canonical_fixed_dv_stats"]
        .mean()
        .reset_index(name="mean_delta")
        .sort_values(["mean_delta", "model"], ascending=[True, True])["model"]
        .tolist()
    )


def write_by_model_summary(df: pd.DataFrame, order: list[str], summary_csv: Path) -> pd.DataFrame:
    rows = []
    order_rank = {model: i for i, model in enumerate(order)}
    for (model, eval_target), block in df.groupby(["model", "eval_target"]):
        rows.append(
            {
                "model_order": order_rank[model],
                "model": model,
                "display_name": block["display_name"].iloc[0],
                "model_label": block["model_label"].iloc[0],
                "eval_target": eval_target,
                "eval_label": dict((key, label) for key, label, _color in EVAL_SPECS)[
                    eval_target
                ],
                "fit_scope": block["fit_scope"].iloc[0],
                "fit_scope_label": block["fit_scope_label"].iloc[0],
                "target_weight": float(block["target_weight"].iloc[0]),
                "n_subjects": int(block["subject"].nunique()),
                "canonical_mean": float(block["canonical_fixed_dv_stats_mrsa"].mean()),
                "fit_mean": float(block["mrsa_loso"].mean()),
                "delta_mean": float(block["delta_vs_canonical_fixed_dv_stats"].mean()),
                "delta_sem": sem(
                    block["delta_vs_canonical_fixed_dv_stats"].to_numpy(dtype=float)
                ),
                "delta_min": float(block["delta_vs_canonical_fixed_dv_stats"].min()),
                "delta_max": float(block["delta_vs_canonical_fixed_dv_stats"].max()),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["model_order", "eval_target"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    return summary


def x_position(model_x: int, eval_target: str, subject: str, *, fit: bool) -> float:
    eval_center = EVAL_CENTER_OFFSETS[eval_target]
    fit_offset = FIT_OFFSET if fit else CANONICAL_OFFSET
    subject_offset = SUBJECT_OFFSETS.get(subject, 0.0) * 0.28
    return model_x + eval_center + fit_offset + subject_offset


def draw_absolute_panel(ax, df: pd.DataFrame, *, order: list[str]) -> None:
    for model_x, model in enumerate(order):
        for eval_target, _label, color in EVAL_SPECS:
            block = df[df["model"].eq(model) & df["eval_target"].eq(eval_target)].copy()
            for row in block.itertuples(index=False):
                x0 = x_position(model_x, eval_target, row.subject, fit=False)
                x1 = x_position(model_x, eval_target, row.subject, fit=True)
                ax.plot(
                    [x0, x1],
                    [row.canonical_fixed_dv_stats_mrsa, row.mrsa_loso],
                    color=color,
                    linewidth=0.45,
                    alpha=0.24,
                    zorder=1,
                )
                ax.scatter(
                    x0,
                    row.canonical_fixed_dv_stats_mrsa,
                    s=POINT_SIZE,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=0.55,
                    alpha=0.95,
                    zorder=3,
                )
                ax.scatter(
                    x1,
                    row.mrsa_loso,
                    s=POINT_SIZE,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.35,
                    alpha=0.95,
                    zorder=4,
                )

            center = model_x + EVAL_CENTER_OFFSETS[eval_target]
            canonical_mean = float(block["canonical_fixed_dv_stats_mrsa"].mean())
            fit_mean = float(block["mrsa_loso"].mean())
            ax.scatter(
                center + CANONICAL_OFFSET,
                canonical_mean,
                marker="D",
                s=MEAN_SIZE,
                facecolor="white",
                edgecolor=color,
                linewidth=0.75,
                zorder=6,
            )
            ax.scatter(
                center + FIT_OFFSET,
                fit_mean,
                marker="D",
                s=MEAN_SIZE,
                facecolor=color,
                edgecolor="white",
                linewidth=0.35,
                zorder=7,
            )

    ax.set_title("Absolute mixed RSA", fontsize=FONT_SMALL, fontweight="bold")
    ax.set_ylabel("Mixed RSA ($r_s$)", fontsize=FONT_AXIS)
    ax.grid(axis="y", alpha=0.22, linewidth=0.45)


def draw_delta_panel(ax, df: pd.DataFrame, *, order: list[str]) -> None:
    ax.axhline(0, color="0.35", linestyle="--", linewidth=0.8, alpha=0.75, zorder=0)
    for model_x, model in enumerate(order):
        for eval_target, _label, color in EVAL_SPECS:
            block = df[df["model"].eq(model) & df["eval_target"].eq(eval_target)].copy()
            center = model_x + EVAL_CENTER_OFFSETS[eval_target]
            for row in block.itertuples(index=False):
                subject_offset = SUBJECT_OFFSETS.get(row.subject, 0.0) * 0.62
                ax.scatter(
                    center + subject_offset,
                    row.delta_vs_canonical_fixed_dv_stats,
                    s=POINT_SIZE,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.35,
                    alpha=0.95,
                    zorder=4,
                )
            vals = block["delta_vs_canonical_fixed_dv_stats"].to_numpy(dtype=float)
            mean = float(np.mean(vals))
            err = sem(vals)
            ax.vlines(
                center,
                mean - err,
                mean + err,
                color="0.15",
                linewidth=0.9,
                zorder=5,
            )
            ax.scatter(
                center,
                mean,
                marker="D",
                s=MEAN_SIZE,
                facecolor=color,
                edgecolor="white",
                linewidth=0.35,
                zorder=6,
            )

    ax.set_title("Delta vs canonical", fontsize=FONT_SMALL, fontweight="bold")
    ax.set_ylabel("Fit minus canonical", fontsize=FONT_AXIS)
    ax.grid(axis="y", alpha=0.22, linewidth=0.45)


def set_model_axis(ax, df: pd.DataFrame, order: list[str]) -> None:
    labels = []
    for model in order:
        row = df[df["model"].eq(model)].iloc[0]
        labels.append(row.model_label)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=FONT_TICK)


def make_figure(df: pd.DataFrame, order: list[str], pdf: Path, png: Path) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(W_DOUBLE, 4.65),
        constrained_layout=True,
    )
    draw_absolute_panel(axes[0], df, order=order)
    draw_delta_panel(axes[1], df, order=order)
    for ax in axes:
        set_model_axis(ax, df, order)

    fit_label = str(df["fit_scope_label"].iloc[0])
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="0.20",
            markersize=4.7,
            label="Canonical",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="0.35",
            markeredgecolor="white",
            markersize=4.7,
            label="Sensitivity fit",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="0.35",
            markeredgecolor="white",
            markersize=5.2,
            label="Model mean",
        ),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_CSTIM,
               markeredgecolor="white", markersize=4.7, label="CSTIM"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_BASELINE,
               markeredgecolor="white", markersize=4.7, label="Baseline"),
    ]
    axes[0].legend(
        handles=handles,
        loc="best",
        frameon=True,
        framealpha=0.94,
        edgecolor="none",
        fontsize=FONT_TICK,
        ncol=1,
    )

    weight = float(df["target_weight"].iloc[0])
    fig.suptitle(
        f"{fit_label} across SOTA models, canonical comparison weight={weight:g}",
        fontsize=FONT_SUPTITLE,
        fontweight="bold",
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf)
    fig.savefig(png, dpi=DPI)
    print(f"Saved {pdf}")
    print(f"Saved {png}")
    plt.close(fig)


def plot_fit_scope(fit_scope: str) -> None:
    score_csv, summary_csv, pdf, png = paths_for_fit_scope(fit_scope)
    df = load_scores(score_csv, fit_scope)
    order = model_order(df)
    write_by_model_summary(df, order, summary_csv)
    make_figure(df, order, pdf, png)
    print(f"Saved {summary_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fit-scope",
        default="deepvision-plus-cstim",
        choices=[*sorted(FIT_SCOPE_CONFIG), "all"],
    )
    args = parser.parse_args()

    scopes = sorted(FIT_SCOPE_CONFIG) if args.fit_scope == "all" else [args.fit_scope]
    for fit_scope in scopes:
        plot_fit_scope(fit_scope)


if __name__ == "__main__":
    main()
