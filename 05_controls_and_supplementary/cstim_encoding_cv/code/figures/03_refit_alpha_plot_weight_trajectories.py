#!/usr/bin/env python3
"""Plot target-adaptation weighting trajectories."""

from __future__ import annotations

import _paths  # noqa: F401
from _paths import FIGURES_DIR, RESULTS_DIR

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cstims.target_adaptation import MODEL_SET_TITLES, PANEL_ORDER, mean_ci
from cstims.paper.style_improved import (
    COLOR_BASELINE,
    COLOR_CSTIM,
    DPI,
    FONT,
    W_DOUBLE,
    apply_style,
)


apply_style()

REFIT_RESULTS_DIR = RESULTS_DIR / "03_refit_alpha"
OUTPUT_FIGURES_DIR = FIGURES_DIR / "03_refit_alpha"
OUTPUT_PNG_DIR = OUTPUT_FIGURES_DIR / "png"
SCORE_CSV = REFIT_RESULTS_DIR / "plus4700" / "scores.csv"
FONT_TINY = FONT.get("tiny", FONT.get("small", FONT["tick"]))
FONT_AXIS = 9
FONT_XTICK = 7


def summarize(df: pd.DataFrame, value_col: str = "mrsa_loso") -> pd.DataFrame:
    rows = []
    for (model_set, eval_target, weight), block in df.groupby(
        ["model_set", "eval_target", "target_weight"]
    ):
        mean, lo, hi = mean_ci(block[value_col].to_numpy())
        rows.append(
            {
                "model_set": model_set,
                "eval_target": eval_target,
                "target_weight": float(weight),
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
    return {w: i for i, w in enumerate(sorted(weights))}


def weight_label(weight: float) -> str:
    if np.isposinf(weight):
        return "target-only"
    if abs(weight) >= 1000:
        return f"{weight / 1000:g}k"
    return f"{weight:g}"


def draw_panel(ax, df: pd.DataFrame, model_set: str, *, delta: bool) -> None:
    sub = df[df["model_set"].eq(model_set)].copy()
    if sub.empty:
        ax.set_visible(False)
        return
    weights = sorted(sub["target_weight"].unique())
    xpos = x_positions(weights)
    sub["x"] = sub["target_weight"].map(xpos)

    for eval_target, color, label in [
        ("cstim_loso", COLOR_CSTIM, "CSTIM LOSO"),
        ("vicco_heldout", COLOR_BASELINE, "Baseline held out"),
    ]:
        block = sub[sub["eval_target"].eq(eval_target)].copy()
        if block.empty:
            continue
        ycol = "delta_vs_original" if delta else "mrsa_loso"
        for _key, line in block.groupby(["subject", "model"]):
            line = line.sort_values("target_weight")
            ax.plot(
                line["x"],
                line[ycol],
                color=color,
                linewidth=0.35,
                alpha=0.12,
                zorder=1,
            )
        summ = summarize(block, value_col=ycol).sort_values("target_weight")
        x = summ["target_weight"].map(xpos).to_numpy(dtype=float)
        ax.fill_between(x, summ["lo"], summ["hi"], color=color, alpha=0.13, linewidth=0)
        ax.plot(
            x,
            summ["mean"],
            marker="o",
            markersize=3.6,
            linewidth=1.35,
            color=color,
            label=label,
            zorder=4,
        )

    if delta:
        ax.axhline(0, color="0.35", linestyle="--", linewidth=0.8, alpha=0.75, zorder=0)
        if weights and max(weights) >= 1000:
            ax.axvline(
                xpos[max(weights)] - 0.5,
                color="0.65",
                linestyle=":",
                linewidth=0.65,
                alpha=0.75,
                zorder=0,
            )
    else:
        for eval_target, color in [("cstim_loso", COLOR_CSTIM), ("vicco_heldout", COLOR_BASELINE)]:
            ref = sub[sub["eval_target"].eq(eval_target)].drop_duplicates(
                ["subject", "model", "model_set"]
            )
            vals = ref["original_best_shared_mrsa"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                ax.axhline(
                    float(vals.mean()),
                    color=color,
                    linestyle="--",
                    linewidth=0.85,
                    alpha=0.70,
                    zorder=0,
                )

    ax.set_title(MODEL_SET_TITLES[model_set], fontsize=FONT["small"], fontweight="bold")
    ax.set_xticks([xpos[w] for w in weights])
    ax.set_xticklabels([weight_label(w) for w in weights], fontsize=FONT_XTICK)
    ax.set_xlabel("Target sample weight", fontsize=FONT_AXIS)
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


def load_scores(*, include_extreme: bool) -> pd.DataFrame:
    df = pd.read_csv(SCORE_CSV)
    df["target_weight"] = df["target_weight"].astype(float)
    return df[df["eval_target"].isin(["cstim_loso", "vicco_heldout"])].copy()


def plot(delta: bool = False) -> None:
    df = load_scores(include_extreme=delta)
    fig, axes = plt.subplots(
        1,
        len(PANEL_ORDER),
        figsize=(W_DOUBLE, 3.35),
        sharey=False,
        constrained_layout=True,
    )
    for i, (ax, model_set) in enumerate(zip(axes, PANEL_ORDER)):
        draw_panel(ax, df, model_set, delta=delta)
        if i == 0:
            ax.set_ylabel(
                "Delta mixed RSA vs original" if delta else "Mixed RSA ($r_s$)",
                fontsize=FONT_AXIS,
            )
            ax.legend(
                loc="lower right",
                fontsize=FONT_TINY,
                frameon=True,
                framealpha=0.92,
                edgecolor="none",
            )
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PNG_DIR.mkdir(parents=True, exist_ok=True)
    stem = (
        "target_adaptation_weight_delta_trajectory_cached"
        if delta
        else "target_adaptation_weight_trajectory_cached"
    )
    pdf = OUTPUT_FIGURES_DIR / f"{stem}.pdf"
    png = OUTPUT_PNG_DIR / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=DPI)
    print(f"Saved {pdf}")
    print(f"Saved {png}")
    plt.close(fig)


def main() -> None:
    plot(delta=False)
    plot(delta=True)


if __name__ == "__main__":
    main()
