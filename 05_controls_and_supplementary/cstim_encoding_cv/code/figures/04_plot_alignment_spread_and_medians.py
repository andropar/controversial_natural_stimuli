#!/usr/bin/env python3
"""Plot median and spread of alignment across layers and refit target weights."""

from __future__ import annotations

import _paths  # noqa: F401
from _paths import FIGURES_DIR, LAYER_SWEEP_ROOT, RESULTS_DIR

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cstims import constants
from cstims.paper.style_improved import (
    COLOR_BASELINE,
    COLOR_CSTIM,
    COLOR_TRAIN,
    DPI,
    FONT,
    W_DOUBLE,
    apply_style,
)
from cstims.target_adaptation import MODEL_SET_TITLES, PANEL_ORDER
from layers_config import get_layer_set


apply_style()

DENSE_LAYER_SCORES_CSV = (
    LAYER_SWEEP_ROOT / "results" / "mrsa_dense_all_eval_layer_scores.csv"
)
REFIT_SCORE_CSV = RESULTS_DIR / "03_refit_alpha" / "plus4700" / "scores.csv"

OUTPUT_RESULTS_DIR = RESULTS_DIR / "04_alignment_spread"
OUTPUT_FIGURES_DIR = FIGURES_DIR / "04_alignment_spread"
OUTPUT_PNG_DIR = OUTPUT_FIGURES_DIR / "png"
CURVE_SUMMARY_CSV = OUTPUT_RESULTS_DIR / "layer_depth_curve_summary_from_dense.csv"

GRID = np.linspace(0.0, 1.0, 31)
FONT_TINY = FONT.get("tiny", FONT.get("small", FONT["tick"]))
FONT_AXIS = 9
FONT_NOTE = 8
FONT_XTICK = 7

LAYER_CONDITIONS = [
    ("cstim", "CSTIM", COLOR_CSTIM, "-", 0.14),
    ("vicco", "Vicco", COLOR_BASELINE, "-", 0.14),
    ("shared", "Shared", COLOR_TRAIN, "--", 0.10),
]
REFIT_CONDITIONS = [
    ("cstim_loso", "CSTIM LOSO", COLOR_CSTIM),
    ("vicco_heldout", "Held-out Vicco", COLOR_BASELINE),
]


def layer_maps() -> tuple[dict[str, dict[str, int]], dict[str, dict[str, float]]]:
    specs = get_layer_set("dense")
    index_maps = {}
    frac_maps = {}
    for model, layers in specs.items():
        names = [name for name, _shape in layers]
        denom = max(len(names) - 1, 1)
        index_maps[model] = {name: idx for idx, name in enumerate(names)}
        frac_maps[model] = {name: idx / denom for idx, name in enumerate(names)}
    return index_maps, frac_maps


def build_curve_summary_from_dense(*, force: bool = False) -> pd.DataFrame:
    if (
        CURVE_SUMMARY_CSV.exists()
        and not force
        and CURVE_SUMMARY_CSV.stat().st_mtime >= DENSE_LAYER_SCORES_CSV.stat().st_mtime
    ):
        return pd.read_csv(CURVE_SUMMARY_CSV)

    keep_model_sets = set(PANEL_ORDER) | {"vicco", "deepvision_shared"}
    keep_eval_targets = {"cstim", "vicco", "shared"}
    group_cols = ["model", "display_name", "layer", "eval_target", "model_set"]
    usecols = group_cols + ["mrsa"]
    chunks = []
    for chunk in pd.read_csv(DENSE_LAYER_SCORES_CSV, usecols=usecols, chunksize=1_000_000):
        chunk = chunk[
            chunk["eval_target"].isin(keep_eval_targets)
            & chunk["model_set"].isin(keep_model_sets)
        ].copy()
        if chunk.empty:
            continue
        chunks.append(
            chunk.groupby(group_cols, as_index=False)
            .agg(mrsa_sum=("mrsa", "sum"), n=("mrsa", "count"))
        )

    if not chunks:
        raise RuntimeError(f"No layer rows found in {DENSE_LAYER_SCORES_CSV}")

    summary = (
        pd.concat(chunks, ignore_index=True)
        .groupby(group_cols, as_index=False)
        .agg(mrsa_sum=("mrsa_sum", "sum"), n=("n", "sum"))
    )
    summary["mrsa"] = summary["mrsa_sum"] / summary["n"].clip(lower=1)

    index_maps, frac_maps = layer_maps()
    summary["layer_index"] = [
        index_maps.get(model, {}).get(layer, np.nan)
        for model, layer in zip(summary["model"], summary["layer"])
    ]
    summary["layer_frac"] = [
        frac_maps.get(model, {}).get(layer, np.nan)
        for model, layer in zip(summary["model"], summary["layer"])
    ]
    summary = summary.dropna(subset=["layer_index", "layer_frac"]).copy()
    summary["layer_index"] = summary["layer_index"].astype(int)
    summary = summary[
        [
            "model",
            "layer",
            "eval_target",
            "model_set",
            "mrsa",
            "layer_index",
            "layer_frac",
            "display_name",
            "n",
        ]
    ].sort_values(["model", "eval_target", "model_set", "layer_index"])

    OUTPUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(CURVE_SUMMARY_CSV, index=False)
    print(f"Saved {CURVE_SUMMARY_CSV}")
    return summary


def finite_percentile(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(values, percentile))


def summarize_values(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "n": int(len(values)),
        "p10": finite_percentile(values, 10),
        "q25": finite_percentile(values, 25),
        "median": finite_percentile(values, 50),
        "q75": finite_percentile(values, 75),
        "p90": finite_percentile(values, 90),
        "min": float(values.min()) if len(values) else float("nan"),
        "max": float(values.max()) if len(values) else float("nan"),
    }


def model_order(model_set: str) -> list[str]:
    models = constants.MODEL_SETS.get(model_set, [])
    return list(models)


def condition_filter(
    summary: pd.DataFrame,
    *,
    model: str,
    panel_model_set: str,
    condition: str,
) -> pd.DataFrame:
    if condition == "cstim":
        target = "cstim"
        source_set = panel_model_set
    elif condition == "vicco":
        target = "vicco"
        source_set = "vicco"
    elif condition == "shared":
        target = "shared"
        source_set = "deepvision_shared"
    else:
        raise ValueError(f"Unknown layer condition: {condition}")
    return summary[
        summary["model"].eq(model)
        & summary["eval_target"].eq(target)
        & summary["model_set"].eq(source_set)
    ].sort_values("layer_frac")


def interpolate_curve(curve: pd.DataFrame, grid: np.ndarray) -> np.ndarray | None:
    curve = curve[["layer_frac", "mrsa"]].dropna().drop_duplicates("layer_frac")
    curve = curve.sort_values("layer_frac")
    if curve.empty:
        return None
    x = curve["layer_frac"].to_numpy(dtype=float)
    y = curve["mrsa"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) == 0:
        return None
    if len(x) == 1:
        return np.full_like(grid, y[0], dtype=float)
    return np.interp(grid, x, y)


def layer_matrix(
    summary: pd.DataFrame,
    *,
    model_set: str,
    condition: str,
) -> tuple[np.ndarray, list[str]]:
    rows = []
    used_models = []
    for model in model_order(model_set):
        curve = condition_filter(
            summary,
            model=model,
            panel_model_set=model_set,
            condition=condition,
        )
        y = interpolate_curve(curve, GRID)
        if y is None:
            continue
        rows.append(y)
        used_models.append(model)
    if not rows:
        return np.empty((0, len(GRID))), []
    return np.vstack(rows), used_models


def build_layer_spread_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_set in PANEL_ORDER:
        for condition, label, _color, _linestyle, _alpha in LAYER_CONDITIONS:
            values, used_models = layer_matrix(
                summary,
                model_set=model_set,
                condition=condition,
            )
            for idx, depth in enumerate(GRID):
                stats = summarize_values(values[:, idx] if len(values) else np.array([]))
                rows.append(
                    {
                        "model_set": model_set,
                        "condition": condition,
                        "condition_label": label,
                        "layer_frac": float(depth),
                        "n_models": len(used_models),
                        **stats,
                    }
                )
    out = pd.DataFrame(rows)
    OUTPUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_RESULTS_DIR / "layer_depth_spread_summary.csv"
    out.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    return out


def draw_layer_panel(
    ax,
    layer_summary: pd.DataFrame,
    raw_summary: pd.DataFrame,
    *,
    model_set: str,
    show_legend: bool,
) -> None:
    for condition, label, color, linestyle, fill_alpha in LAYER_CONDITIONS:
        panel = layer_summary[
            layer_summary["model_set"].eq(model_set)
            & layer_summary["condition"].eq(condition)
        ].sort_values("layer_frac")
        if panel.empty:
            continue

        values, used_models = layer_matrix(
            raw_summary,
            model_set=model_set,
            condition=condition,
        )
        trace_alpha = 0.055 if condition != "shared" else 0.030
        for row in values:
            ax.plot(
                GRID,
                row,
                color=color,
                linewidth=0.35,
                alpha=trace_alpha,
                zorder=1,
            )

        x = panel["layer_frac"].to_numpy(dtype=float)
        ax.fill_between(
            x,
            panel["p10"].to_numpy(dtype=float),
            panel["p90"].to_numpy(dtype=float),
            color=color,
            alpha=fill_alpha * 0.42,
            linewidth=0,
            zorder=2,
        )
        ax.fill_between(
            x,
            panel["q25"].to_numpy(dtype=float),
            panel["q75"].to_numpy(dtype=float),
            color=color,
            alpha=fill_alpha,
            linewidth=0,
            zorder=3,
        )
        ax.plot(
            x,
            panel["median"].to_numpy(dtype=float),
            color=color,
            linestyle=linestyle,
            linewidth=1.45 if condition != "shared" else 1.15,
            label=label,
            zorder=5,
        )

        if condition == "cstim":
            ax.text(
                0.02,
                0.96,
                f"n={len(used_models)} models",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=FONT_NOTE,
                color="0.25",
            )

    ax.set_title(MODEL_SET_TITLES[model_set], fontsize=FONT["small"], fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Layer depth", fontsize=FONT_AXIS)
    ax.grid(axis="y", alpha=0.22, linewidth=0.45)
    if show_legend:
        ax.legend(
            loc="lower left",
            frameon=True,
            framealpha=0.92,
            edgecolor="none",
            fontsize=FONT_TINY,
        )


def plot_layer_depth_spread() -> None:
    raw_summary = build_curve_summary_from_dense()
    layer_summary = build_layer_spread_summary(raw_summary)

    fig, axes = plt.subplots(
        1,
        len(PANEL_ORDER),
        figsize=(W_DOUBLE, 3.45),
        sharey=True,
        constrained_layout=True,
    )
    for idx, (ax, model_set) in enumerate(zip(axes, PANEL_ORDER)):
        draw_layer_panel(
            ax,
            layer_summary,
            raw_summary,
            model_set=model_set,
            show_legend=(idx == 0),
        )
        if idx == 0:
            ax.set_ylabel("Mixed RSA ($r_s$)", fontsize=FONT_AXIS)
        else:
            ax.set_ylabel("")
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", left=False, labelleft=False)

    fig.suptitle(
        "Median alignment and spread across dense layers",
        fontsize=FONT["title"],
        fontweight="bold",
    )
    stem = "target_adaptation_layer_depth_spread_median_cached"
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PNG_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUTPUT_FIGURES_DIR / f"{stem}.pdf"
    out_png = OUTPUT_PNG_DIR / f"{stem}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=DPI)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    plt.close(fig)


def format_weight(weight: float) -> str:
    if np.isposinf(weight):
        return "target-only"
    if abs(weight) >= 1000:
        return f"{weight / 1000:g}k"
    return f"{weight:g}"


def x_positions(weights: list[float]) -> dict[float, int]:
    return {float(weight): idx for idx, weight in enumerate(sorted(weights))}


def load_refit_scores() -> pd.DataFrame:
    df = pd.read_csv(REFIT_SCORE_CSV)
    df["target_weight"] = df["target_weight"].astype(float)
    df = df[df["eval_target"].isin([name for name, _label, _color in REFIT_CONDITIONS])].copy()
    keys = ["subject", "model", "model_set", "eval_target"]
    base = df[np.isclose(df["target_weight"], 0.0)][keys + ["mrsa_loso"]].rename(
        columns={"mrsa_loso": "mrsa_weight0"}
    )
    df = df.merge(base, on=keys, how="left", validate="many_to_one")
    df["delta_vs_weight0"] = df["mrsa_loso"] - df["mrsa_weight0"]
    return df


def build_refit_weight_spread_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_set, eval_target, target_weight), block in df.groupby(
        ["model_set", "eval_target", "target_weight"], sort=False
    ):
        for view, value_col in [
            ("absolute", "mrsa_loso"),
            ("delta_vs_weight0", "delta_vs_weight0"),
        ]:
            stats = summarize_values(block[value_col].to_numpy(dtype=float))
            rows.append(
                {
                    "model_set": model_set,
                    "eval_target": eval_target,
                    "target_weight": float(target_weight),
                    "view": view,
                    "n_models": block["model"].nunique(),
                    "n_subjects": block["subject"].nunique(),
                    **stats,
                }
            )
    out = pd.DataFrame(rows)
    OUTPUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_RESULTS_DIR / "refit_weight_spread_summary.csv"
    out.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    return out


def draw_refit_panel(
    ax,
    summary: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    model_set: str,
    view: str,
    weights: list[float],
    show_xticks: bool,
    show_legend: bool,
) -> None:
    xpos = x_positions(weights)
    panel_scores = scores[scores["model_set"].eq(model_set)].copy()
    panel_summary = summary[
        summary["model_set"].eq(model_set) & summary["view"].eq(view)
    ].copy()

    y_col = "mrsa_loso" if view == "absolute" else "delta_vs_weight0"
    for eval_target, label, color in REFIT_CONDITIONS:
        block_scores = panel_scores[panel_scores["eval_target"].eq(eval_target)].copy()
        block_scores["x"] = block_scores["target_weight"].map(xpos)
        for _key, line in block_scores.groupby(["subject", "model"]):
            line = line.sort_values("target_weight")
            ax.plot(
                line["x"].to_numpy(dtype=float),
                line[y_col].to_numpy(dtype=float),
                color=color,
                linewidth=0.32,
                alpha=0.055,
                zorder=1,
            )

        block = panel_summary[panel_summary["eval_target"].eq(eval_target)].copy()
        if block.empty:
            continue
        block["x"] = block["target_weight"].map(xpos)
        block = block.sort_values("target_weight")
        x = block["x"].to_numpy(dtype=float)
        ax.fill_between(
            x,
            block["p10"].to_numpy(dtype=float),
            block["p90"].to_numpy(dtype=float),
            color=color,
            alpha=0.055,
            linewidth=0,
            zorder=2,
        )
        ax.fill_between(
            x,
            block["q25"].to_numpy(dtype=float),
            block["q75"].to_numpy(dtype=float),
            color=color,
            alpha=0.15,
            linewidth=0,
            zorder=3,
        )
        ax.plot(
            x,
            block["median"].to_numpy(dtype=float),
            color=color,
            marker="o",
            markersize=3.2,
            linewidth=1.35,
            label=label,
            zorder=5,
        )

    if view == "delta_vs_weight0":
        ax.axhline(0, color="0.35", linestyle="--", linewidth=0.75, alpha=0.75, zorder=0)

    ax.text(
        0.02,
        0.96,
        f"n={panel_scores['model'].nunique()} models, {panel_scores['subject'].nunique()} subjects",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_NOTE,
        color="0.25",
    )
    ax.grid(axis="y", alpha=0.22, linewidth=0.45)
    ax.set_xlim(-0.35, len(weights) - 0.65)
    if show_xticks:
        ax.set_xticks([xpos[weight] for weight in weights])
        ax.set_xticklabels(
            [format_weight(weight) for weight in weights],
            rotation=48,
            ha="right",
            fontsize=FONT_XTICK,
        )
        ax.set_xlabel("Target sample weight", fontsize=FONT_AXIS)
    else:
        ax.set_xticks([xpos[weight] for weight in weights])
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    if show_legend:
        ax.legend(
            loc="lower left",
            frameon=True,
            framealpha=0.92,
            edgecolor="none",
            fontsize=FONT_TINY,
        )


def plot_refit_weight_spread() -> None:
    scores = load_refit_scores()
    summary = build_refit_weight_spread_summary(scores)
    weights = sorted(float(weight) for weight in scores["target_weight"].unique())

    fig, axes = plt.subplots(
        2,
        len(PANEL_ORDER),
        figsize=(W_DOUBLE, 6.45),
        sharex=True,
        constrained_layout=True,
    )
    views = [
        ("absolute", "Selected-layer mixed RSA ($r_s$)"),
        ("delta_vs_weight0", "Delta mixed RSA vs weight 0"),
    ]
    for row_idx, (view, ylabel) in enumerate(views):
        row_values = summary[summary["view"].eq(view)][["p10", "p90"]].to_numpy(dtype=float)
        row_values = row_values[np.isfinite(row_values)]
        if len(row_values):
            if view == "delta_vs_weight0":
                span = max(abs(float(row_values.min())), abs(float(row_values.max()))) * 1.08
                y_limits = (-max(span, 0.015), max(span, 0.015))
            else:
                y_limits = (
                    min(-0.035, float(row_values.min()) - 0.02),
                    float(row_values.max()) + 0.025,
                )
        else:
            y_limits = (-0.05, 0.75)

        for col_idx, model_set in enumerate(PANEL_ORDER):
            ax = axes[row_idx, col_idx]
            ax.set_ylim(*y_limits)
            draw_refit_panel(
                ax,
                summary,
                scores,
                model_set=model_set,
                view=view,
                weights=weights,
                show_xticks=(row_idx == len(views) - 1),
                show_legend=(row_idx == 0 and col_idx == 0),
            )
            if row_idx == 0:
                ax.set_title(
                    MODEL_SET_TITLES[model_set],
                    fontsize=FONT["small"],
                    fontweight="bold",
                )
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=FONT_AXIS)
            else:
                ax.set_ylabel("")
                ax.spines["left"].set_visible(False)
                ax.tick_params(axis="y", left=False, labelleft=False)

    fig.suptitle(
        "Median alignment and spread across target-weight refits",
        fontsize=FONT["title"],
        fontweight="bold",
    )
    stem = "target_adaptation_refit_weight_spread_median_cached"
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PNG_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUTPUT_FIGURES_DIR / f"{stem}.pdf"
    out_png = OUTPUT_PNG_DIR / f"{stem}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=DPI)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    plt.close(fig)


def main() -> None:
    plot_layer_depth_spread()
    plot_refit_weight_spread()


if __name__ == "__main__":
    main()
