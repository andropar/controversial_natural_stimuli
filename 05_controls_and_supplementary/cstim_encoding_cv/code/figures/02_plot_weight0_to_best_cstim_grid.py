#!/usr/bin/env python3
"""Plot weight-0 to best-CSTIM-weight mixed-RSA values and deltas by model."""

from __future__ import annotations

import _paths  # noqa: F401
from _paths import FIGURES_DIR, PNG_DIR, RESULTS_DIR

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from cstims.paper import config
from cstims.target_adaptation import MODEL_SET_TITLES, PANEL_ORDER, SHORT_MODEL_NAMES, sem
from cstims.paper.style_improved import (
    COLOR_BASELINE,
    COLOR_CSTIM,
    DPI,
    FONT,
    W_DOUBLE,
    apply_style,
)


apply_style()

SCORE_CSV = RESULTS_DIR / "target_adaptation_weighted_scores.csv"
POINTS_CSV = RESULTS_DIR / "target_adaptation_weight0_to_best_cstim_points.csv"
SUMMARY_CSV = RESULTS_DIR / "target_adaptation_weight0_to_best_cstim_summary.csv"
FIGURE_STEM = "target_adaptation_weight0_to_best_cstim_grid_cached"

ROW_SPECS = [
    ("mean_best", "absolute", "Model-set mean-best", "Mixed RSA"),
    ("mean_best", "delta", "Model-set mean-best", "Delta mixed RSA vs w0"),
    ("model_best", "absolute", "Per-model best", "Mixed RSA"),
    ("model_best", "delta", "Per-model best", "Delta mixed RSA vs w0"),
]

FONT_TICK = 6
FONT_NOTE = 8
FONT_AXIS = 9
FONT_TITLE = FONT.get("small", 10)
POINT_SIZE = 11
BOX_WIDTH = 0.28
DELTA_OFFSET = 0.19
ABS_OFFSETS = {
    "cstim_w0": -0.31,
    "cstim_best": -0.13,
    "vicco_w0": 0.13,
    "vicco_best": 0.31,
}
PAIR_LINE_ALPHA = 0.13
PAIR_LINE_WIDTH = 0.26


def load_weight_scores() -> pd.DataFrame:
    df = pd.read_csv(SCORE_CSV)
    df["target_weight"] = df["target_weight"].astype(float)
    df = df[df["eval_target"].isin(["cstim_loso", "vicco_heldout"])].copy()
    keys = ["subject", "model", "display_name", "selected_layer", "model_set", "target_weight"]
    cstim = df[df["eval_target"].eq("cstim_loso")][keys + ["mrsa_loso"]].rename(
        columns={"mrsa_loso": "cstim_loso"}
    )
    vicco = df[df["eval_target"].eq("vicco_heldout")][keys + ["mrsa_loso"]].rename(
        columns={"mrsa_loso": "vicco_heldout"}
    )
    wide = cstim.merge(vicco, on=keys, how="inner", validate="one_to_one")
    if wide.empty:
        raise RuntimeError("No matched CSTIM/Vicco score rows found.")
    return wide


def best_weight_by_delta(block: pd.DataFrame) -> float:
    means = block.groupby("target_weight")["cstim_loso"].mean().sort_index()
    if 0.0 not in means.index:
        raise RuntimeError("Cannot select best weight without target_weight=0 rows.")
    deltas = means - float(means.loc[0.0])
    best = (
        deltas.reset_index(name="mean_cstim_delta")
        .sort_values(["mean_cstim_delta", "target_weight"], ascending=[False, True])
        .iloc[0]
    )
    return float(best["target_weight"])


def build_points_for_choice(
    block: pd.DataFrame,
    *,
    selection_mode: str,
    selected_weights: dict[str, float],
) -> pd.DataFrame:
    base = block[np.isclose(block["target_weight"], 0.0)].copy()
    rows = []
    for model, model_base in base.groupby("model"):
        selected_weight = selected_weights[model]
        best = block[block["model"].eq(model) & np.isclose(block["target_weight"], selected_weight)]
        merge_keys = ["subject", "model", "display_name", "selected_layer", "model_set"]
        merged = model_base[merge_keys + ["cstim_loso", "vicco_heldout"]].merge(
            best[merge_keys + ["cstim_loso", "vicco_heldout"]],
            on=merge_keys,
            how="inner",
            suffixes=("_w0", "_best"),
            validate="one_to_one",
        )
        merged["selection_mode"] = selection_mode
        merged["selected_weight"] = selected_weight
        merged["delta_cstim_loso"] = merged["cstim_loso_best"] - merged["cstim_loso_w0"]
        merged["delta_vicco_heldout"] = (
            merged["vicco_heldout_best"] - merged["vicco_heldout_w0"]
        )
        rows.append(merged)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_delta_points(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_set in PANEL_ORDER:
        block = wide[wide["model_set"].eq(model_set)].copy()
        if block.empty:
            continue

        mean_best_weight = best_weight_by_delta(block)
        rows.append(
            build_points_for_choice(
                block,
                selection_mode="mean_best",
                selected_weights={
                    model: mean_best_weight for model in sorted(block["model"].unique())
                },
            )
        )

        rows.append(
            build_points_for_choice(
                block,
                selection_mode="model_best",
                selected_weights={
                    model: best_weight_by_delta(model_block)
                    for model, model_block in block.groupby("model")
                },
            )
        )

    points = pd.concat([row for row in rows if not row.empty], ignore_index=True)
    points = points[
        [
            "selection_mode",
            "model_set",
            "subject",
            "model",
            "display_name",
            "selected_layer",
            "selected_weight",
            "cstim_loso_w0",
            "cstim_loso_best",
            "delta_cstim_loso",
            "vicco_heldout_w0",
            "vicco_heldout_best",
            "delta_vicco_heldout",
        ]
    ].sort_values(["selection_mode", "model_set", "model", "subject"])
    return points.reset_index(drop=True)


def summarize_points(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["selection_mode", "model_set", "model", "display_name", "selected_weight"]
    for keys, block in points.groupby(group_cols, sort=False):
        selection_mode, model_set, model, display_name, selected_weight = keys
        rows.append(
            {
                "selection_mode": selection_mode,
                "model_set": model_set,
                "model": model,
                "display_name": display_name,
                "selected_weight": float(selected_weight),
                "n_subjects": block["subject"].nunique(),
                "mean_cstim_loso_w0": float(block["cstim_loso_w0"].mean()),
                "mean_cstim_loso_best": float(block["cstim_loso_best"].mean()),
                "mean_delta_cstim_loso": float(block["delta_cstim_loso"].mean()),
                "sem_delta_cstim_loso": sem(block["delta_cstim_loso"].to_numpy(dtype=float)),
                "mean_vicco_heldout_w0": float(block["vicco_heldout_w0"].mean()),
                "mean_vicco_heldout_best": float(block["vicco_heldout_best"].mean()),
                "mean_delta_vicco_heldout": float(block["delta_vicco_heldout"].mean()),
                "sem_delta_vicco_heldout": sem(
                    block["delta_vicco_heldout"].to_numpy(dtype=float)
                ),
            }
        )
    return pd.DataFrame(rows)


def short_model_label(model: str) -> str:
    return SHORT_MODEL_NAMES.get(model, config.MODEL_DISPLAY_NAMES.get(model, model))


def row_model_order(
    points: pd.DataFrame,
    *,
    model_set: str,
    mode: str,
    view: str,
) -> list[str]:
    panel = points[points["model_set"].eq(model_set) & points["selection_mode"].eq(mode)].copy()
    if panel.empty:
        return []
    metric_col = "cstim_loso_best" if view == "absolute" else "delta_cstim_loso"
    order = (
        panel.groupby("model")[metric_col]
        .mean()
        .reset_index(name="metric")
        .sort_values(["metric", "model"], ascending=[False, True])["model"]
        .tolist()
    )
    return order


def model_orders(points: pd.DataFrame) -> dict[tuple[str, str, str], list[str]]:
    orders = {}
    for mode, view, _row_label, _ylabel in ROW_SPECS:
        for model_set in PANEL_ORDER:
            orders[(mode, view, model_set)] = row_model_order(
                points,
                model_set=model_set,
                mode=mode,
                view=view,
            )
    return orders


def draw_mean_sem_box(
    ax,
    x: float,
    vals: np.ndarray,
    *,
    color: str,
    alpha: float,
    width: float = BOX_WIDTH,
) -> None:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return
    mean = float(vals.mean())
    median = float(np.median(vals))
    err = sem(vals)
    lo, hi = mean - err, mean + err
    if hi - lo < 1e-5:
        lo, hi = mean - 0.0008, mean + 0.0008
    ax.add_patch(
        plt.Rectangle(
            (x - width / 2, lo),
            width,
            hi - lo,
            facecolor=color,
            edgecolor=color,
            linewidth=0.55,
            alpha=alpha,
            zorder=3,
        )
    )
    ax.hlines(mean, x - width / 2, x + width / 2, colors="white", linewidth=0.7, zorder=5)
    ax.scatter(
        x,
        median,
        marker="D",
        s=12,
        facecolor="0.12",
        edgecolor="white",
        linewidth=0.25,
        zorder=7,
    )


def scatter_value(
    ax,
    x: float,
    y: float,
    *,
    color: str,
    filled: bool,
) -> None:
    ax.scatter(
        x,
        y,
        s=POINT_SIZE,
        facecolors=color if filled else "white",
        edgecolors="white" if filled else color,
        linewidths=0.35,
        alpha=0.92,
        zorder=6,
    )


def panel_weight_label(panel: pd.DataFrame, mode: str) -> str:
    weights = sorted(float(w) for w in panel["selected_weight"].unique())
    if not weights:
        return ""
    if mode == "mean_best":
        return f"mean-best w={weights[0]:g}"
    return "per-model w*"


def set_x_labels(ax, panel: pd.DataFrame, order: list[str], mode: str) -> None:
    labels = []
    for model in order:
        row = panel[panel["model"].eq(model)].iloc[0]
        label = short_model_label(model)
        if mode == "model_best":
            label = f"{label}\nw={float(row.selected_weight):g}"
        labels.append(label)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels, rotation=48, ha="right", fontsize=FONT_TICK)


def draw_absolute_panel(
    ax,
    panel: pd.DataFrame,
    order: list[str],
) -> None:
    x = np.arange(len(order))
    for i, model in enumerate(order):
        block = panel[panel["model"].eq(model)]
        if block.empty:
            continue

        for row in block.itertuples():
            ax.plot(
                [x[i] + ABS_OFFSETS["cstim_w0"], x[i] + ABS_OFFSETS["cstim_best"]],
                [row.cstim_loso_w0, row.cstim_loso_best],
                color=COLOR_CSTIM,
                linewidth=PAIR_LINE_WIDTH,
                alpha=PAIR_LINE_ALPHA,
                zorder=2,
            )
            ax.plot(
                [x[i] + ABS_OFFSETS["vicco_w0"], x[i] + ABS_OFFSETS["vicco_best"]],
                [row.vicco_heldout_w0, row.vicco_heldout_best],
                color=COLOR_BASELINE,
                linewidth=PAIR_LINE_WIDTH,
                alpha=PAIR_LINE_ALPHA,
                zorder=2,
            )
            scatter_value(
                ax,
                x[i] + ABS_OFFSETS["cstim_w0"],
                row.cstim_loso_w0,
                color=COLOR_CSTIM,
                filled=False,
            )
            scatter_value(
                ax,
                x[i] + ABS_OFFSETS["cstim_best"],
                row.cstim_loso_best,
                color=COLOR_CSTIM,
                filled=True,
            )
            scatter_value(
                ax,
                x[i] + ABS_OFFSETS["vicco_w0"],
                row.vicco_heldout_w0,
                color=COLOR_BASELINE,
                filled=False,
            )
            scatter_value(
                ax,
                x[i] + ABS_OFFSETS["vicco_best"],
                row.vicco_heldout_best,
                color=COLOR_BASELINE,
                filled=True,
            )

        draw_mean_sem_box(
            ax,
            x[i] + ABS_OFFSETS["cstim_w0"],
            block["cstim_loso_w0"].to_numpy(dtype=float),
            color=COLOR_CSTIM,
            alpha=0.22,
        )
        draw_mean_sem_box(
            ax,
            x[i] + ABS_OFFSETS["cstim_best"],
            block["cstim_loso_best"].to_numpy(dtype=float),
            color=COLOR_CSTIM,
            alpha=0.70,
        )
        draw_mean_sem_box(
            ax,
            x[i] + ABS_OFFSETS["vicco_w0"],
            block["vicco_heldout_w0"].to_numpy(dtype=float),
            color=COLOR_BASELINE,
            alpha=0.22,
        )
        draw_mean_sem_box(
            ax,
            x[i] + ABS_OFFSETS["vicco_best"],
            block["vicco_heldout_best"].to_numpy(dtype=float),
            color=COLOR_BASELINE,
            alpha=0.70,
        )


def draw_delta_panel(
    ax,
    panel: pd.DataFrame,
    order: list[str],
) -> None:
    x = np.arange(len(order))
    for i, model in enumerate(order):
        block = panel[panel["model"].eq(model)]
        if block.empty:
            continue
        c_vals = block["delta_cstim_loso"].to_numpy(dtype=float)
        v_vals = block["delta_vicco_heldout"].to_numpy(dtype=float)
        draw_mean_sem_box(ax, x[i] - DELTA_OFFSET, c_vals, color=COLOR_CSTIM, alpha=0.70)
        draw_mean_sem_box(
            ax,
            x[i] + DELTA_OFFSET,
            v_vals,
            color=COLOR_BASELINE,
            alpha=0.70,
        )

        for row in block.itertuples():
            ax.plot(
                [x[i] - DELTA_OFFSET, x[i] + DELTA_OFFSET],
                [row.delta_cstim_loso, row.delta_vicco_heldout],
                color="#777777",
                linewidth=PAIR_LINE_WIDTH,
                alpha=PAIR_LINE_ALPHA,
                zorder=2,
            )
            scatter_value(
                ax,
                x[i] - DELTA_OFFSET,
                row.delta_cstim_loso,
                color=COLOR_CSTIM,
                filled=True,
            )
            scatter_value(
                ax,
                x[i] + DELTA_OFFSET,
                row.delta_vicco_heldout,
                color=COLOR_BASELINE,
                filled=True,
            )

    ax.axhline(0, color="0.35", linestyle="--", linewidth=0.75, alpha=0.75, zorder=1)


def draw_panel(
    ax,
    points: pd.DataFrame,
    model_set: str,
    mode: str,
    view: str,
    order: list[str],
    *,
    show_xticks: bool,
) -> None:
    panel = points[points["model_set"].eq(model_set) & points["selection_mode"].eq(mode)].copy()
    if panel.empty:
        ax.set_visible(False)
        return

    if view == "absolute":
        draw_absolute_panel(ax, panel, order)
    else:
        draw_delta_panel(ax, panel, order)

    ax.grid(axis="y", alpha=0.22, linewidth=0.45)
    ax.set_xlim(-0.70, len(order) - 0.30)
    ax.text(
        0.02,
        0.94,
        panel_weight_label(panel, mode),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_NOTE,
        color="0.24",
    )
    if show_xticks:
        set_x_labels(ax, panel, order, mode)
    else:
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=FONT_TICK)


def plot_delta_grid(points: pd.DataFrame, summary: pd.DataFrame) -> None:
    orders = model_orders(points)
    abs_values = points[
        ["cstim_loso_w0", "cstim_loso_best", "vicco_heldout_w0", "vicco_heldout_best"]
    ].to_numpy(dtype=float)
    abs_values = abs_values[np.isfinite(abs_values)]
    abs_ylim = (-0.035, float(abs_values.max()) * 1.05)

    all_deltas = points[["delta_cstim_loso", "delta_vicco_heldout"]].to_numpy(dtype=float)
    all_deltas = all_deltas[np.isfinite(all_deltas)]
    delta_span = max(float(np.max(np.abs(all_deltas))) * 1.12, 0.015)
    delta_ylim = (-delta_span, delta_span)

    ratios = [
        max(
            summary[summary["model_set"].eq(model_set)]["model"].nunique(),
            1,
        )
        for model_set in PANEL_ORDER
    ]
    fig = plt.figure(figsize=(W_DOUBLE, 15.2))
    gs = fig.add_gridspec(
        len(ROW_SPECS),
        len(PANEL_ORDER),
        width_ratios=ratios,
        wspace=0.08,
        hspace=0.56,
        left=0.055,
        right=0.995,
        top=0.945,
        bottom=0.075,
    )

    for row_idx, (mode, view, row_label, ylabel) in enumerate(ROW_SPECS):
        for col_idx, model_set in enumerate(PANEL_ORDER):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.set_ylim(*(abs_ylim if view == "absolute" else delta_ylim))
            draw_panel(
                ax,
                points,
                model_set,
                mode,
                view,
                orders[(mode, view, model_set)],
                show_xticks=True,
            )
            if row_idx == 0:
                ax.set_title(
                    MODEL_SET_TITLES[model_set],
                    y=0.995,
                    fontsize=FONT_TITLE,
                    fontweight="bold",
                )
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=FONT_AXIS)
                ax.text(
                    -0.24,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=FONT_AXIS,
                    fontweight="bold",
                    clip_on=False,
                )
            else:
                ax.set_ylabel("")
                ax.spines["left"].set_visible(False)
                ax.tick_params(axis="y", left=False, labelleft=False)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLOR_CSTIM,
            markeredgecolor=COLOR_CSTIM,
            markersize=4.8,
            label="CSTIM LOSO",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLOR_BASELINE,
            markeredgecolor=COLOR_BASELINE,
            markersize=4.8,
            label="Held-out Vicco",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="0.25",
            markersize=4.8,
            label="weight 0",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="0.25",
            markeredgecolor="0.25",
            markersize=4.8,
            label="selected best",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="0.12",
            markeredgecolor="0.12",
            markersize=4.2,
            label="median",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.987),
        ncol=5,
        frameon=False,
        fontsize=FONT.get("small", 10),
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES_DIR / f"{FIGURE_STEM}.pdf"
    png = PNG_DIR / f"{FIGURE_STEM}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=DPI)
    print(f"Saved {pdf}")
    print(f"Saved {png}")
    plt.close(fig)


def main() -> None:
    wide = load_weight_scores()
    points = build_delta_points(wide)
    summary = summarize_points(points)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    points.to_csv(POINTS_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved {POINTS_CSV}")
    print(f"Saved {SUMMARY_CSV}")
    plot_delta_grid(points, summary)


if __name__ == "__main__":
    main()
