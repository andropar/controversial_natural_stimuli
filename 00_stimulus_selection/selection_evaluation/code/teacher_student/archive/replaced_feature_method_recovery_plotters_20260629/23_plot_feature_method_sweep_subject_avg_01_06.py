#!/usr/bin/env python3
"""Create subject-averaged revisions of feature-method sweep figures 01-06."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


SCRIPT = Path(__file__).resolve()


def find_eval_root(path: Path) -> Path:
    for parent in path.parents:
        if parent.name == "selection_evaluation":
            return parent
    raise RuntimeError(f"Could not find selection_evaluation parent for {path}")


EVAL_ROOT = find_eval_root(SCRIPT)
DEFAULT_RESULTS_DIR = (
    EVAL_ROOT
    / "feature_method_sweep_recovery"
    / "teacher_student"
    / "results"
    / "pool_size_sweep_new_methods_20260617_161534_refit1000"
    / "teacher_student_independent_refit_refit1000_rdm_score_spearman_response_empcal_ns20_rand100_rr3_fastgpu"
)
DEFAULT_PLOTS_DIR = DEFAULT_RESULTS_DIR / "plots" / "subject_avg_01_06"

SUBJECT_TRACKS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
MODEL_SET_ORDER = ["sota", "training_objective", "architecture", "dataset"]
MODEL_SET_LABELS = {
    "sota": "SOTA",
    "training_objective": "Training objective",
    "architecture": "Architecture",
    "dataset": "Dataset",
}
POOL_SIZE_ORDER = [
    1_000,
    10_000,
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    5_000_000,
    10_000_000,
]
METHOD_ORDER = [
    "raw_only_mean_min",
    "raw_only_mean_min_no_attenuation",
    "sub01_only_mean_min",
    "sub01_only_mean_min_no_attenuation",
    "raw_enc_w05_mean_min",
    "raw_enc_w05_mean_min_no_attenuation",
]
METHOD_LABELS = {
    "raw_only_mean_min": "Raw",
    "raw_only_mean_min_no_attenuation": "Raw no atten.",
    "sub01_only_mean_min": "Sub-01",
    "sub01_only_mean_min_no_attenuation": "Sub-01 no atten.",
    "raw_enc_w05_mean_min": "Raw+enc",
    "raw_enc_w05_mean_min_no_attenuation": "Raw+enc no atten.",
}
METHOD_COLORS = {
    "raw_only_mean_min": "#5477C4",
    "raw_only_mean_min_no_attenuation": "#A3BEFA",
    "sub01_only_mean_min": "#CC6F47",
    "sub01_only_mean_min_no_attenuation": "#F0986E",
    "raw_enc_w05_mean_min": "#71B436",
    "raw_enc_w05_mean_min_no_attenuation": "#A3D576",
}
TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
    "random": "#464C55",
}


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "savefig.edgecolor": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.titlecolor": TOKENS["ink"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.75,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Aptos",
                "Inter",
                "Segoe UI",
                "DejaVu Sans",
                "Arial",
                "sans-serif",
            ],
        },
    )


def require_columns(df: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def add_method_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["selection_method_id"].isin(METHOD_ORDER)].copy()
    out["method_label"] = out["selection_method_id"].map(METHOD_LABELS)
    out["method_order"] = out["selection_method_id"].map(
        {method_id: idx for idx, method_id in enumerate(METHOD_ORDER)}
    )
    return out


def join_sorted(values: pd.Series) -> str:
    return ",".join(sorted({str(value) for value in values.dropna()}))


def pool_label(value: float, _position: Optional[int] = None) -> str:
    value = float(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def snr_label(value: float, _position: Optional[int] = None) -> str:
    if value >= 1:
        return f"{value:g}"
    return f"{value:.2g}"


def sorted_present(values: pd.Series, preferred: list[str]) -> list[str]:
    present = list(dict.fromkeys(values.dropna().astype(str).tolist()))
    ordered = [value for value in preferred if value in present]
    ordered.extend([value for value in present if value not in ordered])
    return ordered


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(
        0.055,
        0.985,
        title,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.055,
        0.952,
        subtitle,
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
    )


def format_pool_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.set_xlim(min(POOL_SIZE_ORDER) * 0.82, max(POOL_SIZE_ORDER) * 1.22)
    ax.set_xticks(POOL_SIZE_ORDER)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(pool_label))
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)


def finish_axes(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, color=TOKENS["grid"], linewidth=0.75)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def subject_average_metric(metric: pd.DataFrame) -> pd.DataFrame:
    required = [
        "selection_model_set",
        "pool_size",
        "selection_method_id",
        "track",
        "subset_type",
        "recovery_accuracy",
    ]
    require_columns(metric, required, Path("empirical_snr_by_pool_method_track.csv"))
    data = add_method_metadata(metric)
    data = data[data["track"].isin(SUBJECT_TRACKS)].copy()
    if data.empty:
        raise ValueError("No subject-track rows remained after filtering.")
    keys = [
        "selection_model_set",
        "pool_size",
        "selection_method_id",
        "method_label",
        "method_order",
        "subset_type",
    ]
    return (
        data.groupby(keys, as_index=False, sort=False)
        .agg(
            recovery_accuracy=("recovery_accuracy", "mean"),
            n_subject_tracks=("track", "nunique"),
            subject_tracks=("track", join_sorted),
        )
        .assign(track="subject_avg")
        .sort_values(["selection_model_set", "method_order", "pool_size", "subset_type"])
    )


def selected_pool_average(subject_metric: pd.DataFrame) -> pd.DataFrame:
    selected = subject_metric[subject_metric["subset_type"] == "selected"].copy()
    keys = ["pool_size", "selection_method_id", "method_label", "method_order"]
    return (
        selected.groupby(keys, as_index=False, sort=False)
        .agg(
            recovery_accuracy=("recovery_accuracy", "mean"),
            n_model_sets=("selection_model_set", "nunique"),
            n_subject_tracks=("n_subject_tracks", "max"),
            subject_tracks=("subject_tracks", join_sorted),
        )
        .assign(track="subject_avg")
        .sort_values(["method_order", "pool_size"])
    )


def lift_by_model(subject_metric: pd.DataFrame) -> pd.DataFrame:
    pivot = subject_metric.pivot_table(
        index=[
            "selection_model_set",
            "pool_size",
            "selection_method_id",
            "method_label",
            "method_order",
            "n_subject_tracks",
            "subject_tracks",
        ],
        columns="subset_type",
        values="recovery_accuracy",
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.dropna(subset=["selected", "random"]).copy()
    pivot["lift"] = pivot["selected"] - pivot["random"]
    pivot["lift_pp"] = 100.0 * pivot["lift"]
    pivot["track"] = "subject_avg"
    return pivot.sort_values(["selection_model_set", "method_order", "pool_size"])


def lift_pool_average(lift_model: pd.DataFrame) -> pd.DataFrame:
    keys = ["pool_size", "selection_method_id", "method_label", "method_order"]
    return (
        lift_model.groupby(keys, as_index=False, sort=False)
        .agg(
            random=("random", "mean"),
            selected=("selected", "mean"),
            lift=("lift", "mean"),
            lift_pp=("lift_pp", "mean"),
            n_model_sets=("selection_model_set", "nunique"),
            n_subject_tracks=("n_subject_tracks", "max"),
            subject_tracks=("subject_tracks", join_sorted),
        )
        .assign(track="subject_avg")
        .sort_values(["method_order", "pool_size"])
    )


def subject_average_noise_curves(combined: pd.DataFrame) -> pd.DataFrame:
    required = [
        "selection_model_set",
        "pool_size",
        "selection_method_id",
        "track",
        "subset_type",
        "relative_snr",
        "recovery_accuracy",
    ]
    require_columns(combined, required, Path("combined_discriminability.csv"))
    data = add_method_metadata(combined)
    data = data[data["track"].isin(SUBJECT_TRACKS)].copy()
    if data.empty:
        raise ValueError("No subject-track rows remained in combined noise curves.")
    keys = [
        "selection_model_set",
        "pool_size",
        "selection_method_id",
        "method_label",
        "method_order",
        "relative_snr",
        "subset_type",
    ]
    return (
        data.groupby(keys, as_index=False, sort=False)
        .agg(
            recovery_accuracy=("recovery_accuracy", "mean"),
            n_subject_tracks=("track", "nunique"),
            subject_tracks=("track", join_sorted),
        )
        .assign(track="subject_avg")
        .sort_values(
            [
                "selection_model_set",
                "method_order",
                "pool_size",
                "subset_type",
                "relative_snr",
            ]
        )
    )


def draw_method_lines(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    linestyle: str = "-",
    alpha: float = 1.0,
    linewidth: float = 1.65,
    label_prefix: str = "",
) -> None:
    for method_id in METHOD_ORDER:
        part = data[data["selection_method_id"] == method_id].sort_values(x_col)
        if part.empty:
            continue
        label = f"{label_prefix}{METHOD_LABELS[method_id]}".strip()
        ax.plot(
            part[x_col].to_numpy(),
            part[y_col].to_numpy(),
            color=METHOD_COLORS[method_id],
            linestyle=linestyle,
            linewidth=linewidth,
            marker="o" if x_col == "pool_size" else None,
            markersize=3.5,
            alpha=alpha,
            label=label,
        )


def method_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=METHOD_COLORS[method_id], linewidth=1.8, label=METHOD_LABELS[method_id])
        for method_id in METHOD_ORDER
    ]


def plot_01_selected_recovery(data: pd.DataFrame, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    add_header(
        fig,
        "Selected recovery by pool size, subject average",
        "Empirical-SNR selected rows; subject tracks sub-01, sub-03, sub-05, sub-06, and sub-07 averaged, then model sets averaged.",
    )
    draw_method_lines(ax, data, x_col="pool_size", y_col="recovery_accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Recovery accuracy")
    ax.set_xlabel("Candidate pool size")
    format_pool_axis(ax)
    finish_axes(ax)
    fig.legend(
        handles=method_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.13, top=0.72)
    save_figure(fig, out_base)


def plot_02_lift(data: pd.DataFrame, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    add_header(
        fig,
        "Selected-minus-random lift by pool size, subject average",
        "Lift is selected recovery minus random recovery in percentage points; subject tracks and model sets are averaged.",
    )
    ax.axhline(0, color=TOKENS["random"], linestyle=":", linewidth=1.5)
    draw_method_lines(ax, data, x_col="pool_size", y_col="lift_pp")
    y_pad = 2.0
    ax.set_ylim(min(-2, float(data["lift_pp"].min()) - y_pad), float(data["lift_pp"].max()) + y_pad)
    ax.set_ylabel("Lift (pp)")
    ax.set_xlabel("Candidate pool size")
    format_pool_axis(ax)
    finish_axes(ax)
    fig.legend(
        handles=[
            *method_legend_handles(),
            Line2D([0], [0], color=TOKENS["random"], linestyle=":", linewidth=1.5, label="Zero lift"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.13, top=0.72)
    save_figure(fig, out_base)


def plot_03_heatmap(data: pd.DataFrame, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    add_header(
        fig,
        "10M selected-minus-random lift, subject average",
        "Heatmap uses 10M candidate pools after averaging subject tracks; columns retain model-set differences.",
    )
    method_order = [METHOD_LABELS[m] for m in METHOD_ORDER]
    model_sets = sorted_present(data["selection_model_set"], MODEL_SET_ORDER)
    matrix = (
        data.pivot_table(
            index="method_label",
            columns="selection_model_set",
            values="lift_pp",
            aggfunc="mean",
        )
        .reindex(index=method_order, columns=model_sets)
    )
    vmax = float(np.nanmax(np.abs(matrix.to_numpy())))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=sns.diverging_palette(25, 220, s=80, l=60, as_cmap=True),
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=1.0,
        linecolor=TOKENS["panel"],
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "Lift (pp)"},
    )
    ax.set_xlabel("Model set")
    ax.set_ylabel("Method")
    ax.set_xticklabels([MODEL_SET_LABELS.get(x.get_text(), x.get_text()) for x in ax.get_xticklabels()], rotation=25, ha="right")
    ax.tick_params(axis="both", labelsize=8.5)
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.18, top=0.80)
    save_figure(fig, out_base)


def plot_04_noise_curves(data: pd.DataFrame, out_base: Path) -> None:
    model_sets = sorted_present(data["selection_model_set"], MODEL_SET_ORDER)
    fig, axes = plt.subplots(len(model_sets), 1, figsize=(11.5, 11.2), sharex=True, sharey=True, squeeze=False)
    add_header(
        fig,
        "All recovery noise curves, subject average",
        "Each panel averages subject tracks; color encodes method, line style encodes selected/random, and opacity increases with pool size.",
    )
    present_pools = set(data["pool_size"].dropna().astype(int).tolist())
    pools = [pool for pool in POOL_SIZE_ORDER if pool in present_pools]
    pools.extend(sorted(present_pools - set(pools)))
    alpha_by_pool = {
        pool: 0.16 + 0.74 * idx / max(len(pools) - 1, 1)
        for idx, pool in enumerate(pools)
    }
    for row_idx, model_set in enumerate(model_sets):
        ax = axes[row_idx][0]
        sub = data[data["selection_model_set"] == model_set]
        for pool in pools:
            pool_df = sub[sub["pool_size"] == pool]
            for subset_type, linestyle in [("random", ":"), ("selected", "-")]:
                subset_df = pool_df[pool_df["subset_type"] == subset_type]
                draw_method_lines(
                    ax,
                    subset_df,
                    x_col="relative_snr",
                    y_col="recovery_accuracy",
                    linestyle=linestyle,
                    alpha=alpha_by_pool[pool],
                    linewidth=1.0,
                )
        ax.set_xscale("log")
        ax.set_xlim(float(data["relative_snr"].min()) * 0.85, float(data["relative_snr"].max()) * 1.18)
        ax.set_ylim(0, 1.02)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(snr_label))
        ax.set_ylabel(f"{MODEL_SET_LABELS.get(model_set, model_set)}\nRecovery")
        finish_axes(ax)
    axes[-1][0].set_xlabel("Relative SNR")
    fig.legend(
        handles=[
            *method_legend_handles(),
            Line2D([0], [0], color=TOKENS["ink"], linestyle="-", linewidth=1.4, label="Selected"),
            Line2D([0], [0], color=TOKENS["ink"], linestyle=":", linewidth=1.4, label="Random"),
            Line2D([0], [0], color=TOKENS["muted"], linewidth=1.0, alpha=0.20, label="1k pool"),
            Line2D([0], [0], color=TOKENS["muted"], linewidth=1.8, alpha=0.90, label="10M pool"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=4,
        frameon=False,
        fontsize=8.2,
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.07, top=0.76, hspace=0.24)
    save_figure(fig, out_base)


def plot_05_recovery_by_model(data: pd.DataFrame, out_base: Path) -> None:
    model_sets = sorted_present(data["selection_model_set"], MODEL_SET_ORDER)
    fig, axes = plt.subplots(len(model_sets), 1, figsize=(11.0, 10.5), sharex=True, sharey=True, squeeze=False)
    add_header(
        fig,
        "Empirical-SNR recovery by pool size and model set, subject average",
        "Subject tracks are averaged; color encodes method and line style separates selected stimuli from random subsets.",
    )
    for row_idx, model_set in enumerate(model_sets):
        ax = axes[row_idx][0]
        sub = data[data["selection_model_set"] == model_set]
        for subset_type, linestyle in [("random", ":"), ("selected", "-")]:
            draw_method_lines(
                ax,
                sub[sub["subset_type"] == subset_type],
                x_col="pool_size",
                y_col="recovery_accuracy",
                linestyle=linestyle,
                linewidth=1.35 if subset_type == "random" else 1.8,
                alpha=0.78 if subset_type == "random" else 1.0,
            )
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(f"{MODEL_SET_LABELS.get(model_set, model_set)}\nRecovery")
        format_pool_axis(ax)
        finish_axes(ax)
    axes[-1][0].set_xlabel("Candidate pool size")
    fig.legend(
        handles=[
            *method_legend_handles(),
            Line2D([0], [0], color=TOKENS["ink"], linestyle="-", linewidth=1.7, label="Selected"),
            Line2D([0], [0], color=TOKENS["ink"], linestyle=":", linewidth=1.7, label="Random"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=4,
        frameon=False,
        fontsize=8.2,
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.07, top=0.76, hspace=0.24)
    save_figure(fig, out_base)


def plot_06_lift_by_model(data: pd.DataFrame, out_base: Path) -> None:
    model_sets = sorted_present(data["selection_model_set"], MODEL_SET_ORDER)
    fig, axes = plt.subplots(len(model_sets), 1, figsize=(11.0, 10.5), sharex=True, sharey=True, squeeze=False)
    add_header(
        fig,
        "Empirical-SNR selected-minus-random lift by pool size and model set, subject average",
        "Lift is selected recovery minus random recovery in percentage points after averaging subject tracks.",
    )
    y_min = min(-2, float(data["lift_pp"].min()) - 2)
    y_max = float(data["lift_pp"].max()) + 2
    for row_idx, model_set in enumerate(model_sets):
        ax = axes[row_idx][0]
        sub = data[data["selection_model_set"] == model_set]
        ax.axhline(0, color=TOKENS["random"], linestyle=":", linewidth=1.4)
        draw_method_lines(ax, sub, x_col="pool_size", y_col="lift_pp")
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel(f"{MODEL_SET_LABELS.get(model_set, model_set)}\nLift (pp)")
        format_pool_axis(ax)
        finish_axes(ax)
    axes[-1][0].set_xlabel("Candidate pool size")
    fig.legend(
        handles=[
            *method_legend_handles(),
            Line2D([0], [0], color=TOKENS["random"], linestyle=":", linewidth=1.4, label="Zero lift"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=4,
        frameon=False,
        fontsize=8.2,
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.07, top=0.76, hspace=0.24)
    save_figure(fig, out_base)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_csv(path: Path, usecols: Optional[list[str]] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, usecols=usecols)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="Defaults to a csv/ subdirectory under --plots-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_dir = args.csv_dir if args.csv_dir is not None else args.plots_dir / "csv"
    use_chart_theme()

    metric_path = args.results_dir / "empirical_snr_by_pool_method_track.csv"
    combined_path = args.results_dir / "combined_discriminability.csv"
    metric = read_csv(metric_path)
    subject_metric = subject_average_metric(metric)
    selected_avg = selected_pool_average(subject_metric)
    lift_model = lift_by_model(subject_metric)
    lift_avg = lift_pool_average(lift_model)
    heatmap_10m = lift_model[lift_model["pool_size"] == 10_000_000].copy()

    combined = read_csv(
        combined_path,
        usecols=[
            "selection_model_set",
            "pool_size",
            "selection_method_id",
            "track",
            "subset_type",
            "relative_snr",
            "recovery_accuracy",
        ],
    )
    noise_curves = subject_average_noise_curves(combined)

    outputs: list[Path] = []
    csv_outputs = {
        "01_avg_subjects_selected_recovery_accuracy_by_pool.csv": selected_avg,
        "02_avg_subjects_selected_minus_random_lift_by_pool.csv": lift_avg,
        "03_avg_subjects_lift_heatmap_10m_by_method_model_set.csv": heatmap_10m,
        "04_avg_subjects_all_noise_curves_plot_data.csv": noise_curves,
        "05_avg_subjects_empirical_snr_recovery_by_pool_model_set.csv": subject_metric,
        "06_avg_subjects_empirical_snr_lift_by_pool_model_set.csv": lift_model,
    }
    for filename, df in csv_outputs.items():
        path = csv_dir / filename
        write_csv(df, path)
        outputs.append(path)

    plot_jobs = [
        (
            "01_avg_subjects_selected_recovery_accuracy_by_pool",
            lambda stem: plot_01_selected_recovery(selected_avg, args.plots_dir / stem),
        ),
        (
            "02_avg_subjects_selected_minus_random_lift_by_pool",
            lambda stem: plot_02_lift(lift_avg, args.plots_dir / stem),
        ),
        (
            "03_avg_subjects_lift_heatmap_10m_by_method_model_set",
            lambda stem: plot_03_heatmap(heatmap_10m, args.plots_dir / stem),
        ),
        (
            "04_avg_subjects_all_noise_curves_model_method_pool_subset",
            lambda stem: plot_04_noise_curves(noise_curves, args.plots_dir / stem),
        ),
        (
            "05_avg_subjects_empirical_snr_recovery_by_pool_model_set",
            lambda stem: plot_05_recovery_by_model(subject_metric, args.plots_dir / stem),
        ),
        (
            "06_avg_subjects_empirical_snr_lift_by_pool_model_set",
            lambda stem: plot_06_lift_by_model(lift_model, args.plots_dir / stem),
        ),
    ]
    for stem, plotter in plot_jobs:
        plotter(stem)
        outputs.extend([args.plots_dir / f"{stem}.png", args.plots_dir / f"{stem}.svg"])

    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
