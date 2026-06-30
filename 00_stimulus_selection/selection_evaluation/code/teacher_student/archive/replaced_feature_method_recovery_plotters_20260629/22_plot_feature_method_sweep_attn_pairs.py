#!/usr/bin/env python3
"""Plot paired attenuation variants for the completed pool-size recovery sweep."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
DEFAULT_PLOTS_DIR = DEFAULT_RESULTS_DIR / "plots" / "attn_pairs"

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

METHOD_PAIRS = {
    "raw_only": (
        "raw_only_mean_min",
        "raw_only_mean_min_no_attenuation",
    ),
    "sub01_only": (
        "sub01_only_mean_min",
        "sub01_only_mean_min_no_attenuation",
    ),
    "raw_enc_w05": (
        "raw_enc_w05_mean_min",
        "raw_enc_w05_mean_min_no_attenuation",
    ),
}
METHOD_FAMILY_ORDER = ["raw_only", "sub01_only", "raw_enc_w05"]
METHOD_FAMILY_LABELS = {
    "raw_only": "Raw only",
    "sub01_only": "Sub-01 only",
    "raw_enc_w05": "Raw + enc w0.5",
}
VARIANT_LABELS = {
    "attenuated": "Attenuated",
    "no_attenuation": "No attenuation",
}
VARIANT_COLORS = {
    "attenuated": "#5477C4",
    "no_attenuation": "#CC6F47",
}
VARIANT_MARKERS = {
    "attenuated": "o",
    "no_attenuation": "s",
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


def method_lookup() -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for family, (attenuated, no_attenuation) in METHOD_PAIRS.items():
        out[attenuated] = (family, "attenuated")
        out[no_attenuation] = (family, "no_attenuation")
    return out


def add_method_metadata(df: pd.DataFrame) -> pd.DataFrame:
    lookup = method_lookup()
    out = df.copy()
    out["method_family"] = out["selection_method_id"].map(
        lambda method: lookup.get(str(method), (None, None))[0]
    )
    out["attenuation_variant"] = out["selection_method_id"].map(
        lambda method: lookup.get(str(method), (None, None))[1]
    )
    out = out.dropna(subset=["method_family", "attenuation_variant"]).copy()
    out["method_family_label"] = out["method_family"].map(METHOD_FAMILY_LABELS)
    out["attenuation_label"] = out["attenuation_variant"].map(VARIANT_LABELS)
    return out


def sorted_present(values: pd.Series, preferred: list) -> list:
    present = list(dict.fromkeys(values.dropna().tolist()))
    ordered = [value for value in preferred if value in present]
    ordered.extend([value for value in present if value not in ordered])
    return ordered


def join_sorted(values: pd.Series) -> str:
    return ",".join(sorted({str(value) for value in values.dropna()}))


def require_columns(df: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def subject_average_by_pool(metric: pd.DataFrame) -> pd.DataFrame:
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
        raise ValueError("No subject-track rows remained after filtering out raw tracks.")
    keys = [
        "selection_model_set",
        "pool_size",
        "selection_method_id",
        "method_family",
        "method_family_label",
        "attenuation_variant",
        "attenuation_label",
        "subset_type",
    ]
    return (
        data.groupby(keys, as_index=False, sort=False)
        .agg(
            recovery_accuracy=("recovery_accuracy", "mean"),
            n_subject_tracks=("track", "nunique"),
            subject_tracks=("track", join_sorted),
        )
        .sort_values(["selection_model_set", "method_family", "pool_size", "subset_type"])
    )


def make_pool_plot_data(metric: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    avg = subject_average_by_pool(metric)
    selected = avg[avg["subset_type"] == "selected"].copy()
    random = avg[avg["subset_type"] == "random"].copy()
    baseline_keys = [
        "selection_model_set",
        "pool_size",
        "method_family",
        "method_family_label",
    ]
    baseline = (
        random.groupby(baseline_keys, as_index=False, sort=False)
        .agg(
            recovery_accuracy=("recovery_accuracy", "mean"),
            n_subject_tracks=("n_subject_tracks", "max"),
            subject_tracks=("subject_tracks", join_sorted),
            n_variant_methods=("selection_method_id", "nunique"),
            random_method_ids=("selection_method_id", join_sorted),
        )
        .assign(
            subset_type="random",
            series_type="random_baseline",
            selection_method_id="shared_random",
            attenuation_variant="shared_random",
            attenuation_label="Shared random",
        )
    )
    selected_plot = selected.assign(
        series_type="selected",
        n_variant_methods=1,
        random_method_ids="",
    )
    plot_columns = [
        "selection_model_set",
        "pool_size",
        "method_family",
        "method_family_label",
        "series_type",
        "selection_method_id",
        "attenuation_variant",
        "attenuation_label",
        "subset_type",
        "recovery_accuracy",
        "n_subject_tracks",
        "subject_tracks",
        "n_variant_methods",
        "random_method_ids",
    ]
    recovery_plot = pd.concat(
        [selected_plot[plot_columns], baseline[plot_columns]],
        ignore_index=True,
    )

    lift = selected.merge(
        baseline[
            [
                "selection_model_set",
                "pool_size",
                "method_family",
                "recovery_accuracy",
                "n_variant_methods",
                "random_method_ids",
            ]
        ].rename(columns={"recovery_accuracy": "shared_random_recovery_accuracy"}),
        on=["selection_model_set", "pool_size", "method_family"],
        how="left",
    )
    lift = lift.rename(columns={"recovery_accuracy": "selected_recovery_accuracy"})
    lift["lift"] = lift["selected_recovery_accuracy"] - lift["shared_random_recovery_accuracy"]
    lift["lift_pp"] = 100.0 * lift["lift"]
    lift = lift[
        [
            "selection_model_set",
            "pool_size",
            "method_family",
            "method_family_label",
            "selection_method_id",
            "attenuation_variant",
            "attenuation_label",
            "selected_recovery_accuracy",
            "shared_random_recovery_accuracy",
            "lift",
            "lift_pp",
            "n_subject_tracks",
            "subject_tracks",
            "n_variant_methods",
            "random_method_ids",
        ]
    ]
    return recovery_plot, lift


def subject_average_noise_curves(combined: pd.DataFrame, pool_size: int) -> pd.DataFrame:
    required = [
        "selection_model_set",
        "pool_size",
        "selection_method_id",
        "track",
        "subset_type",
        "relative_snr",
        "noise_mult",
        "recovery_accuracy",
    ]
    require_columns(combined, required, Path("combined_discriminability.csv"))
    data = add_method_metadata(combined)
    data = data[(data["pool_size"] == pool_size) & data["track"].isin(SUBJECT_TRACKS)].copy()
    if data.empty:
        raise ValueError(f"No subject-track rows remained for pool_size={pool_size}.")
    keys = [
        "selection_model_set",
        "pool_size",
        "selection_method_id",
        "method_family",
        "method_family_label",
        "attenuation_variant",
        "attenuation_label",
        "relative_snr",
        "noise_mult",
        "subset_type",
    ]
    return (
        data.groupby(keys, as_index=False, sort=False)
        .agg(
            recovery_accuracy=("recovery_accuracy", "mean"),
            n_subject_tracks=("track", "nunique"),
            subject_tracks=("track", join_sorted),
        )
        .sort_values(
            [
                "selection_model_set",
                "method_family",
                "selection_method_id",
                "relative_snr",
                "subset_type",
            ]
        )
    )


def make_noise_curve_plot_data(combined: pd.DataFrame, pool_size: int) -> pd.DataFrame:
    avg = subject_average_noise_curves(combined, pool_size)
    selected = avg[avg["subset_type"] == "selected"].copy()
    random = avg[avg["subset_type"] == "random"].copy()
    baseline_keys = [
        "selection_model_set",
        "pool_size",
        "method_family",
        "method_family_label",
        "relative_snr",
        "noise_mult",
    ]
    baseline = (
        random.groupby(baseline_keys, as_index=False, sort=False)
        .agg(
            recovery_accuracy=("recovery_accuracy", "mean"),
            n_subject_tracks=("n_subject_tracks", "max"),
            subject_tracks=("subject_tracks", join_sorted),
            n_variant_methods=("selection_method_id", "nunique"),
            random_method_ids=("selection_method_id", join_sorted),
        )
        .assign(
            subset_type="random",
            series_type="random_baseline",
            selection_method_id="shared_random",
            attenuation_variant="shared_random",
            attenuation_label="Shared random",
        )
    )
    selected_plot = selected.assign(
        series_type="selected",
        n_variant_methods=1,
        random_method_ids="",
    )
    columns = [
        "selection_model_set",
        "pool_size",
        "method_family",
        "method_family_label",
        "series_type",
        "selection_method_id",
        "attenuation_variant",
        "attenuation_label",
        "subset_type",
        "relative_snr",
        "noise_mult",
        "recovery_accuracy",
        "n_subject_tracks",
        "subject_tracks",
        "n_variant_methods",
        "random_method_ids",
    ]
    return pd.concat([selected_plot[columns], baseline[columns]], ignore_index=True)


def pool_label(value: float, _position: Optional[int] = None) -> str:
    value = float(value)
    if value >= 1_000_000:
        amount = value / 1_000_000
        return f"{amount:g}M"
    if value >= 1_000:
        amount = value / 1_000
        return f"{amount:g}k"
    return f"{value:g}"


def snr_label(value: float, _position: Optional[int] = None) -> str:
    if value >= 1:
        return f"{value:g}"
    return f"{value:.2g}"


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
        0.958,
        subtitle,
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
    )


def facet_grid(
    data: pd.DataFrame,
    title: str,
    subtitle: str,
    *,
    figsize: tuple[float, float],
    sharey: bool = True,
) -> tuple[plt.Figure, list[list[plt.Axes]], list[str], list[str]]:
    model_sets = sorted_present(data["selection_model_set"], MODEL_SET_ORDER)
    families = sorted_present(data["method_family"], METHOD_FAMILY_ORDER)
    fig, axes = plt.subplots(
        len(model_sets),
        len(families),
        figsize=figsize,
        sharex=True,
        sharey=sharey,
        squeeze=False,
    )
    add_header(fig, title, subtitle)
    for row_idx, model_set in enumerate(model_sets):
        for col_idx, family in enumerate(families):
            ax = axes[row_idx][col_idx]
            ax.set_axisbelow(True)
            ax.grid(True, axis="both", color=TOKENS["grid"], linewidth=0.75)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color(TOKENS["axis"])
            ax.spines["bottom"].set_color(TOKENS["axis"])
            if row_idx == 0:
                ax.set_title(METHOD_FAMILY_LABELS.get(family, family), fontsize=10.5, pad=9)
            if col_idx == 0:
                ax.set_ylabel(MODEL_SET_LABELS.get(model_set, model_set), fontsize=9.5)
            else:
                ax.set_ylabel("")
    return fig, axes, model_sets, families


def draw_selected_and_random(
    ax: plt.Axes,
    sub: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    selected_label: str,
    random_label: str,
) -> None:
    baseline = sub[sub["series_type"] == "random_baseline"].sort_values(x_col)
    if not baseline.empty:
        ax.plot(
            baseline[x_col].to_numpy(),
            baseline[y_col].to_numpy(),
            color=TOKENS["random"],
            linestyle=":",
            linewidth=1.8,
            label=random_label,
            zorder=2,
        )
    selected = sub[sub["series_type"] == "selected"].copy()
    for variant in ["attenuated", "no_attenuation"]:
        part = selected[selected["attenuation_variant"] == variant].sort_values(x_col)
        if part.empty:
            continue
        ax.plot(
            part[x_col].to_numpy(),
            part[y_col].to_numpy(),
            color=VARIANT_COLORS[variant],
            linestyle="-",
            marker=VARIANT_MARKERS[variant],
            markersize=4.0,
            linewidth=1.65,
            label=f"{VARIANT_LABELS[variant]} {selected_label}",
            zorder=3,
        )


def add_common_legend(fig: plt.Figure, *, lift: bool = False) -> None:
    random_label = "Shared random baseline" if not lift else "Shared random baseline (0 pp)"
    handles = [
        Line2D(
            [0],
            [0],
            color=VARIANT_COLORS["attenuated"],
            marker=VARIANT_MARKERS["attenuated"],
            linewidth=1.65,
            markersize=4,
            label="Attenuated selected",
        ),
        Line2D(
            [0],
            [0],
            color=VARIANT_COLORS["no_attenuation"],
            marker=VARIANT_MARKERS["no_attenuation"],
            linewidth=1.65,
            markersize=4,
            label="No attenuation selected",
        ),
        Line2D(
            [0],
            [0],
            color=TOKENS["random"],
            linestyle=":",
            linewidth=1.8,
            label=random_label,
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.983),
        frameon=False,
        ncol=2,
        fontsize=8.5,
        handlelength=2.4,
        columnspacing=1.4,
    )


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def model_row_figsize(data: pd.DataFrame, *, full_height: float) -> tuple[float, float]:
    n_rows = max(1, len(sorted_present(data["selection_model_set"], MODEL_SET_ORDER)))
    if n_rows >= len(MODEL_SET_ORDER):
        height = full_height
    else:
        per_row = (full_height - 2.2) / len(MODEL_SET_ORDER)
        height = 2.2 + per_row * n_rows
    return (14.5, max(4.3, height))


def plot_recovery_by_pool(data: pd.DataFrame, out_base: Path) -> None:
    fig, axes, model_sets, families = facet_grid(
        data,
        "All-subject recovery by pool size",
        "Subject tracks sub-01, sub-03, sub-05, sub-06, and sub-07 averaged; random baseline is averaged across each attenuated/no-attenuation pair.",
        figsize=model_row_figsize(data, full_height=11.0),
    )
    for row_idx, model_set in enumerate(model_sets):
        for col_idx, family in enumerate(families):
            ax = axes[row_idx][col_idx]
            sub = data[
                (data["selection_model_set"] == model_set)
                & (data["method_family"] == family)
            ]
            draw_selected_and_random(
                ax,
                sub,
                x_col="pool_size",
                y_col="recovery_accuracy",
                selected_label="selected",
                random_label="shared random",
            )
            ax.set_xscale("log")
            ax.set_xlim(min(POOL_SIZE_ORDER) * 0.82, max(POOL_SIZE_ORDER) * 1.22)
            ax.set_ylim(0.0, 1.02)
            ax.set_xticks(POOL_SIZE_ORDER)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(pool_label))
            ax.tick_params(axis="x", labelsize=8, rotation=0)
            ax.tick_params(axis="y", labelsize=8)
            if col_idx == 0:
                ax.set_ylabel(
                    f"{MODEL_SET_LABELS.get(model_set, model_set)}\nRecovery accuracy",
                    fontsize=9.5,
                )
            if row_idx == len(model_sets) - 1:
                ax.set_xlabel("Candidate pool size", fontsize=9)
            else:
                ax.set_xlabel("")
    add_common_legend(fig)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.07, top=0.875, hspace=0.26, wspace=0.10)
    save_figure(fig, out_base)


def plot_lift_by_pool(data: pd.DataFrame, out_base: Path) -> None:
    fig, axes, model_sets, families = facet_grid(
        data,
        "All-subject selected-minus-random lift by pool size",
        "Lift is selected recovery minus the paired shared random baseline, expressed in percentage points across averaged subject tracks.",
        figsize=model_row_figsize(data, full_height=11.0),
        sharey=True,
    )
    y_min = min(-2.0, float(data["lift_pp"].min()) - 2.0)
    y_max = max(2.0, float(data["lift_pp"].max()) + 2.0)
    for row_idx, model_set in enumerate(model_sets):
        for col_idx, family in enumerate(families):
            ax = axes[row_idx][col_idx]
            sub = data[
                (data["selection_model_set"] == model_set)
                & (data["method_family"] == family)
            ]
            ax.axhline(0.0, color=TOKENS["random"], linestyle=":", linewidth=1.8, zorder=1)
            for variant in ["attenuated", "no_attenuation"]:
                part = sub[sub["attenuation_variant"] == variant].sort_values("pool_size")
                if part.empty:
                    continue
                ax.plot(
                    part["pool_size"].to_numpy(),
                    part["lift_pp"].to_numpy(),
                    color=VARIANT_COLORS[variant],
                    linestyle="-",
                    marker=VARIANT_MARKERS[variant],
                    markersize=4.0,
                    linewidth=1.65,
                    zorder=3,
                )
            ax.set_xscale("log")
            ax.set_xlim(min(POOL_SIZE_ORDER) * 0.82, max(POOL_SIZE_ORDER) * 1.22)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(POOL_SIZE_ORDER)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(pool_label))
            ax.tick_params(axis="x", labelsize=8, rotation=0)
            ax.tick_params(axis="y", labelsize=8)
            if col_idx == 0:
                ax.set_ylabel(
                    f"{MODEL_SET_LABELS.get(model_set, model_set)}\nLift (pp)",
                    fontsize=9.5,
                )
            if row_idx == len(model_sets) - 1:
                ax.set_xlabel("Candidate pool size", fontsize=9)
            else:
                ax.set_xlabel("")
    add_common_legend(fig, lift=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.07, top=0.875, hspace=0.26, wspace=0.10)
    save_figure(fig, out_base)


def plot_noise_curves_10m(data: pd.DataFrame, out_base: Path) -> None:
    fig, axes, model_sets, families = facet_grid(
        data,
        "All-subject recovery noise curves at 10M pool size",
        "X axis is relative SNR on a log scale; subject tracks are averaged and each random curve is averaged across paired variants.",
        figsize=model_row_figsize(data, full_height=11.5),
    )
    x_min = float(data["relative_snr"].min()) * 0.85
    x_max = float(data["relative_snr"].max()) * 1.18
    for row_idx, model_set in enumerate(model_sets):
        for col_idx, family in enumerate(families):
            ax = axes[row_idx][col_idx]
            sub = data[
                (data["selection_model_set"] == model_set)
                & (data["method_family"] == family)
            ]
            draw_selected_and_random(
                ax,
                sub,
                x_col="relative_snr",
                y_col="recovery_accuracy",
                selected_label="selected",
                random_label="shared random",
            )
            ax.axvline(1.0, color=TOKENS["muted"], linestyle="--", linewidth=1.0, alpha=0.8)
            ax.set_xscale("log")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0.0, 1.02)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(snr_label))
            ax.tick_params(axis="x", labelsize=8, rotation=0)
            ax.tick_params(axis="y", labelsize=8)
            if col_idx == 0:
                ax.set_ylabel(
                    f"{MODEL_SET_LABELS.get(model_set, model_set)}\nRecovery accuracy",
                    fontsize=9.5,
                )
            if row_idx == len(model_sets) - 1:
                ax.set_xlabel("Relative SNR", fontsize=9)
            else:
                ax.set_xlabel("")
    add_common_legend(fig)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.07, top=0.875, hspace=0.26, wspace=0.10)
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
    parser.add_argument("--noise-pool-size", type=int, default=10_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_chart_theme()

    metric_path = args.results_dir / "empirical_snr_by_pool_method_track.csv"
    combined_path = args.results_dir / "combined_discriminability.csv"
    csv_dir = args.csv_dir if args.csv_dir is not None else args.plots_dir / "csv"

    metric = read_csv(metric_path)
    recovery_pool, lift_pool = make_pool_plot_data(metric)

    combined_cols = [
        "selection_model_set",
        "selection_method_id",
        "pool_size",
        "track",
        "subset_type",
        "relative_snr",
        "noise_mult",
        "recovery_accuracy",
    ]
    combined = read_csv(combined_path, usecols=combined_cols)
    noise_curves = make_noise_curve_plot_data(combined, args.noise_pool_size)

    outputs: list[Path] = []
    recovery_csv = csv_dir / "15_avg_all_subjects_attn_pairs_recovery_by_pool_method_family.csv"
    lift_csv = csv_dir / "16_avg_all_subjects_attn_pairs_lift_by_pool_method_family.csv"
    noise_csv = csv_dir / "17_avg_all_subjects_attn_pairs_noise_curves_10M.csv"
    write_csv(recovery_pool, recovery_csv)
    write_csv(lift_pool, lift_csv)
    write_csv(noise_curves, noise_csv)
    outputs.extend([recovery_csv, lift_csv, noise_csv])

    plot_recovery_by_pool(
        recovery_pool,
        args.plots_dir / "15_avg_all_subjects_attn_pairs_recovery_by_pool_method_family",
    )
    plot_lift_by_pool(
        lift_pool,
        args.plots_dir / "16_avg_all_subjects_attn_pairs_lift_by_pool_method_family",
    )
    plot_noise_curves_10m(
        noise_curves,
        args.plots_dir / "17_avg_all_subjects_attn_pairs_noise_curves_10M",
    )
    for stem in [
        "15_avg_all_subjects_attn_pairs_recovery_by_pool_method_family",
        "16_avg_all_subjects_attn_pairs_lift_by_pool_method_family",
        "17_avg_all_subjects_attn_pairs_noise_curves_10M",
    ]:
        outputs.extend([args.plots_dir / f"{stem}.png", args.plots_dir / f"{stem}.svg"])

    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
