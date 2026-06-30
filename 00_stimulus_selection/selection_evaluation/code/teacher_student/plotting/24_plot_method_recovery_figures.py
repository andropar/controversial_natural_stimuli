#!/usr/bin/env python3
"""Plot method-level teacher/student recovery figures from available outputs.

The script intentionally discovers result files instead of relying on a single
hard-coded run. It supports the pool-size feature-method sweep, refit-robust
outputs, and future fixed-pool methods if their eval ids follow
``model_set__method_id`` or ``model_set__pool_*__method_id``.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


SCRIPT = Path(__file__).resolve()
EVAL_ROOT = next(parent for parent in SCRIPT.parents if parent.name == "selection_evaluation")
DEFAULT_RESULTS_ROOT = (
    EVAL_ROOT / "feature_method_sweep_recovery" / "teacher_student" / "results"
)
DEFAULT_FIGURES_ROOT = (
    EVAL_ROOT / "feature_method_sweep_recovery" / "teacher_student" / "figures"
)
DEFAULT_DATA_DIR = DEFAULT_FIGURES_ROOT / "data" / "method_recovery"

MODEL_SET_ORDER = ["all_models", "sota", "training_objective", "architecture", "dataset"]
MODEL_SET_LABELS = {
    "all_models": "All models",
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
SUBJECT_TRACKS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]

METHOD_FAMILY_ORDER = [
    "raw_only",
    "sub01_only",
    "raw_enc_w05",
    "sub01_max_min",
    "raw_enc_w05_max_min",
    "avg_subjects_enc",
    "refit_robust",
]
METHOD_FAMILY_LABELS = {
    "raw_only": "Raw only",
    "sub01_only": "Sub-01 only",
    "raw_enc_w05": "Raw + enc w0.5",
    "sub01_max_min": "Sub-01 max/min",
    "raw_enc_w05_max_min": "Raw + enc max/min",
    "avg_subjects_enc": "Avg-subject enc",
    "refit_robust": "Refit robust",
}
METHOD_METADATA = {
    "raw_only_mean_min": ("raw_only", "attenuated", "Attenuated"),
    "raw_only_mean_min_no_attenuation": (
        "raw_only",
        "no_attenuation",
        "No attenuation",
    ),
    "sub01_only_mean_min": ("sub01_only", "attenuated", "Attenuated"),
    "sub01_only_mean_min_no_attenuation": (
        "sub01_only",
        "no_attenuation",
        "No attenuation",
    ),
    "raw_enc_w05_mean_min": ("raw_enc_w05", "attenuated", "Attenuated"),
    "raw_enc_w05_mean_min_no_attenuation": (
        "raw_enc_w05",
        "no_attenuation",
        "No attenuation",
    ),
    "sub01_only_max_min": ("sub01_max_min", "max_min", "Max/min"),
    "raw_enc_w05_max_min": ("raw_enc_w05_max_min", "max_min", "Max/min"),
    "avg_subjects_enc_mean_min": (
        "avg_subjects_enc",
        "avg_subjects",
        "Avg subjects",
    ),
    "sub01_eval_augmented_loo_refit_robust": (
        "refit_robust",
        "refit_robust",
        "Refit robust",
    ),
}
VARIANT_ORDER = [
    "attenuated",
    "no_attenuation",
    "max_min",
    "avg_subjects",
    "refit_robust",
]
VARIANT_COLORS = {
    "attenuated": "#5477C4",
    "no_attenuation": "#CC6F47",
    "max_min": "#6F9E3E",
    "avg_subjects": "#7C5CC4",
    "refit_robust": "#2B8C84",
}
VARIANT_MARKERS = {
    "attenuated": "o",
    "no_attenuation": "s",
    "max_min": "D",
    "avg_subjects": "^",
    "refit_robust": "P",
}
TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#202431",
    "muted": "#687083",
    "grid": "#E6E9F1",
    "axis": "#D5DAE6",
    "random": "#3B4048",
}


@dataclass(frozen=True)
class EvalId:
    model_set: str
    method_id: str
    pool_dir: str
    pool_size: float


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


def parse_pool_size(pool_dir: str) -> float:
    if not pool_dir.startswith("pool_"):
        return float("nan")
    return float(int(pool_dir.split("_", 1)[1]))


def parse_eval_id(eval_id: str) -> EvalId | None:
    robust_suffix = "_sub01_eval_augmented_loo_refit_robust"
    if eval_id.endswith(robust_suffix):
        return EvalId(
            model_set=eval_id[: -len(robust_suffix)],
            method_id="sub01_eval_augmented_loo_refit_robust",
            pool_dir="",
            pool_size=float("nan"),
        )
    parts = eval_id.split("__")
    if len(parts) >= 3 and parts[1].startswith("pool_"):
        return EvalId(
            model_set=parts[0],
            method_id="__".join(parts[2:]),
            pool_dir=parts[1],
            pool_size=parse_pool_size(parts[1]),
        )
    if len(parts) >= 2:
        return EvalId(
            model_set=parts[0],
            method_id="__".join(parts[1:]),
            pool_dir="",
            pool_size=float("nan"),
        )
    return None


def method_metadata(method_id: str) -> tuple[str, str, str] | None:
    return METHOD_METADATA.get(method_id)


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def discover_recovery(results_root: Path, condition: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(results_root.rglob("discriminability.csv")):
        if "_teacher_cache" in path.parts or "payloads" in path.parts:
            continue
        parsed = parse_eval_id(path.parent.name)
        if parsed is None:
            continue
        meta = method_metadata(parsed.method_id)
        if meta is None:
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"Skipping unreadable CSV {path}: {exc}")
            continue
        if df.empty:
            continue
        if "rdm_calibration_comparison" not in df.columns:
            continue
        df = df[df["rdm_calibration_comparison"].astype(str) == condition].copy()
        if df.empty:
            continue
        if "eval_refit_mode" in df.columns:
            df = df[df["eval_refit_mode"].fillna("").astype(str) == "eval_augmented_loo"]
        if df.empty:
            continue
        family, variant, variant_label = meta
        df["eval_id"] = path.parent.name
        df["selection_model_set"] = parsed.model_set
        df["selection_method_id"] = parsed.method_id
        df["pool_dir"] = parsed.pool_dir
        df["pool_size"] = parsed.pool_size
        df["method_family"] = family
        df["method_family_label"] = METHOD_FAMILY_LABELS[family]
        df["method_variant"] = variant
        df["method_variant_label"] = variant_label
        df["source_csv"] = relative_path(path, results_root)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["relative_snr"] = out["relative_snr"].astype(float)
    out["noise_mult"] = out["noise_mult"].astype(float)
    out["recovery_accuracy"] = out["recovery_accuracy"].astype(float)
    out["pool_size"] = pd.to_numeric(out["pool_size"], errors="coerce")
    return out


def ordered_present(values: Iterable[str], preferred: list[str]) -> list[str]:
    present = list(dict.fromkeys(pd.Series(list(values)).dropna().astype(str)))
    return [x for x in preferred if x in present] + [x for x in present if x not in preferred]


def format_pool(value: float, _position: int | None = None) -> str:
    if not np.isfinite(value):
        return ""
    value = float(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def format_snr(value: float, _position: int | None = None) -> str:
    if value >= 1:
        return f"{value:g}"
    return f"{value:.2g}"


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_base.parent / "pdf" / out_base.with_suffix(".pdf").name
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def finish_panel(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, color=TOKENS["grid"], linewidth=0.75)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.tick_params(axis="both", labelsize=8)


def build_plot_tracks(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    raw = data[data["track"].astype(str) == "raw"].copy()
    if not raw.empty:
        raw["plot_track"] = "raw"
        rows.append(raw)

    subject = data[data["track"].astype(str).isin(SUBJECT_TRACKS)].copy()
    if not subject.empty:
        keys = [
            "eval_id",
            "selection_model_set",
            "selection_method_id",
            "pool_dir",
            "pool_size",
            "method_family",
            "method_family_label",
            "method_variant",
            "method_variant_label",
            "subset_type",
            "relative_snr",
            "noise_mult",
            "rdm_calibration_comparison",
            "eval_refit_mode",
            "source_csv",
        ]
        keys = [key for key in keys if key in subject.columns]
        avg = (
            subject.groupby(keys, as_index=False, sort=False, dropna=False)
            .agg(
                recovery_accuracy=("recovery_accuracy", "mean"),
                n_subject_tracks=("track", "nunique"),
                subject_tracks=("track", lambda s: ",".join(sorted(set(map(str, s))))),
            )
            .assign(track="subject_avg", plot_track="subject_avg")
        )
        rows.append(avg)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def final_display_rows(plot_tracks: pd.DataFrame, pool_size: int) -> pd.DataFrame:
    target_pool = np.isclose(plot_tracks["pool_size"], float(pool_size), equal_nan=False)
    refit_robust = plot_tracks["method_family"].astype(str) == "refit_robust"
    out = plot_tracks[
        (plot_tracks["plot_track"] == "subject_avg") & (target_pool | refit_robust)
    ].copy()
    out = out[(out["relative_snr"] >= 0.1) & (out["relative_snr"] <= 10.0)].copy()
    return out


def draw_snr_panel(ax: plt.Axes, sub: pd.DataFrame) -> None:
    if sub.empty:
        ax.set_visible(False)
        return
    random = sub[sub["subset_type"].astype(str) == "random"]
    if not random.empty:
        baseline = (
            random.groupby("relative_snr", as_index=False, sort=True)
            .agg(recovery_accuracy=("recovery_accuracy", "mean"))
            .sort_values("relative_snr")
        )
        ax.plot(
            baseline["relative_snr"].to_numpy(),
            baseline["recovery_accuracy"].to_numpy(),
            color=TOKENS["random"],
            linestyle=":",
            linewidth=1.8,
            zorder=2,
        )
    selected = sub[sub["subset_type"].astype(str) == "selected"]
    for variant in ordered_present(selected["method_variant"], VARIANT_ORDER):
        part = selected[selected["method_variant"].astype(str) == variant].sort_values(
            "relative_snr"
        )
        if part.empty:
            continue
        ax.plot(
            part["relative_snr"].to_numpy(),
            part["recovery_accuracy"].to_numpy(),
            color=VARIANT_COLORS.get(variant, "#555555"),
            marker=VARIANT_MARKERS.get(variant, "o"),
            linewidth=1.7,
            markersize=3.6,
            zorder=3,
        )
    ax.axvline(1.0, color=TOKENS["muted"], linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.085, 11.8)
    ax.set_ylim(0.0, 1.02)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_snr))
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
    finish_panel(ax)


def plot_final_snr(data: pd.DataFrame, condition: str, pool_size: int, figures_root: Path) -> pd.DataFrame:
    plot_data = final_display_rows(build_plot_tracks(data), pool_size)
    if plot_data.empty:
        return plot_data
    model_sets = ordered_present(plot_data["selection_model_set"], MODEL_SET_ORDER)
    families = ordered_present(plot_data["method_family"], METHOD_FAMILY_ORDER)
    fig, axes = plt.subplots(
        len(model_sets),
        len(families),
        figsize=(3.9 * len(families), 2.35 * len(model_sets) + 1.4),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for row_idx, model_set in enumerate(model_sets):
        for col_idx, family in enumerate(families):
            ax = axes[row_idx][col_idx]
            sub = plot_data[
                (plot_data["selection_model_set"] == model_set)
                & (plot_data["method_family"] == family)
            ]
            draw_snr_panel(ax, sub)
            if row_idx == 0:
                ax.set_title(METHOD_FAMILY_LABELS.get(family, family), fontsize=10.5, pad=8)
            if col_idx == 0:
                ax.set_ylabel(
                    f"{MODEL_SET_LABELS.get(model_set, model_set)}\nRecovery accuracy",
                    fontsize=9.5,
                )
            else:
                ax.set_ylabel("")
            if row_idx == len(model_sets) - 1:
                ax.set_xlabel("Relative SNR", fontsize=9)
            else:
                ax.set_xlabel("")
    title = f"Teacher/student recovery by selection method ({condition.replace('_', ' ')})"
    if "refit_robust" in set(plot_data["method_family"].astype(str)):
        subtitle = (
            f"Pool-sweep methods use {format_pool(pool_size)} selected stimuli and subject-averaged "
            "tracks; refit robust is shown from its fixed selection evals where available."
        )
    else:
        subtitle = (
            f"Pool-sweep methods use {format_pool(pool_size)} selected stimuli and subject-averaged "
            "tracks."
        )
    fig.text(0.045, 0.985, title, ha="left", va="top", fontsize=15, fontweight="semibold")
    fig.text(0.045, 0.955, subtitle, ha="left", va="top", fontsize=9.5, color=TOKENS["muted"])
    fig.legend(
        handles=legend_handles(plot_data),
        loc="upper right",
        bbox_to_anchor=(0.99, 0.945),
        frameon=False,
        ncol=3,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.075, top=0.84, hspace=0.30, wspace=0.10)
    save_figure(fig, figures_root / f"method_recovery_{condition}_snr_curves_10M")
    return plot_data


def empirical_snr_rows(plot_tracks: pd.DataFrame) -> pd.DataFrame:
    return plot_tracks[
        (plot_tracks["plot_track"] == "subject_avg")
        & np.isclose(plot_tracks["relative_snr"].astype(float), 1.0)
        & plot_tracks["pool_size"].notna()
        & (plot_tracks["method_family"] != "refit_robust")
    ].copy()


def pool_plot_rows(plot_tracks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    emp = empirical_snr_rows(plot_tracks)
    if emp.empty:
        return emp, emp
    selected = emp[emp["subset_type"].astype(str) == "selected"].copy()
    random = emp[emp["subset_type"].astype(str) == "random"].copy()
    baseline = (
        random.groupby(
            ["selection_model_set", "method_family", "method_family_label", "pool_size"],
            as_index=False,
            sort=False,
        )
        .agg(recovery_accuracy=("recovery_accuracy", "mean"))
        .rename(columns={"recovery_accuracy": "shared_random_recovery_accuracy"})
    )
    recovery = pd.concat(
        [
            selected.assign(series_type="selected"),
            baseline.assign(
                selection_method_id="shared_random",
                method_variant="shared_random",
                method_variant_label="Shared random",
                subset_type="random",
                series_type="random",
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    lift = selected.merge(
        baseline,
        on=["selection_model_set", "method_family", "method_family_label", "pool_size"],
        how="left",
    )
    lift["lift"] = lift["recovery_accuracy"] - lift["shared_random_recovery_accuracy"]
    lift["lift_pp"] = 100.0 * lift["lift"]
    return recovery, lift


def draw_pool_panel(
    ax: plt.Axes,
    sub: pd.DataFrame,
    *,
    y_col: str,
    lift: bool = False,
) -> None:
    if sub.empty:
        ax.set_visible(False)
        return
    if lift:
        ax.axhline(0.0, color=TOKENS["random"], linestyle=":", linewidth=1.5, zorder=1)
        selected = sub
    else:
        baseline = sub[sub["series_type"].astype(str) == "random"].sort_values("pool_size")
        if not baseline.empty:
            ax.plot(
                baseline["pool_size"].to_numpy(),
                baseline[y_col].to_numpy(),
                color=TOKENS["random"],
                linestyle=":",
                linewidth=1.8,
                zorder=2,
            )
        selected = sub[sub["series_type"].astype(str) == "selected"]
    for variant in ordered_present(selected["method_variant"], VARIANT_ORDER):
        part = selected[selected["method_variant"].astype(str) == variant].sort_values("pool_size")
        if part.empty:
            continue
        ax.plot(
            part["pool_size"].to_numpy(),
            part[y_col].to_numpy(),
            color=VARIANT_COLORS.get(variant, "#555555"),
            marker=VARIANT_MARKERS.get(variant, "o"),
            linewidth=1.65,
            markersize=3.6,
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_xlim(min(POOL_SIZE_ORDER) * 0.82, max(POOL_SIZE_ORDER) * 1.22)
    ax.set_xticks(POOL_SIZE_ORDER)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_pool))
    if not lift:
        ax.set_ylim(0.0, 1.02)
    finish_panel(ax)
    ax.tick_params(axis="x", labelsize=7, rotation=28)
    for label in ax.get_xticklabels():
        label.set_ha("right")


def plot_pool_grid(
    plot_data: pd.DataFrame,
    condition: str,
    figures_root: Path,
    *,
    y_col: str,
    stem: str,
    title_metric: str,
    lift: bool = False,
) -> None:
    if plot_data.empty:
        return
    model_sets = ordered_present(plot_data["selection_model_set"], MODEL_SET_ORDER)
    families = ordered_present(plot_data["method_family"], METHOD_FAMILY_ORDER)
    fig, axes = plt.subplots(
        len(model_sets),
        len(families),
        figsize=(3.9 * len(families), 2.35 * len(model_sets) + 1.4),
        squeeze=False,
        sharex=True,
        sharey=not lift,
    )
    if lift:
        y_pad = 2.0
        y_min = min(-2.0, float(plot_data[y_col].min()) - y_pad)
        y_max = max(2.0, float(plot_data[y_col].max()) + y_pad)
    else:
        y_min, y_max = 0.0, 1.02
    for row_idx, model_set in enumerate(model_sets):
        for col_idx, family in enumerate(families):
            ax = axes[row_idx][col_idx]
            sub = plot_data[
                (plot_data["selection_model_set"] == model_set)
                & (plot_data["method_family"] == family)
            ]
            draw_pool_panel(ax, sub, y_col=y_col, lift=lift)
            ax.set_ylim(y_min, y_max)
            if row_idx == 0:
                ax.set_title(METHOD_FAMILY_LABELS.get(family, family), fontsize=10.5, pad=8)
            if col_idx == 0:
                ylabel = "Lift (pp)" if lift else "Recovery accuracy"
                ax.set_ylabel(f"{MODEL_SET_LABELS.get(model_set, model_set)}\n{ylabel}", fontsize=9.5)
            else:
                ax.set_ylabel("")
            if row_idx == len(model_sets) - 1:
                ax.set_xlabel("Candidate pool size", fontsize=9)
            else:
                ax.set_xlabel("")
    title = f"{title_metric} by pool size ({condition.replace('_', ' ')})"
    subtitle = "Empirical-SNR rows; subject tracks sub-01, sub-03, sub-05, sub-06, and sub-07 averaged."
    fig.text(0.045, 0.985, title, ha="left", va="top", fontsize=15, fontweight="semibold")
    fig.text(0.045, 0.955, subtitle, ha="left", va="top", fontsize=9.5, color=TOKENS["muted"])
    fig.legend(
        handles=legend_handles(plot_data, include_random=not lift),
        loc="upper right",
        bbox_to_anchor=(0.99, 0.945),
        frameon=False,
        ncol=3,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.085, top=0.84, hspace=0.30, wspace=0.10)
    save_figure(fig, figures_root / stem)


def legend_handles(data: pd.DataFrame, include_random: bool = True) -> list[Line2D]:
    handles: list[Line2D] = []
    for variant in ordered_present(data["method_variant"], VARIANT_ORDER):
        if variant == "shared_random":
            continue
        label = data.loc[data["method_variant"].astype(str) == variant, "method_variant_label"].iloc[0]
        handles.append(
            Line2D(
                [0],
                [0],
                color=VARIANT_COLORS.get(variant, "#555555"),
                marker=VARIANT_MARKERS.get(variant, "o"),
                linewidth=1.7,
                markersize=4,
                label=str(label),
            )
        )
    if include_random:
        handles.append(
            Line2D(
                [0],
                [0],
                color=TOKENS["random"],
                linestyle=":",
                linewidth=1.8,
                label="Shared random",
            )
        )
    return handles


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--figures-root", type=Path, default=DEFAULT_FIGURES_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--conditions",
        default="clean_to_noisy,noisy_to_noisy",
        help="Comma-separated rdm_calibration_comparison values.",
    )
    parser.add_argument("--final-pool-size", type=int, default=10_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_chart_theme()
    args.figures_root.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    for condition in conditions:
        data = discover_recovery(args.results_root, condition)
        if data.empty:
            print(f"No available teacher/student recovery rows for {condition}")
            continue
        write_csv(data, args.data_dir / f"method_recovery_available_rows_{condition}.csv")

        final_data = plot_final_snr(data, condition, args.final_pool_size, args.figures_root)
        if not final_data.empty:
            write_csv(final_data, args.data_dir / f"method_recovery_snr_curves_10M_{condition}.csv")

        tracks = build_plot_tracks(data)
        recovery_pool, lift_pool = pool_plot_rows(tracks)
        if not recovery_pool.empty:
            write_csv(
                recovery_pool,
                args.data_dir / f"method_recovery_pool_recovery_empirical_snr_{condition}.csv",
            )
            plot_pool_grid(
                recovery_pool,
                condition,
                args.figures_root,
                y_col="recovery_accuracy",
                stem=f"method_recovery_{condition}_pool_recovery_empirical_snr",
                title_metric="Teacher/student recovery",
            )
        if not lift_pool.empty:
            write_csv(
                lift_pool,
                args.data_dir / f"method_recovery_pool_lift_empirical_snr_{condition}.csv",
            )
            plot_pool_grid(
                lift_pool,
                condition,
                args.figures_root,
                y_col="lift_pp",
                stem=f"method_recovery_{condition}_pool_lift_empirical_snr",
                title_metric="Selected-minus-random lift",
                lift=True,
            )
        print(f"Wrote method recovery figures for {condition}")


if __name__ == "__main__":
    main()
