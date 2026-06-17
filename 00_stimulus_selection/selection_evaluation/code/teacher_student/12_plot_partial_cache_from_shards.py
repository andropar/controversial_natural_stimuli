#!/usr/bin/env python3
"""Plot teacher/student recovery curves directly from partial cache shards."""

from __future__ import annotations

import argparse
import math
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SCRIPT = Path(__file__).resolve()
EVAL_ROOT = SCRIPT.parents[2]
RESULTS = EVAL_ROOT / "final_stimuli_recovery" / "teacher_student" / "results"
FIGURES = EVAL_ROOT / "final_stimuli_recovery" / "teacher_student" / "figures"

MODEL_ORDER = ["all_models", "sota", "training_objective", "architecture", "dataset"]
EXPECTED_MODELS = {
    "all_models": 20,
    "sota": 6,
    "training_objective": 5,
    "architecture": 5,
    "dataset": 5,
}
RAW_TRACK = "raw"
ENCODING_TRACKS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
DISPLAY_TRACK_ORDER = ["raw", "encoding_avg"]
COLORS = {"selected": "#4C78A8", "random": "#E45756"}
CI_MULT = 1.96

NEEDED_COLUMNS = [
    "model_set",
    "track",
    "track_type",
    "metric",
    "corr_type",
    "subset_type",
    "subset_idx",
    "recovered_correct",
    "teacher_margin",
    "noise_mult",
    "noise_ceiling",
    "relative_snr",
    "noise_sample_idx",
    "refit_repeat_idx",
    "refit_pool_size",
    "refit_train_n",
    "refit_val_n",
    "eval_noise_mode",
    "fit_noise_calibration",
    "eval_refit_mode",
    "n_equivalence_classes",
]


@dataclass
class GroupStats:
    correct_sum: float = 0.0
    margin_sum: float = 0.0
    n_units: int = 0
    noise_samples: set[int] = field(default_factory=set)
    meta: dict[str, Any] = field(default_factory=dict)


def ordered(values: pd.Series | list[str], preferred: list[str]) -> list[str]:
    present = list(dict.fromkeys(pd.Series(values).dropna().astype(str)))
    return [x for x in preferred if x in present] + [x for x in present if x not in preferred]


def compute_log_auc(noise_mult: pd.Series, values: pd.Series) -> float:
    x = np.asarray(noise_mult, dtype=float)
    y = np.asarray(values, dtype=float)
    ok = np.isfinite(x) & (x > 0) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 2:
        return float("nan")
    order = np.argsort(x)
    log_x = np.log10(x[order])
    y = y[order]
    span = log_x[-1] - log_x[0]
    if span <= 0:
        return float("nan")
    return float(np.trapezoid(y, log_x) / span)


def sample_sd(values: pd.Series) -> float:
    vals = pd.Series(values).dropna().astype(float).to_numpy()
    if len(vals) < 2:
        return float("nan")
    return float(np.std(vals, ddof=1))


def sample_sem(values: pd.Series) -> float:
    vals = pd.Series(values).dropna().astype(float).to_numpy()
    if len(vals) < 2:
        return float("nan")
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def binomial_sem(p: pd.Series, n: pd.Series) -> pd.Series:
    p = p.astype(float)
    n = n.astype(float).clip(lower=1.0)
    return np.sqrt(np.clip(p * (1.0 - p), 0.0, None) / n)


def model_set_from_result_dir(name: str, suffix: str) -> str:
    suffix = suffix.lstrip("_")
    marker = f"_{suffix}"
    if name.endswith(marker):
        return name[: -len(marker)]
    for model_set in sorted(MODEL_ORDER, key=len, reverse=True):
        if name.startswith(f"{model_set}_"):
            return model_set
    return name.split("_teacher_student_", 1)[0]


def refit_size_from_suffix(suffix: str) -> int | None:
    match = re.search(r"_refit(\d+)_", suffix)
    if not match:
        return None
    return int(match.group(1))


def variant_from_suffix(suffix: str) -> str:
    if "eval_augmented_loo" in suffix:
        return "eval_augmented_loo"
    if "independent_refit" in suffix:
        return "independent"
    return "unknown"


def cache_files(results_root: Path, suffix: str) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for result_dir in sorted(results_root.glob(f"*{suffix}")):
        cache_root = result_dir / "_teacher_cache"
        if not cache_root.exists():
            continue
        model_set = model_set_from_result_dir(result_dir.name, suffix)
        for path in sorted(cache_root.glob("**/*.csv")):
            files.append((model_set, path))
    return files


def read_columns(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def aggregate_cache_suffix(
    results_root: Path,
    suffix: str,
    *,
    chunksize: int,
) -> pd.DataFrame:
    stats: dict[tuple[Any, ...], GroupStats] = defaultdict(GroupStats)
    files = cache_files(results_root, suffix)
    if not files:
        return pd.DataFrame()

    key_cols = [
        "model_set",
        "track",
        "track_type",
        "noise_mult",
        "noise_ceiling",
        "subset_type",
        "subset_idx",
        "refit_repeat_idx",
    ]
    meta_cols = [
        "metric",
        "corr_type",
        "relative_snr",
        "refit_pool_size",
        "refit_train_n",
        "refit_val_n",
        "eval_noise_mode",
        "fit_noise_calibration",
        "eval_refit_mode",
        "n_equivalence_classes",
    ]

    for file_idx, (model_set, path) in enumerate(files, start=1):
        if file_idx == 1 or file_idx % 50 == 0 or file_idx == len(files):
            print(f"  {suffix}: reading shard {file_idx}/{len(files)}", flush=True)
        header = read_columns(path)
        usecols = [col for col in NEEDED_COLUMNS if col in header]
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
            if chunk.empty:
                continue
            if "model_set" not in chunk:
                chunk["model_set"] = model_set
            else:
                chunk["model_set"] = chunk["model_set"].fillna(model_set)
            if "refit_repeat_idx" not in chunk:
                chunk["refit_repeat_idx"] = 0
            if "relative_snr" not in chunk:
                chunk["relative_snr"] = 1.0 / chunk["noise_mult"].astype(float)
            if "eval_refit_mode" not in chunk:
                chunk["eval_refit_mode"] = variant_from_suffix(suffix)
            if "track_type" not in chunk:
                chunk["track_type"] = chunk["track"]
            if "n_equivalence_classes" not in chunk:
                chunk["n_equivalence_classes"] = chunk["model_set"].map(EXPECTED_MODELS)
            chunk["recovered_correct"] = chunk["recovered_correct"].astype(float)
            chunk["teacher_margin"] = chunk["teacher_margin"].astype(float)

            for key, group in chunk.groupby(key_cols, sort=False, dropna=False):
                entry = stats[key]
                entry.correct_sum += float(group["recovered_correct"].sum())
                entry.margin_sum += float(group["teacher_margin"].sum())
                entry.n_units += int(len(group))
                if "noise_sample_idx" in group:
                    entry.noise_samples.update(
                        int(x) for x in group["noise_sample_idx"].dropna().unique()
                    )
                if not entry.meta:
                    first = group.iloc[0]
                    entry.meta = {
                        col: first[col]
                        for col in meta_cols
                        if col in group.columns and pd.notna(first[col])
                    }

    rows: list[dict[str, Any]] = []
    for key, entry in stats.items():
        (
            model_set,
            track,
            track_type,
            noise_mult,
            noise_ceiling,
            subset_type,
            subset_idx,
            refit_repeat_idx,
        ) = key
        n_units = max(entry.n_units, 1)
        rows.append(
            {
                "model_set": str(model_set),
                "recovery_orientation": "teacher_student_independent_refit_rdm_recovery",
                "track": str(track),
                "track_type": str(track_type),
                "metric": entry.meta.get("metric", "cosine"),
                "corr_type": entry.meta.get("corr_type", "spearman"),
                "noise_mult": float(noise_mult),
                "relative_snr": float(entry.meta.get("relative_snr", 1.0 / float(noise_mult))),
                "noise_ceiling": float(noise_ceiling),
                "subset_type": str(subset_type),
                "subset_idx": int(subset_idx),
                "refit_repeat_idx": int(refit_repeat_idx),
                "recovery_accuracy": float(entry.correct_sum / n_units),
                "error_prob": float(1.0 - entry.correct_sum / n_units),
                "mean_margin": float(entry.margin_sum / n_units),
                "n_units": int(entry.n_units),
                "n_models": int(EXPECTED_MODELS.get(str(model_set), 0) or 0),
                "n_equivalence_classes": int(
                    entry.meta.get(
                        "n_equivalence_classes",
                        EXPECTED_MODELS.get(str(model_set), 0) or 0,
                    )
                ),
                "n_noise_samples": int(len(entry.noise_samples))
                if entry.noise_samples
                else 1,
                "refit_pool_size": int(entry.meta.get("refit_pool_size", -1)),
                "refit_train_n": int(entry.meta.get("refit_train_n", -1)),
                "refit_val_n": int(entry.meta.get("refit_val_n", -1)),
                "eval_noise_mode": entry.meta.get("eval_noise_mode", "response"),
                "fit_noise_calibration": entry.meta.get("fit_noise_calibration", "rdm_empirical"),
                "eval_refit_mode": entry.meta.get("eval_refit_mode", variant_from_suffix(suffix)),
            }
        )
    subset = pd.DataFrame(rows)
    if subset.empty:
        return subset

    agg_keys = [
        "model_set",
        "recovery_orientation",
        "track",
        "track_type",
        "metric",
        "corr_type",
        "noise_mult",
        "relative_snr",
        "noise_ceiling",
        "subset_type",
    ]
    out_rows: list[dict[str, Any]] = []
    for key, group in subset.groupby(agg_keys, sort=False, dropna=False):
        weights = group["n_units"].to_numpy(float)
        acc = float(np.average(group["recovery_accuracy"], weights=weights))
        margin = float(np.average(group["mean_margin"], weights=weights))
        acc_vals = group["recovery_accuracy"].astype(float).to_numpy()
        margin_vals = group["mean_margin"].astype(float).to_numpy()
        if len(acc_vals) > 1:
            acc_sd = float(np.std(acc_vals, ddof=1))
            acc_sem = float(acc_sd / math.sqrt(len(acc_vals)))
        else:
            acc_sd = float("nan")
            acc_sem = float(math.sqrt(max(acc * (1.0 - acc), 0.0) / max(weights.sum(), 1.0)))
        if len(margin_vals) > 1:
            margin_sd = float(np.std(margin_vals, ddof=1))
            margin_sem = float(margin_sd / math.sqrt(len(margin_vals)))
        else:
            margin_sd = float("nan")
            margin_sem = float("nan")
        out_rows.append(
            dict(zip(agg_keys, key))
            | {
                "recovery_accuracy": acc,
                "recovery_accuracy_sd": acc_sd,
                "recovery_accuracy_sem": acc_sem,
                "error_prob": float(1.0 - acc),
                "mean_margin": margin,
                "mean_margin_sd": margin_sd,
                "mean_margin_sem": margin_sem,
                "n_units": int(group["n_units"].sum()),
                "n_subsets": int(group["subset_idx"].nunique()),
                "n_refit_repeats": int(group["refit_repeat_idx"].nunique()),
                "n_models": int(group["n_models"].max()),
                "n_equivalence_classes": int(group["n_equivalence_classes"].max()),
                "n_noise_samples": int(group["n_noise_samples"].max()),
                "refit_pool_size": int(group["refit_pool_size"].max()),
                "refit_train_n": int(group["refit_train_n"].max()),
                "refit_val_n": int(group["refit_val_n"].max()),
                "eval_noise_mode": group["eval_noise_mode"].iloc[0],
                "fit_noise_calibration": group["fit_noise_calibration"].iloc[0],
                "eval_refit_mode": group["eval_refit_mode"].iloc[0],
            }
        )
    return pd.DataFrame(out_rows)


def make_raw_and_encoding_avg(data: pd.DataFrame) -> pd.DataFrame:
    raw = data[data["track"] == RAW_TRACK].copy()
    if not raw.empty:
        raw["report_track"] = "raw"
        raw["n_subject_tracks"] = 0
        raw["subject_tracks"] = ""
        raw["recovery_accuracy_sem"] = raw["recovery_accuracy_sem"].combine_first(
            binomial_sem(raw["recovery_accuracy"], raw["n_units"])
        )

    enc = data[data["track"].isin(ENCODING_TRACKS)].copy()
    enc_rows: list[dict[str, Any]] = []
    keys = [
        "model_set",
        "recovery_orientation",
        "metric",
        "corr_type",
        "noise_mult",
        "relative_snr",
        "noise_ceiling",
        "subset_type",
        "eval_noise_mode",
        "fit_noise_calibration",
        "eval_refit_mode",
    ]
    if not enc.empty:
        for key, group in enc.groupby(keys, sort=False, dropna=False):
            by_track = (
                group.groupby("track", as_index=False)
                .agg(
                    recovery_accuracy=("recovery_accuracy", "mean"),
                    error_prob=("error_prob", "mean"),
                    mean_margin=("mean_margin", "mean"),
                    n_units=("n_units", "sum"),
                    n_subsets=("n_subsets", "max"),
                    n_refit_repeats=("n_refit_repeats", "max"),
                    n_models=("n_models", "max"),
                    n_equivalence_classes=("n_equivalence_classes", "max"),
                    n_noise_samples=("n_noise_samples", "max"),
                    refit_pool_size=("refit_pool_size", "max"),
                    refit_train_n=("refit_train_n", "max"),
                    refit_val_n=("refit_val_n", "max"),
                )
            )
            present_tracks = sorted(set(by_track["track"].astype(str)))
            row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
            row |= {
                "track": "encoding_avg",
                "report_track": "encoding_avg",
                "track_type": "encoding_average_partial",
                "recovery_accuracy": float(by_track["recovery_accuracy"].mean()),
                "recovery_accuracy_sd": sample_sd(by_track["recovery_accuracy"]),
                "recovery_accuracy_sem": sample_sem(by_track["recovery_accuracy"]),
                "error_prob": float(by_track["error_prob"].mean()),
                "mean_margin": float(by_track["mean_margin"].mean()),
                "n_units": int(by_track["n_units"].sum()),
                "n_subsets": int(by_track["n_subsets"].max()),
                "n_refit_repeats": int(by_track["n_refit_repeats"].max()),
                "n_models": int(by_track["n_models"].max()),
                "n_equivalence_classes": int(by_track["n_equivalence_classes"].max()),
                "n_noise_samples": int(by_track["n_noise_samples"].max()),
                "refit_pool_size": int(by_track["refit_pool_size"].max()),
                "refit_train_n": int(by_track["refit_train_n"].max()),
                "refit_val_n": int(by_track["refit_val_n"].max()),
                "n_subject_tracks": len(present_tracks),
                "subject_tracks": ",".join(present_tracks),
            }
            enc_rows.append(row)
    enc_avg = pd.DataFrame(enc_rows)
    out = pd.concat([raw, enc_avg], ignore_index=True, sort=False)
    if out.empty:
        return out
    out["report_track"] = pd.Categorical(
        out["report_track"],
        categories=DISPLAY_TRACK_ORDER,
        ordered=True,
    )
    out["model_set"] = pd.Categorical(out["model_set"], categories=MODEL_ORDER, ordered=True)
    return out.sort_values(["model_set", "report_track", "subset_type", "noise_mult"])


def make_auc(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["model_set", "report_track", "track_type", "subset_type"]
    for key, group in summary.groupby(keys, sort=False, observed=True):
        rows.append(
            dict(zip(keys, key))
            | {
                "error_auc": compute_log_auc(group["noise_mult"], group["error_prob"]),
                "recovery_accuracy_auc": compute_log_auc(
                    group["noise_mult"], group["recovery_accuracy"]
                ),
                "mean_margin_auc": compute_log_auc(group["noise_mult"], group["mean_margin"]),
                "n_noise_levels": int(group["noise_mult"].nunique()),
                "n_models": int(group["n_models"].max()),
                "n_equivalence_classes": int(group["n_equivalence_classes"].max()),
                "n_subject_tracks": int(group["n_subject_tracks"].max()),
                "n_refit_repeats": int(group["n_refit_repeats"].max()),
                "refit_pool_size": int(group["refit_pool_size"].max()),
                "eval_refit_mode": group["eval_refit_mode"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def empirical_snr_table(summary: pd.DataFrame) -> pd.DataFrame:
    emp = summary[np.isclose(summary["noise_mult"].astype(float), 1.0)].copy()
    if emp.empty:
        return emp
    selected = emp[emp["subset_type"] == "selected"][
        [
            "model_set",
            "report_track",
            "recovery_accuracy",
            "mean_margin",
            "n_units",
            "n_refit_repeats",
            "refit_pool_size",
        ]
    ].rename(
        columns={
            "recovery_accuracy": "selected_recovery_accuracy",
            "mean_margin": "selected_mean_margin",
            "n_units": "selected_n_units",
        }
    )
    random = emp[emp["subset_type"] == "random"][
        ["model_set", "report_track", "recovery_accuracy", "mean_margin", "n_units"]
    ].rename(
        columns={
            "recovery_accuracy": "random_recovery_accuracy",
            "mean_margin": "random_mean_margin",
            "n_units": "random_n_units",
        }
    )
    out = selected.merge(random, on=["model_set", "report_track"], how="outer")
    out["selected_minus_random"] = (
        out["selected_recovery_accuracy"] - out["random_recovery_accuracy"]
    )
    out["report_track"] = pd.Categorical(
        out["report_track"],
        categories=DISPLAY_TRACK_ORDER,
        ordered=True,
    )
    out["model_set"] = pd.Categorical(out["model_set"], categories=MODEL_ORDER, ordered=True)
    return out.sort_values(["model_set", "report_track"])


def plot_curves(summary: pd.DataFrame, out_base: Path, title: str) -> None:
    model_sets = ordered(summary["model_set"].astype(str), MODEL_ORDER)
    tracks = ordered(summary["report_track"].astype(str), DISPLAY_TRACK_ORDER)
    fig, axes = plt.subplots(
        len(tracks),
        len(model_sets),
        figsize=(3.4 * len(model_sets), 2.65 * len(tracks)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row, track in enumerate(tracks):
        for col, model_set in enumerate(model_sets):
            ax = axes[row, col]
            sub = summary[
                (summary["model_set"].astype(str) == model_set)
                & (summary["report_track"].astype(str) == track)
            ]
            if sub.empty:
                ax.set_axis_off()
                continue
            for subset_type, linestyle in [("random", "--"), ("selected", "-")]:
                g = sub[sub["subset_type"] == subset_type].sort_values("relative_snr")
                if g.empty:
                    continue
                x = g["relative_snr"].astype(float).to_numpy()
                y = g["recovery_accuracy"].astype(float).to_numpy()
                if subset_type == "random":
                    band = g["recovery_accuracy_sd"].astype(float).fillna(0.0).to_numpy()
                    alpha = 0.18
                else:
                    band = (
                        CI_MULT
                        * g["recovery_accuracy_sem"].astype(float).fillna(0.0).to_numpy()
                    )
                    alpha = 0.16
                ax.fill_between(
                    x,
                    np.clip(y - band, 0.0, 1.0),
                    np.clip(y + band, 0.0, 1.0),
                    color=COLORS[subset_type],
                    alpha=alpha,
                    linewidth=0,
                    zorder=1,
                )
                ax.plot(
                    x,
                    y,
                    linestyle,
                    color=COLORS[subset_type],
                    linewidth=1.9,
                    label=subset_type.capitalize(),
                    zorder=3,
                )
            chance_n = float(sub["n_equivalence_classes"].max())
            if np.isfinite(chance_n) and chance_n > 0:
                ax.axhline(1.0 / chance_n, color="#666666", linestyle=":", linewidth=0.8)
            ax.axvline(1.0, color="#222222", linestyle="-.", linewidth=0.8, alpha=0.65)
            ax.set_xscale("log")
            ax.set_ylim(0.0, 1.04)
            ax.grid(color="#dddddd", linewidth=0.65, alpha=0.75)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(model_set.replace("_", " "), fontsize=10)
            if col == 0:
                label = "Raw" if track == "raw" else "Encoding avg"
                ax.set_ylabel(f"{label}\nRecovery accuracy", fontsize=9)
            if row == len(tracks) - 1:
                ax.set_xlabel("Relative SNR")
            repeats = int(sub["n_refit_repeats"].max())
            ax.text(
                0.02,
                0.04,
                f"rr={repeats}",
                transform=ax.transAxes,
                fontsize=7,
                color="#444444",
                va="bottom",
                ha="left",
            )
    handles = [
        Line2D([0], [0], color=COLORS["random"], linestyle="--", lw=1.9, label="Random"),
        Line2D([0], [0], color=COLORS["selected"], linestyle="-", lw=1.9, label="Selected"),
        Patch(facecolor=COLORS["random"], alpha=0.18, label="Random SD"),
        Patch(facecolor=COLORS["selected"], alpha=0.16, label="Selected 95% CI"),
        Line2D([0], [0], color="#222222", linestyle="-.", lw=0.8, label="Empirical SNR"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(title, y=1.08)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_refit_sweep(combined: pd.DataFrame, out_base: Path) -> None:
    data = combined[
        (combined["eval_refit_mode"] == "independent")
        & np.isclose(combined["noise_mult"].astype(float), 1.0)
    ].copy()
    if data.empty:
        return
    data["refit_pool_size"] = data["refit_pool_size"].astype(int)
    model_sets = ordered(data["model_set"].astype(str), MODEL_ORDER)
    tracks = ordered(data["report_track"].astype(str), DISPLAY_TRACK_ORDER)
    fig, axes = plt.subplots(
        len(tracks),
        len(model_sets),
        figsize=(3.4 * len(model_sets), 2.55 * len(tracks)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row, track in enumerate(tracks):
        for col, model_set in enumerate(model_sets):
            ax = axes[row, col]
            sub = data[
                (data["model_set"].astype(str) == model_set)
                & (data["report_track"].astype(str) == track)
            ]
            if sub.empty:
                ax.set_axis_off()
                continue
            for subset_type, linestyle in [("random", "--"), ("selected", "-")]:
                g = sub[sub["subset_type"] == subset_type].sort_values("refit_pool_size")
                if g.empty:
                    continue
                ax.plot(
                    g["refit_pool_size"],
                    g["recovery_accuracy"],
                    linestyle,
                    marker="o",
                    markersize=3.5,
                    color=COLORS[subset_type],
                    linewidth=1.8,
                    label=subset_type.capitalize(),
                )
            chance_n = float(sub["n_equivalence_classes"].max())
            if np.isfinite(chance_n) and chance_n > 0:
                ax.axhline(1.0 / chance_n, color="#666666", linestyle=":", linewidth=0.8)
            ax.set_xscale("log")
            ax.set_ylim(0.0, 1.04)
            ax.grid(color="#dddddd", linewidth=0.65, alpha=0.75)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(model_set.replace("_", " "), fontsize=10)
            if col == 0:
                label = "Raw" if track == "raw" else "Encoding avg"
                ax.set_ylabel(f"{label}\nRecovery at empirical SNR", fontsize=9)
            if row == len(tracks) - 1:
                ax.set_xlabel("Refit pool size")
    handles = [
        Line2D([0], [0], color=COLORS["random"], linestyle="--", marker="o", lw=1.8, label="Random"),
        Line2D([0], [0], color=COLORS["selected"], linestyle="-", marker="o", lw=1.8, label="Selected"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Independent refit sweep: empirical SNR partial-cache results", y=1.06)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    summary: pd.DataFrame,
    *,
    figures_root: Path,
    name: str,
    title: str,
    png_dir: Path,
) -> pd.DataFrame:
    curves_csv = figures_root / f"{name}_curves_summary.csv"
    auc_csv = figures_root / f"{name}_auc_summary.csv"
    emp_csv = figures_root / f"{name}_empirical_snr.csv"
    summary.to_csv(curves_csv, index=False)
    make_auc(summary).to_csv(auc_csv, index=False)
    empirical_snr_table(summary).to_csv(emp_csv, index=False)
    plot_curves(summary, figures_root / name, title)
    png = figures_root / f"{name}.png"
    if png.exists():
        png_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(png, png_dir / png.name)
    print(f"Wrote {png}", flush=True)
    print(f"Wrote {curves_csv}", flush=True)
    return summary


def default_suffixes() -> list[str]:
    base = "_teacher_student_independent_refit_refit{size}_rdm_score_spearman_response_empcal_ns20_rand100_rr3_fastgpu"
    return [base.format(size=size) for size in [100, 500, 1000, 5000, 10000]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    parser.add_argument("--figures-root", type=Path, default=FIGURES)
    parser.add_argument("--data-suffix", action="append", default=None)
    parser.add_argument("--name-prefix", default="teacher_student_recovery_partial_cache")
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    args.figures_root.mkdir(parents=True, exist_ok=True)
    png_dir = args.figures_root / "png"
    summaries: list[pd.DataFrame] = []
    suffixes = args.data_suffix or default_suffixes()
    for suffix in suffixes:
        print(f"Aggregating {suffix}", flush=True)
        disc = aggregate_cache_suffix(args.results_root, suffix, chunksize=args.chunksize)
        if disc.empty:
            print(f"  no cache rows found for {suffix}", flush=True)
            continue
        summary = make_raw_and_encoding_avg(disc)
        if summary.empty:
            print(f"  no raw/encoding rows found for {suffix}", flush=True)
            continue
        refit_size = refit_size_from_suffix(suffix)
        variant = variant_from_suffix(suffix)
        summary["data_suffix"] = suffix
        summary["variant"] = variant
        if refit_size is not None:
            summary["refit_pool_size_from_suffix"] = refit_size
        size_label = f"refit{refit_size}" if refit_size is not None else "unknown_refit"
        name = f"{args.name_prefix}_{variant}_{size_label}_ns20_rand100_rr3_fastgpu"
        title = f"Teacher/student recovery from partial cache ({variant}, {size_label})"
        summaries.append(
            write_outputs(
                summary,
                figures_root=args.figures_root,
                name=name,
                title=title,
                png_dir=png_dir,
            )
        )

    if summaries:
        combined = pd.concat(summaries, ignore_index=True, sort=False)
        combined_csv = args.figures_root / f"{args.name_prefix}_all_available_curves_summary.csv"
        combined.to_csv(combined_csv, index=False)
        sweep_base = args.figures_root / f"{args.name_prefix}_independent_refit_sweep_empirical_snr"
        plot_refit_sweep(combined, sweep_base)
        sweep_png = sweep_base.with_suffix(".png")
        if sweep_png.exists():
            png_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sweep_png, png_dir / sweep_png.name)
            print(f"Wrote {sweep_png}", flush=True)
        print(f"Wrote {combined_csv}", flush=True)


if __name__ == "__main__":
    main()
