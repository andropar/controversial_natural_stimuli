#!/usr/bin/env python3
"""Plot all currently available final-stimuli teacher/student refit-size results."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


SCRIPT = Path(__file__).resolve()
EVAL_ROOT = SCRIPT.parents[2]
ROOT = next(p for p in SCRIPT.parents if (p / "src" / "cstims").exists())
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RESULTS = EVAL_ROOT / "final_stimuli_recovery" / "teacher_student" / "results"
FIGURES = EVAL_ROOT / "final_stimuli_recovery" / "teacher_student" / "figures"
DEFAULT_PATTERN = "rdm_score_spearman_response_empcal_ns20_rand100_rr3_fastgpu"
MODEL_ORDER = ["all_models", "sota", "training_objective", "architecture", "dataset"]
RAW_TRACK = "raw"
ENCODING_TRACKS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
DISPLAY_TRACK_ORDER = ["raw", "encoding_avg"]
SUBSET_STYLE = {
    "selected": {"color": "#4C78A8", "linestyle": "-", "label": "Selected"},
    "random": {"color": "#E45756", "linestyle": "--", "label": "Random"},
}
CI_MULT = 1.96
MODE_DISPLAY_NAMES = {
    "independent": "Separate-train refit",
    "eval_augmented_loo": "Eval-included LOO refit",
}


def sample_sd(values: pd.Series | np.ndarray) -> float:
    vals = pd.Series(values).dropna().astype(float).to_numpy()
    if len(vals) < 2:
        return float("nan")
    return float(np.std(vals, ddof=1))


def sample_sem(values: pd.Series | np.ndarray) -> float:
    vals = pd.Series(values).dropna().astype(float).to_numpy()
    if len(vals) < 2:
        return float("nan")
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def binomial_sem(p: pd.Series, n: pd.Series) -> pd.Series:
    p = p.astype(float)
    n = n.astype(float).clip(lower=1.0)
    return np.sqrt(np.clip(p * (1.0 - p), 0.0, None) / n)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def mode_display_name(mode: str) -> str:
    return MODE_DISPLAY_NAMES.get(mode, mode.replace("_", " "))


def load_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def infer_model_set(run_dir: Path, metadata: dict[str, Any]) -> str:
    if metadata.get("model_set"):
        return str(metadata["model_set"])
    name = run_dir.name
    for model_set in MODEL_ORDER:
        if name.startswith(model_set + "_"):
            return model_set
    return name.split("_teacher_student", 1)[0]


def infer_eval_refit_mode(run_dir: Path, metadata: dict[str, Any]) -> str:
    if metadata.get("eval_refit_mode"):
        return str(metadata["eval_refit_mode"])
    name = run_dir.name
    if "eval_augmented_loo" in name:
        return "eval_augmented_loo"
    return "independent"


def infer_refit_pool_size(run_dir: Path, metadata: dict[str, Any]) -> int:
    if metadata.get("refit_pool_size") is not None:
        return int(metadata["refit_pool_size"])
    name = run_dir.name
    match = re.search(r"refit(?:_|)(\d+)", name)
    if match:
        return int(match.group(1))
    match = re.search(r"_([0-9]+)k_", name)
    if match:
        return int(match.group(1)) * 1000
    return 0


def discover_runs(results_root: Path, pattern: str) -> list[Path]:
    out = []
    for run_dir in sorted(results_root.iterdir()):
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        if "teacher_student" not in name:
            continue
        if "teacher_alignment" in name:
            continue
        if pattern and pattern not in name:
            continue
        if (run_dir / "metadata.json").exists() or (run_dir / "discriminability.csv").exists():
            out.append(run_dir)
    return out


def read_completed_summary(run_dir: Path, metadata: dict[str, Any]) -> pd.DataFrame:
    path = run_dir / "discriminability.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        data = pd.read_csv(path)
    except Exception as exc:
        print(f"Skipping unreadable summary {path}: {exc}", flush=True)
        return pd.DataFrame()
    if data.empty:
        return data
    model_set = infer_model_set(run_dir, metadata)
    refit_pool_size = infer_refit_pool_size(run_dir, metadata)
    eval_refit_mode = infer_eval_refit_mode(run_dir, metadata)
    n_models = len(metadata.get("model_names", [])) or safe_int(data.get("n_models", pd.Series([0])).max())
    data["source_dir"] = run_dir.name
    data["source_kind"] = "summary"
    data["is_partial"] = False
    data["model_set"] = data.get("model_set", model_set)
    data["refit_pool_size"] = data.get("refit_pool_size", refit_pool_size)
    data["eval_refit_mode"] = data.get("eval_refit_mode", eval_refit_mode).fillna(eval_refit_mode)
    data["n_models"] = data.get("n_models", n_models)
    data["n_cache_shards_read"] = np.nan
    data["n_expected_cache_shards"] = np.nan
    data["cache_shard_fraction"] = 1.0
    data["n_teachers_observed"] = n_models
    data["n_teachers_expected"] = n_models
    if "relative_snr" not in data:
        data["relative_snr"] = 1.0 / data["noise_mult"].astype(float)
    if "recovery_accuracy_sem" not in data:
        data["recovery_accuracy_sem"] = binomial_sem(data["recovery_accuracy"], data["n_units"])
    return data


def expected_shards_for_run(metadata: dict[str, Any], *, report_track: str | None = None) -> int:
    model_names = metadata.get("model_names") or []
    tracks = [track.get("name") for track in metadata.get("tracks", []) if track.get("name")]
    n_models = len(model_names)
    n_repeats = int(metadata.get("n_refit_repeats") or 1)
    if report_track == "raw":
        n_tracks = 1 if RAW_TRACK in tracks else 0
    elif report_track == "encoding_avg":
        n_tracks = len([track for track in tracks if track in ENCODING_TRACKS])
    else:
        n_tracks = len(tracks)
    return n_models * n_repeats * n_tracks


def summarize_cache(run_dir: Path, metadata: dict[str, Any]) -> pd.DataFrame:
    cache_root = run_dir / "_teacher_cache"
    if not cache_root.exists():
        return pd.DataFrame()

    model_set = infer_model_set(run_dir, metadata)
    n_models = len(metadata.get("model_names", []))
    unit_stats: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(
        lambda: {"n_units": 0, "correct_sum": 0.0, "margin_sum": 0.0, "noise_samples": set()}
    )
    observed_shards: set[tuple[str, int, str]] = set()
    source_files = 0
    usecols = [
        "model_set",
        "track",
        "track_type",
        "metric",
        "corr_type",
        "subset_type",
        "subset_idx",
        "teacher_model",
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

    for path in sorted(cache_root.rglob("*.csv")):
        if path.name.endswith(".tmp"):
            continue
        try:
            header = pd.read_csv(path, nrows=0)
            cols = [col for col in usecols if col in header.columns]
            df = pd.read_csv(path, usecols=cols)
        except Exception as exc:
            print(f"Skipping unreadable cache shard {path}: {exc}", flush=True)
            continue
        if df.empty or "recovered_correct" not in df or "teacher_margin" not in df:
            continue
        source_files += 1
        if "model_set" not in df:
            df["model_set"] = model_set
        if "relative_snr" not in df:
            df["relative_snr"] = 1.0 / df["noise_mult"].astype(float)
        if "refit_repeat_idx" not in df:
            df["refit_repeat_idx"] = 0
        if "eval_refit_mode" not in df:
            df["eval_refit_mode"] = infer_eval_refit_mode(run_dir, metadata)
        if "n_equivalence_classes" not in df:
            df["n_equivalence_classes"] = n_models
        df["recovered_correct"] = df["recovered_correct"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
        for (track, repeat, teacher), _ in df.groupby(
            ["track", "refit_repeat_idx", "teacher_model"], sort=False, dropna=False
        ):
            observed_shards.add((str(track), int(repeat), str(teacher)))

        group_cols = [
            "model_set",
            "track",
            "track_type",
            "metric",
            "corr_type",
            "noise_mult",
            "relative_snr",
            "noise_ceiling",
            "subset_type",
            "subset_idx",
            "refit_repeat_idx",
            "eval_noise_mode",
            "fit_noise_calibration",
            "eval_refit_mode",
            "refit_pool_size",
            "refit_train_n",
            "refit_val_n",
            "n_equivalence_classes",
        ]
        grouped = (
            df.groupby(group_cols, sort=False, dropna=False)
            .agg(
                n_units=("recovered_correct", "size"),
                correct_sum=("recovered_correct", "sum"),
                margin_sum=("teacher_margin", "sum"),
                n_noise_samples=("noise_sample_idx", "nunique"),
            )
            .reset_index()
        )
        for row in grouped.itertuples(index=False):
            row_dict = row._asdict()
            key = tuple(row_dict[col] for col in group_cols)
            stat = unit_stats[key]
            stat["n_units"] += int(row_dict["n_units"])
            stat["correct_sum"] += float(row_dict["correct_sum"])
            stat["margin_sum"] += float(row_dict["margin_sum"])
            stat["noise_samples"].add(int(row_dict["n_noise_samples"]))

    if not unit_stats:
        return pd.DataFrame()

    unit_rows = []
    group_cols = [
        "model_set",
        "track",
        "track_type",
        "metric",
        "corr_type",
        "noise_mult",
        "relative_snr",
        "noise_ceiling",
        "subset_type",
        "subset_idx",
        "refit_repeat_idx",
        "eval_noise_mode",
        "fit_noise_calibration",
        "eval_refit_mode",
        "refit_pool_size",
        "refit_train_n",
        "refit_val_n",
        "n_equivalence_classes",
    ]
    for key, stat in unit_stats.items():
        n_units = int(stat["n_units"])
        unit_rows.append(
            dict(zip(group_cols, key))
            | {
                "recovery_accuracy": float(stat["correct_sum"] / max(n_units, 1)),
                "error_prob": float(1.0 - stat["correct_sum"] / max(n_units, 1)),
                "mean_margin": float(stat["margin_sum"] / max(n_units, 1)),
                "n_units": n_units,
                "n_noise_samples": int(max(stat["noise_samples"] or {0})),
            }
        )
    unit = pd.DataFrame(unit_rows)

    agg_keys = [
        "model_set",
        "track",
        "track_type",
        "metric",
        "corr_type",
        "noise_mult",
        "relative_snr",
        "noise_ceiling",
        "subset_type",
        "eval_noise_mode",
        "fit_noise_calibration",
        "eval_refit_mode",
        "refit_pool_size",
        "refit_train_n",
        "refit_val_n",
        "n_equivalence_classes",
    ]
    rows = []
    expected_all = expected_shards_for_run(metadata)
    for key, group in unit.groupby(agg_keys, sort=False, dropna=False):
        weights = group["n_units"].astype(float).to_numpy()
        acc_values = group["recovery_accuracy"].astype(float).to_numpy()
        margin_values = group["mean_margin"].astype(float).to_numpy()
        acc = float(np.average(acc_values, weights=weights))
        margin = float(np.average(margin_values, weights=weights))
        rows.append(
            dict(zip(agg_keys, key))
            | {
                "recovery_orientation": "teacher_student_independent_refit_rdm_recovery",
                "recovery_accuracy": acc,
                "recovery_accuracy_sd": sample_sd(acc_values),
                "recovery_accuracy_sem": sample_sem(acc_values)
                if len(acc_values) > 1
                else math.sqrt(max(acc * (1.0 - acc), 0.0) / max(float(weights.sum()), 1.0)),
                "error_prob": float(1.0 - acc),
                "mean_margin": margin,
                "mean_margin_sd": sample_sd(margin_values),
                "mean_margin_sem": sample_sem(margin_values),
                "n_units": int(group["n_units"].sum()),
                "n_subsets": int(group["subset_idx"].nunique()),
                "n_refit_repeats": int(group["refit_repeat_idx"].nunique()),
                "n_models": int(n_models),
                "n_noise_samples": int(group["n_noise_samples"].max()),
                "source_dir": run_dir.name,
                "source_kind": "cache",
                "n_cache_shards_read": int(source_files),
                "n_expected_cache_shards": int(expected_all),
                "cache_shard_fraction": (
                    float(source_files / expected_all) if expected_all else float("nan")
                ),
                "n_teachers_observed": int(
                    len({teacher for _track, _repeat, teacher in observed_shards})
                ),
                "n_teachers_expected": int(n_models),
                "is_partial": bool(expected_all and source_files < expected_all),
            }
        )
    return pd.DataFrame(rows)


def make_raw_and_encoding_avg(data: pd.DataFrame) -> pd.DataFrame:
    raw = data[data["track"].astype(str).eq(RAW_TRACK)].copy()
    if not raw.empty:
        raw["report_track"] = "raw"
        raw["n_subject_tracks"] = 0
        raw["subject_tracks"] = ""
        raw["report_cache_shards_read"] = raw["n_cache_shards_read"]
        raw["report_expected_cache_shards"] = raw["n_expected_cache_shards"]

    enc = data[data["track"].astype(str).isin(ENCODING_TRACKS)].copy()
    enc_rows: list[dict[str, Any]] = []
    base_keys = [
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
        "refit_pool_size",
        "refit_train_n",
        "refit_val_n",
        "source_dir",
        "source_kind",
    ]
    keys = [key for key in base_keys if key in enc.columns]
    if not enc.empty:
        for key, group in enc.groupby(keys, sort=False, dropna=False):
            present_tracks = sorted(set(group["track"].astype(str)))
            row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
            row |= {
                "track": "encoding_avg",
                "track_type": "encoding_average",
                "report_track": "encoding_avg",
                "recovery_accuracy": float(group["recovery_accuracy"].mean()),
                "recovery_accuracy_sd": sample_sd(group["recovery_accuracy"]),
                "recovery_accuracy_sem": sample_sem(group["recovery_accuracy"]),
                "error_prob": float(group["error_prob"].mean()),
                "mean_margin": float(group["mean_margin"].mean()),
                "mean_margin_sd": sample_sd(group["mean_margin"]),
                "mean_margin_sem": sample_sem(group["mean_margin"]),
                "n_units": int(group["n_units"].sum()),
                "n_subsets": int(group["n_subsets"].max()),
                "n_refit_repeats": int(group["n_refit_repeats"].max()),
                "n_models": int(group["n_models"].max()),
                "n_equivalence_classes": int(group["n_equivalence_classes"].max()),
                "n_noise_samples": int(group["n_noise_samples"].max()),
                "n_subject_tracks": len(present_tracks),
                "subject_tracks": ",".join(present_tracks),
                "n_cache_shards_read": group["n_cache_shards_read"].max(),
                "n_expected_cache_shards": group["n_expected_cache_shards"].max(),
                "report_cache_shards_read": group["n_cache_shards_read"].sum(min_count=1),
                "report_expected_cache_shards": group["n_expected_cache_shards"].sum(min_count=1),
                "cache_shard_fraction": (
                    group["n_cache_shards_read"].sum(min_count=1)
                    / group["n_expected_cache_shards"].sum(min_count=1)
                )
                if group["n_expected_cache_shards"].notna().any()
                else 1.0,
                "n_teachers_observed": int(group["n_teachers_observed"].max()),
                "n_teachers_expected": int(group["n_teachers_expected"].max()),
                "is_partial": bool(group["is_partial"].any()),
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
    return out.sort_values(
        ["eval_refit_mode", "model_set", "report_track", "refit_pool_size", "subset_type", "noise_mult"]
    )


def empirical_snr_table(summary: pd.DataFrame) -> pd.DataFrame:
    emp = summary[np.isclose(summary["noise_mult"].astype(float), 1.0)].copy()
    if emp.empty:
        return emp
    keys = ["eval_refit_mode", "model_set", "report_track", "refit_pool_size", "source_dir"]
    selected = emp[emp["subset_type"].eq("selected")].rename(
        columns={
            "recovery_accuracy": "selected_recovery_accuracy",
            "mean_margin": "selected_mean_margin",
            "is_partial": "selected_is_partial",
        }
    )
    random = emp[emp["subset_type"].eq("random")].rename(
        columns={
            "recovery_accuracy": "random_recovery_accuracy",
            "mean_margin": "random_mean_margin",
            "is_partial": "random_is_partial",
        }
    )
    keep_selected = keys + [
        "selected_recovery_accuracy",
        "selected_mean_margin",
        "selected_is_partial",
        "cache_shard_fraction",
        "n_teachers_observed",
        "n_teachers_expected",
        "n_refit_repeats",
        "n_subject_tracks",
        "subject_tracks",
    ]
    keep_random = keys + ["random_recovery_accuracy", "random_mean_margin", "random_is_partial"]
    out = selected[keep_selected].merge(random[keep_random], on=keys, how="outer")
    out["selected_minus_random"] = (
        out["selected_recovery_accuracy"] - out["random_recovery_accuracy"]
    )
    out["is_partial"] = out["selected_is_partial"].fillna(False) | out["random_is_partial"].fillna(False)
    return out.sort_values(["eval_refit_mode", "model_set", "report_track", "refit_pool_size"])


def ordered(values: pd.Series, preferred: list[str]) -> list[str]:
    present = list(dict.fromkeys(values.dropna().astype(str)))
    return [x for x in preferred if x in present] + [x for x in present if x not in preferred]


def plot_empirical_snr(summary: pd.DataFrame, out_base: Path) -> None:
    emp = summary[np.isclose(summary["noise_mult"].astype(float), 1.0)].copy()
    if emp.empty:
        return
    model_sets = ordered(emp["model_set"].astype(str), MODEL_ORDER)
    tracks = ordered(emp["report_track"].astype(str), DISPLAY_TRACK_ORDER)
    modes = ordered(emp["eval_refit_mode"].astype(str), ["independent", "eval_augmented_loo"])
    n_rows = len(modes) * len(tracks)
    fig, axes = plt.subplots(
        n_rows,
        len(model_sets),
        figsize=(3.15 * len(model_sets), 2.15 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    row_labels: list[tuple[str, str]] = [(mode, track) for mode in modes for track in tracks]
    all_refits = sorted(emp["refit_pool_size"].dropna().astype(int).unique())
    for r, (mode, track) in enumerate(row_labels):
        for c, model_set in enumerate(model_sets):
            ax = axes[r, c]
            sub = emp[
                emp["eval_refit_mode"].astype(str).eq(mode)
                & emp["report_track"].astype(str).eq(track)
                & emp["model_set"].astype(str).eq(model_set)
            ]
            if sub.empty:
                ax.set_axis_off()
                continue
            for subset_type, style in SUBSET_STYLE.items():
                g = sub[sub["subset_type"].eq(subset_type)].sort_values("refit_pool_size")
                if g.empty:
                    continue
                x = g["refit_pool_size"].astype(float).to_numpy()
                y = g["recovery_accuracy"].astype(float).to_numpy()
                sem = g["recovery_accuracy_sem"].astype(float).fillna(0.0).to_numpy()
                ax.errorbar(
                    x,
                    y,
                    yerr=CI_MULT * sem,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.4,
                    marker="o",
                    markersize=4.2,
                    capsize=2,
                    label=style["label"],
                    zorder=3,
                )
                partial = g["is_partial"].fillna(False).astype(bool).to_numpy()
                if partial.any():
                    ax.scatter(
                        x[partial],
                        y[partial],
                        s=34,
                        facecolors="white",
                        edgecolors=style["color"],
                        linewidths=1.2,
                        zorder=4,
                    )
            chance_n = float(sub["n_equivalence_classes"].max())
            if np.isfinite(chance_n) and chance_n > 0:
                ax.axhline(1.0 / chance_n, color="#666666", linestyle=":", linewidth=0.8)
            ax.set_xscale("log")
            ax.set_ylim(0, 1.04)
            if all_refits:
                ax.set_xticks(all_refits)
                ax.set_xticklabels([str(x) for x in all_refits], rotation=35, ha="right")
            ax.grid(color="#dddddd", linewidth=0.65, alpha=0.75)
            ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(model_set.replace("_", " "), fontsize=9)
            if c == 0:
                track_label = "Raw" if track == "raw" else "Encoding avg"
                mode_label = mode_display_name(mode)
                ax.set_ylabel(f"{mode_label}\n{track_label}\nRecovery", fontsize=8)
            if r == n_rows - 1:
                ax.set_xlabel("Refit pool size", fontsize=8)
    handles = [
        Line2D([0], [0], color=style["color"], linestyle=style["linestyle"], marker="o", label=style["label"])
        for style in SUBSET_STYLE.values()
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#444444",
            marker="o",
            markerfacecolor="white",
            linestyle="none",
            label="Partial cache",
        )
    )
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Teacher/student recovery at empirical SNR by refit pool size", y=1.035)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_noise_curves(summary: pd.DataFrame, out_base: Path) -> None:
    independent = summary[summary["eval_refit_mode"].astype(str).eq("independent")].copy()
    if independent.empty:
        return
    model_sets = ordered(independent["model_set"].astype(str), MODEL_ORDER)
    tracks = ordered(independent["report_track"].astype(str), DISPLAY_TRACK_ORDER)
    refits = sorted(independent["refit_pool_size"].dropna().astype(int).unique())
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, max(len(refits), 1)))
    refit_color = dict(zip(refits, colors))
    fig, axes = plt.subplots(
        len(tracks),
        len(model_sets),
        figsize=(3.15 * len(model_sets), 2.55 * len(tracks)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for r, track in enumerate(tracks):
        for c, model_set in enumerate(model_sets):
            ax = axes[r, c]
            sub = independent[
                independent["report_track"].astype(str).eq(track)
                & independent["model_set"].astype(str).eq(model_set)
            ]
            if sub.empty:
                ax.set_axis_off()
                continue
            for refit in refits:
                for subset_type, linestyle in [("selected", "-"), ("random", "--")]:
                    g = sub[
                        sub["refit_pool_size"].astype(int).eq(refit)
                        & sub["subset_type"].eq(subset_type)
                    ].sort_values("relative_snr")
                    if g.empty:
                        continue
                    label = f"{refit} {subset_type}" if (r == 0 and c == 0) else None
                    ax.plot(
                        g["relative_snr"].astype(float),
                        g["recovery_accuracy"].astype(float),
                        color=refit_color[refit],
                        linestyle=linestyle,
                        linewidth=1.25,
                        alpha=0.9,
                        label=label,
                    )
                    partial = g["is_partial"].fillna(False).any()
                    if partial:
                        emp = g[np.isclose(g["noise_mult"].astype(float), 1.0)]
                        if not emp.empty:
                            ax.scatter(
                                emp["relative_snr"].astype(float),
                                emp["recovery_accuracy"].astype(float),
                                s=22,
                                facecolors="white",
                                edgecolors=[refit_color[refit]],
                                linewidths=0.9,
                                zorder=4,
                            )
            chance_n = float(sub["n_equivalence_classes"].max())
            if np.isfinite(chance_n) and chance_n > 0:
                ax.axhline(1.0 / chance_n, color="#666666", linestyle=":", linewidth=0.8)
            ax.axvline(1.0, color="#222222", linestyle="-.", linewidth=0.75, alpha=0.65)
            ax.set_xscale("log")
            ax.set_ylim(0, 1.04)
            ax.grid(color="#dddddd", linewidth=0.65, alpha=0.75)
            ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(model_set.replace("_", " "), fontsize=9)
            if c == 0:
                ax.set_ylabel(("Raw" if track == "raw" else "Encoding avg") + "\nRecovery", fontsize=8)
            if r == len(tracks) - 1:
                ax.set_xlabel("Relative SNR", fontsize=8)
    refit_handles = [
        Line2D([0], [0], color=refit_color[refit], linewidth=1.6, label=str(refit))
        for refit in refits
    ]
    subset_handles = [
        Line2D([0], [0], color="#333333", linestyle="-", linewidth=1.4, label="Selected"),
        Line2D([0], [0], color="#333333", linestyle="--", linewidth=1.4, label="Random"),
    ]
    fig.legend(
        handles=refit_handles + subset_handles,
        loc="upper center",
        ncol=min(len(refit_handles) + len(subset_handles), 8),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        f"{mode_display_name('independent')} teacher/student recovery curves by refit pool size",
        y=1.065,
    )
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def copy_png(figures_root: Path, name: str) -> None:
    png_dir = figures_root / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    src = figures_root / f"{name}.png"
    if src.exists():
        (png_dir / src.name).write_bytes(src.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    parser.add_argument("--figures-root", type=Path, default=FIGURES)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--name", default="teacher_student_refit_size_available")
    args = parser.parse_args()

    run_dirs = discover_runs(args.results_root, args.pattern)
    if not run_dirs:
        raise SystemExit(f"No teacher/student result dirs matched pattern: {args.pattern}")

    summaries = []
    inventory_rows = []
    for run_dir in run_dirs:
        metadata = load_metadata(run_dir)
        completed = read_completed_summary(run_dir, metadata)
        if completed.empty:
            summary = summarize_cache(run_dir, metadata)
        else:
            summary = completed
        if summary.empty:
            continue
        summaries.append(summary)
        inventory_rows.append(
            {
                "source_dir": run_dir.name,
                "model_set": infer_model_set(run_dir, metadata),
                "eval_refit_mode": infer_eval_refit_mode(run_dir, metadata),
                "refit_pool_size": infer_refit_pool_size(run_dir, metadata),
                "source_kind": str(summary["source_kind"].iloc[0]),
                "is_partial": bool(summary["is_partial"].any()),
                "n_cache_shards_read": summary["n_cache_shards_read"].max(),
                "n_expected_cache_shards": summary["n_expected_cache_shards"].max(),
                "cache_shard_fraction": summary["cache_shard_fraction"].max(),
                "n_rows_summary": len(summary),
            }
        )

    if not summaries:
        raise SystemExit("No usable summary or cache data found.")

    data = pd.concat(summaries, ignore_index=True, sort=False)
    report = make_raw_and_encoding_avg(data)
    if report.empty:
        raise SystemExit("No raw or encoding rows found after aggregation.")

    args.figures_root.mkdir(parents=True, exist_ok=True)
    summary_csv = args.figures_root / f"{args.name}_summary.csv"
    empirical_csv = args.figures_root / f"{args.name}_empirical_snr.csv"
    inventory_csv = args.figures_root / f"{args.name}_run_inventory.csv"
    report.to_csv(summary_csv, index=False)
    empirical_snr_table(report).to_csv(empirical_csv, index=False)
    pd.DataFrame(inventory_rows).sort_values(
        ["eval_refit_mode", "model_set", "refit_pool_size", "source_dir"]
    ).to_csv(inventory_csv, index=False)

    empirical_name = f"{args.name}_empirical_snr"
    curves_name = f"{args.name}_curves"
    plot_empirical_snr(report, args.figures_root / empirical_name)
    plot_noise_curves(report, args.figures_root / curves_name)
    copy_png(args.figures_root, empirical_name)
    copy_png(args.figures_root, curves_name)

    print(f"Wrote {summary_csv}")
    print(f"Wrote {empirical_csv}")
    print(f"Wrote {inventory_csv}")
    print(f"Wrote {args.figures_root / (empirical_name + '.png')}")
    print(f"Wrote {args.figures_root / (curves_name + '.png')}")


if __name__ == "__main__":
    main()
