#!/usr/bin/env python3
"""Plot feature-method teacher/student recovery curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


SCRIPT = Path(__file__).resolve()
EVAL_ROOT = SCRIPT.parents[2]
SWEEP_ROOT = EVAL_ROOT / "feature_method_sweep_recovery"
DEFAULT_RUN = SWEEP_ROOT / "teacher_student" / "results" / "sota_20260611_112941"
DEFAULT_RESULTS_NAME = "teacher_student_rdm_score_spearman_response_empcal_ns20"
DEFAULT_FIGURES = SWEEP_ROOT / "teacher_student" / "figures"
METHOD_ORDER = [
    "raw_only_mean_min",
    "raw_only_mean_min_no_attenuation",
    "sub01_only_mean_min",
    "sub01_only_mean_min_no_attenuation",
    "raw_enc_w05_mean_min",
    "raw_enc_w05_mean_min_no_attenuation",
]
METHOD_LABELS = {
    "raw_only_mean_min": "Raw features only",
    "raw_only_mean_min_no_attenuation": "Raw features only (no attenuation)",
    "sub01_only_mean_min": "Sub-01 only (current)",
    "sub01_only_mean_min_no_attenuation": "Sub-01 only (no attenuation)",
    "raw_enc_w05_mean_min": "Intended (Raw + enc, mean/min)",
    "raw_enc_w05_mean_min_no_attenuation": "Intended (Raw + enc, no attenuation)",
}
METHOD_COLORS = {
    "raw_only_mean_min": "#4C78A8",
    "raw_only_mean_min_no_attenuation": "#8FB7DD",
    "sub01_only_mean_min": "#F28E2B",
    "sub01_only_mean_min_no_attenuation": "#FFBE7D",
    "raw_enc_w05_mean_min": "#59A14F",
    "raw_enc_w05_mean_min_no_attenuation": "#8CD17D",
}
RAW_TRACK = "raw"
ENCODING_TRACKS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
REPORT_TRACKS = ["raw", "encoding_avg"]
CI_MULT = 1.96


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


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def ordered_methods(methods: list[str]) -> list[str]:
    requested = [method for method in methods if method]
    return [
        *[method for method in METHOD_ORDER if method in requested],
        *[method for method in requested if method not in METHOD_ORDER],
    ]


def method_color(method_id: str, index: int) -> object:
    if method_id in METHOD_COLORS:
        return METHOD_COLORS[method_id]
    return plt.get_cmap("tab10")(index % 10)


def load_results(run_dir: Path, results_name: str, methods: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for method_id in methods:
        path = run_dir / results_name / method_id / "discriminability.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["method_id"] = method_id
        df["method_label"] = METHOD_LABELS.get(method_id, method_id)
        df["source_dir"] = str(path.parent.relative_to(run_dir))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    if "eval_refit_mode" in data:
        data["eval_refit_mode"] = data["eval_refit_mode"].fillna("independent")
    if "relative_snr" not in data:
        data["relative_snr"] = 1.0 / data["noise_mult"].astype(float)
    return data


def make_raw_and_encoding_avg(data: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    raw = data[data["track"] == RAW_TRACK].copy()
    if not raw.empty:
        raw["report_track"] = "raw"
        raw["n_subject_tracks"] = 0
        raw["subject_tracks"] = ""
        if "recovery_accuracy_sd" not in raw:
            raw["recovery_accuracy_sd"] = np.nan
        if "recovery_accuracy_sem" in raw:
            raw["recovery_accuracy_sem"] = raw["recovery_accuracy_sem"].combine_first(
                binomial_sem(raw["recovery_accuracy"], raw["n_units"])
            )
        else:
            raw["recovery_accuracy_sem"] = binomial_sem(raw["recovery_accuracy"], raw["n_units"])
        if "n_refit_repeats" not in raw:
            raw["n_refit_repeats"] = 1

    enc = data[data["track"].isin(ENCODING_TRACKS)].copy()
    if not enc.empty and "n_refit_repeats" not in enc:
        enc["n_refit_repeats"] = 1
    enc_rows: list[dict[str, object]] = []
    base_keys = [
        "method_id",
        "method_label",
        "metric",
        "corr_type",
        "noise_mult",
        "relative_snr",
        "noise_ceiling",
        "subset_type",
        "eval_noise_mode",
        "fit_noise_calibration",
        "eval_refit_mode",
        "source_dir",
    ]
    keys = [key for key in base_keys if key in enc.columns]
    if not enc.empty:
        for key, group in enc.groupby(keys, sort=False, dropna=False):
            by_track = (
                group.groupby("track", as_index=False)
                .agg(
                    recovery_accuracy=("recovery_accuracy", "mean"),
                    error_prob=("error_prob", "mean"),
                    mean_margin=("mean_margin", "mean"),
                    n_units=("n_units", "sum"),
                    n_models=("n_models", "max"),
                    n_equivalence_classes=("n_equivalence_classes", "max"),
                    n_noise_samples=("n_noise_samples", "max"),
                    refit_pool_size=("refit_pool_size", "max"),
                    refit_train_n=("refit_train_n", "max"),
                    refit_val_n=("refit_val_n", "max"),
                    n_refit_repeats=("n_refit_repeats", "max"),
                )
            )
            present_tracks = sorted(set(group["track"].astype(str)))
            row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
            row |= {
                "track": "encoding_avg",
                "report_track": "encoding_avg",
                "track_type": "encoding_average",
                "recovery_accuracy": float(by_track["recovery_accuracy"].mean()),
                "recovery_accuracy_sd": sample_sd(by_track["recovery_accuracy"]),
                "recovery_accuracy_sem": sample_sem(by_track["recovery_accuracy"]),
                "error_prob": float(by_track["error_prob"].mean()),
                "mean_margin": float(by_track["mean_margin"].mean()),
                "n_units": int(by_track["n_units"].sum()),
                "n_subsets": int(group["n_subsets"].max()),
                "n_models": int(by_track["n_models"].max()),
                "n_equivalence_classes": int(by_track["n_equivalence_classes"].max()),
                "n_noise_samples": int(by_track["n_noise_samples"].max()),
                "refit_pool_size": int(by_track["refit_pool_size"].max()),
                "refit_train_n": int(by_track["refit_train_n"].max()),
                "refit_val_n": int(by_track["refit_val_n"].max()),
                "n_refit_repeats": int(by_track["n_refit_repeats"].max()),
                "n_subject_tracks": len(present_tracks),
                "subject_tracks": ",".join(present_tracks),
            }
            enc_rows.append(row)
    enc_avg = pd.DataFrame(enc_rows)
    out = pd.concat([raw, enc_avg], ignore_index=True, sort=False)
    if out.empty:
        return out
    out["method_id"] = pd.Categorical(
        out["method_id"],
        categories=ordered_methods(methods),
        ordered=True,
    )
    out["report_track"] = pd.Categorical(
        out["report_track"],
        categories=REPORT_TRACKS,
        ordered=True,
    )
    return out.sort_values(["method_id", "report_track", "subset_type", "noise_mult"])


def make_auc(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["method_id", "method_label", "report_track", "track_type", "subset_type"]
    for key, group in summary.groupby(keys, sort=False, observed=True):
        rows.append(
            dict(zip(keys, key))
            | {
                "error_auc": compute_log_auc(group["noise_mult"], group["error_prob"]),
                "recovery_accuracy_auc": compute_log_auc(
                    group["noise_mult"],
                    group["recovery_accuracy"],
                ),
                "mean_margin_auc": compute_log_auc(group["noise_mult"], group["mean_margin"]),
                "n_noise_levels": int(group["noise_mult"].nunique()),
                "n_models": int(group["n_models"].max()) if "n_models" in group else np.nan,
                "n_subject_tracks": (
                    int(group["n_subject_tracks"].max())
                    if "n_subject_tracks" in group
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def random_repeat_summary(summary: pd.DataFrame) -> pd.DataFrame:
    random = summary[summary["subset_type"] == "random"].copy()
    if random.empty:
        return pd.DataFrame()
    base_keys = [
        "report_track",
        "metric",
        "corr_type",
        "noise_mult",
        "relative_snr",
        "noise_ceiling",
        "eval_noise_mode",
        "fit_noise_calibration",
        "eval_refit_mode",
    ]
    keys = [key for key in base_keys if key in random.columns]
    rows: list[dict[str, object]] = []
    for key, group in random.groupby(keys, sort=False, dropna=False, observed=True):
        repeats = (
            group.groupby("method_id", observed=True, as_index=False)
            .agg(
                recovery_accuracy=("recovery_accuracy", "mean"),
                error_prob=("error_prob", "mean"),
                mean_margin=("mean_margin", "mean"),
                n_units=("n_units", "sum"),
                n_models=("n_models", "max"),
                n_equivalence_classes=("n_equivalence_classes", "max"),
                n_subject_tracks=("n_subject_tracks", "max"),
            )
            .dropna(subset=["method_id"])
        )
        if repeats.empty:
            continue
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        acc = repeats["recovery_accuracy"]
        margin = repeats["mean_margin"]
        methods = [
            str(method)
            for method in repeats["method_id"].astype(str)
            if str(method) != "nan"
        ]
        row |= {
            "subset_type": "random_repeat_mean",
            "recovery_accuracy": float(acc.mean()),
            "recovery_accuracy_sd": sample_sd(acc),
            "recovery_accuracy_sem": sample_sem(acc),
            "error_prob": float(repeats["error_prob"].mean()),
            "error_prob_sd": sample_sd(repeats["error_prob"]),
            "error_prob_sem": sample_sem(repeats["error_prob"]),
            "mean_margin": float(margin.mean()),
            "mean_margin_sd": sample_sd(margin),
            "mean_margin_sem": sample_sem(margin),
            "n_random_repeats": int(repeats["method_id"].nunique()),
            "random_repeat_methods": ",".join(methods),
            "n_units": int(repeats["n_units"].sum()),
            "n_models": int(repeats["n_models"].max()),
            "n_equivalence_classes": int(repeats["n_equivalence_classes"].max()),
            "n_subject_tracks": int(repeats["n_subject_tracks"].max()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def empirical_snr_table(summary: pd.DataFrame) -> pd.DataFrame:
    emp = summary[np.isclose(summary["noise_mult"].astype(float), 1.0)].copy()
    if emp.empty:
        return emp
    selected = emp[emp["subset_type"] == "selected"][
        [
            "method_id",
            "method_label",
            "report_track",
            "recovery_accuracy",
            "mean_margin",
            "n_subject_tracks",
            "subject_tracks",
        ]
    ].rename(
        columns={
            "recovery_accuracy": "selected_recovery_accuracy",
            "mean_margin": "selected_mean_margin",
        }
    )
    random = random_repeat_summary(emp)
    if random.empty:
        random = emp[emp["subset_type"] == "random"][
            ["report_track", "recovery_accuracy", "mean_margin"]
        ].rename(
            columns={
                "recovery_accuracy": "random_recovery_accuracy",
                "mean_margin": "random_mean_margin",
            }
        )
    else:
        random = random[
            [
                "report_track",
                "recovery_accuracy",
                "recovery_accuracy_sd",
                "recovery_accuracy_sem",
                "mean_margin",
                "mean_margin_sd",
                "mean_margin_sem",
                "n_random_repeats",
                "random_repeat_methods",
            ]
        ].rename(
            columns={
                "recovery_accuracy": "random_recovery_accuracy",
                "recovery_accuracy_sd": "random_recovery_accuracy_sd",
                "recovery_accuracy_sem": "random_recovery_accuracy_sem",
                "mean_margin": "random_mean_margin",
                "mean_margin_sd": "random_mean_margin_sd",
                "mean_margin_sem": "random_mean_margin_sem",
            }
        )
    out = selected.merge(random, on=["report_track"], how="outer")
    out["selected_minus_random"] = (
        out["selected_recovery_accuracy"] - out["random_recovery_accuracy"]
    )
    return out.sort_values(["method_id", "report_track"])


def add_random_repeat_band(ax: plt.Axes, sub: pd.DataFrame) -> None:
    random = random_repeat_summary(sub)
    if random.empty:
        return
    random = random.sort_values("relative_snr")
    x = random["relative_snr"].astype(float).to_numpy()
    y = random["recovery_accuracy"].astype(float).to_numpy()
    spread = random["recovery_accuracy_sd"].astype(float).fillna(0.0).to_numpy()
    lo = np.clip(y - spread, 0.0, 1.0)
    hi = np.clip(y + spread, 0.0, 1.0)
    ax.fill_between(
        x,
        lo,
        hi,
        color="#222222",
        alpha=0.20,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        x,
        y,
        linestyle="--",
        color="#222222",
        linewidth=2.0,
        alpha=0.9,
        zorder=3,
    )


def plot_curves(summary: pd.DataFrame, out_base: Path, methods: list[str]) -> None:
    spaces = [track for track in REPORT_TRACKS if track in set(summary["report_track"].astype(str))]
    fig, axes = plt.subplots(1, len(spaces), figsize=(6.2 * len(spaces), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, report_track in zip(axes, spaces):
        sub = summary[summary["report_track"].astype(str) == report_track]
        add_random_repeat_band(ax, sub)
        for color_idx, method_id in enumerate(ordered_methods(methods)):
            method_df = sub[sub["method_id"].astype(str) == method_id]
            if method_df.empty:
                continue
            color = method_color(method_id, color_idx)
            g = method_df[method_df["subset_type"] == "selected"].sort_values("relative_snr")
            if g.empty:
                continue
            x = g["relative_snr"].astype(float).to_numpy()
            y = g["recovery_accuracy"].astype(float).to_numpy()
            if "recovery_accuracy_sem" in g:
                ci = (
                    CI_MULT
                    * g["recovery_accuracy_sem"].astype(float).fillna(0.0).to_numpy()
                )
                ax.fill_between(
                    x,
                    np.clip(y - ci, 0.0, 1.0),
                    np.clip(y + ci, 0.0, 1.0),
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                    zorder=2,
                )
            ax.plot(
                x,
                y,
                linestyle="-",
                color=color,
                linewidth=2.0,
                alpha=1.0,
                zorder=4,
            )
        if "n_equivalence_classes" in sub:
            chance_n = float(sub["n_equivalence_classes"].max())
        else:
            chance_n = float(sub["n_models"].max()) if "n_models" in sub else np.nan
        if np.isfinite(chance_n) and chance_n > 0:
            ax.axhline(1.0 / chance_n, color="#666666", linestyle=":", linewidth=0.8)
        ax.axvline(
            1.0,
            color="#222222",
            linestyle="-.",
            linewidth=0.8,
            alpha=0.65,
        )
        subj = sub["subject_tracks"].dropna().astype(str)
        if report_track == "encoding_avg" and not subj.empty:
            tracks = sorted(set(",".join(x for x in subj if x).split(",")) - {""})
            if tracks:
                ax.text(
                    0.02,
                    0.04,
                    ", ".join(tracks),
                    transform=ax.transAxes,
                    fontsize=7,
                    color="#444444",
                    va="bottom",
                    ha="left",
                )
        ax.set_xscale("log")
        ax.set_ylim(0, 1.04)
        ax.grid(color="#dddddd", linewidth=0.65, alpha=0.75)
        ax.set_axisbelow(True)
        ax.set_title("Raw" if report_track == "raw" else "Encoding avg")
        ax.set_xlabel("Relative SNR")
    axes[0].set_ylabel("Recovery accuracy")
    method_handles = [
        plt.Line2D(
            [0],
            [0],
            color=method_color(m, idx),
            linewidth=2.0,
            label=METHOD_LABELS.get(m, m),
        )
        for idx, m in enumerate(ordered_methods(methods))
        if m in set(summary["method_id"].astype(str))
    ]
    style_handles = [
        plt.Line2D(
            [0],
            [0],
            color="#222222",
            linestyle="--",
            linewidth=2.0,
            label="Random mean",
        ),
        Patch(
            facecolor="#222222",
            edgecolor="none",
            alpha=0.20,
            label="Random +/- SD",
        ),
        Patch(
            facecolor="#777777",
            edgecolor="none",
            alpha=0.18,
            label="Selected 95% CI",
        ),
        plt.Line2D(
            [0],
            [0],
            color="#222222",
            linestyle="-.",
            linewidth=0.8,
            alpha=0.65,
            label="Empirical SNR",
        ),
    ]
    fig.legend(
        handles=method_handles + style_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        fontsize=8,
    )
    fig.suptitle("Teacher/student recovery: feature-method sweep", y=1.15)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--results-name", default=DEFAULT_RESULTS_NAME)
    parser.add_argument("--figures-root", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--methods", default=",".join(METHOD_ORDER))
    parser.add_argument("--name", default="teacher_student_recovery_feature_method_sweep_ns20")
    args = parser.parse_args()

    methods = parse_csv_list(args.methods)
    data = load_results(args.run_dir, args.results_name, methods)
    if data.empty:
        raise SystemExit("No completed feature-method teacher/student results found.")
    summary = make_raw_and_encoding_avg(data, methods)
    args.figures_root.mkdir(parents=True, exist_ok=True)
    summary_csv = args.figures_root / f"{args.name}_raw_encoding_avg_summary.csv"
    auc_csv = args.figures_root / f"{args.name}_raw_encoding_avg_auc.csv"
    empirical_csv = args.figures_root / f"{args.name}_empirical_snr.csv"
    random_csv = args.figures_root / f"{args.name}_random_repeat_summary.csv"
    summary.to_csv(summary_csv, index=False)
    make_auc(summary).to_csv(auc_csv, index=False)
    empirical_snr_table(summary).to_csv(empirical_csv, index=False)
    random_repeat_summary(summary).to_csv(random_csv, index=False)
    plot_curves(summary, args.figures_root / args.name, methods)
    png_dir = args.figures_root / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    png = args.figures_root / f"{args.name}.png"
    if png.exists():
        (png_dir / png.name).write_bytes(png.read_bytes())
    print(f"Wrote {summary_csv}")
    print(f"Wrote {auc_csv}")
    print(f"Wrote {empirical_csv}")
    print(f"Wrote {random_csv}")
    print(f"Wrote {args.figures_root / (args.name + '.png')}")


if __name__ == "__main__":
    main()
