#!/usr/bin/env python3
"""Plot intermediate feature-method teacher/student recovery from cache shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
EVAL_ROOT = SCRIPT.parents[2]
SWEEP_ROOT = EVAL_ROOT / "feature_method_sweep_recovery"
DEFAULT_RUN = SWEEP_ROOT / "teacher_student" / "results" / "sota_20260611_112941"
DEFAULT_RESULTS_NAME = "teacher_student_rdm_score_spearman_response_empcal_ns20"
DEFAULT_FIGURES = SWEEP_ROOT / "teacher_student" / "figures"
METHOD_ORDER = [
    "raw_only_mean_min",
    "sub01_only_mean_min",
    "raw_enc_w05_mean_min",
    "paper_effective_identity_sub01_mean_min_no_attenuation",
]
METHOD_LABELS = {
    "raw_only_mean_min": "Raw features only",
    "sub01_only_mean_min": "Sub-01 only (current)",
    "raw_enc_w05_mean_min": "Intended (Raw + enc, mean/min)",
    "paper_effective_identity_sub01_mean_min_no_attenuation": "Sub-01 only (no attenuation)",
}
METHOD_COLORS = {
    "raw_only_mean_min": "#4C78A8",
    "sub01_only_mean_min": "#F28E2B",
    "raw_enc_w05_mean_min": "#59A14F",
    "paper_effective_identity_sub01_mean_min_no_attenuation": "#9C755F",
}
EXPECTED_TEACHERS = 6
RAW_TRACK = "raw"
ENCODING_TRACKS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
REPORT_TRACKS = ["raw", "encoding_avg"]


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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


def read_cache(run_dir: Path, results_name: str, methods: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    root = run_dir / results_name
    for method_id in methods:
        cache_root = root / method_id / "_teacher_cache"
        if not cache_root.exists():
            continue
        for path in sorted(cache_root.glob("**/*.csv")):
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                print(f"Skipping unreadable cache shard {path}: {exc}")
                continue
            if df.empty:
                continue
            df["method_id"] = method_id
            df["method_label"] = METHOD_LABELS.get(method_id, method_id)
            df["source_dir"] = str((root / method_id).relative_to(run_dir))
            df["cache_file"] = path.name
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    if "eval_refit_mode" in data:
        data["eval_refit_mode"] = data["eval_refit_mode"].fillna("independent")
    if "relative_snr" not in data:
        data["relative_snr"] = 1.0 / data["noise_mult"].astype(float)
    return data


def discriminability_from_cache(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base_keys = [
        "method_id",
        "method_label",
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
        "source_dir",
    ]
    keys = [key for key in base_keys if key in data.columns]
    for key, group in data.groupby(keys, sort=False, dropna=False):
        row = dict(zip(keys, key))
        correct = group["recovered_correct"].astype(float)
        n_done = int(group["teacher_model"].nunique())
        row |= {
            "recovery_accuracy": float(correct.mean()),
            "error_prob": float(1.0 - correct.mean()),
            "mean_margin": float(group["teacher_margin"].mean()),
            "n_units": int(len(group)),
            "n_subsets": int(group["subset_idx"].nunique())
            if row["subset_type"] == "random"
            else 1,
            "n_models": EXPECTED_TEACHERS,
            "n_teacher_models_done": n_done,
            "n_teacher_models_expected": EXPECTED_TEACHERS,
            "teacher_coverage": float(n_done / EXPECTED_TEACHERS),
            "n_equivalence_classes": int(group["n_equivalence_classes"].max())
            if "n_equivalence_classes" in group
            else EXPECTED_TEACHERS,
            "n_noise_samples": int(group["noise_sample_idx"].nunique()),
            "refit_pool_size": int(group["refit_pool_size"].max()),
            "refit_train_n": int(group["refit_train_n"].max()),
            "refit_val_n": int(group["refit_val_n"].max()),
        }
        if "response_noise_std" in group:
            row["response_noise_std"] = float(group["response_noise_std"].mean())
        if "achieved_fit_rdm_reliability" in group:
            row["achieved_fit_rdm_reliability"] = float(
                group["achieved_fit_rdm_reliability"].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def make_raw_and_encoding_avg(summary: pd.DataFrame) -> pd.DataFrame:
    raw = summary[summary["track"] == RAW_TRACK].copy()
    if not raw.empty:
        raw["report_track"] = "raw"
        raw["n_subject_tracks"] = 0
        raw["subject_tracks"] = ""
        raw["recovery_accuracy_sd"] = np.nan
        raw["recovery_accuracy_sem"] = binomial_sem(raw["recovery_accuracy"], raw["n_units"])

    enc = summary[summary["track"].isin(ENCODING_TRACKS)].copy()
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
                    n_teacher_models_done=("n_teacher_models_done", "max"),
                    n_teacher_models_expected=("n_teacher_models_expected", "max"),
                    teacher_coverage=("teacher_coverage", "mean"),
                    n_models=("n_models", "max"),
                    n_equivalence_classes=("n_equivalence_classes", "max"),
                    n_noise_samples=("n_noise_samples", "max"),
                    refit_pool_size=("refit_pool_size", "max"),
                    refit_train_n=("refit_train_n", "max"),
                    refit_val_n=("refit_val_n", "max"),
                    response_noise_std=("response_noise_std", "mean"),
                    achieved_fit_rdm_reliability=(
                        "achieved_fit_rdm_reliability",
                        "mean",
                    ),
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
                "n_subsets": int(group["n_subsets"].max()),
                "n_models": int(by_track["n_models"].max()),
                "n_teacher_models_done": int(by_track["n_teacher_models_done"].min()),
                "n_teacher_models_expected": int(by_track["n_teacher_models_expected"].max()),
                "teacher_coverage": float(by_track["teacher_coverage"].mean()),
                "n_equivalence_classes": int(by_track["n_equivalence_classes"].max()),
                "n_noise_samples": int(by_track["n_noise_samples"].max()),
                "refit_pool_size": int(by_track["refit_pool_size"].max()),
                "refit_train_n": int(by_track["refit_train_n"].max()),
                "refit_val_n": int(by_track["refit_val_n"].max()),
                "response_noise_std": float(by_track["response_noise_std"].mean()),
                "achieved_fit_rdm_reliability": float(
                    by_track["achieved_fit_rdm_reliability"].mean()
                ),
                "n_subject_tracks": len(present_tracks),
                "subject_tracks": ",".join(present_tracks),
            }
            enc_rows.append(row)
    enc_avg = pd.DataFrame(enc_rows)
    out = pd.concat([raw, enc_avg], ignore_index=True, sort=False)
    if out.empty:
        return out
    out["method_id"] = pd.Categorical(out["method_id"], categories=METHOD_ORDER, ordered=True)
    out["report_track"] = pd.Categorical(
        out["report_track"],
        categories=REPORT_TRACKS,
        ordered=True,
    )
    return out.sort_values(["method_id", "report_track", "subset_type", "noise_mult"])


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
            group.groupby("method_id", as_index=False, observed=True)
            .agg(
                recovery_accuracy=("recovery_accuracy", "mean"),
                error_prob=("error_prob", "mean"),
                mean_margin=("mean_margin", "mean"),
                n_units=("n_units", "sum"),
                n_models=("n_models", "max"),
                n_equivalence_classes=("n_equivalence_classes", "max"),
                n_subject_tracks=("n_subject_tracks", "max"),
                n_teacher_models_done=("n_teacher_models_done", "min"),
                n_teacher_models_expected=("n_teacher_models_expected", "max"),
                teacher_coverage=("teacher_coverage", "mean"),
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
            "n_teacher_models_done": int(repeats["n_teacher_models_done"].min()),
            "n_teacher_models_expected": int(repeats["n_teacher_models_expected"].max()),
            "teacher_coverage": float(repeats["teacher_coverage"].mean()),
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
            "n_teacher_models_done",
            "n_teacher_models_expected",
            "teacher_coverage",
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
    sem = random["recovery_accuracy_sem"].astype(float).fillna(0.0).to_numpy()
    ax.fill_between(
        x,
        np.clip(y - sem, 0.0, 1.0),
        np.clip(y + sem, 0.0, 1.0),
        color="#222222",
        alpha=0.14,
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


def plot_curves(summary: pd.DataFrame, out_base: Path) -> None:
    spaces = [track for track in REPORT_TRACKS if track in set(summary["report_track"].astype(str))]
    fig, axes = plt.subplots(1, len(spaces), figsize=(6.2 * len(spaces), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, report_track in zip(axes, spaces):
        sub = summary[summary["report_track"].astype(str) == report_track]
        add_random_repeat_band(ax, sub)
        for method_id in METHOD_ORDER:
            method_df = sub[sub["method_id"].astype(str) == method_id]
            if method_df.empty:
                continue
            color = METHOD_COLORS.get(method_id, "#777777")
            g = method_df[method_df["subset_type"] == "selected"].sort_values("relative_snr")
            if g.empty:
                continue
            x = g["relative_snr"].astype(float).to_numpy()
            y = g["recovery_accuracy"].astype(float).to_numpy()
            if "recovery_accuracy_sem" in g:
                sem = g["recovery_accuracy_sem"].astype(float).fillna(0.0).to_numpy()
                ax.fill_between(
                    x,
                    np.clip(y - sem, 0.0, 1.0),
                    np.clip(y + sem, 0.0, 1.0),
                    color=color,
                    alpha=0.12,
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
        chance_n = float(sub["n_equivalence_classes"].max())
        if np.isfinite(chance_n) and chance_n > 0:
            ax.axhline(1.0 / chance_n, color="#666666", linestyle=":", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.04)
        ax.grid(color="#dddddd", linewidth=0.65, alpha=0.75)
        ax.set_axisbelow(True)
        ax.set_title("Raw" if report_track == "raw" else "Encoding avg")
        ax.set_xlabel("Relative SNR")
        coverage = sub["teacher_coverage"].dropna()
        subj = sub["subject_tracks"].dropna().astype(str)
        subtitle = ""
        if not coverage.empty:
            subtitle = f"{coverage.min():.0%}-{coverage.max():.0%} teachers"
        if report_track == "encoding_avg" and not subj.empty:
            tracks = sorted(set(",".join(x for x in subj if x).split(",")) - {""})
            if tracks:
                subtitle = f"{', '.join(tracks)}; {subtitle}"
        ax.text(
            0.02,
            0.04,
            subtitle,
            transform=ax.transAxes,
            fontsize=7,
            color="#444444",
            va="bottom",
            ha="left",
        )
    axes[0].set_ylabel("Recovery accuracy")
    method_handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS[m], linewidth=2.0, label=METHOD_LABELS[m])
        for m in METHOD_ORDER
        if m in set(summary["method_id"].astype(str))
    ]
    style_handles = [
        plt.Line2D(
            [0],
            [0],
            color="#222222",
            linestyle="--",
            linewidth=2.0,
            label="Random mean +/- SEM",
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
    fig.suptitle(
        "Intermediate feature-method teacher/student RDM-score recovery",
        y=1.15,
    )
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
    parser.add_argument(
        "--name",
        default="feature_method_sweep_teacher_student_rdm_score_intermediate_ns20",
    )
    args = parser.parse_args()

    methods = parse_csv_list(args.methods)
    data = read_cache(args.run_dir, args.results_name, methods)
    if data.empty:
        raise SystemExit("No feature-method cache shards found.")
    disc = discriminability_from_cache(data)
    summary = make_raw_and_encoding_avg(disc)
    if summary.empty:
        raise SystemExit("No raw or encoding rows found in cache shards.")

    args.figures_root.mkdir(parents=True, exist_ok=True)
    disc_csv = args.figures_root / f"{args.name}_per_track_summary.csv"
    summary_csv = args.figures_root / f"{args.name}_raw_encoding_avg_summary.csv"
    empirical_csv = args.figures_root / f"{args.name}_empirical_snr.csv"
    random_csv = args.figures_root / f"{args.name}_random_repeat_summary.csv"
    disc.to_csv(disc_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    empirical_snr_table(summary).to_csv(empirical_csv, index=False)
    random_repeat_summary(summary).to_csv(random_csv, index=False)
    plot_curves(summary, args.figures_root / args.name)
    png_dir = args.figures_root / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    png = args.figures_root / f"{args.name}.png"
    if png.exists():
        (png_dir / png.name).write_bytes(png.read_bytes())

    print(f"Wrote {disc_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {empirical_csv}")
    print(f"Wrote {random_csv}")
    print(f"Wrote {args.figures_root / (args.name + '.png')}")


if __name__ == "__main__":
    main()
