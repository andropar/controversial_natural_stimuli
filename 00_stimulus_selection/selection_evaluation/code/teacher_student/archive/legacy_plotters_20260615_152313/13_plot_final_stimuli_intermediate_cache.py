#!/usr/bin/env python3
"""Plot intermediate teacher/student recovery directly from cache shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
EVAL_ROOT = SCRIPT.parents[2]
RESULTS = EVAL_ROOT / "final_stimuli_recovery" / "teacher_student" / "results"
FIGURES = EVAL_ROOT / "final_stimuli_recovery" / "teacher_student" / "figures"
DATA_SUFFIX = "_teacher_student_independent_refit_1k_rdm_score_spearman_response_empcal_ns20"
MODEL_ORDER = ["all_models", "sota", "training_objective", "architecture", "dataset"]
EXPECTED_TEACHERS = {
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


def model_set_from_source_dir(name: str, suffix: str) -> str:
    suffix = suffix.lstrip("_")
    marker = f"_{suffix}"
    if name.endswith(marker):
        return name[: -len(marker)]
    return name


def read_cache(results_root: Path, suffix: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for result_dir in sorted(results_root.glob(f"*{suffix}")):
        if not result_dir.is_dir():
            continue
        cache_root = result_dir / "_teacher_cache"
        if not cache_root.exists():
            continue
        source_model_set = model_set_from_source_dir(result_dir.name, suffix)
        paths = sorted(cache_root.glob("**/*.csv"))
        for path in paths:
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                print(f"Skipping unreadable cache shard {path}: {exc}")
                continue
            if df.empty:
                continue
            df["source_dir"] = result_dir.name
            df["cache_file"] = path.name
            if "model_set" not in df:
                df["model_set"] = source_model_set
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
    keys = [
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
        "source_dir",
    ]
    for key, group in data.groupby(keys, sort=False, dropna=False):
        row = dict(zip(keys, key))
        correct = group["recovered_correct"].astype(float)
        model_set = str(row["model_set"])
        expected = EXPECTED_TEACHERS.get(model_set, int(group["teacher_model"].nunique()))
        row |= {
            "recovery_accuracy": float(correct.mean()),
            "error_prob": float(1.0 - correct.mean()),
            "mean_margin": float(group["teacher_margin"].mean()),
            "n_units": int(len(group)),
            "n_subsets": int(group["subset_idx"].nunique())
            if row["subset_type"] == "random"
            else 1,
            "n_teacher_models_done": int(group["teacher_model"].nunique()),
            "n_teacher_models_expected": int(expected),
            "teacher_coverage": float(group["teacher_model"].nunique() / expected)
            if expected
            else np.nan,
            "n_equivalence_classes": int(group["n_equivalence_classes"].max())
            if "n_equivalence_classes" in group
            else int(expected),
            "n_noise_samples": int(group["noise_sample_idx"].nunique()),
            "refit_pool_size": int(group["refit_pool_size"].max()),
            "refit_train_n": int(group["refit_train_n"].max()),
            "refit_val_n": int(group["refit_val_n"].max()),
        }
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
    keys = [
        "model_set",
        "metric",
        "corr_type",
        "noise_mult",
        "relative_snr",
        "noise_ceiling",
        "subset_type",
        "eval_noise_mode",
        "fit_noise_calibration",
        "source_dir",
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
                    n_teacher_models_done=("n_teacher_models_done", "max"),
                    n_teacher_models_expected=("n_teacher_models_expected", "max"),
                    teacher_coverage=("teacher_coverage", "mean"),
                    n_equivalence_classes=("n_equivalence_classes", "max"),
                    n_noise_samples=("n_noise_samples", "max"),
                    refit_pool_size=("refit_pool_size", "max"),
                    refit_train_n=("refit_train_n", "max"),
                    refit_val_n=("refit_val_n", "max"),
                )
            )
            present_tracks = sorted(set(by_track["track"].astype(str)))
            row = dict(zip(keys, key))
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
                "n_teacher_models_done": int(by_track["n_teacher_models_done"].min()),
                "n_teacher_models_expected": int(by_track["n_teacher_models_expected"].max()),
                "teacher_coverage": float(by_track["teacher_coverage"].mean()),
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
    out["model_set"] = pd.Categorical(out["model_set"], categories=MODEL_ORDER, ordered=True)
    out["report_track"] = pd.Categorical(
        out["report_track"],
        categories=DISPLAY_TRACK_ORDER,
        ordered=True,
    )
    return out.sort_values(["model_set", "report_track", "subset_type", "noise_mult"])


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
    random = emp[emp["subset_type"] == "random"][
        ["model_set", "report_track", "recovery_accuracy", "mean_margin"]
    ].rename(
        columns={
            "recovery_accuracy": "random_recovery_accuracy",
            "mean_margin": "random_mean_margin",
        }
    )
    out = selected.merge(random, on=["model_set", "report_track"], how="outer")
    out["selected_minus_random"] = (
        out["selected_recovery_accuracy"] - out["random_recovery_accuracy"]
    )
    return out.sort_values(["model_set", "report_track"])


def ordered(values: pd.Series, preferred: list[str]) -> list[str]:
    present = list(dict.fromkeys(values.dropna().astype(str)))
    return [x for x in preferred if x in present] + [x for x in present if x not in preferred]


def plot_summary(summary: pd.DataFrame, out_base: Path) -> None:
    model_sets = ordered(summary["model_set"].astype(str), MODEL_ORDER)
    report_tracks = ordered(summary["report_track"].astype(str), DISPLAY_TRACK_ORDER)
    fig, axes = plt.subplots(
        len(report_tracks),
        len(model_sets),
        figsize=(3.5 * len(model_sets), 2.8 * len(report_tracks)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for r, report_track in enumerate(report_tracks):
        for c, model_set in enumerate(model_sets):
            ax = axes[r, c]
            sub = summary[
                (summary["report_track"].astype(str) == report_track)
                & (summary["model_set"].astype(str) == model_set)
            ]
            if sub.empty:
                ax.set_axis_off()
                continue
            for subset_type, ls in [("random", "--"), ("selected", "-")]:
                g = sub[sub["subset_type"] == subset_type].sort_values("relative_snr")
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
                        color=COLORS[subset_type],
                        alpha=0.12 if subset_type == "selected" else 0.09,
                        linewidth=0,
                        zorder=1,
                    )
                ax.plot(
                    x,
                    y,
                    ls,
                    color=COLORS[subset_type],
                    linewidth=1.9,
                    label=subset_type.capitalize(),
                    zorder=3,
                )
            chance_n = float(sub["n_equivalence_classes"].max())
            if np.isfinite(chance_n) and chance_n > 0:
                ax.axhline(1.0 / chance_n, color="#666666", linestyle=":", linewidth=0.8)
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
            ax.set_xscale("log")
            ax.set_ylim(0, 1.04)
            ax.grid(color="#dddddd", linewidth=0.65, alpha=0.75)
            ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(model_set.replace("_", " "), fontsize=10)
            if c == 0:
                label = "Raw" if report_track == "raw" else "Encoding avg"
                ax.set_ylabel(f"{label}\nRecovery accuracy", fontsize=9)
            if r == len(report_tracks) - 1:
                ax.set_xlabel("Relative SNR")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Intermediate teacher/student recovery from completed cache shards", y=1.06)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    parser.add_argument("--figures-root", type=Path, default=FIGURES)
    parser.add_argument("--data-suffix", default=DATA_SUFFIX)
    parser.add_argument("--name", default="teacher_student_rdm_score_intermediate_cache_ns20")
    args = parser.parse_args()

    data = read_cache(args.results_root, args.data_suffix)
    if data.empty:
        raise SystemExit("No cache shards found.")
    disc = discriminability_from_cache(data)
    summary = make_raw_and_encoding_avg(disc)
    if summary.empty:
        raise SystemExit("No raw or encoding rows found in cache shards.")

    args.figures_root.mkdir(parents=True, exist_ok=True)
    disc_csv = args.figures_root / f"{args.name}_per_track_summary.csv"
    summary_csv = args.figures_root / f"{args.name}_raw_encoding_avg_summary.csv"
    empirical_csv = args.figures_root / f"{args.name}_empirical_snr.csv"
    disc.to_csv(disc_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    empirical_snr_table(summary).to_csv(empirical_csv, index=False)
    plot_summary(summary, args.figures_root / args.name)
    png_dir = args.figures_root / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    png = args.figures_root / f"{args.name}.png"
    if png.exists():
        (png_dir / png.name).write_bytes(png.read_bytes())

    print(f"Wrote {disc_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {empirical_csv}")
    print(f"Wrote {args.figures_root / (args.name + '.png')}")


if __name__ == "__main__":
    main()
