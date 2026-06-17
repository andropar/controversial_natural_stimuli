#!/usr/bin/env python3
"""Plot raw and subject-averaged encoding teacher/student recovery curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
RESULTS = SCRIPT.parents[1] / "results"
FIGURES = SCRIPT.parents[1] / "figures"
DATA_SUFFIX = "_teacher_student_independent_refit_1k_rdm_score_spearman_response_empcal_ns20"
MODEL_ORDER = ["all_models", "sota", "training_objective", "architecture", "dataset"]
RAW_TRACK = "raw"
ENCODING_TRACKS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
DISPLAY_TRACK_ORDER = ["raw", "encoding_avg"]
COLORS = {"selected": "#4C78A8", "random": "#E45756"}


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


def load_results(results_root: Path, suffix: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for model_set in MODEL_ORDER:
        path = results_root / f"{model_set}{suffix}" / "discriminability.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["source_dir"] = path.parent.name
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    data = pd.concat(rows, ignore_index=True)
    if "eval_refit_mode" in data:
        data["eval_refit_mode"] = data["eval_refit_mode"].fillna("independent")
    if "relative_snr" not in data:
        data["relative_snr"] = 1.0 / data["noise_mult"].astype(float)
    return data


def make_raw_and_encoding_avg(data: pd.DataFrame) -> pd.DataFrame:
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
        "source_dir",
    ]
    keys = [key for key in base_keys if key in enc.columns]
    if not enc.empty:
        for key, group in enc.groupby(keys, sort=False, dropna=False):
            present_tracks = sorted(set(group["track"].astype(str)))
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
    out["report_track"] = pd.Categorical(
        out["report_track"],
        categories=DISPLAY_TRACK_ORDER,
        ordered=True,
    )
    return out.sort_values(["model_set", "report_track", "subset_type", "noise_mult"])


def make_auc(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["model_set", "report_track", "track_type", "subset_type"]
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
                "n_equivalence_classes": (
                    int(group["n_equivalence_classes"].max())
                    if "n_equivalence_classes" in group
                    else np.nan
                ),
                "n_subject_tracks": (
                    int(group["n_subject_tracks"].max())
                    if "n_subject_tracks" in group
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def empirical_snr_table(summary: pd.DataFrame) -> pd.DataFrame:
    emp = summary[np.isclose(summary["noise_mult"].astype(float), 1.0)].copy()
    if emp.empty:
        return emp
    selected = emp[emp["subset_type"] == "selected"][
        ["model_set", "report_track", "recovery_accuracy", "mean_margin"]
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
    out["report_track"] = pd.Categorical(
        out["report_track"],
        categories=DISPLAY_TRACK_ORDER,
        ordered=True,
    )
    out["model_set"] = pd.Categorical(out["model_set"], categories=MODEL_ORDER, ordered=True)
    return out.sort_values(["model_set", "report_track"])


def ordered(values: pd.Series, preferred: list[str]) -> list[str]:
    present = list(dict.fromkeys(values.dropna().astype(str)))
    return [x for x in preferred if x in present] + [x for x in present if x not in preferred]


def plot_summary(summary: pd.DataFrame, out_base: Path) -> None:
    model_sets = ordered(summary["model_set"], MODEL_ORDER)
    report_tracks = ordered(summary["report_track"].astype(str), DISPLAY_TRACK_ORDER)
    fig, axes = plt.subplots(
        len(report_tracks),
        len(model_sets),
        figsize=(3.4 * len(model_sets), 2.65 * len(report_tracks)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for r, report_track in enumerate(report_tracks):
        for c, model_set in enumerate(model_sets):
            ax = axes[r, c]
            sub = summary[
                (summary["report_track"].astype(str) == report_track)
                & (summary["model_set"] == model_set)
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
            if "n_equivalence_classes" in sub:
                chance_n = float(sub["n_equivalence_classes"].max())
            else:
                chance_n = float(sub["n_models"].max())
            if np.isfinite(chance_n) and chance_n > 0:
                ax.axhline(1.0 / chance_n, color="#666666", linestyle=":", linewidth=0.8)
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
    fig.suptitle("Teacher/student recovery: raw and subject-averaged encoding", y=1.06)
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
    parser.add_argument("--data-suffix", default=DATA_SUFFIX)
    parser.add_argument("--name", default="teacher_student_rdm_score_raw_encoding_avg")
    args = parser.parse_args()

    data = load_results(args.results_root, args.data_suffix)
    if data.empty:
        raise SystemExit("No completed RDM-score discriminability.csv files found.")
    summary = make_raw_and_encoding_avg(data)
    if summary.empty:
        raise SystemExit("No raw or encoding-track rows found.")

    args.figures_root.mkdir(parents=True, exist_ok=True)
    summary_csv = args.figures_root / f"{args.name}_curves_summary.csv"
    auc_csv = args.figures_root / f"{args.name}_auc_summary.csv"
    empirical_csv = args.figures_root / f"{args.name}_empirical_snr.csv"
    summary.to_csv(summary_csv, index=False)
    auc = make_auc(summary)
    auc.to_csv(auc_csv, index=False)
    empirical_snr_table(summary).to_csv(empirical_csv, index=False)
    plot_name = f"{args.name}_recovery_curves"
    plot_summary(summary, args.figures_root / plot_name)
    copy_png(args.figures_root, plot_name)

    print(f"Wrote {summary_csv}")
    print(f"Wrote {auc_csv}")
    print(f"Wrote {empirical_csv}")


if __name__ == "__main__":
    main()
