#!/usr/bin/env python3
"""Plot corrected RDM-score teacher/student recovery curves."""

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
DATA_SUFFIX = "_teacher_student_independent_refit_1k_rdm_score_rdm"
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
    if "relative_snr" not in data:
        data["relative_snr"] = 1.0 / data["noise_mult"].astype(float)
    return data


def ordered(values: pd.Series, preferred: list[str]) -> list[str]:
    present = list(dict.fromkeys(values.dropna().astype(str)))
    return [x for x in preferred if x in present] + [x for x in present if x not in preferred]


def make_raw_and_encoding_avg(data: pd.DataFrame) -> pd.DataFrame:
    raw = data[data["track"] == RAW_TRACK].copy()
    if not raw.empty:
        raw["report_track"] = "raw"
        raw["n_subject_tracks"] = 0
        raw["subject_tracks"] = ""

    enc = data[data["track"].isin(ENCODING_TRACKS)].copy()
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
            by_track = (
                group.groupby("track", as_index=False)
                .agg(
                    recovery_accuracy=("recovery_accuracy", "mean"),
                    error_prob=("error_prob", "mean"),
                    mean_margin=("mean_margin", "mean"),
                    n_units=("n_units", "sum"),
                    n_models=("n_models", "max"),
                    n_equivalence_classes=("n_equivalence_classes", "max")
                    if "n_equivalence_classes" in group
                    else ("n_models", "max"),
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
                "track_type": "encoding_average",
                "recovery_accuracy": float(by_track["recovery_accuracy"].mean()),
                "error_prob": float(by_track["error_prob"].mean()),
                "mean_margin": float(by_track["mean_margin"].mean()),
                "n_units": int(by_track["n_units"].sum()),
                "n_subsets": int(group["n_subsets"].max()) if "n_subsets" in group else np.nan,
                "n_models": int(by_track["n_models"].max()),
                "n_equivalence_classes": int(by_track["n_equivalence_classes"].max()),
                "n_noise_samples": int(by_track["n_noise_samples"].max()),
                "refit_pool_size": int(by_track["refit_pool_size"].max())
                if "refit_pool_size" in by_track
                else np.nan,
                "refit_train_n": int(by_track["refit_train_n"].max())
                if "refit_train_n" in by_track
                else np.nan,
                "refit_val_n": int(by_track["refit_val_n"].max())
                if "refit_val_n" in by_track
                else np.nan,
                "n_subject_tracks": len(present_tracks),
                "subject_tracks": ",".join(present_tracks),
            }
            enc_rows.append(row)
    enc_avg = pd.DataFrame(enc_rows)
    out = pd.concat([raw, enc_avg], ignore_index=True, sort=False)
    if out.empty:
        return out
    out["track"] = pd.Categorical(out["track"], categories=DISPLAY_TRACK_ORDER, ordered=True)
    return out.sort_values(["model_set", "track", "subset_type", "noise_mult"])


def plot_curves(data: pd.DataFrame, metric: str, ylabel: str, out_base: Path) -> None:
    model_sets = ordered(data["model_set"], MODEL_ORDER)
    tracks = ordered(data["track"].astype(str), DISPLAY_TRACK_ORDER)
    n_rows = len(tracks)
    n_cols = len(model_sets)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.35 * n_cols, 2.25 * n_rows),
        sharex=True,
        sharey=metric == "recovery_accuracy",
        squeeze=False,
    )
    for r, track in enumerate(tracks):
        for c, model_set in enumerate(model_sets):
            ax = axes[r, c]
            sub = data[(data["track"] == track) & (data["model_set"] == model_set)]
            if sub.empty:
                ax.set_axis_off()
                continue
            for subset_type, ls in [("random", "--"), ("selected", "-")]:
                g = sub[sub["subset_type"] == subset_type].sort_values("relative_snr")
                if g.empty:
                    continue
                ax.plot(
                    g["relative_snr"],
                    g[metric],
                    ls,
                    color=COLORS[subset_type],
                    linewidth=1.8,
                    label=subset_type.capitalize(),
                )
            if metric == "recovery_accuracy" and "n_models" in sub:
                chance = 1.0 / float(sub["n_models"].max())
                ax.axhline(chance, color="#666666", linestyle=":", linewidth=0.8)
                ax.set_ylim(0, 1.04)
            ax.set_xscale("log")
            ax.grid(color="#dddddd", linewidth=0.65, alpha=0.75)
            ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(model_set.replace("_", " "), fontsize=10)
            if c == 0:
                label = "Raw" if track == "raw" else "Encoding avg"
                ax.set_ylabel(f"{label}\n{ylabel}", fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel("Relative SNR")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("RDM-score teacher/student recovery across noise levels", y=1.035)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_auc(auc: pd.DataFrame, out_base: Path) -> None:
    if auc.empty:
        return
    model_sets = ordered(auc["model_set"], MODEL_ORDER)
    tracks = ordered(auc["track"].astype(str), DISPLAY_TRACK_ORDER)
    n_cols = len(model_sets)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, model_set in zip(axes, model_sets):
        sub = auc[auc["model_set"] == model_set]
        x = np.arange(len(tracks), dtype=float)
        width = 0.34
        for offset, subset_type in [(-width / 2, "random"), (width / 2, "selected")]:
            vals = []
            for track in tracks:
                hit = sub[(sub["track"] == track) & (sub["subset_type"] == subset_type)]
                vals.append(float(hit["error_auc"].iloc[0]) if not hit.empty else np.nan)
            ax.bar(
                x + offset,
                vals,
                width=width,
                color=COLORS[subset_type],
                edgecolor="#222222",
                linewidth=0.5,
                label=subset_type.capitalize(),
            )
        ax.set_title(model_set.replace("_", " "), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["Raw" if track == "raw" else "Encoding avg" for track in tracks],
            rotation=35,
            ha="right",
        )
        ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Error-probability AUC")
    axes[-1].legend(frameon=False)
    fig.suptitle("RDM-score teacher/student recovery AUC", y=1.02)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_auc(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["model_set", "track", "track_type", "subset_type"]
    for key, group in data.groupby(keys, sort=False):
        rows.append(
            dict(zip(keys, key))
            | {
                "error_auc": compute_log_auc(group["noise_mult"], group["error_prob"]),
                "recovery_accuracy_auc": compute_log_auc(
                    group["noise_mult"], group["recovery_accuracy"]
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


def copy_pngs(figures_root: Path, names: list[str]) -> None:
    png_dir = figures_root / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = figures_root / f"{name}.png"
        if src.exists():
            (png_dir / src.name).write_bytes(src.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    parser.add_argument("--figures-root", type=Path, default=FIGURES)
    parser.add_argument("--data-suffix", default=DATA_SUFFIX)
    args = parser.parse_args()

    data = load_results(args.results_root, args.data_suffix)
    if data.empty:
        raise SystemExit("No completed full-grid RDM-score discriminability.csv files found.")
    data = make_raw_and_encoding_avg(data)
    if data.empty:
        raise SystemExit("No raw or encoding-track rows found.")

    args.figures_root.mkdir(parents=True, exist_ok=True)
    summary_csv = args.figures_root / "teacher_student_rdm_score_curves_summary.csv"
    auc_csv = args.figures_root / "teacher_student_rdm_score_auc_summary.csv"
    data.sort_values(["model_set", "track", "subset_type", "noise_mult"]).to_csv(
        summary_csv,
        index=False,
    )
    auc = make_auc(data)
    auc.sort_values(["model_set", "track", "subset_type"]).to_csv(auc_csv, index=False)

    plot_curves(
        data,
        "recovery_accuracy",
        "Recovery accuracy",
        args.figures_root / "teacher_student_rdm_score_recovery_curves",
    )
    plot_curves(
        data,
        "mean_margin",
        "Teacher margin",
        args.figures_root / "teacher_student_rdm_score_margin_curves",
    )
    plot_auc(auc, args.figures_root / "teacher_student_rdm_score_error_auc")
    copy_pngs(
        args.figures_root,
        [
            "teacher_student_rdm_score_recovery_curves",
            "teacher_student_rdm_score_margin_curves",
            "teacher_student_rdm_score_error_auc",
        ],
    )

    print(f"Wrote {summary_csv}")
    print(f"Wrote {auc_csv}")


if __name__ == "__main__":
    main()
