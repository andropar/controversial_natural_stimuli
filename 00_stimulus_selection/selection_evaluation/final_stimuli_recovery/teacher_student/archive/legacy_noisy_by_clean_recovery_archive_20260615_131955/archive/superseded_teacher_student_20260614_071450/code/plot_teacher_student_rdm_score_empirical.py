#!/usr/bin/env python3
"""Plot corrected RDM-score teacher/student empirical-SNR results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
RESULTS = SCRIPT.parents[1] / "results"
FIGURES = SCRIPT.parents[1] / "figures"
MODEL_ORDER = ["all_models", "sota", "training_objective", "architecture", "dataset"]
TRACK_ORDER = ["raw", "sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
SUBSET_ORDER = ["random", "selected"]
COLORS = {"random": "#E45756", "selected": "#4C78A8"}


def load_results(results_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(results_root.glob("*_teacher_student_independent_refit_1k_rdm_score_rdm*empirical/discriminability.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["source_dir"] = path.parent.name
        df["source_mtime"] = path.stat().st_mtime
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    data = pd.concat(rows, ignore_index=True)
    data = data[np.isclose(data["noise_mult"].astype(float), 1.0)].copy()
    key_cols = ["model_set", "track", "track_type", "subset_type", "noise_mult", "eval_noise_mode"]
    data = (
        data.sort_values(["source_mtime", "source_dir"])
        .drop_duplicates(key_cols, keep="last")
        .reset_index(drop=True)
    )
    return data


def ordered(values: pd.Series, preferred: list[str]) -> list[str]:
    present = list(dict.fromkeys(values.dropna().astype(str)))
    return [x for x in preferred if x in present] + [x for x in present if x not in preferred]


def plot_metric(data: pd.DataFrame, metric: str, ylabel: str, out_base: Path) -> None:
    model_sets = ordered(data["model_set"], MODEL_ORDER)
    n_cols = max(len(model_sets), 1)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.1 * n_cols, 4.2), sharey=metric == "recovery_accuracy")
    axes = np.atleast_1d(axes)

    for ax, model_set in zip(axes, model_sets):
        sub = data[data["model_set"] == model_set].copy()
        tracks = ordered(sub["track"], TRACK_ORDER)
        x = np.arange(len(tracks), dtype=float)
        width = 0.34
        for offset, subset_type in [(-width / 2, "random"), (width / 2, "selected")]:
            vals = []
            for track in tracks:
                hit = sub[(sub["track"] == track) & (sub["subset_type"] == subset_type)]
                vals.append(float(hit[metric].iloc[0]) if not hit.empty else np.nan)
            ax.bar(
                x + offset,
                vals,
                width=width,
                label=subset_type.capitalize(),
                color=COLORS[subset_type],
                alpha=0.9,
                edgecolor="#222222",
                linewidth=0.5,
            )
        ax.set_title(model_set.replace("_", " "))
        ax.set_xticks(x)
        ax.set_xticklabels(tracks, rotation=35, ha="right")
        ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        if metric == "recovery_accuracy":
            ax.set_ylim(0, 1.04)
            ax.axhline(1.0 / max(float(sub["n_models"].max()), 1.0), color="#555555", linestyle=":", linewidth=1)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(frameon=False, loc="lower right" if metric == "recovery_accuracy" else "best")
    fig.suptitle("RDM-score teacher/student recovery at empirical SNR", y=1.02)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    parser.add_argument("--figures-root", type=Path, default=FIGURES)
    args = parser.parse_args()

    data = load_results(args.results_root)
    if data.empty:
        raise SystemExit("No completed RDM-score empirical discriminability.csv files found.")

    out_csv = args.figures_root / "teacher_student_rdm_score_empirical_summary.csv"
    args.figures_root.mkdir(parents=True, exist_ok=True)
    (args.figures_root / "png").mkdir(parents=True, exist_ok=True)
    data.sort_values(["model_set", "track", "subset_type"]).to_csv(out_csv, index=False)

    recovery_base = args.figures_root / "teacher_student_rdm_score_empirical_recovery"
    margin_base = args.figures_root / "teacher_student_rdm_score_empirical_margin"
    plot_metric(data, "recovery_accuracy", "Recovery accuracy", recovery_base)
    plot_metric(data, "mean_margin", "Teacher margin vs nearest competitor", margin_base)

    for base in [recovery_base, margin_base]:
        for ext in [".png", ".pdf"]:
            src = base.with_suffix(ext)
            dst_dir = args.figures_root / "png" if ext == ".png" else args.figures_root
            dst = dst_dir / src.name
            if src.resolve() != dst.resolve():
                dst.write_bytes(src.read_bytes())

    print(f"Wrote {out_csv}")
    print(f"Wrote {recovery_base.with_suffix('.png')}")
    print(f"Wrote {margin_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
