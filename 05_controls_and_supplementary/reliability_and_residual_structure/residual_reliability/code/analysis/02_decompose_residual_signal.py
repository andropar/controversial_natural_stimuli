#!/usr/bin/env python3
"""
Decompose the residual-reliability result into interpretable pieces.

This script starts from results/residual_rsa.csv and asks four related questions:

1. How close is the cross-validated model ensemble to the correlation ceiling?
2. How much within-subject split-half reliability remains after residualization?
3. How much leave-one-subject-out shared structure remains after residualization?
4. Which of those quantities changes for diagnostic stimuli relative to baseline?

The key distinction is that absolute LOSO residual can stay flat even when the
diagnostic set leaves a larger *fraction* of the available shared signal
unexplained. The paired contrasts make that distinction explicit.

Outputs:
    results/residual_decomposition_summary.csv
    results/residual_decomposition_contrasts.csv
    figures/residual_decomposition.{pdf,png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

STAGE = Path(__file__).resolve().parents[2]
SHARE_ROOT = STAGE.parents[2]
sys.path.insert(0, str(SHARE_ROOT / "src"))

import matplotlib.pyplot as plt

from cstims.paper.style_improved import apply_style, DPI, FONT, W_DOUBLE

apply_style()

DATA_CSV = STAGE / "results" / "residual_rsa.csv"
OUT_DATA = STAGE / "results"
OUT_FIG = STAGE / "figures"

GROUPS = ["vicco", "all_models", "sota", "architecture", "dataset", "training_objective"]
GROUP_LABELS = {
    "vicco": "Baseline",
    "all_models": "All models",
    "sota": "SOTA",
    "architecture": "Arch.",
    "dataset": "Dataset",
    "training_objective": "Train. obj.",
}

COLOR_BASELINE = "#2980B9"
COLOR_CSTIM = "#D64541"
COLOR_DARK = "#333333"

METRIC_LABELS = {
    "ensemble_gap_to_correlation_ceiling": "Ensemble gap to ceiling",
    "ensemble_fraction_ceiling": "Ensemble / ceiling",
    "fullfit_gap_to_correlation_ceiling": "Full-fit ensemble gap to ceiling",
    "fullfit_fraction_ceiling": "Full-fit ensemble / ceiling",
    "within_residual_fraction": "Within-subject residual fraction",
    "loso_residual_fraction": "LOSO residual fraction",
    "r_loso_residual": "Absolute LOSO residual",
    "residual_split_half_reliability": "Residual split-half reliability",
    "r_loso_brain": "Total LOSO brain RSA",
    "r_ensemble_cv": "Ensemble RSA",
    "r_ensemble_full_fit": "Full-fit ensemble RSA",
    "correlation_ceiling": "Correlation ceiling",
}

RATIO_METRICS = {
    "ensemble_fraction_ceiling",
    "fullfit_fraction_ceiling",
    "within_residual_fraction",
    "within_model_removed_fraction",
    "loso_residual_fraction",
    "loso_model_removed_fraction",
}


def _sem(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def _ci95(x: pd.Series) -> tuple[float, float]:
    x = x.dropna()
    if len(x) < 2:
        return (np.nan, np.nan)
    mean = x.mean()
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(mean - half), float(mean + half)


def _sign_test_pvalue(diffs: pd.Series) -> float:
    diffs = diffs.dropna()
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return np.nan
    n_pos = int((diffs > 0).sum())
    return float(stats.binomtest(n_pos, n=len(diffs), p=0.5).pvalue)


def _wilcoxon_pvalue(diffs: pd.Series) -> float:
    diffs = diffs.dropna()
    diffs = diffs[diffs != 0]
    if len(diffs) < 2:
        return np.nan
    try:
        return float(stats.wilcoxon(diffs, zero_method="wilcox", method="auto").pvalue)
    except ValueError:
        return np.nan


def load_subject_level() -> pd.DataFrame:
    """Average baseline bootstraps within subject and add derived metrics."""
    df = pd.read_csv(DATA_CSV)
    sample_counts = (
        df.groupby(["rsa_type", "stimulus_group", "stimulus_type", "subject"], as_index=False)
        ["bootstrap_idx"]
        .nunique()
        .rename(columns={"bootstrap_idx": "n_subsamples_averaged"})
    )
    subj = (
        df.groupby(["rsa_type", "stimulus_group", "stimulus_type", "subject"], as_index=False)
        .mean(numeric_only=True)
    )
    subj = subj.merge(
        sample_counts,
        on=["rsa_type", "stimulus_group", "stimulus_type", "subject"],
        how="left",
    )
    subj["baseline_sampling"] = np.where(
        subj["stimulus_group"] == "vicco",
        "n_matched_without_replacement",
        "none",
    )

    subj["ensemble_fraction_ceiling"] = (
        subj["r_ensemble_cv"] / subj["correlation_ceiling"]
    )
    subj["fullfit_gap_to_correlation_ceiling"] = (
        subj["correlation_ceiling"] - subj["r_ensemble_full_fit"]
    )
    subj["fullfit_fraction_ceiling"] = (
        subj["r_ensemble_full_fit"] / subj["correlation_ceiling"]
    )
    subj["within_residual_fraction"] = (
        subj["residual_split_half_reliability"] / subj["noise_ceiling_reliability"]
    )
    subj["within_model_removed"] = (
        subj["noise_ceiling_reliability"] - subj["residual_split_half_reliability"]
    )
    subj["within_model_removed_fraction"] = 1 - subj["within_residual_fraction"]
    subj["loso_model_removed"] = subj["r_loso_brain"] - subj["r_loso_residual"]
    subj["loso_model_removed_fraction"] = 1 - subj["loso_residual_fraction"]

    # Guard against tiny negative/over-one ratios caused by noisy denominators.
    for col in [
        "ensemble_fraction_ceiling",
        "fullfit_fraction_ceiling",
        "within_residual_fraction",
        "within_model_removed_fraction",
        "loso_residual_fraction",
        "loso_model_removed_fraction",
    ]:
        subj[col] = subj[col].replace([np.inf, -np.inf], np.nan)

    return subj


def make_summary(subj: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "r_single_best",
        "r_ensemble_cv",
        "r_ensemble_full_fit",
        "correlation_ceiling",
        "ensemble_gap_to_correlation_ceiling",
        "ensemble_fraction_ceiling",
        "fullfit_gap_to_correlation_ceiling",
        "fullfit_fraction_ceiling",
        "noise_ceiling_reliability",
        "residual_split_half_reliability",
        "within_residual_fraction",
        "within_model_removed",
        "within_model_removed_fraction",
        "r_loso_brain",
        "r_loso_residual",
        "loso_residual_fraction",
        "loso_model_removed",
        "loso_model_removed_fraction",
    ]

    rows = []
    for (rsa_type, group), g in subj.groupby(["rsa_type", "stimulus_group"]):
        row = {
            "rsa_type": rsa_type,
            "stimulus_group": group,
            "inference_unit": "subject",
            "n_subjects": g["subject"].nunique(),
            "mean_n_subsamples_averaged": float(g["n_subsamples_averaged"].mean()),
        }
        for metric in metrics:
            lo, hi = _ci95(g[metric])
            row[f"{metric}_mean"] = float(g[metric].mean())
            row[f"{metric}_sem"] = _sem(g[metric])
            row[f"{metric}_ci95_lo"] = lo
            row[f"{metric}_ci95_hi"] = hi
            if metric in RATIO_METRICS:
                row[f"{metric}_n_below_0"] = int((g[metric] < 0).sum())
                row[f"{metric}_n_above_1"] = int((g[metric] > 1).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["rsa_type", "stimulus_group"])


def make_contrasts(subj: pd.DataFrame) -> pd.DataFrame:
    metrics = list(METRIC_LABELS)
    rows = []
    for rsa_type in sorted(subj["rsa_type"].unique()):
        base = subj[(subj.rsa_type == rsa_type) & (subj.stimulus_group == "vicco")]
        base = base.set_index("subject")
        for group in GROUPS:
            if group == "vicco":
                continue
            cur = subj[(subj.rsa_type == rsa_type) & (subj.stimulus_group == group)]
            cur = cur.set_index("subject")
            shared = sorted(set(base.index) & set(cur.index))
            if len(shared) < 2:
                continue
            for metric in metrics:
                diffs = cur.loc[shared, metric] - base.loc[shared, metric]
                diffs = diffs.dropna()
                if len(diffs) < 2:
                    continue
                ci_lo, ci_hi = _ci95(diffs)
                t_res = stats.ttest_1samp(diffs, 0.0, nan_policy="omit")
                rows.append({
                    "rsa_type": rsa_type,
                    "stimulus_group": group,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "inference_unit": "subject",
                    "n_subjects": len(diffs),
                    "baseline_mean": float(base.loc[diffs.index, metric].mean()),
                    "diagnostic_mean": float(cur.loc[diffs.index, metric].mean()),
                    "mean_diff_diagnostic_minus_baseline": float(diffs.mean()),
                    "sem_diff": _sem(diffs),
                    "ci95_lo": ci_lo,
                    "ci95_hi": ci_hi,
                    "t": float(t_res.statistic),
                    "p_ttest": float(t_res.pvalue),
                    "p_wilcoxon": _wilcoxon_pvalue(diffs),
                    "p_sign_two_sided": _sign_test_pvalue(diffs),
                    "n_positive": int((diffs > 0).sum()),
                    "n_negative": int((diffs < 0).sum()),
                    "ratio_metric_unbounded": bool(metric in RATIO_METRICS),
                })
    return pd.DataFrame(rows).sort_values(["rsa_type", "stimulus_group", "metric"])


def _bar_metric(
    ax,
    subj: pd.DataFrame,
    rsa_type: str,
    metric: str,
    ylabel: str,
    title: str,
    ylim: tuple[float, float] | None = None,
    overlay_metric: str | None = None,
    overlay_label: str | None = None,
) -> None:
    rows = subj[subj.rsa_type == rsa_type]
    xs = np.arange(len(GROUPS))
    means = []
    sems = []
    colors = []
    for group in GROUPS:
        g = rows[rows.stimulus_group == group]
        means.append(g[metric].mean())
        sems.append(_sem(g[metric]))
        colors.append(COLOR_BASELINE if group == "vicco" else COLOR_CSTIM)

    ax.bar(
        xs, means, yerr=sems, color=colors, alpha=0.62, edgecolor=colors,
        linewidth=0.8, capsize=2, error_kw=dict(ecolor="0.25", elinewidth=0.7),
    )
    for x, group in zip(xs, GROUPS):
        vals = rows.loc[rows.stimulus_group == group, metric].to_numpy()
        jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) else []
        ax.scatter(np.full(len(vals), x) + jitter, vals, s=16, color=COLOR_DARK,
                   alpha=0.62, linewidth=0, zorder=5)
        if overlay_metric is not None:
            overlay_vals = rows.loc[rows.stimulus_group == group, overlay_metric].dropna()
            if len(overlay_vals):
                ax.hlines(
                    overlay_vals.mean(), x - 0.30, x + 0.30,
                    colors="black", linewidth=1.2, zorder=6,
                )
    ax.set_xticks(xs)
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUPS], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=FONT["title"])
    if overlay_label:
        ax.text(
            0.99, 0.96, overlay_label, ha="right", va="top",
            transform=ax.transAxes, fontsize=FONT["small"] - 1, color="black",
        )
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.20, linewidth=0.5)


def _paired_panel(ax, subj: pd.DataFrame, metric: str, title: str, ylabel: str,
                  ylim: tuple[float, float] | None = None) -> None:
    rows = subj[(subj.rsa_type == "mixed") & (subj.stimulus_group.isin(["vicco", "all_models"]))]
    wide = rows.pivot(index="subject", columns="stimulus_group", values=metric).dropna()
    x0, x1 = 0, 1
    for _, row in wide.iterrows():
        ax.plot([x0, x1], [row["vicco"], row["all_models"]], color="#777777",
                linewidth=0.8, alpha=0.65, zorder=1)
        ax.scatter([x0, x1], [row["vicco"], row["all_models"]],
                   color=[COLOR_BASELINE, COLOR_CSTIM], s=22, zorder=3)
    means = wide[["vicco", "all_models"]].mean()
    ax.plot([x0, x1], [means["vicco"], means["all_models"]], color="black",
            linewidth=2.0, zorder=4)
    diff = wide["all_models"] - wide["vicco"]
    ax.text(0.50, 0.96, f"Delta={diff.mean():+.2f}",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=FONT["small"], color=COLOR_DARK)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["Baseline", "All-model\ndiagnostic"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=FONT["title"])
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.20, linewidth=0.5)


def make_figure(subj: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE, 6.8))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.13,
                        hspace=0.48, wspace=0.30)

    _bar_metric(
        axes[0, 0], subj, "mixed", "ensemble_gap_to_correlation_ceiling",
        "Correlation units", "A. Mixed ensemble gap", ylim=(0, 0.35),
        overlay_metric="fullfit_gap_to_correlation_ceiling",
        overlay_label="black tick: full-fit upper bound",
    )
    _bar_metric(
        axes[0, 1], subj, "mixed", "within_residual_fraction",
        "Fraction of split-half reliability", "B. Within-subject residual fraction",
        ylim=(0, 1.05)
    )
    _bar_metric(
        axes[1, 0], subj, "mixed", "loso_residual_fraction",
        "Fraction of LOSO brain RSA", "C. Shared residual fraction", ylim=(0, 1.05)
    )
    _paired_panel(
        axes[1, 1], subj, "within_residual_fraction",
        "D. Paired all-model contrast", "Residual fraction", ylim=(0, 1.05)
    )

    fig.suptitle(
        "Subject-level residual diagnostics after model-ensemble removal",
        fontsize=FONT["title"] + 1,
    )
    return fig


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    subj = load_subject_level()
    summary = make_summary(subj)
    contrasts = make_contrasts(subj)

    summary_path = OUT_DATA / "residual_decomposition_summary.csv"
    contrast_path = OUT_DATA / "residual_decomposition_contrasts.csv"
    summary.to_csv(summary_path, index=False)
    contrasts.to_csv(contrast_path, index=False)
    print(f"Saved {summary_path}")
    print(f"Saved {contrast_path}")

    fig = make_figure(subj)
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"residual_decomposition.{ext}"
        fig.savefig(out, dpi=DPI if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)

    key = contrasts[
        (contrasts.rsa_type == "mixed")
        & (contrasts.stimulus_group == "all_models")
        & (contrasts.metric.isin([
            "ensemble_gap_to_correlation_ceiling",
            "fullfit_gap_to_correlation_ceiling",
            "within_residual_fraction",
            "loso_residual_fraction",
            "r_loso_residual",
        ]))
    ][[
        "metric_label", "baseline_mean", "diagnostic_mean",
        "mean_diff_diagnostic_minus_baseline", "ci95_lo", "ci95_hi",
        "p_ttest", "p_wilcoxon", "p_sign_two_sided",
    ]]
    print("\nMixed all-model diagnostic vs baseline:")
    print(key.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
