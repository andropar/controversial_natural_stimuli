#!/usr/bin/env python3
"""Plot full visual-cortex parcel mixed-RSA summaries.

The figure intentionally shows both the endpoint distribution and the mean
with uncertainty, following the paper figure-making notes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, StrMethodFormatter
from scipy.stats import ttest_1samp, wilcoxon

PAPER = Path(__file__).resolve().parents[1]
PROJECT = PAPER.parents[1]
FIG_STYLE = PAPER / "figures"
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(FIG_STYLE))

from style_improved import (  # noqa: E402
    DPI,
    FONT,
    OKABE_ITO,
    W_DOUBLE,
    add_panel_label,
    apply_style,
    shade,
)


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGURES = HERE / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

GROUP_ORDER = [
    "Early visual",
    "Extended early/dorsal",
    "Motion",
    "Mid-level/lateral",
    "Medial scene",
    "Ventral object/scene",
]

AGGREGATE_ORDER = ["Full visual", "hlvis"]
ROI_ORDER = AGGREGATE_ORDER + GROUP_ORDER
ROI_DISPLAY = {
    "Full visual": "Full visual",
    "hlvis": "High-level visual",
    "Early visual": "Early visual",
    "Extended early/dorsal": "Extended early/dorsal",
    "Motion": "Motion",
    "Mid-level/lateral": "Mid-level/lateral",
    "Medial scene": "Medial scene",
    "Ventral object/scene": "Ventral object/scene",
}
ROI_DISPLAY_SHORT = {
    "Full visual": "Full\nvisual",
    "hlvis": "High-level\nvisual",
    "Early visual": "Early\nvisual",
    "Extended early/dorsal": "Ext. early\n/dorsal",
    "Motion": "Motion",
    "Mid-level/lateral": "Mid-level\n/lateral",
    "Medial scene": "Medial\nscene",
    "Ventral object/scene": "Ventral\nobject/scene",
}

ROI_KIND = {
    "Full visual": "aggregate",
    "hlvis": "aggregate",
    "Early visual": "group",
    "Extended early/dorsal": "group",
    "Motion": "group",
    "Mid-level/lateral": "group",
    "Medial scene": "group",
    "Ventral object/scene": "group",
}

METRICS = [
    {
        "column": "delta",
        "title": "Alignment delta",
        "xlabel": "Controversial - baseline mixed RSA",
        "reference": 0.0,
        "formatter": StrMethodFormatter("{x:.2f}"),
    },
    {
        "column": "relative_delta",
        "title": "Relative delta",
        "xlabel": "Delta as fraction of baseline",
        "reference": 0.0,
        "formatter": FuncFormatter(lambda x, _pos: f"{x:.0%}"),
    },
    {
        "column": "log2_spread_ratio",
        "title": "Model-spread change",
        "xlabel": "log2(controversial / baseline)",
        "reference": 0.0,
        "formatter": StrMethodFormatter("{x:.1f}"),
    },
]


def sem(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) <= 1:
        return np.nan
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def summarize_for_plot(endpoints: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for roi_label, grp in endpoints.groupby("roi_label", sort=False):
        row = {
            "roi_label": roi_label,
            "roi_level": grp["roi_level"].iloc[0],
            "n_endpoints": int(grp[["subject", "model_set"]].drop_duplicates().shape[0]),
            "mean_n_voxels": float(grp["n_voxels"].mean()),
            "sem_n_voxels": sem(grp["n_voxels"]),
        }
        for metric in METRICS:
            col = metric["column"]
            row[f"{col}_mean"] = float(grp[col].mean())
            row[f"{col}_sem"] = sem(grp[col])
            row[f"{col}_ci95"] = 1.96 * row[f"{col}_sem"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["plot_order"] = summary["roi_label"].map({r: i for i, r in enumerate(ROI_ORDER)})
    return summary.sort_values("plot_order").drop(columns="plot_order")


def holm_adjust(p_values: pd.Series) -> pd.Series:
    p = p_values.to_numpy(dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return pd.Series(out, index=p_values.index)

    finite_idx = np.flatnonzero(finite)
    order = finite_idx[np.argsort(p[finite])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        adjusted = min((m - rank) * p[idx], 1.0)
        running = max(running, adjusted)
        out[idx] = running
    return pd.Series(out, index=p_values.index)


def paired_comparisons(endpoints: pd.DataFrame) -> pd.DataFrame:
    """Paired subject x model-set ROI comparisons for plotted metrics."""
    rows = []
    index = ["subject", "model_set"]
    references = ["Early visual", "hlvis"]
    for metric in METRICS:
        col = metric["column"]
        wide = endpoints.pivot_table(index=index, columns="roi_label", values=col, aggfunc="mean")
        for reference in references:
            if reference not in wide:
                continue
            for roi_label in ROI_ORDER:
                if roi_label == reference or roi_label not in wide:
                    continue
                paired = wide[[roi_label, reference]].dropna()
                diff = paired[roi_label] - paired[reference]
                diff = diff[np.isfinite(diff)]
                if len(diff) < 2:
                    continue
                t_res = ttest_1samp(diff, popmean=0.0, nan_policy="omit")
                try:
                    w_res = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                    wilcoxon_p = float(w_res.pvalue)
                except ValueError:
                    wilcoxon_p = np.nan
                diff_sem = float(diff.std(ddof=1) / np.sqrt(len(diff)))
                rows.append(
                    {
                        "metric": col,
                        "roi_label": roi_label,
                        "reference_roi_label": reference,
                        "roi_display": ROI_DISPLAY[roi_label],
                        "reference_roi_display": ROI_DISPLAY[reference],
                        "n_paired_endpoints": int(len(diff)),
                        "mean_difference": float(diff.mean()),
                        "sem_difference": diff_sem,
                        "ci95_difference": 1.96 * diff_sem,
                        "median_difference": float(diff.median()),
                        "t": float(t_res.statistic),
                        "p": float(t_res.pvalue),
                        "wilcoxon_p": wilcoxon_p,
                    }
                )

    comparisons = pd.DataFrame(rows)
    if comparisons.empty:
        return comparisons
    comparisons["p_holm"] = np.nan
    for (_metric, _reference), idx in comparisons.groupby(["metric", "reference_roi_label"]).groups.items():
        comparisons.loc[idx, "p_holm"] = holm_adjust(comparisons.loc[idx, "p"]).to_numpy()
    comparisons["roi_order"] = comparisons["roi_label"].map({r: i for i, r in enumerate(ROI_ORDER)})
    comparisons["reference_order"] = comparisons["reference_roi_label"].map({r: i for i, r in enumerate(ROI_ORDER)})
    return comparisons.sort_values(["metric", "reference_order", "roi_order"]).drop(
        columns=["roi_order", "reference_order"]
    )


def prepare_endpoints(path: Path) -> pd.DataFrame:
    endpoints = pd.read_csv(path)
    keep = endpoints["roi_label"].isin(ROI_ORDER)
    endpoints = endpoints[keep].copy()
    endpoints["roi_kind"] = endpoints["roi_label"].map(ROI_KIND)
    endpoints["plot_order"] = endpoints["roi_label"].map({r: i for i, r in enumerate(ROI_ORDER)})
    endpoints["log2_spread_ratio"] = np.log2(endpoints["spread_ratio"])
    endpoints.loc[~np.isfinite(endpoints["log2_spread_ratio"]), "log2_spread_ratio"] = np.nan
    endpoints = endpoints.sort_values(["plot_order", "subject", "model_set"])
    missing = [r for r in ROI_ORDER if r not in set(endpoints["roi_label"])]
    if missing:
        raise ValueError(f"missing ROI rows in endpoint summary: {missing}")
    return endpoints


def symmetric_limits(values: pd.Series, reference: float, pad: float = 0.10) -> tuple[float, float]:
    values = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    values = np.concatenate([values, np.asarray([reference], dtype=float)])
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if lo < reference < hi:
        span = max(reference - lo, hi - reference)
        lo, hi = reference - span, reference + span
    width = hi - lo
    if not np.isfinite(width) or width <= 0:
        width = 1.0
    return lo - pad * width, hi + pad * width


def plot_metric(ax, endpoints: pd.DataFrame, summary: pd.DataFrame, metric: dict, x_positions: dict[str, float]) -> None:
    col = metric["column"]
    mean_col = f"{col}_mean"
    ci_col = f"{col}_ci95"

    rng = np.random.default_rng(1187)
    raw_color = "#A8A8A8"
    mean_color = OKABE_ITO["bluish_green"] if col == "log2_spread_ratio" else OKABE_ITO["blue"]
    aggregate_color = shade(mean_color, -0.20)

    for roi_label in ROI_ORDER:
        roi_rows = endpoints[endpoints["roi_label"] == roi_label]
        x = x_positions[roi_label]
        jitter = rng.uniform(-0.16, 0.16, size=len(roi_rows))
        ax.scatter(
            x + jitter,
            roi_rows[col],
            s=10,
            c=raw_color,
            alpha=0.45,
            linewidths=0,
            zorder=2,
        )

    for _, row in summary.iterrows():
        roi_label = row["roi_label"]
        x = x_positions[roi_label]
        y = row[mean_col]
        ci = row[ci_col]
        color = aggregate_color if ROI_KIND[roi_label] == "aggregate" else mean_color
        ax.errorbar(
            x,
            y,
            yerr=ci,
            fmt="o",
            markersize=4.5,
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=0,
            zorder=4,
        )

    ax.axhline(metric["reference"], color="#3A3A3A", linewidth=0.8, linestyle=":", zorder=1)
    ax.axvline(x_positions["Early visual"] - 0.5, color="#D0D0D0", linewidth=0.7, zorder=1)
    ax.set_title(metric["title"], fontsize=FONT["title"], pad=5)
    ax.set_ylabel(metric["xlabel"], fontsize=FONT["axis_label"])
    ax.set_xlim(-0.6, len(ROI_ORDER) - 0.4)
    ax.set_xticks([x_positions[r] for r in ROI_ORDER])
    ax.set_xticklabels([ROI_DISPLAY_SHORT[r] for r in ROI_ORDER], rotation=35, ha="right")
    ax.set_ylim(*symmetric_limits(endpoints[col], metric["reference"]))
    if isinstance(metric["formatter"], FuncFormatter):
        ax.yaxis.set_major_formatter(metric["formatter"])
    else:
        ax.yaxis.set_major_formatter(metric["formatter"])
    ax.tick_params(axis="both", labelsize=FONT["tick"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.45)
    ax.grid(axis="x", visible=False)


def make_figure(endpoints: pd.DataFrame, summary: pd.DataFrame, output_stem: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(W_DOUBLE, 3.75),
        sharey=True,
        constrained_layout=False,
    )
    x_positions = {roi: i for i, roi in enumerate(ROI_ORDER)}

    for ax, metric, label in zip(axes, METRICS, ["a", "b", "c"]):
        plot_metric(ax, endpoints, summary, metric, x_positions)
        add_panel_label(ax, label, x=-0.12, y=1.05)

    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.31, top=0.88, wspace=0.30)
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=DATA / "visual_parcel_endpoint_summary.csv",
        help="Endpoint summary produced by 03_visual_parcel_mixed_rsa.py.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=FIGURES / "visual_parcel_roi_summary",
        help="Output path without extension.",
    )
    args = parser.parse_args()

    endpoints = prepare_endpoints(args.summary)
    summary = summarize_for_plot(endpoints)
    comparisons = paired_comparisons(endpoints)
    group_summary_path = DATA / "visual_group_summary.csv"
    comparisons_path = DATA / "visual_group_comparisons.csv"
    summary.to_csv(group_summary_path, index=False)
    comparisons.to_csv(comparisons_path, index=False)
    make_figure(endpoints, summary, args.output_stem)

    print(f"wrote {group_summary_path}")
    print(f"wrote {comparisons_path}")
    print(f"wrote {args.output_stem.with_suffix('.pdf')}")
    print(f"wrote {args.output_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
