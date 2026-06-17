#!/usr/bin/env python3
"""ROI baseline-alignment levels against controversial-minus-baseline deltas."""
from __future__ import annotations

import sys
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
PROJECT = PAPER.parents[1]
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(PROJECT))

from cstims.paper.style_improved import DPI, FONT, OKABE_ITO, W_DOUBLE, add_panel_label, apply_style  # noqa: E402


HERE = Path(__file__).resolve().parent
DATA = HERE / "results"
FIGURES = HERE / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

ROI_ORDER = [
    "Full visual",
    "hlvis",
    "Early visual",
    "Extended early/dorsal",
    "Motion",
    "Mid-level/lateral",
    "Medial scene",
    "Ventral object/scene",
]
ROI_DISPLAY = {
    "Full visual": "Full visual",
    "hlvis": "High-level visual",
    "Early visual": "Early visual",
    "Extended early/dorsal": "Ext. early/dorsal",
    "Motion": "Motion",
    "Mid-level/lateral": "Mid-level/lateral",
    "Medial scene": "Medial scene",
    "Ventral object/scene": "Ventral object/scene",
}
ROI_COLORS = {
    "Full visual": "#666666",
    "hlvis": OKABE_ITO["reddish_purple"],
    "Early visual": OKABE_ITO["sky_blue"],
    "Extended early/dorsal": OKABE_ITO["blue"],
    "Motion": OKABE_ITO["orange"],
    "Mid-level/lateral": OKABE_ITO["bluish_green"],
    "Medial scene": OKABE_ITO["yellow"],
    "Ventral object/scene": OKABE_ITO["vermillion"],
}
def sem(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) <= 1:
        return np.nan
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def load_endpoints() -> pd.DataFrame:
    df = pd.read_csv(DATA / "visual_parcel_endpoint_summary.csv")
    df = df[df["roi_label"].isin(ROI_ORDER)].copy()
    df["roi_order"] = df["roi_label"].map({r: i for i, r in enumerate(ROI_ORDER)})
    return df.sort_values(["roi_order", "subject", "model_set"])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for roi_label, grp in df.groupby("roi_label", sort=False):
        rows.append(
            {
                "roi_label": roi_label,
                "roi_display": ROI_DISPLAY[roi_label],
                "n_endpoints": int(grp[["subject", "model_set"]].drop_duplicates().shape[0]),
                "baseline_mean": float(grp["score_baseline"].mean()),
                "baseline_sem": sem(grp["score_baseline"]),
                "cstim_mean": float(grp["score_cstim"].mean()),
                "cstim_sem": sem(grp["score_cstim"]),
                "delta_mean": float(grp["delta"].mean()),
                "delta_sem": sem(grp["delta"]),
                "relative_delta_mean": float(grp["relative_delta"].mean()),
                "n_voxels_mean": float(grp["n_voxels"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["baseline_ci95"] = 1.96 * out["baseline_sem"]
    out["cstim_ci95"] = 1.96 * out["cstim_sem"]
    out["delta_ci95"] = 1.96 * out["delta_sem"]
    out["roi_order"] = out["roi_label"].map({r: i for i, r in enumerate(ROI_ORDER)})
    out = out.sort_values("roi_order").drop(columns="roi_order")
    out.to_csv(DATA / "roi_baseline_delta_summary.csv", index=False)
    return out


def plot_scatter(ax, endpoints: pd.DataFrame, summary: pd.DataFrame) -> None:
    rng = np.random.default_rng(4001)
    for roi_label in ROI_ORDER:
        color = ROI_COLORS[roi_label]
        sub = endpoints[endpoints["roi_label"] == roi_label]
        ax.scatter(
            sub["score_baseline"] + rng.normal(0, 0.0025, len(sub)),
            sub["delta"] + rng.normal(0, 0.0025, len(sub)),
            s=11,
            color=color,
            alpha=0.22,
            linewidths=0,
            zorder=2,
        )
        row = summary[summary["roi_label"] == roi_label].iloc[0]
        ax.errorbar(
            row["baseline_mean"],
            row["delta_mean"],
            xerr=row["baseline_ci95"],
            yerr=row["delta_ci95"],
            fmt="o",
            markersize=5.5,
            color=color,
            ecolor=color,
            elinewidth=0.9,
            capsize=0,
            zorder=4,
        )
    ax.axhline(0, color="#333333", lw=0.8, ls=":", zorder=1)
    ax.set_xlabel("Baseline mixed RSA", fontsize=FONT["axis_label"])
    ax.set_ylabel("CSTIMS - baseline", fontsize=FONT["axis_label"])
    ax.set_title("Baseline alignment vs. delta", fontsize=FONT["title"], pad=5)
    ax.grid(True, color="#E6E6E6", linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=FONT["tick"])
    add_panel_label(ax, "a", x=-0.12, y=1.05)


def plot_slope(ax, summary: pd.DataFrame) -> None:
    y = np.arange(len(ROI_ORDER))
    for i, roi_label in enumerate(ROI_ORDER):
        row = summary[summary["roi_label"] == roi_label].iloc[0]
        color = ROI_COLORS[roi_label]
        ax.plot(
            [row["baseline_mean"], row["cstim_mean"]],
            [i, i],
            color=color,
            lw=1.2,
            alpha=0.85,
            zorder=2,
        )
        ax.errorbar(
            row["baseline_mean"],
            i - 0.055,
            xerr=row["baseline_ci95"],
            fmt="o",
            markersize=4.5,
            color="#555555",
            ecolor="#555555",
            elinewidth=0.8,
            capsize=0,
            zorder=4,
        )
        ax.errorbar(
            row["cstim_mean"],
            i + 0.055,
            xerr=row["cstim_ci95"],
            fmt="o",
            markersize=4.5,
            color=color,
            ecolor=color,
            elinewidth=0.8,
            capsize=0,
            zorder=4,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([ROI_DISPLAY[r] for r in ROI_ORDER], fontsize=FONT["tick"] - 1)
    ax.invert_yaxis()
    ax.set_xlabel("Mixed RSA", fontsize=FONT["axis_label"])
    ax.set_title("Absolute levels", fontsize=FONT["title"], pad=5)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.45)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=FONT["tick"])
    add_panel_label(ax, "b", x=-0.18, y=1.05)


def main() -> None:
    apply_style()
    endpoints = load_endpoints()
    summary = summarize(endpoints)
    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.55), gridspec_kw={"width_ratios": [1.15, 1.0]})
    plot_scatter(axes[0], endpoints, summary)
    plot_slope(axes[1], summary)
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.16, top=0.88, wspace=0.36)
    for ext in ("pdf", "png"):
        out = FIGURES / f"roi_baseline_vs_delta.{ext}"
        fig.savefig(out, dpi=DPI)
        print(f"Saved {out}")
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
