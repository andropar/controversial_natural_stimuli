#!/usr/bin/env python3
"""Generate compact summary plots for the full-data rerun outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


THIS = Path(__file__).resolve()
RERUN_ROOT = THIS.parents[1]
SHARE_ROOT = THIS.parents[3]
HELPERS = SHARE_ROOT / "src"
sys.path.insert(0, str(HELPERS))

from cstims import constants, paths


DATA_DIR = RERUN_ROOT / "results"
FIG_DIR = RERUN_ROOT / "figures"
ROI_ORDER = [
    "EVC",
    "ventral",
    "lateral",
    "dorsal",
    "general",
    "EBA",
    "FFA",
    "PPA",
    "LOTC",
    "floc_all",
    "ventral_lateral_floc",
]


def savefig(fig, stem: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def model_order(df: pd.DataFrame):
    all_models = list(constants.MODEL_SETS["all_models"])
    present = set(df["model"])
    return [m for m in all_models if m in present] + sorted(present - set(all_models))


def plot_heatmap(summary: pd.DataFrame, value_col: str, stem: str, title: str, out_dir: Path):
    sub = summary[
        summary["model_set"].eq("all_models")
        & summary["stimulus_type"].eq("controversial")
    ].copy()
    if sub.empty:
        return
    models = model_order(sub)
    rois = [r for r in ROI_ORDER if r in set(sub["roi"])]
    mat = (
        sub.pivot_table(index="model", columns="roi", values=value_col, aggfunc="mean")
        .reindex(index=models, columns=rois)
    )
    labels = [constants.MODEL_DISPLAY_NAMES.get(m, m) for m in mat.index]

    fig_h = max(5.0, 0.32 * len(labels) + 1.8)
    fig_w = max(7.0, 0.68 * len(rois) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(rois)))
    ax.set_xticklabels(rois, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel("ROI")
    ax.set_ylabel("Model")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(value_col)
    savefig(fig, stem, out_dir)


def plot_roi_bars(summary: pd.DataFrame, value_col: str, stem: str, title: str, out_dir: Path):
    sub = summary[summary["stimulus_type"].eq("controversial")].copy()
    if sub.empty:
        return
    rois = [r for r in ROI_ORDER if r in set(sub["roi"])]
    agg = (
        sub.groupby(["roi", "model_set"], as_index=False)[value_col]
        .mean()
        .pivot(index="roi", columns="model_set", values=value_col)
        .reindex(rois)
    )
    model_sets = [s for s in constants.MODEL_SETS.keys() if s in agg.columns]
    x = np.arange(len(agg.index))
    width = 0.8 / max(1, len(model_sets))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, model_set in enumerate(model_sets):
        offset = (i - (len(model_sets) - 1) / 2) * width
        ax.bar(x + offset, agg[model_set].to_numpy(dtype=float), width, label=model_set)
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index, rotation=35, ha="right")
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.legend(frameon=False, ncols=2, fontsize=8)
    savefig(fig, stem, out_dir)


def plot_metric(path: Path, metric: str, out_dir: Path):
    if not path.exists():
        print(f"[skip] missing {path}", flush=True)
        return
    df = pd.read_csv(path)
    value_col = f"{metric}_mean"
    plot_heatmap(
        df,
        value_col,
        f"paper_layer_{metric}_all_models_controversial_heatmap",
        f"Paper-layer {metric}: all-models controversial set",
        out_dir,
    )
    plot_roi_bars(
        df,
        value_col,
        f"paper_layer_{metric}_controversial_roi_summary",
        f"Paper-layer {metric}: controversial set by ROI",
        out_dir,
    )


def plot_crsa_mrsa_delta(out_dir: Path):
    crsa_path = DATA_DIR / "paper_layer_crsa_by_roi_summary.csv"
    mrsa_path = DATA_DIR / "paper_layer_mrsa_by_roi_summary.csv"
    if not crsa_path.exists() or not mrsa_path.exists():
        return
    crsa = pd.read_csv(crsa_path)
    mrsa = pd.read_csv(mrsa_path)
    keys = ["roi", "model_set", "model", "display_name", "stimulus_type"]
    merged = crsa.merge(mrsa, on=keys, how="inner", suffixes=("_crsa", "_mrsa"))
    sub = merged[
        merged["model_set"].eq("all_models")
        & merged["stimulus_type"].eq("controversial")
    ].copy()
    if sub.empty:
        return
    sub["delta_mrsa_minus_crsa"] = sub["mrsa_mean"] - sub["crsa_mean"]
    rois = [r for r in ROI_ORDER if r in set(sub["roi"])]
    agg = sub.groupby("roi", as_index=False)["delta_mrsa_minus_crsa"].mean().set_index("roi").reindex(rois)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    vals = agg["delta_mrsa_minus_crsa"].to_numpy(dtype=float)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.bar(np.arange(len(agg.index)), vals, color="#4d7c8a")
    ax.set_xticks(np.arange(len(agg.index)))
    ax.set_xticklabels(agg.index, rotation=35, ha="right")
    ax.set_ylabel("mRSA - cRSA")
    ax.set_title("Paper-layer mRSA vs cRSA: all-models controversial set")
    savefig(fig, "paper_layer_mrsa_minus_crsa_roi_delta", out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=FIG_DIR)
    args = parser.parse_args()

    plot_metric(DATA_DIR / "paper_layer_crsa_by_roi_summary.csv", "crsa", args.out_dir)
    plot_metric(DATA_DIR / "paper_layer_mrsa_by_roi_summary.csv", "mrsa", args.out_dir)
    plot_crsa_mrsa_delta(args.out_dir)
    print(f"Wrote plots under {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
