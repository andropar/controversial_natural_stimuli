#!/usr/bin/env python3
"""Plot noisy-by-clean pairwise dominance and correlation-margin curves."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[5]
HELPERS_DIR = ROOT / "shared" / "code" / "paper_helpers"
sys.path.insert(0, str(HELPERS_DIR))
sys.path.insert(0, str(HELPERS_DIR / "figures"))

from style_improved import (  # noqa: E402
    DPI,
    FONT,
    MODEL_SET_DISPLAY_SHORT,
    MODEL_SET_ORDER,
    apply_style,
)


apply_style()

DEFAULT_RESULTS = SCRIPT.parents[1] / "results"
DEFAULT_FIGURES = SCRIPT.parents[1] / "figures"
ENCODING_TRACKS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
SNR_TICKS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
SNR_TICK_LABELS = ["0.01", "0.03", "0.1", "0.3", "1", "3", "10"]
MODEL_SET_COLORS = {
    "all_models": "#222222",
    "sota": "#6A3D9A",
    "training_objective": "#8C564B",
    "architecture": "#009E73",
    "dataset": "#E7298A",
}


def model_set_color(name: str) -> str:
    return MODEL_SET_COLORS.get(name, "#666666")


def load_pairwise(results_root: Path, model_set: str) -> pd.DataFrame:
    path = results_root / f"{model_set}_noisy_by_clean_boot" / "pairwise_margin.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["snr"] = 1.0 / df["noise_mult"].astype(float)
    return df


def aggregate_tracks(
    df: pd.DataFrame,
    tracks: list[str],
    value_col: str,
    selected_std_col: str,
    random_std_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    sub = df[df["track"].isin(tracks)].copy()
    if sub.empty:
        return None

    selected = (
        sub[sub["subset_type"] == "selected"]
        .groupby("snr", as_index=False)
        .agg(value=(value_col, "mean"), std=(selected_std_col, "mean"))
        .sort_values("snr")
    )
    random = (
        sub[sub["subset_type"] == "random"]
        .groupby("snr", as_index=False)
        .agg(value=(value_col, "mean"), std=(random_std_col, "mean"))
        .sort_values("snr")
    )
    if selected.empty or random.empty:
        return None

    return (
        selected["snr"].to_numpy(float),
        selected["value"].to_numpy(float),
        selected["std"].fillna(0).to_numpy(float),
        random["value"].to_numpy(float),
        random["std"].fillna(0).to_numpy(float),
    )


def style_axis(ax, ylabel: str, show_xlabel: bool) -> None:
    ax.axvline(1.0, color="#444444", lw=1.0, ls=(0, (4, 2)), alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlim(0.009, 11.0)
    ax.set_xticks(SNR_TICKS)
    ax.set_xticklabels(SNR_TICK_LABELS if show_xlabel else [])
    ax.set_xlabel("Relative signal-to-noise ratio" if show_xlabel else "")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.30)


def plot_track_group(results_root: Path, figures_root: Path, name: str, tracks: list[str]) -> None:
    fig, axes = plt.subplots(
        2,
        len(MODEL_SET_ORDER),
        figsize=(11.4, 4.8),
        sharex=True,
        constrained_layout=True,
    )

    all_margin_values: list[float] = []
    for col_idx, model_set in enumerate(MODEL_SET_ORDER):
        df = load_pairwise(results_root, model_set)
        title = MODEL_SET_DISPLAY_SHORT.get(model_set, model_set)
        color = model_set_color(model_set)

        for row_idx, (
            value_col,
            selected_std_col,
            random_std_col,
            ylabel,
            chance,
        ) in enumerate(
            [
                (
                    "pairwise_dominance",
                    "pairwise_dominance_mc_std",
                    "pairwise_dominance_subset_std",
                    "Pairwise dominance",
                    0.5,
                ),
                (
                    "mean_margin",
                    "mean_margin_mc_std",
                    "mean_margin_subset_std",
                    "Mean corr. margin",
                    0.0,
                ),
            ]
        ):
            ax = axes[row_idx, col_idx]
            if row_idx == 0:
                ax.set_title(title, pad=3)
            if df.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue

            agg = aggregate_tracks(
                df,
                tracks,
                value_col,
                selected_std_col,
                random_std_col,
            )
            if agg is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue

            x, selected, selected_std, random, random_std = agg
            ax.axhline(chance, color="#999999", lw=0.8, ls=":", zorder=1)
            ax.plot(x, random, color="#6F6F6F", lw=1.0, ls="--", alpha=0.75, label="Random")
            ax.plot(x, selected, color=color, lw=1.35, alpha=0.96, label="Controversial")

            if np.any(np.isfinite(random_std)):
                ax.fill_between(
                    x,
                    random - random_std,
                    random + random_std,
                    color="#6F6F6F",
                    alpha=0.10,
                    linewidth=0,
                )
            if np.any(np.isfinite(selected_std)):
                ax.fill_between(
                    x,
                    selected - selected_std,
                    selected + selected_std,
                    color=color,
                    alpha=0.08,
                    linewidth=0,
                )
            if value_col == "mean_margin":
                all_margin_values.extend(selected[np.isfinite(selected)].tolist())
                all_margin_values.extend(random[np.isfinite(random)].tolist())

            style_axis(
                ax,
                ylabel if col_idx == 0 else "",
                show_xlabel=row_idx == 1,
            )
            if col_idx > 0:
                ax.set_ylabel("")
                plt.setp(ax.get_yticklabels(), visible=False)

    for ax in axes[0]:
        ax.set_ylim(0.45, 1.02)

    if all_margin_values:
        vals = np.asarray(all_margin_values, dtype=float)
        limit = max(abs(np.nanmin(vals)), abs(np.nanmax(vals)), 0.02) * 1.18
        for ax in axes[1]:
            ax.set_ylim(-limit, limit)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 1.04),
            fontsize=FONT["small"],
        )

    figures_root.mkdir(parents=True, exist_ok=True)
    png_dir = figures_root / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = figures_root / f"pairwise_dominance_margin_{name}.pdf"
    png_path = png_dir / f"pairwise_dominance_margin_{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figures-root", type=Path, default=DEFAULT_FIGURES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_track_group(args.results_root, args.figures_root, "raw", ["raw"])
    plot_track_group(args.results_root, args.figures_root, "encoding", ENCODING_TRACKS)


if __name__ == "__main__":
    main()
