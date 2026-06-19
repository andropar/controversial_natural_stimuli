"""
plot_simulation_vs_fmri.py

Two-panel figure:
  Panel a: Pairwise scatter — simulation distance vs fMRI |mRSA diff|
           (all_models, encoding track, controversial stimuli, subject-averaged)
           Within-set pairs colored by model set; cross-set pairs in gray.
  Panel b: Mean ρ per model set — controversial (selected) vs vicco (baseline)
           (encoding track, per-subject dots + mean)

Usage:
  python plot_simulation_vs_fmri.py
"""

import sys
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))
from cstims import constants, paths

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy import stats as sp_stats

from cstims.paper.style_improved import apply_style, FONT, DPI, W_DOUBLE

apply_style()

DATA_DIR = paths.simulation_data_dir()
FIG_DIR = Path(__file__).resolve().parent

MODEL_SET_COLORS = {
    "all_models":         "#9B59B6",
    "sota":               "#2ECC71",
    "architecture":       "#E74C3C",
    "training_objective": "#3498DB",
    "dataset":            "#F39C12",
}

MODEL_SET_LABELS = {
    "all_models":         "All models",
    "sota":               "SOTA",
    "architecture":       "Architecture",
    "training_objective": "Training obj.",
    "dataset":            "Dataset",
}

MODEL_SET_MARKERS = {
    "all_models":         "o",
    "sota":               "s",
    "architecture":       "^",
    "training_objective": "D",
    "dataset":            "v",
}

# Within-set membership for coloring scatter pairs
SET_MODELS = {ms: set(models) for ms, models in constants.MODEL_SETS.items()
              if ms != "all_models"}

PLOT_ORDER = ["sota", "architecture", "training_objective", "dataset"]


def _get_set_membership(pair: str) -> str | None:
    """Return which individual model set both models in a pair belong to, or None."""
    m1, m2 = pair.split("||")
    for set_name in PLOT_ORDER:
        if m1 in SET_MODELS[set_name] and m2 in SET_MODELS[set_name]:
            return set_name
    return None


def plot_pairwise_scatter(ax, pairwise_df: pd.DataFrame, summary_df: pd.DataFrame):
    """
    Panel a: scatter of subject-averaged sim distance vs fMRI |mRSA diff|.
    all_models, encoding track, controversial stimuli.
    Within-set pairs colored; cross-set gray.
    """
    df = pairwise_df[
        (pairwise_df["model_set"] == "all_models") &
        (pairwise_df["track_type"] == "encoding") &
        (pairwise_df["stimulus_type"] == "controversial")
    ]

    # Subject-average per pair
    agg = df.groupby("pair").agg(
        sim_distance=("sim_distance", "mean"),
        fmri_score_diff=("fmri_score_diff", "mean"),
    ).reset_index()
    agg["subset"] = agg["pair"].apply(_get_set_membership)

    # Cross-set (gray background)
    bg = agg[agg["subset"].isna()]
    ax.scatter(
        bg["sim_distance"], bg["fmri_score_diff"],
        c="#CCCCCC", s=16, alpha=0.5, edgecolors="none",
        zorder=2, label=f"Cross-set (n={len(bg)})",
    )

    # Within-set (colored overlay)
    for ms in PLOT_ORDER:
        sub = agg[agg["subset"] == ms]
        if sub.empty:
            continue
        ax.scatter(
            sub["sim_distance"], sub["fmri_score_diff"],
            c=MODEL_SET_COLORS[ms],
            marker=MODEL_SET_MARKERS[ms],
            s=32, alpha=0.9, edgecolors="white", linewidths=0.4,
            zorder=4,
            label=f"{MODEL_SET_LABELS[ms]} (n={len(sub)})",
        )

    # Regression line over all pairs
    slope, intercept, _, _, _ = sp_stats.linregress(
        agg["sim_distance"], agg["fmri_score_diff"]
    )
    x_line = np.linspace(agg["sim_distance"].min(), agg["sim_distance"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "k--", alpha=0.35, linewidth=1, zorder=3)

    # Stats annotation from summary
    row = summary_df[
        (summary_df["model_set"] == "all_models") &
        (summary_df["track_type"] == "encoding") &
        (summary_df["stimulus_type"] == "controversial")
    ]
    if not row.empty:
        mean_rho = row.iloc[0]["mean_rho"]
        rho_sd = row.iloc[0]["rho_sd"]
        ax.text(
            0.04, 0.97,
            f"$\\bar{{\\rho}}$ = {mean_rho:.2f} ± {rho_sd:.2f}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=FONT["annotation"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.85, edgecolor="#CCCCCC", linewidth=0.5),
        )

    ax.set_xlabel("Simulation pairwise distance\n(1 − RDM correlation, encoding space)",
                  fontsize=FONT["axis_label"])
    ax.set_ylabel("fMRI |ΔmRSA|", fontsize=FONT["axis_label"])
    ax.set_title("All models — encoding space (n = 190 pairs)", fontsize=FONT["title"])
    ax.legend(frameon=True, framealpha=0.9, fontsize=FONT["small"],
              loc="upper right", markerscale=0.8, handletextpad=0.3)


COLORS_B = {
    "controversial": "#444444",
    "vicco":         "#BBBBBB",
}


def plot_vicco_comparison(ax, pairwise_df: pd.DataFrame, summary_df: pd.DataFrame):
    """
    Panel b: Mean ρ per model set (encoding track), controversial vs vicco.
    Individual subject dots + mean tick. training_objective annotated separately
    (VICReg leverage inflates it; shown but labeled).
    """
    rho_df = (
        pairwise_df[pairwise_df["track_type"] == "encoding"]
        .groupby(["model_set", "stimulus_type", "subject"])["subject_rho"]
        .first()
        .reset_index()
    )

    ms_order = ["all_models"] + PLOT_ORDER
    x_positions = np.arange(len(ms_order))
    offset = 0.13
    jitter = 0.05

    rng = np.random.default_rng(0)

    for i, ms in enumerate(ms_order):
        for stim_type, xoff in [("controversial", -offset), ("vicco", offset)]:
            color = COLORS_B[stim_type]
            sub = rho_df[
                (rho_df["model_set"] == ms) &
                (rho_df["stimulus_type"] == stim_type)
            ]["subject_rho"].values

            if len(sub) == 0:
                continue

            jx = i + xoff + rng.uniform(-jitter, jitter, len(sub))
            ax.scatter(jx, sub, color=color, s=18, alpha=0.65,
                       edgecolors="none", zorder=3)

            mean_val = np.mean(sub)
            ax.plot([i + xoff - 0.09, i + xoff + 0.09], [mean_val, mean_val],
                    color=color, linewidth=2.2, zorder=4, solid_capstyle="round")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.35, zorder=1)

    # Annotate training_objective as VICReg-inflated
    to_idx = ms_order.index("training_objective")
    ax.text(to_idx, ax.get_ylim()[1] if ax.get_ylim()[1] > 0.5 else 0.95,
            "†", fontsize=FONT["annotation"] + 2, ha="center", va="bottom",
            color="#888888")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [MODEL_SET_LABELS[ms] for ms in ms_order],
        fontsize=FONT["tick"], rotation=20, ha="right",
    )
    ax.set_ylabel("Spearman $\\rho$\n(sim distance vs. fMRI |ΔmRSA|)",
                  fontsize=FONT["axis_label"])
    ax.set_title("Encoding track: selected vs. vicco stimuli", fontsize=FONT["title"])

    h1 = mlines.Line2D([], [], color=COLORS_B["controversial"],
                       marker="o", linestyle="none", markersize=5,
                       label="Selected (controversial)")
    h2 = mlines.Line2D([], [], color=COLORS_B["vicco"],
                       marker="o", linestyle="none", markersize=5,
                       label="Vicco (baseline)")
    ax.legend(handles=[h1, h2], frameon=True, framealpha=0.9,
              fontsize=FONT["annotation"], loc="lower right")
    ax.text(0.01, 0.01, "† VICReg leverage", transform=ax.transAxes,
            fontsize=FONT["small"], color="#888888", va="bottom")


def main():
    pairwise_path = DATA_DIR / "option_b_pairwise.csv"
    summary_path = DATA_DIR / "prediction_summary.csv"

    if not pairwise_path.exists():
        print("No data found. Run 01_compare_simulation_to_fmri.py first.")
        return

    pairwise_df = pd.read_csv(pairwise_path)
    summary_df = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 6.5))

    plot_pairwise_scatter(axes[0], pairwise_df, summary_df)
    axes[0].text(-0.14, 1.06, "a", transform=axes[0].transAxes,
                 fontsize=FONT["panel_label"], fontweight="bold", va="top")

    plot_vicco_comparison(axes[1], pairwise_df, summary_df)
    axes[1].text(-0.14, 1.06, "b", transform=axes[1].transAxes,
                 fontsize=FONT["panel_label"], fontweight="bold", va="top")

    plt.tight_layout(w_pad=3.5)

    for ext in ["pdf", "png"]:
        out = FIG_DIR / f"simulation_vs_fmri.{ext}"
        fig.savefig(out, dpi=DPI)
        print(f"Saved {out}")

    plt.close()


if __name__ == "__main__":
    main()
