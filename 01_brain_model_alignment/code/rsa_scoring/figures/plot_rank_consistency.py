#!/usr/bin/env python3
"""
Plot model ranking consistency across subjects (bump chart).

Each line = a model. X-axis = participants. Y-axis = rank (1 = best).
Lines crossing = rank changes. Labels on both sides for readability.

Outputs:
    figures/rank_consistency_{method}.pdf/png
    figures/rank_consistency_{method}_data.csv

Usage:
    python plot_rank_consistency.py
"""

import sys
from pathlib import Path
from itertools import combinations

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))  # project root
sys.path.insert(0, str(_PAPER / "figures"))  # for shared figure style
import config

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

MODEL_SETS = config.MODEL_SETS
MODEL_DISPLAY_NAMES = config.MODEL_DISPLAY_NAMES
SUBJECTS = config.SUBJECTS
RSA_DATA_DIR = config.RSA_DATA_DIR
FIGURES_DIR = Path(__file__).resolve().parent

from style_improved import apply_style, FONT, DPI, W_DOUBLE

apply_style()

TITLE_MAP = {
    "training_objective": "Training Objective",
    "sota": "SOTA",
    "architecture": "Architecture",
    "dataset": "Dataset",
}

PANEL_LABELS = ["a", "b", "c", "d"]

# Hand-picked distinct colors for each model set to avoid confusion
SET_COLORS = {
    "sota": {
        "dinov2_vitl14": "#1f77b4",
        "slip_vit_l_slip": "#ff7f0e",
        "slip_vit_l_simclr": "#2ca02c",
        "openclip_vit_so400m_14_siglip_webli": "#d62728",
        "timm_vit_large_patch14_clip_224_laion2b": "#9467bd",
        "torchvision_convnext_base_imagenet1k_v1": "#8c564b",
    },
    "architecture": {
        "cornet_s": "#1f77b4",
        "torchvision_resnet50_imagenet1k_v1": "#ff7f0e",
        "torchvision_convnext_base_imagenet1k_v1": "#2ca02c",
        "torchvision_vgg16_imagenet1k_v1": "#d62728",
        "torchvision_vit_l_16_imagenet1k_v1": "#9467bd",
    },
    "training_objective": {
        "vissl_resnet50_barlowtwins": "#1f77b4",
        "vissl_resnet50_mocov2": "#ff7f0e",
        "vissl_resnet50_supervised": "#2ca02c",
        "vicreg_resnet50": "#d62728",
        "robustness_imagenet_l2_eps3": "#9467bd",
    },
    "dataset": {
        "timm_vit_large_patch14_clip_quickgelu_224_openai": "#1f77b4",
        "openclip_vit_l_14_quickgelu_metaclip_400m": "#ff7f0e",
        "openclip_vit_l_14_quickgelu_metaclip_fullcc": "#2ca02c",
        "timm_vit_large_patch14_clip_224_dfn2b": "#d62728",
        "openclip_vit_l_14_laion400m_e31": "#9467bd",
    },
}

SUBJECT_LABELS = {
    "sub-01": "P1",
    "sub-03": "P2",
    "sub-05": "P3",
    "sub-06": "P4",
    "sub-07": "P5",
}


def load_all_scores(method: str) -> pd.DataFrame:
    """Load scores for all subjects for a given method."""
    filename = f"{method}_scores.csv"
    dfs = []
    for subject in SUBJECTS:
        path = RSA_DATA_DIR / subject / filename
        if path.exists():
            dfs.append(pd.read_csv(path))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def compute_rankings(df: pd.DataFrame, model_set: str, score_col: str) -> pd.DataFrame:
    """Compute per-subject rankings for a model set."""
    models = MODEL_SETS[model_set]
    cstim = df[
        (df["model_set"] == model_set) &
        (df["stimulus_type"] == "controversial")
    ]

    rows = []
    for subject in cstim["subject"].unique():
        sub_df = cstim[cstim["subject"] == subject]
        sub_df = sub_df[sub_df["model"].isin(models)]
        if sub_df.empty:
            continue
        ranked = sub_df.sort_values(score_col, ascending=False)
        for rank, (_, row) in enumerate(ranked.iterrows(), 1):
            rows.append({
                "subject": subject,
                "model": row["model"],
                "display_name": MODEL_DISPLAY_NAMES.get(row["model"], row["model"]),
                "score": row[score_col],
                "rank": rank,
            })

    return pd.DataFrame(rows)


def compute_rank_consistency(rankings_df: pd.DataFrame) -> float:
    """Compute mean pairwise Spearman rank correlation across subjects."""
    subjects = sorted(rankings_df["subject"].unique())
    if len(subjects) < 2:
        return np.nan

    correlations = []
    for s1, s2 in combinations(subjects, 2):
        r1 = rankings_df[rankings_df["subject"] == s1].set_index("model")["rank"]
        r2 = rankings_df[rankings_df["subject"] == s2].set_index("model")["rank"]
        common = r1.index.intersection(r2.index)
        if len(common) < 3:
            continue
        rho, _ = stats.spearmanr(r1[common], r2[common])
        correlations.append(rho)

    return np.mean(correlations) if correlations else np.nan


def plot_bump_chart(ax, rankings_df: pd.DataFrame, model_set: str, title: str,
                    panel_label: str = None):
    """Draw a publication-quality bump chart."""
    subjects = sorted(rankings_df["subject"].unique())
    n_subjects = len(subjects)
    models_in_set = MODEL_SETS[model_set]
    # Sort models by their mean rank for consistent ordering
    mean_ranks = rankings_df.groupby("model")["rank"].mean()
    models = sorted(rankings_df["model"].unique(), key=lambda m: mean_ranks.get(m, 99))
    n_models = len(models)

    if n_subjects == 0 or n_models == 0:
        ax.set_visible(False)
        return

    subject_positions = {s: i for i, s in enumerate(subjects)}
    colors = SET_COLORS.get(model_set, {})

    for model in models:
        model_df = rankings_df[rankings_df["model"] == model].set_index("subject")
        xs = []
        ys = []
        for s in subjects:
            if s in model_df.index:
                xs.append(subject_positions[s])
                ys.append(model_df.loc[s, "rank"])

        if not xs:
            continue

        color = colors.get(model, "#888888")
        display_name = MODEL_DISPLAY_NAMES.get(model, model)

        ax.plot(xs, ys, "o-", color=color, linewidth=1.5, markersize=5,
                alpha=0.85, zorder=3, solid_capstyle="round")

        # Label on the left
        ax.text(xs[0] - 0.15, ys[0], display_name,
                va="center", ha="right", fontsize=FONT["small"], color=color,
                fontweight="bold")

        # Label on the right
        ax.text(xs[-1] + 0.15, ys[-1], display_name,
                va="center", ha="left", fontsize=FONT["small"], color=color,
                fontweight="bold")

    # Rank consistency
    rho = compute_rank_consistency(rankings_df)

    ax.set_xticks(range(n_subjects))
    ax.set_xticklabels([SUBJECT_LABELS.get(s, s) for s in subjects], fontsize=FONT["tick"])
    ax.set_xlabel("Participant")
    ax.set_ylabel("Rank")
    ax.set_ylim(n_models + 0.5, 0.5)
    ax.set_yticks(range(1, n_models + 1))
    ax.set_xlim(-1.2, n_subjects - 1 + 1.2)
    ax.set_title(f"{title}   ($\\bar{{\\rho}}$ = {rho:.2f})", fontweight="bold")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)

    # Panel label
    if panel_label:
        ax.text(-0.22, 1.05, panel_label, transform=ax.transAxes,
                fontsize=FONT["panel_label"], fontweight="bold", va="top")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for method, score_col in [("wrsa_transfer", "wrsa_transfer"), ("crsa", "crsa")]:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")

        df = load_all_scores(method)
        if df.empty:
            print(f"  No data for {method}")
            continue

        model_sets_ordered = ["sota", "training_objective", "architecture", "dataset"]
        n_sets = len(model_sets_ordered)

        # Taller figure with more horizontal space for labels
        fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE, 11.0))
        axes = axes.flatten()

        all_ranking_data = []

        for i, model_set in enumerate(model_sets_ordered):
            rankings = compute_rankings(df, model_set, score_col)
            if rankings.empty:
                axes[i].set_visible(False)
                continue

            rho = compute_rank_consistency(rankings)
            print(f"  {model_set}: mean rank rho = {rho:.2f}")

            rankings["model_set"] = model_set
            rankings["method"] = method
            all_ranking_data.append(rankings)

            plot_bump_chart(
                axes[i], rankings, model_set,
                TITLE_MAP.get(model_set, model_set),
                panel_label=PANEL_LABELS[i],
            )

        plt.tight_layout(w_pad=5.0, h_pad=2.5)

        for ext in ["pdf", "png"]:
            out = FIGURES_DIR / f"rank_consistency_{method}.{ext}"
            fig.savefig(out)
            print(f"  Saved {out}")
        plt.close(fig)

        if all_ranking_data:
            ranking_df = pd.concat(all_ranking_data, ignore_index=True)
            ranking_df.to_csv(
                FIGURES_DIR / f"rank_consistency_{method}_data.csv", index=False
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
