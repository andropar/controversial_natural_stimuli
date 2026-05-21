#!/usr/bin/env python3
"""Plot encoding model results.

Usage:
    python plot_encoding_results.py /path/to/run_dir
    python plot_encoding_results.py  # uses most recent run

    # Also generate brain maps (requires loading encoding models)
    python plot_encoding_results.py --brain-maps
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

SCRIPT_DIR = Path(__file__).resolve().parent


def _find_share_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists() and (path / "src" / "cstims").exists():
            return path
    return start.parents[1]


PROJECT_ROOT = _find_share_root(SCRIPT_DIR)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

# Model metadata for grouping
MODEL_FAMILIES = {
    "resnet": ["resnet50", "resnet18", "resnet101", "resnet152"],
    "vit": ["vit", "deit", "beit", "dinov2"],
    "clip": ["clip"],
    "convnext": ["convnext"],
    "efficientnet": ["efficientnet"],
    "vgg": ["vgg"],
    "alexnet": ["alexnet"],
}

TRAINING_OBJECTIVES = {
    "supervised": ["supervised", "imagenet1k", "imagenet21k"],
    "clip": ["clip", "openclip", "laion"],
    "mocov2": ["mocov2", "moco"],
    "barlowtwins": ["barlowtwins", "barlow"],
    "simclr": ["simclr"],
    "dino": ["dino", "dinov2"],
    "swav": ["swav"],
    "mae": ["mae"],
}


def find_latest_run(results_root: Path) -> Path:
    """Find the most recent encoding run directory."""
    runs = sorted(results_root.glob("encoding_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No encoding runs found in {results_root}")
    return runs[-1]


def classify_model_family(model_name: str) -> str:
    """Classify model into architecture family."""
    model_lower = model_name.lower()
    for family, keywords in MODEL_FAMILIES.items():
        if any(kw in model_lower for kw in keywords):
            return family
    return "other"


def classify_training_objective(model_name: str) -> str:
    """Classify model by training objective."""
    model_lower = model_name.lower()
    for objective, keywords in TRAINING_OBJECTIVES.items():
        if any(kw in model_lower for kw in keywords):
            return objective
    return "other"


def add_model_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add model family and training objective columns."""
    df = df.copy()
    df["family"] = df["model"].apply(classify_model_family)
    df["objective"] = df["model"].apply(classify_training_objective)
    return df


def shorten_model_name(name: str) -> str:
    """Shorten model name for display."""
    # Remove common prefixes
    for prefix in ["torchvision_", "timm_", "vissl_", "clip_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    # Truncate if still too long
    if len(name) > 30:
        name = name[:27] + "..."
    return name


def plot_model_ranking(df: pd.DataFrame, out_dir: Path):
    """Bar plot of models ranked by veRSA."""
    # Average across subjects if multiple
    if df["subject"].nunique() > 1:
        model_df = df.groupby("model").agg({
            "veRSA": ["mean", "std"],
            "voxel_r_median": "mean",
        }).reset_index()
        model_df.columns = ["model", "veRSA", "veRSA_std", "voxel_r_median"]
    else:
        model_df = df[["model", "veRSA", "voxel_r_median"]].copy()
        model_df["veRSA_std"] = 0

    model_df = model_df.sort_values("veRSA", ascending=True)
    model_df["model_short"] = model_df["model"].apply(shorten_model_name)

    fig, ax = plt.subplots(figsize=(10, max(6, len(model_df) * 0.4)))

    y_pos = np.arange(len(model_df))
    bars = ax.barh(y_pos, model_df["veRSA"], xerr=model_df["veRSA_std"],
                   capsize=3, color=plt.cm.viridis(model_df["veRSA"] / model_df["veRSA"].max()))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_df["model_short"])
    ax.set_xlabel("veRSA (cross-dataset)")
    ax.set_title("Encoding Model Performance (veRSA)")

    # Add value labels
    for i, (v, s) in enumerate(zip(model_df["veRSA"], model_df["veRSA_std"])):
        ax.text(v + s + 0.01, i, f"{v:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "model_ranking.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "model_ranking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: model_ranking.pdf/png")


def plot_per_dataset_breakdown(df: pd.DataFrame, out_dir: Path):
    """Grouped bar plot showing veRSA per dataset for each model."""
    # Find dataset columns
    dataset_cols = sorted([c for c in df.columns if c.startswith("veRSA_")
                          and c not in ["veRSA_overall", "veRSA_mean_across_datasets",
                                       "veRSA_std_across_datasets"]])

    if not dataset_cols:
        print("  Skipping per-dataset plot (no dataset columns)")
        return

    # Average across subjects
    if df["subject"].nunique() > 1:
        plot_df = df.groupby("model")[dataset_cols].mean().reset_index()
    else:
        plot_df = df[["model"] + dataset_cols].copy()

    plot_df["model_short"] = plot_df["model"].apply(shorten_model_name)

    # Sort by mean veRSA across datasets
    plot_df["mean_veRSA"] = plot_df[dataset_cols].mean(axis=1)
    plot_df = plot_df.sort_values("mean_veRSA", ascending=False)

    # Melt for seaborn
    melt_df = plot_df.melt(
        id_vars=["model_short"],
        value_vars=dataset_cols,
        var_name="dataset",
        value_name="veRSA"
    )
    melt_df["dataset"] = melt_df["dataset"].str.replace("veRSA_", "")

    fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.5)))

    sns.barplot(
        data=melt_df,
        y="model_short",
        x="veRSA",
        hue="dataset",
        ax=ax,
        order=plot_df["model_short"].tolist(),
    )

    ax.set_xlabel("veRSA")
    ax.set_ylabel("")
    ax.set_title("Encoding Performance by Dataset")
    ax.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    fig.savefig(out_dir / "per_dataset_breakdown.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "per_dataset_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: per_dataset_breakdown.pdf/png")


def plot_versa_vs_voxelr(df: pd.DataFrame, out_dir: Path):
    """Scatter plot of veRSA vs voxel-wise correlation."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Color by model if multiple subjects, otherwise just scatter
    if df["subject"].nunique() > 1:
        for model in df["model"].unique():
            model_df = df[df["model"] == model]
            ax.scatter(model_df["voxel_r_median"], model_df["veRSA"],
                      label=shorten_model_name(model), alpha=0.7, s=60)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    else:
        ax.scatter(df["voxel_r_median"], df["veRSA"], alpha=0.7, s=80)
        # Add model labels
        for _, row in df.iterrows():
            ax.annotate(shorten_model_name(row["model"]),
                       (row["voxel_r_median"], row["veRSA"]),
                       fontsize=7, alpha=0.8,
                       xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Voxel-wise r (median)")
    ax.set_ylabel("veRSA")
    ax.set_title("Encoding Performance: veRSA vs Voxel Correlation")

    # Add diagonal reference
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.3, zorder=0)

    plt.tight_layout()
    fig.savefig(out_dir / "versa_vs_voxelr.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "versa_vs_voxelr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: versa_vs_voxelr.pdf/png")


def plot_cross_dataset_heatmap(df: pd.DataFrame, out_dir: Path):
    """Heatmap of model x dataset veRSA."""
    dataset_cols = sorted([c for c in df.columns if c.startswith("veRSA_")
                          and c not in ["veRSA_overall", "veRSA_mean_across_datasets",
                                       "veRSA_std_across_datasets"]])

    if not dataset_cols or len(dataset_cols) < 2:
        print("  Skipping heatmap (need at least 2 datasets)")
        return

    # Average across subjects
    if df["subject"].nunique() > 1:
        heatmap_df = df.groupby("model")[dataset_cols].mean()
    else:
        heatmap_df = df.set_index("model")[dataset_cols]

    # Rename columns
    heatmap_df.columns = [c.replace("veRSA_", "") for c in heatmap_df.columns]

    # Sort by mean veRSA
    heatmap_df["mean"] = heatmap_df.mean(axis=1)
    heatmap_df = heatmap_df.sort_values("mean", ascending=False).drop(columns=["mean"])

    # Shorten model names for display
    heatmap_df.index = [shorten_model_name(m) for m in heatmap_df.index]

    fig, ax = plt.subplots(figsize=(8, max(6, len(heatmap_df) * 0.4)))

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "veRSA"},
    )

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Model")
    ax.set_title("Cross-Dataset Generalization")

    plt.tight_layout()
    fig.savefig(out_dir / "cross_dataset_heatmap.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "cross_dataset_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: cross_dataset_heatmap.pdf/png")


def plot_alpha_distribution(df: pd.DataFrame, out_dir: Path):
    """Distribution of chosen alpha values."""
    if "alpha_median" not in df.columns or df["alpha_median"].isna().all():
        print("  Skipping alpha plot (no alpha data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of log alpha
    ax = axes[0]
    log_alphas = np.log10(df["alpha_median"].dropna())
    ax.hist(log_alphas, bins=20, edgecolor="black", alpha=0.7)
    ax.set_xlabel("log10(alpha)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Optimal Alpha")

    # Alpha by model
    ax = axes[1]
    model_order = df.groupby("model")["alpha_median"].median().sort_values().index
    plot_df = df[["model", "alpha_median"]].copy()
    plot_df["model_short"] = plot_df["model"].apply(shorten_model_name)

    if len(model_order) <= 15:
        sns.boxplot(data=plot_df, y="model_short", x="alpha_median", ax=ax,
                   order=[shorten_model_name(m) for m in model_order])
        ax.set_xscale("log")
        ax.set_xlabel("Alpha (median)")
        ax.set_ylabel("")
    else:
        # Too many models, just show scatter
        ax.scatter(df["alpha_median"], df["veRSA"], alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("Alpha (median)")
        ax.set_ylabel("veRSA")
    ax.set_title("Optimal Alpha by Model")

    plt.tight_layout()
    fig.savefig(out_dir / "alpha_distribution.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "alpha_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: alpha_distribution.pdf/png")


def plot_alpha_stability(df: pd.DataFrame, out_dir: Path):
    """Visualize alpha stability across CV folds."""
    if "alpha_cv_median" not in df.columns or df["alpha_cv_median"].isna().all():
        print("  Skipping alpha stability (no fold alpha data)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Histogram of alpha CV across models
    ax = axes[0]
    cv_values = df["alpha_cv_median"].dropna()
    ax.hist(cv_values, bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(cv_values.median(), color="red", linestyle="--",
               label=f"Median: {cv_values.median():.3f}")
    ax.set_xlabel("Alpha CV (median across voxels)")
    ax.set_ylabel("Number of models")
    ax.set_title("Alpha Stability Distribution")
    ax.legend()

    # 2. Fold-fold correlation distribution
    ax = axes[1]
    if "alpha_fold_corr_mean" in df.columns and not df["alpha_fold_corr_mean"].isna().all():
        corr_values = df["alpha_fold_corr_mean"].dropna()
        ax.hist(corr_values, bins=20, edgecolor="black", alpha=0.7)
        ax.axvline(corr_values.median(), color="red", linestyle="--",
                   label=f"Median: {corr_values.median():.3f}")
        ax.set_xlabel("Mean fold-fold correlation (log alpha)")
        ax.set_ylabel("Number of models")
        ax.set_title("Alpha Consistency Across Folds")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No fold correlation data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Alpha Consistency Across Folds")

    # 3. Alpha stability vs veRSA
    ax = axes[2]
    valid = df[["alpha_cv_median", "veRSA"]].dropna()
    if len(valid) > 2:
        ax.scatter(valid["alpha_cv_median"], valid["veRSA"], alpha=0.6, s=50)
        ax.set_xlabel("Alpha CV (median)")
        ax.set_ylabel("veRSA")
        r, p = stats.spearmanr(valid["alpha_cv_median"], valid["veRSA"])
        ax.set_title(f"Stability vs Performance\nSpearman r={r:.3f}, p={p:.2e}")

        # Add model labels for outliers
        for _, row in valid.iterrows():
            if row["alpha_cv_median"] > valid["alpha_cv_median"].quantile(0.9):
                model_name = df.loc[row.name, "model"] if "model" in df.columns else ""
                ax.annotate(shorten_model_name(model_name)[:10],
                           (row["alpha_cv_median"], row["veRSA"]),
                           fontsize=7, alpha=0.7,
                           xytext=(3, 3), textcoords="offset points")
    else:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Stability vs Performance")

    plt.tight_layout()
    fig.savefig(out_dir / "alpha_stability.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "alpha_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: alpha_stability.pdf/png")


def plot_subject_consistency(df: pd.DataFrame, out_dir: Path):
    """Show consistency across subjects if multiple subjects."""
    if df["subject"].nunique() <= 1:
        print("  Skipping subject consistency (single subject)")
        return

    # Pivot to get model x subject
    pivot_df = df.pivot(index="model", columns="subject", values="veRSA")
    pivot_df.index = [shorten_model_name(m) for m in pivot_df.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Heatmap
    ax = axes[0]
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
    ax.set_title("veRSA by Model and Subject")

    # Correlation across subjects
    ax = axes[1]
    subjects = pivot_df.columns.tolist()
    if len(subjects) >= 2:
        corr_matrix = pivot_df.corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                   vmin=0, vmax=1, ax=ax)
        ax.set_title("Subject-Subject Correlation (Model Rankings)")

    plt.tight_layout()
    fig.savefig(out_dir / "subject_consistency.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "subject_consistency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: subject_consistency.pdf/png")


def plot_training_objective_comparison(df: pd.DataFrame, out_dir: Path):
    """Compare models grouped by training objective (supervised vs self-supervised)."""
    df = add_model_metadata(df)

    # Average across subjects if needed
    if df["subject"].nunique() > 1:
        plot_df = df.groupby(["model", "objective"]).agg({
            "veRSA": "mean",
            "voxel_r_median": "mean",
        }).reset_index()
    else:
        plot_df = df[["model", "objective", "veRSA", "voxel_r_median"]].copy()

    # Filter to objectives with enough models
    obj_counts = plot_df["objective"].value_counts()
    valid_objs = obj_counts[obj_counts >= 1].index.tolist()
    plot_df = plot_df[plot_df["objective"].isin(valid_objs)]

    if len(valid_objs) < 2:
        print("  Skipping training objective plot (not enough objectives)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Box plot of veRSA by objective
    ax = axes[0]
    order = plot_df.groupby("objective")["veRSA"].median().sort_values(ascending=False).index
    sns.boxplot(data=plot_df, x="objective", y="veRSA", order=order, ax=ax)
    sns.stripplot(data=plot_df, x="objective", y="veRSA", order=order,
                  color="black", alpha=0.5, size=4, ax=ax)
    ax.set_xlabel("Training Objective")
    ax.set_ylabel("veRSA")
    ax.set_title("Encoding Performance by Training Objective")
    ax.tick_params(axis='x', rotation=45)

    # Violin plot
    ax = axes[1]
    sns.violinplot(data=plot_df, x="objective", y="veRSA", order=order, ax=ax, inner="box")
    ax.set_xlabel("Training Objective")
    ax.set_ylabel("veRSA")
    ax.set_title("veRSA Distribution by Training Objective")
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    fig.savefig(out_dir / "training_objective_comparison.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "training_objective_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: training_objective_comparison.pdf/png")


def plot_generalization_gap(df: pd.DataFrame, out_dir: Path):
    """Compare LOO-CV score (in-sample) vs cross-dataset veRSA (true generalization)."""
    if "loo_cv_score_mean" not in df.columns:
        print("  Skipping generalization gap (no LOO-CV scores)")
        return

    # Average across subjects
    if df["subject"].nunique() > 1:
        plot_df = df.groupby("model").agg({
            "loo_cv_score_mean": "mean",
            "veRSA": "mean",
        }).reset_index()
    else:
        plot_df = df[["model", "loo_cv_score_mean", "veRSA"]].copy()

    plot_df["generalization_gap"] = plot_df["loo_cv_score_mean"] - plot_df["veRSA"]
    plot_df["model_short"] = plot_df["model"].apply(shorten_model_name)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Scatter: LOO-CV vs veRSA
    ax = axes[0]
    ax.scatter(plot_df["loo_cv_score_mean"], plot_df["veRSA"], alpha=0.7, s=60)

    # Add diagonal
    lims = [0, max(plot_df["loo_cv_score_mean"].max(), plot_df["veRSA"].max()) * 1.1]
    ax.plot(lims, lims, "k--", alpha=0.3, label="y=x (no gap)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Correlation
    r, p = stats.pearsonr(plot_df["loo_cv_score_mean"], plot_df["veRSA"])
    ax.set_xlabel("LOO-CV Score (in-sample)")
    ax.set_ylabel("veRSA (cross-dataset)")
    ax.set_title(f"Generalization: In-sample vs Cross-dataset\nr={r:.3f}, p={p:.3g}")
    ax.legend()

    # Bar plot of generalization gap
    ax = axes[1]
    sorted_df = plot_df.sort_values("generalization_gap", ascending=True)
    colors = ["green" if g < 0 else "red" for g in sorted_df["generalization_gap"]]
    ax.barh(range(len(sorted_df)), sorted_df["generalization_gap"], color=colors, alpha=0.7)
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df["model_short"], fontsize=8)
    ax.axvline(0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Gap (LOO-CV - veRSA)")
    ax.set_title("Generalization Gap\n(negative = generalizes better than expected)")

    # Distribution of gap
    ax = axes[2]
    ax.hist(plot_df["generalization_gap"], bins=15, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", label="No gap")
    ax.axvline(plot_df["generalization_gap"].mean(), color="blue", linestyle="--",
               label=f"Mean: {plot_df['generalization_gap'].mean():.3f}")
    ax.set_xlabel("Generalization Gap")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Generalization Gap")
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_dir / "generalization_gap.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "generalization_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: generalization_gap.pdf/png")


def plot_dataset_difficulty(df: pd.DataFrame, out_dir: Path):
    """Analyze which datasets are easiest/hardest to predict."""
    dataset_cols = sorted([c for c in df.columns if c.startswith("veRSA_")
                          and c not in ["veRSA_overall", "veRSA_mean_across_datasets",
                                       "veRSA_std_across_datasets"]])

    if len(dataset_cols) < 2:
        print("  Skipping dataset difficulty (need multiple datasets)")
        return

    # Compute mean/std across models for each dataset
    dataset_stats = []
    for col in dataset_cols:
        dataset_name = col.replace("veRSA_", "")
        vals = df[col].dropna()
        dataset_stats.append({
            "dataset": dataset_name,
            "mean_veRSA": vals.mean(),
            "std_veRSA": vals.std(),
            "median_veRSA": vals.median(),
            "min_veRSA": vals.min(),
            "max_veRSA": vals.max(),
        })

    stats_df = pd.DataFrame(dataset_stats).sort_values("mean_veRSA", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar plot with error bars
    ax = axes[0]
    x = range(len(stats_df))
    ax.bar(x, stats_df["mean_veRSA"], yerr=stats_df["std_veRSA"],
           capsize=5, color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(stats_df))))
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df["dataset"], rotation=45, ha="right")
    ax.set_ylabel("veRSA (mean ± std across models)")
    ax.set_title("Dataset Difficulty\n(higher = easier to predict)")

    # Box plot showing distribution across models
    ax = axes[1]
    plot_data = []
    for col in dataset_cols:
        dataset_name = col.replace("veRSA_", "")
        for val in df[col].dropna():
            plot_data.append({"dataset": dataset_name, "veRSA": val})
    plot_df = pd.DataFrame(plot_data)

    order = stats_df["dataset"].tolist()
    sns.boxplot(data=plot_df, x="dataset", y="veRSA", order=order, ax=ax)
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel("veRSA")
    ax.set_title("veRSA Distribution per Dataset")

    plt.tight_layout()
    fig.savefig(out_dir / "dataset_difficulty.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "dataset_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: dataset_difficulty.pdf/png")


def plot_dataset_agreement(df: pd.DataFrame, out_dir: Path):
    """Analyze whether datasets agree on which models are best."""
    dataset_cols = sorted([c for c in df.columns if c.startswith("veRSA_")
                          and c not in ["veRSA_overall", "veRSA_mean_across_datasets",
                                       "veRSA_std_across_datasets"]])

    if len(dataset_cols) < 2:
        print("  Skipping dataset agreement (need multiple datasets)")
        return

    # Average across subjects
    if df["subject"].nunique() > 1:
        model_df = df.groupby("model")[dataset_cols].mean()
    else:
        model_df = df.set_index("model")[dataset_cols]

    # Rename columns
    model_df.columns = [c.replace("veRSA_", "") for c in model_df.columns]

    # Correlation matrix between datasets
    corr_matrix = model_df.corr(method="spearman")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Correlation heatmap
    ax = axes[0]
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
               vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title("Dataset Agreement on Model Rankings\n(Spearman correlation)")

    # Pairwise scatter for most/least agreeing pairs
    ax = axes[1]
    # Find pair with lowest correlation
    np.fill_diagonal(corr_matrix.values, 1)  # Ignore self-correlation
    min_idx = np.unravel_index(corr_matrix.values.argmin(), corr_matrix.shape)
    ds1, ds2 = corr_matrix.columns[min_idx[0]], corr_matrix.columns[min_idx[1]]

    ax.scatter(model_df[ds1], model_df[ds2], alpha=0.7, s=60)
    r = corr_matrix.loc[ds1, ds2]
    ax.set_xlabel(f"veRSA on {ds1}")
    ax.set_ylabel(f"veRSA on {ds2}")
    ax.set_title(f"Least Agreeing Datasets\n{ds1} vs {ds2} (r={r:.2f})")

    # Add model labels for outliers
    for model in model_df.index:
        x, y = model_df.loc[model, ds1], model_df.loc[model, ds2]
        # Label if outlier (far from diagonal)
        diff = abs(x - y)
        if diff > model_df[[ds1, ds2]].diff(axis=1).iloc[:, 1].abs().quantile(0.8):
            ax.annotate(shorten_model_name(model), (x, y), fontsize=7,
                       xytext=(5, 5), textcoords="offset points")

    plt.tight_layout()
    fig.savefig(out_dir / "dataset_agreement.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "dataset_agreement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: dataset_agreement.pdf/png")


def plot_ood_analysis(df: pd.DataFrame, out_dir: Path):
    """Special analysis of OOD (out-of-distribution) generalization."""
    if "veRSA_OOD" not in df.columns:
        print("  Skipping OOD analysis (no OOD column)")
        return

    # Get other dataset columns
    other_cols = [c for c in df.columns if c.startswith("veRSA_")
                  and c not in ["veRSA_OOD", "veRSA_overall", "veRSA_mean_across_datasets",
                               "veRSA_std_across_datasets"]]

    if not other_cols:
        print("  Skipping OOD analysis (no other datasets)")
        return

    # Average across subjects
    if df["subject"].nunique() > 1:
        plot_df = df.groupby("model").agg({
            "veRSA_OOD": "mean",
            "veRSA": "mean",
            **{col: "mean" for col in other_cols}
        }).reset_index()
    else:
        plot_df = df[["model", "veRSA_OOD", "veRSA"] + other_cols].copy()

    # Compute mean of non-OOD datasets
    plot_df["veRSA_in_distribution"] = plot_df[other_cols].mean(axis=1)
    plot_df["ood_advantage"] = plot_df["veRSA_OOD"] - plot_df["veRSA_in_distribution"]
    plot_df["model_short"] = plot_df["model"].apply(shorten_model_name)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Scatter: OOD vs in-distribution
    ax = axes[0]
    ax.scatter(plot_df["veRSA_in_distribution"], plot_df["veRSA_OOD"], alpha=0.7, s=60)

    # Add diagonal
    lims = [0, max(plot_df["veRSA_in_distribution"].max(), plot_df["veRSA_OOD"].max()) * 1.1]
    ax.plot(lims, lims, "k--", alpha=0.3, label="y=x")

    r, p = stats.pearsonr(plot_df["veRSA_in_distribution"], plot_df["veRSA_OOD"])
    ax.set_xlabel("veRSA (In-Distribution Mean)")
    ax.set_ylabel("veRSA (OOD)")
    ax.set_title(f"OOD vs In-Distribution Performance\nr={r:.3f}")
    ax.legend()

    # OOD advantage bar plot
    ax = axes[1]
    sorted_df = plot_df.sort_values("ood_advantage", ascending=True)
    colors = ["green" if a > 0 else "red" for a in sorted_df["ood_advantage"]]
    ax.barh(range(len(sorted_df)), sorted_df["ood_advantage"], color=colors, alpha=0.7)
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df["model_short"], fontsize=8)
    ax.axvline(0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("OOD Advantage (OOD - In-Distribution)")
    ax.set_title("OOD Generalization Advantage\n(positive = better on OOD)")

    # Histogram
    ax = axes[2]
    ax.hist(plot_df["ood_advantage"], bins=15, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--")
    ax.axvline(plot_df["ood_advantage"].mean(), color="blue", linestyle="--",
               label=f"Mean: {plot_df['ood_advantage'].mean():.3f}")
    ax.set_xlabel("OOD Advantage")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of OOD Advantage")
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_dir / "ood_analysis.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "ood_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: ood_analysis.pdf/png")


def plot_alpha_vs_performance(df: pd.DataFrame, out_dir: Path):
    """Analyze relationship between optimal alpha and performance."""
    if "alpha_median" not in df.columns or df["alpha_median"].isna().all():
        print("  Skipping alpha vs performance (no alpha data)")
        return

    df = add_model_metadata(df)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Scatter: alpha vs veRSA
    ax = axes[0]
    scatter = ax.scatter(df["alpha_median"], df["veRSA"],
                        c=df["objective"].astype("category").cat.codes,
                        alpha=0.7, s=60, cmap="tab10")
    ax.set_xscale("log")
    ax.set_xlabel("Optimal Alpha (median)")
    ax.set_ylabel("veRSA")

    r, p = stats.spearmanr(np.log10(df["alpha_median"]), df["veRSA"])
    ax.set_title(f"Alpha vs Performance\nSpearman r={r:.3f}, p={p:.3g}")

    # Alpha vs voxel_r
    ax = axes[1]
    ax.scatter(df["alpha_median"], df["voxel_r_median"], alpha=0.7, s=60)
    ax.set_xscale("log")
    ax.set_xlabel("Optimal Alpha (median)")
    ax.set_ylabel("Voxel-wise r (median)")

    r, p = stats.spearmanr(np.log10(df["alpha_median"]), df["voxel_r_median"])
    ax.set_title(f"Alpha vs Voxel Correlation\nSpearman r={r:.3f}, p={p:.3g}")

    # Alpha distribution by training objective
    ax = axes[2]
    plot_df = df[["objective", "alpha_median"]].dropna()
    if plot_df["objective"].nunique() > 1:
        order = plot_df.groupby("objective")["alpha_median"].median().sort_values().index
        sns.boxplot(data=plot_df, x="objective", y="alpha_median", order=order, ax=ax)
        ax.set_yscale("log")
        ax.set_xlabel("Training Objective")
        ax.set_ylabel("Optimal Alpha")
        ax.set_title("Regularization Strength by Objective")
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.text(0.5, 0.5, "Need multiple objectives", ha="center", va="center",
               transform=ax.transAxes)

    plt.tight_layout()
    fig.savefig(out_dir / "alpha_vs_performance.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "alpha_vs_performance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: alpha_vs_performance.pdf/png")


def plot_radar_profiles(df: pd.DataFrame, out_dir: Path):
    """Radar/spider plot showing model profiles across datasets."""
    dataset_cols = sorted([c for c in df.columns if c.startswith("veRSA_")
                          and c not in ["veRSA_overall", "veRSA_mean_across_datasets",
                                       "veRSA_std_across_datasets"]])

    if len(dataset_cols) < 3:
        print("  Skipping radar plot (need at least 3 datasets)")
        return

    # Average across subjects and get top models
    if df["subject"].nunique() > 1:
        model_df = df.groupby("model")[dataset_cols + ["veRSA"]].mean()
    else:
        model_df = df.set_index("model")[dataset_cols + ["veRSA"]]

    # Select top 5 models by overall veRSA
    top_models = model_df.nlargest(5, "veRSA").index.tolist()

    # Rename columns
    datasets = [c.replace("veRSA_", "") for c in dataset_cols]

    # Create radar plot
    angles = np.linspace(0, 2 * np.pi, len(datasets), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = plt.cm.tab10(np.linspace(0, 1, len(top_models)))

    for i, model in enumerate(top_models):
        values = model_df.loc[model, dataset_cols].values.tolist()
        values += values[:1]  # Close the polygon

        ax.plot(angles, values, 'o-', linewidth=2, label=shorten_model_name(model),
               color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(datasets, size=10)
    ax.set_title("Top 5 Models: Dataset Profiles", size=14, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    fig.savefig(out_dir / "radar_profiles.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "radar_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: radar_profiles.pdf/png")


def plot_summary_dashboard(df: pd.DataFrame, out_dir: Path):
    """Create a summary dashboard with key insights."""
    df = add_model_metadata(df)

    dataset_cols = sorted([c for c in df.columns if c.startswith("veRSA_")
                          and c not in ["veRSA_overall", "veRSA_mean_across_datasets",
                                       "veRSA_std_across_datasets"]])

    fig = plt.figure(figsize=(16, 12))

    # 1. Top models bar
    ax1 = fig.add_subplot(2, 3, 1)
    if df["subject"].nunique() > 1:
        model_means = df.groupby("model")["veRSA"].mean().sort_values(ascending=False).head(10)
    else:
        model_means = df.set_index("model")["veRSA"].sort_values(ascending=False).head(10)

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(model_means)))
    ax1.barh(range(len(model_means)), model_means.values[::-1], color=colors[::-1])
    ax1.set_yticks(range(len(model_means)))
    ax1.set_yticklabels([shorten_model_name(m) for m in model_means.index[::-1]], fontsize=9)
    ax1.set_xlabel("veRSA")
    ax1.set_title("Top 10 Models")

    # 2. Training objective comparison
    ax2 = fig.add_subplot(2, 3, 2)
    obj_means = df.groupby("objective")["veRSA"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    ax2.bar(range(len(obj_means)), obj_means["mean"], yerr=obj_means["std"],
           capsize=3, color=plt.cm.Set2(np.linspace(0, 1, len(obj_means))))
    ax2.set_xticks(range(len(obj_means)))
    ax2.set_xticklabels(obj_means.index, rotation=45, ha="right")
    ax2.set_ylabel("veRSA")
    ax2.set_title("By Training Objective")

    # 3. Dataset difficulty
    ax3 = fig.add_subplot(2, 3, 3)
    if dataset_cols:
        dataset_means = df[dataset_cols].mean().sort_values(ascending=False)
        dataset_means.index = [c.replace("veRSA_", "") for c in dataset_means.index]
        ax3.bar(range(len(dataset_means)), dataset_means.values,
               color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(dataset_means))))
        ax3.set_xticks(range(len(dataset_means)))
        ax3.set_xticklabels(dataset_means.index, rotation=45, ha="right")
        ax3.set_ylabel("Mean veRSA")
        ax3.set_title("Dataset Difficulty")
    else:
        ax3.text(0.5, 0.5, "No per-dataset data", ha="center", va="center")

    # 4. veRSA vs voxel_r
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(df["voxel_r_median"], df["veRSA"], alpha=0.6, s=40,
               c=df["objective"].astype("category").cat.codes, cmap="tab10")
    ax4.set_xlabel("Voxel-wise r")
    ax4.set_ylabel("veRSA")
    r, _ = stats.pearsonr(df["voxel_r_median"].dropna(), df["veRSA"].dropna())
    ax4.set_title(f"veRSA vs Voxel-r (r={r:.2f})")

    # 5. Alpha distribution
    ax5 = fig.add_subplot(2, 3, 5)
    if "alpha_median" in df.columns and not df["alpha_median"].isna().all():
        ax5.hist(np.log10(df["alpha_median"].dropna()), bins=20, edgecolor="black", alpha=0.7)
        ax5.set_xlabel("log10(Alpha)")
        ax5.set_ylabel("Count")
        ax5.set_title(f"Optimal Alpha Distribution\nMedian: {df['alpha_median'].median():.0f}")
    else:
        ax5.text(0.5, 0.5, "No alpha data", ha="center", va="center")

    # 6. Key stats text box
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")

    stats_text = f"""
    Summary Statistics
    ══════════════════

    Models evaluated: {df['model'].nunique()}
    Subjects: {df['subject'].nunique()}

    Best model: {model_means.index[0]}
    Best veRSA: {model_means.values[0]:.4f}

    Mean veRSA: {df['veRSA'].mean():.4f}
    Std veRSA: {df['veRSA'].std():.4f}

    Best objective: {obj_means.index[0]}
    """

    if dataset_cols:
        easiest = dataset_means.index[0]
        hardest = dataset_means.index[-1]
        stats_text += f"""
    Easiest dataset: {easiest}
    Hardest dataset: {hardest}
    """

    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle("Encoding Model Benchmark Summary", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "summary_dashboard.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "summary_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: summary_dashboard.pdf/png")


def plot_roi_comparison(df: pd.DataFrame, out_dir: Path):
    """Compare veRSA scores between visual and hlvis ROIs."""
    if "veRSA_hlvis" not in df.columns or df["veRSA_hlvis"].isna().all():
        print("  Skipping ROI comparison (no hlvis data)")
        return

    # Use veRSA_visual if available, otherwise fall back to veRSA
    visual_col = "veRSA_visual" if "veRSA_visual" in df.columns else "veRSA"

    # Average across subjects if multiple
    if df["subject"].nunique() > 1:
        plot_df = df.groupby("model").agg({
            visual_col: "mean",
            "veRSA_hlvis": "mean",
        }).reset_index()
    else:
        plot_df = df[["model", visual_col, "veRSA_hlvis"]].copy()

    plot_df["model_short"] = plot_df["model"].apply(shorten_model_name)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Scatter: visual vs hlvis
    ax = axes[0]
    ax.scatter(plot_df[visual_col], plot_df["veRSA_hlvis"], alpha=0.7, s=60)

    # Add model labels
    for _, row in plot_df.iterrows():
        ax.annotate(
            row["model_short"][:10],
            (row[visual_col], row["veRSA_hlvis"]),
            fontsize=7, alpha=0.7,
            xytext=(3, 3), textcoords="offset points"
        )

    # Add diagonal
    lims = [
        min(plot_df[visual_col].min(), plot_df["veRSA_hlvis"].min()) - 0.02,
        max(plot_df[visual_col].max(), plot_df["veRSA_hlvis"].max()) + 0.02,
    ]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y=x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Correlation
    r, p = stats.spearmanr(plot_df[visual_col], plot_df["veRSA_hlvis"])
    ax.set_xlabel("veRSA (visual ROI)")
    ax.set_ylabel("veRSA (hlvis ROI)")
    ax.set_title(f"ROI Comparison\nSpearman ρ = {r:.3f}")
    ax.legend()

    # 2. Bar plot comparing ROIs
    ax = axes[1]
    sorted_df = plot_df.sort_values(visual_col, ascending=False)
    x = np.arange(len(sorted_df))
    width = 0.35

    ax.bar(x - width/2, sorted_df[visual_col], width, label="visual", alpha=0.8)
    ax.bar(x + width/2, sorted_df["veRSA_hlvis"], width, label="hlvis", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_df["model_short"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("veRSA")
    ax.set_title("veRSA by Model and ROI")
    ax.legend()

    # 3. ROI difference bar plot
    ax = axes[2]
    plot_df["roi_diff"] = plot_df["veRSA_hlvis"] - plot_df[visual_col]
    sorted_diff = plot_df.sort_values("roi_diff", ascending=True)
    colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in sorted_diff["roi_diff"]]

    ax.barh(range(len(sorted_diff)), sorted_diff["roi_diff"], color=colors, alpha=0.8)
    ax.set_yticks(range(len(sorted_diff)))
    ax.set_yticklabels(sorted_diff["model_short"], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Δ veRSA (hlvis - visual)")
    ax.set_title("ROI Difference\n(green = better on hlvis)")

    plt.tight_layout()
    fig.savefig(out_dir / "roi_comparison.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "roi_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: roi_comparison.pdf/png")


def plot_model_ranking_by_roi(df: pd.DataFrame, out_dir: Path):
    """Bar plot of models ranked by veRSA, showing both ROIs."""
    if "veRSA_hlvis" not in df.columns or df["veRSA_hlvis"].isna().all():
        print("  Skipping ROI ranking (no hlvis data)")
        return

    visual_col = "veRSA_visual" if "veRSA_visual" in df.columns else "veRSA"

    # Average across subjects if multiple
    if df["subject"].nunique() > 1:
        model_df = df.groupby("model").agg({
            visual_col: ["mean", "std"],
            "veRSA_hlvis": ["mean", "std"],
        }).reset_index()
        model_df.columns = ["model", "visual_mean", "visual_std", "hlvis_mean", "hlvis_std"]
    else:
        model_df = df[["model", visual_col, "veRSA_hlvis"]].copy()
        model_df.columns = ["model", "visual_mean", "hlvis_mean"]
        model_df["visual_std"] = 0
        model_df["hlvis_std"] = 0

    # Sort by hlvis score
    model_df = model_df.sort_values("hlvis_mean", ascending=True)
    model_df["model_short"] = model_df["model"].apply(shorten_model_name)

    fig, ax = plt.subplots(figsize=(12, max(6, len(model_df) * 0.5)))

    y_pos = np.arange(len(model_df))
    height = 0.35

    # Visual ROI bars
    ax.barh(y_pos - height/2, model_df["visual_mean"], height,
            xerr=model_df["visual_std"], capsize=2,
            label="visual", alpha=0.7, color="steelblue")

    # hlvis ROI bars
    ax.barh(y_pos + height/2, model_df["hlvis_mean"], height,
            xerr=model_df["hlvis_std"], capsize=2,
            label="hlvis", alpha=0.7, color="darkorange")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_df["model_short"])
    ax.set_xlabel("veRSA")
    ax.set_title("Model Ranking by ROI")
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(out_dir / "model_ranking_by_roi.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "model_ranking_by_roi.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: model_ranking_by_roi.pdf/png")


def compute_voxelwise_r(enc_model, benchmark):
    """Compute per-voxel prediction correlation using odd/even split."""
    from cstims.feature_extraction.universal_extractor import UniversalFeatureExtractor

    # Extract features
    extractor = UniversalFeatureExtractor(
        model_name=enc_model.model_name,
        layer=enc_model.layer,
        source=enc_model.source,
        device="cuda" if Path("/SSD/").exists() else "cpu",
        aggregation="auto",
    )

    image_paths = benchmark.stimulus_data.image_path.tolist()
    features_list = []

    for i in range(0, len(image_paths), 32):
        batch_paths = image_paths[i:i+32]
        from PIL import Image
        batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
        import torch
        batch_tensors = [extractor.preprocess(img) for img in batch_images]
        batch = torch.stack(batch_tensors).to(extractor.device)

        with torch.no_grad():
            feats = extractor.extract(batch)
        if isinstance(feats, torch.Tensor):
            feats = feats.detach().cpu().numpy()
        feats = np.asarray(feats).reshape(feats.shape[0], -1).astype(np.float32)
        features_list.append(feats)

    features = np.vstack(features_list)

    # Get responses
    responses = benchmark.response_data.to_numpy().T  # (n_images, n_voxels)

    # Odd/even split
    X_test = features[1::2]
    Y_test = responses[1::2]

    # Standardize features using stored stats
    X_test_scaled = (X_test - enc_model.feature_mean) / np.maximum(enc_model.feature_scale, 1e-6)

    # Predict
    Y_pred = X_test_scaled @ enc_model.weights + enc_model.intercept

    # Compute per-voxel r
    from scipy.stats import pearsonr
    n_voxels = Y_test.shape[1]
    voxel_r = np.zeros(n_voxels)
    for v in range(n_voxels):
        if np.std(Y_test[:, v]) > 0 and np.std(Y_pred[:, v]) > 0:
            voxel_r[v] = pearsonr(Y_test[:, v], Y_pred[:, v])[0]

    return voxel_r


def plot_brain_maps(df: pd.DataFrame, run_dir: Path, out_dir: Path, top_n: int = 5):
    """Generate brain maps showing voxel-wise prediction scores.

    Creates:
    1. Voxel-wise prediction accuracy (r) for top models
    2. Comparison of best vs worst model
    3. Difference map showing where models differ
    """
    import json
    from cstims.encoding.model import LinearEncodingModel
    from cstims.datasets.deepvision import DeepVisionBenchmark

    print("  Generating brain maps...")

    # Find model directories
    model_dirs = []
    for subdir in run_dir.iterdir():
        if not subdir.is_dir():
            continue
        npz_file = subdir / "encoding_model.npz"
        if not npz_file.exists():
            continue

        # Try metadata.json first, fall back to npz metadata
        meta_file = subdir / "metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
        else:
            # Load metadata from npz file
            try:
                with np.load(npz_file, allow_pickle=True) as z:
                    if "metadata" in z:
                        metadata_arr = z["metadata"]
                        metadata_str = metadata_arr.item() if hasattr(metadata_arr, 'item') else str(metadata_arr)
                        meta = json.loads(metadata_str)
                    else:
                        continue
            except Exception as e:
                print(f"    Warning: Failed to load {npz_file}: {e}")
                continue

        model_dirs.append({
            "dir": subdir,
            "npz": npz_file,
            "model": meta.get("model_name", meta.get("model", "unknown")),
            "subject": meta.get("subject", "unknown"),
            "voxel_set": meta.get("voxel_set", "visual"),
            "veRSA": meta.get("metrics", {}).get("veRSA_pearson_r", 0),
            "voxel_r_median": meta.get("metrics", {}).get("voxel_r_median", 0),
        })

    if not model_dirs:
        print("    No encoding models found for brain maps")
        return

    # Sort by veRSA
    model_dirs = sorted(model_dirs, key=lambda x: x["veRSA"] or 0, reverse=True)
    print(f"    Found {len(model_dirs)} encoding models")

    # Group by subject
    subjects = sorted(set(m["subject"] for m in model_dirs))

    from cstims.paths import deepvision_fmri_root

    cache_root = Path(
        os.environ.get(
            "CSTIMS_DEEPVISION_CACHE_ROOT",
            PROJECT_ROOT / "01_brain_model_alignment/cache_or_heavy/brain_data",
        )
    )
    deepvision_root = deepvision_fmri_root()

    for subject in subjects:
        subj_models = [m for m in model_dirs if m["subject"] == subject]
        if len(subj_models) < 2:
            continue

        # Get top and bottom model
        best_model = subj_models[0]
        worst_model = subj_models[-1]

        try:
            best_enc = LinearEncodingModel.load(best_model["npz"])
            worst_enc = LinearEncodingModel.load(worst_model["npz"])
        except Exception as e:
            print(f"    Error loading models for {subject}: {e}")
            continue

        # Load benchmark for computing voxel-wise r
        print(f"    Computing voxel-wise r for {subject}...")
        try:
            benchmark = DeepVisionBenchmark(
                cache_root=cache_root,
                deepvision_fmri_root=deepvision_root,
                subject=subject,
                voxel_set=best_model.get("voxel_set", "visual"),
                build_rdms=False,
            )

            best_voxel_r = compute_voxelwise_r(best_enc, benchmark)
            worst_voxel_r = compute_voxelwise_r(worst_enc, benchmark)

            # Create brain volumes for voxel-wise r
            best_r_vol = best_enc.to_volume(best_voxel_r, fill_value=np.nan)
            worst_r_vol = worst_enc.to_volume(worst_voxel_r, fill_value=np.nan)
            diff_r_vol = best_enc.to_volume(best_voxel_r - worst_voxel_r, fill_value=np.nan)

            # Save as NIfTI for interactive viewing
            nifti_dir = out_dir / "nifti"
            nifti_dir.mkdir(exist_ok=True)

            best_enc.to_nifti(best_voxel_r, nifti_dir / f"voxelr_best_{subject}_{shorten_model_name(best_model['model'])}.nii.gz", fill_value=0)
            worst_enc.to_nifti(worst_voxel_r, nifti_dir / f"voxelr_worst_{subject}_{shorten_model_name(worst_model['model'])}.nii.gz", fill_value=0)
            best_enc.to_nifti(best_voxel_r - worst_voxel_r, nifti_dir / f"voxelr_diff_{subject}_best_minus_worst.nii.gz", fill_value=0)
            print(f"    Saved NIfTI files to {nifti_dir}")

        except Exception as e:
            print(f"    Could not compute voxel-wise r: {e}")
            print("    Falling back to alpha visualization...")
            best_r_vol = best_enc.to_volume(np.log10(best_enc.alphas + 1e-6), fill_value=np.nan)
            worst_r_vol = worst_enc.to_volume(np.log10(worst_enc.alphas + 1e-6), fill_value=np.nan)
            diff_r_vol = None

        # Find good slices (with data)
        vol_shape = best_r_vol.shape
        z_mid = vol_shape[2] // 2
        y_mid = vol_shape[1] // 2
        x_mid = vol_shape[0] // 2

        # Find slices with most non-nan voxels
        best_mask = ~np.isnan(best_r_vol)
        z_counts = best_mask.sum(axis=(0, 1))
        y_counts = best_mask.sum(axis=(0, 2))
        x_counts = best_mask.sum(axis=(1, 2))

        z_best = np.argmax(z_counts) if z_counts.max() > 0 else z_mid
        y_best = np.argmax(y_counts) if y_counts.max() > 0 else y_mid
        x_best = np.argmax(x_counts) if x_counts.max() > 0 else x_mid

        # Determine colormap limits
        vmin = min(np.nanmin(best_r_vol), np.nanmin(worst_r_vol))
        vmax = max(np.nanmax(best_r_vol), np.nanmax(worst_r_vol))

        # Plot comparison figure: voxel-wise prediction accuracy
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))

        # Best model - 3 views
        ax = axes[0, 0]
        slice_data = best_r_vol[:, :, z_best].T
        im = ax.imshow(slice_data, cmap="RdYlGn", origin="lower", aspect="equal",
                       vmin=vmin, vmax=vmax)
        ax.set_title(f"Best: {shorten_model_name(best_model['model'])}\nAxial (z={z_best})")
        ax.axis("off")
        plt.colorbar(im, ax=ax, label="Voxel r")

        ax = axes[0, 1]
        slice_data = best_r_vol[:, y_best, :].T
        ax.imshow(slice_data, cmap="RdYlGn", origin="lower", aspect="equal",
                  vmin=vmin, vmax=vmax)
        ax.set_title(f"Coronal (y={y_best})")
        ax.axis("off")

        ax = axes[0, 2]
        slice_data = best_r_vol[x_best, :, :].T
        ax.imshow(slice_data, cmap="RdYlGn", origin="lower", aspect="equal",
                  vmin=vmin, vmax=vmax)
        ax.set_title(f"Sagittal (x={x_best})")
        ax.axis("off")

        # Worst model - 3 views
        ax = axes[1, 0]
        slice_data = worst_r_vol[:, :, z_best].T
        im = ax.imshow(slice_data, cmap="RdYlGn", origin="lower", aspect="equal",
                       vmin=vmin, vmax=vmax)
        ax.set_title(f"Worst: {shorten_model_name(worst_model['model'])}\nAxial (z={z_best})")
        ax.axis("off")
        plt.colorbar(im, ax=ax, label="Voxel r")

        ax = axes[1, 1]
        slice_data = worst_r_vol[:, y_best, :].T
        ax.imshow(slice_data, cmap="RdYlGn", origin="lower", aspect="equal",
                  vmin=vmin, vmax=vmax)
        ax.set_title(f"Coronal (y={y_best})")
        ax.axis("off")

        ax = axes[1, 2]
        slice_data = worst_r_vol[x_best, :, :].T
        ax.imshow(slice_data, cmap="RdYlGn", origin="lower", aspect="equal",
                  vmin=vmin, vmax=vmax)
        ax.set_title(f"Sagittal (x={x_best})")
        ax.axis("off")

        fig.suptitle(
            f"Voxel-wise prediction accuracy (r): {subject}\n"
            f"Best: veRSA={best_model['veRSA']:.3f}, r_med={best_model['voxel_r_median']:.3f} | "
            f"Worst: veRSA={worst_model['veRSA']:.3f}, r_med={worst_model['voxel_r_median']:.3f}",
            fontsize=11
        )
        plt.tight_layout()

        out_path = out_dir / f"brain_voxelr_comparison_{subject}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {out_path.name}")

        # Plot difference map (best - worst)
        if diff_r_vol is not None:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))

            diff_max = np.nanmax(np.abs(diff_r_vol))

            ax = axes[0]
            slice_data = diff_r_vol[:, :, z_best].T
            im = ax.imshow(slice_data, cmap="RdBu_r", origin="lower", aspect="equal",
                           vmin=-diff_max, vmax=diff_max)
            ax.set_title(f"Axial (z={z_best})")
            ax.axis("off")
            plt.colorbar(im, ax=ax, label="Δr (best - worst)")

            ax = axes[1]
            slice_data = diff_r_vol[:, y_best, :].T
            ax.imshow(slice_data, cmap="RdBu_r", origin="lower", aspect="equal",
                      vmin=-diff_max, vmax=diff_max)
            ax.set_title(f"Coronal (y={y_best})")
            ax.axis("off")

            ax = axes[2]
            slice_data = diff_r_vol[x_best, :, :].T
            ax.imshow(slice_data, cmap="RdBu_r", origin="lower", aspect="equal",
                      vmin=-diff_max, vmax=diff_max)
            ax.set_title(f"Sagittal (x={x_best})")
            ax.axis("off")

            fig.suptitle(
                f"Difference in voxel-wise r: {shorten_model_name(best_model['model'])} − "
                f"{shorten_model_name(worst_model['model'])}\n(Red = best model better)",
                fontsize=11
            )
            plt.tight_layout()

            out_path = out_dir / f"brain_voxelr_difference_{subject}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved: {out_path.name}")

        # Create mosaic of top N models showing voxel r (single slice)
        # Skip if voxel-wise r computation failed (we'd need benchmark for each model)
        print(f"    Creating mosaic for top {top_n} models...")

        top_models = subj_models[:min(top_n, len(subj_models))]
        n_models = len(top_models)
        n_cols = min(5, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else list(axes)
        else:
            axes = list(axes.flatten())

        for i, model_info in enumerate(top_models):
            ax = axes[i]

            try:
                enc = LinearEncodingModel.load(model_info["npz"])
                # Use alphas as proxy for performance (log scale)
                alpha_vol = enc.to_volume(-np.log10(enc.alphas + 1e-6))  # Invert: higher = better predicted
                slice_data = alpha_vol[:, :, z_best].T

                im = ax.imshow(slice_data, cmap="viridis", origin="lower", aspect="equal")
                ax.set_title(f"{shorten_model_name(model_info['model'])[:15]}\nveRSA={model_info['veRSA']:.3f}", fontsize=9)
                ax.axis("off")
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"Top {n_models} models - inverse log(α) at z={z_best}: {subject}\n(brighter = lower regularization = easier to predict)", fontsize=10)
        plt.tight_layout()

        out_path = out_dir / f"brain_alpha_mosaic_{subject}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {out_path.name}")

    print("  Brain maps complete!")


def main():
    parser = argparse.ArgumentParser(description="Plot encoding model results")
    parser.add_argument("run_dir", nargs="?", default=None, help="Path to run directory")
    parser.add_argument("--brain-maps", action="store_true", help="Generate brain map visualizations")
    args = parser.parse_args()

    # Determine run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        script_dir = Path(__file__).parent
        results_root = script_dir / "results"
        if not results_root.exists():
            print(f"Results directory not found: {results_root}")
            sys.exit(1)
        run_dir = find_latest_run(results_root)

    print(f"Plotting results from: {run_dir}")
    print("=" * 60)

    # Load combined results
    csv_path = run_dir / "combined_results.csv"
    if not csv_path.exists():
        print(f"Combined results not found: {csv_path}")
        print("Run combine_results.py first!")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} results ({df['model'].nunique()} models, "
          f"{df['subject'].nunique()} subjects)")

    # Create figures directory
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    print("\nGenerating plots...")

    # Basic plots
    plot_model_ranking(df, fig_dir)
    plot_per_dataset_breakdown(df, fig_dir)
    plot_versa_vs_voxelr(df, fig_dir)
    plot_cross_dataset_heatmap(df, fig_dir)
    plot_alpha_distribution(df, fig_dir)
    plot_alpha_stability(df, fig_dir)
    plot_subject_consistency(df, fig_dir)

    # Advanced analyses
    plot_training_objective_comparison(df, fig_dir)
    plot_generalization_gap(df, fig_dir)
    plot_dataset_difficulty(df, fig_dir)
    plot_dataset_agreement(df, fig_dir)
    plot_ood_analysis(df, fig_dir)
    plot_alpha_vs_performance(df, fig_dir)
    plot_radar_profiles(df, fig_dir)

    # ROI comparison (if hlvis data available)
    plot_roi_comparison(df, fig_dir)
    plot_model_ranking_by_roi(df, fig_dir)

    # Summary dashboard (do last)
    plot_summary_dashboard(df, fig_dir)

    # Brain maps (if requested)
    if args.brain_maps:
        plot_brain_maps(df, run_dir, fig_dir)

    print(f"\nAll figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
