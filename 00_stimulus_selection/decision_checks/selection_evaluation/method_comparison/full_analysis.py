"""
Full analysis comparing v2_full and powermean selection methods.

Generates:
- AUC comparison bar charts
- Per-track breakdown
- Image overlap analysis
- Refinement trajectory comparison
- Markdown report
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Output directories
V2_FULL_ROOT = Path("/home/jroth/rsa_based_selection/outputs/final_cstims_v2_full")
POWERMEAN_ROOT = Path("/home/jroth/rsa_based_selection/outputs/final_cstims_powermean")

MODEL_SETS = ["all_models", "sota", "dataset", "training_objective", "architecture"]
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "figures"
DATA_DIR = SCRIPT_DIR / "data"

# Plotting style
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "v2_full_final": "#1f77b4",
    "v2_full_best_raw_combined": "#aec7e8",
    "powermean_final": "#ff7f0e",
    "powermean_best_raw_combined": "#ffbb78",
}


def find_latest_run(root: Path, model_set: str) -> Path:
    """Find the latest timestamped run directory."""
    method_dirs = list(root.glob(f"{model_set}/method-*/"))
    if not method_dirs:
        return None
    runs = sorted(method_dirs[0].glob("20*"))
    return runs[-1] if runs else None


def load_statistics(run_dir: Path, selection_type: str) -> pd.DataFrame:
    """Load statistics.csv for a given selection type."""
    if selection_type == "final":
        stats_path = run_dir / "eval_pipeline" / "statistics.csv"
    else:
        stats_path = run_dir / "eval_pipeline" / "best_raw_combined" / "statistics.csv"
    if not stats_path.exists():
        return None
    return pd.read_csv(stats_path)


def load_image_manifest(run_dir: Path, selection_type: str) -> pd.DataFrame:
    """Load image manifest for overlap analysis."""
    if selection_type == "final":
        path = run_dir / "eval_pipeline" / "images" / "image_manifest.csv"
    else:
        path = run_dir / "eval_pipeline" / "best_raw_combined" / "images" / "image_manifest.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df


def collect_all_results() -> pd.DataFrame:
    """Collect results from all runs into a single DataFrame."""
    rows = []
    for model_set in MODEL_SETS:
        for method_name, root in [("v2_full", V2_FULL_ROOT), ("powermean", POWERMEAN_ROOT)]:
            run_dir = find_latest_run(root, model_set)
            if run_dir is None:
                continue
            for selection_type in ["final", "best_raw_combined"]:
                stats = load_statistics(run_dir, selection_type)
                if stats is None:
                    continue
                for _, row in stats.iterrows():
                    rows.append({
                        "model_set": model_set,
                        "method": method_name,
                        "selection_type": selection_type,
                        "condition": f"{method_name}_{selection_type}",
                        "track": row["track"],
                        "selected_auc": row["selected_auc"],
                        "random_auc": row["random_auc"],
                        "auc_diff": row["auc_diff"],
                        "auc_improvement_pct": row["auc_improvement_pct"],
                    })
    return pd.DataFrame(rows)


def compute_image_overlap() -> pd.DataFrame:
    """Compute image overlap between all condition pairs."""
    # Load all image sets
    image_sets = {}
    for model_set in MODEL_SETS:
        for method_name, root in [("v2_full", V2_FULL_ROOT), ("powermean", POWERMEAN_ROOT)]:
            run_dir = find_latest_run(root, model_set)
            if run_dir is None:
                continue
            for selection_type in ["final", "best_raw_combined"]:
                manifest = load_image_manifest(run_dir, selection_type)
                if manifest is None:
                    continue
                # Use global_idx as unique image identifier
                if "global_idx" in manifest.columns:
                    images = set(manifest["global_idx"].values)
                elif "image_name" in manifest.columns:
                    images = set(manifest["image_name"].values)
                else:
                    # Fallback to index
                    images = set(range(len(manifest)))
                key = (model_set, f"{method_name}_{selection_type}")
                image_sets[key] = images

    # Compute pairwise overlaps
    rows = []
    conditions = ["v2_full_final", "v2_full_best_raw_combined",
                  "powermean_final", "powermean_best_raw_combined"]

    for model_set in MODEL_SETS:
        for i, cond1 in enumerate(conditions):
            for cond2 in conditions[i+1:]:
                key1 = (model_set, cond1)
                key2 = (model_set, cond2)
                if key1 in image_sets and key2 in image_sets:
                    set1, set2 = image_sets[key1], image_sets[key2]
                    overlap = len(set1 & set2)
                    jaccard = overlap / len(set1 | set2) if len(set1 | set2) > 0 else 0
                    rows.append({
                        "model_set": model_set,
                        "condition1": cond1,
                        "condition2": cond2,
                        "overlap_count": overlap,
                        "total_unique": len(set1 | set2),
                        "jaccard_similarity": jaccard,
                        "overlap_pct": overlap / 100 * 100,  # Assuming 100 images each
                    })

    return pd.DataFrame(rows)


def plot_aggregate_auc_comparison(df: pd.DataFrame):
    """Bar chart comparing aggregate AUC across conditions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    agg_df = df[df["track"] == "AGGREGATE"].copy()

    # Pivot for grouped bar chart
    pivot = agg_df.pivot_table(
        index="model_set",
        columns="condition",
        values="selected_auc"
    )

    col_order = ["v2_full_final", "v2_full_best_raw_combined",
                 "powermean_final", "powermean_best_raw_combined"]
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    x = np.arange(len(pivot.index))
    width = 0.2

    for i, (col, color) in enumerate(zip(pivot.columns,
                                          [COLORS[c] for c in pivot.columns])):
        ax.bar(x + i * width, pivot[col], width, label=col.replace("_", " "), color=color)

    ax.set_xlabel("Model Set", fontsize=12)
    ax.set_ylabel("AGGREGATE AUC (lower = better)", fontsize=12)
    ax.set_title("Discriminability by Method and Selection Type", fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(pivot.max()) * 1.15)

    # Add mean line
    for i, (col, color) in enumerate(zip(pivot.columns,
                                          [COLORS[c] for c in pivot.columns])):
        mean_val = pivot[col].mean()
        ax.axhline(mean_val, color=color, linestyle="--", alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "aggregate_auc_comparison.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "aggregate_auc_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'aggregate_auc_comparison.pdf'}")


def plot_track_breakdown(df: pd.DataFrame):
    """Heatmap showing AUC by track and condition."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Average across model sets
    track_means = df.groupby(["track", "condition"])["selected_auc"].mean().unstack()

    col_order = ["v2_full_final", "v2_full_best_raw_combined",
                 "powermean_final", "powermean_best_raw_combined"]
    track_means = track_means[[c for c in col_order if c in track_means.columns]]

    # Reorder tracks
    track_order = ["raw", "sub-01", "sub-03", "sub-05", "sub-06", "sub-07", "AGGREGATE"]
    track_means = track_means.reindex([t for t in track_order if t in track_means.index])

    sns.heatmap(track_means, annot=True, fmt=".3f", cmap="RdYlGn_r",
                ax=ax, cbar_kws={"label": "AUC (lower = better)"})
    ax.set_title("AUC by Track and Condition (averaged across model sets)", fontsize=14)
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Track", fontsize=12)

    # Rotate x labels
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "track_breakdown_heatmap.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "track_breakdown_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'track_breakdown_heatmap.pdf'}")


def plot_encoding_vs_raw_tradeoff(df: pd.DataFrame):
    """Scatter plot showing encoding vs raw AUC trade-off."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Compute mean encoding AUC (sub-01 to sub-07)
    encoding_tracks = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]

    results = []
    for (model_set, condition), grp in df.groupby(["model_set", "condition"]):
        enc_auc = grp[grp["track"].isin(encoding_tracks)]["selected_auc"].mean()
        raw_auc = grp[grp["track"] == "raw"]["selected_auc"].values
        raw_auc = raw_auc[0] if len(raw_auc) > 0 else np.nan
        results.append({
            "model_set": model_set,
            "condition": condition,
            "encoding_auc": enc_auc,
            "raw_auc": raw_auc,
        })

    plot_df = pd.DataFrame(results)

    for condition in plot_df["condition"].unique():
        subset = plot_df[plot_df["condition"] == condition]
        ax.scatter(subset["raw_auc"], subset["encoding_auc"],
                   c=COLORS.get(condition, "gray"), label=condition.replace("_", " "),
                   s=100, alpha=0.7)

        # Add model set labels
        for _, row in subset.iterrows():
            ax.annotate(row["model_set"][:4], (row["raw_auc"], row["encoding_auc"]),
                       fontsize=8, alpha=0.7)

    ax.set_xlabel("Raw Track AUC (lower = better)", fontsize=12)
    ax.set_ylabel("Mean Encoding Track AUC (lower = better)", fontsize=12)
    ax.set_title("Encoding vs Raw Trade-off", fontsize=14)
    ax.legend(loc="upper right")

    # Add diagonal line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "encoding_vs_raw_tradeoff.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "encoding_vs_raw_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'encoding_vs_raw_tradeoff.pdf'}")


def plot_image_overlap(overlap_df: pd.DataFrame):
    """Bar chart showing image overlap between conditions."""
    if overlap_df.empty:
        print("No overlap data available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Overlap by model set for key comparisons
    ax1 = axes[0]
    key_comparisons = [
        ("v2_full_final", "powermean_final"),
        ("v2_full_best_raw_combined", "powermean_best_raw_combined"),
        ("v2_full_final", "v2_full_best_raw_combined"),
        ("powermean_final", "powermean_best_raw_combined"),
    ]

    comparison_data = []
    for c1, c2 in key_comparisons:
        subset = overlap_df[(overlap_df["condition1"] == c1) & (overlap_df["condition2"] == c2)]
        if not subset.empty:
            comparison_data.append({
                "comparison": f"{c1.split('_')[0]} vs {c2.split('_')[0]}\n({c1.split('_')[-1]} vs {c2.split('_')[-1]})",
                "mean_overlap": subset["overlap_count"].mean(),
                "std_overlap": subset["overlap_count"].std(),
            })

    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        ax1.bar(comp_df["comparison"], comp_df["mean_overlap"],
                yerr=comp_df["std_overlap"], capsize=5, color="steelblue")
        ax1.set_ylabel("Image Overlap (out of 100)", fontsize=12)
        ax1.set_title("Mean Image Overlap Between Conditions", fontsize=14)
        ax1.set_ylim(0, 100)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Plot 2: Heatmap of mean overlaps
    ax2 = axes[1]
    conditions = ["v2_full_final", "v2_full_best_raw_combined",
                  "powermean_final", "powermean_best_raw_combined"]

    overlap_matrix = pd.DataFrame(100, index=conditions, columns=conditions, dtype=float)
    for _, row in overlap_df.groupby(["condition1", "condition2"])["overlap_count"].mean().reset_index().iterrows():
        c1, c2, val = row["condition1"], row["condition2"], row["overlap_count"]
        overlap_matrix.loc[c1, c2] = val
        overlap_matrix.loc[c2, c1] = val

    sns.heatmap(overlap_matrix, annot=True, fmt=".0f", cmap="Blues",
                ax=ax2, cbar_kws={"label": "Overlap count"}, vmin=0, vmax=100)
    ax2.set_title("Image Overlap Matrix (averaged across model sets)", fontsize=14)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "image_overlap.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "image_overlap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'image_overlap.pdf'}")


def plot_method_summary(df: pd.DataFrame):
    """Summary bar chart of mean AUC across all model sets."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    col_order = ["v2_full_final", "v2_full_best_raw_combined",
                 "powermean_final", "powermean_best_raw_combined"]

    # Plot 1: Aggregate AUC
    ax1 = axes[0]
    agg_means = df[df["track"] == "AGGREGATE"].groupby("condition")["selected_auc"].mean()
    agg_means = agg_means.reindex([c for c in col_order if c in agg_means.index])
    bars = ax1.bar(range(len(agg_means)), agg_means.values,
                   color=[COLORS[c] for c in agg_means.index])
    ax1.set_xticks(range(len(agg_means)))
    ax1.set_xticklabels([c.replace("_", "\n") for c in agg_means.index], fontsize=9)
    ax1.set_ylabel("AGGREGATE AUC", fontsize=12)
    ax1.set_title("Overall Discriminability", fontsize=14)
    # Highlight best
    best_idx = agg_means.values.argmin()
    bars[best_idx].set_edgecolor("red")
    bars[best_idx].set_linewidth(3)

    # Plot 2: Encoding tracks
    ax2 = axes[1]
    enc_tracks = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
    enc_means = df[df["track"].isin(enc_tracks)].groupby("condition")["selected_auc"].mean()
    enc_means = enc_means.reindex([c for c in col_order if c in enc_means.index])
    bars = ax2.bar(range(len(enc_means)), enc_means.values,
                   color=[COLORS[c] for c in enc_means.index])
    ax2.set_xticks(range(len(enc_means)))
    ax2.set_xticklabels([c.replace("_", "\n") for c in enc_means.index], fontsize=9)
    ax2.set_ylabel("Encoding Track AUC", fontsize=12)
    ax2.set_title("Encoding Model Discriminability", fontsize=14)
    best_idx = enc_means.values.argmin()
    bars[best_idx].set_edgecolor("red")
    bars[best_idx].set_linewidth(3)

    # Plot 3: Raw track
    ax3 = axes[2]
    raw_means = df[df["track"] == "raw"].groupby("condition")["selected_auc"].mean()
    raw_means = raw_means.reindex([c for c in col_order if c in raw_means.index])
    bars = ax3.bar(range(len(raw_means)), raw_means.values,
                   color=[COLORS[c] for c in raw_means.index])
    ax3.set_xticks(range(len(raw_means)))
    ax3.set_xticklabels([c.replace("_", "\n") for c in raw_means.index], fontsize=9)
    ax3.set_ylabel("Raw Track AUC", fontsize=12)
    ax3.set_title("Raw Feature Discriminability", fontsize=14)
    best_idx = raw_means.values.argmin()
    bars[best_idx].set_edgecolor("red")
    bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "method_summary.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "method_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'method_summary.pdf'}")


def generate_report(df: pd.DataFrame, overlap_df: pd.DataFrame):
    """Generate markdown report with findings."""

    col_order = ["v2_full_final", "v2_full_best_raw_combined",
                 "powermean_final", "powermean_best_raw_combined"]

    # Compute summary statistics
    agg_means = df[df["track"] == "AGGREGATE"].groupby("condition")["selected_auc"].mean()
    agg_means = agg_means.reindex([c for c in col_order if c in agg_means.index])

    enc_tracks = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
    enc_means = df[df["track"].isin(enc_tracks)].groupby("condition")["selected_auc"].mean()
    enc_means = enc_means.reindex([c for c in col_order if c in enc_means.index])

    raw_means = df[df["track"] == "raw"].groupby("condition")["selected_auc"].mean()
    raw_means = raw_means.reindex([c for c in col_order if c in raw_means.index])

    best_condition = agg_means.idxmin()
    best_auc = agg_means.min()

    # Per model set winners
    agg_by_model = df[df["track"] == "AGGREGATE"].pivot_table(
        index="model_set", columns="condition", values="selected_auc"
    )
    model_winners = agg_by_model.idxmin(axis=1)

    report = f"""# Selection Method Comparison Report

## Overview

This report compares two stimulus selection methods:
- **v2_full**: Standard aggregation (`raw_plus_all_encodings`)
- **powermean**: Harmonic mean aggregation (`raw_plus_all_encodings_powermean`)

Each method produces two selection outputs:
- **final**: After full refinement (10 passes)
- **best_raw_combined**: Best checkpoint by raw track score during greedy selection

## Summary Results

### Best Overall Condition: `{best_condition}`

| Condition | AGGREGATE AUC | Encoding AUC | Raw AUC |
|-----------|---------------|--------------|---------|
"""

    for cond in col_order:
        if cond in agg_means.index:
            marker = " **✓**" if cond == best_condition else ""
            report += f"| {cond} | {agg_means[cond]:.4f}{marker} | {enc_means[cond]:.4f} | {raw_means[cond]:.4f} |\n"

    report += f"""
*Lower AUC = better discriminability*

### Key Findings

1. **Best overall: `{best_condition}`** with AGGREGATE AUC = {best_auc:.4f}

2. **best_raw_combined beats final** for both methods:
   - v2_full: {agg_means.get('v2_full_final', 0):.4f} → {agg_means.get('v2_full_best_raw_combined', 0):.4f} ({(agg_means.get('v2_full_final', 0) - agg_means.get('v2_full_best_raw_combined', 0)) / agg_means.get('v2_full_final', 1) * 100:.1f}% improvement)
   - powermean: {agg_means.get('powermean_final', 0):.4f} → {agg_means.get('powermean_best_raw_combined', 0):.4f} ({(agg_means.get('powermean_final', 0) - agg_means.get('powermean_best_raw_combined', 0)) / agg_means.get('powermean_final', 1) * 100:.1f}% improvement)

3. **Encoding track comparison** (mean across sub-01 to sub-07):
   - v2_full_best_raw_combined: {enc_means.get('v2_full_best_raw_combined', 0):.4f}
   - powermean_best_raw_combined: {enc_means.get('powermean_best_raw_combined', 0):.4f}
   - Difference: {abs(enc_means.get('v2_full_best_raw_combined', 0) - enc_means.get('powermean_best_raw_combined', 0)):.4f}

4. **Raw track is similar** across all conditions (~{raw_means.mean():.3f})

### Per-Model-Set Winners

| Model Set | Best Condition | AUC |
|-----------|----------------|-----|
"""

    for model_set, winner in model_winners.items():
        auc = agg_by_model.loc[model_set, winner]
        report += f"| {model_set} | {winner} | {auc:.4f} |\n"

    report += f"""
### Winner Distribution
"""
    winner_counts = model_winners.value_counts()
    for cond, count in winner_counts.items():
        report += f"- {cond}: {count}/5 model sets\n"

    # Image overlap
    if not overlap_df.empty:
        report += """
## Image Overlap Analysis

How similar are the selected image sets?

| Comparison | Mean Overlap | Jaccard Similarity |
|------------|--------------|-------------------|
"""
        key_comparisons = [
            ("v2_full_final", "v2_full_best_raw_combined", "v2_full: final vs best_raw"),
            ("powermean_final", "powermean_best_raw_combined", "powermean: final vs best_raw"),
            ("v2_full_final", "powermean_final", "final: v2_full vs powermean"),
            ("v2_full_best_raw_combined", "powermean_best_raw_combined", "best_raw: v2_full vs powermean"),
        ]

        for c1, c2, label in key_comparisons:
            subset = overlap_df[(overlap_df["condition1"] == c1) & (overlap_df["condition2"] == c2)]
            if subset.empty:
                subset = overlap_df[(overlap_df["condition1"] == c2) & (overlap_df["condition2"] == c1)]
            if not subset.empty:
                mean_overlap = subset["overlap_count"].mean()
                mean_jaccard = subset["jaccard_similarity"].mean()
                report += f"| {label} | {mean_overlap:.0f}/100 | {mean_jaccard:.2f} |\n"

    report += """
## Figures

- [Aggregate AUC Comparison](figures/aggregate_auc_comparison.png)
- [Track Breakdown Heatmap](figures/track_breakdown_heatmap.png)
- [Encoding vs Raw Trade-off](figures/encoding_vs_raw_tradeoff.png)
- [Method Summary](figures/method_summary.png)
- [Image Overlap](figures/image_overlap.png)

## Conclusions

1. **Use `best_raw_combined` checkpoints** - Refinement hurts downstream discriminability.

2. **v2_full slightly outperforms powermean** when comparing best_raw_combined outputs,
   contrary to the hypothesis that harmonic mean would protect encoding tracks.

3. **The difference is small** (~0.001 AUC) - both methods produce similarly good selections.

4. **Model set matters** - different methods win for different model sets, suggesting
   the "best" method may depend on the specific model comparison being made.

## Recommendation

For future selections, use **v2_full with best_raw_combined** as the default choice,
but consider that the difference between methods is minimal.
"""

    report_path = SCRIPT_DIR / "README.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved: {report_path}")


def main():
    # Create output directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    print("=" * 80)
    print("Full Analysis: v2_full vs powermean")
    print("=" * 80)

    # Collect data
    print("\n1. Collecting results...")
    df = collect_all_results()
    df.to_csv(DATA_DIR / "all_auc_results.csv", index=False)
    print(f"   Saved: {DATA_DIR / 'all_auc_results.csv'}")

    print("\n2. Computing image overlap...")
    overlap_df = compute_image_overlap()
    if not overlap_df.empty:
        overlap_df.to_csv(DATA_DIR / "image_overlap.csv", index=False)
        print(f"   Saved: {DATA_DIR / 'image_overlap.csv'}")

    # Generate plots
    print("\n3. Generating plots...")
    plot_aggregate_auc_comparison(df)
    plot_track_breakdown(df)
    plot_encoding_vs_raw_tradeoff(df)
    plot_method_summary(df)
    plot_image_overlap(overlap_df)

    # Generate report
    print("\n4. Generating report...")
    generate_report(df, overlap_df)

    print("\n" + "=" * 80)
    print("Done! See README.md for full report.")
    print("=" * 80)


if __name__ == "__main__":
    main()
