"""
Compare AUC discriminability between v2_full and powermean selection methods.

Compares 4 conditions:
- v2_full + final
- v2_full + best_raw_combined
- powermean + final
- powermean + best_raw_combined
"""

import pandas as pd
from pathlib import Path

# Output directories
V2_FULL_ROOT = Path("/home/jroth/rsa_based_selection/outputs/final_cstims_v2_full")
POWERMEAN_ROOT = Path("/home/jroth/rsa_based_selection/outputs/final_cstims_powermean")

MODEL_SETS = ["all_models", "sota", "dataset", "training_objective", "architecture"]


def find_latest_run(root: Path, model_set: str) -> Path:
    """Find the latest timestamped run directory."""
    method_dir = list(root.glob(f"{model_set}/method-*/"))[0]
    runs = sorted(method_dir.glob("20*"))
    return runs[-1] if runs else None


def load_statistics(run_dir: Path, selection_type: str) -> pd.DataFrame:
    """Load statistics.csv for a given selection type."""
    if selection_type == "final":
        stats_path = run_dir / "eval_pipeline" / "statistics.csv"
    else:  # best_raw_combined
        stats_path = run_dir / "eval_pipeline" / "best_raw_combined" / "statistics.csv"

    if not stats_path.exists():
        return None
    return pd.read_csv(stats_path)


def collect_all_results():
    """Collect results from all runs into a single DataFrame."""
    rows = []

    for model_set in MODEL_SETS:
        for method_name, root in [("v2_full", V2_FULL_ROOT), ("powermean", POWERMEAN_ROOT)]:
            run_dir = find_latest_run(root, model_set)
            if run_dir is None:
                print(f"No run found for {method_name}/{model_set}")
                continue

            for selection_type in ["final", "best_raw_combined"]:
                stats = load_statistics(run_dir, selection_type)
                if stats is None:
                    print(f"No stats for {method_name}/{model_set}/{selection_type}")
                    continue

                for _, row in stats.iterrows():
                    rows.append({
                        "model_set": model_set,
                        "method": method_name,
                        "selection_type": selection_type,
                        "track": row["track"],
                        "selected_auc": row["selected_auc"],
                        "random_auc": row["random_auc"],
                        "auc_diff": row["auc_diff"],
                        "auc_improvement_pct": row["auc_improvement_pct"],
                    })

    return pd.DataFrame(rows)


def main():
    print("=" * 80)
    print("AUC Comparison: v2_full vs powermean")
    print("=" * 80)

    df = collect_all_results()

    # Save raw data
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "all_auc_results.csv", index=False)
    print(f"\nSaved raw results to {output_dir / 'all_auc_results.csv'}")

    # Create condition column for easier comparison
    df["condition"] = df["method"] + "_" + df["selection_type"]

    # =========================================================================
    # Table 1: AGGREGATE AUC by condition (across all model sets)
    # =========================================================================
    print("\n" + "=" * 80)
    print("AGGREGATE AUC (lower = better discriminability)")
    print("=" * 80)

    agg_df = df[df["track"] == "AGGREGATE"].copy()

    # Pivot: model_set x condition
    pivot = agg_df.pivot_table(
        index="model_set",
        columns="condition",
        values="selected_auc"
    )

    # Reorder columns for readability
    col_order = ["v2_full_final", "v2_full_best_raw_combined",
                 "powermean_final", "powermean_best_raw_combined"]
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    print("\nSelected AUC by model_set and condition:")
    print(pivot.round(4).to_string())

    # Mean across model sets
    print("\nMean across all model sets:")
    print(pivot.mean().round(4).to_string())

    # =========================================================================
    # Table 2: Per-track breakdown (averaged across model sets)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PER-TRACK AUC (averaged across model sets)")
    print("=" * 80)

    track_means = df.groupby(["track", "condition"])["selected_auc"].mean().unstack()
    track_means = track_means[[c for c in col_order if c in track_means.columns]]

    print(track_means.round(4).to_string())

    # =========================================================================
    # Table 3: Encoding tracks only (the key comparison for powermean)
    # =========================================================================
    print("\n" + "=" * 80)
    print("ENCODING TRACKS ONLY (sub-01 to sub-07)")
    print("=" * 80)

    encoding_tracks = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
    enc_df = df[df["track"].isin(encoding_tracks)]

    enc_means = enc_df.groupby("condition")["selected_auc"].mean()
    enc_means = enc_means.reindex([c for c in col_order if c in enc_means.index])

    print("\nMean encoding track AUC:")
    print(enc_means.round(4).to_string())

    # =========================================================================
    # Table 4: Raw track only
    # =========================================================================
    print("\n" + "=" * 80)
    print("RAW TRACK ONLY")
    print("=" * 80)

    raw_df = df[df["track"] == "raw"]
    raw_means = raw_df.groupby("condition")["selected_auc"].mean()
    raw_means = raw_means.reindex([c for c in col_order if c in raw_means.index])

    print("\nMean raw track AUC:")
    print(raw_means.round(4).to_string())

    # =========================================================================
    # Summary: Best condition per model_set
    # =========================================================================
    print("\n" + "=" * 80)
    print("BEST CONDITION PER MODEL SET (lowest AGGREGATE AUC)")
    print("=" * 80)

    best_per_model = pivot.idxmin(axis=1)
    print(best_per_model.to_string())

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    overall_best = pivot.mean().idxmin()
    print(f"\nBest overall condition: {overall_best}")
    print(f"Mean AGGREGATE AUC: {pivot.mean()[overall_best]:.4f}")


if __name__ == "__main__":
    main()
