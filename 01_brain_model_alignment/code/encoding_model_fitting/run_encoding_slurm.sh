#!/bin/bash
#
# Submit parallel SLURM jobs for encoding model fitting.
#
# Submits jobs for all combinations of subjects × model batches.
# With ~150 models, 10 models/batch = 15 batches × 5 subjects = 75 jobs.
#
# Usage:
#   ./run_encoding_slurm.sh [--dry-run]
#
# The --dry-run flag prints commands without submitting.

set -e

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/fit_encoding_hydra.py"
START_SLURM="${CSTIMS_SLURM_WRAPPER:-}"

# Job resources (GPU for feature extraction, CPU for ridge)
MEM="64000"

# Subjects
SUBJECTS=(
    "sub-01"
    "sub-03"
    "sub-05"
    "sub-06"
    "sub-07"
)

# Model batching: 5 models per batch, ceiling division for remainder
# 5 subjects × 5 batches = 25 jobs total
MODELS_PER_BATCH=5
TOTAL_MODELS=21  # From model_list.csv (excluding header)
NUM_BATCHES=$(( (TOTAL_MODELS + MODELS_PER_BATCH - 1) / MODELS_PER_BATCH ))  # = 5 batches

# Evaluation method: random_kfold, group_kfold, or odd_even
EVAL_METHOD="random_kfold"
N_FOLDS=5

# Preprocessing options
RESPONSE_ZSCORE=false
SRP=false
SCALE_FEATURES=false

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Output directory (shared across all jobs)
OUTPUT_DIR="${SCRIPT_DIR}/results/encoding_${TIMESTAMP}"

# ============================================================================
# Parse arguments
# ============================================================================

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
fi

if [[ -z "$START_SLURM" ]]; then
    if [[ "$DRY_RUN" == "true" ]]; then
        START_SLURM="<CSTIMS_SLURM_WRAPPER>"
    else
        echo "ERROR: set CSTIMS_SLURM_WRAPPER to a local Slurm submission wrapper."
        exit 1
    fi
fi

# ============================================================================
# Submit jobs
# ============================================================================

echo "========================================"
echo "DeepVision Encoding Model Fitting"
echo "========================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "Subjects: ${SUBJECTS[*]}"
echo "Model batches: ${NUM_BATCHES} (${MODELS_PER_BATCH} models each)"
echo "Eval method: ${EVAL_METHOD} (n_folds=${N_FOLDS})"
echo "Preprocessing: response_zscore=${RESPONSE_ZSCORE}, srp=${SRP}, scale_features=${SCALE_FEATURES}"
echo "Total jobs: $((${#SUBJECTS[@]} * NUM_BATCHES))"
echo ""

TOTAL_JOBS=0
JOB_IDS=()

for SUBJECT in "${SUBJECTS[@]}"; do
    echo "----------------------------------------"
    echo "Subject: ${SUBJECT}"
    echo "----------------------------------------"

    for BATCH in $(seq 0 $((NUM_BATCHES - 1))); do
        # Build hydra arguments
        HYDRA_ARGS=(
            "parallel.subject=${SUBJECT}"
            "parallel.model_batch=${BATCH}"
            "parallel.models_per_batch=${MODELS_PER_BATCH}"
            "fitting.eval_method=${EVAL_METHOD}"
            "fitting.n_folds=${N_FOLDS}"
            "fitting.scale_features=${SCALE_FEATURES}"
            "preprocessing.response_zscore=${RESPONSE_ZSCORE}"
            "preprocessing.srp=${SRP}"
            "hydra.run.dir=${OUTPUT_DIR}"
        )

        # Build command
        CMD="python ${START_SLURM} --conda-env deepjuice --gpu --mem ${MEM} ${SCRIPT} ${HYDRA_ARGS[*]}"

        if [[ "$DRY_RUN" == "true" ]]; then
            echo "  [DRY RUN] Batch ${BATCH}: $CMD"
            JOB_ID="FAKE_${SUBJECT}_batch${BATCH}"
        else
            # Submit and capture output
            OUTPUT=$($CMD 2>&1)

            # Extract job ID
            JOB_ID=$(echo "$OUTPUT" | grep -oP 'Submitted batch job \K\d+' || \
                     echo "$OUTPUT" | grep -oP 'job[= ]+\K\d+' || \
                     echo "$OUTPUT" | grep -oP '\d{5,}' | head -1)

            if [[ -z "$JOB_ID" ]]; then
                echo "  ERROR: Could not extract job ID from: $OUTPUT"
                continue
            fi

            echo "  Batch ${BATCH}: Job ${JOB_ID}"
        fi

        JOB_IDS+=("$JOB_ID")
        TOTAL_JOBS=$((TOTAL_JOBS + 1))
    done
done

echo ""
echo "========================================"
echo "Submitted ${TOTAL_JOBS} jobs"
echo "Output directory: ${OUTPUT_DIR}"
echo "Monitor with: squeue -u \$USER"
echo "========================================"

# ============================================================================
# Create a combine script to run after all jobs complete
# ============================================================================

COMBINE_SCRIPT="${OUTPUT_DIR}/combine_results.py"

if [[ "$DRY_RUN" != "true" ]]; then
    mkdir -p "${OUTPUT_DIR}"
    cat > "${COMBINE_SCRIPT}" << 'PYEOF'
#!/usr/bin/env python3
"""Combine results from parallel encoding model fitting jobs.

Merges per-batch parquet files into per-subject and final combined files.
Also generates summary statistics and reports on alpha stability.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

def load_metadata_for_subject(run_dir: Path, subject: str):
    """Load all metadata files for a subject."""
    metadata = []
    for subdir in run_dir.glob(f"{subject}_*"):
        meta_file = subdir / "metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
                meta["dir"] = str(subdir)
                metadata.append(meta)
    return metadata


if __name__ == "__main__":
    run_dir = Path(__file__).parent
    subjects = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]

    print(f"Combining results in {run_dir}")
    print("=" * 60)

    all_results = []

    # Combine parquet files
    for subject in subjects:
        batch_files = sorted(run_dir.glob(f"results_{subject}_batch*.parquet"))

        if batch_files:
            subject_df = pd.concat(
                [pd.read_parquet(f) for f in batch_files],
                ignore_index=True,
            )
            subject_df = subject_df.drop_duplicates(
                subset=["model", "layer", "subject"],
                keep="last",
            )
            subject_df.to_parquet(run_dir / f"results_{subject}.parquet", index=False)
            all_results.append(subject_df)
            print(f"{subject}: {len(subject_df)} models ({len(batch_files)} batches)")
        else:
            # Check for single result file
            result_file = run_dir / f"results_{subject}.parquet"
            if result_file.exists():
                df = pd.read_parquet(result_file)
                all_results.append(df)
                print(f"{subject}: {len(df)} models")
            else:
                print(f"{subject}: No results found")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_parquet(run_dir / "results_all.parquet", index=False)
        print(f"\nSaved {len(final_df)} total results")

        # Summary statistics
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)

        # Per-subject stats
        print("\nPer-subject veRSA:")
        subj_stats = final_df.groupby("subject")["veRSA"].agg(["mean", "std", "min", "max"])
        print(subj_stats.to_string())

        # Overall stats
        print(f"\nOverall veRSA: mean={final_df['veRSA'].mean():.4f}, std={final_df['veRSA'].std():.4f}")
        print(f"Overall voxel_r_median: mean={final_df['voxel_r_median'].mean():.4f}")

        # Top models
        print("\n=== Top 10 Models (by veRSA, averaged across subjects) ===")
        top = (
            final_df.groupby("model")["veRSA"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        for i, (model, score) in enumerate(top.items(), 1):
            print(f"  {i:2d}. {model[:50]:50s} {score:.4f}")

        # Alpha stability report (if available)
        if "alpha_cv_median" in final_df.columns:
            has_stability = final_df["alpha_cv_median"].notna()
            if has_stability.any():
                print("\n" + "=" * 60)
                print("ALPHA STABILITY REPORT")
                print("=" * 60)

                stab_df = final_df[has_stability]
                print(f"\nAlpha CV (coefficient of variation across folds):")
                print(f"  Mean: {stab_df['alpha_cv_median'].mean():.3f}")
                print(f"  Max:  {stab_df['alpha_cv_median'].max():.3f}")

                if "alpha_fold_corr_mean" in stab_df.columns:
                    print(f"\nFold-fold alpha correlation:")
                    print(f"  Mean: {stab_df['alpha_fold_corr_mean'].mean():.3f}")
                    print(f"  Min:  {stab_df['alpha_fold_corr_mean'].min():.3f}")

                if "veRSA_fold_mean" in stab_df.columns:
                    print(f"\nveRSA (fold mean vs overall):")
                    print(f"  Fold mean avg: {stab_df['veRSA_fold_mean'].mean():.4f}")
                    print(f"  Overall avg:   {stab_df['veRSA_visual'].mean():.4f}")

    # Also collect metadata from directories
    print("\n" + "=" * 60)
    print("COLLECTING METADATA FROM MODEL DIRECTORIES")
    print("=" * 60)

    all_metadata = []
    for subject in subjects:
        metadata = load_metadata_for_subject(run_dir, subject)
        all_metadata.extend(metadata)
        print(f"{subject}: {len(metadata)} model directories")

    if all_metadata:
        # Create summary from metadata
        rows = []
        for m in all_metadata:
            row = {
                "model": m["model"],
                "subject": m["subject"],
                "layer": m["layer"],
                "veRSA_visual": m["metrics"].get("veRSA_visual") or m["metrics"].get("veRSA_pearson_r"),
                "veRSA_hlvis": m["metrics"].get("veRSA_hlvis"),
                "veRSA_fold_mean": m["metrics"].get("veRSA_fold_mean"),
                "voxel_r_median": m["metrics"].get("voxel_r_median"),
                "alpha_median": m.get("alphas", {}).get("chosen_median"),
                "eval_method": m.get("eval_method"),
            }
            # Add alpha stability if available
            if "alpha_stability" in m and m["alpha_stability"]:
                row["alpha_cv_median"] = m["alpha_stability"].get("cv_median")
                row["alpha_fold_corr_mean"] = m["alpha_stability"].get("fold_correlation_mean")
            rows.append(row)
        meta_df = pd.DataFrame(rows)
        meta_df.to_parquet(run_dir / "metadata_summary.parquet", index=False)
        print(f"\nSaved metadata summary: {len(meta_df)} rows")

    print("\nDone!")
PYEOF
    chmod +x "${COMBINE_SCRIPT}"
    echo ""
    echo "After all jobs complete, run:"
    echo "  python ${COMBINE_SCRIPT}"
fi
