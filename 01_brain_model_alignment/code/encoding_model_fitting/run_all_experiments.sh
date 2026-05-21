#!/bin/bash
#
# Run encoding model fitting with all combinations of:
#   - eval_method: odd_even, group_kfold
#   - scale_features: true, false
#
# This creates 4 separate runs, each with all subjects and models.
#
# Usage:
#   ./run_all_experiments.sh [--dry-run] [--local]
#
# Options:
#   --dry-run   Print commands without executing
#   --local     Run locally instead of submitting to SLURM
#
# Output directories:
#   results/encoding_YYYYMMDD_HHMMSS_oddeven_scaled/
#   results/encoding_YYYYMMDD_HHMMSS_oddeven_unscaled/
#   results/encoding_YYYYMMDD_HHMMSS_groupkfold_scaled/
#   results/encoding_YYYYMMDD_HHMMSS_groupkfold_unscaled/

set -e

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/fit_encoding_hydra.py"
START_SLURM="${CSTIMS_SLURM_WRAPPER:-}"
COMBINE_SCRIPT="${SCRIPT_DIR}/combine_results.py"
PLOT_SCRIPT="${SCRIPT_DIR}/plot_encoding_results.py"

# Job resources
MEM="64000"

# Subjects
SUBJECTS=(
    "sub-01"
    "sub-03"
    "sub-05"
    "sub-06"
    "sub-07"
)

# Model batching
MODELS_PER_BATCH=5
TOTAL_MODELS=21
NUM_BATCHES=$(( (TOTAL_MODELS + MODELS_PER_BATCH - 1) / MODELS_PER_BATCH ))

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Experiment configurations: (eval_method, scale_features, suffix)
declare -a EXPERIMENTS=(
    "odd_even:true:oddeven_scaled"
    "odd_even:false:oddeven_unscaled"
#   "group_kfold:true:groupkfold_scaled"
    "group_kfold:false:groupkfold_unscaled"
)

# ============================================================================
# Parse arguments
# ============================================================================

DRY_RUN=false
LOCAL=false

for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            echo "=== DRY RUN MODE ==="
            ;;
        --local)
            LOCAL=true
            echo "=== LOCAL MODE (no SLURM) ==="
            ;;
    esac
done

if [[ "$LOCAL" != "true" && -z "$START_SLURM" ]]; then
    if [[ "$DRY_RUN" == "true" ]]; then
        START_SLURM="<CSTIMS_SLURM_WRAPPER>"
    else
        echo "ERROR: set CSTIMS_SLURM_WRAPPER to a local Slurm submission wrapper, or pass --local."
        exit 1
    fi
fi

# ============================================================================
# Submit experiments
# ============================================================================

echo "========================================"
echo "DeepVision Encoding - Full Experiment Suite"
echo "========================================"
echo "Timestamp: ${TIMESTAMP}"
echo "Subjects: ${SUBJECTS[*]}"
echo "Model batches: ${NUM_BATCHES} (${MODELS_PER_BATCH} models each)"
echo "Experiments: ${#EXPERIMENTS[@]}"
echo ""

TOTAL_JOBS=0
declare -A OUTPUT_DIRS

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    # Parse experiment config
    IFS=':' read -r EVAL_METHOD SCALE_FEATURES SUFFIX <<< "$EXPERIMENT"

    OUTPUT_DIR="${SCRIPT_DIR}/results/encoding_${TIMESTAMP}_${SUFFIX}"
    OUTPUT_DIRS[$SUFFIX]="$OUTPUT_DIR"

    echo "========================================"
    echo "Experiment: ${SUFFIX}"
    echo "  eval_method: ${EVAL_METHOD}"
    echo "  scale_features: ${SCALE_FEATURES}"
    echo "  output: ${OUTPUT_DIR}"
    echo "========================================"

    if [[ "$DRY_RUN" != "true" ]]; then
        mkdir -p "${OUTPUT_DIR}"
    fi

    for SUBJECT in "${SUBJECTS[@]}"; do
        echo "  Subject: ${SUBJECT}"

        for BATCH in $(seq 0 $((NUM_BATCHES - 1))); do
            # Build hydra arguments
            HYDRA_ARGS=(
                "parallel.subject=${SUBJECT}"
                "parallel.model_batch=${BATCH}"
                "parallel.models_per_batch=${MODELS_PER_BATCH}"
                "fitting.eval_method=${EVAL_METHOD}"
                "fitting.scale_features=${SCALE_FEATURES}"
                "hydra.run.dir=${OUTPUT_DIR}"
            )

            if [[ "$LOCAL" == "true" ]]; then
                # Run locally
                CMD="python ${SCRIPT} ${HYDRA_ARGS[*]}"
            else
                # Submit to SLURM
                CMD="python ${START_SLURM} --mem ${MEM} --gpu --conda-env deepjuice ${SCRIPT} ${HYDRA_ARGS[*]}"
            fi

            if [[ "$DRY_RUN" == "true" ]]; then
                echo "    [DRY RUN] Batch ${BATCH}: ${EVAL_METHOD}, scale=${SCALE_FEATURES}"
            else
                if [[ "$LOCAL" == "true" ]]; then
                    echo "    Running batch ${BATCH}..."
                    $CMD
                else
                    # Submit and capture job ID
                    OUTPUT=$($CMD 2>&1)
                    JOB_ID=$(echo "$OUTPUT" | grep -oP 'Submitted batch job \K\d+' || \
                             echo "$OUTPUT" | grep -oP 'job[= ]+\K\d+' || \
                             echo "$OUTPUT" | grep -oP '\d{5,}' | head -1)

                    if [[ -n "$JOB_ID" ]]; then
                        echo "    Batch ${BATCH}: Job ${JOB_ID}"
                    else
                        echo "    Batch ${BATCH}: Submitted (could not extract job ID)"
                    fi
                fi
            fi

            TOTAL_JOBS=$((TOTAL_JOBS + 1))
        done
    done
    echo ""
done

# ============================================================================
# Summary
# ============================================================================

echo "========================================"
echo "SUMMARY"
echo "========================================"
echo "Total jobs submitted: ${TOTAL_JOBS}"
echo ""
echo "Output directories:"
for SUFFIX in "${!OUTPUT_DIRS[@]}"; do
    echo "  ${SUFFIX}: ${OUTPUT_DIRS[$SUFFIX]}"
done
echo ""

if [[ "$DRY_RUN" != "true" ]]; then
    echo "After all jobs complete, run for each experiment:"
    echo ""
    for SUFFIX in "${!OUTPUT_DIRS[@]}"; do
        echo "  # ${SUFFIX}"
        echo "  python ${COMBINE_SCRIPT} ${OUTPUT_DIRS[$SUFFIX]}"
        echo "  python ${PLOT_SCRIPT} ${OUTPUT_DIRS[$SUFFIX]}"
        echo ""
    done

    # Create a convenience script to combine all
    COMBINE_ALL="${SCRIPT_DIR}/results/combine_all_${TIMESTAMP}.sh"
    cat > "${COMBINE_ALL}" << EOF
#!/bin/bash
# Combine and plot all experiments from ${TIMESTAMP}

set -e

COMBINE_SCRIPT="${COMBINE_SCRIPT}"
PLOT_SCRIPT="${PLOT_SCRIPT}"

EOF

    for SUFFIX in "${!OUTPUT_DIRS[@]}"; do
        cat >> "${COMBINE_ALL}" << EOF
echo "Processing: ${SUFFIX}"
python \${COMBINE_SCRIPT} ${OUTPUT_DIRS[$SUFFIX]}
python \${PLOT_SCRIPT} ${OUTPUT_DIRS[$SUFFIX]}
echo ""

EOF
    done

    cat >> "${COMBINE_ALL}" << 'EOF'
echo "Done! All experiments processed."
EOF

    chmod +x "${COMBINE_ALL}"
    echo "Or run all at once:"
    echo "  ${COMBINE_ALL}"
fi

echo ""
echo "Monitor jobs with: squeue -u \$USER"
