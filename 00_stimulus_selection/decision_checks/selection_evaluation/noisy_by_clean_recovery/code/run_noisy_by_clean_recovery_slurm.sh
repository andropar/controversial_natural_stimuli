#!/usr/bin/env bash
#
# Submit the noisy-by-clean recovery rerun on Raven via the same SLURM wrapper
# pattern used by the original evaluation scripts.
#
# Usage:
#   bash code/run_noisy_by_clean_recovery_slurm.sh [--dry-run]
#
# Defaults match the original unique-evaluation run, except for the corrected
# noisy-by-clean recovery orientation inside compute_noisy_by_clean_recovery.py.

set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}
SCRIPT="${ROOT}/00_stimulus_selection/decision_checks/selection_evaluation/noisy_by_clean_recovery/code/compute_noisy_by_clean_recovery.py"
PLOT="${ROOT}/00_stimulus_selection/decision_checks/selection_evaluation/noisy_by_clean_recovery/code/plot_noisy_by_clean_recovery.py"
START_SLURM="${CSTIMS_SLURM_WRAPPER:-/u/rothj/laion_natural/scripts/start_as_slurm_job.py}"
PYTHON=${PYTHON:-python}

MEM=${MEM:-64000}
ENV_NAME=${ENV_NAME:-raven}
WHICH_SELECTION=${WHICH_SELECTION:-final}
SEED=${SEED:-42}
N_RANDOM_SUBSETS=${N_RANDOM_SUBSETS:-50}
N_NOISE_SAMPLES=${N_NOISE_SAMPLES:-100}
N_BOOTSTRAP=${N_BOOTSTRAP:-500}

MODEL_SETS=(
  "all_models"
  "architecture"
  "dataset"
  "sota"
  "training_objective"
)

DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --model-sets)
      IFS=',' read -r -a MODEL_SETS <<< "$2"
      shift 2
      ;;
    --mem)
      MEM="$2"
      shift 2
      ;;
    --env)
      ENV_NAME="$2"
      shift 2
      ;;
    --which-selection)
      WHICH_SELECTION="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --n-random-subsets)
      N_RANDOM_SUBSETS="$2"
      shift 2
      ;;
    --n-noise-samples)
      N_NOISE_SAMPLES="$2"
      shift 2
      ;;
    --n-bootstrap)
      N_BOOTSTRAP="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

if [[ "$DRY_RUN" == "true" ]]; then
  echo "=== DRY RUN MODE ==="
elif [[ ! -f "$START_SLURM" ]]; then
  echo "ERROR: SLURM wrapper not found: $START_SLURM"
  echo "Set CSTIMS_SLURM_WRAPPER to the Raven wrapper path if it differs."
  exit 1
fi

echo "========================================"
echo "Noisy-by-clean recovery SLURM submission"
echo "========================================"
echo "Root: ${ROOT}"
echo "SLURM wrapper: ${START_SLURM}"
echo "Model sets: ${MODEL_SETS[*]}"
echo "Memory: ${MEM} MB"
echo "Parameters: env=${ENV_NAME}, which_selection=${WHICH_SELECTION}, seed=${SEED}, n_random_subsets=${N_RANDOM_SUBSETS}, n_noise_samples=${N_NOISE_SAMPLES}, n_bootstrap=${N_BOOTSTRAP}"
echo "Metric/correlation: from payload config, matching original eval defaults"
echo ""

TOTAL_JOBS=0
JOB_IDS=()

for MODEL_SET in "${MODEL_SETS[@]}"; do
  MODEL_SET=$(echo "$MODEL_SET" | xargs)
  [[ -z "$MODEL_SET" ]] && continue

  ARGS=(
    "${SCRIPT}"
    --model-sets "${MODEL_SET}"
    --env "${ENV_NAME}"
    --which-selection "${WHICH_SELECTION}"
    --seed "${SEED}"
    --n-random-subsets "${N_RANDOM_SUBSETS}"
    --n-noise-samples "${N_NOISE_SAMPLES}"
    --n-bootstrap "${N_BOOTSTRAP}"
    --unique-encodings
  )
  CMD=("${PYTHON}" "${START_SLURM}" --gpu --mem "${MEM}" "${ARGS[@]}")

  echo "----------------------------------------"
  echo "Model set: ${MODEL_SET}"
  echo "----------------------------------------"

  if [[ "$DRY_RUN" == "true" ]]; then
    print_command "${CMD[@]}"
    JOB_ID="DRY_${MODEL_SET}"
  else
    OUTPUT=$("${CMD[@]}" 2>&1)
    JOB_ID=$(echo "$OUTPUT" | grep -oP 'Submitted batch job \K\d+' || \
             echo "$OUTPUT" | grep -oP 'job[= ]+\K\d+' || \
             echo "$OUTPUT" | grep -oP '\d{5,}' | head -1)

    if [[ -z "$JOB_ID" ]]; then
      echo "ERROR: Could not extract job ID from:"
      echo "$OUTPUT"
      exit 1
    fi

    echo "Submitted job: ${JOB_ID}"
  fi

  JOB_IDS+=("$JOB_ID")
  TOTAL_JOBS=$((TOTAL_JOBS + 1))
  echo ""
done

echo "========================================"
echo "Submitted ${TOTAL_JOBS} jobs"
echo "Job IDs: ${JOB_IDS[*]}"
echo "Monitor with: squeue -u \$USER"
echo ""
echo "After all jobs finish, generate the figure with:"
print_command "${PYTHON}" "${PLOT}"
echo "========================================"
