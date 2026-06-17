#!/usr/bin/env bash
#
# Submit fitted teacher/student recovery jobs on Raven.

set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}
SCRIPT="${ROOT}/00_stimulus_selection/selection_evaluation/noisy_by_clean_recovery/code/compute_teacher_student_recovery.py"
PLOT="${ROOT}/00_stimulus_selection/selection_evaluation/noisy_by_clean_recovery/code/plot_teacher_student_recovery.py"
START_SLURM="${CSTIMS_SLURM_WRAPPER:-/u/rothj/laion_natural/scripts/start_as_slurm_job.py}"
PYTHON=${PYTHON:-python}

MEM=${MEM:-64000}
ENV_NAME=${ENV_NAME:-raven}
WHICH_SELECTION=${WHICH_SELECTION:-final}
SEED=${SEED:-42}
N_RANDOM_SUBSETS=${N_RANDOM_SUBSETS:-20}
N_RANDOM_IMAGES=${N_RANDOM_IMAGES:-10000}
N_SPLITS=${N_SPLITS:-8}
N_NOISE_SAMPLES=${N_NOISE_SAMPLES:-20}
ALPHAS=${ALPHAS:-0.001,0.01,0.1,1,10,100}

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
    --n-random-images)
      N_RANDOM_IMAGES="$2"
      shift 2
      ;;
    --n-splits)
      N_SPLITS="$2"
      shift 2
      ;;
    --n-noise-samples)
      N_NOISE_SAMPLES="$2"
      shift 2
      ;;
    --alphas)
      ALPHAS="$2"
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
echo "Teacher/student recovery SLURM submission"
echo "========================================"
echo "Root: ${ROOT}"
echo "SLURM wrapper: ${START_SLURM}"
echo "Model sets: ${MODEL_SETS[*]}"
echo "Memory: ${MEM} MB"
echo "Parameters: env=${ENV_NAME}, which_selection=${WHICH_SELECTION}, seed=${SEED}, n_random_subsets=${N_RANDOM_SUBSETS}, n_random_images=${N_RANDOM_IMAGES}, n_splits=${N_SPLITS}, n_noise_samples=${N_NOISE_SAMPLES}, alphas=${ALPHAS}"
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
    --n-random-images "${N_RANDOM_IMAGES}"
    --n-splits "${N_SPLITS}"
    --n-noise-samples "${N_NOISE_SAMPLES}"
    --alphas "${ALPHAS}"
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
echo "After all jobs finish, generate the figures with:"
print_command "${PYTHON}" "${PLOT}"
echo "========================================"
