#!/bin/bash
#
# Submit the feature-only method sweep on Raven.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

START_SLURM="${CSTIMS_SLURM_WRAPPER:-/u/rothj/laion_natural/scripts/start_as_slurm_job.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_SET="${MODEL_SET:-sota}"
TARGET_SIZE="${TARGET_SIZE:-100}"
MAX_RAM_GB="${MAX_RAM_GB:-50}"
BATCH_SIZE="${BATCH_SIZE:-2500}"
ADAPTIVE_BATCH_SIZE="${ADAPTIVE_BATCH_SIZE:-0}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-0}"
MIN_BATCH_SIZE="${MIN_BATCH_SIZE:-256}"
MEM="${MEM:-128000}"
N_RANDOM_SUBSETS="${N_RANDOM_SUBSETS:-50}"
N_NOISE_SAMPLES="${N_NOISE_SAMPLES:-100}"
N_BOOTSTRAP="${N_BOOTSTRAP:-500}"
PROGRESS_EVERY_BATCHES="${PROGRESS_EVERY_BATCHES:-10}"
SEED="${SEED:-42}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/00_stimulus_selection/selection_evaluation/feature_method_sweep/results/${MODEL_SET}_${TIMESTAMP}}"

CMD=(
  "${PYTHON_BIN}" "${START_SLURM}" --gpu --mem "${MEM}"
  "${SCRIPT_DIR}/feature_method_sweep.py"
  --env raven
  --model-set "${MODEL_SET}"
  --target-size "${TARGET_SIZE}"
  --max-ram-gb "${MAX_RAM_GB}"
  --batch-size "${BATCH_SIZE}"
  --seed "${SEED}"
  --n-random-subsets "${N_RANDOM_SUBSETS}"
  --n-noise-samples "${N_NOISE_SAMPLES}"
  --n-bootstrap "${N_BOOTSTRAP}"
  --progress-every-batches "${PROGRESS_EVERY_BATCHES}"
  --output-root "${OUTPUT_ROOT}"
)

if [[ "${ADAPTIVE_BATCH_SIZE}" == "1" ]]; then
  CMD+=(--adaptive-batch-size)
  CMD+=(--max-batch-size "${MAX_BATCH_SIZE}")
  CMD+=(--min-batch-size "${MIN_BATCH_SIZE}")
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=(${EXTRA_ARGS})
  CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi

echo "Submitting feature method sweep"
echo "Output root: ${OUTPUT_ROOT}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
