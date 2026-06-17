#!/bin/bash
#
# Submit the feature-only method sweep on Raven.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

START_SLURM="${CSTIMS_SLURM_WRAPPER:-/u/rothj/laion_natural/scripts/start_as_slurm_job.py}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/data/home_roth/miniforge3/bin/python" ]]; then
    PYTHON_BIN="/data/home_roth/miniforge3/bin/python"
  elif [[ -x "/u/rothj/miniforge3/bin/python" ]]; then
    PYTHON_BIN="/u/rothj/miniforge3/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi
CONDA_PREFIX_GUESS="$(cd "$(dirname "${PYTHON_BIN}")/.." 2>/dev/null && pwd || true)"
CONDA_LIB="${CONDA_LIB:-${CONDA_PREFIX_GUESS}/lib}"
if [[ -d "${CONDA_LIB}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"
fi

MODEL_SET="${MODEL_SET:-sota}"
METHODS="${METHODS:-}"
POOL_FEATURE_DIR="${POOL_FEATURE_DIR:-}"
RANDOM_FEATURE_DIR="${RANDOM_FEATURE_DIR:-${POOL_FEATURE_DIR}}"
N_RANDOM_IMAGES="${N_RANDOM_IMAGES:-}"
TARGET_SIZE="${TARGET_SIZE:-100}"
MAX_RAM_GB="${MAX_RAM_GB:-50}"
MAX_IMAGES="${MAX_IMAGES:-}"
POOL_SIZES="${POOL_SIZES:-}"
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
SKIP_SELECTION="${SKIP_SELECTION:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SHARED_ENCODINGS="${SHARED_ENCODINGS:-1}"
IMAGE_FILTER="${IMAGE_FILTER:-1}"
FILTER_MIN_RESOLUTION="${FILTER_MIN_RESOLUTION:-1000}"
FILTER_NATURAL_PROB_THRESHOLD="${FILTER_NATURAL_PROB_THRESHOLD:-0.85}"
FILTER_DOWNLOAD_TIMEOUT="${FILTER_DOWNLOAD_TIMEOUT:-10.0}"
FILTER_MAX_ATTEMPTS_PER_ITERATION="${FILTER_MAX_ATTEMPTS_PER_ITERATION:-1000}"
FILTER_PARALLEL_BATCH_SIZE="${FILTER_PARALLEL_BATCH_SIZE:-1}"
FILTER_CLASSIFIER_PATH="${FILTER_CLASSIFIER_PATH:-}"
FILTER_SAVE_IMAGES="${FILTER_SAVE_IMAGES:-1}"
ALLOW_FILTER_FALLBACK="${ALLOW_FILTER_FALLBACK:-0}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/00_stimulus_selection/feature_method_sweep/results/${MODEL_SET}_${TIMESTAMP}}"
RUN_LOCAL="${RUN_LOCAL:-0}"

SCRIPT_CMD=(
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

if [[ -n "${MAX_IMAGES}" ]]; then
  SCRIPT_CMD+=(--max-images "${MAX_IMAGES}")
fi

if [[ -n "${POOL_SIZES}" ]]; then
  SCRIPT_CMD+=(--pool-sizes "${POOL_SIZES}")
fi

if [[ -n "${METHODS}" ]]; then
  SCRIPT_CMD+=(--methods "${METHODS}")
fi

if [[ -n "${POOL_FEATURE_DIR}" ]]; then
  SCRIPT_CMD+=(--pool-feature-dir "${POOL_FEATURE_DIR}")
fi

if [[ -n "${RANDOM_FEATURE_DIR}" ]]; then
  SCRIPT_CMD+=(--random-feature-dir "${RANDOM_FEATURE_DIR}")
fi

if [[ -n "${N_RANDOM_IMAGES}" ]]; then
  SCRIPT_CMD+=(--n-random-images "${N_RANDOM_IMAGES}")
fi

if [[ "${ADAPTIVE_BATCH_SIZE}" == "1" ]]; then
  SCRIPT_CMD+=(--adaptive-batch-size)
  SCRIPT_CMD+=(--max-batch-size "${MAX_BATCH_SIZE}")
  SCRIPT_CMD+=(--min-batch-size "${MIN_BATCH_SIZE}")
fi

if [[ "${SKIP_SELECTION}" == "1" ]]; then
  SCRIPT_CMD+=(--skip-selection)
fi

if [[ "${SKIP_EVAL}" == "1" ]]; then
  SCRIPT_CMD+=(--skip-eval)
fi

if [[ "${SHARED_ENCODINGS}" == "1" ]]; then
  SCRIPT_CMD+=(--shared-encodings)
else
  SCRIPT_CMD+=(--unique-encodings)
fi

if [[ "${IMAGE_FILTER}" == "1" ]]; then
  SCRIPT_CMD+=(--filter-min-resolution "${FILTER_MIN_RESOLUTION}")
  SCRIPT_CMD+=(--filter-natural-prob-threshold "${FILTER_NATURAL_PROB_THRESHOLD}")
  SCRIPT_CMD+=(--filter-download-timeout "${FILTER_DOWNLOAD_TIMEOUT}")
  SCRIPT_CMD+=(--filter-max-attempts-per-iteration "${FILTER_MAX_ATTEMPTS_PER_ITERATION}")
  SCRIPT_CMD+=(--filter-parallel-batch-size "${FILTER_PARALLEL_BATCH_SIZE}")
  if [[ -n "${FILTER_CLASSIFIER_PATH}" ]]; then
    SCRIPT_CMD+=(--filter-classifier-path "${FILTER_CLASSIFIER_PATH}")
  fi
  if [[ "${FILTER_SAVE_IMAGES}" != "1" ]]; then
    SCRIPT_CMD+=(--disable-filter-image-save)
  fi
  if [[ "${ALLOW_FILTER_FALLBACK}" == "1" ]]; then
    SCRIPT_CMD+=(--allow-filter-fallback)
  fi
else
  SCRIPT_CMD+=(--disable-image-filter)
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=(${EXTRA_ARGS})
  SCRIPT_CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi

if [[ "${RUN_LOCAL}" == "1" ]]; then
  CMD=("${PYTHON_BIN}" "${SCRIPT_CMD[@]}")
else
  CMD=("${PYTHON_BIN}" "${START_SLURM}" --gpu --mem "${MEM}" "${SCRIPT_CMD[@]}")
fi

echo "Submitting feature method sweep"
echo "Output root: ${OUTPUT_ROOT}"
echo "Shared encodings: ${SHARED_ENCODINGS}"
echo "Image filter: ${IMAGE_FILTER}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
