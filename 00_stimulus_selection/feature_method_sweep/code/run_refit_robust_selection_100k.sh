#!/usr/bin/env bash
#
# Submit and resume refit-robust greedy selection as a single queued Raven run.
#
# The script submits one compute job, then queues a small dependency-gated
# continuation job. The continuation checks selected_indices.npy and, if the
# target size is not reached, submits the next compute job with --resume.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT}"

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

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-refit_robust_selection}"
MODEL_SET="${MODEL_SET:-sota}"
RUN_NAME="${RUN_NAME:-${MODEL_SET}_${RUN_TAG}_${RUN_STAMP}}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/00_stimulus_selection/feature_method_sweep/results}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULTS_ROOT}/${RUN_NAME}}"
QUEUE_DIR="${QUEUE_DIR:-${RESULTS_ROOT}/refit_robust_queue_${RUN_STAMP}}"

SOURCE_RUN="${SOURCE_RUN:-${RESULTS_ROOT}/sota_20260611_111622}"
POOL_FEATURE_DIR="${POOL_FEATURE_DIR:-}"
MAX_RAM_GB="${MAX_RAM_GB:-300}"
MAX_IMAGES="${MAX_IMAGES:-}"
METHOD_ID="${METHOD_ID:-sub01_eval_augmented_loo_refit_robust}"
INITIAL_FROM_METHOD="${INITIAL_FROM_METHOD:-sub01_only_mean_min}"
ENV_NAME="${ENV_NAME:-raven}"
TRACK="${TRACK:-sub-01}"
ENCODING_ROI_SUBSET="${ENCODING_ROI_SUBSET:-hlvis}"
UNIQUE_ENCODINGS="${UNIQUE_ENCODINGS:-1}"
TARGET_SIZE="${TARGET_SIZE:-100}"
INIT_SIZE="${INIT_SIZE:-3}"
SEED="${SEED:-42}"
METRIC="${METRIC:-cosine}"
CORR_TYPE="${CORR_TYPE:-spearman}"
TARGET_DIM="${TARGET_DIM:-0}"
TOP_K_PROXY="${TOP_K_PROXY:-1024}"
RANDOM_SHORTLIST="${RANDOM_SHORTLIST:-0}"
PROXY_BATCH_SIZE="${PROXY_BATCH_SIZE:-2048}"
REFIT_POOL_SIZE="${REFIT_POOL_SIZE:-1000}"
REFIT_VAL_SIZE="${REFIT_VAL_SIZE:-200}"
N_NOISE_SAMPLES="${N_NOISE_SAMPLES:-1}"
NOISE_MULT="${NOISE_MULT:-1.0}"
NOISE_CEILING="${NOISE_CEILING:-0.46}"
ALPHAS="${ALPHAS:-0.001,0.01,0.1,1,10,100}"
FIT_NOISE_CALIBRATION="${FIT_NOISE_CALIBRATION:-rdm_empirical}"
CALIBRATION_IMAGES="${CALIBRATION_IMAGES:-100}"
CALIBRATION_NOISE_SAMPLES="${CALIBRATION_NOISE_SAMPLES:-2}"
CALIBRATION_MAX_ITER="${CALIBRATION_MAX_ITER:-8}"
TEACHER_AGGREGATION="${TEACHER_AGGREGATION:-mean}"
REFIT_OBJECTIVE="${REFIT_OBJECTIVE:-accuracy_margin}"
KERNEL_BATCH_SIZE="${KERNEL_BATCH_SIZE:-4096}"
REFIT_SCORE_WORKERS="${REFIT_SCORE_WORKERS:-4}"
PRECOMPUTE_BASE_KERNELS="${PRECOMPUTE_BASE_KERNELS:-1}"
ALLOW_REFIT_SELECTION_OVERLAP="${ALLOW_REFIT_SELECTION_OVERLAP:-0}"
IMAGE_FILTER="${IMAGE_FILTER:-1}"
FILTER_MIN_RESOLUTION="${FILTER_MIN_RESOLUTION:-1000}"
FILTER_NATURAL_PROB_THRESHOLD="${FILTER_NATURAL_PROB_THRESHOLD:-0.85}"
FILTER_DOWNLOAD_TIMEOUT="${FILTER_DOWNLOAD_TIMEOUT:-10.0}"
FILTER_MAX_ATTEMPTS_PER_ITERATION="${FILTER_MAX_ATTEMPTS_PER_ITERATION:-1000}"
FILTER_PARALLEL_BATCH_SIZE="${FILTER_PARALLEL_BATCH_SIZE:-1}"
FILTER_CLASSIFIER_PATH="${FILTER_CLASSIFIER_PATH:-}"
FILTER_SAVE_IMAGES="${FILTER_SAVE_IMAGES:-1}"
ALLOW_FILTER_FALLBACK="${ALLOW_FILTER_FALLBACK:-0}"
DEVICE="${DEVICE:-cuda}"
MEM="${MEM:-640000}"
USE_GPU="${USE_GPU:-1}"
RUN_LOCAL="${RUN_LOCAL:-0}"
SUBMIT="${SUBMIT:-1}"
QUEUE_CONTINUATIONS="${QUEUE_CONTINUATIONS:-1}"
MAX_RESUME_ROUNDS="${MAX_RESUME_ROUNDS:-30}"
RESUME_ROUND="${RESUME_ROUND:-0}"
CONTINUE_OUTPUT_ROOT="${CONTINUE_OUTPUT_ROOT:-}"
EXTRA_ARGS_BASE="${EXTRA_ARGS:-}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

mkdir -p "${QUEUE_DIR}"

extract_slurm_job_id() {
  local log_path="$1"
  awk '
    /^[[:space:]]*[0-9]+[[:space:]]*$/ {
      id=$1
    }
    /Submitted/ || /batch job/ {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^[0-9]+$/) {
          id=$i
        }
      }
    }
    END {
      if (id != "") {
        print id
      }
    }
  ' "${log_path}"
}

check_complete() {
  local output_root="$1"
  "${PYTHON_BIN}" - "${output_root}" "${METHOD_ID}" "${TARGET_SIZE}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

output_root = Path(sys.argv[1])
method_id = sys.argv[2]
target_size = int(sys.argv[3])
path = output_root / "payloads" / method_id / "selected_indices.npy"

if not path.exists():
    print(f"not complete: missing {path}", file=sys.stderr)
    sys.exit(1)
try:
    n_selected = int(np.load(path, mmap_mode="r").shape[0])
except Exception as exc:
    print(f"not complete: could not read {path}: {exc}", file=sys.stderr)
    sys.exit(1)
if n_selected < target_size:
    print(f"not complete: {n_selected}/{target_size}", file=sys.stderr)
    sys.exit(1)
print(f"complete: {n_selected}/{target_size}")
PY
}

build_script_cmd() {
  local output_root="$1"
  local resume="$2"
  SCRIPT_CMD=(
    "${SCRIPT_DIR}/refit_robust_selection.py"
    --source-run "${SOURCE_RUN}"
    --output-root "${output_root}"
    --method-id "${METHOD_ID}"
    --initial-from-method "${INITIAL_FROM_METHOD}"
    --env "${ENV_NAME}"
    --model-set "${MODEL_SET}"
    --track "${TRACK}"
    --encoding-roi-subset "${ENCODING_ROI_SUBSET}"
    --target-size "${TARGET_SIZE}"
    --init-size "${INIT_SIZE}"
    --seed "${SEED}"
    --max-ram-gb "${MAX_RAM_GB}"
    --metric "${METRIC}"
    --corr-type "${CORR_TYPE}"
    --refit-pool-size "${REFIT_POOL_SIZE}"
    --refit-val-size "${REFIT_VAL_SIZE}"
    --top-k-proxy "${TOP_K_PROXY}"
    --random-shortlist "${RANDOM_SHORTLIST}"
    --proxy-batch-size "${PROXY_BATCH_SIZE}"
    --target-dim "${TARGET_DIM}"
    --fit-noise-calibration "${FIT_NOISE_CALIBRATION}"
    --calibration-images "${CALIBRATION_IMAGES}"
    --calibration-noise-samples "${CALIBRATION_NOISE_SAMPLES}"
    --calibration-max-iter "${CALIBRATION_MAX_ITER}"
    --n-noise-samples "${N_NOISE_SAMPLES}"
    --noise-mult "${NOISE_MULT}"
    --noise-ceiling "${NOISE_CEILING}"
    --alphas "${ALPHAS}"
    --teacher-aggregation "${TEACHER_AGGREGATION}"
    --refit-objective "${REFIT_OBJECTIVE}"
    --kernel-batch-size "${KERNEL_BATCH_SIZE}"
    --refit-score-workers "${REFIT_SCORE_WORKERS}"
    --device "${DEVICE}"
  )

  if [[ "${UNIQUE_ENCODINGS}" == "1" ]]; then
    SCRIPT_CMD+=(--unique-encodings)
  fi
  if [[ -n "${POOL_FEATURE_DIR}" ]]; then
    SCRIPT_CMD+=(--pool-feature-dir "${POOL_FEATURE_DIR}")
  fi
  if [[ -n "${MAX_IMAGES}" ]]; then
    SCRIPT_CMD+=(--max-images "${MAX_IMAGES}")
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
  if [[ "${PRECOMPUTE_BASE_KERNELS}" != "1" ]]; then
    SCRIPT_CMD+=(--no-precompute-base-kernels)
  fi
  if [[ "${ALLOW_REFIT_SELECTION_OVERLAP}" == "1" ]]; then
    SCRIPT_CMD+=(--allow-refit-selection-overlap)
  fi
  if [[ "${resume}" == "1" ]]; then
    SCRIPT_CMD+=(--resume)
  fi
  if [[ -n "${EXTRA_ARGS_BASE}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARRAY=(${EXTRA_ARGS_BASE})
    SCRIPT_CMD+=("${EXTRA_ARGS_ARRAY[@]}")
  fi
}

print_command() {
  local -n cmd_ref=$1
  printf 'Command:'
  printf ' %q' "${cmd_ref[@]}"
  printf '\n'
}

submit_refit_job() {
  local output_root="$1"
  local resume="$2"
  local submit_log="$3"
  build_script_cmd "${output_root}" "${resume}"

  if [[ "${RUN_LOCAL}" == "1" ]]; then
    CMD=("${PYTHON_BIN}" -u "${SCRIPT_CMD[@]}")
  else
    SLURM_ARGS=(--mem "${MEM}")
    if [[ "${USE_GPU}" == "1" ]]; then
      SLURM_ARGS=(--gpu "${SLURM_ARGS[@]}")
    fi
    CMD=("${PYTHON_BIN}" "${START_SLURM}" "${SLURM_ARGS[@]}" "${SCRIPT_CMD[@]}")
  fi

  echo "Submitting refit-robust selection"
  echo "Output root: ${output_root}"
  echo "Resume: ${resume}"
  echo "Queue dir: ${QUEUE_DIR}"
  print_command CMD

  set +e
  "${CMD[@]}" 2>&1 | tee "${submit_log}"
  local submit_status="${PIPESTATUS[0]}"
  set -e

  if [[ "${submit_status}" -ne 0 ]]; then
    echo "Refit-robust submission failed; see ${submit_log}" >&2
    exit "${submit_status}"
  fi

  if [[ "${RUN_LOCAL}" == "1" ]]; then
    SUBMITTED_JOB_ID="local"
    return
  fi

  SUBMITTED_JOB_ID="$(extract_slurm_job_id "${submit_log}")"
  if [[ -z "${SUBMITTED_JOB_ID}" ]]; then
    echo "Could not parse Slurm job id; see ${submit_log}" >&2
    exit 3
  fi
}

write_export_if_set_to_script() {
  local script_path="$1"
  local name="$2"
  local value="${!name-}"
  if [[ -n "${value}" ]]; then
    printf 'export %s=%q\n' "${name}" "${value}" >> "${script_path}"
  fi
}

queue_continuation() {
  local output_root="$1"
  local dependency_job_id="$2"
  local next_round="$3"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local submit_script="${QUEUE_DIR}/refit_robust_continue_round${next_round}_after_${dependency_job_id}_${stamp}.slurm.sh"
  local submit_log="${QUEUE_DIR}/refit_robust_continue_round${next_round}_after_${dependency_job_id}_${stamp}.%j.log"

  cat > "${submit_script}" <<EOF
#!/bin/bash
#SBATCH --job-name=continue_refit_robust
#SBATCH --time=00:15:00
#SBATCH --mem=2G
#SBATCH --output=${submit_log}
#SBATCH --error=${submit_log}

set -euo pipefail

cd "${ROOT}"
EOF

  printf 'export CONTINUE_OUTPUT_ROOT=%q\n' "${output_root}" >> "${submit_script}"
  printf 'export RESUME_ROUND=%q\n' "${next_round}" >> "${submit_script}"
  printf 'export RUN_STAMP=%q\n' "${RUN_STAMP}" >> "${submit_script}"
  printf 'export RUN_NAME=%q\n' "${RUN_NAME}" >> "${submit_script}"
  printf 'export OUTPUT_ROOT=%q\n' "${output_root}" >> "${submit_script}"
  printf 'export QUEUE_DIR=%q\n' "${QUEUE_DIR}" >> "${submit_script}"
  printf 'export EXTRA_ARGS=%q\n' "${EXTRA_ARGS_BASE}" >> "${submit_script}"

  for name in \
    START_SLURM PYTHON_BIN CONDA_LIB RUN_TAG MODEL_SET RESULTS_ROOT SOURCE_RUN \
    POOL_FEATURE_DIR MAX_RAM_GB MAX_IMAGES METHOD_ID INITIAL_FROM_METHOD TRACK UNIQUE_ENCODINGS \
    ENV_NAME ENCODING_ROI_SUBSET TARGET_SIZE INIT_SIZE SEED METRIC CORR_TYPE \
    TARGET_DIM TOP_K_PROXY RANDOM_SHORTLIST \
    PROXY_BATCH_SIZE REFIT_POOL_SIZE REFIT_VAL_SIZE N_NOISE_SAMPLES NOISE_MULT \
    NOISE_CEILING ALPHAS FIT_NOISE_CALIBRATION CALIBRATION_IMAGES \
    CALIBRATION_NOISE_SAMPLES CALIBRATION_MAX_ITER TEACHER_AGGREGATION \
    REFIT_OBJECTIVE KERNEL_BATCH_SIZE REFIT_SCORE_WORKERS PRECOMPUTE_BASE_KERNELS \
    ALLOW_REFIT_SELECTION_OVERLAP IMAGE_FILTER FILTER_MIN_RESOLUTION \
    FILTER_NATURAL_PROB_THRESHOLD FILTER_DOWNLOAD_TIMEOUT \
    FILTER_MAX_ATTEMPTS_PER_ITERATION FILTER_PARALLEL_BATCH_SIZE \
    FILTER_CLASSIFIER_PATH FILTER_SAVE_IMAGES ALLOW_FILTER_FALLBACK \
    DEVICE MEM USE_GPU RUN_LOCAL QUEUE_CONTINUATIONS \
    MAX_RESUME_ROUNDS OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS \
    CUDA_VISIBLE_DEVICES CSTIMS_SLURM_WRAPPER; do
    write_export_if_set_to_script "${submit_script}" "${name}"
  done

  cat >> "${submit_script}" <<EOF

echo "Continuing refit-robust selection at \$(date --iso-8601=seconds)"
echo "Output root: \${CONTINUE_OUTPUT_ROOT}"
echo "Resume round: \${RESUME_ROUND}"

bash "${SCRIPT_DIR}/run_refit_robust_selection_100k.sh"
EOF

  chmod +x "${submit_script}"

  echo "Queueing continuation after job ${dependency_job_id}"
  echo "  continuation_script=${submit_script}"
  sbatch --dependency="afterany:${dependency_job_id}" "${submit_script}"
}

if [[ -n "${CONTINUE_OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="${CONTINUE_OUTPUT_ROOT}"
  echo "Continuation mode"
  echo "Output root: ${OUTPUT_ROOT}"
  echo "Resume round: ${RESUME_ROUND}/${MAX_RESUME_ROUNDS}"

  if check_complete "${OUTPUT_ROOT}"; then
    echo "Run is complete; no further resume job needed."
    exit 0
  fi

  if (( RESUME_ROUND > MAX_RESUME_ROUNDS )); then
    echo "Reached MAX_RESUME_ROUNDS=${MAX_RESUME_ROUNDS}." >&2
    exit 4
  fi

  if [[ "${SUBMIT}" == "0" ]]; then
    build_script_cmd "${OUTPUT_ROOT}" "1"
    echo
    echo "SUBMIT=0 continuation dry run. Would submit resume job:"
    if [[ "${RUN_LOCAL}" == "1" ]]; then
      DRY_CMD=("${PYTHON_BIN}" -u "${SCRIPT_CMD[@]}")
    else
      DRY_SLURM_ARGS=(--mem "${MEM}")
      if [[ "${USE_GPU}" == "1" ]]; then
        DRY_SLURM_ARGS=(--gpu "${DRY_SLURM_ARGS[@]}")
      fi
      DRY_CMD=("${PYTHON_BIN}" "${START_SLURM}" "${DRY_SLURM_ARGS[@]}" "${SCRIPT_CMD[@]}")
    fi
    print_command DRY_CMD
    exit 0
  fi

  resume_log="${QUEUE_DIR}/refit_robust_resume_round${RESUME_ROUND}_${RUN_STAMP}.log"
  submit_refit_job "${OUTPUT_ROOT}" "1" "${resume_log}"
  echo "  resume_job_id=${SUBMITTED_JOB_ID}"

  if [[ "${QUEUE_CONTINUATIONS}" == "1" && "${RUN_LOCAL}" != "1" ]]; then
    queue_continuation "${OUTPUT_ROOT}" "${SUBMITTED_JOB_ID}" "$((RESUME_ROUND + 1))"
  fi
  exit 0
fi

echo "Refit-robust selection queue"
echo "Repo: ${ROOT}"
echo "Run stamp: ${RUN_STAMP}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Source run: ${SOURCE_RUN}"
echo "Pool feature dir: ${POOL_FEATURE_DIR:-<source-run default>}"
echo "Model set: ${MODEL_SET}"
echo "Track: ${TRACK}"
echo "Target size: ${TARGET_SIZE}"
echo "Top-k proxy: ${TOP_K_PROXY}"
echo "Refit pool/val: ${REFIT_POOL_SIZE}/${REFIT_VAL_SIZE}"
echo "Image filter: ${IMAGE_FILTER}"
if [[ "${IMAGE_FILTER}" == "1" ]]; then
  echo "Filter max attempts per iteration: ${FILTER_MAX_ATTEMPTS_PER_ITERATION}"
fi
echo "Queue continuations: ${QUEUE_CONTINUATIONS}"
echo "Queue dir: ${QUEUE_DIR}"

if [[ "${SUBMIT}" == "0" ]]; then
  build_script_cmd "${OUTPUT_ROOT}" "0"
  echo
  echo "SUBMIT=0 dry run."
  if [[ "${RUN_LOCAL}" == "1" ]]; then
    DRY_CMD=("${PYTHON_BIN}" -u "${SCRIPT_CMD[@]}")
  else
    DRY_SLURM_ARGS=(--mem "${MEM}")
    if [[ "${USE_GPU}" == "1" ]]; then
      DRY_SLURM_ARGS=(--gpu "${DRY_SLURM_ARGS[@]}")
    fi
    DRY_CMD=("${PYTHON_BIN}" "${START_SLURM}" "${DRY_SLURM_ARGS[@]}" "${SCRIPT_CMD[@]}")
  fi
  print_command DRY_CMD
  exit 0
fi

initial_log="${QUEUE_DIR}/refit_robust_initial_submit_${RUN_STAMP}.log"
submit_refit_job "${OUTPUT_ROOT}" "0" "${initial_log}"
echo "  initial_job_id=${SUBMITTED_JOB_ID}"

if [[ "${QUEUE_CONTINUATIONS}" == "1" && "${RUN_LOCAL}" != "1" ]]; then
  continuation_log="${QUEUE_DIR}/refit_robust_initial_continuation_${RUN_STAMP}.log"
  set +e
  queue_continuation "${OUTPUT_ROOT}" "${SUBMITTED_JOB_ID}" "1" \
    2>&1 | tee "${continuation_log}"
  continuation_status="${PIPESTATUS[0]}"
  set -e
  if [[ "${continuation_status}" -ne 0 ]]; then
    echo "Continuation queueing failed; see ${continuation_log}" >&2
    exit "${continuation_status}"
  fi
fi
