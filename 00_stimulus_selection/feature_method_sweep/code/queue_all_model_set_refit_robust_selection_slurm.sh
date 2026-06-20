#!/usr/bin/env bash
#
# Submit refit-robust selection for every model set, using one fixed natural pool
# size determined by the largest feature-method-sweep POOL_SIZES entry unless
# MAX_IMAGES is explicitly set, and keep queueing dependency-gated resume jobs
# until each run is complete.
#
# Usage on Raven:
#   bash 00_stimulus_selection/feature_method_sweep/code/queue_all_model_set_refit_robust_selection_slurm.sh
#
# Defaults:
#   MODEL_SETS=sota,all_models,training_objective,architecture,dataset
#   POOL_SIZES=1k,10k,50k,100k,250k,500k,1M,5M,10M
#   MAX_RAM_GB=300
#   TARGET_SIZE=100

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

START_SLURM="${CSTIMS_SLURM_WRAPPER:-/u/rothj/laion_natural/scripts/start_as_slurm_job.py}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/data/home_roth/miniforge3/bin/python" ]]; then
    PYTHON_BIN="/data/home_roth/miniforge3/bin/python"
  elif [[ -x "/u/rothj/miniforge3/bin/python" ]]; then
    PYTHON_BIN="/u/rothj/miniforge3/bin/python"
  elif [[ -x "/u/rothj/conda-envs/laion/bin/python" ]]; then
    PYTHON_BIN="/u/rothj/conda-envs/laion/bin/python"
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
MODEL_SETS="${MODEL_SETS:-sota,all_models,training_objective,architecture,dataset}"
POOL_SIZES="${POOL_SIZES:-1k,10k,50k,100k,250k,500k,1M,5M,10M}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/00_stimulus_selection/feature_method_sweep/results}"
QUEUE_DIR="${QUEUE_DIR:-${RESULTS_ROOT}/all_model_set_refit_robust_queue_${RUN_STAMP}}"

METHOD_ID="${METHOD_ID:-sub01_eval_augmented_loo_refit_robust}"
ENV_NAME="${ENV_NAME:-raven}"
export CSTIMS_PATH_ENV="${CSTIMS_PATH_ENV:-${ENV_NAME}}"
TRACK="${TRACK:-sub-01}"
ENCODING_ROI_SUBSET="${ENCODING_ROI_SUBSET:-hlvis}"
UNIQUE_ENCODINGS="${UNIQUE_ENCODINGS:-0}"
TARGET_SIZE="${TARGET_SIZE:-100}"
INIT_SIZE="${INIT_SIZE:-3}"
SEED="${SEED:-42}"
METRIC="${METRIC:-cosine}"
CORR_TYPE="${CORR_TYPE:-spearman}"
MAX_RAM_GB="${MAX_RAM_GB:-300}"
MAX_IMAGES="${MAX_IMAGES:-}"
POOL_FEATURE_DIR="${POOL_FEATURE_DIR:-}"

TARGET_DIM="${TARGET_DIM:-0}"
TOP_K_PROXY="${TOP_K_PROXY:-1024}"
RANDOM_SHORTLIST="${RANDOM_SHORTLIST:-0}"
PROXY_BATCH_SIZE="${PROXY_BATCH_SIZE:-2048}"
PROXY_NOISE_CALIB_EXAMPLES="${PROXY_NOISE_CALIB_EXAMPLES:-1000}"
PROXY_NOISE_CALIB_REPEATS="${PROXY_NOISE_CALIB_REPEATS:-100}"
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
PRECOMPUTE_BASE_KERNELS="${PRECOMPUTE_BASE_KERNELS:-0}"
ALLOW_REFIT_SELECTION_OVERLAP="${ALLOW_REFIT_SELECTION_OVERLAP:-0}"
NO_PROXY_ATTENUATION="${NO_PROXY_ATTENUATION:-0}"

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
MEM="${MEM:-64000}"
USE_GPU="${USE_GPU:-1}"
RUN_LOCAL="${RUN_LOCAL:-0}"
SUBMIT="${SUBMIT:-1}"
QUEUE_CONTINUATIONS="${QUEUE_CONTINUATIONS:-${SUBMIT_RESUME:-1}}"
MAX_RESUME_ROUNDS="${MAX_RESUME_ROUNDS:-30}"
RESUME_ROUND="${RESUME_ROUND:-0}"
CONTINUE_MODEL_SET="${CONTINUE_MODEL_SET:-}"
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

parse_count_token() {
  local token="$1"
  token="${token//_/}"
  token="${token// /}"
  token="${token,,}"
  local factor=1
  case "${token}" in
    *k)
      factor=1000
      token="${token%k}"
      ;;
    *m)
      factor=1000000
      token="${token%m}"
      ;;
  esac
  awk -v value="${token}" -v factor="${factor}" '
    BEGIN {
      if (value !~ /^[0-9]+([.][0-9]+)?$/) {
        exit 1
      }
      printf "%.0f\n", value * factor
    }
  '
}

normalize_count_value() {
  local value="$1"
  local parsed
  if ! parsed="$(parse_count_token "${value}")"; then
    echo "Invalid count value: ${value}" >&2
    exit 2
  fi
  printf '%s\n' "${parsed}"
}

max_pool_size_from_pool_sizes() {
  local pool_sizes="$1"
  local token parsed
  local max_pool_size=0
  IFS=',' read -ra tokens <<< "${pool_sizes}"
  for token in "${tokens[@]}"; do
    [[ -z "${token// /}" ]] && continue
    parsed="$(normalize_count_value "${token}")"
    if (( parsed > max_pool_size )); then
      max_pool_size="${parsed}"
    fi
  done
  if (( max_pool_size <= 0 )); then
    echo "POOL_SIZES did not contain a positive pool size: ${pool_sizes}" >&2
    exit 2
  fi
  printf '%s\n' "${max_pool_size}"
}

max_images_for_model_set() {
  local model_set="$1"
  local upper_model_set
  upper_model_set="$(printf '%s' "${model_set}" | tr '[:lower:]' '[:upper:]')"
  local model_set_override_var="MAX_IMAGES_${upper_model_set}"
  local model_set_override="${!model_set_override_var:-}"
  if [[ -n "${model_set_override}" ]]; then
    normalize_count_value "${model_set_override}"
    return
  fi
  if [[ -n "${MAX_IMAGES}" ]]; then
    normalize_count_value "${MAX_IMAGES}"
    return
  fi
  max_pool_size_from_pool_sizes "${POOL_SIZES}"
}

check_complete() {
  local output_root="$1"
  "${PYTHON_BIN}" - "${output_root}" "${METHOD_ID}" "${TARGET_SIZE}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

output_root = Path(sys.argv[1])
method_id = sys.argv[2]
target_size = int(sys.argv[3])
method_dir = output_root / "payloads" / method_id

required = [
    output_root / "run_config.json",
    method_dir / "selected_indices.npy",
    method_dir / "selection_trace.csv",
    method_dir / "candidate_scores.csv",
    method_dir / "selected_image_records.csv",
    method_dir / "selected_stimuli_data.pkl",
    method_dir / "method_config.json",
    method_dir / "refit_robust_summary.json",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    print("not complete", file=sys.stderr)
    for path in missing:
        print(f"missing: {path}", file=sys.stderr)
    sys.exit(1)

try:
    n_selected = int(np.load(method_dir / "selected_indices.npy", mmap_mode="r").shape[0])
except Exception as exc:
    print(f"not complete: could not read selected_indices.npy: {exc}", file=sys.stderr)
    sys.exit(1)
if n_selected < target_size:
    print(f"not complete: {n_selected}/{target_size}", file=sys.stderr)
    sys.exit(1)

try:
    with (method_dir / "refit_robust_summary.json").open("r") as f:
        summary = json.load(f)
except Exception as exc:
    print(f"not complete: could not read refit_robust_summary.json: {exc}", file=sys.stderr)
    sys.exit(1)
if int(summary.get("n_selected", 0)) < target_size:
    print(
        f"not complete: summary has {summary.get('n_selected')}/{target_size}",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"complete: {n_selected}/{target_size}")
PY
}

build_script_cmd() {
  local model_set="$1"
  local output_root="$2"
  local resume="$3"
  local effective_max_images
  effective_max_images="$(max_images_for_model_set "${model_set}")"
  SCRIPT_CMD=(
    "${SCRIPT_DIR}/refit_robust_selection.py"
    --output-root "${output_root}"
    --method-id "${METHOD_ID}"
    --env "${ENV_NAME}"
    --model-set "${model_set}"
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
    --proxy-noise-calib-examples "${PROXY_NOISE_CALIB_EXAMPLES}"
    --proxy-noise-calib-repeats "${PROXY_NOISE_CALIB_REPEATS}"
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
  if [[ -n "${effective_max_images}" ]]; then
    SCRIPT_CMD+=(--max-images "${effective_max_images}")
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
  if [[ "${NO_PROXY_ATTENUATION}" == "1" ]]; then
    SCRIPT_CMD+=(--no-proxy-attenuation)
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

build_submit_cmd() {
  if [[ "${RUN_LOCAL}" == "1" ]]; then
    SUBMIT_CMD=("${PYTHON_BIN}" -u "${SCRIPT_CMD[@]}")
  else
    SLURM_ARGS=(--mem "${MEM}")
    if [[ "${USE_GPU}" == "1" ]]; then
      SLURM_ARGS=(--gpu "${SLURM_ARGS[@]}")
    fi
    SUBMIT_CMD=("${PYTHON_BIN}" "${START_SLURM}" "${SLURM_ARGS[@]}" "${SCRIPT_CMD[@]}")
  fi
}

submit_refit_job() {
  local model_set="$1"
  local output_root="$2"
  local resume="$3"
  local submit_log="$4"
  build_script_cmd "${model_set}" "${output_root}" "${resume}"
  build_submit_cmd

  echo "Submitting refit-robust selection"
  echo "Model set: ${model_set}"
  echo "Output root: ${output_root}"
  echo "Resume: ${resume}"
  echo "Queue dir: ${QUEUE_DIR}"
  print_command SUBMIT_CMD

  set +e
  "${SUBMIT_CMD[@]}" 2>&1 | tee "${submit_log}"
  local submit_status="${PIPESTATUS[0]}"
  set -e

  if [[ "${submit_status}" -ne 0 ]]; then
    echo "Refit-robust submission failed for ${model_set}; see ${submit_log}" >&2
    exit "${submit_status}"
  fi

  if [[ "${RUN_LOCAL}" == "1" ]]; then
    SUBMITTED_JOB_ID="local"
    return
  fi

  SUBMITTED_JOB_ID="$(extract_slurm_job_id "${submit_log}")"
  if [[ -z "${SUBMITTED_JOB_ID}" ]]; then
    echo "Could not parse Slurm job id for ${model_set}; see ${submit_log}" >&2
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
  local model_set="$1"
  local output_root="$2"
  local dependency_job_id="$3"
  local next_round="$4"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local submit_script="${QUEUE_DIR}/${model_set}_continue_round${next_round}_after_${dependency_job_id}_${stamp}.slurm.sh"
  local submit_log="${QUEUE_DIR}/${model_set}_continue_round${next_round}_after_${dependency_job_id}_${stamp}.%j.log"

  cat > "${submit_script}" <<EOF
#!/bin/bash
#SBATCH --job-name=continue_${model_set}_refit_robust
#SBATCH --time=00:15:00
#SBATCH --mem=2G
#SBATCH --output=${submit_log}
#SBATCH --error=${submit_log}

set -euo pipefail

cd "${REPO_ROOT}"
EOF

  printf 'export CONTINUE_MODEL_SET=%q\n' "${model_set}" >> "${submit_script}"
  printf 'export CONTINUE_OUTPUT_ROOT=%q\n' "${output_root}" >> "${submit_script}"
  printf 'export RESUME_ROUND=%q\n' "${next_round}" >> "${submit_script}"
  printf 'export RUN_STAMP=%q\n' "${RUN_STAMP}" >> "${submit_script}"
  printf 'export QUEUE_DIR=%q\n' "${QUEUE_DIR}" >> "${submit_script}"
  printf 'export EXTRA_ARGS=%q\n' "${EXTRA_ARGS_BASE}" >> "${submit_script}"

  for name in \
    START_SLURM PYTHON_BIN CONDA_LIB RUN_TAG MODEL_SETS RESULTS_ROOT METHOD_ID \
    ENV_NAME CSTIMS_PATH_ENV TRACK ENCODING_ROI_SUBSET UNIQUE_ENCODINGS TARGET_SIZE INIT_SIZE \
    SEED METRIC CORR_TYPE MAX_RAM_GB MAX_IMAGES POOL_SIZES POOL_FEATURE_DIR TARGET_DIM \
    TOP_K_PROXY RANDOM_SHORTLIST PROXY_BATCH_SIZE PROXY_NOISE_CALIB_EXAMPLES \
    PROXY_NOISE_CALIB_REPEATS REFIT_POOL_SIZE REFIT_VAL_SIZE N_NOISE_SAMPLES \
    NOISE_MULT NOISE_CEILING ALPHAS FIT_NOISE_CALIBRATION CALIBRATION_IMAGES \
    CALIBRATION_NOISE_SAMPLES CALIBRATION_MAX_ITER TEACHER_AGGREGATION \
    REFIT_OBJECTIVE KERNEL_BATCH_SIZE REFIT_SCORE_WORKERS PRECOMPUTE_BASE_KERNELS \
    ALLOW_REFIT_SELECTION_OVERLAP NO_PROXY_ATTENUATION IMAGE_FILTER \
    FILTER_MIN_RESOLUTION FILTER_NATURAL_PROB_THRESHOLD FILTER_DOWNLOAD_TIMEOUT \
    FILTER_MAX_ATTEMPTS_PER_ITERATION FILTER_PARALLEL_BATCH_SIZE \
    FILTER_CLASSIFIER_PATH FILTER_SAVE_IMAGES ALLOW_FILTER_FALLBACK DEVICE MEM \
    USE_GPU RUN_LOCAL SUBMIT QUEUE_CONTINUATIONS MAX_RESUME_ROUNDS \
    OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS PYTHONUNBUFFERED \
    CUDA_VISIBLE_DEVICES CSTIMS_SLURM_WRAPPER; do
    write_export_if_set_to_script "${submit_script}" "${name}"
  done

  cat >> "${submit_script}" <<EOF

echo "Continuing ${model_set} refit-robust selection at \$(date --iso-8601=seconds)"
echo "Output root: \${CONTINUE_OUTPUT_ROOT}"
echo "Resume round: \${RESUME_ROUND}"

bash "${SCRIPT_DIR}/queue_all_model_set_refit_robust_selection_slurm.sh"
EOF

  chmod +x "${submit_script}"

  echo "Queueing continuation for ${model_set} after job ${dependency_job_id}"
  echo "  continuation_script=${submit_script}"
  sbatch --dependency="afterany:${dependency_job_id}" "${submit_script}"
}

if [[ -n "${CONTINUE_MODEL_SET}" ]]; then
  if [[ -z "${CONTINUE_OUTPUT_ROOT}" ]]; then
    echo "CONTINUE_OUTPUT_ROOT must be set in continuation mode." >&2
    exit 2
  fi

  echo "Continuation mode"
  echo "Model set: ${CONTINUE_MODEL_SET}"
  echo "Output root: ${CONTINUE_OUTPUT_ROOT}"
  echo "Resume round: ${RESUME_ROUND}/${MAX_RESUME_ROUNDS}"

  if check_complete "${CONTINUE_OUTPUT_ROOT}"; then
    echo "Run is complete; no further resume job needed."
    exit 0
  fi

  if (( RESUME_ROUND > MAX_RESUME_ROUNDS )); then
    echo "Reached MAX_RESUME_ROUNDS=${MAX_RESUME_ROUNDS} for ${CONTINUE_MODEL_SET}." >&2
    exit 4
  fi

  if [[ "${SUBMIT}" == "0" ]]; then
    build_script_cmd "${CONTINUE_MODEL_SET}" "${CONTINUE_OUTPUT_ROOT}" "1"
    build_submit_cmd
    echo
    echo "SUBMIT=0 continuation dry run. Would submit resume job:"
    print_command SUBMIT_CMD
    exit 0
  fi

  resume_log="${QUEUE_DIR}/${CONTINUE_MODEL_SET}_resume_round${RESUME_ROUND}_${RUN_STAMP}.log"
  echo "Submitting resume job for ${CONTINUE_MODEL_SET}; log=${resume_log}"
  submit_refit_job "${CONTINUE_MODEL_SET}" "${CONTINUE_OUTPUT_ROOT}" "1" "${resume_log}"
  echo "  resume_job_id=${SUBMITTED_JOB_ID}"

  if [[ "${QUEUE_CONTINUATIONS}" == "1" && "${RUN_LOCAL}" != "1" ]]; then
    queue_continuation \
      "${CONTINUE_MODEL_SET}" \
      "${CONTINUE_OUTPUT_ROOT}" \
      "${SUBMITTED_JOB_ID}" \
      "$((RESUME_ROUND + 1))"
  fi
  exit 0
fi

echo "Submitting all-model-set refit-robust selections"
echo "Repo: ${REPO_ROOT}"
echo "Run stamp: ${RUN_STAMP}"
echo "Model sets: ${MODEL_SETS}"
echo "Pool sizes: ${POOL_SIZES}"
echo "Method id: ${METHOD_ID}"
echo "Target size: ${TARGET_SIZE}"
echo "Max RAM GB: ${MAX_RAM_GB}"
echo "Max images override: ${MAX_IMAGES:-<largest POOL_SIZES entry per model set>}"
echo "Slurm mem: ${MEM}"
echo "Unique encodings: ${UNIQUE_ENCODINGS}"
echo "Image filter: ${IMAGE_FILTER}"
echo "Filter max attempts per iteration: ${FILTER_MAX_ATTEMPTS_PER_ITERATION}"
echo "Queue continuations: ${QUEUE_CONTINUATIONS}"
echo "Queue dir: ${QUEUE_DIR}"

if [[ "${SUBMIT}" == "0" ]]; then
  echo
  echo "SUBMIT=0 dry run. Would submit initial jobs and continuation chains:"
  for model_set in ${MODEL_SETS//,/ }; do
    output_root="${RESULTS_ROOT}/${model_set}_${RUN_TAG}_${RUN_STAMP}"
    build_script_cmd "${model_set}" "${output_root}" "0"
    build_submit_cmd
    echo
    echo "MODEL_SET=${model_set} OUTPUT_ROOT=${output_root}"
    echo "Effective max images: $(max_images_for_model_set "${model_set}")"
    print_command SUBMIT_CMD
  done
  exit 0
fi

summary_path="${QUEUE_DIR}/submitted_jobs.tsv"
printf 'model_set\tinitial_job_id\tcontinuation_submitter_job_id\toutput_root\n' > "${summary_path}"

for model_set in ${MODEL_SETS//,/ }; do
  output_root="${RESULTS_ROOT}/${model_set}_${RUN_TAG}_${RUN_STAMP}"
  submit_log="${QUEUE_DIR}/${model_set}_initial_submit_${RUN_STAMP}.log"

  echo
  echo "[$(date --iso-8601=seconds)] submitting initial refit-robust selection for ${model_set}"
  echo "  output_root=${output_root}"
  echo "  submit_log=${submit_log}"

  submit_refit_job "${model_set}" "${output_root}" "0" "${submit_log}"
  initial_job_id="${SUBMITTED_JOB_ID}"
  echo "  initial_job_id=${initial_job_id}"

  continuation_job_id="not_queued"
  if [[ "${QUEUE_CONTINUATIONS}" == "1" && "${RUN_LOCAL}" != "1" ]]; then
    continuation_log="${QUEUE_DIR}/${model_set}_initial_continuation_${RUN_STAMP}.log"
    set +e
    queue_continuation "${model_set}" "${output_root}" "${initial_job_id}" "1" \
      2>&1 | tee "${continuation_log}"
    continuation_status="${PIPESTATUS[0]}"
    set -e
    if [[ "${continuation_status}" -ne 0 ]]; then
      echo "Continuation queueing failed for ${model_set}; see ${continuation_log}" >&2
      exit "${continuation_status}"
    fi
    continuation_job_id="$(extract_slurm_job_id "${continuation_log}")"
    if [[ -z "${continuation_job_id}" ]]; then
      continuation_job_id="unknown"
    fi
    echo "  continuation_submitter_job_id=${continuation_job_id}"
  fi

  printf '%s\t%s\t%s\t%s\n' \
    "${model_set}" \
    "${initial_job_id}" \
    "${continuation_job_id}" \
    "${output_root}" >> "${summary_path}"
done

echo
echo "Done. Submission summary:"
cat "${summary_path}"
