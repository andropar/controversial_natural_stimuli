#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MODEL_SETS="${MODEL_SETS:-sota,all_models,training_objective,architecture,dataset}"
METHOD_ID="${METHOD_ID:-sub01_eval_augmented_loo_refit_robust}"
RUN_TAG="${RUN_TAG:-refit_robust_selection}"
REFIT_RUN_STAMP="${REFIT_RUN_STAMP:-${RUN_STAMP:-}}"
TARGET_SIZE="${TARGET_SIZE:-100}"
EVAL_MODES="${EVAL_MODES:-independent,eval_augmented_loo}"

FEATURE_RESULTS_ROOT="${FEATURE_RESULTS_ROOT:-${REPO_ROOT}/00_stimulus_selection/feature_method_sweep/results}"
TEACHER_ROOT="${TEACHER_ROOT:-${REPO_ROOT}/00_stimulus_selection/selection_evaluation/feature_method_sweep_recovery/teacher_student}"
OUT_RUN="${OUT_RUN:-${TEACHER_ROOT}/results/refit_robust_selection_${REFIT_RUN_STAMP:-latest}}"
STAGED_SELECTION_ROOT="${STAGED_SELECTION_ROOT:-${OUT_RUN}/payloads}"
FIGURES_DIR="${FIGURES_DIR:-${TEACHER_ROOT}/figures}"
LOG_DIR="${LOG_DIR:-${TEACHER_ROOT}/logs/refit_robust_selection_${REFIT_RUN_STAMP:-latest}}"
QUEUE_DIR="${QUEUE_DIR:-${LOG_DIR}/queue}"

PYTHON="${PYTHON:-/u/rothj/conda-envs/deepjuice/bin/python}"
SCRIPT="${SCRIPT:-${SCRIPT_DIR}/01_compute_independent_refit_rdm_recovery.py}"
PLOT_SCRIPT="${PLOT_SCRIPT:-${SCRIPT_DIR}/21_plot_feature_method_sweep_recovery.py}"
RANDOM_FEATURE_DIR="${RANDOM_FEATURE_DIR:-shared/cache_or_heavy/natural_pool_subset_100k_seed42}"

REFIT_SIZE="${REFIT_SIZE:-1000}"
REFIT_VAL_SIZE="${REFIT_VAL_SIZE:-200}"
MAX_REFIT_POOL_SIZE="${MAX_REFIT_POOL_SIZE:-10000}"
N_NOISE_SAMPLES="${N_NOISE_SAMPLES:-20}"
N_RANDOM_SUBSETS="${N_RANDOM_SUBSETS:-100}"
N_RANDOM_IMAGES="${N_RANDOM_IMAGES:-100000}"
N_REFIT_REPEATS="${N_REFIT_REPEATS:-3}"
TRACKS="${TRACKS:-raw,sub-01,sub-03,sub-05,sub-06,sub-07}"
TEACHER_CHUNK_SIZE="${TEACHER_CHUNK_SIZE:-auto}"

GPUS_PER_JOB="${GPUS_PER_JOB:-4}"
MAX_PROCS="${MAX_PROCS:-${GPUS_PER_JOB}}"
GPU_COUNT="${GPU_COUNT:-${GPUS_PER_JOB}}"
GPU_OFFSET="${GPU_OFFSET:-0}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
EVAL_MEM="${EVAL_MEM:-240000}"
EVAL_TIME="${EVAL_TIME:-1-00:00:00}"
GPU_PARTITION="${GPU_PARTITION:-gpu1}"
GPU_QOS="${GPU_QOS:-g0001}"
GPU_ACCOUNT="${GPU_ACCOUNT:-mnpf_gpu}"

POLL_MINUTES="${POLL_MINUTES:-15}"
QUEUE_ATTEMPT="${QUEUE_ATTEMPT:-0}"
MAX_WATCH_ATTEMPTS="${MAX_WATCH_ATTEMPTS:-672}"
SUBMIT="${SUBMIT:-1}"
SUBMIT_PLOTS="${SUBMIT_PLOTS:-1}"
MODE="${MODE:-watch}"

export LD_LIBRARY_PATH="/u/rothj/conda-envs/deepjuice/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

mkdir -p "${OUT_RUN}" "${STAGED_SELECTION_ROOT}" "${FIGURES_DIR}" "${LOG_DIR}" "${QUEUE_DIR}"

detect_latest_refit_stamp() {
  local latest
  latest="$(ls -td "${FEATURE_RESULTS_ROOT}/sota_${RUN_TAG}_"* 2>/dev/null | head -1 || true)"
  if [[ -z "${latest}" ]]; then
    echo "Could not infer REFIT_RUN_STAMP; set REFIT_RUN_STAMP explicitly." >&2
    exit 2
  fi
  local base
  base="$(basename "${latest}")"
  printf '%s\n' "${base#sota_${RUN_TAG}_}"
}

if [[ -z "${REFIT_RUN_STAMP}" ]]; then
  REFIT_RUN_STAMP="$(detect_latest_refit_stamp)"
  OUT_RUN="${TEACHER_ROOT}/results/refit_robust_selection_${REFIT_RUN_STAMP}"
  STAGED_SELECTION_ROOT="${OUT_RUN}/payloads"
  LOG_DIR="${TEACHER_ROOT}/logs/refit_robust_selection_${REFIT_RUN_STAMP}"
  QUEUE_DIR="${LOG_DIR}/queue"
  mkdir -p "${OUT_RUN}" "${STAGED_SELECTION_ROOT}" "${LOG_DIR}" "${QUEUE_DIR}"
fi

model_output_root() {
  local model_set="$1"
  printf '%s/%s_%s_%s\n' "${FEATURE_RESULTS_ROOT}" "${model_set}" "${RUN_TAG}" "${REFIT_RUN_STAMP}"
}

eval_method_id() {
  local model_set="$1"
  printf '%s_%s\n' "${model_set}" "${METHOD_ID}"
}

results_name() {
  local eval_mode="$1"
  local mode_part
  case "${eval_mode}" in
    independent) mode_part="independent_refit" ;;
    eval_augmented_loo) mode_part="eval_augmented_loo" ;;
    eval_augmented_nested_loo) mode_part="eval_augmented_nested_loo" ;;
    *)
      echo "Unknown eval mode: ${eval_mode}" >&2
      exit 2
      ;;
  esac
  printf 'teacher_student_%s_refit%s_rdm_score_spearman_response_empcal_ns%s_rand%s_rr%s_fastgpu\n' \
    "${mode_part}" "${REFIT_SIZE}" "${N_NOISE_SAMPLES}" "${N_RANDOM_SUBSETS}" "${N_REFIT_REPEATS}"
}

check_refit_complete() {
  local output_root="$1"
  "${PYTHON}" - "${output_root}" "${METHOD_ID}" "${TARGET_SIZE}" <<'PY'
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
    for path in missing:
        print(f"missing: {path}", file=sys.stderr)
    sys.exit(1)

try:
    n_selected = int(np.load(method_dir / "selected_indices.npy", mmap_mode="r").shape[0])
except Exception as exc:
    print(f"could not read selected_indices.npy: {exc}", file=sys.stderr)
    sys.exit(1)
if n_selected < target_size:
    print(f"short selected_indices: {n_selected}/{target_size}", file=sys.stderr)
    sys.exit(1)

try:
    with (method_dir / "refit_robust_summary.json").open("r") as f:
        summary = json.load(f)
except Exception as exc:
    print(f"could not read refit_robust_summary.json: {exc}", file=sys.stderr)
    sys.exit(1)
if int(summary.get("n_selected", 0)) < target_size:
    print(f"short summary: {summary.get('n_selected')}/{target_size}", file=sys.stderr)
    sys.exit(1)
print(f"complete: {n_selected}/{target_size}")
PY
}

stage_payload_link() {
  local model_set="$1"
  local output_root
  output_root="$(model_output_root "${model_set}")"
  local target="${output_root}/payloads/${METHOD_ID}"
  local link="${STAGED_SELECTION_ROOT}/$(eval_method_id "${model_set}")"
  if [[ ! -e "${target}/selected_stimuli_data.pkl" ]]; then
    echo "Completed payload not found: ${target}" >&2
    exit 2
  fi
  if [[ -L "${link}" || ! -e "${link}" ]]; then
    ln -sfn "${target}" "${link}"
  elif [[ "$(readlink -f "${link}")" != "$(readlink -f "${target}")" ]]; then
    echo "Refusing to overwrite existing staged payload directory: ${link}" >&2
    exit 2
  fi
}

teacher_count() {
  local payload_dir="$1"
  "${PYTHON}" - "${payload_dir}" <<'PY'
import pickle
import sys
from pathlib import Path

with (Path(sys.argv[1]) / "selected_stimuli_data.pkl").open("rb") as f:
    payload = pickle.load(f)
print(len(payload["model_names"]))
PY
}

teacher_ranges() {
  local model_set="$1"
  local payload_dir="${STAGED_SELECTION_ROOT}/$(eval_method_id "${model_set}")"
  local n
  n="$(teacher_count "${payload_dir}")"
  local chunk_size="${TEACHER_CHUNK_SIZE}"
  if [[ "${chunk_size}" == "auto" ]]; then
    if (( n <= 6 )); then
      chunk_size=3
    else
      chunk_size=5
    fi
  fi
  local start=0
  while (( start < n )); do
    local end=$((start + chunk_size - 1))
    if (( end >= n )); then
      end=$((n - 1))
    fi
    if (( start == end )); then
      printf '%s\n' "${start}"
    else
      printf '%s-%s\n' "${start}" "${end}"
    fi
    start=$((end + 1))
  done
}

eval_methods_csv() {
  local first=1
  for model_set in ${MODEL_SETS//,/ }; do
    if (( first )); then
      first=0
    else
      printf ','
    fi
    eval_method_id "${model_set}"
  done
  printf '\n'
}

run_cache_job() {
  local eval_mode="$1"
  local model_set="$2"
  local teacher_indices="$3"
  local refit_repeat="$4"
  local eval_id
  eval_id="$(eval_method_id "${model_set}")"
  local name
  name="$(results_name "${eval_mode}")"
  local out_dir="${OUT_RUN}/${name}/${eval_id}"
  local safe_indices="${teacher_indices//,/plus}"
  safe_indices="${safe_indices//-/to}"
  local log="${LOG_DIR}/cache_${eval_mode}_${TS}_${eval_id}_${safe_indices}_rr${refit_repeat}.log"
  {
    echo "job_start $(date -Is) mode=${eval_mode} model_set=${model_set} eval_id=${eval_id} teachers=${teacher_indices} repeat=${refit_repeat}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    "${PYTHON}" -u "${SCRIPT}" \
      --model-set "${eval_id}" \
      --selection-root "${STAGED_SELECTION_ROOT}" \
      --tracks "${TRACKS}" \
      --random-feature-dir "${RANDOM_FEATURE_DIR}" \
      --n-random-images "${N_RANDOM_IMAGES}" \
      --refit-pool-size "${REFIT_SIZE}" \
      --refit-val-size "${REFIT_VAL_SIZE}" \
      --max-refit-pool-size "${MAX_REFIT_POOL_SIZE}" \
      --n-refit-repeats "${N_REFIT_REPEATS}" \
      --refit-repeat-indices "${refit_repeat}" \
      --n-random-subsets "${N_RANDOM_SUBSETS}" \
      --n-noise-samples "${N_NOISE_SAMPLES}" \
      --eval-noise-mode response \
      --fit-noise-calibration rdm_empirical \
      --eval-refit-mode "${eval_mode}" \
      --calibration-images 100 \
      --calibration-noise-samples 2 \
      --calibration-max-iter 8 \
      --corr-type spearman \
      --encoding-device cuda \
      --encoding-batch-size 1024 \
      --unique-encodings \
      --teacher-indices "${teacher_indices}" \
      --fast-gpu-batch \
      --cache-only \
      --output-dir "${out_dir}"
    echo "job_done $(date -Is) mode=${eval_mode} model_set=${model_set} eval_id=${eval_id} teachers=${teacher_indices} repeat=${refit_repeat}"
  } > "${log}" 2>&1
}
export -f run_cache_job eval_method_id results_name
export PYTHON SCRIPT STAGED_SELECTION_ROOT TRACKS RANDOM_FEATURE_DIR N_RANDOM_IMAGES
export REFIT_SIZE REFIT_VAL_SIZE MAX_REFIT_POOL_SIZE N_REFIT_REPEATS
export N_RANDOM_SUBSETS N_NOISE_SAMPLES OUT_RUN LOG_DIR TS METHOD_ID
export GPU_COUNT GPU_OFFSET

run_eval() {
  : "${EVAL_MODEL_SET:?EVAL_MODEL_SET must be set in MODE=run_eval}"
  : "${EVAL_MODE:?EVAL_MODE must be set in MODE=run_eval}"
  stage_payload_link "${EVAL_MODEL_SET}"
  local eval_id
  eval_id="$(eval_method_id "${EVAL_MODEL_SET}")"
  local name
  name="$(results_name "${EVAL_MODE}")"
  local job_file="${QUEUE_DIR}/cache_${EVAL_MODE}_${eval_id}_${TS}.tsv"
  : > "${job_file}"
  for refit_repeat in $(seq 0 $((N_REFIT_REPEATS - 1))); do
    while read -r teacher_indices; do
      [[ -z "${teacher_indices}" ]] && continue
      printf '%s\t%s\t%s\t%s\n' "${EVAL_MODE}" "${EVAL_MODEL_SET}" "${teacher_indices}" "${refit_repeat}" >> "${job_file}"
    done < <(teacher_ranges "${EVAL_MODEL_SET}")
  done
  echo "eval_start $(date -Is) model_set=${EVAL_MODEL_SET} eval_id=${eval_id} mode=${EVAL_MODE}"
  echo "job_file=${job_file}"
  echo "output_dir=${OUT_RUN}/${name}/${eval_id}"
  xargs --process-slot-var=JOB_SLOT -P "${MAX_PROCS}" -n 4 bash -c '
    export CUDA_VISIBLE_DEVICES=$(((JOB_SLOT + GPU_OFFSET) % GPU_COUNT))
    run_cache_job "$@"
  ' _ < "${job_file}"
  "${PYTHON}" -u "${SCRIPT}" \
    --model-set "${eval_id}" \
    --selection-root "${STAGED_SELECTION_ROOT}" \
    --tracks "${TRACKS}" \
    --random-feature-dir "${RANDOM_FEATURE_DIR}" \
    --n-random-images "${N_RANDOM_IMAGES}" \
    --refit-pool-size "${REFIT_SIZE}" \
    --refit-val-size "${REFIT_VAL_SIZE}" \
    --max-refit-pool-size "${MAX_REFIT_POOL_SIZE}" \
    --n-refit-repeats "${N_REFIT_REPEATS}" \
    --n-random-subsets "${N_RANDOM_SUBSETS}" \
    --n-noise-samples "${N_NOISE_SAMPLES}" \
    --eval-noise-mode response \
    --fit-noise-calibration rdm_empirical \
    --eval-refit-mode "${EVAL_MODE}" \
    --calibration-images 100 \
    --calibration-noise-samples 2 \
    --calibration-max-iter 8 \
    --corr-type spearman \
    --encoding-device cuda \
    --unique-encodings \
    --fast-gpu-batch \
    --merge-only \
    --output-dir "${OUT_RUN}/${name}/${eval_id}"
  echo "eval_done $(date -Is) model_set=${EVAL_MODEL_SET} eval_id=${eval_id} mode=${EVAL_MODE}"
}

run_plot() {
  : "${EVAL_MODE:?EVAL_MODE must be set in MODE=plot}"
  local name
  name="$(results_name "${EVAL_MODE}")"
  local methods
  methods="$(eval_methods_csv)"
  "${PYTHON}" "${PLOT_SCRIPT}" \
    --run-dir "${OUT_RUN}" \
    --figures-root "${FIGURES_DIR}" \
    --results-name "${name}" \
    --methods "${methods}" \
    --name "teacher_student_recovery_refit_robust_${REFIT_RUN_STAMP}_${EVAL_MODE}"
}

write_env_exports() {
  local script_path="$1"
  for name in \
    MODEL_SETS METHOD_ID RUN_TAG REFIT_RUN_STAMP TARGET_SIZE EVAL_MODES \
    FEATURE_RESULTS_ROOT TEACHER_ROOT OUT_RUN STAGED_SELECTION_ROOT FIGURES_DIR \
    LOG_DIR QUEUE_DIR PYTHON SCRIPT PLOT_SCRIPT RANDOM_FEATURE_DIR REFIT_SIZE \
    REFIT_VAL_SIZE MAX_REFIT_POOL_SIZE N_NOISE_SAMPLES N_RANDOM_SUBSETS \
    N_RANDOM_IMAGES N_REFIT_REPEATS TRACKS TEACHER_CHUNK_SIZE GPUS_PER_JOB \
    MAX_PROCS GPU_COUNT GPU_OFFSET CPUS_PER_TASK EVAL_MEM EVAL_TIME \
    GPU_PARTITION GPU_QOS GPU_ACCOUNT POLL_MINUTES MAX_WATCH_ATTEMPTS \
    SUBMIT SUBMIT_PLOTS OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS \
    PYTHONUNBUFFERED; do
    printf 'export %s=%q\n' "${name}" "${!name-}" >> "${script_path}"
  done
}

extract_sbatch_id() {
  awk '/Submitted batch job/ {print $4}' "$1" | tail -1
}

run_sbatch() {
  local submit_log="$1"
  shift
  local sbatch_output
  if ! sbatch_output="$(sbatch "$@" 2>&1)"; then
    printf '%s\n' "${sbatch_output}" | tee "${submit_log}" >&2
    return 1
  fi
  printf '%s\n' "${sbatch_output}" > "${submit_log}"
  printf '%s\n' "${sbatch_output}" >&2
  extract_sbatch_id "${submit_log}"
}

submit_eval_job() {
  local model_set="$1"
  local eval_mode="$2"
  local eval_id
  eval_id="$(eval_method_id "${model_set}")"
  local script_path="${QUEUE_DIR}/eval_${eval_mode}_${eval_id}_${TS}.slurm.sh"
  local log_path="${QUEUE_DIR}/eval_${eval_mode}_${eval_id}_${TS}.%j.log"
  cat > "${script_path}" <<EOF
#!/bin/bash
#SBATCH --job-name=ts_${model_set}_${eval_mode}
#SBATCH --partition=${GPU_PARTITION}
#SBATCH --qos=${GPU_QOS}
#SBATCH --account=${GPU_ACCOUNT}
#SBATCH --gres=gpu:a100:${GPUS_PER_JOB}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${EVAL_MEM}
#SBATCH --time=${EVAL_TIME}
#SBATCH --output=${log_path}
#SBATCH --error=${log_path}

set -euo pipefail
cd "${REPO_ROOT}"
EOF
  write_env_exports "${script_path}"
  printf 'export MODE=run_eval\n' >> "${script_path}"
  printf 'export EVAL_MODEL_SET=%q\n' "${model_set}" >> "${script_path}"
  printf 'export EVAL_MODE=%q\n' "${eval_mode}" >> "${script_path}"
  printf 'bash %q\n' "${BASH_SOURCE[0]}" >> "${script_path}"
  chmod +x "${script_path}"

  if [[ "${SUBMIT}" == "0" ]]; then
    echo "Would submit eval job: ${script_path}"
    return 0
  fi
  local submit_log="${QUEUE_DIR}/submit_eval_${eval_mode}_${eval_id}_${TS}.log"
  run_sbatch "${submit_log}" "${script_path}"
}

submit_plot_job() {
  local eval_mode="$1"
  shift
  local job_ids=("$@")
  local script_path="${QUEUE_DIR}/plot_${eval_mode}_${TS}.slurm.sh"
  local log_path="${QUEUE_DIR}/plot_${eval_mode}_${TS}.%j.log"
  cat > "${script_path}" <<EOF
#!/bin/bash
#SBATCH --job-name=ts_plot_${eval_mode}
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=${log_path}
#SBATCH --error=${log_path}

set -euo pipefail
cd "${REPO_ROOT}"
EOF
  write_env_exports "${script_path}"
  printf 'export MODE=plot\n' >> "${script_path}"
  printf 'export EVAL_MODE=%q\n' "${eval_mode}" >> "${script_path}"
  printf 'bash %q\n' "${BASH_SOURCE[0]}" >> "${script_path}"
  chmod +x "${script_path}"

  if [[ "${SUBMIT}" == "0" ]]; then
    echo "Would submit plot job after: ${job_ids[*]}"
    echo "  ${script_path}"
    return 0
  fi
  local dependency=""
  if ((${#job_ids[@]} > 0)); then
    local joined
    joined="$(IFS=:; echo "${job_ids[*]}")"
    dependency="--dependency=afterok:${joined}"
  fi
  local submit_log="${QUEUE_DIR}/submit_plot_${eval_mode}_${TS}.log"
  if [[ -n "${dependency}" ]]; then
    run_sbatch "${submit_log}" "${dependency}" "${script_path}"
  else
    run_sbatch "${submit_log}" "${script_path}"
  fi
}

submit_watch_job() {
  local next_attempt="$1"
  local script_path="${QUEUE_DIR}/watch_attempt${next_attempt}_${TS}.slurm.sh"
  local log_path="${QUEUE_DIR}/watch_attempt${next_attempt}_${TS}.%j.log"
  cat > "${script_path}" <<EOF
#!/bin/bash
#SBATCH --job-name=watch_refit_robust_ts
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --output=${log_path}
#SBATCH --error=${log_path}

set -euo pipefail
cd "${REPO_ROOT}"
EOF
  write_env_exports "${script_path}"
  printf 'export MODE=watch\n' >> "${script_path}"
  printf 'export QUEUE_ATTEMPT=%q\n' "${next_attempt}" >> "${script_path}"
  printf 'bash %q\n' "${BASH_SOURCE[0]}" >> "${script_path}"
  chmod +x "${script_path}"

  if [[ "${SUBMIT}" == "0" ]]; then
    echo "Would submit watch job: ${script_path}"
    return 0
  fi
  local submit_log="${QUEUE_DIR}/submit_watch_attempt${next_attempt}_${TS}.log"
  run_sbatch "${submit_log}" --begin="now+${POLL_MINUTES}minutes" "${script_path}"
}

all_complete() {
  local ok=0
  for model_set in ${MODEL_SETS//,/ }; do
    local output_root
    output_root="$(model_output_root "${model_set}")"
    echo "Checking ${model_set}: ${output_root}"
    if check_refit_complete "${output_root}"; then
      stage_payload_link "${model_set}"
    else
      ok=1
    fi
  done
  return "${ok}"
}

submit_all_eval_jobs() {
  echo "Submitting teacher-student recovery evaluations for REFIT_RUN_STAMP=${REFIT_RUN_STAMP}"
  echo "OUT_RUN=${OUT_RUN}"
  for eval_mode in ${EVAL_MODES//,/ }; do
    [[ -z "${eval_mode// /}" ]] && continue
    local job_ids=()
    for model_set in ${MODEL_SETS//,/ }; do
      local job_id
      job_id="$(submit_eval_job "${model_set}" "${eval_mode}" || true)"
      if [[ -n "${job_id}" && "${SUBMIT}" != "0" ]]; then
        job_ids+=("${job_id}")
      fi
    done
    if [[ "${SUBMIT_PLOTS}" == "1" ]]; then
      local plot_job_id
      plot_job_id="$(submit_plot_job "${eval_mode}" "${job_ids[@]}" || true)"
      if [[ -n "${plot_job_id}" ]]; then
        echo "plot_job_id_${eval_mode}=${plot_job_id}"
      fi
    fi
  done
}

run_watch() {
  echo "watch_start $(date -Is) attempt=${QUEUE_ATTEMPT}/${MAX_WATCH_ATTEMPTS}"
  echo "REFIT_RUN_STAMP=${REFIT_RUN_STAMP}"
  echo "MODEL_SETS=${MODEL_SETS}"
  if all_complete; then
    echo "All refit robust outputs complete; queueing teacher-student recovery."
    submit_all_eval_jobs
    echo "watch_done $(date -Is) queued_evals=1"
    return
  fi
  if (( QUEUE_ATTEMPT >= MAX_WATCH_ATTEMPTS )); then
    echo "Reached MAX_WATCH_ATTEMPTS=${MAX_WATCH_ATTEMPTS}; not queueing evaluations." >&2
    exit 4
  fi
  local next_attempt=$((QUEUE_ATTEMPT + 1))
  echo "Not complete yet; scheduling another watch attempt in ${POLL_MINUTES} minutes."
  local watch_job_id
  watch_job_id="$(submit_watch_job "${next_attempt}")"
  echo "watch_job_id=${watch_job_id}"
}

case "${MODE}" in
  watch)
    run_watch
    ;;
  run_eval)
    run_eval
    ;;
  plot)
    run_plot
    ;;
  *)
    echo "Unknown MODE=${MODE}" >&2
    exit 2
    ;;
esac
