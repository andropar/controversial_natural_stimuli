#!/bin/bash
#
# Submit nested pool-size feature-method sweeps for all model sets, including SOTA,
# and keep queueing dependency-gated resume jobs until each run is complete.
#
# Usage on Raven:
#   bash 00_stimulus_selection/feature_method_sweep/code/queue_all_model_set_pool_size_sweeps_slurm.sh
#
# Defaults:
#   MODEL_SETS=sota,all_models,training_objective,architecture,dataset
#   POOL_SIZES=1k,10k,50k,100k,250k,500k,1M,5M,10M
#   FILTER_MAX_ATTEMPTS_PER_ITERATION=1000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/data/home_roth/miniforge3/bin/python" ]]; then
    PYTHON_BIN="/data/home_roth/miniforge3/bin/python"
  elif [[ -x "/u/rothj/miniforge3/bin/python" ]]; then
    PYTHON_BIN="/u/rothj/miniforge3/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-pool_size_sweep_new_methods}"
MODEL_SETS="${MODEL_SETS:-sota,all_models,training_objective,architecture,dataset}"
METHODS="${METHODS:-raw_only_mean_min,raw_only_mean_min_no_attenuation,sub01_only_mean_min,sub01_only_mean_min_no_attenuation,raw_enc_w05_mean_min,raw_enc_w05_mean_min_no_attenuation}"
POOL_SIZES="${POOL_SIZES:-1k,10k,50k,100k,250k,500k,1M,5M,10M}"
TARGET_SIZE="${TARGET_SIZE:-100}"
MAX_RAM_GB="${MAX_RAM_GB:-300}"
MEM="${MEM:-380000}"
BATCH_SIZE="${BATCH_SIZE:-5000}"
ADAPTIVE_BATCH_SIZE="${ADAPTIVE_BATCH_SIZE:-1}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-20000}"
MIN_BATCH_SIZE="${MIN_BATCH_SIZE:-512}"
PROGRESS_EVERY_BATCHES="${PROGRESS_EVERY_BATCHES:-50}"
SEED="${SEED:-42}"
SKIP_EVAL="${SKIP_EVAL:-1}"
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
QUEUE_CONTINUATIONS="${QUEUE_CONTINUATIONS:-${SUBMIT_RESUME:-1}}"
MAX_RESUME_ROUNDS="${MAX_RESUME_ROUNDS:-30}"
SUBMIT="${SUBMIT:-1}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/00_stimulus_selection/feature_method_sweep/results}"
QUEUE_DIR="${QUEUE_DIR:-${RESULTS_ROOT}/all_model_set_pool_sweep_queue_${RUN_STAMP}}"
EXTRA_ARGS_BASE="${EXTRA_ARGS:-}"
CONTINUE_MODEL_SET="${CONTINUE_MODEL_SET:-}"
CONTINUE_OUTPUT_ROOT="${CONTINUE_OUTPUT_ROOT:-}"
RESUME_ROUND="${RESUME_ROUND:-0}"

mkdir -p "${QUEUE_DIR}"

extract_slurm_job_id() {
  local log_path="$1"
  awk '
    /^[[:space:]]*[0-9]+[[:space:]]*$/ {
      id=$1
    }
    /Submitted/ {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^[0-9]+$/) {
          id=$i
        }
      }
    }
    /batch job/ {
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

append_resume_arg() {
  local args="$1"
  case " ${args} " in
    *" --resume "*) printf '%s\n' "${args}" ;;
    *) printf '%s\n' "${args:+${args} }--resume" ;;
  esac
}

check_complete() {
  local output_root="$1"
  "${PYTHON_BIN}" - "${output_root}" "${METHODS}" "${POOL_SIZES}" "${TARGET_SIZE}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

output_root = Path(sys.argv[1])
methods = [item for item in sys.argv[2].split(",") if item]
pool_sizes_arg = sys.argv[3]
target_size = int(sys.argv[4])


def parse_count(token: str) -> int:
    value = token.strip().lower().replace("_", "")
    if value.endswith("k"):
        return int(float(value[:-1]) * 1_000)
    if value.endswith("m"):
        return int(float(value[:-1]) * 1_000_000)
    return int(float(value))


def pool_dir(size: int) -> str:
    return f"pool_{size:09d}"


if pool_sizes_arg.strip():
    roots = [output_root / pool_dir(parse_count(token)) for token in pool_sizes_arg.split(",") if token.strip()]
else:
    roots = [output_root]

missing = []
short = []
for root in roots:
    for method in methods:
        path = root / "payloads" / method / "selected_indices.npy"
        if not path.exists():
            missing.append(str(path))
            continue
        try:
            n_selected = int(np.load(path, mmap_mode="r").shape[0])
        except Exception as exc:
            missing.append(f"{path} ({exc})")
            continue
        if n_selected < target_size:
            short.append(f"{path}: {n_selected}/{target_size}")

if missing or short:
    print("not complete", file=sys.stderr)
    for item in missing[:20]:
        print(f"missing: {item}", file=sys.stderr)
    for item in short[:20]:
        print(f"short: {item}", file=sys.stderr)
    if len(missing) + len(short) > 40:
        print(f"... {len(missing) + len(short) - 40} more incomplete entries", file=sys.stderr)
    sys.exit(1)

print("complete")
PY
}

submit_sweep_job() {
  local model_set="$1"
  local output_root="$2"
  local resume="$3"
  local submit_log="$4"
  local effective_extra_args="${EXTRA_ARGS_BASE}"

  if [[ "${resume}" == "1" ]]; then
    effective_extra_args="$(append_resume_arg "${effective_extra_args}")"
  fi

  set +e
  (
    cd "${REPO_ROOT}"
    MODEL_SET="${model_set}" \
    METHODS="${METHODS}" \
    POOL_SIZES="${POOL_SIZES}" \
    TARGET_SIZE="${TARGET_SIZE}" \
    MAX_RAM_GB="${MAX_RAM_GB}" \
    MEM="${MEM}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    ADAPTIVE_BATCH_SIZE="${ADAPTIVE_BATCH_SIZE}" \
    MAX_BATCH_SIZE="${MAX_BATCH_SIZE}" \
    MIN_BATCH_SIZE="${MIN_BATCH_SIZE}" \
    PROGRESS_EVERY_BATCHES="${PROGRESS_EVERY_BATCHES}" \
    SEED="${SEED}" \
    SKIP_EVAL="${SKIP_EVAL}" \
    SHARED_ENCODINGS="${SHARED_ENCODINGS}" \
    IMAGE_FILTER="${IMAGE_FILTER}" \
    FILTER_MIN_RESOLUTION="${FILTER_MIN_RESOLUTION}" \
    FILTER_NATURAL_PROB_THRESHOLD="${FILTER_NATURAL_PROB_THRESHOLD}" \
    FILTER_DOWNLOAD_TIMEOUT="${FILTER_DOWNLOAD_TIMEOUT}" \
    FILTER_MAX_ATTEMPTS_PER_ITERATION="${FILTER_MAX_ATTEMPTS_PER_ITERATION}" \
    FILTER_PARALLEL_BATCH_SIZE="${FILTER_PARALLEL_BATCH_SIZE}" \
    FILTER_CLASSIFIER_PATH="${FILTER_CLASSIFIER_PATH}" \
    FILTER_SAVE_IMAGES="${FILTER_SAVE_IMAGES}" \
    ALLOW_FILTER_FALLBACK="${ALLOW_FILTER_FALLBACK}" \
    OUTPUT_ROOT="${output_root}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    CONDA_LIB="${CONDA_LIB:-}" \
    CSTIMS_SLURM_WRAPPER="${CSTIMS_SLURM_WRAPPER:-}" \
    POOL_FEATURE_DIR="${POOL_FEATURE_DIR:-}" \
    RANDOM_FEATURE_DIR="${RANDOM_FEATURE_DIR:-}" \
    N_RANDOM_IMAGES="${N_RANDOM_IMAGES:-}" \
    MAX_IMAGES="${MAX_IMAGES:-}" \
    N_RANDOM_SUBSETS="${N_RANDOM_SUBSETS:-}" \
    N_NOISE_SAMPLES="${N_NOISE_SAMPLES:-}" \
    N_BOOTSTRAP="${N_BOOTSTRAP:-}" \
    EXTRA_ARGS="${effective_extra_args}" \
    bash "${SCRIPT_DIR}/run_pool_size_sweep_slurm.sh"
  ) 2>&1 | tee "${submit_log}"
  local submit_status="${PIPESTATUS[0]}"
  set -e

  if [[ "${submit_status}" -ne 0 ]]; then
    echo "Sweep submission failed for ${model_set}; see ${submit_log}" >&2
    exit "${submit_status}"
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
#SBATCH --job-name=continue_${model_set}_pool_sweep
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
    RUN_TAG MODEL_SETS METHODS POOL_SIZES TARGET_SIZE MAX_RAM_GB MEM BATCH_SIZE \
    ADAPTIVE_BATCH_SIZE MAX_BATCH_SIZE MIN_BATCH_SIZE PROGRESS_EVERY_BATCHES \
    SEED SKIP_EVAL SHARED_ENCODINGS IMAGE_FILTER FILTER_MIN_RESOLUTION \
    FILTER_NATURAL_PROB_THRESHOLD FILTER_DOWNLOAD_TIMEOUT \
    FILTER_MAX_ATTEMPTS_PER_ITERATION FILTER_PARALLEL_BATCH_SIZE \
    FILTER_CLASSIFIER_PATH FILTER_SAVE_IMAGES ALLOW_FILTER_FALLBACK \
    QUEUE_CONTINUATIONS MAX_RESUME_ROUNDS RESULTS_ROOT PYTHON_BIN CONDA_LIB \
    CSTIMS_SLURM_WRAPPER POOL_FEATURE_DIR RANDOM_FEATURE_DIR N_RANDOM_IMAGES \
    MAX_IMAGES N_RANDOM_SUBSETS N_NOISE_SAMPLES N_BOOTSTRAP; do
    write_export_if_set_to_script "${submit_script}" "${name}"
  done

  cat >> "${submit_script}" <<EOF

echo "Continuing ${model_set} pool-size sweep at \$(date --iso-8601=seconds)"
echo "Output root: \${CONTINUE_OUTPUT_ROOT}"
echo "Resume round: \${RESUME_ROUND}"

bash "${SCRIPT_DIR}/queue_all_model_set_pool_size_sweeps_slurm.sh"
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

  resume_log="${QUEUE_DIR}/${CONTINUE_MODEL_SET}_resume_round${RESUME_ROUND}_${RUN_STAMP}.log"
  echo "Submitting resume job for ${CONTINUE_MODEL_SET}; log=${resume_log}"
  submit_sweep_job "${CONTINUE_MODEL_SET}" "${CONTINUE_OUTPUT_ROOT}" "1" "${resume_log}"
  echo "  resume_job_id=${SUBMITTED_JOB_ID}"

  if [[ "${QUEUE_CONTINUATIONS}" == "1" ]]; then
    queue_continuation \
      "${CONTINUE_MODEL_SET}" \
      "${CONTINUE_OUTPUT_ROOT}" \
      "${SUBMITTED_JOB_ID}" \
      "$((RESUME_ROUND + 1))"
  fi
  exit 0
fi

echo "Submitting all-model-set feature-method pool-size sweeps"
echo "Repo: ${REPO_ROOT}"
echo "Run stamp: ${RUN_STAMP}"
echo "Model sets: ${MODEL_SETS}"
echo "Methods: ${METHODS}"
echo "Pool sizes: ${POOL_SIZES}"
echo "Shared encodings: ${SHARED_ENCODINGS}"
echo "Image filter: ${IMAGE_FILTER}"
echo "Filter max attempts per iteration: ${FILTER_MAX_ATTEMPTS_PER_ITERATION}"
echo "Queue continuations: ${QUEUE_CONTINUATIONS}"
echo "Queue dir: ${QUEUE_DIR}"

if [[ "${SUBMIT}" == "0" ]]; then
  echo
  echo "SUBMIT=0 dry run. Would submit initial jobs and continuation chains:"
  for model_set in ${MODEL_SETS//,/ }; do
    output_root="${RESULTS_ROOT}/${model_set}_${RUN_TAG}_${RUN_STAMP}"
    echo "  MODEL_SET=${model_set} OUTPUT_ROOT=${output_root}"
  done
  exit 0
fi

summary_path="${QUEUE_DIR}/submitted_jobs.tsv"
printf 'model_set\tinitial_job_id\tcontinuation_submitter_job_id\toutput_root\n' > "${summary_path}"

for model_set in ${MODEL_SETS//,/ }; do
  output_root="${RESULTS_ROOT}/${model_set}_${RUN_TAG}_${RUN_STAMP}"
  submit_log="${QUEUE_DIR}/${model_set}_initial_submit_${RUN_STAMP}.log"

  echo
  echo "[$(date --iso-8601=seconds)] submitting initial sweep for ${model_set}"
  echo "  output_root=${output_root}"
  echo "  submit_log=${submit_log}"

  submit_sweep_job "${model_set}" "${output_root}" "0" "${submit_log}"
  initial_job_id="${SUBMITTED_JOB_ID}"
  echo "  initial_job_id=${initial_job_id}"

  continuation_job_id="not_queued"
  if [[ "${QUEUE_CONTINUATIONS}" == "1" ]]; then
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
