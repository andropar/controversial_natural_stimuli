#!/bin/bash
#
# Submit nested pool-size feature-method sweeps for the non-SOTA model sets and
# immediately queue one resume submitter behind each sweep.
#
# Usage on Raven:
#   bash 00_stimulus_selection/feature_method_sweep/code/queue_other_model_set_pool_size_sweeps_slurm.sh
#
# By default this submits four initial sweeps now:
#   all_models, training_objective, architecture, dataset
#
# For each initial sweep, it parses the Slurm job id from the submission output
# and queues queue_pool_size_sweep_resume_slurm.sh with afterany:<initial_job_id>.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-pool_size_sweep_new_methods}"
MODEL_SETS="${MODEL_SETS:-all_models,training_objective,architecture,dataset}"

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
SUBMIT_RESUME="${SUBMIT_RESUME:-1}"
SUBMIT="${SUBMIT:-1}"

RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/00_stimulus_selection/feature_method_sweep/results}"
QUEUE_DIR="${QUEUE_DIR:-${RESULTS_ROOT}/other_model_set_pool_sweep_queue_${RUN_STAMP}}"
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

echo "Submitting other-model-set pool-size sweeps"
echo "Repo: ${REPO_ROOT}"
echo "Run stamp: ${RUN_STAMP}"
echo "Model sets: ${MODEL_SETS}"
echo "Methods: ${METHODS}"
echo "Pool sizes: ${POOL_SIZES}"
echo "Shared encodings: ${SHARED_ENCODINGS}"
echo "Image filter: ${IMAGE_FILTER}"
echo "Filter max attempts per iteration: ${FILTER_MAX_ATTEMPTS_PER_ITERATION}"
echo "Queue dir: ${QUEUE_DIR}"

if [[ "${SUBMIT}" == "0" ]]; then
  echo
  echo "SUBMIT=0 dry run. Would submit:"
  for model_set in ${MODEL_SETS//,/ }; do
    output_root="${RESULTS_ROOT}/${model_set}_${RUN_TAG}_${RUN_STAMP}"
    echo "  MODEL_SET=${model_set} OUTPUT_ROOT=${output_root}"
  done
  exit 0
fi

summary_path="${QUEUE_DIR}/submitted_jobs.tsv"
printf 'model_set\tinitial_job_id\tresume_submitter_job_id\toutput_root\n' > "${summary_path}"

for model_set in ${MODEL_SETS//,/ }; do
  output_root="${RESULTS_ROOT}/${model_set}_${RUN_TAG}_${RUN_STAMP}"
  submit_log="${QUEUE_DIR}/${model_set}_initial_submit_${RUN_STAMP}.log"
  resume_log="${QUEUE_DIR}/${model_set}_resume_submit_${RUN_STAMP}.log"

  echo
  echo "[$(date --iso-8601=seconds)] submitting initial sweep for ${model_set}"
  echo "  output_root=${output_root}"
  echo "  submit_log=${submit_log}"

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
    PYTHON_BIN="${PYTHON_BIN:-}" \
    CONDA_LIB="${CONDA_LIB:-}" \
    CSTIMS_SLURM_WRAPPER="${CSTIMS_SLURM_WRAPPER:-}" \
    POOL_FEATURE_DIR="${POOL_FEATURE_DIR:-}" \
    RANDOM_FEATURE_DIR="${RANDOM_FEATURE_DIR:-}" \
    N_RANDOM_IMAGES="${N_RANDOM_IMAGES:-}" \
    MAX_IMAGES="${MAX_IMAGES:-}" \
    N_RANDOM_SUBSETS="${N_RANDOM_SUBSETS:-}" \
    N_NOISE_SAMPLES="${N_NOISE_SAMPLES:-}" \
    N_BOOTSTRAP="${N_BOOTSTRAP:-}" \
    EXTRA_ARGS="${EXTRA_ARGS:-}" \
    bash "${SCRIPT_DIR}/run_pool_size_sweep_slurm.sh"
  ) 2>&1 | tee "${submit_log}"
  submit_status="${PIPESTATUS[0]}"
  set -e

  if [[ "${submit_status}" -ne 0 ]]; then
    echo "Initial sweep submission failed for ${model_set}; see ${submit_log}" >&2
    exit "${submit_status}"
  fi

  initial_job_id="$(extract_slurm_job_id "${submit_log}")"
  if [[ -z "${initial_job_id}" ]]; then
    echo "Could not parse initial Slurm job id for ${model_set}; see ${submit_log}" >&2
    exit 3
  fi

  echo "  initial_job_id=${initial_job_id}"

  resume_job_id=""
  if [[ "${SUBMIT_RESUME}" == "1" ]]; then
    echo "[$(date --iso-8601=seconds)] queueing resume behind ${model_set} job ${initial_job_id}"
    echo "  resume_log=${resume_log}"
    set +e
    (
      cd "${REPO_ROOT}"
      CURRENT_JOB_ID="${initial_job_id}" \
      OUTPUT_ROOT="${output_root}" \
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
      PYTHON_BIN="${PYTHON_BIN:-}" \
      CONDA_LIB="${CONDA_LIB:-}" \
      CSTIMS_SLURM_WRAPPER="${CSTIMS_SLURM_WRAPPER:-}" \
      POOL_FEATURE_DIR="${POOL_FEATURE_DIR:-}" \
      RANDOM_FEATURE_DIR="${RANDOM_FEATURE_DIR:-}" \
      N_RANDOM_IMAGES="${N_RANDOM_IMAGES:-}" \
      MAX_IMAGES="${MAX_IMAGES:-}" \
      N_RANDOM_SUBSETS="${N_RANDOM_SUBSETS:-}" \
      N_NOISE_SAMPLES="${N_NOISE_SAMPLES:-}" \
      N_BOOTSTRAP="${N_BOOTSTRAP:-}" \
      EXTRA_ARGS="${EXTRA_ARGS:-}" \
      bash "${SCRIPT_DIR}/queue_pool_size_sweep_resume_slurm.sh"
    ) 2>&1 | tee "${resume_log}"
    resume_status="${PIPESTATUS[0]}"
    set -e

    if [[ "${resume_status}" -ne 0 ]]; then
      echo "Resume queueing failed for ${model_set}; see ${resume_log}" >&2
      exit "${resume_status}"
    fi

    resume_job_id="$(extract_slurm_job_id "${resume_log}")"
    if [[ -z "${resume_job_id}" ]]; then
      echo "Warning: could not parse resume submitter job id for ${model_set}; see ${resume_log}" >&2
      resume_job_id="unknown"
    fi
    echo "  resume_submitter_job_id=${resume_job_id}"
  else
    echo "SUBMIT_RESUME=0; not queueing resume for ${model_set}"
  fi

  printf '%s\t%s\t%s\t%s\n' \
    "${model_set}" \
    "${initial_job_id}" \
    "${resume_job_id:-not_queued}" \
    "${output_root}" >> "${summary_path}"
done

echo
echo "Done. Submission summary:"
cat "${summary_path}"
