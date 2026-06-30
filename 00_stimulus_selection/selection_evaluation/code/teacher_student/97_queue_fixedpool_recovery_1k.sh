#!/usr/bin/env bash
set -euo pipefail

# Queue teacher/student recovery for fixed-pool feature-method selections that do
# not fit the pool_* directory layout expected by 96_queue_pool_size_sweep_recovery_1k.sh.
#
# This script builds a temporary pool-style symlink staging tree and then reuses
# the existing Slurm queue machinery. It intentionally sets TARGET_SIZE=1 because
# fixed-pool avg-subjects selections include two short payloads; the evaluator
# uses the actual selected count in each payload.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
QUEUE_SCRIPT="${SCRIPT_DIR}/96_queue_pool_size_sweep_recovery_1k.sh"

FEATURE_RESULTS_ROOT="${FEATURE_RESULTS_ROOT:-${REPO_ROOT}/00_stimulus_selection/feature_method_sweep/results}"
TEACHER_ROOT="${TEACHER_ROOT:-/ptmp/rothj/controversial_natural_stimuli/00_stimulus_selection/selection_evaluation/feature_method_sweep_recovery/teacher_student}"
STAGING_ROOT="${STAGING_ROOT:-${TEACHER_ROOT}/staging/fixedpool_missing_methods_20260622_20260623}"

REFIT_SIZE="${REFIT_SIZE:-1000}"
N_NOISE_SAMPLES="${N_NOISE_SAMPLES:-20}"
N_RANDOM_SUBSETS="${N_RANDOM_SUBSETS:-100}"
N_REFIT_REPEATS="${N_REFIT_REPEATS:-3}"
TRACKS="${TRACKS:-raw,sub-01,sub-03,sub-05,sub-06,sub-07}"
NOISE_MULTS="${NOISE_MULTS:-0.1,0.143844988829,0.206913808111,0.297635144163,0.428133239872,0.615848211066,0.88586679041,1,1.2742749857,1.83298071083,2.63665089873,3,3.79269019073,5,5.45559478117,7.84759970351,10}"
NOISE_MULTS_LABEL="${NOISE_MULTS_LABEL:-snr0p1to10}"

SUBMIT="${SUBMIT:-1}"
BUILD_NPY_CACHE="${BUILD_NPY_CACHE:-0}"
CACHE_GROUPS="${CACHE_GROUPS:-12}"
MERGE_GROUPS="${MERGE_GROUPS:-3}"
CACHE_TIME="${CACHE_TIME:-1-00:00:00}"
MERGE_TIME="${MERGE_TIME:-04:00:00}"
SUMMARY_TIME="${SUMMARY_TIME:-02:00:00}"

MAX_MIN_SOURCE_SUFFIX="fixedpool_max_min_sub01_rawenc_20260622_154443"
AVG_SUBJECTS_SOURCE_SUFFIX="fixedpool_avg_subjects_enc_mean_min_20260623_1215"

pool_dir_name() {
  case "$1" in
    10M) printf 'pool_010000000' ;;
    2.5M) printf 'pool_002500000' ;;
    *)
      echo "Unsupported fixed-pool token: $1" >&2
      return 2
      ;;
  esac
}

csv_to_words() {
  printf '%s\n' "$1" | tr ',' ' '
}

stage_group() {
  local source_suffix="$1"
  local staged_run_tag="$2"
  local staged_run_stamp="$3"
  local model_sets_csv="$4"
  local methods_csv="$5"
  local pool_token="$6"
  local pool_dir
  pool_dir="$(pool_dir_name "${pool_token}")"

  for model_set in $(csv_to_words "${model_sets_csv}"); do
    local source_root="${FEATURE_RESULTS_ROOT}/${model_set}_${source_suffix}"
    if [[ ! -d "${source_root}/payloads" ]]; then
      echo "Missing source payload root: ${source_root}/payloads" >&2
      return 1
    fi
    local staged_root="${STAGING_ROOT}/${model_set}_${staged_run_tag}_${staged_run_stamp}/${pool_dir}/payloads"
    mkdir -p "${staged_root}"
    for method_id in $(csv_to_words "${methods_csv}"); do
      local source_payload="${source_root}/payloads/${method_id}"
      if [[ ! -d "${source_payload}" ]]; then
        echo "Missing source payload: ${source_payload}" >&2
        return 1
      fi
      ln -sfn "${source_payload}" "${staged_root}/${method_id}"
    done
  done
}

queue_variant() {
  local staged_run_tag="$1"
  local staged_run_stamp="$2"
  local model_sets="$3"
  local methods="$4"
  local pool_sizes="$5"
  local variant="$6"

  local mode
  local calibration
  local results_name
  local variant_suffix
  case "${variant}" in
    independent)
      mode="independent"
      calibration="clean_to_noisy"
      results_name="teacher_student_independent_refit_refit${REFIT_SIZE}_rdm_score_spearman_response_empcal_${NOISE_MULTS_LABEL}_ns${N_NOISE_SAMPLES}_rand${N_RANDOM_SUBSETS}_rr${N_REFIT_REPEATS}_fastgpu"
      variant_suffix="independent_${NOISE_MULTS_LABEL}"
      ;;
    clean_to_noisy)
      mode="eval_augmented_loo"
      calibration="clean_to_noisy"
      results_name="teacher_student_eval_augmented_loo_clean_to_noisy_refit${REFIT_SIZE}_rdm_score_spearman_response_empcal_${NOISE_MULTS_LABEL}_ns${N_NOISE_SAMPLES}_rand${N_RANDOM_SUBSETS}_rr${N_REFIT_REPEATS}_fastgpu"
      variant_suffix="eval_augmented_loo_clean_to_noisy_${NOISE_MULTS_LABEL}"
      ;;
    noisy_to_noisy)
      mode="eval_augmented_loo"
      calibration="noisy_to_noisy"
      results_name="teacher_student_eval_augmented_loo_noisy_to_noisy_refit${REFIT_SIZE}_rdm_score_spearman_response_empcal_${NOISE_MULTS_LABEL}_ns${N_NOISE_SAMPLES}_rand${N_RANDOM_SUBSETS}_rr${N_REFIT_REPEATS}_fastgpu"
      variant_suffix="eval_augmented_loo_noisy_to_noisy_${NOISE_MULTS_LABEL}"
      ;;
    *)
      echo "Unknown variant: ${variant}" >&2
      return 2
      ;;
  esac

  local out_run="${TEACHER_ROOT}/results/${staged_run_tag}_${staged_run_stamp}_refit${REFIT_SIZE}"
  local log_dir="${TEACHER_ROOT}/logs/${staged_run_tag}_${staged_run_stamp}_refit${REFIT_SIZE}_${variant_suffix}"

  echo "queue_variant ${staged_run_tag}_${staged_run_stamp} variant=${variant}"
  env \
    SUBMIT="${SUBMIT}" \
    BUILD_NPY_CACHE="${BUILD_NPY_CACHE}" \
    FEATURE_RESULTS_ROOT="${STAGING_ROOT}" \
    TEACHER_ROOT="${TEACHER_ROOT}" \
    RUN_TAG="${staged_run_tag}" \
    RUN_STAMP="${staged_run_stamp}" \
    MODEL_SETS="${model_sets}" \
    METHODS="${methods}" \
    POOL_SIZES="${pool_sizes}" \
    TARGET_SIZE="1" \
    REFIT_SIZE="${REFIT_SIZE}" \
    N_NOISE_SAMPLES="${N_NOISE_SAMPLES}" \
    NOISE_MULTS="${NOISE_MULTS}" \
    NOISE_MULTS_LABEL="${NOISE_MULTS_LABEL}" \
    N_RANDOM_SUBSETS="${N_RANDOM_SUBSETS}" \
    N_REFIT_REPEATS="${N_REFIT_REPEATS}" \
    TRACKS="${TRACKS}" \
    EVAL_REFIT_MODE="${mode}" \
    RDM_CALIBRATION_COMPARISON="${calibration}" \
    OUT_RUN="${out_run}" \
    STAGED_SELECTION_ROOT="${out_run}/payloads" \
    RESULTS_NAME="${results_name}" \
    LOG_DIR="${log_dir}" \
    CACHE_GROUPS="${CACHE_GROUPS}" \
    MERGE_GROUPS="${MERGE_GROUPS}" \
    CACHE_TIME="${CACHE_TIME}" \
    MERGE_TIME="${MERGE_TIME}" \
    SUMMARY_TIME="${SUMMARY_TIME}" \
    MODE="submit" \
    bash "${QUEUE_SCRIPT}"
}

queue_group() {
  local source_suffix="$1"
  local staged_run_tag="$2"
  local staged_run_stamp="$3"
  local model_sets="$4"
  local methods="$5"
  local pool_sizes="$6"

  stage_group "${source_suffix}" "${staged_run_tag}" "${staged_run_stamp}" "${model_sets}" "${methods}" "${pool_sizes}"
  queue_variant "${staged_run_tag}" "${staged_run_stamp}" "${model_sets}" "${methods}" "${pool_sizes}" independent
  queue_variant "${staged_run_tag}" "${staged_run_stamp}" "${model_sets}" "${methods}" "${pool_sizes}" clean_to_noisy
  queue_variant "${staged_run_tag}" "${staged_run_stamp}" "${model_sets}" "${methods}" "${pool_sizes}" noisy_to_noisy
}

main() {
  mkdir -p "${STAGING_ROOT}"

  queue_group \
    "${MAX_MIN_SOURCE_SUFFIX}" \
    "fixedpool_max_min_sub01_rawenc" \
    "20260622_154443" \
    "sota,training_objective,architecture,dataset" \
    "sub01_only_max_min,raw_enc_w05_max_min" \
    "10M"

  queue_group \
    "${MAX_MIN_SOURCE_SUFFIX}" \
    "fixedpool_max_min_sub01_rawenc_all_models" \
    "20260622_154443" \
    "all_models" \
    "sub01_only_max_min,raw_enc_w05_max_min" \
    "2.5M"

  queue_group \
    "${AVG_SUBJECTS_SOURCE_SUFFIX}" \
    "fixedpool_avg_subjects_enc_mean_min" \
    "20260623_1215" \
    "sota,training_objective,architecture,dataset,all_models" \
    "avg_subjects_enc_mean_min" \
    "10M"
}

main "$@"
