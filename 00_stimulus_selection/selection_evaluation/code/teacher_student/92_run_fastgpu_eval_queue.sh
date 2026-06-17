#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/home_roth/cstims_share"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_LABEL="${RUN_LABEL:-fastgpu_ns20_rand100_rr3}"
REFIT_SIZE="${REFIT_SIZE:-1000}"
REFIT_VAL_SIZE="${REFIT_VAL_SIZE:-200}"
MAX_REFIT_POOL_SIZE="${MAX_REFIT_POOL_SIZE:-10000}"
N_NOISE_SAMPLES="${N_NOISE_SAMPLES:-20}"
N_RANDOM_SUBSETS="${N_RANDOM_SUBSETS:-100}"
N_REFIT_REPEATS="${N_REFIT_REPEATS:-3}"
MAX_PROCS="${MAX_PROCS:-8}"
GPU_COUNT="${GPU_COUNT:-8}"
GPU_OFFSET="${GPU_OFFSET:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

FEATURE_PAYLOAD_RUN="${FEATURE_PAYLOAD_RUN:-00_stimulus_selection/feature_method_sweep/results/sota_pool100k_seed42_raw_sub01_rawenc_w05_meanmin_attenuation_20260615_153656}"
FEATURE_SELECTION_ROOT="${FEATURE_PAYLOAD_RUN}/payloads"

FINAL_ROOT="00_stimulus_selection/selection_evaluation/final_stimuli_recovery/teacher_student"
FEATURE_ROOT="00_stimulus_selection/selection_evaluation/feature_method_sweep_recovery/teacher_student"
FINAL_RESULTS_DIR="${FINAL_ROOT}/results"
FINAL_FIGURES_DIR="${FINAL_ROOT}/figures"
FINAL_LOG_DIR="${FINAL_ROOT}/logs"
FEATURE_OUT_RUN="${FEATURE_ROOT}/results/$(basename "$FEATURE_PAYLOAD_RUN")"
FEATURE_FIGURES_DIR="${FEATURE_ROOT}/figures"
FEATURE_LOG_DIR="${FEATURE_ROOT}/logs"
QUEUE_LOG_DIR="00_stimulus_selection/selection_evaluation/code/teacher_student/logs"
mkdir -p "$FINAL_RESULTS_DIR" "$FINAL_FIGURES_DIR" "$FINAL_LOG_DIR"
mkdir -p "$FEATURE_OUT_RUN" "$FEATURE_FIGURES_DIR" "$FEATURE_LOG_DIR" "$QUEUE_LOG_DIR"

export LD_LIBRARY_PATH="/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-}"
export PYTHON="/data/home_roth/miniforge3/bin/python"
export SCRIPT="00_stimulus_selection/selection_evaluation/code/teacher_student/01_compute_independent_refit_rdm_recovery.py"
export FINAL_PLOT="00_stimulus_selection/selection_evaluation/code/teacher_student/11_plot_final_stimuli_recovery.py"
export FEATURE_PLOT="00_stimulus_selection/selection_evaluation/code/teacher_student/21_plot_feature_method_sweep_recovery.py"
export RANDOM_FEATURE_DIR="shared/cache_or_heavy/natural_pool_subset_100k_seed42"
export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
export TS RUN_LABEL REFIT_SIZE REFIT_VAL_SIZE MAX_REFIT_POOL_SIZE
export N_NOISE_SAMPLES N_RANDOM_SUBSETS N_REFIT_REPEATS
export MAX_PROCS GPU_COUNT GPU_OFFSET
export FINAL_RESULTS_DIR FINAL_FIGURES_DIR FINAL_LOG_DIR
export FEATURE_SELECTION_ROOT FEATURE_OUT_RUN FEATURE_FIGURES_DIR FEATURE_LOG_DIR

FINAL_MODEL_JOBS=(
  "all_models raw 0-4"
  "all_models raw 5-9"
  "all_models raw 10-14"
  "all_models raw 15-19"
  "all_models sub-01 0-4"
  "all_models sub-01 5-9"
  "all_models sub-01 10-14"
  "all_models sub-01 15-19"
  "all_models sub-03 0-4"
  "all_models sub-03 5-9"
  "all_models sub-03 10-14"
  "all_models sub-03 15-19"
  "all_models sub-05 0-4"
  "all_models sub-05 5-9"
  "all_models sub-05 10-14"
  "all_models sub-05 15-19"
  "all_models sub-06 0-4"
  "all_models sub-06 5-9"
  "all_models sub-06 10-14"
  "all_models sub-06 15-19"
  "all_models sub-07 0-4"
  "all_models sub-07 5-9"
  "all_models sub-07 10-14"
  "all_models sub-07 15-19"
  "sota raw 0-2"
  "sota raw 3-5"
  "sota sub-01 0-2"
  "sota sub-01 3-5"
  "sota sub-03 0-2"
  "sota sub-03 3-5"
  "sota sub-05 0-2"
  "sota sub-05 3-5"
  "sota sub-06 0-2"
  "sota sub-06 3-5"
  "sota sub-07 0-2"
  "sota sub-07 3-5"
  "training_objective raw 0-4"
  "training_objective sub-01 0-4"
  "training_objective sub-03 0-4"
  "training_objective sub-05 0-4"
  "training_objective sub-06 0-4"
  "training_objective sub-07 0-4"
  "architecture raw 0-4"
  "architecture sub-01 0-4"
  "architecture sub-03 0-4"
  "architecture sub-05 0-4"
  "architecture sub-06 0-4"
  "architecture sub-07 0-4"
  "dataset raw 0-4"
  "dataset sub-01 0-4"
  "dataset sub-03 0-4"
  "dataset sub-05 0-4"
  "dataset sub-06 0-4"
  "dataset sub-07 0-4"
)

FEATURE_METHODS=(
  raw_only_mean_min
  raw_only_mean_min_no_attenuation
  sub01_only_mean_min
  sub01_only_mean_min_no_attenuation
  raw_enc_w05_mean_min
  raw_enc_w05_mean_min_no_attenuation
)

final_suffix() {
  local mode="$1"
  local mode_part
  if [ "$mode" = "eval_augmented_loo" ]; then
    mode_part="eval_augmented_loo"
  else
    mode_part="independent_refit"
  fi
  echo "teacher_student_${mode_part}_refit${REFIT_SIZE}_rdm_score_spearman_response_empcal_ns${N_NOISE_SAMPLES}_rand${N_RANDOM_SUBSETS}_rr${N_REFIT_REPEATS}_fastgpu"
}
export -f final_suffix

run_final_job() {
  local mode="$1"
  local model_set="$2"
  local track="$3"
  local teacher_indices="$4"
  local refit_repeat="$5"
  local suffix
  suffix="$(final_suffix "$mode")"
  local out_dir="${FINAL_RESULTS_DIR}/${model_set}_${suffix}"
  local safe_indices="${teacher_indices//,/plus}"
  safe_indices="${safe_indices//-/to}"
  local log="${FINAL_LOG_DIR}/fastgpu_${mode}_${TS}_${model_set}_${track}_${safe_indices}_rr${refit_repeat}.log"
  {
    echo "job_start $(date -Is) stage=final mode=${mode} model_set=${model_set} track=${track} teachers=${teacher_indices} repeat=${refit_repeat}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    "$PYTHON" -u "$SCRIPT" \
      --model-set "$model_set" \
      --tracks "$track" \
      --random-feature-dir "$RANDOM_FEATURE_DIR" \
      --n-random-images 100000 \
      --refit-pool-size "$REFIT_SIZE" \
      --refit-val-size "$REFIT_VAL_SIZE" \
      --max-refit-pool-size "$MAX_REFIT_POOL_SIZE" \
      --n-refit-repeats "$N_REFIT_REPEATS" \
      --refit-repeat-indices "$refit_repeat" \
      --n-random-subsets "$N_RANDOM_SUBSETS" \
      --n-noise-samples "$N_NOISE_SAMPLES" \
      --eval-noise-mode response \
      --fit-noise-calibration rdm_empirical \
      --eval-refit-mode "$mode" \
      --calibration-images 100 \
      --calibration-noise-samples 2 \
      --calibration-max-iter 8 \
      --corr-type spearman \
      --encoding-device cuda \
      --encoding-batch-size 1024 \
      --teacher-indices "$teacher_indices" \
      --fast-gpu-batch \
      --cache-only \
      --output-dir "$out_dir"
    echo "job_done $(date -Is) stage=final mode=${mode} model_set=${model_set} track=${track} teachers=${teacher_indices} repeat=${refit_repeat}"
  } > "$log" 2>&1
}
export -f run_final_job

run_final_stage() {
  local mode="$1"
  local suffix
  suffix="$(final_suffix "$mode")"
  local job_file="${FINAL_LOG_DIR}/fastgpu_final_${mode}_${TS}.tsv"
  : > "$job_file"
  for refit_repeat in $(seq 0 $((N_REFIT_REPEATS - 1))); do
    for job in "${FINAL_MODEL_JOBS[@]}"; do
      printf '%s\t%s\n' "$job" "$refit_repeat" >> "$job_file"
    done
  done
  echo "stage_start $(date -Is) final ${mode}; job_file=${job_file}"
  xargs --process-slot-var=JOB_SLOT -P "$MAX_PROCS" -n 4 bash -c '
    export CUDA_VISIBLE_DEVICES=$(((JOB_SLOT + GPU_OFFSET) % GPU_COUNT))
    run_final_job "$0" "$@"
  ' "$mode" < "$job_file"
  for model_set in all_models sota training_objective architecture dataset; do
    "$PYTHON" -u "$SCRIPT" \
      --model-set "$model_set" \
      --tracks raw,sub-01,sub-03,sub-05,sub-06,sub-07 \
      --random-feature-dir "$RANDOM_FEATURE_DIR" \
      --n-random-images 100000 \
      --refit-pool-size "$REFIT_SIZE" \
      --refit-val-size "$REFIT_VAL_SIZE" \
      --max-refit-pool-size "$MAX_REFIT_POOL_SIZE" \
      --n-refit-repeats "$N_REFIT_REPEATS" \
      --n-random-subsets "$N_RANDOM_SUBSETS" \
      --n-noise-samples "$N_NOISE_SAMPLES" \
      --eval-noise-mode response \
      --fit-noise-calibration rdm_empirical \
      --eval-refit-mode "$mode" \
      --calibration-images 100 \
      --calibration-noise-samples 2 \
      --calibration-max-iter 8 \
      --corr-type spearman \
      --encoding-device cuda \
      --fast-gpu-batch \
      --merge-only \
      --output-dir "${FINAL_RESULTS_DIR}/${model_set}_${suffix}"
  done
  "$PYTHON" "$FINAL_PLOT" \
    --results-root "$FINAL_RESULTS_DIR" \
    --figures-root "$FINAL_FIGURES_DIR" \
    --data-suffix "_${suffix}" \
    --name "teacher_student_recovery_final_stimuli_${RUN_LABEL}_${mode}"
  echo "stage_done $(date -Is) final ${mode}"
}

feature_results_name() {
  local mode="$1"
  local mode_part
  if [ "$mode" = "eval_augmented_loo" ]; then
    mode_part="eval_augmented_loo"
  else
    mode_part="independent_refit"
  fi
  echo "teacher_student_${mode_part}_refit${REFIT_SIZE}_rdm_score_spearman_response_empcal_ns${N_NOISE_SAMPLES}_rand${N_RANDOM_SUBSETS}_rr${N_REFIT_REPEATS}_fastgpu"
}
export -f feature_results_name

run_feature_job() {
  local mode="$1"
  local method="$2"
  local teacher_indices="$3"
  local refit_repeat="$4"
  local name
  name="$(feature_results_name "$mode")"
  local out_dir="${FEATURE_OUT_RUN}/${name}/${method}"
  local safe_indices="${teacher_indices//,/plus}"
  safe_indices="${safe_indices//-/to}"
  local log="${FEATURE_LOG_DIR}/fastgpu_${mode}_${TS}_${method}_${safe_indices}_rr${refit_repeat}.log"
  {
    echo "job_start $(date -Is) stage=feature mode=${mode} method=${method} teachers=${teacher_indices} repeat=${refit_repeat}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    "$PYTHON" -u "$SCRIPT" \
      --model-set "$method" \
      --selection-root "$FEATURE_SELECTION_ROOT" \
      --tracks raw,sub-01,sub-03,sub-05,sub-06,sub-07 \
      --random-feature-dir "$RANDOM_FEATURE_DIR" \
      --n-random-images 100000 \
      --refit-pool-size "$REFIT_SIZE" \
      --refit-val-size "$REFIT_VAL_SIZE" \
      --max-refit-pool-size "$MAX_REFIT_POOL_SIZE" \
      --n-refit-repeats "$N_REFIT_REPEATS" \
      --refit-repeat-indices "$refit_repeat" \
      --n-random-subsets "$N_RANDOM_SUBSETS" \
      --n-noise-samples "$N_NOISE_SAMPLES" \
      --eval-noise-mode response \
      --fit-noise-calibration rdm_empirical \
      --eval-refit-mode "$mode" \
      --calibration-images 100 \
      --calibration-noise-samples 2 \
      --calibration-max-iter 8 \
      --corr-type spearman \
      --encoding-device cuda \
      --encoding-batch-size 1024 \
      --unique-encodings \
      --teacher-indices "$teacher_indices" \
      --fast-gpu-batch \
      --cache-only \
      --output-dir "$out_dir"
    echo "job_done $(date -Is) stage=feature mode=${mode} method=${method} teachers=${teacher_indices} repeat=${refit_repeat}"
  } > "$log" 2>&1
}
export -f run_feature_job

run_feature_stage() {
  local mode="$1"
  local name
  name="$(feature_results_name "$mode")"
  local job_file="${FEATURE_LOG_DIR}/fastgpu_feature_${mode}_${TS}.tsv"
  : > "$job_file"
  for refit_repeat in $(seq 0 $((N_REFIT_REPEATS - 1))); do
    for method in "${FEATURE_METHODS[@]}"; do
      printf '%s\t0-2\t%s\n' "$method" "$refit_repeat" >> "$job_file"
      printf '%s\t3-5\t%s\n' "$method" "$refit_repeat" >> "$job_file"
    done
  done
  echo "stage_start $(date -Is) feature ${mode}; payload=${FEATURE_PAYLOAD_RUN}; job_file=${job_file}"
  xargs --process-slot-var=JOB_SLOT -P "$MAX_PROCS" -n 3 bash -c '
    export CUDA_VISIBLE_DEVICES=$(((JOB_SLOT + GPU_OFFSET) % GPU_COUNT))
    run_feature_job "$0" "$@"
  ' "$mode" < "$job_file"
  for method in "${FEATURE_METHODS[@]}"; do
    "$PYTHON" -u "$SCRIPT" \
      --model-set "$method" \
      --selection-root "$FEATURE_SELECTION_ROOT" \
      --tracks raw,sub-01,sub-03,sub-05,sub-06,sub-07 \
      --random-feature-dir "$RANDOM_FEATURE_DIR" \
      --n-random-images 100000 \
      --refit-pool-size "$REFIT_SIZE" \
      --refit-val-size "$REFIT_VAL_SIZE" \
      --max-refit-pool-size "$MAX_REFIT_POOL_SIZE" \
      --n-refit-repeats "$N_REFIT_REPEATS" \
      --n-random-subsets "$N_RANDOM_SUBSETS" \
      --n-noise-samples "$N_NOISE_SAMPLES" \
      --eval-noise-mode response \
      --fit-noise-calibration rdm_empirical \
      --eval-refit-mode "$mode" \
      --calibration-images 100 \
      --calibration-noise-samples 2 \
      --calibration-max-iter 8 \
      --corr-type spearman \
      --encoding-device cuda \
      --unique-encodings \
      --fast-gpu-batch \
      --merge-only \
      --output-dir "${FEATURE_OUT_RUN}/${name}/${method}"
  done
  "$PYTHON" "$FEATURE_PLOT" \
    --run-dir "$FEATURE_OUT_RUN" \
    --figures-root "$FEATURE_FIGURES_DIR" \
    --results-name "$name" \
    --methods "$(IFS=,; echo "${FEATURE_METHODS[*]}")" \
    --name "teacher_student_recovery_feature_method_sweep_${RUN_LABEL}_${mode}"
  echo "stage_done $(date -Is) feature ${mode}"
}

echo "queue_start $(date -Is)"
echo "RUN_LABEL=${RUN_LABEL}; REFIT_SIZE=${REFIT_SIZE}; MAX_REFIT_POOL_SIZE=${MAX_REFIT_POOL_SIZE}"
echo "N_NOISE_SAMPLES=${N_NOISE_SAMPLES}; N_RANDOM_SUBSETS=${N_RANDOM_SUBSETS}; N_REFIT_REPEATS=${N_REFIT_REPEATS}"
echo "MAX_PROCS=${MAX_PROCS}; GPU_COUNT=${GPU_COUNT}; GPU_OFFSET=${GPU_OFFSET}; threads=${OMP_NUM_THREADS}/${MKL_NUM_THREADS}/${OPENBLAS_NUM_THREADS}"
echo "FEATURE_PAYLOAD_RUN=${FEATURE_PAYLOAD_RUN}"

run_final_stage independent
run_feature_stage independent
run_final_stage eval_augmented_loo
run_feature_stage eval_augmented_loo

echo "queue_done $(date -Is)"
