#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/home_roth/cstims_share"
cd "$ROOT"

WAIT_FOR_SCREEN="${WAIT_FOR_SCREEN:-}"
if [ -n "$WAIT_FOR_SCREEN" ]; then
  echo "waiting_for_screen ${WAIT_FOR_SCREEN} $(date -Is)"
  while screen -ls | grep -q "$WAIT_FOR_SCREEN"; do
    sleep 300
  done
  echo "wait_done ${WAIT_FOR_SCREEN} $(date -Is)"
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_LABEL="${RUN_LABEL:-old_sota_apples_fastgpu_ns20_rand100_rr3}"
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

FEATURE_PAYLOAD_RUN="${FEATURE_PAYLOAD_RUN:-00_stimulus_selection/feature_method_sweep/results/sota_20260611_112941}"
FEATURE_SELECTION_ROOT="${FEATURE_PAYLOAD_RUN}/payloads"
FEATURE_ROOT="00_stimulus_selection/selection_evaluation/feature_method_sweep_recovery/teacher_student"
FEATURE_OUT_RUN="${FEATURE_ROOT}/results/$(basename "$FEATURE_PAYLOAD_RUN")"
FEATURE_FIGURES_DIR="${FEATURE_ROOT}/figures"
FEATURE_LOG_DIR="${FEATURE_ROOT}/logs"
RESULTS_NAME="teacher_student_independent_refit_refit${REFIT_SIZE}_rdm_score_spearman_response_empcal_ns${N_NOISE_SAMPLES}_rand${N_RANDOM_SUBSETS}_rr${N_REFIT_REPEATS}_fastgpu"
FIGURE_NAME="teacher_student_recovery_feature_method_sweep_$(basename "$FEATURE_PAYLOAD_RUN")_fastgpu_ns${N_NOISE_SAMPLES}_rand${N_RANDOM_SUBSETS}_rr${N_REFIT_REPEATS}_independent"
mkdir -p "$FEATURE_OUT_RUN" "$FEATURE_FIGURES_DIR" "$FEATURE_LOG_DIR"

export LD_LIBRARY_PATH="/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-}"
export PYTHON="/data/home_roth/miniforge3/bin/python"
export SCRIPT="00_stimulus_selection/selection_evaluation/code/teacher_student/01_compute_independent_refit_rdm_recovery.py"
export FEATURE_PLOT="00_stimulus_selection/selection_evaluation/code/teacher_student/21_plot_feature_method_sweep_recovery.py"
export RANDOM_FEATURE_DIR="shared/cache_or_heavy/natural_pool_subset_100k_seed42"
export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
export TS RUN_LABEL REFIT_SIZE REFIT_VAL_SIZE MAX_REFIT_POOL_SIZE
export N_NOISE_SAMPLES N_RANDOM_SUBSETS N_REFIT_REPEATS
export MAX_PROCS GPU_COUNT GPU_OFFSET
export FEATURE_SELECTION_ROOT FEATURE_OUT_RUN FEATURE_FIGURES_DIR FEATURE_LOG_DIR
export RESULTS_NAME FIGURE_NAME

FEATURE_METHODS=(
  raw_only_mean_min
  sub01_only_mean_min
  raw_enc_w05_mean_min
)

run_feature_job() {
  local method="$1"
  local teacher_indices="$2"
  local refit_repeat="$3"
  local out_dir="${FEATURE_OUT_RUN}/${RESULTS_NAME}/${method}"
  local safe_indices="${teacher_indices//,/plus}"
  safe_indices="${safe_indices//-/to}"
  local log="${FEATURE_LOG_DIR}/fastgpu_old_apples_independent_${TS}_${method}_${safe_indices}_rr${refit_repeat}.log"
  {
    echo "job_start $(date -Is) stage=old_feature_independent method=${method} teachers=${teacher_indices} repeat=${refit_repeat}"
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
      --eval-refit-mode independent \
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
    echo "job_done $(date -Is) stage=old_feature_independent method=${method} teachers=${teacher_indices} repeat=${refit_repeat}"
  } > "$log" 2>&1
}
export -f run_feature_job

run_feature_stage() {
  local job_file="${FEATURE_LOG_DIR}/fastgpu_old_apples_feature_independent_${TS}.tsv"
  : > "$job_file"
  for refit_repeat in $(seq 0 $((N_REFIT_REPEATS - 1))); do
    for method in "${FEATURE_METHODS[@]}"; do
      printf '%s\t0-2\t%s\n' "$method" "$refit_repeat" >> "$job_file"
      printf '%s\t3-5\t%s\n' "$method" "$refit_repeat" >> "$job_file"
    done
  done
  echo "stage_start $(date -Is) old_feature_independent; payload=${FEATURE_PAYLOAD_RUN}; job_file=${job_file}"
  xargs --process-slot-var=JOB_SLOT -P "$MAX_PROCS" -n 3 bash -c '
    export CUDA_VISIBLE_DEVICES=$(((JOB_SLOT + GPU_OFFSET) % GPU_COUNT))
    run_feature_job "$@"
  ' _ < "$job_file"

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
      --eval-refit-mode independent \
      --calibration-images 100 \
      --calibration-noise-samples 2 \
      --calibration-max-iter 8 \
      --corr-type spearman \
      --encoding-device cuda \
      --unique-encodings \
      --fast-gpu-batch \
      --merge-only \
      --output-dir "${FEATURE_OUT_RUN}/${RESULTS_NAME}/${method}"
  done

  "$PYTHON" "$FEATURE_PLOT" \
    --run-dir "$FEATURE_OUT_RUN" \
    --results-name "$RESULTS_NAME" \
    --figures-root "$FEATURE_FIGURES_DIR" \
    --methods "$(IFS=,; echo "${FEATURE_METHODS[*]}")" \
    --name "$FIGURE_NAME"
  echo "stage_done $(date -Is) old_feature_independent"
}

echo "queue_start $(date -Is)"
echo "RUN_LABEL=${RUN_LABEL}; FEATURE_PAYLOAD_RUN=${FEATURE_PAYLOAD_RUN}"
echo "RESULTS_NAME=${RESULTS_NAME}; methods=${FEATURE_METHODS[*]}"
run_feature_stage
echo "queue_done $(date -Is)"
