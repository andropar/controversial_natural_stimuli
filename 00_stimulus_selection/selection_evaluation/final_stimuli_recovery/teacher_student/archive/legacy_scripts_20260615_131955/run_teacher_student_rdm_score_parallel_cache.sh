#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/home_roth/cstims_share"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="00_stimulus_selection/selection_evaluation/noisy_by_clean_recovery/logs"
mkdir -p "$LOG_DIR"
JOB_FILE="$LOG_DIR/teacher_student_rdm_score_parallel_jobs_${TS}.tsv"

cat > "$JOB_FILE" <<'JOBS'
all_models raw 0-4
all_models raw 5-9
all_models raw 10-14
all_models raw 15-19
all_models sub-01 0-4
all_models sub-01 5-9
all_models sub-01 10-14
all_models sub-01 15-19
all_models sub-03 0-4
all_models sub-03 5-9
all_models sub-03 10-14
all_models sub-03 15-19
all_models sub-05 0-4
all_models sub-05 5-9
all_models sub-05 10-14
all_models sub-05 15-19
all_models sub-06 0-4
all_models sub-06 5-9
all_models sub-06 10-14
all_models sub-06 15-19
all_models sub-07 0-4
all_models sub-07 5-9
all_models sub-07 10-14
all_models sub-07 15-19
sota raw 0-5
sota sub-01 0-5
sota sub-03 0-5
sota sub-05 0-5
sota sub-06 0-5
sota sub-07 0-5
training_objective raw 0-4
training_objective sub-01 0-4
training_objective sub-03 0-4
training_objective sub-05 0-4
training_objective sub-06 0-4
training_objective sub-07 0-4
architecture raw 0-4
architecture sub-01 0-4
architecture sub-03 0-4
architecture sub-05 0-4
architecture sub-06 0-4
architecture sub-07 0-4
dataset raw 0-4
dataset sub-01 0-4
dataset sub-03 0-4
dataset sub-05 0-4
dataset sub-06 0-4
dataset sub-07 0-4
JOBS

export LD_LIBRARY_PATH="/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-}"
export PYTHON="/data/home_roth/miniforge3/bin/python"
export SCRIPT="00_stimulus_selection/selection_evaluation/noisy_by_clean_recovery/code/compute_teacher_student_independent_refit_rdm_score.py"
export RANDOM_FEATURE_DIR="shared/cache_or_heavy/natural_pool_subset_100k_seed42"
export DATA_SUFFIX="${DATA_SUFFIX:-teacher_student_independent_refit_1k_rdm_score_rdm}"
export EVAL_NOISE_MODE="${EVAL_NOISE_MODE:-rdm}"
export CORR_TYPE="${CORR_TYPE:-pearson}"
export FIT_NOISE_CALIBRATION="${FIT_NOISE_CALIBRATION:-response}"
export CALIBRATION_IMAGES="${CALIBRATION_IMAGES:-100}"
export CALIBRATION_NOISE_SAMPLES="${CALIBRATION_NOISE_SAMPLES:-2}"
export CALIBRATION_MAX_ITER="${CALIBRATION_MAX_ITER:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MAX_PROCS="${MAX_PROCS:-8}"
export TS LOG_DIR

run_job() {
  local model_set="$1"
  local track="$2"
  local teacher_indices="$3"
  local out_dir="00_stimulus_selection/selection_evaluation/noisy_by_clean_recovery/results/${model_set}_${DATA_SUFFIX}"
  local safe_indices="${teacher_indices//,/plus}"
  safe_indices="${safe_indices//-/to}"
  local log="${LOG_DIR}/teacher_student_rdm_score_parallel_${TS}_${model_set}_${track}_${safe_indices}.log"
  {
    echo "job_start $(date -Is) model_set=${model_set} track=${track} teacher_indices=${teacher_indices}"
    "$PYTHON" -u "$SCRIPT" \
      --model-set "$model_set" \
      --tracks "$track" \
      --random-feature-dir "$RANDOM_FEATURE_DIR" \
      --n-random-images 100000 \
      --refit-pool-size 1000 \
      --refit-val-size 200 \
      --n-random-subsets 20 \
      --n-noise-samples 5 \
      --eval-noise-mode "$EVAL_NOISE_MODE" \
      --fit-noise-calibration "$FIT_NOISE_CALIBRATION" \
      --calibration-images "$CALIBRATION_IMAGES" \
      --calibration-noise-samples "$CALIBRATION_NOISE_SAMPLES" \
      --calibration-max-iter "$CALIBRATION_MAX_ITER" \
      --corr-type "$CORR_TYPE" \
      --encoding-device cuda \
      --teacher-indices "$teacher_indices" \
      --cache-only \
      --output-dir "$out_dir"
    echo "job_done $(date -Is) model_set=${model_set} track=${track} teacher_indices=${teacher_indices}"
  } > "$log" 2>&1
}
export -f run_job

echo "Parallel cache run ${TS}"
echo "Job file: ${JOB_FILE}"
echo "MAX_PROCS=${MAX_PROCS}; OMP/MKL/OPENBLAS threads=${OMP_NUM_THREADS}/${MKL_NUM_THREADS}/${OPENBLAS_NUM_THREADS}"
echo "DATA_SUFFIX=${DATA_SUFFIX}; EVAL_NOISE_MODE=${EVAL_NOISE_MODE}; CORR_TYPE=${CORR_TYPE}; FIT_NOISE_CALIBRATION=${FIT_NOISE_CALIBRATION}"
echo "CALIBRATION_IMAGES=${CALIBRATION_IMAGES}; CALIBRATION_NOISE_SAMPLES=${CALIBRATION_NOISE_SAMPLES}; CALIBRATION_MAX_ITER=${CALIBRATION_MAX_ITER}"

xargs -P "$MAX_PROCS" -n 3 bash -c 'run_job "$@"' _ < "$JOB_FILE"

for model_set in all_models sota training_objective architecture dataset; do
  out_dir="00_stimulus_selection/selection_evaluation/noisy_by_clean_recovery/results/${model_set}_${DATA_SUFFIX}"
  "$PYTHON" -u "$SCRIPT" \
    --model-set "$model_set" \
    --tracks raw,sub-01,sub-03,sub-05,sub-06,sub-07 \
    --random-feature-dir "$RANDOM_FEATURE_DIR" \
    --n-random-images 100000 \
    --refit-pool-size 1000 \
    --refit-val-size 200 \
    --n-random-subsets 20 \
    --n-noise-samples 5 \
    --eval-noise-mode "$EVAL_NOISE_MODE" \
    --fit-noise-calibration "$FIT_NOISE_CALIBRATION" \
    --calibration-images "$CALIBRATION_IMAGES" \
    --calibration-noise-samples "$CALIBRATION_NOISE_SAMPLES" \
    --calibration-max-iter "$CALIBRATION_MAX_ITER" \
    --corr-type "$CORR_TYPE" \
    --encoding-device cuda \
    --merge-only \
    --output-dir "$out_dir"
done

"$PYTHON" 00_stimulus_selection/selection_evaluation/noisy_by_clean_recovery/code/plot_teacher_student_rdm_score_curves.py --data-suffix "_${DATA_SUFFIX}"
echo "Parallel cache run complete $(date -Is)"
