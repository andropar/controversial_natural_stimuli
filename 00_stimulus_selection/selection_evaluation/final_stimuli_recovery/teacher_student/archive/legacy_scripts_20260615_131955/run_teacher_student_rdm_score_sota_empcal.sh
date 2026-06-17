#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/home_roth/cstims_share"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="00_stimulus_selection/selection_evaluation/noisy_by_clean_recovery/logs"
mkdir -p "$LOG_DIR"
JOB_FILE="$LOG_DIR/teacher_student_rdm_score_sota_empcal_jobs_${TS}.tsv"

cat > "$JOB_FILE" <<'JOBS'
sota raw 0-2
sota raw 3-5
sota sub-01 0-2
sota sub-01 3-5
sota sub-03 0-2
sota sub-03 3-5
sota sub-05 0-2
sota sub-05 3-5
sota sub-06 0-2
sota sub-06 3-5
sota sub-07 0-2
sota sub-07 3-5
JOBS

export LD_LIBRARY_PATH="/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-}"
export PYTHON="/data/home_roth/miniforge3/bin/python"
export SCRIPT="00_stimulus_selection/selection_evaluation/noisy_by_clean_recovery/code/compute_teacher_student_independent_refit_rdm_score.py"
export PLOT="00_stimulus_selection/selection_evaluation/noisy_by_clean_recovery/code/plot_teacher_student_rdm_score_curves.py"
export RANDOM_FEATURE_DIR="shared/cache_or_heavy/natural_pool_subset_100k_seed42"
export DATA_SUFFIX="teacher_student_independent_refit_1k_rdm_score_spearman_response_empcal"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-3}"
export MAX_PROCS="${MAX_PROCS:-8}"
export GPU_COUNT="${GPU_COUNT:-8}"
export TS LOG_DIR GPU_COUNT

run_job() {
  local model_set="$1"
  local track="$2"
  local teacher_indices="$3"
  local out_dir="00_stimulus_selection/selection_evaluation/noisy_by_clean_recovery/results/${model_set}_${DATA_SUFFIX}"
  local safe_indices="${teacher_indices//-/to}"
  local log="${LOG_DIR}/teacher_student_rdm_sota_empcal_${TS}_${track}_${safe_indices}.log"
  {
    echo "job_start $(date -Is) model_set=${model_set} track=${track} teacher_indices=${teacher_indices}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    "$PYTHON" -u "$SCRIPT" \
      --model-set "$model_set" \
      --tracks "$track" \
      --random-feature-dir "$RANDOM_FEATURE_DIR" \
      --n-random-images 100000 \
      --refit-pool-size 1000 \
      --refit-val-size 200 \
      --n-random-subsets 20 \
      --n-noise-samples 5 \
      --eval-noise-mode response \
      --fit-noise-calibration rdm_empirical \
      --calibration-images 100 \
      --calibration-noise-samples 2 \
      --calibration-max-iter 8 \
      --corr-type spearman \
      --encoding-device cuda \
      --teacher-indices "$teacher_indices" \
      --cache-only \
      --output-dir "$out_dir"
    echo "job_done $(date -Is) model_set=${model_set} track=${track} teacher_indices=${teacher_indices}"
  } > "$log" 2>&1
}
export -f run_job

echo "SOTA empirical calibrated run ${TS}"
echo "Job file: ${JOB_FILE}"
echo "MAX_PROCS=${MAX_PROCS}; GPU_COUNT=${GPU_COUNT}; OMP/MKL/OPENBLAS=${OMP_NUM_THREADS}/${MKL_NUM_THREADS}/${OPENBLAS_NUM_THREADS}"

xargs --process-slot-var=JOB_SLOT -P "$MAX_PROCS" -n 3 bash -c '
  export CUDA_VISIBLE_DEVICES=$((JOB_SLOT % GPU_COUNT))
  run_job "$@"
' _ < "$JOB_FILE"

out_dir="00_stimulus_selection/selection_evaluation/noisy_by_clean_recovery/results/sota_${DATA_SUFFIX}"
"$PYTHON" -u "$SCRIPT" \
  --model-set sota \
  --tracks raw,sub-01,sub-03,sub-05,sub-06,sub-07 \
  --random-feature-dir "$RANDOM_FEATURE_DIR" \
  --n-random-images 100000 \
  --refit-pool-size 1000 \
  --refit-val-size 200 \
  --n-random-subsets 20 \
  --n-noise-samples 5 \
  --eval-noise-mode response \
  --fit-noise-calibration rdm_empirical \
  --calibration-images 100 \
  --calibration-noise-samples 2 \
  --calibration-max-iter 8 \
  --corr-type spearman \
  --encoding-device cuda \
  --merge-only \
  --output-dir "$out_dir"

"$PYTHON" "$PLOT" --data-suffix "_${DATA_SUFFIX}" || true
echo "SOTA empirical calibrated run complete $(date -Is)"
