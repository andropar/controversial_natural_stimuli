#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/home_roth/cstims_share"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_LABEL="${RUN_LABEL:-ns20}"
EVAL_DIR="00_stimulus_selection/selection_evaluation/final_stimuli_recovery/teacher_student"
RESULTS_DIR="${EVAL_DIR}/results"
FIGURES_DIR="${EVAL_DIR}/figures"
LOG_DIR="${EVAL_DIR}/logs"
mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$FIGURES_DIR"
JOB_FILE="$LOG_DIR/teacher_student_rdm_score_allsets_empcal_${RUN_LABEL}_jobs_${TS}.tsv"
BASE_JOB_FILE="$LOG_DIR/teacher_student_rdm_score_allsets_empcal_${RUN_LABEL}_base_jobs_${TS}.tsv"

cat > "$BASE_JOB_FILE" <<'JOBS'
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
export SCRIPT="00_stimulus_selection/selection_evaluation/code/teacher_student/01_compute_independent_refit_rdm_recovery.py"
export PLOT="00_stimulus_selection/selection_evaluation/code/teacher_student/11_plot_final_stimuli_recovery.py"
export RANDOM_FEATURE_DIR="shared/cache_or_heavy/natural_pool_subset_100k_seed42"
export DATA_SUFFIX="${DATA_SUFFIX:-teacher_student_independent_refit_1k_rdm_score_spearman_response_empcal_ns20}"
export EVAL_REFIT_MODE="${EVAL_REFIT_MODE:-independent}"
export N_NOISE_SAMPLES="${N_NOISE_SAMPLES:-20}"
export N_RANDOM_SUBSETS="${N_RANDOM_SUBSETS:-20}"
export N_REFIT_REPEATS="${N_REFIT_REPEATS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-3}"
export MAX_PROCS="${MAX_PROCS:-8}"
export GPU_COUNT="${GPU_COUNT:-8}"
export GPU_OFFSET="${GPU_OFFSET:-0}"
export TS RUN_LABEL LOG_DIR GPU_COUNT GPU_OFFSET RESULTS_DIR FIGURES_DIR

: > "$JOB_FILE"
for refit_repeat in $(seq 0 $((N_REFIT_REPEATS - 1))); do
  awk -v refit_repeat="$refit_repeat" 'NF {print $0 "\t" refit_repeat}' "$BASE_JOB_FILE" >> "$JOB_FILE"
done

run_job() {
  local model_set="$1"
  local track="$2"
  local teacher_indices="$3"
  local refit_repeat="$4"
  local out_dir="${RESULTS_DIR}/${model_set}_${DATA_SUFFIX}"
  local safe_indices="${teacher_indices//,/plus}"
  safe_indices="${safe_indices//-/to}"
  local log="${LOG_DIR}/teacher_student_rdm_allsets_empcal_${RUN_LABEL}_${TS}_${model_set}_${track}_${safe_indices}_refit${refit_repeat}.log"
  {
    echo "job_start $(date -Is) model_set=${model_set} track=${track} teacher_indices=${teacher_indices} refit_repeat=${refit_repeat}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    "$PYTHON" -u "$SCRIPT" \
      --model-set "$model_set" \
      --tracks "$track" \
      --random-feature-dir "$RANDOM_FEATURE_DIR" \
      --n-random-images 100000 \
      --refit-pool-size 1000 \
      --refit-val-size 200 \
      --n-refit-repeats "$N_REFIT_REPEATS" \
      --refit-repeat-indices "$refit_repeat" \
      --n-random-subsets "$N_RANDOM_SUBSETS" \
      --n-noise-samples "$N_NOISE_SAMPLES" \
      --eval-noise-mode response \
      --fit-noise-calibration rdm_empirical \
      --eval-refit-mode "$EVAL_REFIT_MODE" \
      --calibration-images 100 \
      --calibration-noise-samples 2 \
      --calibration-max-iter 8 \
      --corr-type spearman \
      --encoding-device cuda \
      --teacher-indices "$teacher_indices" \
      --cache-only \
      --output-dir "$out_dir"
    echo "job_done $(date -Is) model_set=${model_set} track=${track} teacher_indices=${teacher_indices} refit_repeat=${refit_repeat}"
  } > "$log" 2>&1
}
export -f run_job

echo "All-model-set empirical calibrated ${RUN_LABEL} run ${TS}"
echo "Job file: ${JOB_FILE}"
echo "MAX_PROCS=${MAX_PROCS}; GPU_COUNT=${GPU_COUNT}; OMP/MKL/OPENBLAS=${OMP_NUM_THREADS}/${MKL_NUM_THREADS}/${OPENBLAS_NUM_THREADS}"
echo "GPU_OFFSET=${GPU_OFFSET}"
echo "N_NOISE_SAMPLES=${N_NOISE_SAMPLES}; N_RANDOM_SUBSETS=${N_RANDOM_SUBSETS}; N_REFIT_REPEATS=${N_REFIT_REPEATS}; DATA_SUFFIX=${DATA_SUFFIX}; EVAL_REFIT_MODE=${EVAL_REFIT_MODE}"

xargs --process-slot-var=JOB_SLOT -P "$MAX_PROCS" -n 4 bash -c '
  export CUDA_VISIBLE_DEVICES=$(((JOB_SLOT + GPU_OFFSET) % GPU_COUNT))
  run_job "$@"
' _ < "$JOB_FILE"

for model_set in all_models sota training_objective architecture dataset; do
  out_dir="${RESULTS_DIR}/${model_set}_${DATA_SUFFIX}"
  "$PYTHON" -u "$SCRIPT" \
    --model-set "$model_set" \
    --tracks raw,sub-01,sub-03,sub-05,sub-06,sub-07 \
    --random-feature-dir "$RANDOM_FEATURE_DIR" \
    --n-random-images 100000 \
    --refit-pool-size 1000 \
    --refit-val-size 200 \
    --n-refit-repeats "$N_REFIT_REPEATS" \
    --n-random-subsets "$N_RANDOM_SUBSETS" \
    --n-noise-samples "$N_NOISE_SAMPLES" \
    --eval-noise-mode response \
    --fit-noise-calibration rdm_empirical \
    --eval-refit-mode "$EVAL_REFIT_MODE" \
    --calibration-images 100 \
    --calibration-noise-samples 2 \
    --calibration-max-iter 8 \
    --corr-type spearman \
    --encoding-device cuda \
    --merge-only \
    --output-dir "$out_dir"
done

"$PYTHON" "$PLOT" \
  --results-root "$RESULTS_DIR" \
  --figures-root "$FIGURES_DIR" \
  --data-suffix "_${DATA_SUFFIX}" \
  --name "teacher_student_recovery_final_stimuli_${RUN_LABEL}"
echo "All-model-set empirical calibrated ${RUN_LABEL} run complete $(date -Is)"
