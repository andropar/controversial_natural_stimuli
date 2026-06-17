#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/home_roth/cstims_share"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
FEATURE_ROOT="00_stimulus_selection/feature_method_sweep"
PAYLOAD_RUN="${FEATURE_ROOT}/results/sota_20260611_112941"
SELECTION_ROOT="${PAYLOAD_RUN}/payloads"
METHOD_ROOT="00_stimulus_selection/selection_evaluation/feature_method_sweep_recovery/teacher_student"
OUT_RUN="${METHOD_ROOT}/results/sota_20260611_112941"
FIGURES_DIR="${METHOD_ROOT}/figures"
LOG_DIR="${METHOD_ROOT}/logs"
mkdir -p "$LOG_DIR" "$OUT_RUN" "$FIGURES_DIR"

RUN_LABEL="${RUN_LABEL:-refit_sweep_ns20_rand100_rr3}"
REFIT_SIZES="${REFIT_SIZES:-100 500 1000 5000 10000}"
MAX_REFIT_POOL_SIZE="${MAX_REFIT_POOL_SIZE:-10000}"
EVAL_REFIT_MODE="${EVAL_REFIT_MODE:-independent}"
N_NOISE_SAMPLES="${N_NOISE_SAMPLES:-20}"
N_RANDOM_SUBSETS="${N_RANDOM_SUBSETS:-100}"
N_REFIT_REPEATS="${N_REFIT_REPEATS:-3}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
MAX_PROCS="${MAX_PROCS:-2}"
GPU_COUNT="${GPU_COUNT:-7}"
GPU_OFFSET="${GPU_OFFSET:-5}"

export LD_LIBRARY_PATH="/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-}"
export PYTHON="/data/home_roth/miniforge3/bin/python"
export SCRIPT="00_stimulus_selection/selection_evaluation/code/teacher_student/01_compute_independent_refit_rdm_recovery.py"
export PLOT="00_stimulus_selection/selection_evaluation/code/teacher_student/21_plot_feature_method_sweep_recovery.py"
export RANDOM_FEATURE_DIR="shared/cache_or_heavy/natural_pool_subset_100k_seed42"
export SELECTION_ROOT OUT_RUN FIGURES_DIR LOG_DIR RUN_LABEL TS
export EVAL_REFIT_MODE N_NOISE_SAMPLES N_RANDOM_SUBSETS N_REFIT_REPEATS
export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
export GPU_COUNT GPU_OFFSET MAX_REFIT_POOL_SIZE

BASE_JOB_FILE="${LOG_DIR}/teacher_student_feature_refit_sweep_base_jobs_${TS}.tsv"
JOB_FILE="${LOG_DIR}/teacher_student_feature_refit_sweep_${RUN_LABEL}_jobs_${TS}.tsv"

cat > "$BASE_JOB_FILE" <<'JOBS'
raw_only_mean_min 0-2
raw_only_mean_min 3-5
sub01_only_mean_min 0-2
sub01_only_mean_min 3-5
raw_enc_w05_mean_min 0-2
raw_enc_w05_mean_min 3-5
paper_effective_identity_sub01_mean_min_no_attenuation 0-2
paper_effective_identity_sub01_mean_min_no_attenuation 3-5
JOBS

: > "$JOB_FILE"
for refit_size in $REFIT_SIZES; do
  refit_val_size=$((refit_size / 5))
  if [ "$refit_val_size" -lt 1 ]; then
    refit_val_size=1
  fi
  for refit_repeat in $(seq 0 $((N_REFIT_REPEATS - 1))); do
    awk -v refit_size="$refit_size" -v refit_val_size="$refit_val_size" -v refit_repeat="$refit_repeat" \
      'NF {print $0 "\t" refit_size "\t" refit_val_size "\t" refit_repeat}' \
      "$BASE_JOB_FILE" >> "$JOB_FILE"
  done
done

results_name() {
  local refit_size="$1"
  local mode_part
  if [ "$EVAL_REFIT_MODE" = "eval_augmented_loo" ]; then
    mode_part="eval_augmented_loo"
  else
    mode_part="independent_refit"
  fi
  echo "teacher_student_${mode_part}_refit${refit_size}_rdm_score_spearman_response_empcal_ns${N_NOISE_SAMPLES}_rand${N_RANDOM_SUBSETS}_rr${N_REFIT_REPEATS}_hybrid"
}
export -f results_name

run_job() {
  local method="$1"
  local teacher_indices="$2"
  local refit_size="$3"
  local refit_val_size="$4"
  local refit_repeat="$5"
  local name
  name="$(results_name "$refit_size")"
  local out_root="${OUT_RUN}/${name}"
  local out_dir="${out_root}/${method}"
  local safe_indices="${teacher_indices//,/plus}"
  safe_indices="${safe_indices//-/to}"
  local log="${LOG_DIR}/teacher_student_feature_refit_sweep_${RUN_LABEL}_${TS}_${method}_refit${refit_size}_${safe_indices}_repeat${refit_repeat}.log"
  {
    echo "job_start $(date -Is) method=${method} teacher_indices=${teacher_indices} refit_size=${refit_size} refit_val_size=${refit_val_size} refit_repeat=${refit_repeat}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    "$PYTHON" -u "$SCRIPT" \
      --model-set "$method" \
      --selection-root "$SELECTION_ROOT" \
      --tracks raw,sub-01,sub-03,sub-05,sub-06,sub-07 \
      --random-feature-dir "$RANDOM_FEATURE_DIR" \
      --n-random-images 100000 \
      --refit-pool-size "$refit_size" \
      --refit-val-size "$refit_val_size" \
      --max-refit-pool-size "$MAX_REFIT_POOL_SIZE" \
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
      --encoding-batch-size 1024 \
      --unique-encodings \
      --teacher-indices "$teacher_indices" \
      --cache-only \
      --output-dir "$out_dir"
    echo "job_done $(date -Is) method=${method} teacher_indices=${teacher_indices} refit_size=${refit_size} refit_repeat=${refit_repeat}"
  } > "$log" 2>&1
}
export -f run_job

echo "Feature-method teacher/student refit sweep ${RUN_LABEL} started $(date -Is)"
echo "Job file: ${JOB_FILE}"
echo "REFIT_SIZES=${REFIT_SIZES}; MAX_REFIT_POOL_SIZE=${MAX_REFIT_POOL_SIZE}"
echo "N_NOISE_SAMPLES=${N_NOISE_SAMPLES}; N_RANDOM_SUBSETS=${N_RANDOM_SUBSETS}; N_REFIT_REPEATS=${N_REFIT_REPEATS}"
echo "MAX_PROCS=${MAX_PROCS}; GPU_COUNT=${GPU_COUNT}; GPU_OFFSET=${GPU_OFFSET}; threads=${OMP_NUM_THREADS}/${MKL_NUM_THREADS}/${OPENBLAS_NUM_THREADS}"

xargs --process-slot-var=JOB_SLOT -P "$MAX_PROCS" -n 5 bash -c '
  export CUDA_VISIBLE_DEVICES=$(((JOB_SLOT + GPU_OFFSET) % GPU_COUNT))
  run_job "$@"
' _ < "$JOB_FILE"

for refit_size in $REFIT_SIZES; do
  refit_val_size=$((refit_size / 5))
  if [ "$refit_val_size" -lt 1 ]; then
    refit_val_size=1
  fi
  name="$(results_name "$refit_size")"
  out_root="${OUT_RUN}/${name}"
  for method in \
    raw_only_mean_min \
    sub01_only_mean_min \
    raw_enc_w05_mean_min \
    paper_effective_identity_sub01_mean_min_no_attenuation
  do
    out_dir="${out_root}/${method}"
    "$PYTHON" -u "$SCRIPT" \
      --model-set "$method" \
      --selection-root "$SELECTION_ROOT" \
      --tracks raw,sub-01,sub-03,sub-05,sub-06,sub-07 \
      --random-feature-dir "$RANDOM_FEATURE_DIR" \
      --n-random-images 100000 \
      --refit-pool-size "$refit_size" \
      --refit-val-size "$refit_val_size" \
      --max-refit-pool-size "$MAX_REFIT_POOL_SIZE" \
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
      --unique-encodings \
      --merge-only \
      --output-dir "$out_dir"
  done
  "$PYTHON" "$PLOT" \
    --run-dir "$OUT_RUN" \
    --figures-root "$FIGURES_DIR" \
    --results-name "$name" \
    --name "teacher_student_recovery_feature_method_sweep_${RUN_LABEL}_refit${refit_size}"
done

echo "Feature-method teacher/student refit sweep ${RUN_LABEL} complete $(date -Is)"
