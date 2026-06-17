#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/home_roth/cstims_share"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_LABEL="${RUN_LABEL:-ns20}"
FEATURE_ROOT="00_stimulus_selection/feature_method_sweep"
PAYLOAD_RUN="${FEATURE_ROOT}/results/sota_20260611_112941"
SELECTION_ROOT="${PAYLOAD_RUN}/payloads"
OUT_NAME="${OUT_NAME:-teacher_student_rdm_score_spearman_response_empcal_ns20}"
METHOD_ROOT="00_stimulus_selection/selection_evaluation/feature_method_sweep_recovery/teacher_student"
OUT_RUN="${METHOD_ROOT}/results/sota_20260611_112941"
OUT_ROOT="${OUT_RUN}/${OUT_NAME}"
FIGURES_DIR="${METHOD_ROOT}/figures"
LOG_DIR="${METHOD_ROOT}/logs"
mkdir -p "$LOG_DIR" "$OUT_ROOT" "$FIGURES_DIR"
JOB_FILE="$LOG_DIR/teacher_student_rdm_score_feature_methods_${RUN_LABEL}_jobs_${TS}.tsv"
BASE_JOB_FILE="$LOG_DIR/teacher_student_rdm_score_feature_methods_${RUN_LABEL}_base_jobs_${TS}.tsv"

cat > "$BASE_JOB_FILE" <<'JOBS'
raw_only_mean_min raw 0-2
raw_only_mean_min raw 3-5
raw_only_mean_min sub-01 0-2
raw_only_mean_min sub-01 3-5
raw_only_mean_min sub-03 0-2
raw_only_mean_min sub-03 3-5
raw_only_mean_min sub-05 0-2
raw_only_mean_min sub-05 3-5
raw_only_mean_min sub-06 0-2
raw_only_mean_min sub-06 3-5
raw_only_mean_min sub-07 0-2
raw_only_mean_min sub-07 3-5
sub01_only_mean_min raw 0-2
sub01_only_mean_min raw 3-5
sub01_only_mean_min sub-01 0-2
sub01_only_mean_min sub-01 3-5
sub01_only_mean_min sub-03 0-2
sub01_only_mean_min sub-03 3-5
sub01_only_mean_min sub-05 0-2
sub01_only_mean_min sub-05 3-5
sub01_only_mean_min sub-06 0-2
sub01_only_mean_min sub-06 3-5
sub01_only_mean_min sub-07 0-2
sub01_only_mean_min sub-07 3-5
raw_enc_w05_mean_min raw 0-2
raw_enc_w05_mean_min raw 3-5
raw_enc_w05_mean_min sub-01 0-2
raw_enc_w05_mean_min sub-01 3-5
raw_enc_w05_mean_min sub-03 0-2
raw_enc_w05_mean_min sub-03 3-5
raw_enc_w05_mean_min sub-05 0-2
raw_enc_w05_mean_min sub-05 3-5
raw_enc_w05_mean_min sub-06 0-2
raw_enc_w05_mean_min sub-06 3-5
raw_enc_w05_mean_min sub-07 0-2
raw_enc_w05_mean_min sub-07 3-5
paper_effective_identity_sub01_mean_min_no_attenuation raw 0-2
paper_effective_identity_sub01_mean_min_no_attenuation raw 3-5
paper_effective_identity_sub01_mean_min_no_attenuation sub-01 0-2
paper_effective_identity_sub01_mean_min_no_attenuation sub-01 3-5
paper_effective_identity_sub01_mean_min_no_attenuation sub-03 0-2
paper_effective_identity_sub01_mean_min_no_attenuation sub-03 3-5
paper_effective_identity_sub01_mean_min_no_attenuation sub-05 0-2
paper_effective_identity_sub01_mean_min_no_attenuation sub-05 3-5
paper_effective_identity_sub01_mean_min_no_attenuation sub-06 0-2
paper_effective_identity_sub01_mean_min_no_attenuation sub-06 3-5
paper_effective_identity_sub01_mean_min_no_attenuation sub-07 0-2
paper_effective_identity_sub01_mean_min_no_attenuation sub-07 3-5
JOBS

export LD_LIBRARY_PATH="/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-}"
export PYTHON="/data/home_roth/miniforge3/bin/python"
export SCRIPT="00_stimulus_selection/selection_evaluation/code/teacher_student/01_compute_independent_refit_rdm_recovery.py"
export PLOT="00_stimulus_selection/selection_evaluation/code/teacher_student/21_plot_feature_method_sweep_recovery.py"
export RANDOM_FEATURE_DIR="shared/cache_or_heavy/natural_pool_subset_100k_seed42"
export EVAL_REFIT_MODE="${EVAL_REFIT_MODE:-independent}"
export N_NOISE_SAMPLES="${N_NOISE_SAMPLES:-20}"
export N_RANDOM_SUBSETS="${N_RANDOM_SUBSETS:-20}"
export N_REFIT_REPEATS="${N_REFIT_REPEATS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MAX_PROCS="${MAX_PROCS:-4}"
export GPU_COUNT="${GPU_COUNT:-8}"
export GPU_OFFSET="${GPU_OFFSET:-0}"
export TS RUN_LABEL LOG_DIR GPU_COUNT GPU_OFFSET PAYLOAD_RUN SELECTION_ROOT OUT_RUN OUT_ROOT FIGURES_DIR

: > "$JOB_FILE"
for refit_repeat in $(seq 0 $((N_REFIT_REPEATS - 1))); do
  awk -v refit_repeat="$refit_repeat" 'NF {print $0 "\t" refit_repeat}' "$BASE_JOB_FILE" >> "$JOB_FILE"
done

run_job() {
  local method="$1"
  local track="$2"
  local teacher_indices="$3"
  local refit_repeat="$4"
  local out_dir="${OUT_ROOT}/${method}"
  local safe_indices="${teacher_indices//,/plus}"
  safe_indices="${safe_indices//-/to}"
  local log="${LOG_DIR}/teacher_student_rdm_feature_methods_${RUN_LABEL}_${TS}_${method}_${track}_${safe_indices}_refit${refit_repeat}.log"
  {
    echo "job_start $(date -Is) method=${method} track=${track} teacher_indices=${teacher_indices} refit_repeat=${refit_repeat}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    "$PYTHON" -u "$SCRIPT" \
      --model-set "$method" \
      --selection-root "$SELECTION_ROOT" \
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
      --encoding-batch-size 1024 \
      --unique-encodings \
      --teacher-indices "$teacher_indices" \
      --cache-only \
      --output-dir "$out_dir"
    echo "job_done $(date -Is) method=${method} track=${track} teacher_indices=${teacher_indices} refit_repeat=${refit_repeat}"
  } > "$log" 2>&1
}
export -f run_job

echo "Feature-method teacher/student RDM-score ${RUN_LABEL} run ${TS}"
echo "Job file: ${JOB_FILE}"
echo "PAYLOAD_RUN=${PAYLOAD_RUN}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "MAX_PROCS=${MAX_PROCS}; GPU_COUNT=${GPU_COUNT}; OMP/MKL/OPENBLAS=${OMP_NUM_THREADS}/${MKL_NUM_THREADS}/${OPENBLAS_NUM_THREADS}"
echo "GPU_OFFSET=${GPU_OFFSET}"
echo "N_NOISE_SAMPLES=${N_NOISE_SAMPLES}; N_RANDOM_SUBSETS=${N_RANDOM_SUBSETS}; N_REFIT_REPEATS=${N_REFIT_REPEATS}; EVAL_REFIT_MODE=${EVAL_REFIT_MODE}"

xargs --process-slot-var=JOB_SLOT -P "$MAX_PROCS" -n 4 bash -c '
  export CUDA_VISIBLE_DEVICES=$(((JOB_SLOT + GPU_OFFSET) % GPU_COUNT))
  run_job "$@"
' _ < "$JOB_FILE"

for method in \
  raw_only_mean_min \
  sub01_only_mean_min \
  raw_enc_w05_mean_min \
  paper_effective_identity_sub01_mean_min_no_attenuation
do
  out_dir="${OUT_ROOT}/${method}"
  "$PYTHON" -u "$SCRIPT" \
    --model-set "$method" \
    --selection-root "$SELECTION_ROOT" \
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
    --unique-encodings \
    --merge-only \
    --output-dir "$out_dir"
done

"$PYTHON" "$PLOT" \
  --run-dir "$OUT_RUN" \
  --figures-root "$FIGURES_DIR" \
  --results-name "$(basename "$OUT_ROOT")" \
  --name "teacher_student_recovery_feature_method_sweep_${RUN_LABEL}"
echo "Feature-method teacher/student RDM-score ${RUN_LABEL} run complete $(date -Is)"
