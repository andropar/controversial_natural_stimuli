#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/home_roth/cstims_share"
cd "$ROOT"

export LD_LIBRARY_PATH="/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/data/home_roth/miniforge3/bin/python"
SCRIPT="00_stimulus_selection/selection_evaluation/noisy_by_clean_recovery/code/compute_teacher_student_independent_refit_rdm_score.py"
PLOT="00_stimulus_selection/selection_evaluation/noisy_by_clean_recovery/code/plot_teacher_student_rdm_score_curves.py"
RANDOM_FEATURE_DIR="shared/cache_or_heavy/natural_pool_subset_100k_seed42"
DATA_SUFFIX="${DATA_SUFFIX:-teacher_student_independent_refit_1k_rdm_score_rdm}"
EVAL_NOISE_MODE="${EVAL_NOISE_MODE:-rdm}"
CORR_TYPE="${CORR_TYPE:-pearson}"
FIT_NOISE_CALIBRATION="${FIT_NOISE_CALIBRATION:-response}"
CALIBRATION_IMAGES="${CALIBRATION_IMAGES:-100}"
CALIBRATION_NOISE_SAMPLES="${CALIBRATION_NOISE_SAMPLES:-2}"
CALIBRATION_MAX_ITER="${CALIBRATION_MAX_ITER:-8}"

for i in $(seq 1 "${1:-72}"); do
  echo "merge_tick $(date -Is)"
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
      --output-dir "$out_dir" || true
  done
  "$PYTHON" "$PLOT" --data-suffix "_${DATA_SUFFIX}" || true
  sleep 300
done
