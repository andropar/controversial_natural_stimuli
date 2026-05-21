#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/home_roth/cstims_share"
RERUN="${ROOT}/05_controls_and_supplementary/full_data_rerun"
PY="/data/home_roth/miniforge3/bin/python"
LOG_DIR="${RERUN}/logs"
SUBJECTS=(sub-01 sub-03 sub-05 sub-06 sub-07)

mkdir -p "${LOG_DIR}" "${RERUN}/figures"
cd "${ROOT}"

trap 'code=$?; echo "[$(date "+%Y-%m-%d %H:%M:%S")] PIPELINE ERROR line=${LINENO} exit=${code}" | tee -a "${LOG_DIR}/pipeline_status.log"; exit ${code}' ERR

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

run_logged() {
  local name="$1"
  shift
  local log="${LOG_DIR}/${name}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(timestamp)] START ${name}" | tee -a "${LOG_DIR}/pipeline_status.log"
  echo "[$(timestamp)] CMD $*" | tee -a "${LOG_DIR}/pipeline_status.log"
  "$@" >"${log}" 2>&1
  echo "[$(timestamp)] DONE ${name} log=${log}" | tee -a "${LOG_DIR}/pipeline_status.log"
}

wait_for_existing_unique_cache_job() {
  local subject="$1"
  while pgrep -f "prepare_deepvision_unique_cache_from_laion.py --subject ${subject}" >/dev/null; do
    echo "[$(timestamp)] WAIT existing unique-cache job for ${subject}" | tee -a "${LOG_DIR}/pipeline_status.log"
    sleep 60
  done
}

echo "[$(timestamp)] PIPELINE START pid=$$ host=$(hostname)" | tee -a "${LOG_DIR}/pipeline_status.log"

run_logged build_best_layer_sofar \
  "${PY}" "${RERUN}/code/build_best_layer_sofar_from_layer_sweep.py"

run_logged check_best_layer_sofar_feature_availability \
  "${PY}" "${RERUN}/code/check_best_layer_sofar_feature_availability.py"

run_logged prepare_cstim_brain_cache \
  "${PY}" "${RERUN}/code/prepare_cstim_brain_cache_from_laion.py"

run_logged compute_paper_layer_crsa_by_roi \
  "${PY}" "${RERUN}/code/compute_paper_layer_crsa_by_roi.py" --n-vicco-boot 1000

for subject in "${SUBJECTS[@]}"; do
  wait_for_existing_unique_cache_job "${subject}"
  run_logged "prepare_deepvision_unique_${subject}" \
    "${PY}" "${RERUN}/code/prepare_deepvision_unique_cache_from_laion.py" --subject "${subject}"
done

run_logged fit_paper_layer_encodings_full_data \
  "${PY}" "${RERUN}/code/fit_paper_layer_encodings_full_data.py" --subject all --models all

run_logged compute_paper_layer_mrsa_by_roi \
  "${PY}" "${RERUN}/code/compute_paper_layer_mrsa_by_roi.py" --n-vicco-boot 1000

run_logged plot_full_data_rerun \
  "${PY}" "${RERUN}/code/plot_full_data_rerun.py"

echo "[$(timestamp)] PIPELINE DONE" | tee -a "${LOG_DIR}/pipeline_status.log"
