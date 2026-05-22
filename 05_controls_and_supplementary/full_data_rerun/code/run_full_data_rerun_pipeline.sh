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

run_if_missing() {
  local name="$1"
  local output="$2"
  shift 2
  if [[ -s "${output}" ]]; then
    echo "[$(timestamp)] SKIP ${name} output_exists=${output}" | tee -a "${LOG_DIR}/pipeline_status.log"
    return
  fi
  run_logged "${name}" "$@"
}

have_cstim_cache() {
  local subject
  for subject in "${SUBJECTS[@]}"; do
    [[ -s "${RERUN}/results/brain_data_cache/${subject}/cstim_betas_averaged.npz" ]] || return 1
    [[ -s "${RERUN}/results/brain_data_cache/${subject}/voxel_metadata.npz" ]] || return 1
    [[ -s "${RERUN}/results/brain_data_cache/${subject}/cstim_stimulus_info.csv" ]] || return 1
  done
  return 0
}

have_all_paper_layer_encodings() {
  local count
  count=$(find "${RERUN}/results/encoding_models/paper_layer" -type f -name encoding_model.npz 2>/dev/null | wc -l)
  [[ "${count}" -ge 100 ]]
}

have_noise_ceilings() {
  [[ -s "${RERUN}/results/rdm_noise_ceilings_by_roi.csv" ]] || return 1
  [[ -s "${RERUN}/results/between_subject_noise_ceilings_by_roi.csv" ]] || return 1
  return 0
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

if have_cstim_cache; then
  echo "[$(timestamp)] SKIP prepare_cstim_brain_cache all_subject_outputs_exist" | tee -a "${LOG_DIR}/pipeline_status.log"
else
  run_logged prepare_cstim_brain_cache \
    "${PY}" "${RERUN}/code/prepare_cstim_brain_cache_from_laion.py"
fi

run_if_missing compute_paper_layer_crsa_by_roi \
  "${RERUN}/results/paper_layer_crsa_by_roi_summary.csv" \
  "${PY}" "${RERUN}/code/compute_paper_layer_crsa_by_roi.py" --n-vicco-boot 1000

for subject in "${SUBJECTS[@]}"; do
  wait_for_existing_unique_cache_job "${subject}"
  if [[ -s "${RERUN}/results/deepvision_unique_cache/${subject}/unique_betas_averaged.npz" ]]; then
    echo "[$(timestamp)] SKIP prepare_deepvision_unique_${subject} output_exists" | tee -a "${LOG_DIR}/pipeline_status.log"
  else
    run_logged "prepare_deepvision_unique_${subject}" \
      "${PY}" "${RERUN}/code/prepare_deepvision_unique_cache_from_laion.py" --subject "${subject}"
  fi
done

if have_all_paper_layer_encodings; then
  echo "[$(timestamp)] SKIP fit_paper_layer_encodings_full_data all_100_outputs_exist" | tee -a "${LOG_DIR}/pipeline_status.log"
else
  run_logged fit_paper_layer_encodings_full_data \
    "${PY}" "${RERUN}/code/fit_paper_layer_encodings_full_data.py" --subject all --models all
fi

run_if_missing compute_paper_layer_mrsa_by_roi \
  "${RERUN}/results/paper_layer_mrsa_by_roi_summary.csv" \
  "${PY}" "${RERUN}/code/compute_paper_layer_mrsa_by_roi.py" --n-vicco-boot 1000

if have_noise_ceilings; then
  echo "[$(timestamp)] SKIP compute_noise_ceilings_by_roi outputs_exist" | tee -a "${LOG_DIR}/pipeline_status.log"
else
  run_logged compute_noise_ceilings_by_roi \
    "${PY}" "${RERUN}/code/compute_noise_ceilings_by_roi.py" --n-vicco-boot 1000 --n-between-vicco-boot 200
fi

run_logged plot_brain_alignment_improved_with_shared \
  "${PY}" "${RERUN}/code/plot_brain_alignment_improved_with_shared.py" --roi all

run_logged plot_roi_spread_alignment_drop_summary \
  "${PY}" "${RERUN}/code/plot_roi_spread_alignment_drop_summary.py"

echo "[$(timestamp)] PIPELINE DONE" | tee -a "${LOG_DIR}/pipeline_status.log"
