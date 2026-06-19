#!/bin/bash
#
# Inventory layer-sweep state on Raven across the old rsa_based_selection
# checkout and the refactored cstims_share checkout.
#
# Run on Raven:
#   cd /u/rothj/controversial_natural_stimuli
#   bash 05_controls_and_supplementary/model_scope_followups/layer_sweep/code/inventory_raven_layer_sweep.sh

set -euo pipefail

NEW_REPO="${NEW_REPO:-/u/rothj/controversial_natural_stimuli}"
OLD_REPO="${OLD_REPO:-/u/rothj/cstims}"
PTMP_NEW="${PTMP_NEW:-/ptmp/rothj/controversial_natural_stimuli}"
PTMP_OLD="${PTMP_OLD:-/ptmp/rothj/cstims}"
PTMP_SHARE="${PTMP_SHARE:-/ptmp/rothj/cstims_share}"
SUBJECTS="${SUBJECTS:-sub-01,sub-03,sub-05,sub-06,sub-07}"

NEW_LAYER="${NEW_REPO}/05_controls_and_supplementary/model_scope_followups/layer_sweep"
OLD_LAYER="${OLD_REPO}/experiments/cstim_paper/11_layer_sweep"

section() {
  printf '\n==== %s ====\n' "$1"
}

exists() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    printf 'YES  %s\n' "${path}"
  else
    printf 'NO   %s\n' "${path}"
  fi
}

du_if_exists() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    du -sh "${path}" 2>/dev/null || true
  fi
}

recent_files() {
  local root="$1"
  local label="$2"
  if [[ ! -d "${root}" ]]; then
    return
  fi
  printf '\n-- recent %s: %s --\n' "${label}" "${root}"
  find "${root}" -maxdepth 5 -type f \
    \( -name '*.csv' -o -name '*.jsonl' -o -name '*.log' -o -name '*.out' -o -name '*.err' \) \
    -printf '%TY-%Tm-%Td %TH:%TM %12s %p\n' 2>/dev/null \
    | sort | tail -80
}

count_files() {
  local root="$1"
  local pattern="$2"
  if [[ -d "${root}" ]]; then
    find "${root}" -type f -name "${pattern}" 2>/dev/null | wc -l | tr -d ' '
  else
    printf '0'
  fi
}

section "Host"
hostname
date --iso-8601=seconds
printf 'USER=%s\n' "${USER:-unknown}"

section "Repo Roots"
for path in \
  "${NEW_REPO}" \
  "${OLD_REPO}" \
  "/u/rothj/rsa_based_selection" \
  "/u/rothj/controversial_natural-stimuli" \
  "${PTMP_NEW}" \
  "${PTMP_OLD}" \
  "${PTMP_SHARE}" \
  "/ptmp/rothj/cstims_laion_natural_subset" \
  "/ptmp/rothj/cstims_laion_natural_subset_memmaps" \
  "/ptmp/rothj/cstims_laion_natural_subset_memmaps_encoded"; do
  exists "${path}"
done

section "Configured Raven Paths"
if [[ -f "${NEW_REPO}/conf/paths/raven.env" ]]; then
  sed -n '1,120p' "${NEW_REPO}/conf/paths/raven.env"
else
  printf 'Missing %s\n' "${NEW_REPO}/conf/paths/raven.env"
fi

section "Layer Sweep Roots"
for root in "${NEW_LAYER}" "${OLD_LAYER}"; do
  exists "${root}"
  du_if_exists "${root}"
  du_if_exists "${root}/results"
  du_if_exists "${root}/cache_or_heavy"
  du_if_exists "${root}/logs"
done

section "Heavy Inputs"
for path in \
  "${PTMP_NEW}/external_data/final_cstims_hdf5_files" \
  "${PTMP_NEW}/external_data/deepvision_fmri" \
  "${PTMP_NEW}/01_brain_model_alignment/cache_or_heavy/deepvision_benchmark_cache" \
  "${PTMP_NEW}/01_brain_model_alignment/cache_or_heavy/cstim_brain_response_cache/data" \
  "${PTMP_NEW}/shared/cache_or_heavy/cstim_paper_feature_cache/feature_cache" \
  "${NEW_REPO}/01_brain_model_alignment/cache_or_heavy/deepvision_benchmark_cache" \
  "${NEW_REPO}/01_brain_model_alignment/cache_or_heavy/cstim_brain_response_cache/data" \
  "${NEW_REPO}/shared/cache_or_heavy/cstim_paper_feature_cache/feature_cache"; do
  exists "${path}"
  du_if_exists "${path}"
done

section "DeepVision Cache Detail"
for root in \
  "${PTMP_NEW}/01_brain_model_alignment/cache_or_heavy/deepvision_benchmark_cache" \
  "${NEW_REPO}/01_brain_model_alignment/cache_or_heavy/deepvision_benchmark_cache"; do
  if [[ ! -d "${root}" ]]; then
    continue
  fi
  printf '\n-- %s --\n' "${root}"
  find "${root}/image_sets" -maxdepth 1 -type f -name '*.csv' -printf '%f\n' 2>/dev/null | sort
  for sub in ${SUBJECTS//,/ }; do
    printf '%s unique image files: ' "${sub}"
    count_files "${root}/image_sets/deepvision_unique_${sub}" '*.jpg'
    printf '\n'
    exists "${root}/voxel_sets/deepvision_unique_${sub}_visual_cve0p20/finalinterp/${sub}/voxel_betas.npy"
    exists "${root}/voxel_sets/deepvision_shared_visual_cve0p20/finalinterp/${sub}/voxel_betas.npy"
  done
done

section "CSTIM Brain Cache Detail"
for root in \
  "${PTMP_NEW}/01_brain_model_alignment/cache_or_heavy/cstim_brain_response_cache/data" \
  "${NEW_REPO}/01_brain_model_alignment/cache_or_heavy/cstim_brain_response_cache/data"; do
  if [[ ! -d "${root}" ]]; then
    continue
  fi
  printf '\n-- %s --\n' "${root}"
  for sub in ${SUBJECTS//,/ }; do
    exists "${root}/${sub}/cstim_betas_averaged.npz"
    exists "${root}/${sub}/voxel_metadata.npz"
    exists "${root}/${sub}/cstim_stimulus_info.csv"
  done
done

section "Layer Sweep Result Files"
for root in "${NEW_LAYER}" "${OLD_LAYER}"; do
  if [[ ! -d "${root}" ]]; then
    continue
  fi
  printf '\n-- %s --\n' "${root}"
  for path in \
    "${root}/results/wrsa_dense_layer_sweep.csv" \
    "${root}/results/wrsa_dense_shared_layer_sweep.csv" \
    "${root}/results/mrsa_dense_all_eval_layer_scores.csv" \
    "${root}/results/mrsa_dense_layer_selection_transfer.csv"; do
    if [[ -f "${path}" ]]; then
      printf '%8s rows  %12s bytes  %s\n' \
        "$(($(wc -l < "${path}") - 1))" \
        "$(stat -c '%s' "${path}")" \
        "${path}"
    else
      printf 'missing %s\n' "${path}"
    fi
  done
done

section "Stream Part Summary"
for part_root in \
  "${NEW_LAYER}/results/stream_parts" \
  "${NEW_LAYER}/results"/stream_parts_raven_dense_* \
  "${OLD_LAYER}/results/stream_parts" \
  "${OLD_LAYER}/results"/stream_parts_raven_dense_*; do
  if [[ ! -d "${part_root}" ]]; then
    continue
  fi
  wrsa_dir="${part_root}/wrsa_dense_layer_sweep"
  shared_dir="${part_root}/wrsa_dense_shared_layer_sweep"
  printf '\n-- %s --\n' "${part_root}"
  du -sh "${part_root}" 2>/dev/null || true
  printf 'wrsa part files:   %s\n' "$(count_files "${wrsa_dir}" '*.csv')"
  printf 'shared part files: %s\n' "$(count_files "${shared_dir}" '*.csv')"
  find "${part_root}" -maxdepth 3 -type f -name '*.csv' \
    -printf '%TY-%Tm-%Td %TH:%TM %12s %p\n' 2>/dev/null \
    | sort | tail -30
done

section "Recent Files"
recent_files "${NEW_LAYER}/results" "new results"
recent_files "${NEW_LAYER}/logs" "new logs"
recent_files "${OLD_LAYER}/results" "old results"
recent_files "${OLD_LAYER}/logs" "old logs"

section "Slurm"
if command -v squeue >/dev/null 2>&1; then
  squeue -u "${USER}" -o '%.18i %.9P %.40j %.8T %.10M %.10l %.6D %R' \
    | grep -Ei 'JOBID|layer|dense|stream|wrsa|encoding|cstim' || true
fi
if command -v sacct >/dev/null 2>&1; then
  sacct -u "${USER}" --starttime now-14days \
    --format=JobID,JobName%40,State,Elapsed,ReqMem,MaxRSS,ExitCode \
    | grep -Ei 'JobID|layer|dense|stream|wrsa|encoding|cstim' || true
fi

section "Recommendation"
cat <<'TXT'
Use an existing result only if both canonical CSVs exist and have plausible row
counts:
  results/wrsa_dense_layer_sweep.csv
  results/wrsa_dense_shared_layer_sweep.csv

If only stream parts exist, prefer merging them instead of rerunning. If stream
parts are missing many subject/model files, starting fresh with the Raven array
submitter is usually faster.
TXT
