#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -z "${ROOT:-}" ]]; then
  ROOT_CANDIDATE="${SCRIPT_DIR}"
  while [[ "${ROOT_CANDIDATE}" != "/" && ! -d "${ROOT_CANDIDATE}/src/cstims" ]]; do
    ROOT_CANDIDATE=$(dirname "${ROOT_CANDIDATE}")
  done
  ROOT="${ROOT_CANDIDATE}"
fi
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x /data/home_roth/miniforge3/bin/python ]]; then
    PYTHON=/data/home_roth/miniforge3/bin/python
  else
    PYTHON=python
  fi
fi
SCRIPT="${ROOT}/00_stimulus_selection/selection_evaluation/code/noisy_by_clean/01_compute_noisy_by_clean_recovery.py"
PLOT="${ROOT}/00_stimulus_selection/selection_evaluation/code/noisy_by_clean/02_plot_noisy_by_clean_recovery.py"
PAIRWISE_PLOT="${ROOT}/00_stimulus_selection/selection_evaluation/code/noisy_by_clean/03_plot_pairwise_margin.py"

if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
elif [[ -d /data/home_roth/miniforge3/lib ]]; then
  export LD_LIBRARY_PATH="/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-}"
fi

"${PYTHON}" "${SCRIPT}" "$@"
"${PYTHON}" "${PLOT}"
"${PYTHON}" "${PAIRWISE_PLOT}"
