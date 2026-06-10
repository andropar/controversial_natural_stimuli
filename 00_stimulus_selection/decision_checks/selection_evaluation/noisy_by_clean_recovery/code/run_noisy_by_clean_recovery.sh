#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x /data/home_roth/miniforge3/bin/python ]]; then
    PYTHON=/data/home_roth/miniforge3/bin/python
  else
    PYTHON=python
  fi
fi
SCRIPT="${ROOT}/00_stimulus_selection/decision_checks/selection_evaluation/noisy_by_clean_recovery/code/compute_noisy_by_clean_recovery.py"
PLOT="${ROOT}/00_stimulus_selection/decision_checks/selection_evaluation/noisy_by_clean_recovery/code/plot_noisy_by_clean_recovery.py"
PAIRWISE_PLOT="${ROOT}/00_stimulus_selection/decision_checks/selection_evaluation/noisy_by_clean_recovery/code/plot_pairwise_margin.py"

if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
elif [[ -d /data/home_roth/miniforge3/lib ]]; then
  export LD_LIBRARY_PATH="/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-}"
fi

"${PYTHON}" "${SCRIPT}" "$@"
"${PYTHON}" "${PLOT}"
"${PYTHON}" "${PAIRWISE_PLOT}"
