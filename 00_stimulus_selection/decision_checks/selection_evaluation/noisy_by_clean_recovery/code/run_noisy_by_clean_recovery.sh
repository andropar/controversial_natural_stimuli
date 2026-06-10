#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}
PYTHON=${PYTHON:-/data/home_roth/miniforge3/bin/python}
SCRIPT="${ROOT}/00_stimulus_selection/decision_checks/selection_evaluation/noisy_by_clean_recovery/code/compute_noisy_by_clean_recovery.py"
PLOT="${ROOT}/00_stimulus_selection/decision_checks/selection_evaluation/noisy_by_clean_recovery/code/plot_noisy_by_clean_recovery.py"

export LD_LIBRARY_PATH="/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-}"

"${PYTHON}" "${SCRIPT}" "$@"
"${PYTHON}" "${PLOT}"
