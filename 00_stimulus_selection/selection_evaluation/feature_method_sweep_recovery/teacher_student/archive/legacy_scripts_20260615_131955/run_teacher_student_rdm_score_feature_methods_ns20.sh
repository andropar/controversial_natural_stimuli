#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/home_roth/cstims_share"
exec "${ROOT}/00_stimulus_selection/selection_evaluation/code/teacher_student/run_feature_method_sweep_recovery.sh" "$@"
