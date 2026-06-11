#!/bin/bash
# Run 02_compute_discriminability.py with unique encodings for all model sets.

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}
SCRIPT=${SCRIPT:-"${ROOT}/00_stimulus_selection/selection_evaluation/code/analysis/02_compute_discriminability.py"}
SELECTION_ROOT=${SELECTION_ROOT:-"${ROOT}/00_stimulus_selection/results/selected_stimuli"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${ROOT}/00_stimulus_selection/selection_evaluation/results"}
SLURM=${SLURM:?Set SLURM to the start_as_slurm_job.py wrapper before running this helper}
TIMESTAMP=20251222_175721

MODEL_SETS=(all_models architecture dataset sota training_objective)

for ms in "${MODEL_SETS[@]}"; do
    result_dir="${SELECTION_ROOT}/${ms}/method-raw_plus_all_encodings/${TIMESTAMP}"
    output_dir="${OUTPUT_ROOT}/${ms}_unique_boot"

    echo "Submitting ${ms}..."
    python "${SLURM}" --gpu --mem 64000 \
        "${SCRIPT}" \
        --result-dir "${result_dir}" \
        --output-dir "${output_dir}" \
        --env raven \
        --unique-encodings
done
