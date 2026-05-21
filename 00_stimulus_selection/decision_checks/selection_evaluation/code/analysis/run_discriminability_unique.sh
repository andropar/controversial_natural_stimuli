#!/bin/bash
# Run 02_compute_discriminability.py with unique encodings for all model sets.

SCRIPT=/u/rothj/cstims/experiments/cstim_paper/00_selection_evaluation/analysis/02_compute_discriminability.py
SLURM=/u/rothj/laion_natural/scripts/start_as_slurm_job.py
SELECTION_ROOT=/u/rothj/cstims/outputs/final_cstims_v2_full
OUTPUT_ROOT=/u/rothj/cstims/experiments/cstim_paper/00_selection_evaluation/data
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
