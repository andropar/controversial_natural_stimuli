# Selection-Evaluation Producer Scripts

This folder contains the source scripts that produced the adjacent
`selection_evaluation` decision-check data and figures.

Source provenance:

- Original source root:
  `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation`
- `analysis/`: selection-evaluation data producers and diagnostic plotting
  helpers.
- `figures/`: paper-facing plotting scripts for the selection-evaluation
  figures.
- `noisy_by_clean/`: shared corrected noisy-by-clean recovery implementation
  used by the final-stimuli and feature-method-sweep recovery entrypoints.
- `teacher_student/`: teacher/student recovery evaluation scripts. The curated
  plotters are `11_plot_final_stimuli_recovery.py` and
  `21_plot_feature_method_sweep_recovery.py`; legacy/intermediate plotters are
  archived under `teacher_student/archive/`.

The corresponding evaluation outputs now live under
`../final_stimuli_recovery/` and `../feature_method_sweep_recovery/`.
Reusable method scripts live here under `noisy_by_clean/` and
`teacher_student/`; result directories keep only analysis-specific bash runners
unless they need real additional orchestration logic.

These scripts were added to the share package after an audit found that the
initial copy step included `00_selection_evaluation` derivatives
(`results/`, `figures/`, `*.md`, `*.html`) but omitted the producer scripts from
the active code copy list.
