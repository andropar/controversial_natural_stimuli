# Selection Objective Validation

This subgroup validates the stimulus-selection objective itself. It is about
whether the selection algorithm and its noise assumptions behave as intended,
not about the final brain-alignment inference.

- `objective_mc_validation/`: Monte Carlo checks of the selection objective.
- `noise_model_validation/`: checks that the objective's noise model is
  calibrated enough for the selection claims that depend on it.
- `laion_decorrelation_pilot/`: two-model LAION pilot comparing noisy raw-track
  selection against a pure clean-decorrelation baseline.
- `laion_decorrelation_laion11/`: same LAION comparison using the 11 full-sample
  model feature spaces that are compatible with the mixed-RSA encoding models;
  this control found only small differences between noise-aware selection and
  pure decorrelation.

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Contents Snapshot

- Folder: `05_controls_and_supplementary/selection_objective_validation`
- Figures in this folder tree: 4
- Data/table-like files in this folder tree: 13
- Python scripts in this folder tree: 4

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Index

| Figure directory | Figures | README |
|---|---:|---|
| `05_controls_and_supplementary/selection_objective_validation/noise_model_validation/figures` | 1 | `05_controls_and_supplementary/selection_objective_validation/noise_model_validation/figures/README.md` |
| `05_controls_and_supplementary/selection_objective_validation/noise_model_validation/figures/png` | 1 | `05_controls_and_supplementary/selection_objective_validation/noise_model_validation/figures/png/README.md` |
| `05_controls_and_supplementary/selection_objective_validation/objective_mc_validation/figures` | 1 | `05_controls_and_supplementary/selection_objective_validation/objective_mc_validation/figures/README.md` |
| `05_controls_and_supplementary/selection_objective_validation/objective_mc_validation/figures/png` | 1 | `05_controls_and_supplementary/selection_objective_validation/objective_mc_validation/figures/png/README.md` |
<!-- END AUTO-FIGURE-PROVENANCE -->
