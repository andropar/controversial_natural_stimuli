# 05 Controls And Supplementary Analyses

This stage contains controls, diagnostics, robustness checks, method
validations, and mechanistic follow-ups. These analyses support interpretation
of the primary brain-model alignment effect but are not all top-level pipeline
stages.

Use this folder when you need to check whether the main result could be
explained by a simpler baseline, a low-level image property, an out-of-domain
artifact, model-scope choices, residual reliability structure, or a selection
objective artifact.

## Subgroups

- `counterfactual_baselines/`: matched or alternative stimulus baselines.
- `low_level_and_ood/`: low-level image statistics and out-of-domain controls.
- `reliability_and_residual_structure/`: residual reliability, unexplained
  variance, and graceful-degradation checks.
- `model_scope_followups/`: layer sweeps, ROI analyses, discriminability, and
  model-scope variants.
- `stimulus_and_pair_diagnostics/`: what the selected stimuli and image pairs
  look like in image, semantic, and brain-space diagnostics.
- `selection_objective_validation/`: simulations and checks for the selection
  objective and noise model.
- `integrated_explanation/`: combined interpretation/report material tying
  several controls together.

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Contents Snapshot

- Folder: `05_controls_and_supplementary`
- Figures in this folder tree: 63
- Data/table-like files in this folder tree: 1535
- Python scripts in this folder tree: 113

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Index

| Figure directory | Figures | README |
|---|---:|---|
| `05_controls_and_supplementary/counterfactual_baselines/figures` | 1 | `05_controls_and_supplementary/counterfactual_baselines/figures/README.md` |
| `05_controls_and_supplementary/counterfactual_baselines/figures/png` | 1 | `05_controls_and_supplementary/counterfactual_baselines/figures/png/README.md` |
| `05_controls_and_supplementary/counterfactual_baselines/figures/supplementary` | 1 | `05_controls_and_supplementary/counterfactual_baselines/figures/supplementary/README.md` |
| `05_controls_and_supplementary/counterfactual_baselines/figures/supplementary/png` | 1 | `05_controls_and_supplementary/counterfactual_baselines/figures/supplementary/png/README.md` |
| `05_controls_and_supplementary/integrated_explanation/figures` | 1 | `05_controls_and_supplementary/integrated_explanation/figures/README.md` |
| `05_controls_and_supplementary/integrated_explanation/figures/png` | 1 | `05_controls_and_supplementary/integrated_explanation/figures/png/README.md` |
| `05_controls_and_supplementary/low_level_and_ood/image_statistics/figures` | 1 | `05_controls_and_supplementary/low_level_and_ood/image_statistics/figures/README.md` |
| `05_controls_and_supplementary/low_level_and_ood/image_statistics/figures/png` | 1 | `05_controls_and_supplementary/low_level_and_ood/image_statistics/figures/png/README.md` |
| `05_controls_and_supplementary/low_level_and_ood/ood_controls/figures` | 2 | `05_controls_and_supplementary/low_level_and_ood/ood_controls/figures/README.md` |
| `05_controls_and_supplementary/low_level_and_ood/ood_controls/figures/png` | 2 | `05_controls_and_supplementary/low_level_and_ood/ood_controls/figures/png/README.md` |
| `05_controls_and_supplementary/low_level_and_ood/ood_controls/figures/supplementary` | 8 | `05_controls_and_supplementary/low_level_and_ood/ood_controls/figures/supplementary/README.md` |
| `05_controls_and_supplementary/low_level_and_ood/ood_controls/figures/supplementary/png` | 8 | `05_controls_and_supplementary/low_level_and_ood/ood_controls/figures/supplementary/png/README.md` |
| `05_controls_and_supplementary/model_scope_followups/discriminability/figures` | 1 | `05_controls_and_supplementary/model_scope_followups/discriminability/figures/README.md` |
| `05_controls_and_supplementary/model_scope_followups/discriminability/figures/png` | 1 | `05_controls_and_supplementary/model_scope_followups/discriminability/figures/png/README.md` |
| `05_controls_and_supplementary/model_scope_followups/layer_sweep/figures` | 4 | `05_controls_and_supplementary/model_scope_followups/layer_sweep/figures/README.md` |
| `05_controls_and_supplementary/model_scope_followups/layer_sweep/figures/png` | 4 | `05_controls_and_supplementary/model_scope_followups/layer_sweep/figures/png/README.md` |
| `05_controls_and_supplementary/model_scope_followups/layer_sweep/figures/supplementary` | 5 | `05_controls_and_supplementary/model_scope_followups/layer_sweep/figures/supplementary/README.md` |
| `05_controls_and_supplementary/model_scope_followups/layer_sweep/figures/supplementary/png` | 5 | `05_controls_and_supplementary/model_scope_followups/layer_sweep/figures/supplementary/png/README.md` |
| `05_controls_and_supplementary/model_scope_followups/roi_analysis/figures` | 1 | `05_controls_and_supplementary/model_scope_followups/roi_analysis/figures/README.md` |
| `05_controls_and_supplementary/model_scope_followups/roi_analysis/figures/png` | 1 | `05_controls_and_supplementary/model_scope_followups/roi_analysis/figures/png/README.md` |
| `05_controls_and_supplementary/reliability_and_residual_structure/residual_reliability/figures` | 1 | `05_controls_and_supplementary/reliability_and_residual_structure/residual_reliability/figures/README.md` |
| `05_controls_and_supplementary/reliability_and_residual_structure/residual_reliability/figures/png` | 1 | `05_controls_and_supplementary/reliability_and_residual_structure/residual_reliability/figures/png/README.md` |
| `05_controls_and_supplementary/reliability_and_residual_structure/residual_reliability/figures/supplementary` | 1 | `05_controls_and_supplementary/reliability_and_residual_structure/residual_reliability/figures/supplementary/README.md` |
| `05_controls_and_supplementary/reliability_and_residual_structure/residual_reliability/figures/supplementary/png` | 1 | `05_controls_and_supplementary/reliability_and_residual_structure/residual_reliability/figures/supplementary/png/README.md` |
| `05_controls_and_supplementary/selection_objective_validation/noise_model_validation/figures` | 1 | `05_controls_and_supplementary/selection_objective_validation/noise_model_validation/figures/README.md` |
| `05_controls_and_supplementary/selection_objective_validation/noise_model_validation/figures/png` | 1 | `05_controls_and_supplementary/selection_objective_validation/noise_model_validation/figures/png/README.md` |
| `05_controls_and_supplementary/selection_objective_validation/objective_mc_validation/figures` | 1 | `05_controls_and_supplementary/selection_objective_validation/objective_mc_validation/figures/README.md` |
| `05_controls_and_supplementary/selection_objective_validation/objective_mc_validation/figures/png` | 1 | `05_controls_and_supplementary/selection_objective_validation/objective_mc_validation/figures/png/README.md` |
| `05_controls_and_supplementary/stimulus_and_pair_diagnostics/stimulus_characterization/figures` | 4 | `05_controls_and_supplementary/stimulus_and_pair_diagnostics/stimulus_characterization/figures/README.md` |
| `05_controls_and_supplementary/stimulus_and_pair_diagnostics/stimulus_characterization/figures/png` | 1 | `05_controls_and_supplementary/stimulus_and_pair_diagnostics/stimulus_characterization/figures/png/README.md` |
<!-- END AUTO-FIGURE-PROVENANCE -->
