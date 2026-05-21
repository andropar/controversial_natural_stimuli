# Noise Model Validation

This folder contains the supplemental validation comparing two noise models used in the
in-silico stimulus-selection analysis:

1. RDM-space noise: add calibrated Gaussian noise directly to RDM vectors.
2. Feature-space Monte Carlo reference: add calibrated Gaussian noise to model features
   and recompute RDMs.

The original completed run was moved from `scripts/claude/validate_noise_approximation`.
The CSV files in `data/` are the frozen results used for the supplement figure.

Run:

```bash
python experiments/cstim_paper/16_noise_model_validation/figures/plot_noise_model_validation.py
```

This writes `noise_model_validation.pdf` and `.png` in `figures/`.

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Contents Snapshot

- Folder: `05_controls_and_supplementary/selection_objective_validation/noise_model_validation`
- Figures in this folder tree: 2
- Data/table-like files in this folder tree: 8
- Python scripts in this folder tree: 2
- Main child folders: `code/`, `data/`, `figures/`
- Direct files: `REPORT.md`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Index

| Figure directory | Figures | README |
|---|---:|---|
| `05_controls_and_supplementary/selection_objective_validation/noise_model_validation/figures` | 1 | `05_controls_and_supplementary/selection_objective_validation/noise_model_validation/figures/README.md` |
| `05_controls_and_supplementary/selection_objective_validation/noise_model_validation/figures/png` | 1 | `05_controls_and_supplementary/selection_objective_validation/noise_model_validation/figures/png/README.md` |
<!-- END AUTO-FIGURE-PROVENANCE -->
