# Figures

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Figure Descriptions

The notes below summarize what each rendered figure shows. The file table that follows points back to the scripts and inputs used to make it.

**`noise_model_validation`.** This validation figure checks the analytical noise model used in the selection
objective. It uses the adjacent staged data/results files named in the provenance table to show whether the
assumed noise behavior matches simulated or empirical selection behavior.

## Contents Snapshot

- Folder: `05_controls_and_supplementary/selection_objective_validation/noise_model_validation/figures`
- Figures in this folder tree: 1
- Data/table-like files in this folder tree: 0
- Python scripts in this folder tree: 0
- Direct files: `noise_model_validation.pdf`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Provenance

Each row is one figure concept; `formats` lists the concrete files present in this folder.

| Figure | Formats | What it shows | Source / derivation | Resources or results | Script | Paper use |
|---|---:|---|---|---|---|---|
| `noise_model_validation` | pdf | validation of the selection noise model | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/16_noise_model_validation/figures/noise_model_validation.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/16_noise_model_validation/figures/noise_model_validation.pdf` | `02_alignment_reliability/data/*noise*.csv`; `noise_ceiling_results.csv`; `correlation_matrix_results.csv`; `discriminability_results.csv`; `ranking_results.csv`; `05_controls_and_supplementary/selection_objective_validation/noise_model_validation/data/candidate_utilities_nc0.3.csv`; plus 5 more | `05_controls_and_supplementary/selection_objective_validation/noise_model_validation/code/figures/plot_noise_model_validation.py` | yes |
<!-- END AUTO-FIGURE-PROVENANCE -->
