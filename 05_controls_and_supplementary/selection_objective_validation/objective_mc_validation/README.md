# Analytical-vs-Monte Carlo objective validation

This experiment checks whether the selection objective changes materially if the
correlation-attenuation approximation is replaced by direct RDM-space Monte
Carlo sampling.

The analysis uses cached LAION-natural feature arrays for the five
Training-Objective models, samples 1,000 calibration images, 64 selected images,
and 300 candidate images, and compares candidate utilities under:

- analytical utility: attenuate the expected Pearson correlation matrix, then
  aggregate utilities;
- RDM-space Monte Carlo utility: add Gaussian noise directly to RDM vectors,
  compute utility for each noisy draw, and average over 512 samples.

At the paper noise ceiling (0.46) and default hard-min objective
(`aggregation_within=mean`, `aggregation_across=min`), analytical and MC
utilities were strongly aligned (Spearman rho = 0.896, top-20 overlap = 80%).
The analytical top candidate was the second-best candidate under the MC
objective. Selecting it instead of the MC top candidate cost 0.00021 MC utility
units, which is 0.29% of the mean MC utility and 4.0% of the candidate utility
range. The MC estimate was systematically lower by 0.00157 utility units, as
expected because the hard minimum is applied before averaging in the MC
estimator.

Linear and smooth aggregations showed near-zero bias and higher agreement:
`mean/mean` rho = 0.974, top-20 overlap = 90%; `smooth_min/smooth_min` rho =
0.976, top-20 overlap = 90%.

## Files

- `analysis/01_validate_analytical_vs_mc_objective.py`: computes cached CSVs.
- `data/summary_by_noise_ceiling.csv`: hard-min objective across noise ceilings.
- `data/summary_by_aggregation.csv`: aggregation controls at NC = 0.46.
- `data/candidate_utilities_nc0.46_mean_min.csv`: per-candidate scatter data.
- `figures/plot_objective_mc_validation.py`: paper-style plotting script.
- `figures/objective_mc_validation.pdf`: supplemental figure.

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Contents Snapshot

- Folder: `05_controls_and_supplementary/selection_objective_validation/objective_mc_validation`
- Figures in this folder tree: 2
- Data/table-like files in this folder tree: 5
- Python scripts in this folder tree: 2
- Main child folders: `code/`, `data/`, `figures/`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Index

| Figure directory | Figures | README |
|---|---:|---|
| `05_controls_and_supplementary/selection_objective_validation/objective_mc_validation/figures` | 1 | `05_controls_and_supplementary/selection_objective_validation/objective_mc_validation/figures/README.md` |
| `05_controls_and_supplementary/selection_objective_validation/objective_mc_validation/figures/png` | 1 | `05_controls_and_supplementary/selection_objective_validation/objective_mc_validation/figures/png/README.md` |
<!-- END AUTO-FIGURE-PROVENANCE -->
