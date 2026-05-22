# Png

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Figure Descriptions

The notes below summarize what each rendered figure shows. The file table that follows points back to the scripts and inputs used to make it.

**`loglik_distributions_raw_improved`.** This plot compares OOD log-likelihood distributions for selected and
reference images. It uses the low-level/OOD control tables to test whether controversial stimuli are globally
less natural or more OOD than their baselines.

**`low_level_dissociation`.** This dissociation figure compares the main alignment/disagreement result with
low-level image-statistic controls. It uses the low-level/OOD control tables to support or reject the claim
that low-level statistics explain the controversial-stimulus effect.

## Contents Snapshot

- Folder: `05_controls_and_supplementary/low_level_and_ood/ood_controls/figures/png`
- Figures in this folder tree: 2
- Data/table-like files in this folder tree: 0
- Python scripts in this folder tree: 0
- Direct files: `loglik_distributions_raw_improved.png`, `low_level_dissociation.png`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Provenance

Each row is one figure concept; `formats` lists the concrete files present in this folder.

| Figure | Formats | What it shows | Source / derivation | Resources or results | Script | Paper use |
|---|---:|---|---|---|---|---|
| `loglik_distributions_raw_improved` | png | OOD log-likelihood distribution comparison | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/06_ood/figures/loglik_distributions_raw_improved.png`; historical share path `data/tier1_analysis_derivatives/cstim_paper/06_ood/figures/loglik_distributions_raw_improved.png` | `05_controls_and_supplementary/low_level_and_ood/ood_controls/data/*.csv`; `pca_loglik.csv`; `OOD_DATA`; `OOD_DATA_DIR`; `05_controls_and_supplementary/low_level_and_ood/ood_controls/data/baseline_subsampling.csv`; `05_controls_and_supplementary/low_level_and_ood/ood_controls/data/baseline_subsampling_summary.csv`; plus 4 more | `05_controls_and_supplementary/low_level_and_ood/ood_controls/code/figures/plot_loglik_distributions_improved.py` | no |
| `low_level_dissociation` | png | low-level image statistics versus alignment dissociation | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/06_ood/figures/low_level_dissociation.png`; historical share path `data/tier1_analysis_derivatives/cstim_paper/06_ood/figures/low_level_dissociation.png` | `05_controls_and_supplementary/low_level_and_ood/image_statistics/data/image_stats.csv`; `wrsa_low_level_subsets.csv`; `wrsa_low_level_subsets_comparison.csv`; `wrsa_transfer_scores.csv`; `rdm_noise_ceilings.csv`; `pca_loglik.csv`; plus 11 more | `05_controls_and_supplementary/low_level_and_ood/ood_controls/code/figures/plot_low_level_deterministic.py` | no |
<!-- END AUTO-FIGURE-PROVENANCE -->
