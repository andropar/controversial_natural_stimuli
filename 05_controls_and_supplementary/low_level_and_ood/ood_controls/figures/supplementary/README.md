# Supplementary

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Figure Descriptions

The notes below summarize what each rendered figure shows. The file table that follows points back to the scripts and inputs used to make it.

**`baseline_subsampling_match_quality`.** This OOD-control diagnostic shows how well baseline subsamples match
the controversial set on the relevant control variables. It uses the low-level/OOD control tables to judge
whether the subsampling control is credible.

**`baseline_subsampling_matched_only`.** This OOD-control plot focuses only on matched baseline subsamples. It
uses the low-level/OOD control tables to show the alignment or control-variable result after unmatched
baseline samples are excluded.

**`baseline_subsampling_scatter_all_models`.** This scatter plot visualizes baseline subsampling across models
or control dimensions. It uses the low-level/OOD control tables to show whether the selected and matched
baseline images occupy comparable control-feature ranges.

**`disagreement_vs_ood_pairs`.** This pair-level OOD diagnostic relates model-disagreement differences to OOD
differences for image pairs. It uses the low-level/OOD control tables because RSA is pairwise and image-level
balance alone is not enough.

**`disagreement_vs_ood_residual`.** This residualized OOD-control figure asks what remains of the
alignment/disagreement relationship after OOD or low-level structure is removed. It uses the low-level/OOD
control tables to separate model-disagreement selection from generic OOD effects.

**`low_level_deterministic_curve`.** This low-level-control curve shows how deterministic low-level matching
or prediction behaves as the control criterion changes. It uses the low-level/OOD control tables to assess
whether low-level image statistics can reproduce the selection/alignment effect.

**`mahalanobis_l2_robustness`.** This robustness plot tests low-level/OOD conclusions with Mahalanobis and L2
control distances. It uses the low-level/OOD control tables to check whether the control result depends on the
specific distance metric.

**`ood_per_model`.** This per-model OOD figure shows how OOD scores or OOD-related effects vary across models.
It uses the low-level/OOD control tables to identify whether a few models drive the OOD relationship.

## Contents Snapshot

- Folder: `05_controls_and_supplementary/low_level_and_ood/ood_controls/figures/supplementary`
- Figures in this folder tree: 8
- Data/table-like files in this folder tree: 0
- Python scripts in this folder tree: 0
- Direct files: `baseline_subsampling_match_quality.pdf`, `baseline_subsampling_matched_only.pdf`, `baseline_subsampling_scatter_all_models.pdf`, `disagreement_vs_ood_pairs.pdf`, `disagreement_vs_ood_residual.pdf`, `low_level_deterministic_curve.pdf`, `mahalanobis_l2_robustness.pdf`, `ood_per_model.pdf`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Provenance

Each row is one figure concept; `formats` lists the concrete files present in this folder.

| Figure | Formats | What it shows | Source / derivation | Resources or results | Script | Paper use |
|---|---:|---|---|---|---|---|
| `baseline_subsampling_match_quality` | pdf | figure derived from baseline subsampling match quality | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/06_ood/figures/baseline_subsampling_match_quality.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/06_ood/figures/baseline_subsampling_match_quality.pdf` | `baseline_subsampling.csv`; `baseline_subsampling_summary.csv`; `SHARE_ROOT`; `DATA_DIR`; `PNG_DIR`; `05_controls_and_supplementary/low_level_and_ood/ood_controls/data/baseline_subsampling.csv`; plus 5 more | `05_controls_and_supplementary/low_level_and_ood/ood_controls/code/figures/plot_baseline_subsampling.py` | no |
| `baseline_subsampling_matched_only` | pdf | figure derived from baseline subsampling matched only | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/06_ood/figures/baseline_subsampling_matched_only.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/06_ood/figures/baseline_subsampling_matched_only.pdf` | `baseline_subsampling.csv`; `baseline_subsampling_summary.csv`; `SHARE_ROOT`; `DATA_DIR`; `PNG_DIR`; `05_controls_and_supplementary/low_level_and_ood/ood_controls/data/baseline_subsampling.csv`; plus 5 more | `05_controls_and_supplementary/low_level_and_ood/ood_controls/code/figures/plot_baseline_subsampling.py` | no |
| `baseline_subsampling_scatter_all_models` | pdf | figure derived from baseline subsampling scatter all models | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/06_ood/figures/baseline_subsampling_scatter_all_models.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/06_ood/figures/baseline_subsampling_scatter_all_models.pdf` | `baseline_subsampling.csv`; `baseline_subsampling_summary.csv`; `SHARE_ROOT`; `DATA_DIR`; `PNG_DIR`; `05_controls_and_supplementary/low_level_and_ood/ood_controls/data/baseline_subsampling.csv`; plus 5 more | `05_controls_and_supplementary/low_level_and_ood/ood_controls/code/figures/plot_baseline_subsampling.py` | no |
| `disagreement_vs_ood_pairs` | pdf | figure derived from disagreement vs ood pairs | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/06_ood/figures/disagreement_vs_ood_pairs.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/06_ood/figures/disagreement_vs_ood_pairs.pdf` | `05_controls_and_supplementary/low_level_and_ood/ood_controls/data/*.csv`; `disagreement_vs_ood_pairs.csv`; `disagreement_vs_ood_images.csv`; `disagreement_vs_ood_summary.csv`; `SHARE_ROOT`; `OOD_DIR`; plus 7 more | `05_controls_and_supplementary/low_level_and_ood/ood_controls/code/figures/plot_disagreement_vs_ood.py` | no |
| `disagreement_vs_ood_residual` | pdf | residual or unexplained brain-structure diagnostic | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/06_ood/figures/disagreement_vs_ood_residual.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/06_ood/figures/disagreement_vs_ood_residual.pdf` | `05_controls_and_supplementary/low_level_and_ood/ood_controls/data/*.csv`; `disagreement_vs_ood_pairs.csv`; `disagreement_vs_ood_images.csv`; `disagreement_vs_ood_summary.csv`; `SHARE_ROOT`; `OOD_DIR`; plus 7 more | `05_controls_and_supplementary/low_level_and_ood/ood_controls/code/figures/plot_disagreement_vs_ood.py` | no |
| `low_level_deterministic_curve` | pdf | figure derived from low level deterministic curve | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/06_ood/figures/low_level_deterministic_curve.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/06_ood/figures/low_level_deterministic_curve.pdf` | `05_controls_and_supplementary/low_level_and_ood/image_statistics/data/image_stats.csv`; `wrsa_low_level_subsets.csv`; `wrsa_low_level_subsets_comparison.csv`; `wrsa_transfer_scores.csv`; `rdm_noise_ceilings.csv`; `pca_loglik.csv`; plus 11 more | `05_controls_and_supplementary/low_level_and_ood/ood_controls/code/figures/plot_low_level_deterministic.py` | no |
| `mahalanobis_l2_robustness` | pdf | low-level/OOD robustness using Mahalanobis and L2 controls | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/06_ood/figures/mahalanobis_l2_robustness.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/06_ood/figures/mahalanobis_l2_robustness.pdf` | `05_controls_and_supplementary/low_level_and_ood/ood_controls/data/*.csv`; `image_stats.csv`; `low_level_mahalanobis_l2norm.csv`; `low_level_mahalanobis_l2norm_summary.csv`; `OUT_DATA`; `05_controls_and_supplementary/low_level_and_ood/ood_controls/data/baseline_subsampling.csv`; plus 5 more | `05_controls_and_supplementary/low_level_and_ood/ood_controls/code/07_l2_mahalanobis_robustness.py` | source for manuscript copy |
| `ood_per_model` | pdf | per-model OOD/alignment summary | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/06_ood/figures/ood_per_model.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/06_ood/figures/ood_per_model.pdf` | `05_controls_and_supplementary/low_level_and_ood/ood_controls/data/*.csv`; `ood_vs_alignment.csv`; `ood_per_model_rho.csv`; `05_controls_and_supplementary/low_level_and_ood/ood_controls/data/baseline_subsampling.csv`; `05_controls_and_supplementary/low_level_and_ood/ood_controls/data/baseline_subsampling_summary.csv`; `05_controls_and_supplementary/low_level_and_ood/ood_controls/data/disagreement_vs_ood_images.csv`; plus 3 more | `05_controls_and_supplementary/low_level_and_ood/ood_controls/code/06_per_set_ood_scatter.py` | yes |
<!-- END AUTO-FIGURE-PROVENANCE -->
