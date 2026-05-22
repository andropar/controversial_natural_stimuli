# Correlation Matrices Summary

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Figure Descriptions

The notes below summarize what each rendered figure shows. The file table that follows points back to the scripts and inputs used to make it.

**`correlation_matrices_margin_curve_raw_improved`.** This selection sanity-check plot follows the
discriminability margin for the model-set comparison in raw feature space. It summarizes how separated the
within-model and between-model RDM correlations are as the clean/noised or baseline condition changes, using
the frozen selection-evaluation CSVs and model-RDM summaries. Use it to see whether the selected stimuli keep
a usable model-separation margin under the relevant noise or baseline setting.

**`correlation_matrices_margin_curve_raw_pool_improved`.** This selection sanity-check plot follows the
discriminability margin for the model-set comparison in raw feature space. It summarizes how separated the
within-model and between-model RDM correlations are as the clean/noised or baseline condition changes, using
the frozen selection-evaluation CSVs and model-RDM summaries. Use it to see whether the selected stimuli keep
a usable model-separation margin under the relevant noise or baseline setting.

**`correlation_matrices_margin_curve_sub-01_improved`.** This selection sanity-check plot follows the
discriminability margin for the model-set comparison in sub-01 encoding/predicted-brain space. It summarizes
how separated the within-model and between-model RDM correlations are as the clean/noised or baseline
condition changes, using the frozen selection-evaluation CSVs and model-RDM summaries. Use it to see whether
the selected stimuli keep a usable model-separation margin under the relevant noise or baseline setting.

**`correlation_matrices_margin_curve_sub-01_pool_improved`.** This selection sanity-check plot follows the
discriminability margin for the model-set comparison in sub-01 encoding/predicted-brain space. It summarizes
how separated the within-model and between-model RDM correlations are as the clean/noised or baseline
condition changes, using the frozen selection-evaluation CSVs and model-RDM summaries. Use it to see whether
the selected stimuli keep a usable model-separation margin under the relevant noise or baseline setting.

**`correlation_matrices_summary_raw_improved`.** This summary figure compares median within-model RDM
correlations against between-model correlations for the model-set comparison in raw feature space. It reports
the diagonal-minus-off-diagonal margin for selected and reference images, using the frozen
selection-evaluation CSVs and model-RDM summaries. The purpose is to show whether the selected stimuli produce
a clearer model identity signal than the corresponding random/reference baseline.

**`correlation_matrices_summary_raw_pool_improved`.** This summary figure compares median within-model RDM
correlations against between-model correlations for the model-set comparison in raw feature space. It reports
the diagonal-minus-off-diagonal margin for selected and reference images, using the frozen
selection-evaluation CSVs and model-RDM summaries. The purpose is to show whether the selected stimuli produce
a clearer model identity signal than the corresponding random/reference baseline.

**`correlation_matrices_summary_sub-01_improved`.** This summary figure compares median within-model RDM
correlations against between-model correlations for the model-set comparison in sub-01
encoding/predicted-brain space. It reports the diagonal-minus-off-diagonal margin for selected and reference
images, using the frozen selection-evaluation CSVs and model-RDM summaries. The purpose is to show whether the
selected stimuli produce a clearer model identity signal than the corresponding random/reference baseline.

**`correlation_matrices_summary_sub-01_pool_improved`.** This summary figure compares median within-model RDM
correlations against between-model correlations for the model-set comparison in sub-01
encoding/predicted-brain space. It reports the diagonal-minus-off-diagonal margin for selected and reference
images, using the frozen selection-evaluation CSVs and model-RDM summaries. The purpose is to show whether the
selected stimuli produce a clearer model identity signal than the corresponding random/reference baseline.

## Contents Snapshot

- Folder: `00_stimulus_selection/decision_checks/selection_evaluation/figures/correlation_matrices_summary`
- Figures in this folder tree: 16
- Data/table-like files in this folder tree: 0
- Python scripts in this folder tree: 0
- Direct files: `correlation_matrices_margin_curve_raw_improved.pdf`, `correlation_matrices_margin_curve_raw_improved.png`, `correlation_matrices_margin_curve_raw_pool_improved.pdf`, `correlation_matrices_margin_curve_raw_pool_improved.png`, `correlation_matrices_margin_curve_sub-01_improved.pdf`, `correlation_matrices_margin_curve_sub-01_improved.png`, `correlation_matrices_margin_curve_sub-01_pool_improved.pdf`, `correlation_matrices_margin_curve_sub-01_pool_improved.png`, `correlation_matrices_summary_raw_improved.pdf`, `correlation_matrices_summary_raw_improved.png`, `correlation_matrices_summary_raw_pool_improved.pdf`, `correlation_matrices_summary_raw_pool_improved.png`

Use the tables below as a trace from rendered files back to the nearby code, staged data, or result folders that produced them.

## Figure Provenance

Each row is one figure concept; `formats` lists the concrete files present in this folder.

| Figure | Formats | What it shows | Source / derivation | Resources or results | Script | Paper use |
|---|---:|---|---|---|---|---|
| `correlation_matrices_margin_curve_raw_improved` | pdf, png | margin-vs-noise summary for model-RDM correlation-matrix checks | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation/figures/correlation_matrices_summary/correlation_matrices_margin_curve_raw_improved.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/00_selection_evaluation/figures/correlation_matrices_summary/correlation_matrices_margin_curve_raw_improved.pdf` | `selection_evaluation/data/*/correlation_matrices.csv`; `{which}_discriminability_{metric}_{corr_type}.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/ablations.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/correlation_matrices.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/discriminability.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/diversity.csv`; plus 2 more | `src/cstims/evaluation/plotting.py`; `src/cstims/evaluation/core.py` | source for manuscript copy |
| `correlation_matrices_margin_curve_raw_pool_improved` | pdf, png | margin-vs-noise summary for model-RDM correlation-matrix checks | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation/figures/correlation_matrices_summary/correlation_matrices_margin_curve_raw_pool_improved.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/00_selection_evaluation/figures/correlation_matrices_summary/correlation_matrices_margin_curve_raw_pool_improved.pdf` | `selection_evaluation/data/*/correlation_matrices.csv`; `{which}_discriminability_{metric}_{corr_type}.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/ablations.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/correlation_matrices.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/discriminability.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/diversity.csv`; plus 2 more | `src/cstims/evaluation/plotting.py`; `src/cstims/evaluation/core.py` | source for manuscript copy |
| `correlation_matrices_margin_curve_sub-01_improved` | pdf, png | margin-vs-noise summary for model-RDM correlation-matrix checks | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation/figures/correlation_matrices_summary/correlation_matrices_margin_curve_sub-01_improved.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/00_selection_evaluation/figures/correlation_matrices_summary/correlation_matrices_margin_curve_sub-01_improved.pdf` | `selection_evaluation/data/*/correlation_matrices.csv`; `{which}_discriminability_{metric}_{corr_type}.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/ablations.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/correlation_matrices.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/discriminability.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/diversity.csv`; plus 2 more | `src/cstims/evaluation/plotting.py`; `src/cstims/evaluation/core.py` | source for manuscript copy |
| `correlation_matrices_margin_curve_sub-01_pool_improved` | pdf, png | margin-vs-noise summary for model-RDM correlation-matrix checks | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation/figures/correlation_matrices_summary/correlation_matrices_margin_curve_sub-01_pool_improved.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/00_selection_evaluation/figures/correlation_matrices_summary/correlation_matrices_margin_curve_sub-01_pool_improved.pdf` | `selection_evaluation/data/*/correlation_matrices.csv`; `{which}_discriminability_{metric}_{corr_type}.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/ablations.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/correlation_matrices.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/discriminability.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/diversity.csv`; plus 2 more | `src/cstims/evaluation/plotting.py`; `src/cstims/evaluation/core.py` | source for manuscript copy |
| `correlation_matrices_summary_raw_improved` | pdf, png | summary of within-group versus between-group model-RDM correlations | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation/figures/correlation_matrices_summary/correlation_matrices_summary_raw_improved.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/00_selection_evaluation/figures/correlation_matrices_summary/correlation_matrices_summary_raw_improved.pdf` | `selection_evaluation/data/*/correlation_matrices.csv`; `{which}_discriminability_{metric}_{corr_type}.csv`; `correlation_matrices.csv`; `correlation_matrices_with_random_noised_pool.csv`; `correlation_matrices_with_random_noised.csv`; `margin_curve.csv`; plus 7 more | `src/cstims/evaluation/plotting.py`; `src/cstims/evaluation/core.py`; `00_stimulus_selection/decision_checks/selection_evaluation/code/figures/plot_correlation_matrices_improved.py` | source for manuscript copy |
| `correlation_matrices_summary_raw_pool_improved` | pdf, png | summary of within-group versus between-group model-RDM correlations | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation/figures/correlation_matrices_summary/correlation_matrices_summary_raw_pool_improved.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/00_selection_evaluation/figures/correlation_matrices_summary/correlation_matrices_summary_raw_pool_improved.pdf` | `selection_evaluation/data/*/correlation_matrices.csv`; `{which}_discriminability_{metric}_{corr_type}.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/ablations.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/correlation_matrices.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/discriminability.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/diversity.csv`; plus 2 more | `src/cstims/evaluation/plotting.py`; `src/cstims/evaluation/core.py` | source for manuscript copy |
| `correlation_matrices_summary_sub-01_improved` | pdf, png | summary of within-group versus between-group model-RDM correlations | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation/figures/correlation_matrices_summary/correlation_matrices_summary_sub-01_improved.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/00_selection_evaluation/figures/correlation_matrices_summary/correlation_matrices_summary_sub-01_improved.pdf` | `selection_evaluation/data/*/correlation_matrices.csv`; `{which}_discriminability_{metric}_{corr_type}.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/ablations.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/correlation_matrices.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/discriminability.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/diversity.csv`; plus 2 more | `src/cstims/evaluation/plotting.py`; `src/cstims/evaluation/core.py` | source for manuscript copy |
| `correlation_matrices_summary_sub-01_pool_improved` | pdf, png | summary of within-group versus between-group model-RDM correlations | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation/figures/correlation_matrices_summary/correlation_matrices_summary_sub-01_pool_improved.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/00_selection_evaluation/figures/correlation_matrices_summary/correlation_matrices_summary_sub-01_pool_improved.pdf` | `selection_evaluation/data/*/correlation_matrices.csv`; `{which}_discriminability_{metric}_{corr_type}.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/ablations.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/correlation_matrices.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/discriminability.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/results/all_models/diversity.csv`; plus 2 more | `src/cstims/evaluation/plotting.py`; `src/cstims/evaluation/core.py` | source for manuscript copy |
<!-- END AUTO-FIGURE-PROVENANCE -->
