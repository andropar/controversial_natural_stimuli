# Supplementary

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Figure Descriptions

The notes below summarize what each rendered figure shows. The file table that follows points back to the scripts and inputs used to make it.

**`controlled_all_models`.** This controlled all-model comparison repeats the alignment analysis under the
relevant control setting. It uses the CRSA/WRSA RSA score tables and noise-ceiling summaries to check that the
main effect is not an artifact of the uncontrolled all-model score calculation.

**`cross_set_comparison`.** This cross-set comparison asks whether the RSA effect transfers across model-set
definitions. It uses the CRSA/WRSA RSA score tables and noise-ceiling summaries to show whether the
controversial-stimulus result is specific to one grouping or robust across groupings.

**`cross_set_out_of_set`.** This cross-set figure focuses on out-of-set transfer: whether model-brain
alignment patterns learned or summarized in one model set generalize to other held-out model sets, using the
CRSA/WRSA RSA score tables and noise-ceiling summaries.

**`large_benchmark_gap_raw_improved`.** This benchmark-gap figure shows the raw, not normalized,
controversial-versus-baseline alignment difference for large benchmark models. It uses the CRSA/WRSA RSA score
tables and noise-ceiling summaries and is useful for comparing absolute RSA effect size before noise-ceiling
scaling.

**`model_rankings`.** This model-ranking figure lists or summarizes which models achieve the strongest brain
alignment under the analyzed condition. It uses the CRSA/WRSA RSA score tables and noise-ceiling summaries and
is mainly a descriptive companion to the rank-shift and rank-consistency analyses.

**`rank_rho_summary_improved`.** This rank-correlation summary shows how stable the model ordering is between
controversial and baseline conditions. It uses the CRSA/WRSA RSA score tables and noise-ceiling summaries to
test whether the stimulus manipulation only changes absolute alignment, or also reshuffles which models look
best.

**`score_distributions`.** This figure plots the distribution of RSA alignment scores rather than only summary
means. It uses the CRSA/WRSA RSA score tables and noise-ceiling summaries to show spread, overlap, and
outliers between controversial and baseline conditions.

## Contents Snapshot

- Folder: `01_brain_model_alignment/figures/rsa_scores/supplementary`
- Figures in this folder tree: 7
- Data/table-like files in this folder tree: 0
- Python scripts in this folder tree: 0
- Direct files: `controlled_all_models.pdf`, `cross_set_comparison.pdf`, `cross_set_out_of_set.pdf`, `large_benchmark_gap_raw_improved.pdf`, `model_rankings.pdf`, `rank_rho_summary_improved.pdf`, `score_distributions.pdf`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Provenance

Each row is one figure concept; `formats` lists the concrete files present in this folder.

| Figure | Formats | What it shows | Source / derivation | Resources or results | Script | Paper use |
|---|---:|---|---|---|---|---|
| `controlled_all_models` | pdf | controlled all-model alignment comparison | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/02_rsa_scores/figures/controlled_all_models.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/02_rsa_scores/figures/controlled_all_models.pdf` | `cross_set_wrsa_scores.csv`; `wrsa_transfer_scores.csv`; `SHARE_ROOT`; `STAGE_DIR`; `RSA_DATA_DIR`; `FIGURES_DIR`; plus 7 more | `01_brain_model_alignment/code/rsa_scoring/figures/plot_controlled_all_models.py` | no |
| `cross_set_comparison` | pdf | cross-set transfer or out-of-set RSA comparison | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/02_rsa_scores/figures/cross_set_comparison.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/02_rsa_scores/figures/cross_set_comparison.pdf` | `crsa_scores.csv`; `wrsa_transfer_scores.csv`; `SHARE_ROOT`; `STAGE_DIR`; `RSA_DATA_DIR`; `FIGURES_DIR`; plus 7 more | `01_brain_model_alignment/code/rsa_scoring/figures/plot_cross_set_comparison.py` | no |
| `cross_set_out_of_set` | pdf | cross-set transfer or out-of-set RSA comparison | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/02_rsa_scores/figures/cross_set_out_of_set.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/02_rsa_scores/figures/cross_set_out_of_set.pdf` | `cross_set_wrsa_scores.csv`; `SHARE_ROOT`; `STAGE_DIR`; `RSA_DATA_DIR`; `FIGURES_DIR`; `PNG_DIR`; plus 6 more | `01_brain_model_alignment/code/rsa_scoring/figures/plot_cross_set_out_of_set.py` | no |
| `large_benchmark_gap_raw_improved` | pdf | raw benchmark gap between controversial and baseline alignment | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/02_rsa_scores/figures/large_benchmark_gap_raw_improved.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/02_rsa_scores/figures/large_benchmark_gap_raw_improved.pdf` | `01_brain_model_alignment/results/rsa_scores/*/{crsa_scores,wrsa_transfer_scores}.csv`; `rsa_large_benchmark_scores.csv`; `cross_set_wrsa_scores.csv`; `wrsa_transfer_scores.csv`; `crsa_scores.csv`; `rdm_noise_ceilings.csv`; plus 13 more | `01_brain_model_alignment/code/rsa_scoring/figures/plot_large_benchmark_gap.py`; `01_brain_model_alignment/code/rsa_scoring/figures/plot_large_benchmark_gap_improved.py` | no |
| `model_rankings` | pdf | model ranking summary from RSA scores | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/02_rsa_scores/figures/model_rankings.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/02_rsa_scores/figures/model_rankings.pdf` | `01_brain_model_alignment/results/rsa_scores/*/{crsa_scores,wrsa_transfer_scores}.csv`; `variance_partitioning.csv`; `encoding_projection.csv`; `cross_stimulus_pairwise.csv`; `rdm_noise_ceilings.csv`; `rank_consistency_wrsa_transfer_data.csv`; plus 11 more | `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/archive/main_paper_figures/plot_missing_figures.py` | no |
| `rank_rho_summary_improved` | pdf | rank-correlation summary for model ordering stability | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/02_rsa_scores/figures/rank_rho_summary_improved.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/02_rsa_scores/figures/rank_rho_summary_improved.pdf` | `01_brain_model_alignment/results/rsa_scores/*/{crsa_scores,wrsa_transfer_scores}.csv`; `rank_correlations.csv`; `SHARE_ROOT`; `STAGE_DIR`; `FIGURES_DIR`; `PNG_DIR`; plus 8 more | `01_brain_model_alignment/code/rsa_scoring/figures/plot_rank_shift_improved.py` | yes |
| `score_distributions` | pdf | distribution of RSA alignment scores across conditions | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/02_rsa_scores/figures/score_distributions.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/02_rsa_scores/figures/score_distributions.pdf` | `01_brain_model_alignment/results/rsa_scores/*/{crsa_scores,wrsa_transfer_scores}.csv`; `wrsa_transfer_scores.csv`; `crsa_scores.csv`; `permutation_test_results.csv`; `SHARE_ROOT`; `STAGE_DIR`; plus 10 more | `01_brain_model_alignment/code/rsa_scoring/figures/plot_score_distributions.py` | no |
<!-- END AUTO-FIGURE-PROVENANCE -->
