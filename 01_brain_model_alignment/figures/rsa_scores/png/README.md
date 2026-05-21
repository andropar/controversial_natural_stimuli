# Png

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Figure Descriptions

The notes below summarize what each rendered figure shows. The file table that follows points back to the scripts and inputs used to make it.

**`brain_alignment_improved`.** This is the main raw brain-model RSA alignment figure. It compares
controversial stimuli against the baseline across model-set families and RSA endpoints, with spread-ratio and
noise-ceiling context where available, using the CRSA/WRSA RSA score tables and noise-ceiling summaries. Read
it as the primary visual summary of the alignment drop or spread change.

**`brain_alignment_improved_with_shared`.** This brain-alignment panel extends the main RSA plot with
shared-model or benchmark comparisons. It compares controversial and baseline images across mixed and fixed
RSA endpoints, using the CRSA/WRSA RSA score tables and noise-ceiling summaries, so it is useful for checking
whether the effect is specific to the evaluated model set or also appears in broader benchmark models.

**`brain_alignment_nc`.** This brain-alignment figure overlays or incorporates noise-ceiling information for
the controversial and baseline conditions. It uses the CRSA/WRSA RSA score tables and noise-ceiling summaries
to help separate an apparent alignment change from a possible reliability or ceiling change.

**`brain_alignment_subset_scatter_improved`.** This subset scatter plot shows how brain-model alignment
changes across stimulus subsets or model-set cells. It is a local diagnostic for whether the main effect is
broadly distributed or driven by a small number of subsets, using the CRSA/WRSA RSA score tables and
noise-ceiling summaries.

**`large_benchmark_gap_norm_improved`.** This benchmark-gap figure shows the normalized difference between
controversial and baseline alignment for large benchmark models. It uses the CRSA/WRSA RSA score tables and
noise-ceiling summaries to show whether high-performing external models close or retain the
controversial-stimulus gap.

**`rank_shift_improved`.** This rank-shift plot visualizes how individual model rankings move between baseline
and controversial conditions. It uses the CRSA/WRSA RSA score tables and noise-ceiling summaries to identify
whether the effect is a broad score shift or a change in relative model ordering.

## Contents Snapshot

- Folder: `01_brain_model_alignment/figures/rsa_scores/png`
- Figures in this folder tree: 6
- Data/table-like files in this folder tree: 0
- Python scripts in this folder tree: 0
- Direct files: `brain_alignment_improved.png`, `brain_alignment_improved_with_shared.png`, `brain_alignment_nc.png`, `brain_alignment_subset_scatter_improved.png`, `large_benchmark_gap_norm_improved.png`, `rank_shift_improved.png`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Provenance

Each row is one figure concept; `formats` lists the concrete files present in this folder.

| Figure | Formats | What it shows | Source / derivation | Data or inputs | Script | Paper use |
|---|---:|---|---|---|---|---|
| `brain_alignment_improved` | png | raw brain-model RSA alignment for controversial versus baseline images | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/02_rsa_scores/figures/brain_alignment_improved.png`; historical share path `data/tier1_analysis_derivatives/cstim_paper/02_rsa_scores/figures/brain_alignment_improved.png` | `01_brain_model_alignment/rsa_scores/*/{crsa_scores,wrsa_transfer_scores}.csv`; `crsa_scores.csv`; `wrsa_transfer_scores.csv`; `rdm_noise_ceilings.csv`; `between_subject_noise_ceilings.csv`; `permutation_test_results.csv`; plus 9 more | `01_brain_model_alignment/code/rsa_scoring/figures/plot_brain_alignment.py`; `01_brain_model_alignment/code/rsa_scoring/figures/plot_brain_alignment_improved.py` | no |
| `brain_alignment_improved_with_shared` | png | brain-model alignment including shared-model/benchmark comparisons | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/02_rsa_scores/figures/brain_alignment_improved_with_shared.png`; historical share path `data/tier1_analysis_derivatives/cstim_paper/02_rsa_scores/figures/brain_alignment_improved_with_shared.png` | `01_brain_model_alignment/rsa_scores/*/{crsa_scores,wrsa_transfer_scores}.csv`; `crsa_scores.csv`; `wrsa_transfer_scores.csv`; `rdm_noise_ceilings.csv`; `between_subject_noise_ceilings.csv`; `permutation_test_results.csv`; plus 9 more | `01_brain_model_alignment/code/rsa_scoring/figures/plot_brain_alignment_improved_with_shared.py` | no |
| `brain_alignment_nc` | png | figure derived from brain alignment nc | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/02_rsa_scores/figures/brain_alignment_nc.png`; historical share path `data/tier1_analysis_derivatives/cstim_paper/02_rsa_scores/figures/brain_alignment_nc.png` | `01_brain_model_alignment/rsa_scores/*/{crsa_scores,wrsa_transfer_scores}.csv`; `crsa_scores.csv`; `wrsa_transfer_scores.csv`; `rdm_noise_ceilings.csv`; `between_subject_noise_ceilings.csv`; `permutation_test_results.csv`; plus 13 more | `01_brain_model_alignment/code/rsa_scoring/figures/plot_brain_alignment.py`; `01_brain_model_alignment/code/rsa_scoring/figures/plot_brain_alignment_improved.py`; `02_alignment_reliability/code/figures/plot_brain_alignment_nc_normalized.py` | no |
| `brain_alignment_subset_scatter_improved` | png | subset-level brain-model alignment scatter comparison | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/02_rsa_scores/figures/brain_alignment_subset_scatter_improved.png`; historical share path `data/tier1_analysis_derivatives/cstim_paper/02_rsa_scores/figures/brain_alignment_subset_scatter_improved.png` | `01_brain_model_alignment/rsa_scores/*/{crsa_scores,wrsa_transfer_scores}.csv`; `FIGURES_DIR`; `wrsa_transfer_scores.csv`; `crsa_scores.csv`; `subset_scores_K{max_stim}.csv`; `STATS_DATA_DIR`; plus 7 more | `01_brain_model_alignment/code/rsa_scoring/figures/plot_brain_alignment_subset_improved.py`; `01_brain_model_alignment/code/rsa_scoring/figures/plot_brain_alignment_subset.py` | no |
| `large_benchmark_gap_norm_improved` | png | normalized benchmark gap between controversial and baseline alignment | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/02_rsa_scores/figures/large_benchmark_gap_norm_improved.png`; historical share path `data/tier1_analysis_derivatives/cstim_paper/02_rsa_scores/figures/large_benchmark_gap_norm_improved.png` | `01_brain_model_alignment/rsa_scores/*/{crsa_scores,wrsa_transfer_scores}.csv`; `rsa_large_benchmark_scores.csv`; `cross_set_wrsa_scores.csv`; `wrsa_transfer_scores.csv`; `crsa_scores.csv`; `rdm_noise_ceilings.csv`; plus 13 more | `01_brain_model_alignment/code/rsa_scoring/figures/plot_large_benchmark_gap.py`; `01_brain_model_alignment/code/rsa_scoring/figures/plot_large_benchmark_gap_improved.py` | no |
| `rank_shift_improved` | png | model rank-shift summary between controversial and baseline conditions | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/02_rsa_scores/figures/rank_shift_improved.png`; historical share path `data/tier1_analysis_derivatives/cstim_paper/02_rsa_scores/figures/rank_shift_improved.png` | `01_brain_model_alignment/rsa_scores/*/{crsa_scores,wrsa_transfer_scores}.csv`; `rank_correlations.csv`; `SHARE_ROOT`; `STAGE_DIR`; `FIGURES_DIR`; `PNG_DIR`; plus 10 more | `01_brain_model_alignment/code/rsa_scoring/figures/plot_rank_shift_improved.py`; `01_brain_model_alignment/code/rsa_scoring/figures/plot_rank_shift.py` | no |
<!-- END AUTO-FIGURE-PROVENANCE -->
