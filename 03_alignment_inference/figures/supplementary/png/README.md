# Png

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Figure Descriptions

The notes below summarize what each rendered figure shows. The file table that follows points back to the scripts and inputs used to make it.

**`rank_null`.** This null-distribution figure gives the permutation or resampling reference for rank-shift
statistics. It uses the CRSA/WRSA RSA score tables and noise-ceiling summaries to show whether the observed
rank changes are larger than expected under the analysis null.

**`spread_amplification`.** This plot summarizes whether controversial stimuli amplify the spread of
model-brain alignment scores relative to baseline. It uses the adjacent staged data/results files named in the
provenance table to test whether the selected images increase separability among models.

## Contents Snapshot

- Folder: `03_alignment_inference/figures/supplementary/png`
- Figures in this folder tree: 2
- Data/table-like files in this folder tree: 0
- Python scripts in this folder tree: 0
- Direct files: `rank_null.png`, `spread_amplification.png`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Provenance

Each row is one figure concept; `formats` lists the concrete files present in this folder.

| Figure | Formats | What it shows | Source / derivation | Resources or results | Script | Paper use |
|---|---:|---|---|---|---|---|
| `rank_null` | png | rank-shift null or permutation reference distribution | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/03_statistics/figures/rank_null.png`; historical share path `data/tier1_analysis_derivatives/cstim_paper/03_statistics/figures/rank_null.png` | `01_brain_model_alignment/results/rsa_scores/*/{crsa_scores,wrsa_transfer_scores}.csv`; `{method}_scores.csv`; `rank_null.csv`; `SHARE_ROOT`; `OUT_DATA`; `03_alignment_inference/data/discriminability_paper_summary.csv`; plus 5 more | `03_alignment_inference/code/11_rank_correlation_null.py` | no |
| `spread_amplification` | png | spread/amplification summary for controversial versus baseline scores | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/03_statistics/figures/spread_amplification.png`; historical share path `data/tier1_analysis_derivatives/cstim_paper/03_statistics/figures/spread_amplification.png` | `spread_statistics.csv`; `spread_statistics_summary.csv`; `SHARE_ROOT`; `DATA_DIR`; `FIGURES_DIR`; `PNG_DIR`; plus 6 more | `03_alignment_inference/code/figures/plot_spread_amplification.py` | no |
<!-- END AUTO-FIGURE-PROVENANCE -->
