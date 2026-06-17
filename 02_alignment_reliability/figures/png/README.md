# Png

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Figure Descriptions

The notes below summarize what each rendered figure shows. The file table that follows points back to the scripts and inputs used to make it.

**`brain_alignment_nc_normalized`.** This plot repeats the brain-model alignment comparison after normalizing
by the estimated noise ceiling. It asks whether the controversial-versus-baseline RSA effect remains after
accounting for measurement reliability, using the CRSA/WRSA RSA score tables and noise-ceiling summaries.

## Contents Snapshot

- Folder: `02_alignment_reliability/figures/png`
- Figures in this folder tree: 1
- Data/table-like files in this folder tree: 0
- Python scripts in this folder tree: 0
- Direct files: `brain_alignment_nc_normalized.png`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Provenance

Each row is one figure concept; `formats` lists the concrete files present in this folder.

| Figure | Formats | What it shows | Source / derivation | Resources or results | Script | Paper use |
|---|---:|---|---|---|---|---|
| `brain_alignment_nc_normalized` | png | noise-ceiling-normalized brain-model alignment | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/03_statistics/figures/brain_alignment_nc_normalized.png`; historical share path `data/tier1_analysis_derivatives/cstim_paper/03_statistics/figures/brain_alignment_nc_normalized.png` | `02_alignment_reliability/results/*noise*.csv`; `01_brain_model_alignment/results/rsa_scores/*/{crsa_scores,wrsa_transfer_scores}.csv`; `crsa_scores.csv`; `wrsa_transfer_scores.csv`; `rdm_noise_ceilings.csv`; `between_subject_noise_ceilings.csv`; plus 13 more | `02_alignment_reliability/code/figures/02_plot_brain_alignment_nc_normalized.py` | no |
<!-- END AUTO-FIGURE-PROVENANCE -->
