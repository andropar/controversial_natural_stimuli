# Supplementary

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Figure Descriptions

The notes below summarize what each rendered figure shows. The file table that follows points back to the scripts and inputs used to make it.

**`nc_harmonised`.** This harmonized noise-ceiling figure places raw alignment, normalized alignment, and
attenuation/reliability readouts on a common scale. It uses the adjacent staged data/results files named in
the provenance table to evaluate whether reliability explains the alignment effect.

**`rdm_noise_ceilings`.** This figure reports RDM reliability or noise-ceiling estimates for the brain data.
It uses the adjacent staged data/results files named in the provenance table to show the maximum plausible RSA
alignment under the measured neural reliability.

## Contents Snapshot

- Folder: `02_alignment_reliability/figures/supplementary`
- Figures in this folder tree: 2
- Data/table-like files in this folder tree: 0
- Python scripts in this folder tree: 0
- Direct files: `nc_harmonised.pdf`, `rdm_noise_ceilings.pdf`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Provenance

Each row is one figure concept; `formats` lists the concrete files present in this folder.

| Figure | Formats | What it shows | Source / derivation | Data or inputs | Script | Paper use |
|---|---:|---|---|---|---|---|
| `nc_harmonised` | pdf | harmonized noise-ceiling and attenuation readout | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/03_statistics/figures/nc_harmonised.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/03_statistics/figures/nc_harmonised.pdf` | `02_alignment_reliability/data/*noise*.csv`; `rdm_noise_ceilings.csv`; `between_subject_noise_ceilings.csv`; `SHARE_ROOT`; `02_alignment_reliability/data/between_subject_noise_ceilings.csv`; `02_alignment_reliability/data/nc_normalized_scores.csv`; plus 3 more | `02_alignment_reliability/code/12_nc_harmonised.py` | yes |
| `rdm_noise_ceilings` | pdf | RDM split-half or subject noise-ceiling estimates | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/03_statistics/figures/rdm_noise_ceilings.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/03_statistics/figures/rdm_noise_ceilings.pdf` | `02_alignment_reliability/data/*noise*.csv`; `rdm_noise_ceilings.csv`; `SHARE_ROOT`; `DATA_DIR`; `FIGURES_DIR`; `PNG_DIR`; plus 5 more | `02_alignment_reliability/code/figures/plot_rdm_noise_ceilings.py` | no |
<!-- END AUTO-FIGURE-PROVENANCE -->
