# 02 Alignment Reliability

This stage contains the reliability and noise-ceiling analyses that qualify the
raw brain-model alignment results in `01_brain_model_alignment/`. Use it to ask
whether a controversial-stimulus alignment change is explained by lower
measurement reliability, or whether the effect remains after noise-ceiling
normalization and attenuation-style checks.

The code in `code/` produces the staged summary tables in `results/` and the
rendered reliability figures in `figures/`.

## How To Read This Stage

Start with `results/nc_normalized_summary.csv` and
`results/noise_ceiling_variant_summary.csv` for the compact answers. The scripts in
`code/` compute subject-level and between-subject noise ceilings, normalize RSA
scores against those ceilings, and summarize whether reliability differences
could explain the alignment effects.

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Contents Snapshot

- Folder: `02_alignment_reliability`
- Figures in this folder tree: 6
- Data/table-like files in this folder tree: 5
- Python scripts in this folder tree: 6
- Main child folders: `code/`, `results/`, `figures/`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Index

| Figure directory | Figures | README |
|---|---:|---|
| `02_alignment_reliability/figures` | 1 | `02_alignment_reliability/figures/README.md` |
| `02_alignment_reliability/figures/png` | 1 | `02_alignment_reliability/figures/png/README.md` |
| `02_alignment_reliability/figures/supplementary` | 2 | `02_alignment_reliability/figures/supplementary/README.md` |
| `02_alignment_reliability/figures/supplementary/png` | 2 | `02_alignment_reliability/figures/supplementary/png/README.md` |
<!-- END AUTO-FIGURE-PROVENANCE -->
