# 03 Alignment Inference

This stage contains the main statistical inference attached to the alignment
effects: spread summaries, permutation tests, bootstrap summaries, rank-null
analyses, subset score summaries, and paper-facing canonical tables. It is
separate from `01_brain_model_alignment/` because `01` stores the alignment
score production itself, while this folder stores the statistical summaries
that interpret those scores.

The code in `code/` consumes the RSA outputs from `01_brain_model_alignment/`
and writes compact result tables to `data/`; the currently retained rendered
inference figures are in `figures/`.

## How To Read This Stage

Use `data/primary_endpoint_summary.csv`,
`data/effect_sizes_brain_alignment.csv`, and
`data/permutation_test_results.csv` for the headline statistical readout. The
remaining tables break the same question down by spread statistics, bootstrap
uncertainty, model family, rank-null analyses, and selected subsets.

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Contents Snapshot

- Folder: `03_alignment_inference`
- Figures in this folder tree: 4
- Data/table-like files in this folder tree: 18
- Python scripts in this folder tree: 9
- Main child folders: `code/`, `data/`, `figures/`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Index

| Figure directory | Figures | README |
|---|---:|---|
| `03_alignment_inference/figures/supplementary` | 2 | `03_alignment_inference/figures/supplementary/README.md` |
| `03_alignment_inference/figures/supplementary/png` | 2 | `03_alignment_inference/figures/supplementary/png/README.md` |
<!-- END AUTO-FIGURE-PROVENANCE -->
