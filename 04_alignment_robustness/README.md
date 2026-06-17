# 04 Alignment Robustness

This stage holds robustness checks for the alignment conclusions, especially
distance-metric sensitivity and model-versus-brain spread diagnostics. These
analyses are downstream of the main RSA score tables but are conceptually
different from the primary inference tests, so they live outside
`03_alignment_inference/`.

The scripts in `code/` generate the robustness tables in `results/` and the
retained robustness figures in `figures/`.

## How To Read This Stage

Start with `results/distance_metric_claim_robustness_summary.csv` for the compact
metric-sensitivity check. The other data tables show the same robustness
questions at finer resolution: distance-metric variants, mixed-distance ranks,
and model-RDM spread diagnostics.

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Contents Snapshot

- Folder: `04_alignment_robustness`
- Figures in this folder tree: 8
- Data/table-like files in this folder tree: 6
- Python scripts in this folder tree: 8
- Main child folders: `code/`, `results/`, `figures/`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Index

| Figure directory | Figures | README |
|---|---:|---|
| `04_alignment_robustness/figures` | 4 | `04_alignment_robustness/figures/README.md` |
| `04_alignment_robustness/figures/png` | 4 | `04_alignment_robustness/figures/png/README.md` |
<!-- END AUTO-FIGURE-PROVENANCE -->
