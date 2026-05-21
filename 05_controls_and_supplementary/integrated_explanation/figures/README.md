# Figures

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Figure Descriptions

The notes below summarize what each rendered figure shows. The file table that follows points back to the scripts and inputs used to make it.

**`explanation_summary`.** This integrated explanation figure combines the reliability,
matched-counterfactual, pair-level variance-partitioning, and residual-structure readouts. It uses the
integrated explanation-analysis tables as the compact answer to whether low-level features, OOD-ness,
semantics, reliability, or model-disagreement structure explain the alignment effect.

## Contents Snapshot

- Folder: `05_controls_and_supplementary/integrated_explanation/figures`
- Figures in this folder tree: 1
- Data/table-like files in this folder tree: 0
- Python scripts in this folder tree: 0
- Direct files: `explanation_summary.pdf`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Provenance

Each row is one figure concept; `formats` lists the concrete files present in this folder.

| Figure | Formats | What it shows | Source / derivation | Data or inputs | Script | Paper use |
|---|---:|---|---|---|---|---|
| `explanation_summary` | pdf | integrated explanation summary combining matched controls, reliability, and variance partitioning | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/18_explain_alignment_effect/figures/explanation_summary.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/18_explain_alignment_effect/figures/explanation_summary.pdf` | `matched_counterfactual_ladder_summary.csv`; `reliability_control_summary.csv`; `pair_variance_partition_summary.csv`; `residual_readout_contrasts.csv`; `05_controls_and_supplementary/integrated_explanation/data/matched_counterfactual_decision_table.csv`; `05_controls_and_supplementary/integrated_explanation/data/matched_counterfactual_ladder_by_cell.csv`; plus 4 more | `05_controls_and_supplementary/integrated_explanation/code/analysis/04_make_explanation_summary_figure.py` | no |
<!-- END AUTO-FIGURE-PROVENANCE -->
