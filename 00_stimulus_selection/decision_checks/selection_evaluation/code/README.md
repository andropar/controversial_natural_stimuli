# Selection-Evaluation Producer Scripts

This folder contains the source scripts that produced the adjacent
`selection_evaluation` decision-check data and figures.

Source provenance:

- Original source root:
  `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation`
- `analysis/`: selection-evaluation data producers and diagnostic plotting
  helpers.
- `figures/`: paper-facing plotting scripts for the selection-evaluation
  figures.

These scripts were added to the share package after an audit found that the
initial copy step included `00_selection_evaluation` derivatives
(`data/`, `figures/`, `*.md`, `*.html`) but omitted the producer scripts from
the active code copy list.
