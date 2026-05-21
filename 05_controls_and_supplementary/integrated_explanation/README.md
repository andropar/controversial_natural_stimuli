# Integrated Explanation

This section asks what explains the controversial-stimulus effect on
brain-model alignment. The primary estimand is the subject-level mixed-RSA
delta:

```text
delta = RSA(controversial) - RSA(counterfactual baseline)
```

The primary explanatory analysis is a matched-counterfactual ladder, not a
large pair-level regression. Pair-level variance partitioning is included as a
mechanistic follow-up with image-blocked cross-validation.

## Pipeline

```bash
/data/home_roth/_stachelschwein/miniforge3/bin/python code/analysis/01_matched_counterfactual_ladder.py
/data/home_roth/_stachelschwein/miniforge3/bin/python code/analysis/02_reliability_and_residual_readout.py
/data/home_roth/_stachelschwein/miniforge3/bin/python code/analysis/03_pair_level_variance_partition.py
/data/home_roth/_stachelschwein/miniforge3/bin/python code/analysis/04_make_explanation_summary_figure.py
```

or:

```bash
/data/home_roth/_stachelschwein/miniforge3/bin/python code/analysis/run_all.py
```

## Outputs

- `data/matched_counterfactual_ladder_by_cell.csv`
- `data/matched_counterfactual_ladder_summary.csv`
- `data/reliability_control_by_cell.csv`
- `data/reliability_control_summary.csv`
- `data/residual_readout_summary.csv`
- `data/pair_variance_partition_by_cell.csv`
- `data/pair_variance_partition_summary.csv`
- `figures/explanation_summary.{pdf,png}`

## Interpretation

- If the delta disappears after low-level matching, the effect is plausibly
  low-level.
- If it disappears after OOD matching, OOD-ness is plausibly doing the work.
- If it persists after the combined low-level + OOD + embedding match, the
  model-disagreement selection explanation is stronger.
- If raw RSA drops but noise-ceiling-normalized RSA does not, reliability is a
  measurement explanation.
- If residual RDM structure remains reliable after removing current model RDMs,
  the stimuli expose reliable brain geometry not captured by the current model
  family.

Set-level regressions over the five controversial sets are intentionally not
used as the primary evidence.

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Contents Snapshot

- Folder: `05_controls_and_supplementary/integrated_explanation`
- Figures in this folder tree: 2
- Data/table-like files in this folder tree: 9
- Python scripts in this folder tree: 5
- Main child folders: `code/`, `data/`, `figures/`
- Direct files: `REPORT.md`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Index

| Figure directory | Figures | README |
|---|---:|---|
| `05_controls_and_supplementary/integrated_explanation/figures` | 1 | `05_controls_and_supplementary/integrated_explanation/figures/README.md` |
| `05_controls_and_supplementary/integrated_explanation/figures/png` | 1 | `05_controls_and_supplementary/integrated_explanation/figures/png/README.md` |
<!-- END AUTO-FIGURE-PROVENANCE -->
