# RSA Score Tables

This folder contains the staged brain-model alignment scores. These are the
tables most downstream stages read when they refer to the cstim
brain-alignment scores.

## Per-Subject Scores

Each subject folder (`sub-01`, `sub-03`, `sub-05`, `sub-06`, `sub-07`) contains:

- `crsa_scores.csv`: classical/fixed RSA scores. Rows with
  `stimulus_type == "controversial"` are the cstim scores; rows with
  `stimulus_type == "vicco"` are baseline bootstrap scores.
- `wrsa_transfer_scores.csv`: transferred weighted/mixed RSA scores using
  fitted encoding models. It uses the same `stimulus_type` convention.
- `cross_set_wrsa_scores.csv`: cross-set weighted RSA checks.

The main columns are `subject`, `model_set`, `model`, `display_name`,
`stimulus_type`, `bootstrap_idx`, `n_stimuli`, and the score column (`crsa` or
`wrsa_transfer`).

## Benchmark Summaries

- `rsa_large_benchmark_scores.csv`: large benchmark model alignment scores.
- `benchmark_upper_tail_deltas.csv`: per-subject upper-tail cstim-vs-baseline
  deltas.
- `benchmark_upper_tail_deltas_summary.csv`: subject-aggregated upper-tail
  summaries.

## Producing Code

The score-producing scripts live in `../code/rsa_scoring/`. Start with
`../code/rsa_scoring/README.md` for the script-to-output map.

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Contents Snapshot

- Folder: `01_brain_model_alignment/rsa_scores`
- Figures in this folder tree: 0
- Data/table-like files in this folder tree: 18
- Python scripts in this folder tree: 0
- Direct files: `benchmark_upper_tail_deltas.csv`, `benchmark_upper_tail_deltas_summary.csv`, `rsa_large_benchmark_scores.csv`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.
<!-- END AUTO-FIGURE-PROVENANCE -->
