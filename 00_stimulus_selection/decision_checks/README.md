# Decision Checks

This folder contains analyses that were used to evaluate or sanity-check the stimulus-selection procedure. 

## `selection_evaluation/`

This is the copied selection-evaluation output for the five final stimulus
sets. It includes per-set summaries and figures that check whether the selected
images separate the model groups better than random or baseline image samples.

Important files under `selection_evaluation/data/<set>/`:

- `summary_report.md`: short human-readable summary for one selected set.
- `summary.csv`: run-level scalar metadata.
- `statistics.csv`: selected-vs-random discriminability statistics.
- `discriminability.csv`: per-track/model discriminability curves and AUCs.
- `correlation_matrices.csv`: model-by-model RDM correlation matrices for
  selected and reference subsets.
- `diversity.csv`: pairwise feature diversity of the selected images.
- `greedy_scores.csv`: selection objective trajectory during greedy selection.
- `refinement.csv`: post-greedy replacement/refinement history.
- `filter_records.csv`: natural-image filtering and validation records.
- `filter_summary.json`: summary of filtering outcomes.
- `noise_calibration.csv`: calibrated noise values used for selection tracks.
- `ablations.csv`: selection/refinement ablation diagnostics.

Cross-set decision-check tables live directly under
`selection_evaluation/data/`:

- `selection_objective_summary.csv`: objective readouts by model set, track,
  subset type, and noise condition.
- `selection_objective_by_model.csv`: model-level objective components.
- `selection_objective_combined.csv`: combined objective summaries.
- `selection_track_composition_summary.csv`: comparison of raw-only,
  group-only, raw-plus-single-subject, and raw-plus-all-subject selection
  compositions.
- `selection_track_contribution_summary.csv`: contribution of all-subject
  encoding tracks relative to baselines.
- `selection_leave_one_subject_out_feasibility.md`: note explaining why a true
  leave-one-subject-out selection audit cannot be inferred from available runs.

## `selection_evaluation/method_comparison/`

This folder contains the selection-method comparison that produced the
leaderboard figure used in the paper. It compares the standard v2/full
selection against the power-mean variant and final selections against
`best_raw_combined` checkpoints.

Important files:

- `README.md`: summary of the comparison and recommendation.
- `data/all_auc_results.csv`: AUC table used by the comparison.
- `data/image_overlap.csv`: selected-image overlap table.
- `figures/track_composition_leaderboard_improved.pdf`: leaderboard figure
  copied into the manuscript as `selection_method_comparison_improved.pdf`.
- `figures/plot_track_composition_leaderboard_improved.py`: plotting script for
  the improved leaderboard.

The `figures/` subfolders contain visual versions of these diagnostics. Where
both original and `_improved` figure versions existed, the package keeps the
newer `_improved` version.

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Contents Snapshot

- Folder: `00_stimulus_selection/decision_checks`
- Figures in this folder tree: 115
- Data/table-like files in this folder tree: 122
- Python scripts in this folder tree: 31

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Index

| Figure directory | Figures | README |
|---|---:|---|
| `00_stimulus_selection/decision_checks/selection_evaluation/figures/correlation_matrices_eval_pipeline` | 10 | `00_stimulus_selection/decision_checks/selection_evaluation/figures/correlation_matrices_eval_pipeline/README.md` |
| `00_stimulus_selection/decision_checks/selection_evaluation/figures/correlation_matrices_eval_pipeline/png` | 10 | `00_stimulus_selection/decision_checks/selection_evaluation/figures/correlation_matrices_eval_pipeline/png/README.md` |
| `00_stimulus_selection/decision_checks/selection_evaluation/figures/correlation_matrices_pool` | 20 | `00_stimulus_selection/decision_checks/selection_evaluation/figures/correlation_matrices_pool/README.md` |
| `00_stimulus_selection/decision_checks/selection_evaluation/figures/correlation_matrices_pool/png` | 20 | `00_stimulus_selection/decision_checks/selection_evaluation/figures/correlation_matrices_pool/png/README.md` |
| `00_stimulus_selection/decision_checks/selection_evaluation/figures/correlation_matrices_vicco` | 20 | `00_stimulus_selection/decision_checks/selection_evaluation/figures/correlation_matrices_vicco/README.md` |
| `00_stimulus_selection/decision_checks/selection_evaluation/figures/correlation_matrices_vicco/png` | 20 | `00_stimulus_selection/decision_checks/selection_evaluation/figures/correlation_matrices_vicco/png/README.md` |
| `00_stimulus_selection/decision_checks/selection_evaluation/figures/insilico_curve` | 7 | `00_stimulus_selection/decision_checks/selection_evaluation/figures/insilico_curve/README.md` |
| `00_stimulus_selection/decision_checks/selection_evaluation/figures/insilico_curve/png` | 7 | `00_stimulus_selection/decision_checks/selection_evaluation/figures/insilico_curve/png/README.md` |
| `00_stimulus_selection/decision_checks/selection_evaluation/method_comparison/figures` | 1 | `00_stimulus_selection/decision_checks/selection_evaluation/method_comparison/figures/README.md` |
<!-- END AUTO-FIGURE-PROVENANCE -->
