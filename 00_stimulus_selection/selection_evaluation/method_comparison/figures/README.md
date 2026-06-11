# Figures

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Figure Descriptions

The notes below summarize what each rendered figure shows. The file table that follows points back to the scripts and inputs used to make it.

**`track_composition_leaderboard_improved`.** This leaderboard ranks candidate selection-track compositions by
in-silico model-recovery AUC. It compares random, raw-only, group-encoding, subject-encoding, and
raw-plus-encoding settings, and identifies the best final track composition used for the frozen selection
inputs, using the frozen selection-evaluation CSVs and model-RDM summaries.

## Contents Snapshot

- Folder: `00_stimulus_selection/selection_evaluation/method_comparison/figures`
- Figures in this folder tree: 1
- Data/table-like files in this folder tree: 0
- Python scripts in this folder tree: 1
- Direct files: `plot_track_composition_leaderboard_improved.py`, `track_composition_leaderboard_improved.pdf`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Provenance

Each row is one figure concept; `formats` lists the concrete files present in this folder.

| Figure | Formats | What it shows | Source / derivation | Resources or results | Script | Paper use |
|---|---:|---|---|---|---|---|
| `track_composition_leaderboard_improved` | pdf | selection-method leaderboard comparing track compositions and checkpoint choices | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/scripts/claude/compare_selection_methods/figures/track_composition_leaderboard_improved.pdf` | `00_stimulus_selection/selection_evaluation/method_comparison/results/*.csv`; `scripts/cursor/outputs/final_aggregate_plot/leaderboard_summary.csv`; `REPO_ROOT`; `OUTPUT_DIR`; `00_stimulus_selection/selection_evaluation/results/all_models/ablations.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/correlation_matrices.csv`; plus 4 more | `00_stimulus_selection/selection_evaluation/method_comparison/figures/plot_track_composition_leaderboard_improved.py` | no |
<!-- END AUTO-FIGURE-PROVENANCE -->
