# Held-Out Unique-Image Baseline

This active analysis constructs an independent natural-image control from each
subject's DeepVision unique images. The intended full run uses repeated 80/20
subject-specific splits, fits evaluation encodings only on the 80% training
fold, and evaluates mixed RSA on:

- controversial stimuli from the later CSTIMS sessions,
- same-session unselected natural-image baseline stimuli,
- held-out unique-image baseline subsets from the earlier DeepVision dataset,
- embedding-PC, low-level-statistic, feature-PPCA-OOD, and combined matched
  held-out unique subsets when common support is adequate,
- high feature-PPCA-OOD held-out subsets as a natural-image stress control.

The low-level covariate builder is `00_compute_unique_low_level_stats.py`. The
main analysis script is `01_heldout_unique_baseline.py`, and
`02_summarize_heldout_unique.py` builds paper-facing aggregate tables and a
summary figure. The analysis writes:

- `data/unique_image_low_level_stats.csv`
- `data/heldout_unique_splits.csv`
- `data/heldout_unique_baseline_results.csv`
- `data/baseline_matching_diagnostics.csv`
- `data/matched_baseline_results.csv`
- `data/heldout_unique_endpoint_by_split.csv`
- `data/heldout_unique_endpoint_summary.csv`
- `data/heldout_unique_aggregate_summary.csv`
- `data/heldout_unique_completion_status.csv`
- `figures/heldout_unique_baseline_summary.{pdf,png}`

Low-level matching for unique images requires per-image low-level statistics for
the subject-specific unique pools. If those statistics are absent, the script
records that status in `baseline_matching_diagnostics.csv` rather than forcing an
unsupported match.

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Contents Snapshot

- Folder: `05_controls_and_supplementary/counterfactual_baselines`
- Figures in this folder tree: 4
- Data/table-like files in this folder tree: 9
- Python scripts in this folder tree: 4
- Main child folders: `code/`, `data/`, `figures/`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Index

| Figure directory | Figures | README |
|---|---:|---|
| `05_controls_and_supplementary/counterfactual_baselines/figures` | 1 | `05_controls_and_supplementary/counterfactual_baselines/figures/README.md` |
| `05_controls_and_supplementary/counterfactual_baselines/figures/png` | 1 | `05_controls_and_supplementary/counterfactual_baselines/figures/png/README.md` |
| `05_controls_and_supplementary/counterfactual_baselines/figures/supplementary` | 1 | `05_controls_and_supplementary/counterfactual_baselines/figures/supplementary/README.md` |
| `05_controls_and_supplementary/counterfactual_baselines/figures/supplementary/png` | 1 | `05_controls_and_supplementary/counterfactual_baselines/figures/supplementary/png/README.md` |
<!-- END AUTO-FIGURE-PROVENANCE -->
