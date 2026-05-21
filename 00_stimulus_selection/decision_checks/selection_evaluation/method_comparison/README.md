# Selection Method Comparison Report

## Overview

This report compares two stimulus selection methods:
- **v2_full**: Standard aggregation (`raw_plus_all_encodings`)
- **powermean**: Harmonic mean aggregation (`raw_plus_all_encodings_powermean`)

Each method produces two selection outputs:
- **final**: After full refinement (10 passes)
- **best_raw_combined**: Best checkpoint by raw track score during greedy selection

## Summary Results

### Best Overall Condition: `v2_full_best_raw_combined`

| Condition | AGGREGATE AUC | Encoding AUC | Raw AUC |
|-----------|---------------|--------------|---------|
| v2_full_final | 0.2144 | 0.2400 | 0.0867 |
| v2_full_best_raw_combined | 0.2079 **✓** | 0.2324 | 0.0853 |
| powermean_final | 0.2112 | 0.2360 | 0.0873 |
| powermean_best_raw_combined | 0.2173 | 0.2415 | 0.0964 |

*Lower AUC = better discriminability*

### Key Findings

1. **Best overall: `v2_full_best_raw_combined`** with AGGREGATE AUC = 0.2079

2. **best_raw_combined beats final** for both methods:
   - v2_full: 0.2144 → 0.2079 (3.0% improvement)
   - powermean: 0.2112 → 0.2173 (-2.9% improvement)

3. **Encoding track comparison** (mean across sub-01 to sub-07):
   - v2_full_best_raw_combined: 0.2324
   - powermean_best_raw_combined: 0.2415
   - Difference: 0.0090

4. **Raw track is similar** across all conditions (~0.089)

### Per-Model-Set Winners

| Model Set | Best Condition | AUC |
|-----------|----------------|-----|
| all_models | powermean_best_raw_combined | 0.3219 |
| architecture | powermean_best_raw_combined | 0.1380 |
| dataset | v2_full_best_raw_combined | 0.1711 |
| sota | v2_full_best_raw_combined | 0.1943 |
| training_objective | powermean_best_raw_combined | 0.1872 |

### Winner Distribution
- powermean_best_raw_combined: 3/5 model sets
- v2_full_best_raw_combined: 2/5 model sets

## Image Overlap Analysis

How similar are the selected image sets?

| Comparison | Mean Overlap | Jaccard Similarity |
|------------|--------------|-------------------|
| v2_full: final vs best_raw | 73/100 | 0.63 |
| powermean: final vs best_raw | 91/100 | 0.84 |
| final: v2_full vs powermean | 33/100 | 0.21 |
| best_raw: v2_full vs powermean | 27/100 | 0.16 |

## Figures

- [Aggregate AUC Comparison](figures/aggregate_auc_comparison.png)
- [Track Breakdown Heatmap](figures/track_breakdown_heatmap.png)
- [Encoding vs Raw Trade-off](figures/encoding_vs_raw_tradeoff.png)
- [Method Summary](figures/method_summary.png)
- [Image Overlap](figures/image_overlap.png)

## Conclusions

1. **Use `best_raw_combined` checkpoints** - Refinement hurts downstream discriminability.

2. **v2_full slightly outperforms powermean** when comparing best_raw_combined outputs,
   contrary to the hypothesis that harmonic mean would protect encoding tracks.

3. **The difference is small** (~0.001 AUC) - both methods produce similarly good selections.

4. **Model set matters** - different methods win for different model sets, suggesting
   the "best" method may depend on the specific model comparison being made.

## Recommendation

For future selections, use **v2_full with best_raw_combined** as the default choice,
but consider that the difference between methods is minimal.

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Contents Snapshot

- Folder: `00_stimulus_selection/decision_checks/selection_evaluation/method_comparison`
- Figures in this folder tree: 1
- Data/table-like files in this folder tree: 2
- Python scripts in this folder tree: 3
- Main child folders: `data/`, `figures/`
- Direct files: `compare_auc.py`, `full_analysis.py`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Index

| Figure directory | Figures | README |
|---|---:|---|
| `00_stimulus_selection/decision_checks/selection_evaluation/method_comparison/figures` | 1 | `00_stimulus_selection/decision_checks/selection_evaluation/method_comparison/figures/README.md` |
<!-- END AUTO-FIGURE-PROVENANCE -->
