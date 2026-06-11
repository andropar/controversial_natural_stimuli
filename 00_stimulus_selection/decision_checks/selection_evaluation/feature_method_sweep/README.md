# Feature Method Sweep

Feature-only audit of stimulus-selection objectives. This intentionally skips
image filtering, image download, and refinement, then evaluates each selected
feature set with the corrected noisy-by-clean recovery code.

Default sweep:

- `raw_only_mean_min`
- `sub01_only_mean_min`
- `paper_effective_identity_sub01_mean_min`
- `raw_enc_w05_mean_min`
- `raw_enc_w05_max_mean`
- `raw_enc_w05_max_min`

The paper-effective condition uses the same effective track-combination bug as
the frozen selected stimuli: raw plus five subject encodings are configured, but
`track_aggregation.agg_method=identity` makes the first track, `sub-01`, the
actual selection score.

## Raven

Submit the default SOTA run:

```bash
bash 00_stimulus_selection/decision_checks/selection_evaluation/feature_method_sweep/code/run_feature_method_sweep_slurm.sh
```

Useful overrides:

```bash
MAX_RAM_GB=100 TARGET_SIZE=100 N_RANDOM_SUBSETS=50 N_BOOTSTRAP=500 \
bash 00_stimulus_selection/decision_checks/selection_evaluation/feature_method_sweep/code/run_feature_method_sweep_slurm.sh
```

The script writes one timestamped output tree under:

```text
00_stimulus_selection/decision_checks/selection_evaluation/feature_method_sweep/results/
```

Important files:

- `payloads/method_manifest.csv`: exact method definitions.
- `payloads/<method_id>/selected_indices.npy`: selected global indices.
- `payloads/<method_id>/selected_image_records.csv`: selected image provenance.
- `eval/<method_id>_noisy_by_clean_boot/auc_significance.csv`: strict top-1 recovery AUC.
- `eval/<method_id>_noisy_by_clean_boot/pairwise_auc.csv`: pairwise dominance and margin AUC.
- `comparison/method_summary.csv`: compact method comparison.
- `comparison/recovery_auc_by_method.csv`: all recovery AUC rows.
- `comparison/pairwise_auc_by_method.csv`: all pairwise AUC rows.

Strict top-1 recovery AUC is lower-is-better. Pairwise dominance and mean-margin
AUCs are higher-is-better.
