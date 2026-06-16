# Feature Method Sweep

Feature-only audit of stimulus-selection objectives. This intentionally skips
image filtering, image download, and refinement, then evaluates each selected
feature set with the corrected noisy-by-clean recovery code.

Default sweep:

- `raw_only_mean_min`
- `raw_only_mean_min_no_attenuation`
- `sub01_only_mean_min`
- `sub01_only_mean_min_no_attenuation`
- `paper_effective_identity_sub01_mean_min`
- `paper_effective_identity_sub01_mean_min_no_attenuation`
- `raw_enc_w05_mean_min`
- `raw_enc_w05_mean_min_no_attenuation`
- `raw_enc_w05_max_mean`
- `raw_enc_w05_max_min`

The paper-effective condition uses the same effective track-combination bug as
the frozen selected stimuli: raw plus five subject encodings are configured, but
`track_aggregation.agg_method=identity` makes the first track, `sub-01`, the
actual selection score.

The no-attenuation paper-effective condition uses the same effective selector as
the frozen selected stimuli, but sets the selection-objective noise to zero. Its
recovery evaluation still uses the run-level target noise.

For local `.npz` pool-cache sweeps, pass `--pool-feature-dir` to the Python
entry point or `POOL_FEATURE_DIR=...` to the shell wrapper. When
`--random-feature-dir` is omitted, recovery can reuse the same pool cache as the
random baseline by setting `RANDOM_FEATURE_DIR="$POOL_FEATURE_DIR"`.

## Raven

Submit the default SOTA run:

```bash
bash 00_stimulus_selection/feature_method_sweep/code/run_feature_method_sweep_slurm.sh
```

Submit the nested candidate-pool-size sweep used to test pool-size effects:

```bash
bash 00_stimulus_selection/feature_method_sweep/code/run_pool_size_sweep_slurm.sh
```

By default this runs the six current methods over nested pool sizes
`1k,10k,50k,100k,250k,500k,1M,5M,10M`, writes one standard payload directory per
pool under `results/<run_name>/pool_*`, and skips the legacy noisy-by-clean
evaluation. Evaluate those payloads with the current teacher-student recovery
pipeline.

Useful overrides:

```bash
MAX_RAM_GB=100 TARGET_SIZE=100 N_RANDOM_SUBSETS=50 N_BOOTSTRAP=500 \
bash 00_stimulus_selection/feature_method_sweep/code/run_feature_method_sweep_slurm.sh
```

Adaptive batch sizing can grow the candidate batch size and back off if CUDA
runs out of memory:

```bash
ADAPTIVE_BATCH_SIZE=1 BATCH_SIZE=2500 MAX_BATCH_SIZE=10000 PROGRESS_EVERY_BATCHES=1 \
bash 00_stimulus_selection/feature_method_sweep/code/run_feature_method_sweep_slurm.sh
```

The selection script writes payloads and progress under:

```text
00_stimulus_selection/feature_method_sweep/results/
```

The noisy-by-clean recovery evaluation for those payloads is written under:

```text
00_stimulus_selection/selection_evaluation/feature_method_sweep_recovery/noisy_by_clean/results/
```

Plot the recovery summaries with:

```bash
python 00_stimulus_selection/selection_evaluation/feature_method_sweep_recovery/noisy_by_clean/04_plot_summary.py
```

Important files:

- `payloads/method_manifest.csv`: exact method definitions.
- `selection_progress_latest.json`: latest selection progress state.
- `selection_progress.jsonl`: complete selection progress stream.
- `payloads/<method_id>/selected_indices.npy`: selected global indices.
- `payloads/<method_id>/selected_image_records.csv`: selected image provenance.
- `<run_name>/eval/<method_id>_noisy_by_clean_boot/auc_significance.csv`: strict top-1 recovery AUC.
- `<run_name>/eval/<method_id>_noisy_by_clean_boot/pairwise_auc.csv`: pairwise dominance and margin AUC.
- `<run_name>/comparison/method_summary.csv`: compact method comparison.
- `<run_name>/comparison/recovery_auc_by_method.csv`: all recovery AUC rows.
- `<run_name>/comparison/pairwise_auc_by_method.csv`: all pairwise AUC rows.

Strict top-1 recovery AUC is lower-is-better. Pairwise dominance and mean-margin
AUCs are higher-is-better.
