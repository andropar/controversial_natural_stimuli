# Noisy-by-clean recovery check

This directory is an isolated rerun of the in-silico recovery/AUC analysis with
the recovery matrix oriented as:

```text
score[noisy_model, clean_model] = corr(noisy RDM, clean RDM)
```

The original `selection_evaluation/code/analysis/02_compute_discriminability.py`
used the opposite orientation for the recovery score tensor. The scripts here
reuse the original downstream aggregation, bootstrap, AUC, and plotting code, but
replace the score construction with noisy-by-clean recovery.

Outputs are written locally under:

- `results/<model_set>_noisy_by_clean_boot/`
- `figures/noisy_by_clean_recovery.{pdf,png}`

For the proper Raven rerun, use the original candidate pool via the Raven path
config. Do not pass `--random-feature-dir`:

```bash
export CSTIMS_SHARE_ROOT="$PWD"
bash 00_stimulus_selection/decision_checks/selection_evaluation/noisy_by_clean_recovery/code/run_noisy_by_clean_recovery.sh \
  --env raven \
  --n-random-subsets 50 \
  --n-noise-samples 100 \
  --n-bootstrap 1000
```

That uses:

- selection payloads from `00_stimulus_selection/results/selected_stimuli/`
- random-baseline features from the candidate pool in
  `00_stimulus_selection/resources/configs/paths/raven.yaml`
- unique per-subject encoding roots from `shared/code/paper_helpers/config.py`

The wrapper keeps `--random-feature-dir` only for local smoke tests with cached
`.npz` features:

```bash
bash 00_stimulus_selection/decision_checks/selection_evaluation/noisy_by_clean_recovery/code/run_noisy_by_clean_recovery.sh \
  --random-feature-dir shared/cache_or_heavy/natural_pool_subset_10k \
  --n-random-subsets 10 \
  --n-noise-samples 50 \
  --n-bootstrap 100
```

Existing small-cache outputs in this directory are not the full candidate-pool
rerun; the Raven command above overwrites them.
