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
- `figures/pairwise_dominance_margin_{raw,encoding}.{pdf,png}`

For the proper Raven rerun, submit the model-set jobs through the same SLURM
wrapper pattern as the original evaluation. Do not pass `--random-feature-dir`:

```bash
bash 00_stimulus_selection/selection_evaluation/noisy_by_clean_recovery/code/run_noisy_by_clean_recovery_slurm.sh
```

By default this uses `/u/rothj/laion_natural/scripts/start_as_slurm_job.py`.
If Raven has a different wrapper path, set `CSTIMS_SLURM_WRAPPER` first.

The submitter uses the original unique-evaluation parameters:

- `env = raven`
- `which_selection = final`
- `n_random_subsets = 50`
- `n_noise_samples = 100`
- `n_bootstrap = 500`
- `seed = 42`
- metric/correlation from the payload config, with the original fallbacks
  `cosine`/`spearman`
- unique per-subject encodings

That uses:

- selection payloads from `00_stimulus_selection/results/selected_stimuli/`
- random-baseline features from the candidate pool in
  `00_stimulus_selection/resources/configs/paths/raven.yaml`
- unique per-subject encoding roots from `cstims.paper.config`

After all five jobs finish, generate the plot:

```bash
python 00_stimulus_selection/selection_evaluation/noisy_by_clean_recovery/code/plot_noisy_by_clean_recovery.py
python 00_stimulus_selection/selection_evaluation/noisy_by_clean_recovery/code/plot_pairwise_margin.py
```

The pairwise diagnostic writes two additional files in each model-set result
directory:

- `pairwise_margin.csv`: curves for pairwise dominance and mean correlation margin
- `pairwise_auc.csv`: AUC summaries for those curves

The wrapper keeps `--random-feature-dir` only for local smoke tests with cached
`.npz` features:

```bash
bash 00_stimulus_selection/selection_evaluation/noisy_by_clean_recovery/code/run_noisy_by_clean_recovery.sh \
  --random-feature-dir shared/cache_or_heavy/natural_pool_subset_10k \
  --n-random-subsets 10 \
  --n-noise-samples 50 \
  --n-bootstrap 100
```

Existing small-cache outputs in this directory are not the full candidate-pool
rerun; the Raven command above overwrites them.
