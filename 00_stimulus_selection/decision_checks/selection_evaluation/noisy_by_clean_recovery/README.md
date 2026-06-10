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

The local workspace does not contain the old cluster memmap paths from the
selection payloads, so the compute wrapper can load random-baseline features from
`shared/cache_or_heavy/natural_pool_subset_10k`.

Run completed in this workspace with:

```text
n_random_subsets = 10
n_noise_samples = 50
n_bootstrap = 100
```

The verified local natural-pool cache is incomplete for two dataset models and
one legacy all-models-only model. The compute output therefore writes
`model_roster.csv` in each model-set result directory; in this run `all_models`
uses 18 cached models and `dataset` uses 3 cached models.
