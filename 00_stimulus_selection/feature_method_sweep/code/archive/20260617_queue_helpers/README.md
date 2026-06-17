# Archived queue helpers

These scripts were superseded by:

```text
00_stimulus_selection/feature_method_sweep/code/queue_all_model_set_pool_size_sweeps_slurm.sh
```

The active script launches all model-set pool-size sweeps, including SOTA, and
queues dependency-gated continuation submitters that keep submitting `--resume`
jobs until each run reaches the requested `target_size`.
