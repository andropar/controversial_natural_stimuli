# RSA Scoring Code

This folder computes the brain-alignment score tables for the controversial
stimuli and baseline images.

## Main Scripts

- `01_compute_crsa.py`: computes classical/fixed RSA (`crsa`). It extracts model
  features for each stimulus group, builds model RDMs and brain RDMs, then
  writes `../../results/rsa_scores/<subject>/crsa_scores.csv`.
- `02_compute_wrsa_transfer.py`: computes transferred weighted/mixed RSA
  (`wrsa_transfer`). It applies DeepVision-trained encoding models to the cstim
  images, builds predicted-brain RDMs, compares them to measured brain RDMs,
  then writes `../../results/rsa_scores/<subject>/wrsa_transfer_scores.csv`.
- `03_compute_cross_set_wrsa.py`: evaluates cross-set weighted RSA, producing
  `../../results/rsa_scores/<subject>/cross_set_wrsa_scores.csv`.
- `04_compute_rsa_large_benchmark.py`: computes alignment scores for the larger
  benchmark model set, producing
  `../../results/rsa_scores/rsa_large_benchmark_scores.csv`.
- `05_compute_rsa_deepvision_shared.py`: computes the shared DeepVision
  comparison scores.

The `figures/` subfolder contains plotting scripts that read the staged score
tables and write rendered figures under `../../figures/rsa_scores/`.

## Inputs And Outputs

Inputs:

- Brain caches from `../brain_data_preparation/`, accessed through the stage
  path helpers as `resources/` or copied heavy cache locations.
- Cstim and Vicco image payloads from the external HDF5/image root documented in
  `../../../external_data/README.md`.
- Model metadata from `../../../00_stimulus_selection/resources/`.
- Fitted encoding models from `../../results/encoding_models/` for
  `wrsa_transfer`.

Outputs:

- `../../results/rsa_scores/sub-*/crsa_scores.csv`
- `../../results/rsa_scores/sub-*/wrsa_transfer_scores.csv`
- `../../results/rsa_scores/sub-*/cross_set_wrsa_scores.csv`
- `../../results/rsa_scores/rsa_large_benchmark_scores.csv`
- `../../results/rsa_scores/benchmark_upper_tail_deltas*.csv`

The per-subject cstim rows are marked with
`stimulus_type == "controversial"`. Baseline rows use
`stimulus_type == "vicco"` with bootstrap samples.
