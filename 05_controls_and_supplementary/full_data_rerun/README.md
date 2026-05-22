# Full Data Rerun

Supplementary rerun of the main brain-alignment analysis using the fully
preprocessed LAION-fMRI release in:

```text
/data/home_roth/datasets/LAION-fMRI
```

The published tree currently has the public LAION-fMRI sessions. The private
cstims sessions still need to be synced from AWS into the same BIDS-style
`derivatives/glmsingle-tedana/{subject}/ses-*/func` layout before this rerun can
produce the controversial-stimulus brain cache.

## Layout

```text
code/
  sync_private_cstims_from_aws.sh
  prepare_cstim_brain_cache_from_laion.py
  compute_paper_layer_crsa_by_roi.py
  fit_paper_layer_encodings_full_data.py
  compute_paper_layer_mrsa_by_roi.py
  compute_noise_ceilings_by_roi.py
  plot_brain_alignment_improved_with_shared.py
  plot_roi_spread_alignment_drop_summary.py
  run_full_data_rerun_pipeline.sh
  build_best_layer_sofar_from_layer_sweep.py
results/
  brain_data_cache/        # generated; same contract as 01 brain inputs
  paper_layer_crsa_by_roi.csv
  paper_layer_crsa_by_roi_summary.csv
  paper_layer_mrsa_by_roi.csv
  paper_layer_mrsa_by_roi_summary.csv
  rdm_noise_ceilings_by_roi.csv
  between_subject_noise_ceilings_by_roi.csv
  roi_spread_alignment_drop_summary.csv
  best_shared_layer_sofar_from_layer_sweep.csv
figures/
  brain_alignment_improved_with_shared.pdf
  roi_spread_alignment_drop_summary.pdf
  roi_spread_alignment_drop_by_model_set/
  brain_alignment_by_roi/
logs/
```

## Workflow

1. Sync private cstim sessions from AWS, once the bucket/prefix is known:

   ```bash
   AWS_PROFILE=<profile> \
   CSTIMS_AWS_URI=s3://<bucket>/<prefix> \
   bash 05_controls_and_supplementary/full_data_rerun/code/sync_private_cstims_from_aws.sh
   ```

2. Build the cstim brain cache from the fully preprocessed GLMsingle-tedana
   derivatives:

   ```bash
   python 05_controls_and_supplementary/full_data_rerun/code/prepare_cstim_brain_cache_from_laion.py
   ```

3. Run the RSA scoring against `results/brain_data_cache`. The scoring scripts can
   be copied/adapted from `01_brain_model_alignment/code/rsa_scoring` once the
   brain cache exists and the layer-sweep best-layer encoding run has finished.
   The downstream score table should include a `roi` column and loop over the
   named masks saved in `voxel_metadata.npz`.

The resumable end-to-end runner is:

```bash
screen -dmS full_data_rerun bash -lc \
  'cd /data/home_roth/cstims_share && bash 05_controls_and_supplementary/full_data_rerun/code/run_full_data_rerun_pipeline.sh >> 05_controls_and_supplementary/full_data_rerun/logs/pipeline_screen.log 2>&1'
```

Progress is written to:

```text
logs/pipeline_status.log
logs/pipeline_screen.log
```

## Completed Runs

- Synced private cstim `ses-32`, `ses-33`, and `ses-34` SingletrialBetas files
  from `s3://laion-fmri/derivatives/glmsingle-tedana`.
- Built full-data cstim brain caches for:

  ```text
  sub-01, sub-03, sub-05, sub-06, sub-07
  ```

- Ran paper-layer direct/classical RSA (`cRSA`) by ROI using cached paper-layer
  model features:

  ```text
  results/paper_layer_crsa_by_roi.csv
  results/paper_layer_crsa_by_roi_summary.csv
  ```

  This output has 2,257,255 rows: five subjects, 20 models, five model sets,
  11 ROI groupings, controversial-stimulus rows, and 1000 Vicco bootstrap rows.

- Refit full-data paper-layer encoding models on DeepVision unique images and
  ran paper-layer mixed RSA (`mRSA`) by ROI:

  ```text
  results/paper_layer_mrsa_by_roi.csv
  results/paper_layer_mrsa_by_roi_summary.csv
  ```

- Computed ROI-specific full-data noise ceilings for cstim and Vicco:

  ```text
  results/rdm_noise_ceilings_by_roi.csv
  results/between_subject_noise_ceilings_by_roi.csv
  ```

  The main and per-ROI brain-alignment plots are noise-ceiling normalized:

  ```text
  figures/brain_alignment_improved_with_shared.pdf
  figures/brain_alignment_by_roi/
  ```

  A compact ROI comparison of model spread and median alignment drop is saved
  as:

  ```text
  results/roi_spread_alignment_drop_summary.csv
  figures/roi_spread_alignment_drop_summary.pdf        # all_models bar plot
  figures/roi_spread_alignment_drop_by_model_set/      # one bar plot per model set
  ```

- Snapshotted the best shared-selected layer available so far from the ongoing
  dense layer-sweep stream parts:

  ```text
  results/best_shared_layer_sofar_from_layer_sweep.csv
  ```

  At snapshot time this covered `sub-01` and 18 models, because those were the
  stream parts present on disk.

## mRSA Note

The fully preprocessed cstim cache is in a different voxel grid/mask than the
original paper cache. For example, `sub-01` has 271,258 full-data brain voxels
instead of 257,594 in the original cache. That means the original paper encoding
weights cannot be safely applied to the new cstim betas.

Full-data mRSA/wRSA therefore uses encoding models refit in this same full-data
voxel space before running ROI-wise mRSA transfer.

## Current Assumptions

- Cstim sessions are `ses-32`, `ses-33`, and `ses-34`, matching the original
  paper helper config.
- Trial labels in `*_desc-SingletrialBetas_trials.tsv` contain the stimulus
  image names or labels used by the original cstim label parser.
- ROI masks are saved in brain-space vector form with names prefixed by `roi_`.
  The currently scored per-ROI analyses are:

  ```text
  EVC
  ventral
  lateral
  dorsal
  general
  EBA
  FFA
  PPA
  LOTC
  floc_all
  ventral_lateral_floc
  ```

- The compatibility `hlvis_mask` is set to `roi_ventral_lateral_floc`.
  The broader LAION masks are:

  ```text
  (Noiseceiling4rep > 0.2) & requested ROI mask
  ```

  where the LAION masks are the T1w `laion*` masks under
  `derivatives/rois/{subject}/laion`, and the fLOC masks are the corresponding
  T1w ROI masks under `derivatives/rois/{subject}`.

These assumptions are isolated in the cache builder so they can be changed
without touching the original analysis.
