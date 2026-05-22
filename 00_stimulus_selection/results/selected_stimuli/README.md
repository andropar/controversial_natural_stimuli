# Selected Stimuli

This folder contains the five frozen controversial-stimulus sets used by the
downstream paper analyses:

- `all_models`
- `architecture`
- `dataset`
- `sota`
- `training_objective`

Each set contains three files.

## `selected_stimuli_data.pkl`

Main selected-stimulus payload. This is the file downstream code should use
when it needs the final selected image identities or their selection-time model
features.

It is a Python pickle written by
`../code/select_controversial_stimuli.py`. The payload is a dictionary with
these main fields:

- `selected_global_indices`: final selected global image indices.
- `selected_image_records`: image metadata for the final selected indices,
  including shard/tar references and image names.
- `model_names`: model roster used for the selection run.
- `config`: resolved selection configuration.
- `track_definitions`: raw/encoding tracks used by the selection objective.
- `track_aggregation`: how per-track scores were normalized/combined.
- `var_noise_by_model`: calibrated noise values by track/model.
- `scores`: combined selection objective history.
- `selected_features_raw`: raw model features for the final selected stimuli.
- `selected_features_by_encoding`: subject-encoding-transformed model features
  for the final selected stimuli, when applicable.
- `greedy_indices`: selected indices at the end of greedy selection before
  refinement.
- `greedy_image_records`: image metadata for `greedy_indices`.
- `greedy_features_raw` and `greedy_features_by_encoding`: feature payloads for
  the greedy selection.
- `best_raw_combined_indices`: best selection encountered under the raw-combined
  score during refinement.
- `best_raw_combined_image_records`: image metadata for
  `best_raw_combined_indices`.
- `best_raw_combined_features_raw` and
  `best_raw_combined_features_by_encoding`: feature payloads for that variant.
- `best_raw_combined_score` and `best_raw_combined_pass`: score and refinement
  pass for the best raw-combined variant.
- `refinement_history`: replacement records from post-greedy refinement.
- `filter_records`: natural-image filter records for candidate images.
- `multi_view`, `encoding_multi`, `scores_per_view_history`, and
  `scores_per_rep_history`: compatibility fields for earlier/multi-track
  selection modes.

The pickle references Python objects from the original selection environment
such as OmegaConf containers and NumPy arrays. It is safest to load it from an
environment with the copied project dependencies available.

## `checkpoint.pkl`

Minimal resume checkpoint for the optimizer. It is mainly useful for provenance
or for resuming/reconstructing a run, not for ordinary downstream analysis.

It serializes a `SelectionCheckpoint` object with:

- `phase`: final optimizer phase, typically `complete`.
- `greedy_iteration`: number of completed greedy iterations.
- `refinement_pass` and `refinement_position`: refinement progress.
- `current_indices`: current/final selected global indices.
- `failed_indices`: candidate indices rejected or failed during selection.
- `scores_combined`: combined score history.
- `scores_per_track_history`: per-track score history.
- `refinement_history`: refinement replacement records.
- `var_noise_raw`: calibrated raw-track noise.
- `var_noise_by_encoding`: calibrated encoding-track noise.

## `select_controversial_stimuli.log`

Text log from the frozen run. It records the resolved configuration, model set,
track definitions, output path, selection progress, filtering/refinement
messages, and final save events.

## Provenance

Readable run provenance is in `../manifests/selection_runs.csv`.

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Contents Snapshot

- Folder: `00_stimulus_selection/results/selected_stimuli`
- Figures in this folder tree: 0
- Data/table-like files in this folder tree: 150
- Python scripts in this folder tree: 0

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.
<!-- END AUTO-FIGURE-PROVENANCE -->
