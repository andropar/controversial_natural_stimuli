# 00 Stimulus Selection

This stage contains the frozen controversial-stimulus selections and the
decision checks used to evaluate the selection procedure.

The practical purpose of this folder is to answer: which images were selected,
which model pool and filters produced them, and what checks show that the
selection is doing what it is supposed to do before any brain-alignment analysis
uses the images.

- `code/`: selection entrypoint plus copied selection/evaluation dependencies.
- `resources/`: Hydra configs, model-set definitions, and selection resources.
- `results/selected_stimuli/`: final frozen selected sets.
- `selection_evaluation/`: selection evaluation, diagnostics, and known-bad or
  superseded checks.
- `resources/manifests/selection_runs.csv`: readable provenance for each
  selected set.

## Selected Stimuli

Each folder under `results/selected_stimuli/` contains:

- `selected_stimuli_data.pkl`: main downstream payload with selected global
  image indices, image records, resolved config, model names, raw/encoding
  features, selection scores, refinement history, and filter records.
- `checkpoint.pkl`: optimizer resume/provenance checkpoint.
- `select_controversial_stimuli.log`: text log for the frozen run.

See `results/selected_stimuli/README.md` for the detailed pickle field
description.

## Decision Checks

`selection_evaluation/` contains the sanity checks used to
evaluate the final selection procedure: discriminability, diversity, model-RDM
correlation matrices, natural-image filter records, greedy/refinement score
traces, noise calibration, objective composition, the selection-method
leaderboard, and related figures.

See `selection_evaluation/README.md` for the file-by-file description.

## How To Read This Stage

Start with `results/selected_stimuli/` if you need the actual frozen image sets. Use
`resources/manifests/selection_runs.csv` to identify which run produced a set,
then inspect `resources/` for the selection configuration and `selection_evaluation/`
for the evidence that the chosen set has the intended disagreement structure.

## Selection Method

The selector searches for natural images that make model representational
geometry disagree. For a candidate image, it asks how much the image would
increase model separability if added to the selected set.

The core steps are:

1. Build each model's RDM from pairwise image distances, usually cosine
   distance.
2. Calibrate expected brain-measurement noise to the configured noise ceiling
   (`noise_ceiling_target: 0.46` in `resources/configs/config.yaml`).
3. Estimate the expected RSA correlation matrix under that noise.
4. Score each model by how well it agrees with itself relative to the other
   models.
5. Aggregate the model scores, then aggregate across configured tracks.

The default method used for the main frozen selections is
`raw_plus_all_encodings`: five subject-specific encoding tracks
(`sub-01`, `sub-03`, `sub-05`, `sub-06`, `sub-07`) plus one raw feature track.
The configured weighting gives `0.5` total weight to raw model features and
shares the remaining `0.5` across the encoding tracks.

Relevant implementation files:

- `src/cstims/selection/primitives.py`: pairwise distances, correlations, and
  per-model utility calculations.
- `src/cstims/selection/utility.py`: batch utility calculation and analytical
  scoring.
- `src/cstims/selection/selector.py`: initialization, greedy selection, track
  aggregation, and refinement.
- `src/cstims/noise_estimation.py` and
  `src/cstims/evaluation/model_discrimination.py`: noise calibration and
  model-discrimination helpers.

The main selection entrypoint copied into this stage is
`code/select_controversial_stimuli.py`. Configs live in `resources/configs/`,
including model sets such as `sota`, `all_models`, `architecture`,
`training_objective`, `dataset`, and `small`.

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Contents Snapshot

- Folder: `00_stimulus_selection`
- Figures in this folder tree: 115
- Data/table-like files in this folder tree: 281
- Python scripts in this folder tree: 51
- Main child folders: `code/`, `resources/`, `results/selected_stimuli/`, `selection_evaluation/`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Index

| Figure directory | Figures | README |
|---|---:|---|
| `00_stimulus_selection/selection_evaluation/figures/correlation_matrices_eval_pipeline` | 10 | `00_stimulus_selection/selection_evaluation/figures/correlation_matrices_eval_pipeline/README.md` |
| `00_stimulus_selection/selection_evaluation/figures/correlation_matrices_eval_pipeline/png` | 10 | `00_stimulus_selection/selection_evaluation/figures/correlation_matrices_eval_pipeline/png/README.md` |
| `00_stimulus_selection/selection_evaluation/figures/correlation_matrices_pool` | 20 | `00_stimulus_selection/selection_evaluation/figures/correlation_matrices_pool/README.md` |
| `00_stimulus_selection/selection_evaluation/figures/correlation_matrices_pool/png` | 20 | `00_stimulus_selection/selection_evaluation/figures/correlation_matrices_pool/png/README.md` |
| `00_stimulus_selection/selection_evaluation/figures/correlation_matrices_vicco` | 20 | `00_stimulus_selection/selection_evaluation/figures/correlation_matrices_vicco/README.md` |
| `00_stimulus_selection/selection_evaluation/figures/correlation_matrices_vicco/png` | 20 | `00_stimulus_selection/selection_evaluation/figures/correlation_matrices_vicco/png/README.md` |
| `00_stimulus_selection/selection_evaluation/figures/insilico_curve` | 7 | `00_stimulus_selection/selection_evaluation/figures/insilico_curve/README.md` |
| `00_stimulus_selection/selection_evaluation/figures/insilico_curve/png` | 7 | `00_stimulus_selection/selection_evaluation/figures/insilico_curve/png/README.md` |
| `00_stimulus_selection/selection_evaluation/method_comparison/figures` | 1 | `00_stimulus_selection/selection_evaluation/method_comparison/figures/README.md` |
<!-- END AUTO-FIGURE-PROVENANCE -->
