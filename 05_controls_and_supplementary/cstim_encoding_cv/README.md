# CSTIM Encoding Cross-Validation

This follow-up asks what happens if CSTIM responses are included in the
encoding fit, while CSTIM evaluation remains out-of-sample.

The primary revised analysis is target adaptation. The dense layer-sweep
`best_on_shared` layer is fixed upstream for each subject/model. For each
CSTIM target set, the encoding model is fit with:

- DeepVision unique responses
- only that 100-image CSTIM target set
- a tunable target-sample weight
- analytic leave-one-sample-out predictions for that target set

Vicco is handled in two ways:

- held-out Vicco predictions from each CSTIM-target-adapted model
- a separate DeepVision unique + Vicco target-adapted model with Vicco analytic
  LOSO predictions

This avoids the earlier pooled-500 design and lets each model set ask whether
including that set's own target samples changes the mixed-RSA scores.

## Completed Target-Adaptation Run

The completed run uses all 100 best-on-shared subject/model/layer selections
covering 20 models and 5 subjects. It produced 5,100 score rows:

- 2,050 CSTIM LOSO rows across model-set memberships and ten weights
- 2,050 held-out Vicco rows from the matching CSTIM-adapted models
- 1,000 Vicco LOSO rows from separate Vicco-adapted models

The final intended run is the fresh SRP5920 version:

- dense best-on-shared selected layers from the layer-sweep transfer table
- fresh `flatten_srp5920_v1` selected-layer feature caches generated under the
  same dense-chunk extractor context as the layer-sweep stream run
- per-voxel ridge alphas selected on DeepVision unique responses with 5 folds
- exact analytic weighted LOSO for each CSTIM target set and for Vicco
- prediction/scoring semantics matched to the dense layer-sweep stream scorer

Fresh selected-layer SRP5920 features were filled locally under:

- `cache_or_heavy/selected_layer_features_srp5920/features/`
- `cache_or_heavy/selected_layer_features_srp5920/dv_features/`

Per-voxel alpha caches are under:

- `cache_or_heavy/target_adaptation_srp5920/alphas/`

The source layer-sweep cache is only read, not modified. The final score rows
all have `feature_dim_analysis=5920`,
`feature_protocol=flatten_srp5920_v1`, and
`alpha_rule=per_voxel_deepvision_unique_ridgecv`.

## Method Details

The plotted target-adaptation analysis is tied to the Figure 2 dense mRSA
ground-truth path. The original/reference scores are loaded from:

- `../model_scope_followups/layer_sweep/results/mrsa_dense_layer_selection_transfer.csv`

Rows are restricted to `selection_rule=best_on_shared`,
`selection_model_set=deepvision_shared`, and the selected layer from that table
is fixed for every subject/model pair. These are the same mRSA values consumed
by `06_manuscript/figures/composites/fig2_cstim_effects/make_fig2_cstim_effects.py`.

For each fixed subject/model/layer selection, the analysis loads:

- DeepVision unique features from
  `cache_or_heavy/selected_layer_features_srp5920/dv_features/{subject}/{model}.npz`
- CSTIM and Vicco features from
  `cache_or_heavy/selected_layer_features_srp5920/features/{model}/{set}.npz`
- DeepVision unique responses from the cached benchmark response files under
  `01_brain_model_alignment/cache_or_heavy/deepvision_benchmark_cache/`
- CSTIM/Vicco averaged betas, stimulus metadata, and hlvis voxel masks from
  `01_brain_model_alignment/cache_or_heavy/cstim_brain_response_cache/data/{subject}/`

Feature extraction follows the dense layer-sweep stream convention:

- layer activations are flattened
- each layer is projected to 5,920 dimensions with deterministic sparse random
  projection
- the SRP seed is stable per `(model, layer)`, matching the dense layer-sweep
  stream pipeline
- selected-layer caches are extracted under the dense layer chunk that contained
  the selected layer in the stream run, rather than as isolated single-layer
  extractions

Per-voxel ridge alphas are selected on DeepVision unique only:

- responses are z-scored per voxel using the DeepVision unique response mean/std
- five random 50/50 train/test splits are generated with seed `42`
- the alpha grid is `logspace(log10(0.1), log10(1e7), 20)`
- `fit_voxelwise_ridgecv_fast(..., scale_features=True)` selects an alpha per
  voxel per fold
- the final per-voxel alpha is the median across folds

For scoring, the selected alphas are fixed. DeepVision unique is the base
training set. Each target condition is then handled as a weighted ridge update:

- CSTIM target adaptation: DeepVision unique plus one 100-image CSTIM model set
- CSTIM evaluation: analytic leave-one-sample-out predictions for that same
  100-image target set
- held-out Vicco evaluation: Vicco predictions from each CSTIM-adapted model
- Vicco baseline adaptation: DeepVision unique plus Vicco, scored with analytic
  Vicco LOSO

Predictions are evaluated with the same convention as the dense layer-sweep
stream scorer (`layer_sweep_stream_predict_v1`). Vicco scores use the same
1,000 no-replacement 100-image bootstrap samples as the layer-sweep source
table.

The target weights are `0, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 47`.
`target_weight=0` is the DeepVision-only baseline under the same
selected-layer/SRP/per-voxel-alpha target-adaptation code path. Mixed RSA is
computed as Spearman correlation between the upper triangles of the
predicted-response correlation-distance RDM and the matching brain-response
correlation-distance RDM.

The delta trajectory figure can also include the optional
`target_weight=4700` fast-update stress test from
`code/analysis/04_score_fast_update_extreme_weight_sensitivity.py`. That
weight makes the 100 CSTIM target samples dominate the weighted objective
while still using fixed DeepVision preprocessing and fixed DeepVision-selected
alphas. It is a diagnostic for near-target-only behavior, not part of the
canonical weight grid.

The score table carries the original Figure 2 source-table score in
`original_best_shared_mrsa`, and
`results/target_adaptation_weight0_repro_check.csv` summarizes the weight-0
reproduction check. In the current run, weight 0 closely reproduces the source
means but is not bitwise identical:

- CSTIM LOSO mean delta: `+0.000348`, max absolute delta: `0.021806`
- held-out Vicco mean delta: `+0.000220`, max absolute delta: `0.018698`

The largest deltas are concentrated in `robustness_imagenet_l2_eps3`. Treat
`original_best_shared_mrsa` as the canonical source reference when reporting
delta values.

Reusable constants, selection-table readers, atomic writers, z-scoring helpers,
and RSA/bootstrap helpers live in `src/cstims/target_adaptation.py`; scripts in
this folder keep only the analysis-specific orchestration.

## Full-Refit Sensitivity

`code/analysis/03_sensitivity_full_refit_target_adaptation.py` is an optional
stress test, not part of the canonical plotted analysis. It refits one CSTIM
target set with all 5,920 SRP features and all hlvis voxels after recomputing
weighted DeepVision+CSTIM feature/response normalization and selecting
per-voxel alphas with weighted analytical LOO RidgeCV.

The completed limited run used the SOTA target set at `target_weight=4`, the
weight with the largest mean SOTA CSTIM improvement in the canonical table. It
covered all 30 SOTA subject/model selections and wrote:

- `results/target_adaptation_full_refit_sensitivity_scores.csv`
- `results/target_adaptation_full_refit_sensitivity_summary.csv`
- `results/target_adaptation_full_refit_sensitivity_metadata.json`
- `results/target_adaptation_full_refit_sensitivity_by_model.csv`
- `figures/target_adaptation_full_refit_sensitivity_by_model.pdf`

For this sensitivity, mean SOTA CSTIM LOSO was `0.264331`, which is `-0.014841`
relative to the canonical fixed-DeepVision-stat weighted update at the same
weight. Held-out Vicco was lower in all 30 rows, with mean delta `-0.044816`
relative to the canonical score.

## CSTIM-Dominant Fast-Update Sensitivity

`code/analysis/04_score_fast_update_extreme_weight_sensitivity.py` is an
optional stress test that reuses the canonical Woodbury weighted update with
fixed DeepVision feature/response normalization and fixed DeepVision-selected
per-voxel alphas. The completed run used `target_weight=4700`, where the 100
CSTIM target samples have total weight about 100 times larger than the
DeepVision unique training set.

It writes:

- `results/target_adaptation_fast_extreme_weight_scores.csv`
- `results/target_adaptation_fast_extreme_weight_summary.csv`
- `results/target_adaptation_fast_extreme_weight_metadata.json`

The result is deliberately not folded into
`results/target_adaptation_weighted_scores.csv`. It is appended only while
making `figures/target_adaptation_weight_delta_trajectory_cached.pdf`, where it
appears as the separated `4.7k` diagnostic point. In the completed run, the
near-CSTIM-only weighting was substantially worse than the original reference:
SOTA mean CSTIM LOSO delta was `-0.082167`, and SOTA held-out baseline delta
was `-0.152210`.

## VGG Cache Repair

On 2026-06-10, VGG-16 rows were repaired after finding that the selected-layer
DeepVision feature cache had been extracted with a different layer context than
the target/Vicco cache.

The relevant VGG classifier block is:

```text
classifier.3 = Linear(...)
classifier.4 = ReLU(inplace=True)
classifier.5 = Dropout(...)
```

When `classifier.3` is extracted together with later classifier nodes, the
returned intermediate tensor is affected by the in-place ReLU. When
`classifier.3` is extracted alone, it remains pre-ReLU. The target/Vicco cache
used multi-layer extraction, while the old DeepVision selected-layer cache used
single-layer extraction for each selected layer. That mismatch made VGG train
on pre-ReLU DeepVision features but score post-ReLU target features, causing
the near-zero VGG CSTIM LOSO result.

`code/analysis/01_cache_selected_layer_srp5920_features.py` was patched so the
DeepVision side is extracted with the same multi-layer model context as the
target side. After that patch:

- VGG selected-layer SRP5920 features were overwritten
- VGG per-voxel alpha caches were overwritten
- 175 VGG score rows were recomputed and merged back into
  `results/target_adaptation_weighted_scores.csv`
- the pre-repair full score table was saved as
  `archive/vgg_cache_repair_20260610/results/target_adaptation_weighted_scores.csv.before_vgg_cache_repair`
- repair provenance was added to
  `results/target_adaptation_run_metadata.json`

The repaired VGG all-model weight-0 CSTIM LOSO mean is on the original Fig. 2
scale again, rather than collapsing near zero.

## Run

Use the project conda environment:

```bash
LD_LIBRARY_PATH=/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-} \
  /data/home_roth/miniforge3/bin/python code/analysis/01_cache_selected_layer_srp5920_features.py \
  --batch-size 4 --device cuda:0 --layers-per-chunk 32 --progress-every 1024

LD_LIBRARY_PATH=/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-} \
  /data/home_roth/miniforge3/bin/python code/analysis/02_score_target_adaptation_srp5920_per_voxel_alpha.py \
  --weights 0,0.25,0.5,1,2,4,8,16,32,47 --n-vicco-boot 1000 --overwrite --overwrite-alpha

LD_LIBRARY_PATH=/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-} \
  /data/home_roth/miniforge3/bin/python code/figures/01_plot_target_weight_trajectories.py

LD_LIBRARY_PATH=/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-} \
  /data/home_roth/miniforge3/bin/python code/figures/02_plot_weight0_to_best_cstim_grid.py
```

Optional full-refit sensitivity:

```bash
LD_LIBRARY_PATH=/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-} \
  /data/home_roth/miniforge3/bin/python code/analysis/03_sensitivity_full_refit_target_adaptation.py \
  --model-set sota --weight 4 --n-vicco-boot 1000 --overwrite

LD_LIBRARY_PATH=/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-} \
  /data/home_roth/miniforge3/bin/python code/figures/05_plot_full_refit_sensitivity.py
```

Optional CSTIM-dominant fast-update sensitivity:

```bash
LD_LIBRARY_PATH=/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-} \
  /data/home_roth/miniforge3/bin/python code/analysis/04_score_fast_update_extreme_weight_sensitivity.py \
  --weights 4700 --n-vicco-boot 1000 --overwrite

LD_LIBRARY_PATH=/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-} \
  /data/home_roth/miniforge3/bin/python code/figures/01_plot_target_weight_trajectories.py
```

If the scorer is interrupted after a checkpoint, resume with:

```bash
LD_LIBRARY_PATH=/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-} \
  /data/home_roth/miniforge3/bin/python code/analysis/02_score_target_adaptation_srp5920_per_voxel_alpha.py \
  --weights 0,0.25,0.5,1,2,4,8,16,32,47 --n-vicco-boot 1000 --resume
```

Main outputs:

- `results/target_adaptation_weighted_scores.csv`
- `results/target_adaptation_weighted_summary.csv`
- `results/target_adaptation_cached_selection_audit.csv`
- `results/target_adaptation_weight0_repro_check.csv`
- `results/target_adaptation_weight0_to_best_cstim_points.csv`
- `results/target_adaptation_weight0_to_best_cstim_summary.csv`
- `results/target_adaptation_full_refit_sensitivity_scores.csv` (optional)
- `results/target_adaptation_full_refit_sensitivity_summary.csv` (optional)
- `results/target_adaptation_full_refit_sensitivity_metadata.json` (optional)
- `results/target_adaptation_full_refit_sensitivity_by_model.csv` (optional)
- `results/target_adaptation_fast_extreme_weight_scores.csv` (optional)
- `results/target_adaptation_fast_extreme_weight_summary.csv` (optional)
- `results/target_adaptation_fast_extreme_weight_metadata.json` (optional)
- `figures/target_adaptation_weight_trajectory_cached.pdf`
- `figures/target_adaptation_weight_delta_trajectory_cached.pdf`
- `figures/target_adaptation_weight0_to_best_cstim_grid_cached.pdf`
- `figures/target_adaptation_full_refit_sensitivity_by_model.pdf` (optional)

Optional diagnostic figure scripts:

- `code/figures/03_plot_fixed_weight_brain_alignment_grid.py`
- `code/figures/04_plot_fixed_weight_cstim_vicco_scatter.py`
- `code/figures/05_plot_full_refit_sensitivity.py`

Archived historical outputs:

- `archive/legacy_pooled_512d/` contains the earlier pooled-500 exploratory
  run and obsolete 512D target-adaptation prototype
- `archive/vgg_cache_repair_20260610/` contains the pre-repair VGG score table
  backup and temporary repaired-row table
