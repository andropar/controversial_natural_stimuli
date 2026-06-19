# CSTIM Encoding Cross-Validation

This follow-up asks what happens if CSTIM responses are included in the
encoding fit while CSTIM evaluation remains out-of-sample.

The active layout is intentionally script-based:

- `code/analysis/01_cache_selected_layer_srp5920_features.py`
- `code/analysis/02_score_target_adaptation_fixed_alpha.py`
- `code/analysis/03_score_target_adaptation_refit_alpha.py`
- `code/figures/02_fixed_alpha_plot_brain_alignment_grid.py`
- `code/figures/02_fixed_alpha_plot_cstim_vicco_scatter.py`
- `code/figures/03_refit_alpha_plot_weight_trajectories.py`
- `code/figures/03_refit_alpha_plot_weight0_to_best_cstim_grid.py`

Historical one-off scripts, old logs, and old flat outputs were moved to:

- `archive/cleanup_20260618_layout/`

The moved result files preserve the previous run outputs. Rerun the active
scripts after the corrected layer sweep and layer-selection tables are refreshed
if new corrected values are needed.

## Method

The analysis fixes the dense layer-sweep `best_on_shared` layer for each
subject/model pair, then uses selected-layer features reduced with
`flatten_srp5920_v1`:

- layer activations are flattened
- each selected layer is projected to 5,920 dimensions with deterministic sparse
  random projection
- the SRP seed is stable per `(model, layer)`
- features are extracted in the dense chunk context used by the layer sweep

The fixed-alpha scorer selects per-voxel ridge alphas on DeepVision unique, then
uses exact analytic weighted LOSO updates for each target set. The refit-alpha
scorer recomputes weighted feature/response normalization and reselects alphas
for every requested subject/model/target-set/weight block.

Mixed RSA is Spearman correlation between the upper triangles of the predicted
response correlation-distance RDM and the matching brain-response
correlation-distance RDM.

## Outputs

Cache script outputs:

- `results/01_cache_srp_features/selection_audit.csv`

Fixed-alpha scorer outputs:

- `results/02_fixed_alpha/scores.csv`
- `results/02_fixed_alpha/summary.csv`
- `results/02_fixed_alpha/metadata.json`
- `results/02_fixed_alpha/cache_and_alpha_audit.csv`
- `results/02_fixed_alpha/figure_data/`

Refit-alpha scorer outputs:

- `results/03_refit_alpha/scores.csv`
- `results/03_refit_alpha/summary.csv`
- `results/03_refit_alpha/metadata.json`
- `results/03_refit_alpha/w4700/`
- `results/03_refit_alpha/plus4700/`
- `results/03_refit_alpha/by_model/all_weights/`
- `results/03_refit_alpha/by_model/w4700/`
- `results/03_refit_alpha/figure_data/`

Figures are grouped the same way:

- `figures/02_fixed_alpha/`
- `figures/02_fixed_alpha/png/`
- `figures/03_refit_alpha/`
- `figures/03_refit_alpha/png/`

Feature caches live under:

- `cache_or_heavy/selected_layer_features_srp5920/features/`
- `cache_or_heavy/selected_layer_features_srp5920/dv_features/`
- `cache_or_heavy/target_adaptation_srp5920/alphas/`

## Run

Use the project conda environment:

```bash
export LD_LIBRARY_PATH=/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/data/home_roth/cstims_share/src:${PYTHONPATH:-}
```

Cache selected-layer SRP5920 features:

```bash
/data/home_roth/miniforge3/bin/python \
  code/analysis/01_cache_selected_layer_srp5920_features.py \
  --batch-size 4 --device cuda:0 --layers-per-chunk 32 --progress-every 1024
```

Write only the current cache audit:

```bash
/data/home_roth/miniforge3/bin/python \
  code/analysis/01_cache_selected_layer_srp5920_features.py \
  --audit-only
```

Run fixed-alpha target adaptation:

```bash
/data/home_roth/miniforge3/bin/python \
  code/analysis/02_score_target_adaptation_fixed_alpha.py \
  --weights 0,0.25,0.5,1,2,4,8,16,32,47 \
  --n-vicco-boot 1000 --overwrite --overwrite-alpha
```

Resume fixed-alpha target adaptation:

```bash
/data/home_roth/miniforge3/bin/python \
  code/analysis/02_score_target_adaptation_fixed_alpha.py \
  --weights 0,0.25,0.5,1,2,4,8,16,32,47 \
  --n-vicco-boot 1000 --resume
```

Run refit-alpha target adaptation:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/data/home_roth/miniforge3/bin/python \
  code/analysis/03_score_target_adaptation_refit_alpha.py \
  --weights 0,0.25,0.5,1,2,4,8,16,32,47 \
  --n-vicco-boot 1000 --resume
```

The refit-alpha script routes legacy output stems into organized directories:

- default or `target_adaptation_full_refit_all_weights` -> `results/03_refit_alpha/`
- `target_adaptation_full_refit_w4700` -> `results/03_refit_alpha/w4700/`
- `target_adaptation_full_refit_all_weights_plus4700` -> `results/03_refit_alpha/plus4700/`
- `target_adaptation_full_refit_all_weights_by_model_*` -> `results/03_refit_alpha/by_model/all_weights/`
- `target_adaptation_full_refit_w4700_by_model_*` -> `results/03_refit_alpha/by_model/w4700/`

Plot refit-alpha figures:

```bash
/data/home_roth/miniforge3/bin/python \
  code/figures/03_refit_alpha_plot_weight_trajectories.py

/data/home_roth/miniforge3/bin/python \
  code/figures/03_refit_alpha_plot_weight0_to_best_cstim_grid.py
```

Plot fixed-alpha diagnostic figures:

```bash
/data/home_roth/miniforge3/bin/python \
  code/figures/02_fixed_alpha_plot_brain_alignment_grid.py --weight 2

/data/home_roth/miniforge3/bin/python \
  code/figures/02_fixed_alpha_plot_cstim_vicco_scatter.py \
  --weight 2 --model-set all_models
```
