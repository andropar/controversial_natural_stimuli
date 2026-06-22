# CSTIM Encoding Cross-Validation

This follow-up asks what happens if CSTIM responses are included in the
encoding fit while CSTIM evaluation remains out-of-sample.

The most recent completed pre-fix Raven run was `20260620_173400_png_full`,
with the fixed/refit retry jobs completed on 2026-06-21:

- fixed-alpha array: `28219232` (`cstim_cv_fixed_png_retry`)
- fixed-alpha collect: `28219233` (`cstim_cv_fixed_collect_retry`)
- refit-alpha array: `28219234` (`cstim_cv_refit_png_retry`)
- refit-alpha collect/figures: `28219235` (`cstim_cv_refit_collect_retry`)

On 2026-06-22, diagnostic job `28226974` showed that the pre-fix
`target_weight=0` rows used an extra evaluation-feature transform and therefore
did not exactly reproduce the dense layer-sweep score. Treat the pre-fix cstim
encoding CV score tables as superseded until the corrected rerun is collected.
The corrected rerun writes the same numbered output directories below.

Root-level files named `results/target_adaptation_*` and root-level
`figures/target_adaptation_*` are older June 10-17 artifacts and are kept only
for provenance.

## Canonical Result Paths

Canonical fixed-alpha table:

- `results/02_fixed_alpha/scores.csv`
- `results/02_fixed_alpha/summary.csv`
- `results/02_fixed_alpha/cache_and_alpha_audit.csv`
- `results/02_fixed_alpha/weight0_repro_check.csv`
- `results/02_fixed_alpha/metadata.json`

Canonical refit-alpha table:

- `results/03_refit_alpha/plus4700/scores.csv`
- `results/03_refit_alpha/plus4700/summary.csv`
- `results/03_refit_alpha/plus4700/weight0_vs_fixed_check.csv`
- `results/03_refit_alpha/plus4700/metadata.json`

Current alignment/spread summaries:

- `results/04_alignment_spread/layer_depth_curve_summary_from_dense.csv`
- `results/04_alignment_spread/layer_depth_spread_summary.csv`
- `results/04_alignment_spread/refit_weight_spread_summary.csv`

Current fixed-alpha figures:

- `figures/02_fixed_alpha/`
- `figures/02_fixed_alpha/png/`

Current refit-alpha figures:

- `figures/03_refit_alpha/`
- `figures/03_refit_alpha/png/`

Current alignment/spread figures:

- `figures/04_alignment_spread/target_adaptation_layer_depth_spread_median_cached.pdf`
- `figures/04_alignment_spread/target_adaptation_refit_weight_spread_median_cached.pdf`
- PNG copies in `figures/04_alignment_spread/png/`

Current visual index:

- `figures/cstim_encoding_cv_new_retry_gallery.html`
- `figures/cstim_encoding_cv_new_retry_contact_sheet.png`

## Result Coverage

The corrected score tables should have the same coverage as the pre-fix tables:
6,120 data rows each, with:

- 5 subjects: `sub-01`, `sub-03`, `sub-05`, `sub-06`, `sub-07`
- 20 paper models
- adaptation targets: `all_models`, `architecture`, `dataset`, `sota`,
  `training_objective`, `vicco`
- evaluation targets: `cstim_loso`, `vicco_heldout`, `vicco_loso`
- target/cstim weights: `0`, `0.25`, `0.5`, `1`, `2`, `4`, `8`, `16`, `32`,
  `47`, `4700`, `inf`

The `inf` rows are target-only fits. The `4700` rows are the large-weight
target-adaptation condition added after the first run.

The layer-depth alignment/spread table
`results/04_alignment_spread/layer_depth_curve_summary_from_dense.csv` has
19,880 data rows and is rebuilt from the current dense layer sweep table at
`../model_scope_followups/layer_sweep/results/mrsa_dense_all_eval_layer_scores.csv`.
This is the table used for the current dense-layer median/spread plots.

## Method

The analysis fixes the dense layer-sweep `best_on_shared` layer for each
subject/model pair, using:

- `../model_scope_followups/layer_sweep/results/mrsa_dense_layer_selection_transfer.csv`
- selected-layer features reduced with `flatten_srp5920_v1`
- deterministic sparse random projection to 5,920 dimensions
- a stable SRP seed per `(model, layer)`

The fixed-alpha scorer selects per-voxel ridge alphas on DeepVision unique and
then applies analytic weighted leave-one-subject-out updates for each target
set. The refit-alpha scorer recomputes weighted feature/response normalization
and reselects alphas for every requested subject/model/target/weight block.

Mixed RSA is Spearman correlation between the upper triangles of the predicted
response correlation-distance RDM and the matching brain-response
correlation-distance RDM.

The current fixed-alpha script uses `cstims.paths.brain_data_dir()` for the
brain cache root. On Raven this resolved to:

```text
/raven/ptmp/rothj/cstims/experiments/cstim_paper/01_brain_data/data
```

That cache-root fix is required for the current retry results. The current
shared target-adaptation helper also keeps evaluation features in the
DeepVision-standardized dual feature space; this makes `target_weight=0`
match the original dense layer-sweep prediction convention.

## How Current Results Were Created

The run provenance is in:

```text
logs/raven_cstim_encoding_cv_20260620_173400_png_full/
```

Generated Slurm scripts from that run are the exact replay record:

- `cache_features_worker.slurm.sh`
- `fixed_alpha_array.slurm.sh`
- `fixed_alpha_collect.slurm.sh`
- `refit_alpha_array.slurm.sh`
- `refit_alpha_collect_and_figures.slurm.sh`
- `score_subject_model_manifest.tsv`

The fixed-alpha array ran one subject/model shard at a time with:

```bash
/u/rothj/conda-envs/deepjuice/bin/python \
  code/analysis/02_score_target_adaptation_fixed_alpha.py \
  --subject <subject> \
  --models <model> \
  --weights 0,0.25,0.5,1,2,4,8,16,32,47,4700,inf \
  --n-vicco-boot 1000 \
  --output-dir results/02_fixed_alpha/by_subject_model_png_full_20260620_173400/<subject>/<model> \
  --overwrite \
  --overwrite-alpha
```

`fixed_alpha_collect.slurm.sh` concatenated those shards into
`results/02_fixed_alpha/scores.csv`, wrote the summary/audit/check files, and
ran:

```bash
python code/figures/02_fixed_alpha_plot_brain_alignment_grid.py --weight 2
python code/figures/02_fixed_alpha_plot_cstim_vicco_scatter.py --weight 2 --model-set all_models
python code/figures/02_fixed_alpha_plot_cstim_vicco_scatter.py --weight 2 --model-set sota
python code/figures/02_fixed_alpha_plot_cstim_vicco_scatter.py --weight 2 --model-set training_objective
python code/figures/02_fixed_alpha_plot_cstim_vicco_scatter.py --weight 2 --model-set architecture
python code/figures/02_fixed_alpha_plot_cstim_vicco_scatter.py --weight 2 --model-set dataset
```

The refit-alpha array ran one subject/model shard at a time with:

```bash
/u/rothj/conda-envs/deepjuice/bin/python \
  code/analysis/03_score_target_adaptation_refit_alpha.py \
  --subject <subject> \
  --models <model> \
  --weights 0,0.25,0.5,1,2,4,8,16,32,47,4700,inf \
  --n-vicco-boot 1000 \
  --output-stem target_adaptation_full_refit_all_weights_by_model_png_full_20260620_173400_<subject>_<model> \
  --canonical-score-csv results/02_fixed_alpha/scores.csv \
  --overwrite
```

`refit_alpha_collect_and_figures.slurm.sh` concatenated those shards into
`results/03_refit_alpha/plus4700/scores.csv`, wrote the summary/check files,
and ran:

```bash
python code/figures/03_refit_alpha_plot_weight_trajectories.py
python code/figures/03_refit_alpha_plot_weight0_to_best_cstim_grid.py
```

The current spread/median plots were generated afterward with:

```bash
python code/figures/04_plot_alignment_spread_and_medians.py
```

That script deliberately rebuilds its dense-layer input from
`../model_scope_followups/layer_sweep/results/mrsa_dense_all_eval_layer_scores.csv`
before plotting, so the layer-depth medians come from the current dense sweep
rather than an older cached summary.

## Caches And Heavy Files

Feature caches live outside the compact result tables:

- `cache_or_heavy/selected_layer_features_srp5920/features/`
- `cache_or_heavy/selected_layer_features_srp5920/dv_features/`
- `cache_or_heavy/target_adaptation_srp5920/alphas/`

The current compact score tables and figures are small enough for git. The full
cache directories and Slurm logs are local Raven artifacts and should be moved
with `rsync` if another server needs the complete run state.

## Interpretation Notes

The pre-fix `20260620_173400_png_full` tables should not be used to interpret
`target_weight=0`: that mismatch was traced to a double evaluation-feature
transform. After the corrected rerun, fixed-alpha `target_weight=0` should match
the original dense layer-sweep mRSA reference up to numerical tolerance. Any
remaining refit-vs-fixed `target_weight=0` delta should be rechecked against the
corrected outputs rather than inferred from the pre-fix run.
