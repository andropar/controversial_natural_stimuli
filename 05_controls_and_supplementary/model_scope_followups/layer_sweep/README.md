# Dense mRSA Layer Sweep Follow-Up

This is the current working layer-sweep control under:

```text
/raven/u/rothj/controversial_natural_stimuli/05_controls_and_supplementary/model_scope_followups/layer_sweep
```

The original `11_layer_sweep` analysis used a configured 5-7 layer set per
model and included older fRSA/mRSA rescue analyses. The current paper-facing
follow-up is different: it is a dense mRSA-transfer layer-selection analysis
for the 20 paper models.

## Current Authoritative Run

Use the outputs from Raven run stamp:

```text
20260620_173400_png_full
```

The exact run record is in:

```text
logs/raven_dense_stream_20260620_173400_png_full/
```

The current full stream parts are in:

```text
results/stream_parts_raven_dense_20260620_173400_png_full/
```

Older smoke runs, OOM retries, and older `stream_parts*` directories are kept
for provenance. They are not the current dense layer-sweep result set.

## What Was Run

The production run used `CSTIMS_PATH_ENV=raven`, the dense layer set from
`code/layers_config.py`, and one Slurm array task per `(subject, model)`.

The generated worker script ran:

```bash
/u/rothj/conda-envs/deepjuice/bin/python \
  code/analysis/07_fit_encodings_layer_sweep.py \
  --subject <subject> \
  --models <model> \
  --mode stream-model \
  --layer-set dense \
  --batch-size auto \
  --batch-candidates 1,2,4,8,16,32 \
  --layers-per-chunk auto \
  --max-layers-per-chunk 128 \
  --max-feature-gb-per-chunk 24 \
  --fit-backend gpu \
  --gpu-fit-dtype float64 \
  --n-fit-jobs 1 \
  --n-score-jobs 3 \
  --n-vicco-boot 1000 \
  --n-shared-boot 1000 \
  --bootstrap-n 100 \
  --extract-prefetch-workers 2 \
  --deepvision-load-jobs 1 \
  --stream-part-root results/stream_parts_raven_dense_20260620_173400_png_full \
  --progress-log results/stream_parts_raven_dense_20260620_173400_png_full/progress_<subject>_<model>.jsonl
```

The generated merge script then ran:

```bash
python code/analysis/07_fit_encodings_layer_sweep.py \
  --mode merge-stream \
  --layer-set dense \
  --subject sub-01,sub-03,sub-05,sub-06,sub-07 \
  --stream-part-root results/stream_parts_raven_dense_20260620_173400_png_full \
  --out-csv results/wrsa_dense_layer_sweep.csv \
  --shared-out-csv results/wrsa_dense_shared_layer_sweep.csv

python code/analysis/14_build_mrsa_layer_selection_tables.py \
  --layer-set dense \
  --wrsa-csv results/wrsa_dense_layer_sweep.csv \
  --shared-csv results/wrsa_dense_shared_layer_sweep.csv
```

For a new Raven run, the source helper is:

```bash
bash code/queue_dense_layer_sweep_raven_slurm.sh
```

Set `RUN_STAMP` if you want deterministic output/log directory names.

## Result Coverage

The current dense mRSA tables contain:

- 20 models
- 5 subjects: `sub-01`, `sub-03`, `sub-05`, `sub-06`, `sub-07`
- 485 model-layer pairs
- evaluation targets: DeepVision shared, Vicco, and the five CSTIM sets
- selection rules: `paper_layer`, `best_on_shared`, `best_on_cstim`

The current analysis did not only test 5-7 layers. That description applies to
the older configured-layer sweep.

Per-model dense layer counts:

| Model | Layers |
|---|---:|
| `cornet_s` | 5 |
| `robustness_imagenet_l2_eps3` | 21 |
| `torchvision_resnet50_imagenet1k_v1` | 22 |
| `vicreg_resnet50` | 22 |
| `vissl_resnet50_supervised` | 22 |
| `vissl_resnet50_mocov2` | 22 |
| `vissl_resnet50_barlowtwins` | 22 |
| `dinov2_vitl14` | 24 |
| `torchvision_vit_l_16_imagenet1k_v1` | 24 |
| ViT-L / CLIP-style models | 25 |
| `openclip_vit_so400m_14_siglip_webli` | 28 |
| `torchvision_vgg16_imagenet1k_v1` | 34 |
| `torchvision_convnext_base_imagenet1k_v1` | 39 |

The dense inventory is "meaningful block/module outputs plus paper taps", not
every FX graph operation. Internal paper taps are retained so the paper-layer
comparison is exact.

## Current Full Data

These are the full current dense outputs in `results/`. They are the files to
copy with `rsync` when another server needs the complete data:

- `wrsa_dense_layer_sweep.csv`: dense CSTIM/Vicco mRSA score table,
  14,271,000 data rows, about 2.9G.
- `wrsa_dense_shared_layer_sweep.csv`: dense DeepVision-shared mRSA score
  table, 14,200,000 data rows, about 3.5G.
- `mrsa_dense_all_eval_layer_scores.csv`: combined per-layer mRSA score table,
  28,471,000 data rows, about 6.6G.
- `stream_parts_raven_dense_20260620_173400_png_full/`: current per-task stream
  parts and progress logs, about 6.7G.

The full `results/` directory is about 29G. The three full current raw dense
CSVs are intentionally ignored by `.gitignore` and are not in git:

- `results/wrsa_dense_layer_sweep.csv`
- `results/wrsa_dense_shared_layer_sweep.csv`
- `results/mrsa_dense_all_eval_layer_scores.csv`

The current stamped stream-part directory is also not in git:

- `results/stream_parts_raven_dense_20260620_173400_png_full/`

It is an untracked local Raven run artifact and should be moved with `rsync`
when the complete run state is needed elsewhere.

## Current Git-Tracked Summaries

Use these compact summaries for normal analysis, review, and plotting:

- `results/mrsa_dense_layer_selection_transfer.csv`: 4,900 data rows. This is
  the main table for comparing `paper_layer`, `best_on_shared`, and
  `best_on_cstim` across shared, Vicco, and CSTIM evaluation targets.
- `results/mrsa_layer_depth_curve_summary.csv`: 19,880 data rows. This is the
  compact layer-depth curve table rebuilt from the current dense all-eval
  scores.
- `results/mrsa_selection_transfer_delta_summary.csv`
- `results/mrsa_brain_alignment_layer_rule_summary.csv`
- `results/mrsa_brain_alignment_best_shared_vs_cstim_layers_summary.csv`
- `results/mrsa_brain_alignment_best_shared_vs_cstim_layers_nc_normalized_summary.csv`
- `results/mrsa_brain_alignment_paper_vs_shared_summary.csv`
- `results/mrsa_brain_alignment_paper_vs_shared_nc_normalized_summary.csv`
- `results/mrsa_brain_alignment_paper_vs_shared_with_shared_summary.csv`
- `results/mrsa_brain_alignment_paper_vs_shared_with_shared_nc_normalized_summary.csv`
- `results/mrsa_selected_layer_shift_summary.csv`

Legacy configured/fRSA outputs from the older analysis are still present in
`results/`, but they are not the active dense mRSA follow-up.

## Current Figures

Current figures in `figures/`:

- `brain_alignment_best_shared_layer_improved.pdf`
- `mrsa_selection_transfer_delta.pdf`
- `mrsa_selected_layer_shift.pdf`
- `mrsa_layer_depth_curves_all_models.pdf`
- `mrsa_layer_depth_curves_architecture.pdf`
- `mrsa_layer_depth_curves_dataset.pdf`
- `mrsa_layer_depth_curves_sota.pdf`
- `mrsa_layer_depth_curves_training_objective.pdf`
- `mrsa_brain_alignment_layer_rules.pdf`
- `mrsa_brain_alignment_best_shared_vs_cstim_layers.pdf`
- `mrsa_brain_alignment_best_shared_vs_cstim_layers_nc_normalized.pdf`
- `mrsa_brain_alignment_paper_vs_shared.pdf`
- `mrsa_brain_alignment_paper_vs_shared_nc_normalized.pdf`
- `mrsa_brain_alignment_paper_vs_shared_with_shared.pdf`
- `mrsa_brain_alignment_paper_vs_shared_with_shared_nc_normalized.pdf`

PNG copies live under `figures/png/`.

The figures were regenerated from the current summary tables with:

```bash
python code/figures/plot_mrsa_selection_transfer.py
python code/figures/plot_mrsa_selected_layer_shift.py
python code/figures/plot_mrsa_layer_depth_curves.py
python code/figures/plot_mrsa_brain_alignment_layer_rules.py
python code/figures/plot_mrsa_best_shared_vs_cstim_layers.py
python code/figures/plot_mrsa_brain_alignment_paper_vs_shared.py
python code/figures/plot_brain_alignment_best_shared_layer_improved.py
```

## Layer-Selection Definitions

- `paper_layer`: the exact layer from
  `00_stimulus_selection/resources/model_list.csv` used in the main paper
  pipeline.
- `best_on_shared`: per subject and model, the dense layer with the best mRSA
  on held-out DeepVision shared images.
- `best_on_cstim`: per subject, model, and CSTIM set, the dense layer with the
  best mRSA on that CSTIM set. This is an oracle/supplementary rule.

Current code keeps internal paper taps in chronological position for layer
depth plots. This matters for ConvNeXt (`features.5.*.block.2`) and the timm
CLIP qkv paper layer (`blocks.18.attn.qkv`); otherwise those internal nodes
look like final layers even though they are not.

DINOv2 dense layers currently use `blocks.{i}.norm2`, matching the paper-layer
tap family. These are internal block taps, not whole-block outputs, so sharp
late-layer drops at `blocks.21.norm2`/`blocks.22.norm2` should be interpreted
as internal-tap behavior rather than a clean final-block readout.

## Transfer Notes

Git contains the compact summaries and figures. It does not contain the current
full 29G dense result tree. To move the complete dense run to another server,
pull it from Raven with `rsync`, for example:

```bash
SRC="rothj@raven04i.mpcdf.mpg.de:/raven/u/rothj/controversial_natural_stimuli"
DEST="/destination/path/controversial_natural_stimuli"

rsync -aH --partial --info=progress2 \
  "$SRC/05_controls_and_supplementary/model_scope_followups/layer_sweep/results" \
  "$SRC/05_controls_and_supplementary/model_scope_followups/layer_sweep/figures" \
  "$SRC/05_controls_and_supplementary/model_scope_followups/layer_sweep/logs" \
  "$DEST/05_controls_and_supplementary/model_scope_followups/layer_sweep/"
```

If direct Raven login is not reachable from the destination server, use the
MPCDF gateway with `ssh -J`.

## Cache Layout

Heavy files in this share copy live under `cache_or_heavy/`. The portable
`_paths.py` files keep outputs relative to this directory while still allowing
imports from the source repository for shared helper modules.
