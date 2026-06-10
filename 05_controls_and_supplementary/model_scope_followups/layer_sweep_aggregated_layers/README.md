# Dense mRSA Layer Sweep Follow-Up

This share copy is the current working version of the layer-sweep control in
`/data/home_roth/cstims_share/05_controls_and_supplementary/model_scope_followups/layer_sweep`.

The original `11_layer_sweep` analysis used a configured 5-7 layer set per
model and included both fRSA and mRSA rescue analyses. The current
supplementary follow-up is different: it focuses on dense mRSA-transfer layer
selection for the 20 paper models.

## Current Status

The current dense mRSA tables contain:

- 20 models
- 5 subjects (`sub-01`, `sub-03`, `sub-05`, `sub-06`, `sub-07`)
- 485 model-layer pairs
- evaluation targets: DeepVision shared, Vicco, and the five cstim sets
- selection rules: paper layer, best on shared, and best on cstim

So no, the current analysis did not only test 5-7 layers. The 5-7 layer
description applies to the older configured sweep.

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

## Main Data

Primary dense outputs in `results/`:

- `wrsa_dense_layer_sweep.csv`: dense cstim/Vicco mRSA scores.
- `wrsa_dense_shared_layer_sweep.csv`: dense DeepVision shared mRSA scores.
- `mrsa_dense_all_eval_layer_scores.csv`: combined per-layer mRSA score table.
- `mrsa_dense_layer_selection_transfer.csv`: paper, best-on-shared, and
  best-on-cstim layer selections evaluated on shared, Vicco, and cstims.

Derived summaries:

- `mrsa_selection_transfer_delta_summary.csv`
- `mrsa_brain_alignment_layer_rule_summary.csv`
- `mrsa_brain_alignment_paper_vs_shared_summary.csv`
- `mrsa_brain_alignment_paper_vs_shared_nc_normalized_summary.csv`
- `mrsa_selected_layer_shift_summary.csv`
- `mrsa_layer_depth_curve_summary.csv`

Legacy configured/fRSA outputs from the older analysis are still present in
`results/`, but they are not the active dense mRSA follow-up.

## Main Figures

Current figures in `figures/`:

- `mrsa_selection_transfer_delta.pdf`
- `mrsa_selected_layer_shift.pdf`
- `mrsa_layer_depth_curves_{all_models,architecture,dataset,sota,training_objective}.pdf`
- `mrsa_brain_alignment_layer_rules.pdf`
- `mrsa_brain_alignment_paper_vs_shared.pdf`
- `mrsa_brain_alignment_paper_vs_shared_nc_normalized.pdf`

PNG copies live under `figures/png/`.

## Main Scripts

Dense mRSA pipeline:

```bash
python code/analysis/run_layer_set_pipeline.py \
  --layer-set dense \
  --gpus 0,1,2,3,4,5,6,7

python code/analysis/run_shared_layer_set_pipeline.py \
  --layer-set dense \
  --gpus 0,1,2,3,4,5,6,7

python code/analysis/14_build_mrsa_layer_selection_tables.py --layer-set dense
```

Figure regeneration:

```bash
python code/figures/plot_mrsa_selection_transfer.py
python code/figures/plot_mrsa_selected_layer_shift.py
python code/figures/plot_mrsa_layer_depth_curves.py
python code/figures/plot_mrsa_brain_alignment_layer_rules.py
python code/figures/plot_mrsa_brain_alignment_paper_vs_shared.py
```

Older scripts `01`-`12` are retained for the configured-layer/fRSA rescue
analysis and for provenance, but the current paper-facing dense mRSA tables
come from scripts `13` and `14` plus the figure scripts above.

## Layer-Selection Notes

- `paper_layer`: the exact layer from `00_stimulus_selection/resources/model_list.csv` used in
  the main paper pipeline.
- `best_on_shared`: per subject and model, the dense layer with the best mRSA
  on held-out DeepVision shared images.
- `best_on_cstim`: per subject, model, and cstim set, the dense layer with the
  best mRSA on that cstim set. This is an oracle/supplementary rule.

Current code keeps internal paper taps in chronological position for layer
depth plots. This matters for ConvNeXt (`features.5.*.block.2`) and the timm
CLIP qkv paper layer (`blocks.18.attn.qkv`); otherwise those internal nodes
look like final layers even though they are not.

DINOv2 dense layers currently use `blocks.{i}.norm2`, matching the paper-layer
tap family. These are internal block taps, not whole-block outputs, so sharp
late-layer drops at `blocks.21.norm2`/`blocks.22.norm2` should be interpreted
as internal-tap behavior rather than a clean final-block readout.

## Cache Layout

Heavy files in this share copy live under `cache_or_heavy/`. The portable
`_paths.py` files keep outputs relative to this share directory while still
allowing imports from the source repository for shared helper modules.
