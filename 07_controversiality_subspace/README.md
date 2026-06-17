# Controversiality Subspace

Toy probe for asking whether a simple CLIP-space direction can predict images
that make two model representational geometries disagree.

The first analysis is CPU-only and uses cached local features from:

`/data/home_roth/_stachelschwein/rsa_based_selection/data/LAION_natural_sample`

The toy objective uses two objective models:

- `torchvision_resnet50_imagenet1k_v1_layer-2.npy`
- `dinov2_vitl14_layer-1.npy`

For a set of images `S`, controversiality is:

```text
C(S) = 1 - corr(RDM_resnet50(S), RDM_dinov2(S))
```

The surrogate analyses use CLIP-space features:

- `openclip_vit_l_14_quickgelu_metaclip_400m_layer0.npy`

The script fits two CPU-only CLIP-space surrogates:

1. a selected-vs-random logistic classifier, kept as a descriptive baseline;
2. a marginal-utility ridge regressor trained on average exact marginal gain,
   `C(context + image) - C(context)`, across random training contexts.

## Run

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/data/home_roth/miniforge3/bin/python \
  07_controversiality_subspace/code/01_toy_clip_surrogate.py
```

Outputs are written to:

- `07_controversiality_subspace/results/`
- `07_controversiality_subspace/figures/`

## Main Questions

1. Can exact greedy selection find 20-image sets with high ResNet-vs-DINO
   controversiality from different initializations?
2. Can a linear CLIP-space classifier distinguish selected images from random
   local natural images?
3. Can a CLIP-space marginal-utility regressor predict exact marginal gain?
4. Can either CLIP-space surrogate propose held-out images that actually form
   high-controversiality sets under the original ResNet-vs-DINO objective?
5. Does CLIP-space MMR diversity improve direct marginal top-20 selection or
   the marginal top-500 proposal pool?

The acquisition evaluation is run separately for the classifier and marginal
regressor. Each compares random held-out sets, direct top-20 surrogate images,
exact greedy selection from a random 500-image held-out subset, exact greedy
selection from the top 500 surrogate-ranked held-out images, and exact greedy
selection from the full held-out pool.

## Current Run

With the default settings, the exact toy optimizer strongly separates random and
controversial training sets:

```text
random_train mean C = 0.688
exact_train  mean C = 1.642
```

The selected-vs-random classifier is weak as an acquisition filter:

```text
held-out selected-vs-random ROC AUC = 0.611
random-K + exact mean C              = 1.434
classifier top-K + exact mean C      = 1.436
```

The marginal-utility regressor has a stronger held-out marginal target fit and
a more useful top-K proposal pool:

```text
marginal label eval Pearson          = 0.697
marginal label eval Spearman         = 0.648
random-K + exact mean C              = 1.482
marginal top-K + exact mean C        = 1.514
full held-out exact mean C           = 1.577
```

CLIP-space diversity regularization gives only a small direct top-20 gain and
does not improve the exact-rescored top-500 proposal pool:

```text
marginal top20 mean C                  = 0.732
best diverse marginal top20 C          = 0.776  (MMR weight = 1.0)
marginal top-K + exact mean C          = 1.514
best diverse marginal top-K + exact C  = 1.500  (MMR weight = 0.5)
```

## Pair-Level CLIP Probe

`code/02_pair_level_clip_disagreement.py` asks whether pairwise CLIP features
can predict pairwise disagreement between ResNet50 and DINOv2. The target is
scale-normalized before comparison:

```text
delta(i, j) = z_train(d_resnet(i, j)) - z_train(d_dino(i, j))
```

The symmetric CLIP pair feature is:

```text
concat(abs(clip_i - clip_j), clip_i * clip_j)
```

Current default run:

```text
signed delta test Pearson       = 0.361
signed delta test Spearman      = 0.361
absolute delta test Pearson     = 0.224
absolute delta test Spearman    = 0.115
```

The rank curve is monotonic at the extremes, but the set-acquisition test is
negative:

```text
random mean C                   = 0.665
pair-CLIP greedy mean C          = 0.364
random-K + exact mean C          = 1.423
pair-CLIP top-K + exact mean C   = 1.405
full exact mean C                = 1.540
```

So CLIP contains some pair-disagreement signal, but this simple pair-level
surrogate is not yet useful for constructing controversial image sets.

## Local Residual Geometry

`code/03_local_residual_geometry.py` is the main representational-geometry
analysis. For each query image `x`, each model gets a similarity profile to a
shared random anchor set:

```text
s_m(x) = [sim_m(x, a_1), ..., sim_m(x, a_K)]
```

Profiles are normalized within model/query (`rankz` by default). The consensus
profile is the model average, and the local residual geometry score is:

```text
C_local(x) = mean_m,anchor (s_m(x) - mean_m s_m(x))^2
```

Default run details:

- 20 cached model feature spaces from `natural_pool_subset_100k_seed42`
- 1000 random anchors
- 100 frozen selected images from `all_models`
- 1000 random comparison images

Current default run:

```text
random mean C_local      = 0.538
selected mean C_local    = 0.616
selected - random        = 0.079
P(selected > random)     = 0.786
```

Largest selected-minus-random model residual increases:

```text
robustness_imagenet_l2_eps3       +0.145
torchvision_vgg16_imagenet1k_v1   +0.121
vissl_resnet50_mocov2             +0.113
vissl_resnet50_supervised         +0.111
dinov2_vitl14                     +0.110
```

This is the first analysis in this folder that cleanly matches the intended
interpretation: selected controversial stimuli have more model-dependent local
neighborhood geometry than random natural images.

## Contrastive Residual Neighbors

`code/04_contrastive_residual_neighbors.py` turns the local residual geometry
score into inspection panels. For the 20 selected images with highest
`C_local`, it computes positive residual anchors:

```text
e_m(x, a) = s_m(x, a) - mean_m s_m(x, a)
```

These are anchors that a model treats as unusually similar to the selected
image, relative to the cross-model consensus. The output unit is:

```text
(selected image, model, top positive-residual anchors)
```

The default panel uses five local-image-compatible models:

- `torchvision_resnet50_imagenet1k_v1`
- `dinov2_vitl14`
- `slip_vit_l_slip`
- `openclip_vit_so400m_14_siglip_webli`
- `robustness_imagenet_l2_eps3`

Outputs:

- `figures/contrastive_residual_neighbors_top20.pdf`: one page per selected
  image.
- `figures/contrastive_residual_neighbors/*.pdf`: individual image panels.
- `results/contrastive_residual_neighbors.csv`: all positive-residual anchors.
- `results/contrastive_residual_model_label_template.csv`: manual cue-label
  template.
- `results/contrastive_residual_strongest_model_pairs.csv`: strongest model
  contrast per selected image.

Current strongest contrast counts among the top 20 selected images:

```text
openclip_siglip vs robustness_imagenet_l2_eps3     8
openclip_siglip vs torchvision_resnet50            7
openclip_siglip vs slip_vit_l_slip                 2
robustness_imagenet_l2_eps3 vs torchvision_resnet50 1
robustness_imagenet_l2_eps3 vs slip_vit_l_slip     1
dinov2_vitl14 vs robustness_imagenet_l2_eps3       1
```

Manual label columns are intentionally blank. The intended workflow is to
inspect the panels first, then label what each model-specific residual
neighborhood appears to share, without forcing a predefined shape/texture/scene
taxonomy too early.

## Annotation App

`code/05_build_annotation_cards.py` builds a browser-labeling manifest for
contrastive residual-neighborhood cards. It uses cached cstim/VICCO feature
arrays from `shared/cache_or_heavy/cstim_paper_feature_cache/feature_cache/vicco`
and image thumbnails from
`/data/labshare/_stachelschwein/SSD/jroth/final_cstims_hdf5_files`.
It is CPU-only.

Current generated app data:

- `annotation_app/data/cards.json`: 1200 cards.
- `annotation_app/data/annotations.jsonl`: append-only annotation output.
- `annotation_app/static/stimuli/`: 792 generated thumbnails.

The card set contains:

- 100 query images from each of the five selected model sets.
- 100 sampled query images from the 292-image VICCO baseline.
- One real residual-neighborhood card and one random anchor-control card per
  query image.
- Subject-averaged brain-pair bins for each displayed central-anchor relation:
  `similar` for `mean_brain_z <= -0.75`, `neutral` for
  `-0.75 < mean_brain_z < 0.75`, and `dissimilar` for
  `mean_brain_z >= 0.75`. Negative z means more brain-similar; positive z
  means more brain-dissimilar.

Run:

```bash
/data/home_roth/miniforge3/bin/python 07_controversiality_subspace/code/05_build_annotation_cards.py
/data/home_roth/miniforge3/bin/python 07_controversiality_subspace/annotation_app/server.py --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765`. Model identities are hidden in the UI but retained
in the card metadata and CSV export. The UI also supports filtering cards by
dominant or contained brain-pair bin and shows per-anchor brain-bin badges.
