# Image Transform Alignment Probe

Exploratory counterfactual analysis for CSTIMS controversial stimuli.

## Goal

This folder tests a model-diagnostic counterfactual question:

> Given the brain-measured geometry for the original stimulus set, what minimal,
> natural-looking edit to image B moves a chosen model's representation of B
> toward the brain-derived relational position of B?

The first prototype uses a single image pair:

- anchor: `image_0022.png`, a dense floral wall
- target: `image_0046.png`, a light stone wall
- model: `torchvision_resnet50_imagenet1k_v1`
- subject target: `sub-06`

The brain places this pair at a very similar end of its z-scored distance
distribution, while ResNet50 places it at a very dissimilar end.

## Important Caveat

The optimized image has no measured brain response. This analysis should not be
read as evidence that the brain would represent the optimized image that way.
It is a model diagnostic: it asks what visual changes are sufficient, under a
specified edit family and image prior, to move a model's pairwise geometry
toward a brain-derived target measured on the original images.

The single-pair version is intentionally a pilot. One scalar target is
underdetermined, so the stronger version should optimize a multi-anchor distance
profile: keep many anchors fixed and edit one target image so the model's
distance profile from the edited target to those anchors better matches the
brain's distance profile from the original target to those anchors.

## Folder Contents

- `code/00_rank_candidate_pairs.py`: ranks model/image/subject candidates using
  existing pair-level brain placement tables and cached selection-time features.
- `code/01_resnet50_pixel_counterfactual.py`: quick pixel-space optimization
  prototype for the selected ResNet50 pair. This is an unconstrained control.
- `code/02_parametric_edit_probe.py`: deterministic low-dimensional edit search
  over grayscale, blur, saturation, contrast, brightness, gamma, and zoom.
- `code/03_lowfreq_residual_counterfactual.py`: low-resolution residual
  optimizer used as a constrained but still non-naturalistic control.
- `code/04_summarize_counterfactual_runs.py`: summary table builder for the
  early counterfactual runs.
- `code/05_guided_diffusion_counterfactual.py`: gradient-guided Stable
  Diffusion/DDIM img2img counterfactual. At each denoising step it decodes the
  predicted clean image, computes the ResNet50 pair-distance loss, backpropagates
  that loss to the current diffusion latent, and applies a stabilized latent
  guidance step before the DDIM update.
- `code/06_guided_diffusion_multi_anchor_profile.py`: multi-anchor profile
  version of the guided diffusion method. It optimizes the edited target image
  against a train-anchor distance profile and evaluates held-out anchors.
- `results/`: CSV outputs.
- `figures/`: diagnostic figures.
- `optimized_images/`: optimized target images and snapshots.

## Run

Use the project conda Python:

```bash
/data/home_roth/miniforge3/bin/python \
  05_controls_and_supplementary/image_transform_alignment_probe/code/00_rank_candidate_pairs.py
```

For the PyTorch prototype, use the conda library path:

```bash
LD_LIBRARY_PATH=/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-} \
/data/home_roth/miniforge3/bin/python \
  05_controls_and_supplementary/image_transform_alignment_probe/code/01_resnet50_pixel_counterfactual.py
```

Run the interpretable parametric edit probe:

```bash
LD_LIBRARY_PATH=/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-} \
/data/home_roth/miniforge3/bin/python \
  05_controls_and_supplementary/image_transform_alignment_probe/code/02_parametric_edit_probe.py
```

Run the gradient-guided diffusion prototype:

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-} \
/data/home_roth/miniforge3/bin/python \
  05_controls_and_supplementary/image_transform_alignment_probe/code/05_guided_diffusion_counterfactual.py \
  --target-mode quantile \
  --resolution 384 \
  --num-steps 20 \
  --strength 0.65 \
  --guidance-step-size 0.05 \
  --guidance-error-scale 0.75 \
  --inner-guidance-steps 2 \
  --latent-clip-value 10 \
  --guidance-scale 5.0 \
  --identity-weight 0.2 \
  --tv-weight 0.01 \
  --seed 1 \
  --run-label guided_quantile_s065_g005_inner2_clip_seed1
```

The diffusion script expects a local Stable Diffusion 1.5 snapshot at:

```text
/data/home_roth/_stachelschwein/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14
```

Run the multi-anchor guided diffusion prototype:

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-} \
/data/home_roth/miniforge3/bin/python \
  05_controls_and_supplementary/image_transform_alignment_probe/code/06_guided_diffusion_multi_anchor_profile.py \
  --target-mode quantile \
  --n-train-anchors 32 \
  --n-holdout-anchors 16 \
  --resolution 384 \
  --num-steps 20 \
  --strength 0.65 \
  --guidance-step-size 0.035 \
  --guidance-error-scale 0.9 \
  --inner-guidance-steps 2 \
  --latent-clip-value 10 \
  --guidance-scale 5.0 \
  --identity-weight 0.2 \
  --tv-weight 0.01 \
  --profile-mse-weight 1.0 \
  --profile-corr-weight 0.2 \
  --seed 3 \
  --run-label guided_profile_k32_hold16_s065_g0035_seed3
```

## Interpretation

The prototype directly optimizes target pixels with:

```text
loss =
  (ResNet50_z_distance(anchor, optimized_target) - brain_z_target)^2
  + identity_weight * MSE(optimized_target, original_target)
  + tv_weight * total_variation(optimized_target)
```

This is intentionally quick and dirty. It is useful for checking whether the
objective is feasible, but it can produce adversarial or unnatural changes. It
should be interpreted as an unconstrained baseline that demonstrates why
natural-image constraints and edit-family restrictions are necessary.

## First-Pass Result

### Pixel Optimization

Both initial pixel-space runs reached the subject-specific brain z target:

| run | original ResNet50 z | target brain z | final ResNet50 z | absolute error |
| --- | ---: | ---: | ---: | ---: |
| default | 2.370 | -2.393 | -2.393 | 0.0002 |
| strict_identity | 2.370 | -2.393 | -2.393 | 0.0001 |

The optimized target remains recognizably a stone wall but gains a structured
pink/green overlay. This is a warning that the single-pair pixel objective is
easy to satisfy through small adversarial or texture-like changes. The next
prototype should constrain the edit space more strongly, ideally with a
multi-anchor objective and a natural-image prior.

### Parametric Edits

The first deterministic edit grid did not get close to the brain-derived target.
Using a quantile-calibrated model target:

| quantity | value |
| --- | ---: |
| live original ResNet50 z | 2.370 |
| raw subject brain z | -2.393 |
| quantile-calibrated model target z | -1.986 |
| best simple-edit ResNet50 z | 1.793 |
| best simple-edit quantile-target error | 3.779 |

The best simple edit was an `edge_blend` at `alpha=0.5`, which substantially
changes the image and still remains far from the target. Lower-cost edits such
as grayscale, blur, crop/zoom, brightness, contrast, saturation, and gamma only
move the model distance modestly. This makes the pixel-space success look
underconstrained rather than an interpretable natural edit.

### Low-Frequency Residuals

A constrained residual optimizer modifies the target through an upsampled
low-resolution residual rather than free pixels. This still permits artificial
color fields, but it suppresses pixel-level high-frequency solutions.

| run | target mode | target z | best z | absolute error | pixel RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| lowfreq_quantile_default | quantile | -1.986 | -1.986 | 0.000002 | 0.107 |
| lowfreq_quantile_r8 | quantile | -1.986 | -1.948 | 0.038 | 0.157 |
| lowfreq_raw_r32 | raw brain z | -2.393 | -2.393 | 0.000005 | 0.123 |
| lowfreq_raw_r8 | raw brain z | -2.393 | -2.082 | 0.312 | 0.203 |

The close low-frequency solutions are real, but they remain visually diagnostic
rather than naturalistic: RGB residuals add broad color fields to the stone
wall. A luminance-only residual improved the distance but did not reach the
target. This supports the next methodological step: either restrict edits to
interpretable parameters and accept partial movement, or introduce a stronger
natural-image prior and evaluate the Pareto frontier.

### Gradient-Guided Diffusion

The current diffusion prototype implements direct gradient guidance through the
diffusion latent, not candidate reranking. For the quantile-calibrated target:

| quantity | value |
| --- | ---: |
| live original ResNet50 z | 1.972 |
| raw subject brain z | -2.393 |
| quantile-calibrated model target z | -1.986 |
| best guided predicted-x0 z | -1.988 |
| best guided absolute error | 0.0016 |
| final denoised sample z | -1.719 |
| final absolute error | 0.267 |
| best guided pixel RMSE | 0.164 |
| final pixel RMSE | 0.176 |

Output files:

- `results/resnet50_pair_0022_0046_sub-06_guided_quantile_s065_g005_inner2_clip_seed1_summary.csv`
- `results/resnet50_pair_0022_0046_sub-06_guided_quantile_s065_g005_inner2_clip_seed1_trajectory.csv`
- `figures/resnet50_pair_0022_0046_sub-06_guided_quantile_s065_g005_inner2_clip_seed1.png`
- `optimized_images/resnet50_pair_0022_0046_sub-06_guided_quantile_s065_g005_inner2_clip_seed1_best_predx0.png`
- `optimized_images/resnet50_pair_0022_0046_sub-06_guided_quantile_s065_g005_inner2_clip_seed1_final.png`

The generated counterfactual stays within a stone-wall-like image manifold and
changes the block geometry/texture enough to move ResNet50 substantially toward
the brain-derived relational target. This remains a single-pair model
counterfactual, not a claim about neural responses to the edited image.

### Multi-Anchor Gradient-Guided Diffusion

The profile version keeps target image `0046` fixed as the edited image and
selects anchors spanning the `sub-06` brain-distance distribution. For each
anchor, the brain distance is quantile-calibrated into the live ResNet50
distance distribution. The loss is optimized on train anchors only; held-out
anchors are evaluated but not used for guidance.

| run | train anchors | held-out anchors | original train RMSE | best train RMSE | original held-out RMSE | best held-out RMSE | original train r | best train r | original held-out r | best held-out r | best pixel RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `guided_profile_k16_hold8_s065_g004_seed2` | 16 | 8 | 2.039 | 0.854 | 2.020 | 1.765 | 0.447 | 0.729 | 0.172 | -0.050 | 0.195 |
| `guided_profile_k32_hold16_s065_g0035_seed3` | 32 | 16 | 2.169 | 0.999 | 2.243 | 1.554 | 0.064 | 0.561 | -0.164 | -0.270 | 0.225 |

The key result is that gradient-guided diffusion can reduce multi-anchor profile
error while keeping the edit visually in the stone-wall image manifold. The
held-out split is also doing useful work: RMSE improves on held-out anchors, but
held-out profile correlation does not yet improve. That means the current
prototype is moving absolute model distances toward the brain-derived profile
better than it is recovering the held-out rank/order structure.

Main output files for the larger profile run:

- `results/resnet50_target_0046_sub-06_guided_profile_k32_hold16_s065_g0035_seed3_summary.csv`
- `results/resnet50_target_0046_sub-06_guided_profile_k32_hold16_s065_g0035_seed3_trajectory.csv`
- `results/resnet50_target_0046_sub-06_guided_profile_k32_hold16_s065_g0035_seed3_anchor_profile.csv`
- `figures/resnet50_target_0046_sub-06_guided_profile_k32_hold16_s065_g0035_seed3.png`
- `optimized_images/resnet50_target_0046_sub-06_guided_profile_k32_hold16_s065_g0035_seed3_best_predx0.png`
- `optimized_images/resnet50_target_0046_sub-06_guided_profile_k32_hold16_s065_g0035_seed3_final.png`

## Recommended Method Roadmap

1. Deterministic parametric edits: blur, grayscale, saturation, contrast,
   brightness, gamma, zoom/crop. Report Pareto tradeoffs between alignment
   improvement and perceptual/pixel deviation.
2. Differentiable parametric edit layer: optimize edit parameters, not pixels.
3. Low-frequency residual optimization with augmentation robustness.
4. Gradient-guided diffusion-latent optimization with explicit stability
   controls, perceptual/identity penalties, and augmentation robustness.
5. Scale multi-anchor gradient-guided diffusion: optimize one edited target
   image against a brain-derived distance profile to many anchors, evaluate on
   held-out anchors, and add augmentation robustness during guidance.

The paper-level version should use a multi-anchor distance-profile objective and
held-out anchors. The single-pair version is best kept as a didactic
visualization and failure-mode check.
