# LAION Selection Objective Control

This control asks whether the project selection objective gives a meaningful
advantage over a pure feature-decorrelation objective on a natural-image pool.

## Methods

We used the 10k-image LAION natural sample in
`/data/home_roth/_stachelschwein/rsa_based_selection/data/LAION_natural_sample`.
To keep the fixed-RSA and mixed-RSA evaluations on the same model set, we
included only models with full 10k cached feature matrices whose dimensionality
matched the existing shared-subject encoding weights. This yielded 11 model
feature spaces:

- `dinov2_vitl14`
- `openclip_vit_so400m_14_siglip_webli`
- `robustness_imagenet_l2_eps3`
- `slip_vit_l_simclr`
- `slip_vit_l_slip`
- `timm_vit_large_patch14_clip_quickgelu_224_openai`
- `torchvision_alexnet_imagenet1k_v1`
- `torchvision_resnet50_imagenet1k_v1`
- `vicreg_resnet50`
- `vissl_resnet50_mocov2`
- `vissl_resnet50_supervised`

Two 50-image sets were selected with identical greedy multitrack settings,
differing only in the assumed RDM noise variance. The `noise_aware` condition
used the calibrated RDM noise variance for each model, matching the project
selection method. The `pure_decorrelation` condition used the same selector
with all RDM noise variances set to zero, so that selection reduced to clean
raw-feature decorrelation.

Both selected image sets were evaluated with the same in-silico noise curve
analysis. In fixed RSA, discriminability was computed from raw model-feature
RDMs. In mixed RSA, raw features were projected through the existing
subject-specific encoding models for `sub-01`, `sub-03`, `sub-05`, `sub-06`,
and `sub-07`, restricted to `hlvis` voxels, and curves were averaged across
subjects. Evaluation used cosine RDMs, correlation similarity, target noise
ceiling 0.46, and 1000 noise samples per noise level.

## Result

The two objectives produced very similar recovery curves. By integrated error
AUC, the noise-aware selector was slightly better for fixed RSA, whereas pure
decorrelation was slightly better for mixed RSA:

| RSA space | Noise-aware error AUC | Pure decorrelation error AUC |
|---|---:|---:|
| Fixed | 0.3376 | 0.3452 |
| Mixed | 0.4143 | 0.4049 |

At the empirical noise multiplier, both methods were close to ceiling in fixed
RSA and high in mixed RSA:

| RSA space | Noise-aware accuracy | Pure decorrelation accuracy |
|---|---:|---:|
| Fixed | 0.992 | 0.997 |
| Mixed | 0.945 | 0.952 |

The practical conclusion is that this LAION natural-image control does not show
a large advantage of the noise-aware objective over pure raw-feature
decorrelation. It is therefore best interpreted as a robustness check: the
choice between these two closely related selection objectives does not appear to
drive a large difference in in-silico model recovery for this 50-image pilot.

## Outputs

- `figures/laion_decorrelation_pilot_curve.pdf`: combined four-curve figure.
- `results/summary_auc.csv`: summary AUC and empirical-noise accuracy table.
- `results/noise_curves.csv`: full fixed/mixed noise curves.
- `results/selection_indices.csv`: selected LAION sample rows for both methods.
- `REPORT.md`: generated run report with model list and calibration values.

The analysis is reproducible with:

```bash
LD_LIBRARY_PATH=/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-} \
/data/home_roth/miniforge3/bin/python \
  05_controls_and_supplementary/selection_objective_validation/laion_decorrelation_pilot/code/01_run_laion_decorrelation_pilot.py \
  --model-set laion11 \
  --output-dir 05_controls_and_supplementary/selection_objective_validation/laion_decorrelation_laion11 \
  --extract-batch-size 256
```
