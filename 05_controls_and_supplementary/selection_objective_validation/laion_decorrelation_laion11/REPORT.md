# LAION Decorrelation Pilot

This pilot compares two raw-feature selection objectives on the 10k-image LAION
natural sample using 11 encoder-compatible model feature spaces:

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

Selection variants:

- `noise_aware`: project raw-track selector with calibrated RDM noise.
- `pure_decorrelation`: identical selector with zero RDM noise variance.

Evaluation spaces:

- `fixed`: raw-feature RDMs.
- `mixed`: subject encoding-predicted RDMs, averaged over 5 subjects.

Configuration:

- target size: 50
- init size: 3
- seed: 42
- model set: `laion11`
- metric: `cosine`
- correlation: `correlation`
- target noise ceiling: 0.46
- evaluation noise samples per point: 1000

Raw calibrated RDM noise variances:

```json
{
  "dinov2_vitl14": 0.01025390625,
  "openclip_vit_so400m_14_siglip_webli": 0.02978515625,
  "robustness_imagenet_l2_eps3": 0.033203125,
  "slip_vit_l_simclr": 0.00048828125,
  "slip_vit_l_slip": 0.02490234375,
  "timm_vit_large_patch14_clip_quickgelu_224_openai": 0.01806640625,
  "torchvision_alexnet_imagenet1k_v1": 0.02294921875,
  "torchvision_resnet50_imagenet1k_v1": 0.02197265625,
  "vicreg_resnet50": 0.0185546875,
  "vissl_resnet50_mocov2": 0.0419921875,
  "vissl_resnet50_supervised": 0.03564453125
}
```

Accuracy at empirical noise multiplier 1:

- fixed / noise_aware: 0.992
- fixed / pure_decorrelation: 0.997
- mixed / noise_aware: 0.945
- mixed / pure_decorrelation: 0.952

Lower error AUC is better. Winners by evaluation space:

- fixed: `noise_aware` (error AUC 0.3376)
- mixed: `pure_decorrelation` (error AUC 0.4049)

Interpretation:

The two objectives produced very similar in-silico recovery curves. The
noise-aware objective was slightly better by fixed-RSA error AUC, whereas pure
decorrelation was slightly better by mixed-RSA error AUC and at the empirical
noise point. Thus, in this LAION natural-image pilot, the noise-aware objective
does not provide a clear practical advantage over pure raw-feature
decorrelation. This control is best treated as evidence that the choice between
these two related selection objectives is unlikely to drive a large difference
in model recovery for this 50-image setting.
