# LAION Decorrelation Pilot

This pilot compares two raw-feature selection objectives on the 10k-image LAION
natural sample using encoder-compatible model features:

- `dinov2_vitl14` (`blocks.19.norm2`)
- `slip_vit_l_slip` (`fc_norm`)

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
- metric: `cosine`
- correlation: `correlation`
- target noise ceiling: 0.46
- evaluation noise samples per point: 1000

Raw calibrated RDM noise variances:

```json
{
  "dinov2_vitl14": 0.01025390625,
  "slip_vit_l_slip": 0.0244140625
}
```

Accuracy at empirical noise multiplier 1:

- fixed / noise_aware: 1.000
- fixed / pure_decorrelation: 1.000
- mixed / noise_aware: 1.000
- mixed / pure_decorrelation: 1.000

Lower error AUC is better. Winners by evaluation space:

- fixed: `noise_aware` (error AUC 0.0924)
- mixed: `noise_aware` (error AUC 0.1174)
