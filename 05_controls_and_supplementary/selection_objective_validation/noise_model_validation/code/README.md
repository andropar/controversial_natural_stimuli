# Noise Model Validation

This folder contains the supplemental validation comparing two noise models used in the
in-silico stimulus-selection analysis:

1. RDM-space noise: add calibrated Gaussian noise directly to RDM vectors.
2. Feature-space Monte Carlo reference: add calibrated Gaussian noise to model features
   and recompute RDMs.

The original completed run was moved from `scripts/claude/validate_noise_approximation`.
The CSV files in `data/` are the frozen results used for the supplement figure.

Run:

```bash
python experiments/cstim_paper/16_noise_model_validation/figures/plot_noise_model_validation.py
```

This writes `noise_model_validation.pdf` and `.png` in `figures/`.

