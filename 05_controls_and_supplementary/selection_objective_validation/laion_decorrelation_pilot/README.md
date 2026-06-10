# LAION Decorrelation Pilot

Two-model control comparing the project raw-track noisy selection objective
against a pure clean-decorrelation baseline on the LAION natural sample. The
current run uses cached encoder-compatible `dinov2_vitl14` and
`slip_vit_l_slip` features and evaluates both fixed RSA and mixed RSA.

Run:

```bash
LD_LIBRARY_PATH=/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-} \
/data/home_roth/miniforge3/bin/python \
05_controls_and_supplementary/selection_objective_validation/laion_decorrelation_pilot/code/01_run_laion_decorrelation_pilot.py
```

Outputs are written to `results/`, `figures/`, and `REPORT.md`.
