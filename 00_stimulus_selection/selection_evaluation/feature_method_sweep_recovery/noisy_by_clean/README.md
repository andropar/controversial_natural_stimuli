# Feature-Method-Sweep Noisy-By-Clean Recovery

This directory applies the shared noisy-by-clean recovery method to
feature-method-sweep payloads.

Scripts:

- `01_compute_recovery.py`: compute recovery curves for sweep payloads.
- `02_compute_full_track_cross_eval.py`: rebuild restricted-track payloads with
  all evaluation tracks, then run recovery.
- `03_compute_robustness_diagnostics.py`: feature-sweep diagnostic comparing
  RDM-level and feature-level noise perturbations.
- `04_plot_summary.py`: plot compact feature-sweep recovery summaries.

Shared implementation lives in
`../../code/noisy_by_clean/`.
