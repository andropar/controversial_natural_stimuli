# Residual reliability and ensemble-ceiling analysis

Is there reliable brain-RDM structure that is not captured by *any* current
vision model (single model or linear combination)?

For each subject × stimulus group × RSA type (fixed / mixed):

1. Compute the brain RDM (hlvis, correlation distance).
2. Compute 20 model RDMs.
   - **Fixed RSA**: correlation-distance RDM of raw model features.
   - **Mixed RSA**: correlation-distance RDM of encoding-predicted voxel
     responses (subject-specific evaluation encoders).
3. Rank-transform all RDM vectors (Spearman convention).
4. Best single model: `max_m spearman(brain, model_m)`.
5. Ensemble fit: ridge regression `brain ~ w · models`, stimulus-level 10-fold
   cross-validation. Report the Spearman between out-of-fold predictions and
   brain.
6. Noise ceiling reliability: split-half reliability of the brain RDM using
   odd/even reps, Spearman–Brown corrected.
7. Correlation ceiling: `sqrt(noise_ceiling_reliability)`. This is the relevant
   upper bound for model-brain correlations.
8. **LOSO residual RSA** (headline metric):
   - residualize each subject's full brain RDM against that subject's full
     model-RDM space using ridge regression;
   - average the residual vectors from the other subjects;
   - correlate the held-out subject residual with the leave-one-subject-out
     residual mean.

A non-trivial LOSO residual RSA means that structure left after removing the
current model family is shared across participants, not merely measurement
noise in one subject.

Important caveat: absolute LOSO residual RSA is not the same as "how much of
the reliable within-subject signal is left." LOSO only sees residual structure
that is aligned across subjects. The diagnostic contrast is therefore reported
both as an absolute residual and as a fraction of the available LOSO brain RSA.
The within-subject split-half residual fraction is the complementary metric for
reliable subject-specific residual structure.

Done separately for each controversial model set and for the baseline (vicco),
with bootstrap resampling on vicco to match N=100.

## Output

- `data/residual_rsa.csv`: one row per
  `(subject, stimulus_group, rsa_type, bootstrap_idx)` with columns
  including:
  - `r_single_best`, `r_single_best_model`
  - `r_ensemble_cv`
  - `noise_ceiling_reliability`
  - `correlation_ceiling`
  - `ensemble_gap_to_correlation_ceiling`
  - `r_loso_brain`
  - `r_loso_residual`
  - `loso_residual_fraction`

The older columns `r_ensemble`, `noise_ceiling`, and `residual_reliability`
are retained as aliases for compatibility, but the figure uses the explicit
new columns.

- `figures/residual_reliability.{pdf,png}`: top row shows the best single
  model, cross-validated ensemble, and correlation ceiling; bottom row shows
  total LOSO brain RSA and LOSO residual RSA after removing model structure.

- `analysis/02_decompose_residual_signal.py` writes:
  - `data/residual_decomposition_summary.csv`
  - `data/residual_decomposition_contrasts.csv`
  - `figures/residual_decomposition.{pdf,png}`

  This is the main interpretive layer. For mixed RSA on the all-model
  diagnostic set, the cross-validated ensemble gap is larger than baseline
  (`0.217` vs `0.089`), the over-generous full-fit ensemble gap is still larger
  (`0.156` vs `0.056`), and the within-subject residual fraction increases
  (`0.727` vs `0.435`). The absolute LOSO residual is essentially unchanged
  (`0.275` vs `0.281`), but it is a larger fraction of the available
  cross-subject brain signal (`0.571` vs `0.458`).

- `analysis/03_cross_set_ensemble_transfer.py` writes:
  - `data/ensemble_transfer.csv`
  - `figures/ensemble_transfer_mixed.{pdf,png}`

  This asks whether ensemble weights learned on baseline images rescue the
  diagnostic set. They do not: baseline-trained weights predict the all-model
  diagnostic brain RDM about as well as within-diagnostic CV (`r=0.419` vs
  `r=0.418`), but both remain far below the diagnostic correlation ceiling
  (`sqrt(NC)=0.635`). This argues that the diagnostic gap is not merely a
  failure to choose the right linear mixture of current models.

## Run

```bash
python analysis/01_compute_residual_rsa.py
PYTHONNOUSERSITE=1 python analysis/02_decompose_residual_signal.py
PYTHONNOUSERSITE=1 python analysis/03_cross_set_ensemble_transfer.py --rsa-type mixed
python figures/plot_residual_reliability.py
```

On this machine, plotting may need the conda package stack rather than
user-site packages:

```bash
PYTHONNOUSERSITE=1 python figures/plot_residual_reliability.py
```

For a more stable baseline estimate, increase the Vicco bootstrap count:

```bash
python analysis/01_compute_residual_rsa.py --n-vicco-bootstraps 100
```

To rerun one slice while iterating:

```bash
python analysis/01_compute_residual_rsa.py --rsa-type mixed --groups all_models
```

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Contents Snapshot

- Folder: `05_controls_and_supplementary/reliability_and_residual_structure/residual_reliability`
- Figures in this folder tree: 4
- Data/table-like files in this folder tree: 5
- Python scripts in this folder tree: 4
- Main child folders: `code/`, `data/`, `figures/`
- Direct files: `residual_reliability_report.html`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Index

| Figure directory | Figures | README |
|---|---:|---|
| `05_controls_and_supplementary/reliability_and_residual_structure/residual_reliability/figures` | 1 | `05_controls_and_supplementary/reliability_and_residual_structure/residual_reliability/figures/README.md` |
| `05_controls_and_supplementary/reliability_and_residual_structure/residual_reliability/figures/png` | 1 | `05_controls_and_supplementary/reliability_and_residual_structure/residual_reliability/figures/png/README.md` |
| `05_controls_and_supplementary/reliability_and_residual_structure/residual_reliability/figures/supplementary` | 1 | `05_controls_and_supplementary/reliability_and_residual_structure/residual_reliability/figures/supplementary/README.md` |
| `05_controls_and_supplementary/reliability_and_residual_structure/residual_reliability/figures/supplementary/png` | 1 | `05_controls_and_supplementary/reliability_and_residual_structure/residual_reliability/figures/supplementary/png/README.md` |
<!-- END AUTO-FIGURE-PROVENANCE -->
