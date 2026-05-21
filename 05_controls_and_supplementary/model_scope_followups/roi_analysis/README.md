# ROI Analysis

This directory contains the active ROI analyses for the CSTIMS paper.

## Full Visual-Cortex Parcel Analysis

`03_visual_parcel_mixed_rsa.py` extends the primary high-level visual
(`hlvis`) analysis to HCP-MMP parcels inside each subject's full
`visual_mask`. It does not refit encoders. Instead, it uses the existing
unique-image encoding models, whose weights cover the full visual mask, to
predict voxel patterns for the controversial images and the same-session
baseline (`vicco`) images.

For each subject, model, model set, and ROI, the script computes mixed RSA as
the Spearman correlation between the model-predicted voxel-pattern RDM and the
observed fMRI voxel-pattern RDM. Baseline scores use 10 deterministic bootstrap
samples of 100 `vicco` images. Endpoint summaries average models within each
subject x model-set x ROI, yielding 25 endpoints for each ROI summary
(5 subjects x 5 model sets).

ROIs include individual HCP-MMP parcels plus grouped visual regions:

- Early visual: V1, V2, V3
- Extended early/dorsal: V3A, V3B, V3CD, V6, V6A, V7, IPS1
- Motion: MT, MST, FST
- Mid-level/lateral: V4, V4t, V8, LO1, LO2, LO3, PIT
- Medial scene: RSC, POS1, POS2
- Ventral object/scene: FFC, VVC, PHA1-3, VMV1-3, PH, PHT, TF, TE2p

The analysis also writes aggregate rows for the full visual mask and `hlvis`.

## Outputs

- `data/visual_parcel_mixed_rsa.csv`: long-form model x ROI x stimulus scores.
- `data/visual_parcel_endpoint_summary.csv`: subject x model-set x ROI endpoints,
  including controversial-baseline delta, relative delta, and model-spread ratio.
- `data/visual_parcel_metadata.csv`: subject-specific ROI voxel counts and atlas
  assignment diagnostics.
- `data/visual_group_summary.csv`: plotted group/aggregate means and +/-1.96 SEM.
- `data/visual_group_comparisons.csv`: paired endpoint tests against early visual
  and high-level visual references. Holm correction is applied within each
  metric x reference family.
- `figures/visual_parcel_roi_summary.pdf` and `.png`: main visual ROI figure.

## Figure Caption Draft

Full visual-cortex ROI analysis. Mixed-RSA deltas compare controversial images
with same-session baseline images after applying the existing unique-image
encoding models to full visual-cortex voxel weights. Light points show
individual subject x model-set endpoints after averaging models within each set
(25 endpoints per ROI); large points show endpoint means and horizontal
intervals show +/-1.96 SEM. Rows show aggregate full visual and high-level
visual masks followed by HCP-MMP parcel groups inside the visual mask. Panel a:
absolute mixed-RSA delta. Panel b: delta as a fraction of baseline mixed RSA.
Panel c: log2 ratio of controversial to baseline model spread, where spread is
the median pairwise difference among model scores within an endpoint.

## Headline Results

The controversial-image alignment drop is much smaller in early visual cortex
than in high-level/ventral regions. Mean absolute delta is -0.011 in early
visual cortex versus -0.137 in `hlvis` and -0.133 in ventral object/scene.
Paired endpoint comparisons confirm that early visual differs from `hlvis`
(delta difference = 0.126 +/- 0.033 CI95, Holm p = 7.97e-7) and from every
other grouped visual ROI on absolute delta.

| ROI | Delta mean | Delta CI95 | Relative delta mean | Relative CI95 |
|---|---:|---:|---:|---:|
| Full visual | -0.074 | 0.020 | -28.9% | 6.9% |
| High-level visual | -0.137 | 0.030 | -29.7% | 6.1% |
| Early visual | -0.011 | 0.014 | -8.3% | 12.0% |
| Extended early/dorsal | -0.063 | 0.023 | -26.9% | 10.3% |
| Motion | -0.083 | 0.029 | -25.6% | 8.6% |
| Mid-level/lateral | -0.095 | 0.023 | -29.4% | 6.4% |
| Medial scene | -0.086 | 0.026 | -37.4% | 14.2% |
| Ventral object/scene | -0.133 | 0.032 | -28.8% | 6.5% |

`hlvis` and ventral object/scene are not distinguishable on absolute delta in
this endpoint test (mean difference = 0.004, Holm p = 0.425), which is
consistent with the original high-level visual result being driven mainly by
ventral/object-scene cortex rather than early visual cortex.

## Reproducing

```bash
python -u experiments/cstim_paper/13_roi_analysis/03_visual_parcel_mixed_rsa.py --overwrite
/home/jroth/.conda/envs/edopt/bin/python experiments/cstim_paper/13_roi_analysis/04_plot_visual_parcel_results.py
```

The plotting command uses the `edopt` environment because the base Python
environment currently has a Matplotlib/NumPy binary compatibility mismatch.

## Legacy Summary

`01_build_roi_summary.py` creates `data/roi_results.csv` from available ROI
outputs. The original CSTIMS brain-data cache exposed `visual_mask` and
`hlvis_mask` but not local parcel-level masks. The full visual-cortex analysis
above now uses the subject-level HCP-MMP atlas under the DeepVision derivatives
directory to build parcel masks in visual-voxel coordinates.

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Contents Snapshot

- Folder: `05_controls_and_supplementary/model_scope_followups/roi_analysis`
- Figures in this folder tree: 2
- Data/table-like files in this folder tree: 7
- Python scripts in this folder tree: 5
- Main child folders: `code/`, `data/`, `figures/`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Index

| Figure directory | Figures | README |
|---|---:|---|
| `05_controls_and_supplementary/model_scope_followups/roi_analysis/figures` | 1 | `05_controls_and_supplementary/model_scope_followups/roi_analysis/figures/README.md` |
| `05_controls_and_supplementary/model_scope_followups/roi_analysis/figures/png` | 1 | `05_controls_and_supplementary/model_scope_followups/roi_analysis/figures/png/README.md` |
<!-- END AUTO-FIGURE-PROVENANCE -->
