# 08 — Image statistics

Do low-level image statistics differ between controversial stimulus sets
(`all_models`, `architecture`, `dataset`, `sota`, `training_objective`) and
the baseline set (`vicco`)? If so, they might (partly) explain the
brain-alignment drop observed in
`02_rsa_scores/04_compute_rsa_large_benchmark.py`.

## Stats computed (per image)

- **Luminance**: mean of grayscale
- **Contrast**: RMS contrast of grayscale
- **Color mean / std**: per-channel mean and std (R, G, B)
- **Saturation**: mean HSV S-channel
- **Spatial frequency**:
  - `sf_slope` — slope of log-log radial FFT power spectrum (1/f slope)
  - `sf_high_low_ratio` — energy in high SFs / low SFs (radial-bin split at median)
- **Edge density**: mean Sobel magnitude
- **Complexity**:
  - `entropy` — Shannon entropy of grayscale histogram
  - `jpeg_ratio` — JPEG(Q=75) bytes / raw pixel bytes

## Run

```
python 01_compute_image_stats.py                       # vicco + 5 controversial sets -> data/image_stats.csv
python 03_compute_training_image_stats.py              # LAION-fMRI shared (encoder training) -> appended
python 02_low_level_rdm_brain_alignment.py             # pixel/stats RDM ↔ brain RDM per set
python figures/plot_stat_distributions.py              # marginal distributions + permutation-test stars vs training
python figures/plot_low_level_brain_alignment.py       # low-level RDM ↔ brain RDM bar chart
```

### Reference distributions
- **LAION-fMRI shared** (`deepvision_train`, n=1492) — the DeepVision images used to fit the encoders that drove selection. Any drift of controversial sets from this distribution indicates the selection process pulled images out of the training-image prior.
- **Baseline** (`vicco`, n=292) — a neutral image pool used during cstim acquisition; acts as a control to check that drift is selection-induced, not a property of all non-training images.

## Analyses

1. **Marginal stat distributions** (`01` → `plot_stat_distributions`) — do the
   stimulus sets differ in basic image properties?
2. **Low-level RDM → brain alignment** (`02` → `plot_low_level_brain_alignment`) —
   does a similarity structure built purely from low-level features explain
   the brain RDM, and does that alignment drop for controversial sets?

A "correlate-stats-with-RSA-drop" analysis isn't run because there are only
6 stimulus sets — no statistical power. Analysis 2 is the set-level test of
whether low-level structure could be driving the drop at all.

## Findings (auto-updated by `02` script printout)

Brain-RDM alignment of low-level RDMs is uniformly small (|r| < 0.06) for
every stimulus set, including vicco. Low-level image structure does not
track the model-RSA drop seen in `02_rsa_scores`; the drop is model-specific.

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Contents Snapshot

- Folder: `05_controls_and_supplementary/low_level_and_ood/image_statistics`
- Figures in this folder tree: 2
- Data/table-like files in this folder tree: 2
- Python scripts in this folder tree: 6
- Main child folders: `code/`, `data/`, `figures/`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Index

| Figure directory | Figures | README |
|---|---:|---|
| `05_controls_and_supplementary/low_level_and_ood/image_statistics/figures` | 1 | `05_controls_and_supplementary/low_level_and_ood/image_statistics/figures/README.md` |
| `05_controls_and_supplementary/low_level_and_ood/image_statistics/figures/png` | 1 | `05_controls_and_supplementary/low_level_and_ood/image_statistics/figures/png/README.md` |
<!-- END AUTO-FIGURE-PROVENANCE -->
