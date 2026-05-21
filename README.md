# Natural controversial stimuli

This repository is organized as a staged analysis share for the natural
controversial-stimulus project. The main question is how images chosen because
vision models disagree on them affect brain-model alignment analyses, and
whether the resulting effects survive reliability checks, statistical
inference, robustness tests, and controls.

Here, "controversial" means controversial for models: the selected natural
images are intended to make different vision models predict different
representational similarity structure, while remaining usable under realistic
brain-measurement noise.

## Mini Guide

Read the numbered folders as a pipeline:

1. Choose the stimulus sets in `00_stimulus_selection/`.
2. Compute brain-model alignment scores in `01_brain_model_alignment/`.
3. Check whether the alignment scores are reliable in
   `02_alignment_reliability/`.
4. Run the main statistical tests in `03_alignment_inference/`.
5. Stress-test the conclusions in `04_alignment_robustness/`.
6. Collect controls and supplementary follow-ups in
   `05_controls_and_supplementary/`.
7. Copy the final staged outputs into the paper build in `06_manuscript/`.

Within a numbered folder, the usual convention is:

- `code/`: scripts that generate that stage's outputs.
- `data/`, `rsa_scores/`, `selected_stimuli/`, or `inputs/`: staged inputs and
  tabular outputs used by later stages.
- `figures/`: rendered analysis figures, with `png/` folders holding raster
  copies where present.
- Local `README.md` files: plain-language notes for what that folder contains
  and, where useful, tables pointing from figures back to scripts and inputs.

Raw external datasets and heavyweight intermediates are not all shipped here.
Those paths are documented in `external_data/`, `shared/`, and the relevant
stage README files.

## Method Capsule

The stimulus-selection objective is based on representational similarity
analysis (RSA):

1. For each model, compute a representational dissimilarity matrix (RDM) over
   candidate images, usually with cosine distance.
2. Calibrate the expected measurement noise against a target noise ceiling
   (`noise_ceiling_target: 0.46` in the shipped configs).
3. Estimate how much a candidate image would increase model separability after
   accounting for that noise.
4. Prefer images where each model is internally consistent but disagrees with
   the other models.
5. Combine evidence across raw feature tracks and brain-encoding tracks. The
   default `raw_plus_all_encodings` method gives half the weight to raw model
   features and half to the subject-specific brain-encoding tracks.

The selection implementation is in `src/cstims/selection/`, with the frozen
selection runs and their configs copied into `00_stimulus_selection/`.

## Folder Map

1. `00_stimulus_selection/` - selected stimuli, selection code, configs,
   resources, and selection decision checks.
2. `01_brain_model_alignment/` - brain-data preparation, encoding models, RSA
   analyses, and alignment score tables.
3. `02_alignment_reliability/` - reliability and noise-ceiling analyses.
4. `03_alignment_inference/` - primary alignment inference, permutation tests,
   bootstrap summaries, and canonical tables.
5. `04_alignment_robustness/` - distance-metric and spread robustness checks.
6. `05_controls_and_supplementary/` - controls, supplementary analyses, and
   follow-up validations.
7. `06_manuscript/` - manuscript source and final paper figures.
8. `src/cstims/` - installable project package.
9. `shared/` - shared helper code and copied heavy derivatives used across
   stages.
10. `external_data/` - documented mount/link location for raw external inputs
   that are not shipped.

## Runtime Notes

- `pyproject.toml` defines the lightweight installable package and optional
  full analysis dependencies.
- `Makefile` contains quick checks such as `make smoke` and
  `make compile-core`.
- `external_data/README.md` documents raw or very large inputs that are expected
  to be mounted or linked locally rather than included in this share.
