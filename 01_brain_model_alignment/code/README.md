# Brain-Model Alignment Code

This folder contains the scripts that prepare brain inputs, fit brain-encoding
models, and compute RSA alignment scores.

## Main Subfolders

- `brain_data_preparation/`: builds subject-level cstim brain caches from raw
  fMRI derivatives. The cache files used by scoring are
  `cstim_betas_averaged.npz`, `cstim_betas_by_rep.npz`,
  `cstim_stimulus_info.csv`, and `voxel_metadata.npz`.
- `encoding_model_fitting/`: fits the ridge encoding models used for
  brain-predicted feature tracks and mixed/weighted RSA.
- `rsa_scoring/`: computes the cstim brain-alignment score tables in
  `../results/rsa_scores/`.

For the common question "where are the cstim brain-alignment scores computed?",
start in `rsa_scoring/README.md`.

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Contents Snapshot

- Folder: `01_brain_model_alignment/code`
- Figures in this folder tree: 0
- Data/table-like files in this folder tree: 0
- Python scripts in this folder tree: 30

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.
<!-- END AUTO-FIGURE-PROVENANCE -->
