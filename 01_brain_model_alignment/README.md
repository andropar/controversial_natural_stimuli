# 01 Brain-Model Alignment

This stage contains the infrastructure and outputs needed to compute
brain-model alignment.

The practical purpose of this folder is to turn the selected images and brain
responses into model-by-brain alignment scores. It contains the brain-cache
preparation code, fitted encoding-model inputs, RSA scoring scripts, and the
score tables consumed by later reliability and inference stages.

- `code/brain_data_preparation/`: brain-data cache/preparation scripts.
- `code/encoding_model_fitting/`: encoding-model fitting scripts.
- `code/rsa_scoring/`: fixed RSA, mixed RSA, transfer, and benchmark scripts.
- `results/encoding_models/`: fitted encoding-model metadata and summary
  outputs, excluding the heavyweight model payloads.
- `results/rsa_scores/`: core alignment score tables.
- `figures/`: alignment figures.
- `cache_or_heavy/`: optional heavy rerun inputs.

## How To Read This Stage

Use `results/rsa_scores/` for the canonical alignment tables. Use `code/rsa_scoring/`
to see how those tables were computed, and `results/encoding_models/` when you
need the fitted subject/model metadata behind the scoring. The `figures/`
folder is a rendered view of those same score outputs, not a separate analysis
source.

## Encoding Models

The brain-encoding models used by selection and alignment learn linear mappings
from model features to voxel responses with ridge regression. The fitting code
is staged in `code/encoding_model_fitting/`, and reusable package helpers live
under `src/cstims/encoding/`.

The standard convention inherited from the source workflow is to fit on the
visual ROI, evaluate/select high-level visual cortex (`hlvis`) where configured,
and use per-voxel ridge regularization selected across a log-spaced alpha grid.
The fitted encoders are then used as brain-predicted feature tracks for
stimulus selection and as inputs to downstream RSA scoring.

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Contents Snapshot

- Folder: `01_brain_model_alignment`
- Figures in this folder tree: 26
- Data/table-like files in this folder tree: 835
- Python scripts in this folder tree: 31
- Main child folders: `code/`, `figures/`, `results/`, `cache_or_heavy/`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Index

| Figure directory | Figures | README |
|---|---:|---|
| `01_brain_model_alignment/figures/rsa_scores` | 6 | `01_brain_model_alignment/figures/rsa_scores/README.md` |
| `01_brain_model_alignment/figures/rsa_scores/png` | 6 | `01_brain_model_alignment/figures/rsa_scores/png/README.md` |
| `01_brain_model_alignment/figures/rsa_scores/supplementary` | 7 | `01_brain_model_alignment/figures/rsa_scores/supplementary/README.md` |
| `01_brain_model_alignment/figures/rsa_scores/supplementary/png` | 7 | `01_brain_model_alignment/figures/rsa_scores/supplementary/png/README.md` |
<!-- END AUTO-FIGURE-PROVENANCE -->
