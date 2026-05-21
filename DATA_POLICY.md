# Data Inclusion Policy

This repository is intended to keep the reorganized analysis code, documentation,
figures, selected-stimulus metadata, and result tables together.

Included:

- Source code, configs, READMEs, manuscript-facing documentation, and figures.
- Result CSV/TSV files, including intermediate analysis tables. CSV/TSV files
  are tracked through Git LFS so large result tables can be uploaded.
- Selected-stimulus manifests, selection metadata, checkpoints, and non-image
  outputs.
- Encoding-model metadata and figures where useful, but not the model payloads.

Excluded:

- Raw or derived beta matrices, voxel beta tables, voxel metadata, and brain
  data caches.
- Encoding model payloads named `encoding_model.npz`.
- Feature caches, scratch outputs, temporary files, and logs.
- Selected-stimulus image payloads under `images/` and `validated_images/`.
- Stimulus-characterization figures that embed selected image thumbnails.
- External data mounts under `external_data/`, except for their README.

The manuscript directory is versioned separately and is represented from this
repository as a Git submodule.
