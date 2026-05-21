# External Data

This directory is the expected mount/link location for raw or very large inputs
that are intentionally not included in the share package.

Expected external roots:

- `deepvision_fmri/`: raw fMRI, GLMsingle, and atlas files.
- `final_cstims_hdf5_files/`: raw selected-stimulus HDF5/image files, if
  rerunning workflows that need the original HDF5 payloads.
- `cstims_laion_natural_subset/`: optional LAION-style tar shards for figure
  examples or stimulus-pool reruns.

The same locations can be overridden with environment variables:

- `CSTIMS_DEEPVISION_FMRI_ROOT`
- `CSTIMS_CSTIM_HDF5_ROOT`
- `CSTIMS_LAION_ROOT`
