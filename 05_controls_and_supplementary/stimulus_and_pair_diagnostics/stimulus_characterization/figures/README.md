# Figures

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Figure Descriptions

The notes below summarize what each rendered figure shows. The file table that follows points back to the scripts and inputs used to make it.

**`high_spread_pairs_consistent`.** This stimulus-characterization figure shows high-spread or
high-disagreement image pairs that are consistent across subjects. It uses the stimulus-characterization and
semantic-clustering tables to make the abstract pair-level brain-placement result visually inspectable.

**`semantic_embedding_pool_clustering`.** This semantic-control figure embeds selected and pool images in a
semantic feature space and shows their cluster structure. It uses the stimulus-characterization and
semantic-clustering tables to check whether controversial stimuli occupy unusual semantic/content regions.

**`stimulus_grids_by_model_set`.** This stimulus grid displays selected images grouped by model-set target. It
uses the stimulus-characterization and semantic-clustering tables to let readers inspect what kinds of images
the selection procedure chose for each target.

**`stimulus_overview`.** This overview grid shows representative selected controversial and baseline stimuli.
It uses the stimulus-characterization and semantic-clustering tables as a visual sanity check on the image
content behind the quantitative RSA results.

## Contents Snapshot

- Folder: `05_controls_and_supplementary/stimulus_and_pair_diagnostics/stimulus_characterization/figures`
- Figures in this folder tree: 4
- Data/table-like files in this folder tree: 0
- Python scripts in this folder tree: 0
- Direct files: `high_spread_pairs_consistent.pdf`, `semantic_embedding_pool_clustering.pdf`, `stimulus_grids_by_model_set.pdf`, `stimulus_overview.pdf`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Provenance

Each row is one figure concept; `formats` lists the concrete files present in this folder.

| Figure | Formats | What it shows | Source / derivation | Resources or results | Script | Paper use |
|---|---:|---|---|---|---|---|
| `high_spread_pairs_consistent` | pdf | high-disagreement image pairs with subject-consistent brain placement | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/figures/appendix_h/high_spread_pairs_consistent.pdf` | `cstim_betas_averaged.npz`; `voxel_metadata.npz`; `cstim_stimulus_info.csv`; `per_pair_spread.csv`; `SHARE_ROOT`; `CSTIM_DIR`; plus 9 more | `05_controls_and_supplementary/stimulus_and_pair_diagnostics/stimulus_characterization/code/figures/create_stimulus_grids.py`; `06_manuscript/paper_figures/source/appendix_h/create_stimulus_grids.py` | yes |
| `semantic_embedding_pool_clustering` | pdf | semantic-embedding clustering of selected and pool images | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/14_stimulus_characterization/figures/semantic_embedding_pool_clustering.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/14_stimulus_characterization/figures/semantic_embedding_pool_clustering.pdf` | `{MODEL}.npz`; `data.pkl`; `semantic_embedding_pool_sample.csv`; `semantic_embedding_cluster_summary.csv`; `semantic_embedding_cluster_model_set_summary.csv`; `semantic_embedding_thumbnail_representatives.csv`; plus 10 more | `05_controls_and_supplementary/stimulus_and_pair_diagnostics/stimulus_characterization/code/02_plot_semantic_embedding_clustering.py` | yes |
| `stimulus_grids_by_model_set` | pdf | stimulus-grid overview by model set | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/figures/appendix_h/stimulus_grids_by_model_set.pdf` | `SHARE_ROOT`; `CSTIM_DIR`; `CSTIM_HDF5_ROOT`; `FIG_DIR`; `05_controls_and_supplementary/stimulus_and_pair_diagnostics/stimulus_characterization/data/content_cluster_summary.csv`; `05_controls_and_supplementary/stimulus_and_pair_diagnostics/stimulus_characterization/data/semantic_embedding_cluster_model_set_summary.csv`; plus 4 more | `05_controls_and_supplementary/stimulus_and_pair_diagnostics/stimulus_characterization/code/figures/create_model_set_grids.py`; `06_manuscript/paper_figures/source/appendix_h/create_model_set_grids.py` | yes |
| `stimulus_overview` | pdf | overview grid of selected controversial and baseline stimuli | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/figures/appendix_h/stimulus_overview.pdf` | `cstim_betas_averaged.npz`; `voxel_metadata.npz`; `cstim_stimulus_info.csv`; `per_pair_spread.csv`; `SHARE_ROOT`; `CSTIM_DIR`; plus 9 more | `05_controls_and_supplementary/stimulus_and_pair_diagnostics/stimulus_characterization/code/figures/create_stimulus_grids.py`; `06_manuscript/paper_figures/source/appendix_h/create_stimulus_grids.py` | yes |
<!-- END AUTO-FIGURE-PROVENANCE -->
