# Png

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Figure Descriptions

The notes below summarize what each rendered figure shows. The file table that follows points back to the scripts and inputs used to make it.

**`distance_metric_claim_robustness`.** This robustness plot summarizes whether the main distance-metric
conclusion changes under alternative distance definitions. It uses the adjacent staged data/results files
named in the provenance table to support or qualify claims based on one RSA metric.

**`distance_metric_robustness_improved`.** This robustness plot reruns the alignment analysis with alternative
distance metrics. It uses the adjacent staged data/results files named in the provenance table to check
whether the controversial-stimulus effect depends on the specific RDM distance definition.

**`mixed_distance_metric_robustness`.** This robustness plot reruns the alignment analysis with alternative
distance metrics. It uses the adjacent staged data/results files named in the provenance table to check
whether the controversial-stimulus effect depends on the specific RDM distance definition.

**`model_vs_brain_spread`.** This spread plot compares how dispersed model distances are relative to brain
distances. It uses the adjacent staged data/results files named in the provenance table to diagnose whether
controversial stimuli amplify model variation more than neural variation.

## Contents Snapshot

- Folder: `04_alignment_robustness/figures/png`
- Figures in this folder tree: 4
- Data/table-like files in this folder tree: 0
- Python scripts in this folder tree: 0
- Direct files: `distance_metric_claim_robustness.png`, `distance_metric_robustness_improved.png`, `mixed_distance_metric_robustness.png`, `model_vs_brain_spread.png`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Provenance

Each row is one figure concept; `formats` lists the concrete files present in this folder.

| Figure | Formats | What it shows | Source / derivation | Resources or results | Script | Paper use |
|---|---:|---|---|---|---|---|
| `distance_metric_claim_robustness` | png | figure derived from distance metric claim robustness | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/03_statistics/figures/distance_metric_claim_robustness.png`; historical share path `data/tier1_analysis_derivatives/cstim_paper/03_statistics/figures/distance_metric_claim_robustness.png` | `distance_metric_claim_robustness.csv`; `distance_metric_claim_robustness_summary.csv`; `DATA_DIR`; `FIGURES_DIR`; `PNG_DIR` | `04_alignment_robustness/code/figures/03_plot_distance_metric_claim_robustness.py` | no |
| `distance_metric_robustness_improved` | png | robustness of alignment conclusions to distance metric choice | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/03_statistics/figures/distance_metric_robustness_improved.png`; historical share path `data/tier1_analysis_derivatives/cstim_paper/03_statistics/figures/distance_metric_robustness_improved.png` | `distance_metric_robustness.csv`; `DATA_DIR`; `FIGURES_DIR`; `PNG_DIR` | `04_alignment_robustness/code/figures/01_plot_fixed_distance_metric_robustness.py` | no |
| `mixed_distance_metric_robustness` | png | robustness of alignment conclusions to distance metric choice | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/03_statistics/figures/mixed_distance_metric_robustness.png`; historical share path `data/tier1_analysis_derivatives/cstim_paper/03_statistics/figures/mixed_distance_metric_robustness.png` | `mixed_distance_metric_robustness.csv`; `mixed_distance_metric_rank_summary.csv`; `DATA_DIR`; `FIGURES_DIR`; `PNG_DIR` | `04_alignment_robustness/code/figures/02_plot_mixed_distance_metric_robustness.py` | no |
| `model_vs_brain_spread` | png | figure derived from model vs brain spread | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/03_statistics/figures/model_vs_brain_spread.png`; historical share path `data/tier1_analysis_derivatives/cstim_paper/03_statistics/figures/model_vs_brain_spread.png` | `02_alignment_reliability/results/rdm_noise_ceilings.csv`; `model_rdm_spreads.csv`; `FIGURES_DIR`; `PNG_DIR` | `04_alignment_robustness/code/figures/04_plot_model_vs_brain_spread.py` | no |
<!-- END AUTO-FIGURE-PROVENANCE -->
