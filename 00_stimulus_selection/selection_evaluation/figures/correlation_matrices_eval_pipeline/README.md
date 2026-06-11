# Correlation Matrices Eval Pipeline

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Figure Descriptions

The notes below summarize what each rendered figure shows. The file table that follows points back to the scripts and inputs used to make it.

**`correlation_matrices_all_models_raw_legacy3_improved`.** This heatmap is a model-RDM correlation sanity
check for the all-model grouping in raw feature space. It compares the selected stimuli with
legacy/evaluation-pipeline baseline, using the frozen selection-evaluation CSVs and model-RDM summaries. Read
the diagonal versus off-diagonal structure as evidence for whether the stimulus set separates the intended
model family rather than collapsing models into one shared geometry.

**`correlation_matrices_all_models_sub-01_legacy3_improved`.** This heatmap is a model-RDM correlation sanity
check for the all-model grouping in sub-01 encoding/predicted-brain space. It compares the selected stimuli
with legacy/evaluation-pipeline baseline, using the frozen selection-evaluation CSVs and model-RDM summaries.
Read the diagonal versus off-diagonal structure as evidence for whether the stimulus set separates the
intended model family rather than collapsing models into one shared geometry.

**`correlation_matrices_architecture_raw_legacy3_improved`.** This heatmap is a model-RDM correlation sanity
check for the architecture grouping in raw feature space. It compares the selected stimuli with
legacy/evaluation-pipeline baseline, using the frozen selection-evaluation CSVs and model-RDM summaries. Read
the diagonal versus off-diagonal structure as evidence for whether the stimulus set separates the intended
model family rather than collapsing models into one shared geometry.

**`correlation_matrices_architecture_sub-01_legacy3_improved`.** This heatmap is a model-RDM correlation
sanity check for the architecture grouping in sub-01 encoding/predicted-brain space. It compares the selected
stimuli with legacy/evaluation-pipeline baseline, using the frozen selection-evaluation CSVs and model-RDM
summaries. Read the diagonal versus off-diagonal structure as evidence for whether the stimulus set separates
the intended model family rather than collapsing models into one shared geometry.

**`correlation_matrices_dataset_raw_legacy3_improved`.** This heatmap is a model-RDM correlation sanity check
for the dataset grouping in raw feature space. It compares the selected stimuli with
legacy/evaluation-pipeline baseline, using the frozen selection-evaluation CSVs and model-RDM summaries. Read
the diagonal versus off-diagonal structure as evidence for whether the stimulus set separates the intended
model family rather than collapsing models into one shared geometry.

**`correlation_matrices_dataset_sub-01_legacy3_improved`.** This heatmap is a model-RDM correlation sanity
check for the dataset grouping in sub-01 encoding/predicted-brain space. It compares the selected stimuli with
legacy/evaluation-pipeline baseline, using the frozen selection-evaluation CSVs and model-RDM summaries. Read
the diagonal versus off-diagonal structure as evidence for whether the stimulus set separates the intended
model family rather than collapsing models into one shared geometry.

**`correlation_matrices_sota_raw_legacy3_improved`.** This heatmap is a model-RDM correlation sanity check for
the state-of-the-art grouping in raw feature space. It compares the selected stimuli with
legacy/evaluation-pipeline baseline, using the frozen selection-evaluation CSVs and model-RDM summaries. Read
the diagonal versus off-diagonal structure as evidence for whether the stimulus set separates the intended
model family rather than collapsing models into one shared geometry.

**`correlation_matrices_sota_sub-01_legacy3_improved`.** This heatmap is a model-RDM correlation sanity check
for the state-of-the-art grouping in sub-01 encoding/predicted-brain space. It compares the selected stimuli
with legacy/evaluation-pipeline baseline, using the frozen selection-evaluation CSVs and model-RDM summaries.
Read the diagonal versus off-diagonal structure as evidence for whether the stimulus set separates the
intended model family rather than collapsing models into one shared geometry.

**`correlation_matrices_training_objective_raw_legacy3_improved`.** This heatmap is a model-RDM correlation
sanity check for the training-objective grouping in raw feature space. It compares the selected stimuli with
legacy/evaluation-pipeline baseline, using the frozen selection-evaluation CSVs and model-RDM summaries. Read
the diagonal versus off-diagonal structure as evidence for whether the stimulus set separates the intended
model family rather than collapsing models into one shared geometry.

**`correlation_matrices_training_objective_sub-01_legacy3_improved`.** This heatmap is a model-RDM correlation
sanity check for the training-objective grouping in sub-01 encoding/predicted-brain space. It compares the
selected stimuli with legacy/evaluation-pipeline baseline, using the frozen selection-evaluation CSVs and
model-RDM summaries. Read the diagonal versus off-diagonal structure as evidence for whether the stimulus set
separates the intended model family rather than collapsing models into one shared geometry.

## Contents Snapshot

- Folder: `00_stimulus_selection/selection_evaluation/figures/correlation_matrices_eval_pipeline`
- Figures in this folder tree: 10
- Data/table-like files in this folder tree: 0
- Python scripts in this folder tree: 0
- Direct files: `correlation_matrices_all_models_raw_legacy3_improved.pdf`, `correlation_matrices_all_models_sub-01_legacy3_improved.pdf`, `correlation_matrices_architecture_raw_legacy3_improved.pdf`, `correlation_matrices_architecture_sub-01_legacy3_improved.pdf`, `correlation_matrices_dataset_raw_legacy3_improved.pdf`, `correlation_matrices_dataset_sub-01_legacy3_improved.pdf`, `correlation_matrices_sota_raw_legacy3_improved.pdf`, `correlation_matrices_sota_sub-01_legacy3_improved.pdf`, `correlation_matrices_training_objective_raw_legacy3_improved.pdf`, `correlation_matrices_training_objective_sub-01_legacy3_improved.pdf`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Provenance

Each row is one figure concept; `formats` lists the concrete files present in this folder.

| Figure | Formats | What it shows | Source / derivation | Resources or results | Script | Paper use |
|---|---:|---|---|---|---|---|
| `correlation_matrices_all_models_raw_legacy3_improved` | pdf | model-by-model RDM correlation matrix for the selected images and a reference baseline | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation/figures/correlation_matrices_eval_pipeline/correlation_matrices_all_models_raw_legacy3_improved.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/00_selection_evaluation/figures/correlation_matrices_eval_pipeline/correlation_matrices_all_models_raw_legacy3_improved.pdf` | `selection_evaluation/data/*/correlation_matrices.csv`; `{which}_discriminability_{metric}_{corr_type}.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/ablations.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/correlation_matrices.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/discriminability.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/diversity.csv`; plus 2 more | `src/cstims/evaluation/plotting.py`; `src/cstims/evaluation/core.py` | no |
| `correlation_matrices_all_models_sub-01_legacy3_improved` | pdf | model-by-model RDM correlation matrix for the selected images and a reference baseline | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation/figures/correlation_matrices_eval_pipeline/correlation_matrices_all_models_sub-01_legacy3_improved.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/00_selection_evaluation/figures/correlation_matrices_eval_pipeline/correlation_matrices_all_models_sub-01_legacy3_improved.pdf` | `selection_evaluation/data/*/correlation_matrices.csv`; `{which}_discriminability_{metric}_{corr_type}.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/ablations.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/correlation_matrices.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/discriminability.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/diversity.csv`; plus 2 more | `src/cstims/evaluation/plotting.py`; `src/cstims/evaluation/core.py` | no |
| `correlation_matrices_architecture_raw_legacy3_improved` | pdf | model-by-model RDM correlation matrix for the selected images and a reference baseline | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation/figures/correlation_matrices_eval_pipeline/correlation_matrices_architecture_raw_legacy3_improved.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/00_selection_evaluation/figures/correlation_matrices_eval_pipeline/correlation_matrices_architecture_raw_legacy3_improved.pdf` | `selection_evaluation/data/*/correlation_matrices.csv`; `{which}_discriminability_{metric}_{corr_type}.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/ablations.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/correlation_matrices.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/discriminability.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/diversity.csv`; plus 2 more | `src/cstims/evaluation/plotting.py`; `src/cstims/evaluation/core.py` | no |
| `correlation_matrices_architecture_sub-01_legacy3_improved` | pdf | model-by-model RDM correlation matrix for the selected images and a reference baseline | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation/figures/correlation_matrices_eval_pipeline/correlation_matrices_architecture_sub-01_legacy3_improved.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/00_selection_evaluation/figures/correlation_matrices_eval_pipeline/correlation_matrices_architecture_sub-01_legacy3_improved.pdf` | `selection_evaluation/data/*/correlation_matrices.csv`; `{which}_discriminability_{metric}_{corr_type}.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/ablations.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/correlation_matrices.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/discriminability.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/diversity.csv`; plus 2 more | `src/cstims/evaluation/plotting.py`; `src/cstims/evaluation/core.py` | no |
| `correlation_matrices_dataset_raw_legacy3_improved` | pdf | model-by-model RDM correlation matrix for the selected images and a reference baseline | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation/figures/correlation_matrices_eval_pipeline/correlation_matrices_dataset_raw_legacy3_improved.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/00_selection_evaluation/figures/correlation_matrices_eval_pipeline/correlation_matrices_dataset_raw_legacy3_improved.pdf` | `selection_evaluation/data/*/correlation_matrices.csv`; `{which}_discriminability_{metric}_{corr_type}.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/ablations.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/correlation_matrices.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/discriminability.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/diversity.csv`; plus 2 more | `src/cstims/evaluation/plotting.py`; `src/cstims/evaluation/core.py` | no |
| `correlation_matrices_dataset_sub-01_legacy3_improved` | pdf | model-by-model RDM correlation matrix for the selected images and a reference baseline | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation/figures/correlation_matrices_eval_pipeline/correlation_matrices_dataset_sub-01_legacy3_improved.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/00_selection_evaluation/figures/correlation_matrices_eval_pipeline/correlation_matrices_dataset_sub-01_legacy3_improved.pdf` | `selection_evaluation/data/*/correlation_matrices.csv`; `{which}_discriminability_{metric}_{corr_type}.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/ablations.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/correlation_matrices.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/discriminability.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/diversity.csv`; plus 2 more | `src/cstims/evaluation/plotting.py`; `src/cstims/evaluation/core.py` | no |
| `correlation_matrices_sota_raw_legacy3_improved` | pdf | model-by-model RDM correlation matrix for the selected images and a reference baseline | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation/figures/correlation_matrices_eval_pipeline/correlation_matrices_sota_raw_legacy3_improved.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/00_selection_evaluation/figures/correlation_matrices_eval_pipeline/correlation_matrices_sota_raw_legacy3_improved.pdf` | `selection_evaluation/data/*/correlation_matrices.csv`; `{which}_discriminability_{metric}_{corr_type}.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/ablations.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/correlation_matrices.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/discriminability.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/diversity.csv`; plus 2 more | `src/cstims/evaluation/plotting.py`; `src/cstims/evaluation/core.py` | no |
| `correlation_matrices_sota_sub-01_legacy3_improved` | pdf | model-by-model RDM correlation matrix for the selected images and a reference baseline | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation/figures/correlation_matrices_eval_pipeline/correlation_matrices_sota_sub-01_legacy3_improved.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/00_selection_evaluation/figures/correlation_matrices_eval_pipeline/correlation_matrices_sota_sub-01_legacy3_improved.pdf` | `selection_evaluation/data/*/correlation_matrices.csv`; `{which}_discriminability_{metric}_{corr_type}.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/ablations.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/correlation_matrices.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/discriminability.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/diversity.csv`; plus 2 more | `src/cstims/evaluation/plotting.py`; `src/cstims/evaluation/core.py` | no |
| `correlation_matrices_training_objective_raw_legacy3_improved` | pdf | model-by-model RDM correlation matrix for the selected images and a reference baseline | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation/figures/correlation_matrices_eval_pipeline/correlation_matrices_training_objective_raw_legacy3_improved.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/00_selection_evaluation/figures/correlation_matrices_eval_pipeline/correlation_matrices_training_objective_raw_legacy3_improved.pdf` | `selection_evaluation/data/*/correlation_matrices.csv`; `{which}_discriminability_{metric}_{corr_type}.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/ablations.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/correlation_matrices.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/discriminability.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/diversity.csv`; plus 2 more | `src/cstims/evaluation/plotting.py`; `src/cstims/evaluation/core.py` | no |
| `correlation_matrices_training_objective_sub-01_legacy3_improved` | pdf | model-by-model RDM correlation matrix for the selected images and a reference baseline | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation/figures/correlation_matrices_eval_pipeline/correlation_matrices_training_objective_sub-01_legacy3_improved.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/00_selection_evaluation/figures/correlation_matrices_eval_pipeline/correlation_matrices_training_objective_sub-01_legacy3_improved.pdf` | `selection_evaluation/data/*/correlation_matrices.csv`; `{which}_discriminability_{metric}_{corr_type}.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/ablations.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/correlation_matrices.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/discriminability.csv`; `00_stimulus_selection/selection_evaluation/results/all_models/diversity.csv`; plus 2 more | `src/cstims/evaluation/plotting.py`; `src/cstims/evaluation/core.py` | no |
<!-- END AUTO-FIGURE-PROVENANCE -->
