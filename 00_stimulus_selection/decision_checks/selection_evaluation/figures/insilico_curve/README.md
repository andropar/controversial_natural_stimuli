# Insilico Curve

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Figure Descriptions

The notes below summarize what each rendered figure shows. The file table that follows points back to the scripts and inputs used to make it.

**`insilico_evaluation_unique_improved`.** This is the in-silico recovery check for the final selected stimuli
against unique-image baselines. It compares model-recovery accuracy in raw feature space and
encoding/predicted-brain space, and the AUC panel shows how selected images perform relative to random
baselines across model groups, using the frozen selection-evaluation CSVs and model-RDM summaries. This is one
of the main sanity checks that the frozen inputs are informative before any fMRI analysis is run.

**`selection_curve_all_models`.** This optimization trace shows how the selection score evolved across greedy
and refinement iterations for the all-model target. It marks the chosen checkpoint and refinement phases,
using the frozen selection-evaluation CSVs and model-RDM summaries. This is the figure to inspect when
checking why the frozen selected set stopped where it did.

**`selection_curve_architecture`.** This optimization trace shows how the selection score evolved across
greedy and refinement iterations for the architecture target. It marks the chosen checkpoint and refinement
phases, using the frozen selection-evaluation CSVs and model-RDM summaries. This is the figure to inspect when
checking why the frozen selected set stopped where it did.

**`selection_curve_compact`.** This optimization trace shows how the selection score evolved across greedy and
refinement iterations for the model-set target. It marks the chosen checkpoint and refinement phases, using
the frozen selection-evaluation CSVs and model-RDM summaries. This is the figure to inspect when checking why
the frozen selected set stopped where it did.

**`selection_curve_dataset`.** This optimization trace shows how the selection score evolved across greedy and
refinement iterations for the dataset target. It marks the chosen checkpoint and refinement phases, using the
frozen selection-evaluation CSVs and model-RDM summaries. This is the figure to inspect when checking why the
frozen selected set stopped where it did.

**`selection_curve_sota`.** This optimization trace shows how the selection score evolved across greedy and
refinement iterations for the state-of-the-art target. It marks the chosen checkpoint and refinement phases,
using the frozen selection-evaluation CSVs and model-RDM summaries. This is the figure to inspect when
checking why the frozen selected set stopped where it did.

**`selection_curve_training_objective`.** This optimization trace shows how the selection score evolved across
greedy and refinement iterations for the training-objective target. It marks the chosen checkpoint and
refinement phases, using the frozen selection-evaluation CSVs and model-RDM summaries. This is the figure to
inspect when checking why the frozen selected set stopped where it did.

## Contents Snapshot

- Folder: `00_stimulus_selection/decision_checks/selection_evaluation/figures/insilico_curve`
- Figures in this folder tree: 7
- Data/table-like files in this folder tree: 0
- Python scripts in this folder tree: 0
- Direct files: `insilico_evaluation_unique_improved.pdf`, `selection_curve_all_models.pdf`, `selection_curve_architecture.pdf`, `selection_curve_compact.pdf`, `selection_curve_dataset.pdf`, `selection_curve_sota.pdf`, `selection_curve_training_objective.pdf`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Provenance

Each row is one figure concept; `formats` lists the concrete files present in this folder.

| Figure | Formats | What it shows | Source / derivation | Data or inputs | Script | Paper use |
|---|---:|---|---|---|---|---|
| `insilico_evaluation_unique_improved` | pdf | in-silico discriminability of selected images against unique-image baselines | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation/figures/insilico_curve/insilico_evaluation_unique_improved.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/00_selection_evaluation/figures/insilico_curve/insilico_evaluation_unique_improved.pdf` | `selection_evaluation/data/*/{statistics,discriminability,greedy_scores,selection_objective_*}.csv`; `{which}_discriminability_{metric}_{corr_type}.csv`; `discriminability.csv`; `auc_significance.csv`; `FIGURES_DIR`; `EVAL_DATA_DIR`; plus 6 more | `src/cstims/evaluation/plotting.py`; `src/cstims/evaluation/core.py`; `00_stimulus_selection/decision_checks/selection_evaluation/code/figures/plot_insilico_evaluation_unique.py`; `00_stimulus_selection/decision_checks/selection_evaluation/code/figures/plot_insilico_evaluation_unique_improved.py`; plus 1 more | yes |
| `selection_curve_all_models` | pdf | selection objective or discriminability curve over selection/evaluation settings | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation/figures/insilico_curve/selection_curve_all_models.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/00_selection_evaluation/figures/insilico_curve/selection_curve_all_models.pdf` | `selection_evaluation/data/*/{statistics,discriminability,greedy_scores,selection_objective_*}.csv`; `{which}_discriminability_{metric}_{corr_type}.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/data/all_models/ablations.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/data/all_models/correlation_matrices.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/data/all_models/discriminability.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/data/all_models/diversity.csv`; plus 2 more | `src/cstims/evaluation/plotting.py`; `src/cstims/evaluation/core.py` | no |
| `selection_curve_architecture` | pdf | selection objective or discriminability curve over selection/evaluation settings | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation/figures/insilico_curve/selection_curve_architecture.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/00_selection_evaluation/figures/insilico_curve/selection_curve_architecture.pdf` | `selection_evaluation/data/*/{statistics,discriminability,greedy_scores,selection_objective_*}.csv`; `{which}_discriminability_{metric}_{corr_type}.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/data/all_models/ablations.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/data/all_models/correlation_matrices.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/data/all_models/discriminability.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/data/all_models/diversity.csv`; plus 2 more | `src/cstims/evaluation/plotting.py`; `src/cstims/evaluation/core.py` | no |
| `selection_curve_compact` | pdf | selection objective or discriminability curve over selection/evaluation settings | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation/figures/insilico_curve/selection_curve_compact.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/00_selection_evaluation/figures/insilico_curve/selection_curve_compact.pdf` | `selection_evaluation/data/*/{statistics,discriminability,greedy_scores,selection_objective_*}.csv`; `{which}_discriminability_{metric}_{corr_type}.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/data/all_models/ablations.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/data/all_models/correlation_matrices.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/data/all_models/discriminability.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/data/all_models/diversity.csv`; plus 2 more | `src/cstims/evaluation/plotting.py`; `src/cstims/evaluation/core.py` | no |
| `selection_curve_dataset` | pdf | selection objective or discriminability curve over selection/evaluation settings | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation/figures/insilico_curve/selection_curve_dataset.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/00_selection_evaluation/figures/insilico_curve/selection_curve_dataset.pdf` | `selection_evaluation/data/*/{statistics,discriminability,greedy_scores,selection_objective_*}.csv`; `{which}_discriminability_{metric}_{corr_type}.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/data/all_models/ablations.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/data/all_models/correlation_matrices.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/data/all_models/discriminability.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/data/all_models/diversity.csv`; plus 2 more | `src/cstims/evaluation/plotting.py`; `src/cstims/evaluation/core.py` | no |
| `selection_curve_sota` | pdf | selection objective or discriminability curve over selection/evaluation settings | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation/figures/insilico_curve/selection_curve_sota.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/00_selection_evaluation/figures/insilico_curve/selection_curve_sota.pdf` | `selection_evaluation/data/*/{statistics,discriminability,greedy_scores,selection_objective_*}.csv`; `{which}_discriminability_{metric}_{corr_type}.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/data/all_models/ablations.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/data/all_models/correlation_matrices.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/data/all_models/discriminability.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/data/all_models/diversity.csv`; plus 2 more | `src/cstims/evaluation/plotting.py`; `src/cstims/evaluation/core.py`; `00_stimulus_selection/decision_checks/selection_evaluation/code/figures/plot_selection_curve.py`; `00_stimulus_selection/decision_checks/selection_evaluation/code/figures/plot_selection_curve_improved.py` | no |
| `selection_curve_training_objective` | pdf | selection objective or discriminability curve over selection/evaluation settings | copied from `/data/home_roth/_stachelschwein/rsa_based_selection/experiments/cstim_paper/00_selection_evaluation/figures/insilico_curve/selection_curve_training_objective.pdf`; historical share path `data/tier1_analysis_derivatives/cstim_paper/00_selection_evaluation/figures/insilico_curve/selection_curve_training_objective.pdf` | `selection_evaluation/data/*/{statistics,discriminability,greedy_scores,selection_objective_*}.csv`; `{which}_discriminability_{metric}_{corr_type}.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/data/all_models/ablations.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/data/all_models/correlation_matrices.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/data/all_models/discriminability.csv`; `00_stimulus_selection/decision_checks/selection_evaluation/data/all_models/diversity.csv`; plus 2 more | `src/cstims/evaluation/plotting.py`; `src/cstims/evaluation/core.py` | no |
<!-- END AUTO-FIGURE-PROVENANCE -->
