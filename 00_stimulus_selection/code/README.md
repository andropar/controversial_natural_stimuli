# Stimulus Selection Code

This folder contains the code path used to produce the frozen selected
controversial-stimulus inputs used by the downstream paper analyses. The paper
results do not come from an arbitrary rerun of this code; they use the copied
frozen runs in `../results/selected_stimuli/`.

## Frozen Selection Run Used Downstream

The selected stimuli used by the paper analyses are the five copied runs listed
in `../manifests/selection_runs.csv`:

| model set | frozen target | original run |
|---|---|---|
| `all_models` | `../results/selected_stimuli/all_models/selected_stimuli_data.pkl` | `outputs/final_cstims_v2_full/all_models/method-raw_plus_all_encodings/20251222_175721` |
| `sota` | `../results/selected_stimuli/sota/selected_stimuli_data.pkl` | `outputs/final_cstims_v2_full/sota/method-raw_plus_all_encodings/20251222_175721` |
| `training_objective` | `../results/selected_stimuli/training_objective/selected_stimuli_data.pkl` | `outputs/final_cstims_v2_full/training_objective/method-raw_plus_all_encodings/20251222_175721` |
| `architecture` | `../results/selected_stimuli/architecture/selected_stimuli_data.pkl` | `outputs/final_cstims_v2_full/architecture/method-raw_plus_all_encodings/20251222_175721` |
| `dataset` | `../results/selected_stimuli/dataset/selected_stimuli_data.pkl` | `outputs/final_cstims_v2_full/dataset/method-raw_plus_all_encodings/20251222_175721` |

All five frozen runs used `target_size=100`, `seed=42`, `metric=cosine`,
`corr_type=correlation`, `use_analytical=true`, `aggregation_within=mean`,
`aggregation_across=min`, `noise_ceiling_target=0.46`, and the image-quality
filter with `min_resolution=1000` and `natural_prob_threshold=0.85`.

The recorded Hydra overrides for the frozen runs were:

```text
paths=raven
corr_type=correlation
use_analytical=true
model_set=<one of all_models,sota,training_objective,architecture,dataset>
method=raw_plus_all_encodings
metric=cosine
aggregation_within=mean
aggregation_across=min
max_ram_gb=300
target_size=100
batch_size=2500
refinement.max_passes=10
refinement.min_replacements=0
image_filter.enabled=true
```

## Important Method Detail

Do not infer the exact scoring rule from the directory name alone. The frozen
run family is named `method-raw_plus_all_encodings`, and each run configured six
tracks:

- five subject-specific encoding tracks: `sub-01`, `sub-03`, `sub-05`,
  `sub-06`, `sub-07`
- one raw feature track: `raw`

However, the frozen `.hydra/config.yaml` files and the copied
`eval_pipeline/summary.csv` files record:

```text
track_agg_method = identity
track_norm_method = zscore
raw_weight = 0.5
track_names = sub-01,sub-03,sub-05,sub-06,sub-07,raw
```

In this code, `identity` aggregation returns the first normalized track in the
track order. Therefore, these frozen paper inputs should be described as the
`raw_plus_all_encodings` selection runs from `20251222_175721`, with the
recorded effective selection score determined by the first subject-encoding
track (`sub-01`) under the stored configuration. They should not be casually
described as the later weighted-mean raw-plus-all-encodings variant unless a
separate rerun/config proves that.

## Selection Procedure

The entry point is `select_controversial_stimuli.py`. It loads model features
for the requested model set, constructs RDM-based model-disagreement scores,
starts from `init_size=3` seed images, greedily adds images until `target_size=100`,
and then runs the configured replacement/refinement passes. Candidate images
were accepted only after passing the image-quality/natural-image filter.

The actual frozen outputs to use are:

- `../results/selected_stimuli/<model_set>/selected_stimuli_data.pkl`: selected image
  records and stimulus payload used downstream.
- `../results/selected_stimuli/<model_set>/checkpoint.pkl`: final selection checkpoint.
- `../results/selected_stimuli/<model_set>/.hydra/config.yaml`: exact frozen config.
- `../results/selected_stimuli/<model_set>/eval_pipeline/`: post-hoc sanity checks and
  diagnostics for the selected set.

The `eval_pipeline/best_raw_combined/` subfolders are diagnostic exports from
selection evaluation. They are useful sanity checks, but they are not the primary
selected-stimulus payload used by the downstream RSA analyses.

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Contents Snapshot

- Folder: `00_stimulus_selection/code`
- Figures in this folder tree: 0
- Data/table-like files in this folder tree: 0
- Python scripts in this folder tree: 20
- Direct files: `select_controversial_stimuli.py`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.
<!-- END AUTO-FIGURE-PROVENANCE -->
