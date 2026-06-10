# Pair Labeling App

Small local web app for labeling all unordered image pairs from the
`all_models` controversial stimulus set.

## Run

From the repository root:

```bash
/data/home_roth/miniforge3/bin/python \
  05_controls_and_supplementary/stimulus_and_pair_diagnostics/pair_labeling_app/code/server.py \
  --port 8123
```

Then open `http://127.0.0.1:8123/`.

The server defaults to the 300-pair anchor-balanced diagnostic queue generated
by:

```bash
/data/home_roth/miniforge3/bin/python \
  05_controls_and_supplementary/stimulus_and_pair_diagnostics/pair_labeling_app/code/build_pair_queue.py
```

That queue contains:

- 100 worst model-brain mismatch pairs, one per anchor image
- 100 spread-matched low-mismatch controls, one per anchor image
- 100 random anchor-balanced pairs

Model-brain mismatch is `abs(mean_brain_z_position)` from the existing
pair-level brain-placement table. To label all 4,950 unordered pairs instead,
pass `--all-pairs`.

The image source defaults to:

`00_stimulus_selection/results/selected_stimuli/all_models/eval_pipeline/images`

Use `--image-dir /path/to/images` to point at a different stimulus image
folder. Labels are autosaved to:

`05_controls_and_supplementary/stimulus_and_pair_diagnostics/pair_labeling_app/results/pair_labels.csv`

## Label Schema

Each row is keyed by `img_i,img_j` with `img_i < img_j`.

- `semantic_similarity`: `same`, `related`, `unrelated`, `unsure`
- `visual_surface_similarity`: `high`, `medium`, `low`, `unsure`
- `shape_layout_similarity`: `high`, `medium`, `low`, `unsure`
- `scene_context_similarity`: `same`, `related`, `different`, `unsure`
- `dominant_relation`:
  - `semantic_match_visual_mismatch`
  - `visual_match_semantic_mismatch`
  - `both_match`
  - `both_mismatch`
  - `mixed_or_ambiguous`
  - `unsure`
- `confidence`: `high`, `medium`, `low`

`notes` is optional and is meant for rare edge cases.

## Keyboard Shortcuts

The interface can be used by mouse alone, but the following shortcuts are
supported:

- Semantic similarity: `q`, `w`, `e`, `r`
- Visual-surface similarity: `a`, `s`, `d`, `f`
- Shape/layout similarity: `z`, `x`, `c`, `v`
- Scene/context similarity: `u`, `i`, `o`, `p`
- Dominant relation: `1`, `2`, `3`, `4`, `5`, `6`
- Confidence: `j`, `k`, `l`
- Navigation: left/right arrows
- Save current pair and move forward: `Enter`

The saved CSV is directly joinable to pair-level model/brain tables using
`img_i,img_j`.
