# VLM-Based Stimulus Annotation Audit

This folder contains a minimal model-assisted perceptual/semantic audit for
the controversial-stimuli manuscript. It uses a vision-language model through
OpenRouter to annotate each stimulus image with structured candidate covariates
such as recognizability, ambiguity, clutter, object count, naturalness, and
semantic composition.

This is not human validation and should not be described as simulating human
perception. The annotations are VLM-derived, human-relevant candidate
covariates for later controls and matching analyses.

## Image Source

The script reads the original stimulus image folders directly:

```bash
/data/labshare/_stachelschwein/SSD/jroth/final_cstims_hdf5_files
```

The loader preserves the original analysis convention:

- `architecture` uses the physical `dataset/` folder.
- `dataset` uses the physical `architecture/` folder.
- `vicco` uses the physical `shared_vicco/` folder.

The default conditions are:

```text
all_models architecture dataset sota training_objective vicco
```

## Files

- `code/01_vlm_annotate_images.py`: load images, call OpenRouter, validate JSON,
  and write annotation tables.
- `outputs/annotations/*.jsonl`: incremental append-only annotations.
- `outputs/annotations/*.csv`: final one-row-per-stimulus table.
- `outputs/annotations/*.parquet`: written if a parquet engine is installed.
- `outputs/raw_responses/<run>/`: raw model responses for debugging.
- `logs/vlm_annotation_failures__*.csv`: failed images and errors.

## Dry Run

From this directory:

```bash
python code/01_vlm_annotate_images.py \
  --dry-run-one-image \
  --condition all_models \
  --model google/gemini-3.5-flash
```

## Small Test

```bash
python code/01_vlm_annotate_images.py \
  --condition all_models vicco \
  --limit 10 \
  --model google/gemini-3.5-flash \
  --image-size 768 \
  --sleep 0.5
```

## Full Annotation

```bash
python code/01_vlm_annotate_images.py \
  --model google/gemini-3.5-flash \
  --image-size 768 \
  --sleep 0.5
```

Parallel MiniMax M3 full run:

```bash
python code/01_vlm_annotate_images.py \
  --model minimax/minimax-m3 \
  --output-name full_minimax_m3_all_stimuli \
  --image-size 768 \
  --workers 24 \
  --sleep 0.02 \
  --timeout 180
```

## Resume

Use a stable `--output-name` for long runs. The script skips stimulus IDs already
present in the run JSONL unless `--overwrite` is passed.

```bash
python code/01_vlm_annotate_images.py \
  --output-name vlm_annotations__gemini_flash_full \
  --model google/gemini-3.5-flash \
  --image-size 768 \
  --sleep 0.5
```

Resume the same run with the same command. Use `--start-index` and `--end-index`
for manual chunking.

## Useful Checks

List selected image counts without calling the API:

```bash
python code/01_vlm_annotate_images.py --list-images
```

View an annotation table locally:

```bash
python code/view_vlm_annotations.py \
  --input outputs/annotations/small_test_vicco_5.csv \
  --port 8765
```

Then open `http://127.0.0.1:8765`.

The viewer also works as a small review app. For each stimulus, every table
column is shown as its own editable row with `Accept`, `Save edit`, `Reject`,
and `Clear` controls. Review state is saved separately as
`outputs/annotations/review_state__<input-stem>.json`, and a reviewed CSV is
written as `outputs/annotations/reviewed__<input-stem>.csv`. The raw VLM output
CSV is not overwritten.

Fallback models to try if the chosen model rejects image input or repeatedly
returns invalid JSON:

```text
google/gemini-2.5-flash
google/gemini-2.5-flash-lite
qwen/qwen2.5-vl-72b-instruct
```
