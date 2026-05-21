# Stimulus Selection Summary Report

## Selection Configuration

| Parameter | Value |
|-----------|-------|
| Method | raw_plus_all_encodings |
| n_selected | 100 |
| n_models | 6 |
| Metric | cosine |
| Correlation type | correlation |

## Discriminability

| Metric | Selected | Random | Improvement |
|--------|----------|--------|-------------|
| AUC (avg) | 0.215 | 0.388 | 44.6% |

### Statistical Significance

| Test | Value |
|------|-------|
| p-value (paired t-test) | 6.74e-27 *** |
| Effect size (Cohen's d) | 1.080 |

## Selection Scores

| Stage | Score |
|-------|-------|
| Greedy final | 0.255 |
| Refinement final | 0.243 |
| Improvement | -4.9% |
| n_replacements | 124 |

## Diversity

| Metric | Value |
|--------|-------|
| Mean pairwise similarity | 0.345 |
| Feature entropy | 2.430 |
| vs Random | +0.0067 |
