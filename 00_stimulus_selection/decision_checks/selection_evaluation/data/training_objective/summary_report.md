# Stimulus Selection Summary Report

## Selection Configuration

| Parameter | Value |
|-----------|-------|
| Method | raw_plus_all_encodings |
| n_selected | 100 |
| n_models | 5 |
| Metric | cosine |
| Correlation type | correlation |

## Discriminability

| Metric | Selected | Random | Improvement |
|--------|----------|--------|-------------|
| AUC (avg) | 0.189 | 0.390 | 51.4% |

### Statistical Significance

| Test | Value |
|------|-------|
| p-value (paired t-test) | 3.74e-34 *** |
| Effect size (Cohen's d) | 1.306 |

## Selection Scores

| Stage | Score |
|-------|-------|
| Greedy final | 0.243 |
| Refinement final | 0.245 |
| Improvement | +0.7% |
| n_replacements | 100 |

## Diversity

| Metric | Value |
|--------|-------|
| Mean pairwise similarity | 0.377 |
| Feature entropy | 0.882 |
| vs Random | +0.0497 |
