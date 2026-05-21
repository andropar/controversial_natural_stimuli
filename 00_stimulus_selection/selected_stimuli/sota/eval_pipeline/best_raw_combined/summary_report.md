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
| AUC (avg) | 0.194 | 0.388 | 49.9% |

### Statistical Significance

| Test | Value |
|------|-------|
| p-value (paired t-test) | 2.37e-27 *** |
| Effect size (Cohen's d) | 1.094 |

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
| Mean pairwise similarity | 0.344 |
| Feature entropy | 2.425 |
| vs Random | +0.0053 |
