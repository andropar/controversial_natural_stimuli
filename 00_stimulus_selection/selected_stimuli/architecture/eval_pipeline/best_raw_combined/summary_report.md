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
| AUC (avg) | 0.160 | 0.339 | 52.9% |

### Statistical Significance

| Test | Value |
|------|-------|
| p-value (paired t-test) | 7.70e-23 *** |
| Effect size (Cohen's d) | 0.955 |

## Selection Scores

| Stage | Score |
|-------|-------|
| Greedy final | 0.255 |
| Refinement final | 0.277 |
| Improvement | +8.7% |
| n_replacements | 94 |

## Diversity

| Metric | Value |
|--------|-------|
| Mean pairwise similarity | 0.198 |
| Feature entropy | 0.913 |
| vs Random | +0.0491 |
