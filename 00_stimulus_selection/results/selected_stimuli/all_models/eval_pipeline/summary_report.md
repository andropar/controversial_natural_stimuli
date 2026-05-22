# Stimulus Selection Summary Report

## Selection Configuration

| Parameter | Value |
|-----------|-------|
| Method | raw_plus_all_encodings |
| n_selected | 100 |
| n_models | 21 |
| Metric | cosine |
| Correlation type | correlation |

## Discriminability

| Metric | Selected | Random | Improvement |
|--------|----------|--------|-------------|
| AUC (avg) | 0.326 | 0.504 | 35.3% |

### Statistical Significance

| Test | Value |
|------|-------|
| p-value (paired t-test) | 1.01e-25 *** |
| Effect size (Cohen's d) | 1.044 |

## Selection Scores

| Stage | Score |
|-------|-------|
| Greedy final | 0.236 |
| Refinement final | 0.250 |
| Improvement | +5.7% |
| n_replacements | 137 |

## Diversity

| Metric | Value |
|--------|-------|
| Mean pairwise similarity | 0.323 |
| Feature entropy | 1.011 |
| vs Random | -0.0041 |
