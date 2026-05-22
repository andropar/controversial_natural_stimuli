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
| AUC (avg) | 0.171 | 0.327 | 47.6% |

### Statistical Significance

| Test | Value |
|------|-------|
| p-value (paired t-test) | 3.62e-25 *** |
| Effect size (Cohen's d) | 1.027 |

## Selection Scores

| Stage | Score |
|-------|-------|
| Greedy final | 0.246 |
| Refinement final | 0.254 |
| Improvement | +3.3% |
| n_replacements | 256 |

## Diversity

| Metric | Value |
|--------|-------|
| Mean pairwise similarity | 0.173 |
| Feature entropy | 2.830 |
| vs Random | +0.0515 |
