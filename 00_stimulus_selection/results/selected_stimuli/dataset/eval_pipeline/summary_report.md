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
| AUC (avg) | 0.182 | 0.327 | 44.3% |

### Statistical Significance

| Test | Value |
|------|-------|
| p-value (paired t-test) | 2.80e-24 *** |
| Effect size (Cohen's d) | 0.999 |

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
| Mean pairwise similarity | 0.182 |
| Feature entropy | 2.883 |
| vs Random | +0.0611 |
