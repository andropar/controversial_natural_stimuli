# Explanation Analysis Report

Generated from `analysis/run_all.py` components on the current cached outputs.

## Primary matched-counterfactual result

The mixed-RSA controversial-minus-baseline delta remains negative for every
available counterfactual baseline in the pooled subject-level summary:

| Counterfactual | Mean delta | Subject-bootstrap 95% CI |
|---|---:|---:|
| Same-session baseline | -0.138 | [-0.147, -0.127] |
| Held-out unique baseline | -0.211 | [-0.228, -0.196] |
| Low-level matched | -0.216 | [-0.237, -0.195] |
| PPCA-OOD matched | -0.203 | [-0.218, -0.189] |
| Embedding-PC matched | -0.232 | [-0.253, -0.215] |
| Low-level + OOD + embedding matched | -0.235 | [-0.258, -0.216] |
| High-PPCA-OOD held-out baseline | -0.160 | [-0.173, -0.146] |

Readout: the current matched baselines do not support low-level statistics,
PPCA-OOD, or embedding-PC structure as sufficient explanations for the
controversial-stimulus mixed-RSA drop.

## Reliability readout

For the same-session baseline, the raw mixed-RSA delta is negative for all five
subjects in all five model sets. Noise-ceiling-normalized deltas remain negative
for all subjects in `all_models`, `dataset`, `sota`, and `training_objective`.
`architecture` is weaker after normalization: 2/5 subjects remain negative.

Readout: reliability is not a sufficient explanation for the main all-models,
dataset, sota, or training-objective effects. It may contribute to the smaller
architecture effect.

## Residual reliable structure

For mixed RSA, the all-models diagnostic set has a larger ensemble gap to the
correlation ceiling than vicco (+0.128), higher residual split-half reliability
(+0.070), higher within-subject residual fraction (+0.292), and higher LOSO
residual fraction (+0.113). Absolute LOSO residual RSA is similar to baseline
(-0.005).

Readout: the all-models set has more reliable brain geometry left outside the
current model-RDM space, mainly as a larger fraction of the available reliable
signal rather than a larger absolute cross-subject residual correlation.

## Pair-level variance partitioning

The pair-level follow-up uses image-blocked cross-validation. Pooled full-model
CV R2 is slightly negative (-0.010), so this table should be treated as a
diagnostic rather than primary evidence. Model-space predictors have the largest
pooled alone-predictive performance (0.021), followed by semantic embeddings
(0.011). Unique drops from the full model are small: model space (+0.008) and
model disagreement (+0.002) are positive on average; low-level, OOD, and
semantic families are not positive after accounting for all families.

Readout: blocked pair-level prediction is weak, but it does not suggest that
low-level or OOD families uniquely explain the effect. The more defensible
evidence remains the matched-counterfactual ladder plus reliability/residual
readout.
