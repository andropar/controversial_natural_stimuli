# 06 — OOD analysis

Are controversial stimuli "just OOD images"? This section disentangles
out-of-distribution drift from model-disagreement signal, using three
descriptive controls applied on top of the per-stimulus PPCA log-likelihoods
from `01_compute_pca_loglik.py`.

## Pipeline

```
01_compute_pca_loglik.py                       -> data/pca_loglik.csv
02_baseline_subsampling.py                     -> data/baseline_subsampling{,_summary}.csv          (OOD-matched bootstraps)
02c_low_level_deterministic_subsets.py         -> data/wrsa_low_level_subsets{,_summary,_comparison}.csv  (deterministic high/low/matched vicco subsets — proper low-level test)
04_disagreement_vs_ood.py                      -> data/disagreement_vs_ood_{pairs,images,summary}.csv
05_quantify_dissociation.py                    -> data/dissociation_summary.csv                     (effect sizes + paired Δwrsa CIs)

figures/plot_loglik_distributions_improved.py
figures/plot_ood_vs_alignment_improved.py
figures/plot_baseline_subsampling.py
figures/plot_low_level_deterministic.py
figures/plot_disagreement_vs_ood.py
```

### Archived analyses (see `archive/`)
- `02b_baseline_subsampling_low_level.py` — null result; vicco bootstrap pool's
  low-level distance does not span cstim levels, so the test is undefined.
  Superseded by 02c (deterministic vicco subsets).
- `03_partial_correlation_control.py` and `03b_low_level_robustness.py` —
  paired-difference regression of Δwrsa on Δood + Δlow at n=5 sets. With only
  5 set-level points and Δlow/Δood collinear, β_low is not distinguishable
  from a set-level permutation null (p=0.17). Descriptive at this resolution,
  not a clean "controlling for" analysis. The summary subsections below are
  preserved for reference; the scripts and outputs live in `archive/`.

## Analyses

### 01 — PPCA log-likelihood (baseline OOD score)
Per (model × subject), fit PPCA on the encoder's training-image features
(and predictions); score every stimulus's loglik; z-score against training.
Defines `loglik_feature_z` and `loglik_pred_z`, the OOD scores used downstream.

### 02 — Within-baseline subsampling control
Reuses the 1000 vicco bootstraps already computed by
`02_rsa_scores/02_compute_wrsa_transfer.py` (deterministic: same seed, same
indices). Per (subject × model_set × model):

  - `cstim_ood`, `cstim_wrsa`         — cstim score
  - `all_boot_ood`, `all_boot_wrsa`   — across all 1000 vicco bootstraps
  - `matched_ood`, `matched_wrsa`     — across the K=50 OOD-closest bootstraps
  - `gap_sd = |cstim_ood − matched_ood| / SD(boot_ood)` — match quality
  - `match_quality ∈ {matched, weak, out_of_range}` (gap thresholds 0.5 / 1.5 SD)
  - `out_of_range = cstim_ood beyond the bootstrap OOD min/max`

The match-quality assessment is necessary because the baseline pool is not
OOD enough to span cstim OOD for most sets. Of 205 cells (5 subjects × 41
model×set combinations), 68 (33%) qualify as `matched`.

**Per-set cells (matched / weak / out_of_range):**
| set | matched | weak | out_of_range |
|---|---|---|---|
| architecture | 17 | 2 | 6 |
| all_models | 33 | 5 | 62 |
| sota | 13 | 1 | 16 |
| training_objective | 5 | 5 | 15 |
| dataset | 0 | 0 | 25 |

`dataset` has zero matchable cells. `all_models` has the largest absolute count
of matched cells. `architecture` is the most consistent (17/25 cells matched).

**On `matched` cells (subject × model averages):**
- `all_models`:         drop_vs_matched = −0.238  (n=33 cells)
- `architecture`:       drop_vs_matched = −0.068  (n=17 cells)
- `sota`:               drop_vs_matched = −0.065  (n=13 cells)
- `training_objective`: drop_vs_matched = −0.287  (n=5  cells)

Where matching is meaningful, the cstim drop survives. For `dataset`
the drop cannot be tested by this design.

### 02b — Bootstrap-matched baseline on low-level distance (NULL — pool too narrow) (archived)
The analog of 02 for low-level distance. The vicco bootstrap pool's mean
low-level distance never reaches cstim levels: `boot_low_max ≈ 3.33` while
cstim means span 3.46 to 4.05. Of 205 cells, **0 are matchable** on
low-level distance — the test is undefined for any cstim set with this
baseline pool. This motivated 02c.

### 02c — Deterministic vicco subsets spanning low-level range (decisive test)
Instead of bootstrap subsets (which cluster narrowly around vicco's overall
mean), we construct DETERMINISTIC vicco subsets of n=100 that span the
low-level distance range:

  - `bottom100`        — 100 lowest-low-level vicco images (mean = 2.22)
  - `middle100`        — 100 middle-low-level vicco images (mean = 3.01)
  - `top100`           — 100 highest-low-level vicco images (mean = 4.25,
                          exceeds every cstim set's mean)
  - `match_<set>`      — 100 vicco images whose mean matches each cstim
                          set's mean (greedy nearest-image; tight band,
                          std ≈ 0.3)
  - `dist_match_<set>` — 100 vicco images whose full distance distribution
                          matches each cstim set's (greedy 1-to-1 quantile
                          pairing; mean + spread + shape, std ≈ 1.0–1.2,
                          comparable to cstim's std). Primary low-level
                          control because mean-matching alone leaves the
                          spread mismatched (cstim std ≈ 0.7, mean-matched
                          baseline std ≈ 0.3).

Compute wRSA on each subset per (subject × model), then summarise.

**Result — vicco wRSA is essentially flat across mean low-level distance:**

| subset | mean_low | mean wRSA (subj × model) |
|---|---|---|
| bottom100 | 2.22 | 0.479 |
| middle100 | 3.01 | 0.486 |
| match_architecture | 3.38 | 0.493 |
| match_training_obj | 3.41 | 0.501 |
| match_dataset | 3.69 | 0.489 |
| match_sota | 3.70 | 0.483 |
| match_all_models | 3.84 | 0.466 |
| top100 | 4.25 | 0.460 |

vicco wRSA varies only 0.04 across nearly a 2× change in mean low-level distance.

**Cstim drops persist against matched-and-beyond-matched vicco subsets:**

| model_set | cstim_wrsa | match_<set>_wrsa | drop_vs_match | top100_wrsa | drop_vs_top100 |
|---|---|---|---|---|---|
| all_models | 0.230 | 0.466 | **−0.236** | 0.460 | **−0.230** |
| architecture | 0.302 | 0.493 | −0.191 | (vicco max 4.25 ≫ cstim 3.46) | — |
| training_objective | 0.374 | 0.501 | −0.127 | — | — |
| sota | 0.294 | 0.483 | −0.189 | — | — |
| dataset | 0.336 | 0.489 | −0.153 | — | — |

For `all_models`, cstim drops 0.23 wRSA below the top100 vicco subset —
which has *higher* mean low-level distance than cstim itself (4.25 vs 4.05).
The cstim drop is **not** explained by low-level distribution shift.

This also resolves the puzzle from 03/03b: the set-level Spearman ρ = −0.9
between Δlow and Δwrsa is **selection-induced confounding**, not causal.
Cstim sets that admit more model disagreement (e.g., all_models with 20
diverse models) also admit images that drift further from training. But
pushing the baseline to the same — or higher — low-level distance does
*not* reproduce the cstim drop. So the across-set Δlow/Δwrsa correlation
is a downstream consequence of selection optimising for disagreement, not
a causal role of low-level distance.

### 03 — Paired-difference regression descriptive control (archived)
Per (subject × model × model_set), one row:
  - `Δwrsa = cstim_wrsa − mean_over_boots(vicco_wrsa)`
  - `Δood  = cstim_ood  − mean_over_boots(vicco_ood)`
  - `Δlow  = low_level_distance(cstim_set) − low_level_distance(vicco)`
    (Mahalanobis from training stat centroid)

Joint regression `Δwrsa = α + β_ood·Δood + β_low·Δlow`, n = 205, with
**subject-cluster bootstrap** (1000 resamples drawing whole subjects with
replacement) for confidence intervals.

| coefficient | estimate | 95% bootstrap CI |
|---|---|---|
| α (intercept) | −0.036 | [−0.061, −0.021] |
| β_ood | +0.025 | [0.000, +0.046] |
| β_low | −0.130 | [−0.139, −0.123] |

R² = 0.32. Per-subject α is negative for **all 5 subjects** (range −0.083 to
−0.015).

**Caveat — see 03b.** The bootstrap CI on β_low is misleadingly tight.
`Δlow` takes only 5 unique values (one per cstim set), so β_low is identified
entirely between sets. The subject-cluster bootstrap captures variation in
subject sample but holds the set→Δlow mapping fixed. The set-level
permutation test (03b) gives p = 0.17 for β_low — i.e., β_low is not
distinguishable from the set-level permutation null.

### 03b — Robustness of the β_low result (archived)
Four checks aimed at the n=5-set identifiability problem in 03:

  (1) **Set-level Pearson / Spearman r at n=5.**
        r(Δlow, Δwrsa)  Pearson −0.83 (p=0.082, n=5),  Spearman −0.90 (p=0.037)
        r(Δood, Δwrsa)  Pearson +0.78 (p=0.120, n=5),  Spearman +0.90 (p=0.037)
      Both predictors rank the 5 sets identically with respect to Δwrsa.
      They cannot be disentangled at this resolution.

  (2) **Permutation test, 5! = 120 permutations of set→Δlow.**
        observed β_low = −0.130   →  permutation two-sided p = 0.167
        observed R²    = 0.323    →  permutation one-sided p = 0.058
      Cannot reject the null that Δlow has no relationship to Δwrsa at
      α=0.05.

  (3) **Leave-one-set-out CV.** Fit on 4 sets, predict the held-out set's
      mean drop. Pearson r(predicted, actual) = 0.76. Per-set abs error
      ranges from 8% (all_models) to 53% (sota, architecture). Predictions
      do not generalize cleanly across sets.

  (4) **Random-predictor null** (1000 random uniform 5-value mappings).
      6.7% of random set-level predictors achieve R² ≥ observed (0.323).
      Observed R² is at the ~93rd percentile of random predictors —
      borderline.

**Honest read of 03 + 03b:**

- At the set level, both Δlow and Δood rank-correlate with Δwrsa
  (Spearman ρ ≈ 0.9 for each, p ≈ 0.04 at n=5). They are collinear at
  this resolution.
- The β_low = −0.13 estimate from 03 reflects 5 between-set points; the
  permutation null says we cannot rule out chance.
- The α = −0.036 "residual drop after controlling for OOD and low-level"
  is a linear extrapolation to Δlow=Δood=0, outside the data range. The
  per-subject α negativity reflects that the linear fit predicts negative
  Δwrsa under that extrapolation; it is not a clean "controlled" residual.
- We can say: cstim sets that drift further from training on low-level
  image stats also tend to show larger alignment drops. We cannot say
  this is *because of* low-level distance specifically — any set-level
  variable that ranks sets in the same order would do equally well, and
  Δood ranks them identically.

The cleanest framing of 03 + 03b for the paper is therefore narrower
than the original draft suggested: alignment drop covaries with both
OOD and low-level shift across the 5 sets, but with n=5 we cannot make
mechanistic attribution. The argument that controversial stimuli are
not "just OOD" rests on 02 (matched-OOD baselines do not reproduce the
drop where matching is feasible) and 04 (image-level residual
disagreement after matched vicco reference). 03 should be cited as a
descriptive correlation, not a "controlling for" analysis.

### 05 — Quantitative dissociation summary
Effect sizes and paired Δwrsa CIs for the dissociation panel
(`figures/low_level_dissociation.{pdf,png}`).

**(1) "Shift exists?"** — Cohen's d cstim vs baseline (oriented so positive d
⇒ cstim is more OOD / less reliable):

| axis | scale | d range across sets |
|---|---|---|
| low-level Mahalanobis distance | per-stimulus (n=100 vs 292) | +0.28 to +0.59 (small–medium) |
| PPCA log-lik z, predicted-response space | per-stimulus pooled (model × subject) | +0.60 to +0.96 (medium–large) |
| PPCA log-lik z, raw-feature space | per-stimulus pooled (model × subject) | +0.57 to +0.83 (medium–large) |
| brain RDM noise ceiling | paired per-subject (n=5) | mixed: +3.62 (all_models), +2.31 (dataset), +1.25 (architecture), −0.49 (training_objective), −0.71 (sota) |

The PPCA shift is consistently medium-to-large; the low-level shift is
small-to-medium; the noise-ceiling shift is heterogeneous (cstim is *less*
reliable for all_models and dataset, *more* reliable for training_objective
and sota).

**(2) "Shift doesn't drive the drop?"** — paired Δwrsa per (subject × model)
between cstim and three control baselines, subject-cluster bootstrap (1000
resamples) for 95% CIs. Negative Δ ⇒ cstim drop persists vs control.

| control | combined Δ [95% CI] |
|---|---|
| mean-matched baseline subset (`match_<set>`) | −0.189 [−0.203, −0.177] |
| **distribution-matched baseline (`dist_match_<set>`)** — primary control | **−0.179 [−0.187, −0.169]** |
| highest-low-level baseline subset (`top100`) | −0.171 [−0.186, −0.153] |
| full baseline | −0.174 [−0.187, −0.162] |
| NC-normalized (full baseline) | −0.218 [−0.242, −0.199] |

All combined CIs exclude zero. Per-set CIs also exclude zero in every set
*except* nc_normalized × architecture (CI [−0.083, +0.030]).

### 04 — Disagreement vs OOD (structural, image / pair level)
Methodology choices:

1. **Per-set model rosters** — disagreement on each cstim set uses only the
   models that drove that set's selection. The vicco reference for that set
   uses the **same model roster**, so disagreement magnitudes are directly
   comparable across cstim and vicco.
2. **Two pair-OOD definitions** — `pair_ood_mean = (ood_i+ood_j)/2` and
   `pair_ood_max = max(ood_i, ood_j)`. Spearman r reported for each.
3. **Reference fit on vicco only** — the reference line
   `image_disagreement = a + b·image_OOD` is fit on vicco using the cstim
   set's roster. Cstim residuals are computed against this matched reference.
4. **VICReg sensitivity** — for sets that include VICReg (`all_models` and
   `training_objective`), both `variant ∈ {all, no_vicreg}` are computed.

**Pair-level Spearman r between disagreement and OOD** (variant=`all`):
| set | r(pair_ood_mean) | r(pair_ood_max) | n_pairs |
|---|---|---|---|
| all_models | −0.009 | −0.035 | 4,950 |
| architecture | −0.026 | −0.028 | 4,950 |
| training_objective | −0.011 | +0.075 | 4,950 |
| sota | +0.055 | +0.081 | 4,950 |
| dataset | −0.082 | −0.066 | 4,950 |
| vicco (all_models ref) | −0.049 | −0.023 | 42,486 |

All correlations |r| ≤ 0.09 — pair-level disagreement and OOD are nearly
orthogonal across all sets.

**Image-level mean residual disagreement** (after vicco-fit reference, matched roster):
| set | mean_residual | n_imgs | variant |
|---|---|---|---|
| all_models | +0.280 | 100 | all |
| all_models | +0.276 | 100 | no_vicreg |
| training_objective | +0.272 | 100 | all |
| training_objective | +0.298 | 100 | no_vicreg |
| sota | +0.123 | 100 | all |
| dataset | +0.123 | 100 | all |
| architecture | **−0.006** | 100 | all |

`architecture` is the exception — its cstim disagreement is well-predicted
by OOD on this set. Plausibly: architecture's 5 models (all supervised
CNN/ViT variants) span a narrow disagreement axis that is largely
captured by OOD-driven variation. The other four sets show substantial
positive residual disagreement at matched OOD level.

VICReg sensitivity (`all_models`): residual is +0.28 with VICReg vs +0.28
without — robust. For `training_objective`, +0.27 (with) vs +0.30 (without)
— also robust to VICReg's presence.

## Outputs

| File | Source | Notes |
|------|--------|-------|
| `data/pca_loglik.csv` | 01 | per-stimulus loglik (feature + pred space) |
| `data/baseline_subsampling.csv` | 02 | per (subj × set × model × bootstrap) OOD + wRSA |
| `data/baseline_subsampling_summary.csv` | 02 | per-cell match quality, drop, gap_sd |
| `data/wrsa_low_level_subsets.csv` | 02c | per (subj × model × subset) wRSA on deterministic vicco subsets |
| `data/wrsa_low_level_subsets_summary.csv` | 02c | per-subset mean wRSA + mean low-level |
| `data/wrsa_low_level_subsets_comparison.csv` | 02c | cstim vs matched-/top-vicco subset drops |
| `data/disagreement_vs_ood_pairs.csv` | 04 | per (set × variant × pair × ref_for_set) |
| `data/disagreement_vs_ood_images.csv` | 04 | per (set × variant × image), incl. residual |
| `data/disagreement_vs_ood_summary.csv` | 04 | per (set × variant) Spearman + residual |
| `data/dissociation_summary.csv` | 05 | per-axis Cohen's d (cstim vs baseline) and per-set Δwrsa with subject-cluster bootstrap CIs |

## Headline finding

The cstim alignment drop is **not** explained by either OOD-ness or
low-level distribution shift. The strongest evidence comes from 02
(matched-OOD bootstrap baselines, where matchable) and 02c (deterministic
high-low-level vicco subsets). 04 adds an independent image-level test
showing model disagreement on cstim exceeds what OOD predicts.

- **02 (OOD-matching).** Where matching is feasible (33% of cells),
  OOD-matched bootstrap baselines do not reproduce the cstim drop.
  Drop_vs_matched: all_models −0.238 (n=33), architecture −0.068 (n=17),
  sota −0.065 (n=13), training_objective −0.287 (n=5). `dataset` cstim is
  beyond bootstrap OOD range; the test is undefined there.
- **02c (low-level matching, deterministic subsets).** vicco wRSA varies
  only 0.04 across mean low-level distance from 2.22 to 4.25. Cstim
  drops 0.13 to 0.24 wRSA below the matched-low-level vicco subset for
  every set, including the top100 vicco subset whose mean low-level
  distance exceeds all_models's. **Low-level shift does not produce the
  drop.**
- **03 + 03b (paired-difference regression).** At the set level (n=5),
  Δlow and Δood are both Spearman-correlated with Δwrsa (ρ = ∓0.90,
  p = 0.04). Permutation p on β_low = 0.17. With 5 set-level points and
  Δlow / Δood collinear, this analysis is descriptive: drop covaries
  with set-level shift but mechanism cannot be identified from the
  regression alone. 02c shows the across-set covariation reflects
  selection-induced confounding, not low-level causation.
- **04 (disagreement vs OOD, image level).** Pair-level r(disagreement,
  OOD) is ≤ 0.09 across all sets. Image-level residual disagreement
  (against matched-roster vicco reference): all_models +0.28,
  training_objective +0.27, sota +0.12, dataset +0.12, architecture
  −0.006 (architecture's narrow 5-model roster doesn't admit
  disagreement signal beyond OOD).

> **Suggested sentence for the paper:** *The cstim alignment drop is not
> explained by out-of-distribution-ness on the encoders' training
> distribution, nor by low-level distribution shift. Vicco wRSA is flat
> across vicco subsets spanning the cstim low-level distance range, and
> wherever the baseline pool admits OOD-matched bootstrap subsets, those
> do not reproduce the drop either. The drop reflects model
> disagreement signal on the selected stimuli that exceeds what either
> OOD or low-level shift would predict.*

## Caveats explicitly preserved here

- The vicco baseline pool (n=292) does not span the OOD range of all cstim
  sets. The `dataset` set is entirely out of range; for cells where
  `match_quality == out_of_range` the comparison reads as "even the
  most-OOD baselines we have don't show this drop", not "OOD-equating
  fails to reproduce the drop". The `match_quality` column makes this
  explicit per cell.
- The paired-difference regression's standard errors come from a
  subject-cluster bootstrap. Within-subject row dependence (same model
  appearing in multiple model_sets) is not resampled; this is treated as
  a descriptive control rather than a formal test.
- Disagreement is computed using the same model roster on cstim and on the
  vicco reference for each set, so magnitudes are matched. But variance
  estimators with M=4–5 models are noisier than with M=20; small-set
  residuals should be interpreted with that in mind.
- For `architecture`, the near-zero residual could either reflect "no
  signal beyond OOD" or that the 5 architecture models occupy a too-narrow
  disagreement axis to express residual signal. We do not distinguish
  between these from this analysis alone.

<!-- BEGIN AUTO-FIGURE-PROVENANCE -->
## Contents Snapshot

- Folder: `05_controls_and_supplementary/low_level_and_ood/ood_controls`
- Figures in this folder tree: 20
- Data/table-like files in this folder tree: 16
- Python scripts in this folder tree: 32
- Main child folders: `code/`, `results/`, `figures/`

Use the tables below as a trace from rendered files back to the nearby code, staged data, score tables, or reports that produced them.

## Figure Index

| Figure directory | Figures | README |
|---|---:|---|
| `05_controls_and_supplementary/low_level_and_ood/ood_controls/figures` | 2 | `05_controls_and_supplementary/low_level_and_ood/ood_controls/figures/README.md` |
| `05_controls_and_supplementary/low_level_and_ood/ood_controls/figures/png` | 2 | `05_controls_and_supplementary/low_level_and_ood/ood_controls/figures/png/README.md` |
| `05_controls_and_supplementary/low_level_and_ood/ood_controls/figures/supplementary` | 8 | `05_controls_and_supplementary/low_level_and_ood/ood_controls/figures/supplementary/README.md` |
| `05_controls_and_supplementary/low_level_and_ood/ood_controls/figures/supplementary/png` | 8 | `05_controls_and_supplementary/low_level_and_ood/ood_controls/figures/supplementary/png/README.md` |
<!-- END AUTO-FIGURE-PROVENANCE -->
