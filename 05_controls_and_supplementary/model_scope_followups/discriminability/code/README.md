# discriminability — Result 4: do controversial stimuli make models empirically more separable?

This is the new main-result component for the cstim paper, addressing the
underdiagnostic-benchmark question directly. Three sub-analyses, all on the
20 paper models × 5 subjects, paper-layer features and encoding models.

## TL;DR

**mRSA-transfer:**
- 65% of model pairs that are statistically tied on baseline (paired-t,
  FDR q>0.05) become separated on `cstim_all_models` (FDR q<0.05). Net
  gain: +18 pairs of empirical discriminability over baseline.
- At the operational sample size (n=100), cstim_all_models separates 152
  out of 190 pairs vs 116 for vicco at the same n.
- However, **cstim is sample-efficient only at n ≥ ~40**. At n ≤ 20,
  vicco baseline actually separates *more* pairs than cstim. The crossover
  is around n=30–40.

**fRSA:**
- Cstim is **not** more diagnostic than vicco baseline at any sample size
  on fRSA. Net loss of 35 separable pairs at n=100. The discriminability
  gain is metric-specific and does not generalize to the unconditional fRSA
  metric.

**Practical implication:** the "cstim improves model comparison" claim is
mRSA-specific and applies at moderate-to-large sample sizes (n ≥ ~40).
Don't claim cstim is universally more sample-efficient — it isn't.

## Which models actually get differentiated (analysis 04, mRSA, all_models)

**Per-model conversion profile** — % of tied near-ceiling pairs containing
the model that become separated on cstim:

| 100% conversion | High (75–90%)              | Mid (50–70%)         | Low (≤35%) | 0%        |
|-----------------|----------------------------|----------------------|------------|-----------|
| SLIP (9/9), Sup-RN50 (5/5), CLIP-OAI (1/1) | MoCoV2 (7/8), CLIP-L400 (4/5), SigLIP (7/9), DINOv2 (10/13) | Sup (4/6), VICReg (4/7), CLIP-L2B (4/7), DFN-2B (4/7), BarlowTw (5/9), MC-Full (6/11) | VGG-16 (1/3), ViT-L/16 (1/3), MC-400M (2/9) | CORnet-S (0/2) |

ConvNeXt-B, Robust-L2, SimCLR-ViT had no tied-near-ceiling pairs (they're
already separable from every near-ceiling model on the baseline).

**Architecture-family conversion** — does cstim distinguish *same-architecture,
different-training* models that vicco can't?

| Family             | n_pairs | tied (NC) | converted | conv. rate | lost |
|--------------------|--------:|----------:|----------:|-----------:|-----:|
| Within ResNet-50   | 15      | 4         | 2         | 50%        | 4    |
| Within ViT-L       | 45      | 18        | 11        | 61%        | 3    |
| Across architectures | 130   | 35        | 24        | 69%        | 12   |

Within-family conversion is **slightly lower** than across-family — cstim
isn't preferentially distinguishing same-architecture-different-training
models. It's broadly diagnostic: for any pair of models that's tied on
baseline, cstim has roughly 50–70% chance of separating them, regardless
of architectural similarity.

**Effect-size shift** — per-pair |Δ_cstim| − |Δ_vicco|:

| Outcome              | n   | median |Δ| gain | mean   |
|----------------------|-----|----------------:|-------:|
| converted            | 37  | +0.085          | +0.091 |
| stayed tied          | 20  | +0.025          | +0.023 |
| remained separated   | 114 | −0.012          | −0.033 |
| lost                 | 19  | −0.060          | −0.057 |

Converted pairs gain ~0.09 mRSA in pairwise difference on cstim vs vicco.
Lost pairs lose ~0.06.

**Where cstim blurs distinctions baseline made (loss profile):**

ViT-L/16 (the standard supervised ViT) loses **6** pair-distinctions on
cstim — the most of any model. Followed by Sup-RN50 (4), MoCoV2 (4),
SimCLR-ViT (3), MC-400M (3). For the rest, ≤2 pairs are lost. So the
"cstim blurs distinctions" failure mode is partly concentrated on a
small set of models, with ViT-L/16 the most-affected outlier.

**Direction analysis:** of the 114 pairs separated on both vicco and
cstim, only **3 flip rank order** (cstim's mean Δ has the opposite sign
from vicco's). When cstim makes a pair distinguishable, it almost always
agrees with vicco on which model is the better fit — cstim isn't
overturning rankings, it's adding new ones.

## Files

```
analysis/
  01_pair_separation.py                       # tied-on-baseline → separated-on-cstim
  01b_pair_separation_singlesubset.py         # sanity check with single-subset baseline
  02_sample_efficiency.py                     # # separated pairs vs n; rank stability vs n
  03_empirical_spread.py                      # consolidated spread metrics
  04_which_models_get_differentiated.py       # per-model, family, effect-size, losses

figures/
  pair_separation.{pdf,png}                   # contingency + per-set conversion
  sample_efficiency.{pdf,png}                 # 2x2: mRSA/fRSA × {separated, rank-stab}
  pair_structure.{pdf,png}                    # 4-panel: per-model, family, effect, losses

results/
  pair_separation_{mrsa,frsa}.csv             # per-pair test results
  pair_separation_summary.csv                 # set-level conversion summary
  pair_separation_singlesubset_*.csv          # sanity-check version
  sample_efficiency.csv                       # per (n, boot, set, metric) raw
  sample_efficiency_summary.csv               # mean/SEM per (n, set, metric)
  empirical_spread.csv                        # spread metrics per (set, metric)
  which_models_per_model_profile.csv          # per-model conversion + loss
  which_models_family_conversion.csv          # within/across-family rates
  which_models_pair_outcomes.csv              # per-pair outcome + effect-size shift
```

## Methodology

**Statistical test:** paired t-test across the 5 subjects on per-subject
mRSA (or fRSA) differences for each model pair. FDR-corrected within
(metric × set) using Benjamini–Hochberg. Wilcoxon signed-rank cannot be
used here: at n=5, its minimum two-sided p-value is 0.0625, which makes the
test useless under any FDR correction.

**Sample-size matching:** cstim has 100 images per set. Vicco has 292.
For pair-separation (a), the canonical baseline uses each subject's mean
mRSA across 1000 vicco bootstraps of 100 images — gives vicco an unfairly
precise baseline estimate, which makes the conversion-rate finding
*more* impressive (vicco is "as easy as possible" to detect differences
on, yet pairs still convert). The single-subset sanity check (01b) repeats
the analysis using one random vicco-100 subset at a time and averages over
100 such subsets; conversion rates are within ±5pp of the bootstrap-mean
version.

**"Tied" definition:** q_vicco > 0.05 (FDR-corrected). Two reported
flavors:
- *strict*: tied = simply not separable on baseline
- *near-ceiling*: also requires both models to have mean vicco mRSA > 0.30
  (or fRSA > 0.15). Filters out "tied because both are bad" pairs.

The headline numbers use the near-ceiling filter, which is the
substantively interesting "genuinely brain-aligned but indistinguishable"
case.

**Sample-efficiency design:** for each n in {10, 20, 40, 60, 80, 100}
and each random subset b (50 subsets per n), compute per-(subject, model)
RSA on the n-image subset, run the paired-t-per-pair + FDR pipeline,
count separated pairs. Compares cstim_all_models vs vicco at matched n.

**Rank-stability metric:** Spearman ρ between per-subject ranking on the
n-image subset and the per-subject ranking on the full 100-image set.
Averaged across 5 subjects, then across 50 bootstraps per n.

## Caveats

- **Per-controlled-set numbers are noisy.** Sets like `architecture` (5
  models, 10 pairs) give conversion rates that move sharply with single
  pair flips. Don't lean on per-set numbers in the abstract; the
  all_models result is the headline.
- **Selection bias is NOT an issue here** because the test uses the
  paper-layer features (no post-hoc layer selection) and the FDR
  correction is honest within set.
- **Single-pair Δs at n=5 subjects can be heavily influenced by one
  subject.** A subject-leave-one-out sensitivity check would be reasonable
  for a final paper version but is not yet included.

## Suggested writeup

> Of 190 model pairs in our 20-model panel, 57 (30%) are statistically
> indistinguishable on the vicco natural-image baseline (paired t,
> FDR q>0.05). On the all-models controversial-stimulus set, 37 of those 57
> (65%) become statistically distinguishable at FDR q<0.05 — a net gain of
> 18 pairs of empirical discriminability over baseline. At matched sample
> size (n=100), cstim_all_models supports 152 separable pairs vs 116 for
> the vicco baseline. The advantage is sample-size-dependent: at small n
> (≤20), vicco actually supports more pair separations than cstim, and the
> crossover occurs at n ≈ 30–40. The empirical-discriminability advantage
> of cstim is specific to the encoding-based mRSA-transfer metric; under
> fixed RSA (model RDM directly correlated to brain RDM), cstim does not
> exceed vicco in pair separability at any sample size.

This is the cleanest answer to the underdiagnostic-benchmark question
and supports the paper's narrative that cstim is empirically diagnostic
in the regime where modern model comparisons operate.
