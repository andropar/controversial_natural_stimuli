# Simple Semantic Annotation Audit

Input: `05_controls_and_supplementary/low_level_and_ood/human_ratings/outputs/annotations/full_minimax_m3_all_stimuli.csv`

This is a descriptive VLM-caption/annotation audit. It is not a human behavioral result and not a causal control.

## Largest numeric shifts by absolute Cohen's d

| condition_label | field_label | baseline_mean | condition_mean | mean_difference | cohens_d |
| --- | --- | --- | --- | --- | --- |
| SOTA | visual clutter | 2.976 | 2.290 | -0.686 | -0.822 |
| Data | visual clutter | 2.976 | 2.300 | -0.676 | -0.807 |
| SOTA | natural-photo typicality | 4.123 | 3.580 | -0.543 | -0.699 |
| Arch. | natural-photo typicality | 4.123 | 3.600 | -0.523 | -0.697 |
| SOTA | scene-centricity | 3.318 | 2.480 | -0.838 | -0.677 |
| All-model | natural-photo typicality | 4.123 | 3.620 | -0.503 | -0.622 |
| Data | distinct object categories | 3.832 | 2.950 | -0.882 | -0.582 |
| SOTA | distinct object categories | 3.832 | 3.000 | -0.832 | -0.572 |
| Train. | visual clutter | 2.976 | 2.530 | -0.446 | -0.555 |
| All-model | distinct object categories | 3.832 | 3.030 | -0.802 | -0.534 |
| Arch. | visual clutter | 2.976 | 2.540 | -0.436 | -0.532 |
| Data | object-centricity | 2.942 | 3.560 | 0.618 | 0.525 |
| All-model | visual clutter | 2.976 | 2.570 | -0.406 | -0.494 |
| Arch. | scene-centricity | 3.318 | 3.910 | 0.592 | 0.482 |
| Train. | scene-centricity | 3.318 | 2.770 | -0.548 | -0.453 |

## Largest boolean shifts by absolute prevalence difference

| condition_label | field_label | baseline_prevalence | condition_prevalence | prevalence_difference |
| --- | --- | --- | --- | --- |
| Train. | face | 0.144 | 0.620 | 0.476 |
| SOTA | person | 0.322 | 0.790 | 0.468 |
| Train. | person | 0.322 | 0.760 | 0.438 |
| Arch. | indoor scene | 0.432 | 0.840 | 0.408 |
| SOTA | face | 0.144 | 0.520 | 0.376 |
| Arch. | outdoor scene | 0.568 | 0.200 | -0.368 |
| Arch. | person | 0.322 | 0.080 | -0.242 |
| All-model | outdoor scene | 0.568 | 0.330 | -0.238 |
| All-model | person | 0.322 | 0.540 | 0.218 |
| SOTA | outdoor scene | 0.568 | 0.360 | -0.208 |
| Data | outdoor scene | 0.568 | 0.370 | -0.198 |
| Data | person | 0.322 | 0.140 | -0.182 |
| Arch. | animal | 0.199 | 0.020 | -0.179 |
| SOTA | unusual viewpoint | 0.277 | 0.120 | -0.157 |
| All-model | animal | 0.199 | 0.050 | -0.149 |

## Largest dominant-content/style/domain prevalence shifts

| condition_label | field | category | baseline_prevalence | condition_prevalence | prevalence_difference |
| --- | --- | --- | --- | --- | --- |
| Data | dominant_content_type | object | 0.260 | 0.580 | 0.320 |
| Train. | dominant_content_type | person_face | 0.072 | 0.360 | 0.288 |
| SOTA | dominant_content_type | person_face | 0.072 | 0.340 | 0.268 |
| Arch. | dominant_content_type | scene | 0.363 | 0.630 | 0.267 |
| SOTA | dominant_content_type | scene | 0.363 | 0.120 | -0.243 |
| Train. | dominant_content_type | scene | 0.363 | 0.130 | -0.233 |
| SOTA | image_style | edited_photo | 0.007 | 0.210 | 0.203 |
| SOTA | image_style | natural_photo | 0.986 | 0.790 | -0.196 |
| Arch. | image_style | natural_photo | 0.986 | 0.800 | -0.186 |
| All-model | image_style | natural_photo | 0.986 | 0.810 | -0.176 |
| All-model | image_style | edited_photo | 0.007 | 0.160 | 0.153 |
| Arch. | image_style | edited_photo | 0.007 | 0.150 | 0.143 |
| All-model | dominant_content_type | object | 0.260 | 0.400 | 0.140 |
| Arch. | dominant_content_type | animal | 0.123 | 0.000 | -0.123 |
| All-model | dominant_content_type | animal | 0.123 | 0.000 | -0.123 |
| Train. | image_style | edited_photo | 0.007 | 0.120 | 0.113 |
| Train. | image_style | natural_photo | 0.986 | 0.880 | -0.106 |
| Data | dominant_content_type | animal | 0.123 | 0.030 | -0.093 |
| All-model | dominant_content_type | scene | 0.363 | 0.270 | -0.093 |
| SOTA | dominant_content_type | animal | 0.123 | 0.040 | -0.083 |

## Top caption-token increases by condition

- All-model: product (+0.29), design (+0.10), decor (+0.10), event (+0.10), interior (+0.09), listing (+0.09), fashion (+0.08), lifestyle (+0.07), room (+0.07), catalog (+0.07), clothing (+0.07), sports (+0.06)
- Arch.: interior (+0.40), home (+0.28), room (+0.24), product (+0.19), residential (+0.18), design (+0.18), furniture (+0.17), table (+0.15), staged (+0.15), wall (+0.14), showroom (+0.14), modern (+0.14)
- Data: product (+0.21), decor (+0.16), decorative (+0.14), display (+0.14), wall (+0.14), interior (+0.12), home (+0.11), against (+0.10), lifestyle (+0.08), room (+0.08), design (+0.08), brick (+0.07)
- SOTA: woman (+0.26), product (+0.25), fashion (+0.20), portrait (+0.19), against (+0.15), man (+0.15), editorial (+0.12), catalog (+0.12), lifestyle (+0.11), dress (+0.11), clothing (+0.11), event (+0.11)
- Train.: portrait (+0.23), woman (+0.15), man (+0.14), product (+0.13), smiling (+0.11), lifestyle (+0.10), posing (+0.09), fashion (+0.08), shirt (+0.08), two (+0.08), dress (+0.07), moment (+0.07)
