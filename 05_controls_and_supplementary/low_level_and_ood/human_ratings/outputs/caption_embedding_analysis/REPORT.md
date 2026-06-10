# Caption-Text Embedding Semantic Audit

Input: `/data/home_roth/cstims_share/05_controls_and_supplementary/low_level_and_ood/human_ratings/outputs/annotations/full_minimax_m3_all_stimuli.csv`

Embeddings: `sentence-transformers/all-MiniLM-L6-v2` over `short_caption`, L2-normalized.
This is a simple descriptive semantic-space diagnostic, not a causal control.

## Embedding Metadata

- embedding_method: sentence_transformer
- embedding_model: sentence-transformers/all-MiniLM-L6-v2
- n_documents: 792
- n_dimensions: 384
- normalized_embeddings: True

## CSTIM-vs-Baseline Summary

| condition_label | cohens_d_distance_to_baseline_centroid | pre_match_centroid_distance | post_match_centroid_distance | fraction_centroid_distance_closed_by_matching | mean_nearest_match_distance |
| --- | --- | --- | --- | --- | --- |
| All-model | 0.219 | 0.186 | 0.156 | 0.159 | 1.061 |
| Arch. | -0.454 | 0.305 | 0.185 | 0.394 | 1.028 |
| Data | -0.336 | 0.184 | 0.133 | 0.276 | 1.020 |
| SOTA | 0.604 | 0.255 | 0.184 | 0.279 | 1.083 |
| Train. | 0.858 | 0.217 | 0.173 | 0.202 | 1.085 |
