# VGG Cache Repair Archive

Backup/provenance files from the VGG-16 SRP5920 repair on 2026-06-10.

- `results/target_adaptation_weighted_scores.csv.before_vgg_cache_repair`:
  full score table before recomputing the VGG rows.
- `results/target_adaptation_weighted_scores_vgg_repair_tmp.csv`: temporary
  table containing the recomputed VGG rows that were merged back into the
  current score table.

See `../../README.md` for the method note explaining the VGG feature-cache
context mismatch.
