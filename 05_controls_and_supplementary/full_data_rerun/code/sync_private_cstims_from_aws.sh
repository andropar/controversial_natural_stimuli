#!/usr/bin/env bash
set -euo pipefail

LAION_FMRI_ROOT="${LAION_FMRI_ROOT:-/data/home_roth/datasets/LAION-fMRI}"
DEST="${LAION_FMRI_ROOT}/derivatives/glmsingle-tedana"

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI is not available on PATH. Install/activate it before syncing." >&2
  exit 2
fi

if [[ -z "${CSTIMS_AWS_URI:-}" ]]; then
  echo "Set CSTIMS_AWS_URI to the private S3 prefix, e.g. s3://bucket/path/glmsingle-tedana" >&2
  exit 2
fi

AWS_PROFILE_ARG=()
if [[ -n "${AWS_PROFILE:-}" ]]; then
  AWS_PROFILE_ARG=(--profile "${AWS_PROFILE}")
fi

mkdir -p "${DEST}"

echo "Syncing private cstim sessions from ${CSTIMS_AWS_URI}"
echo "Destination: ${DEST}"

aws "${AWS_PROFILE_ARG[@]}" s3 sync "${CSTIMS_AWS_URI%/}/" "${DEST}/" \
  --exclude "*" \
  --include "sub-*/ses-32/func/*_desc-SingletrialBetas_trials.tsv" \
  --include "sub-*/ses-32/func/*_stat-effect_desc-SingletrialBetas_statmap.json" \
  --include "sub-*/ses-32/func/*_stat-effect_desc-SingletrialBetas_statmap.nii.gz" \
  --include "sub-*/ses-33/func/*_desc-SingletrialBetas_trials.tsv" \
  --include "sub-*/ses-33/func/*_stat-effect_desc-SingletrialBetas_statmap.json" \
  --include "sub-*/ses-33/func/*_stat-effect_desc-SingletrialBetas_statmap.nii.gz" \
  --include "sub-*/ses-34/func/*_desc-SingletrialBetas_trials.tsv" \
  --include "sub-*/ses-34/func/*_stat-effect_desc-SingletrialBetas_statmap.json" \
  --include "sub-*/ses-34/func/*_stat-effect_desc-SingletrialBetas_statmap.nii.gz" \
  --no-progress

echo "Done. Verify with:"
echo "  find ${DEST} -path '*/ses-3*/func/*SingletrialBetas*' | sort"
