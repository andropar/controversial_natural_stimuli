#!/bin/bash
#
# Submit the nested candidate-pool-size feature method sweep on Raven.
#
# This runs the current six feature-selection variants over multiple nested
# candidate-pool prefixes in one shared greedy pass. It writes one standard
# feature_method_sweep result directory per pool size under OUTPUT_ROOT.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

export MODEL_SET="${MODEL_SET:-sota}"
export METHODS="${METHODS:-raw_only_mean_min,raw_only_mean_min_no_attenuation,sub01_only_mean_min,sub01_only_mean_min_no_attenuation,raw_enc_w05_mean_min,raw_enc_w05_mean_min_no_attenuation}"
export POOL_SIZES="${POOL_SIZES:-1k,10k,50k,100k,250k,500k,1M,5M,10M}"
export TARGET_SIZE="${TARGET_SIZE:-100}"
export MAX_RAM_GB="${MAX_RAM_GB:-300}"
export MEM="${MEM:-380000}"
export BATCH_SIZE="${BATCH_SIZE:-5000}"
export ADAPTIVE_BATCH_SIZE="${ADAPTIVE_BATCH_SIZE:-1}"
export MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-20000}"
export MIN_BATCH_SIZE="${MIN_BATCH_SIZE:-512}"
export PROGRESS_EVERY_BATCHES="${PROGRESS_EVERY_BATCHES:-50}"
export SEED="${SEED:-42}"
export SKIP_EVAL="${SKIP_EVAL:-1}"
export SHARED_ENCODINGS="${SHARED_ENCODINGS:-1}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/00_stimulus_selection/feature_method_sweep/results/${MODEL_SET}_pool_size_sweep_new_methods_${TIMESTAMP}}"

echo "Nested feature-method pool-size sweep"
echo "Methods: ${METHODS}"
echo "Pool sizes: ${POOL_SIZES}"
echo "Shared encodings: ${SHARED_ENCODINGS}"
echo "Output root: ${OUTPUT_ROOT}"

exec "${SCRIPT_DIR}/run_feature_method_sweep_slurm.sh"
