#!/bin/bash
#
# Queue nested pool-size feature-method sweeps for the non-SOTA model sets.
#
# This mirrors run_pool_size_sweep_slurm.sh, but fans it out over model sets.
# If a current pool/sweep job can be inferred from squeue, the submitter is
# queued with afterany:<jobid> so these sweeps start after the current run exits.
#
# Common usage on Raven:
#   bash 00_stimulus_selection/feature_method_sweep/code/queue_other_model_set_pool_size_sweeps_slurm.sh
#
# Explicit dependency:
#   DEPENDENCY_JOB_ID=<job_id> bash 00_stimulus_selection/feature_method_sweep/code/queue_other_model_set_pool_size_sweeps_slurm.sh
#
# Override model sets or resume existing output dirs:
#   MODEL_SETS=training_objective,architecture,dataset RESUME=1 bash ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-pool_size_sweep_new_methods}"
MODEL_SETS="${MODEL_SETS:-all_models,training_objective,architecture,dataset}"

METHODS="${METHODS:-raw_only_mean_min,raw_only_mean_min_no_attenuation,sub01_only_mean_min,sub01_only_mean_min_no_attenuation,raw_enc_w05_mean_min,raw_enc_w05_mean_min_no_attenuation}"
POOL_SIZES="${POOL_SIZES:-1k,10k,50k,100k,250k,500k,1M,5M,10M}"
TARGET_SIZE="${TARGET_SIZE:-100}"
MAX_RAM_GB="${MAX_RAM_GB:-300}"
MEM="${MEM:-380000}"
BATCH_SIZE="${BATCH_SIZE:-5000}"
ADAPTIVE_BATCH_SIZE="${ADAPTIVE_BATCH_SIZE:-1}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-20000}"
MIN_BATCH_SIZE="${MIN_BATCH_SIZE:-512}"
PROGRESS_EVERY_BATCHES="${PROGRESS_EVERY_BATCHES:-50}"
SEED="${SEED:-42}"
SKIP_EVAL="${SKIP_EVAL:-1}"
RESUME="${RESUME:-0}"
SUBMIT="${SUBMIT:-1}"

RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/00_stimulus_selection/feature_method_sweep/results}"
QUEUE_DIR="${QUEUE_DIR:-${RESULTS_ROOT}/other_model_set_pool_sweep_queue_${RUN_STAMP}}"
mkdir -p "${QUEUE_DIR}"

DEPENDENCY_JOB_ID="${DEPENDENCY_JOB_ID:-${CURRENT_JOB_ID:-}}"
AUTO_DEPENDENCY="${AUTO_DEPENDENCY:-1}"
REQUIRE_DEPENDENCY="${REQUIRE_DEPENDENCY:-0}"
DEPENDENCY_JOB_PATTERN="${DEPENDENCY_JOB_PATTERN:-pool|feature|sweep}"

if [[ -z "${DEPENDENCY_JOB_ID}" && "${AUTO_DEPENDENCY}" == "1" ]]; then
  if command -v squeue >/dev/null 2>&1; then
    mapfile -t MATCHING_JOBS < <(
      squeue -h -u "${USER}" -o "%A %j %T" \
        | awk -v pat="${DEPENDENCY_JOB_PATTERN}" 'tolower($0) ~ tolower(pat) {print}'
    )
    if [[ "${#MATCHING_JOBS[@]}" -eq 1 ]]; then
      DEPENDENCY_JOB_ID="$(awk '{print $1}' <<<"${MATCHING_JOBS[0]}")"
      echo "Inferred dependency job ${DEPENDENCY_JOB_ID}: ${MATCHING_JOBS[0]}"
    elif [[ "${#MATCHING_JOBS[@]}" -gt 1 ]]; then
      echo "Could not infer a unique dependency job; matching jobs were:" >&2
      printf '  %s\n' "${MATCHING_JOBS[@]}" >&2
    else
      echo "No matching dependency job found for pattern '${DEPENDENCY_JOB_PATTERN}'." >&2
    fi
  else
    echo "squeue is unavailable; submitting without an inferred dependency." >&2
  fi
fi

if [[ -z "${DEPENDENCY_JOB_ID}" && "${REQUIRE_DEPENDENCY}" == "1" ]]; then
  echo "No dependency job available and REQUIRE_DEPENDENCY=1." >&2
  exit 2
fi

EXTRA_ARGS_COMBINED="${EXTRA_ARGS:-}"
if [[ "${RESUME}" == "1" && " ${EXTRA_ARGS_COMBINED} " != *" --resume "* ]]; then
  EXTRA_ARGS_COMBINED="${EXTRA_ARGS_COMBINED:+${EXTRA_ARGS_COMBINED} }--resume"
fi

SUBMIT_SCRIPT="${QUEUE_DIR}/submit_other_model_set_pool_sweeps_${RUN_STAMP}.slurm.sh"
SUBMIT_LOG="${QUEUE_DIR}/submit_other_model_set_pool_sweeps_${RUN_STAMP}.%j.log"

cat > "${SUBMIT_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH --job-name=submit_other_pool_sweeps
#SBATCH --time=00:20:00
#SBATCH --mem=1G
#SBATCH --output=${SUBMIT_LOG}
#SBATCH --error=${SUBMIT_LOG}

set -euo pipefail

cd "${REPO_ROOT}"
EOF

write_export() {
  local name="$1"
  local value="${!name-}"
  printf 'export %s=%q\n' "${name}" "${value}" >> "${SUBMIT_SCRIPT}"
}

for name in \
  RUN_STAMP RUN_TAG MODEL_SETS METHODS POOL_SIZES TARGET_SIZE MAX_RAM_GB MEM \
  BATCH_SIZE ADAPTIVE_BATCH_SIZE MAX_BATCH_SIZE MIN_BATCH_SIZE \
  PROGRESS_EVERY_BATCHES SEED SKIP_EVAL RESULTS_ROOT EXTRA_ARGS_COMBINED \
  PYTHON_BIN CONDA_LIB CSTIMS_SLURM_WRAPPER POOL_FEATURE_DIR RANDOM_FEATURE_DIR \
  N_RANDOM_IMAGES MAX_IMAGES N_RANDOM_SUBSETS N_NOISE_SAMPLES N_BOOTSTRAP \
  RUN_LOCAL; do
  write_export "${name}"
done

cat >> "${SUBMIT_SCRIPT}" <<'EOF'

echo "Submitting other-model-set pool-size sweeps at $(date --iso-8601=seconds)"
echo "Repo: $(pwd)"
echo "Model sets: ${MODEL_SETS}"
echo "Methods: ${METHODS}"
echo "Pool sizes: ${POOL_SIZES}"

for model_set in ${MODEL_SETS//,/ }; do
  export MODEL_SET="${model_set}"
  export TIMESTAMP="${RUN_STAMP}"
  export OUTPUT_ROOT="${RESULTS_ROOT}/${model_set}_${RUN_TAG}_${RUN_STAMP}"
  export EXTRA_ARGS="${EXTRA_ARGS_COMBINED}"

  echo
  echo "[$(date --iso-8601=seconds)] submitting MODEL_SET=${MODEL_SET}"
  echo "  OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "  EXTRA_ARGS=${EXTRA_ARGS:-<none>}"

  bash "00_stimulus_selection/feature_method_sweep/code/run_pool_size_sweep_slurm.sh"
done

echo "Submitter done at $(date --iso-8601=seconds)"
EOF

chmod +x "${SUBMIT_SCRIPT}"

echo "Prepared other-model-set pool-size sweep submitter:"
echo "  ${SUBMIT_SCRIPT}"
echo "Result dirs will be:"
for model_set in ${MODEL_SETS//,/ }; do
  echo "  ${RESULTS_ROOT}/${model_set}_${RUN_TAG}_${RUN_STAMP}"
done

if [[ "${SUBMIT}" == "0" ]]; then
  echo "SUBMIT=0, not calling sbatch. Run this manually:"
  echo "  bash ${SUBMIT_SCRIPT}"
  exit 0
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable. Submitter script was written but not queued." >&2
  exit 1
fi

SBATCH_ARGS=()
if [[ -n "${DEPENDENCY_JOB_ID}" ]]; then
  SBATCH_ARGS+=(--dependency="afterany:${DEPENDENCY_JOB_ID}")
  echo "Queueing with dependency afterany:${DEPENDENCY_JOB_ID}"
else
  echo "Queueing without dependency"
fi

sbatch "${SBATCH_ARGS[@]}" "${SUBMIT_SCRIPT}"
