#!/bin/bash
#
# Queue a dependency-gated resume for the Raven pool-size feature sweep.
#
# Usage:
#   CURRENT_JOB_ID=<running_job_id> bash 00_stimulus_selection/feature_method_sweep/code/queue_pool_size_sweep_resume_slurm.sh
#
# Optional overrides should match the original run if you changed defaults:
#   OUTPUT_ROOT=... POOL_SIZES=... METHODS=... BATCH_SIZE=... MEM=... CURRENT_JOB_ID=... bash ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

RUN_NAME="${RUN_NAME:-sota_pool_size_sweep_new_methods_20260616_171946}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/00_stimulus_selection/feature_method_sweep/results/${RUN_NAME}}"
CURRENT_JOB_ID="${CURRENT_JOB_ID:-}"
CURRENT_JOB_PATTERN="${CURRENT_JOB_PATTERN:-pool|feature|sweep}"

if [[ -z "${CURRENT_JOB_ID}" ]]; then
  mapfile -t MATCHING_JOBS < <(
    squeue -h -u "${USER}" -o "%A %j %T" \
      | awk -v pat="${CURRENT_JOB_PATTERN}" 'tolower($0) ~ tolower(pat) {print}'
  )
  if [[ "${#MATCHING_JOBS[@]}" -eq 1 ]]; then
    CURRENT_JOB_ID="$(awk '{print $1}' <<<"${MATCHING_JOBS[0]}")"
  else
    echo "Could not infer exactly one current pool-sweep job." >&2
    echo "Set CURRENT_JOB_ID explicitly. Current jobs:" >&2
    squeue -u "${USER}" >&2 || true
    exit 2
  fi
fi

if [[ ! -d "${OUTPUT_ROOT}" ]]; then
  echo "Warning: OUTPUT_ROOT does not exist yet: ${OUTPUT_ROOT}" >&2
fi

mkdir -p "${OUTPUT_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUBMIT_SCRIPT="${OUTPUT_ROOT}/resume_after_${CURRENT_JOB_ID}_${STAMP}.slurm.sh"
SUBMIT_LOG="${OUTPUT_ROOT}/resume_after_${CURRENT_JOB_ID}_${STAMP}.%j.log"

cat > "${SUBMIT_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH --job-name=submit_pool_sweep_resume
#SBATCH --time=00:10:00
#SBATCH --mem=1G
#SBATCH --output=${SUBMIT_LOG}
#SBATCH --error=${SUBMIT_LOG}

set -euo pipefail

cd "${REPO_ROOT}"
EOF

write_export_if_set() {
  local name="$1"
  local value="${!name-}"
  if [[ -n "${value}" ]]; then
    printf 'export %s=%q\n' "${name}" "${value}" >> "${SUBMIT_SCRIPT}"
  fi
}

printf 'export OUTPUT_ROOT=%q\n' "${OUTPUT_ROOT}" >> "${SUBMIT_SCRIPT}"
printf 'export EXTRA_ARGS=%q\n' "${EXTRA_ARGS:+${EXTRA_ARGS} }--resume" >> "${SUBMIT_SCRIPT}"

for name in \
  MODEL_SET METHODS POOL_SIZES TARGET_SIZE MAX_RAM_GB MEM BATCH_SIZE \
  ADAPTIVE_BATCH_SIZE MAX_BATCH_SIZE MIN_BATCH_SIZE PROGRESS_EVERY_BATCHES \
  SEED SKIP_EVAL SHARED_ENCODINGS PYTHON_BIN CONDA_LIB CSTIMS_SLURM_WRAPPER; do
  write_export_if_set "${name}"
done

cat >> "${SUBMIT_SCRIPT}" <<EOF

echo "Submitting pool-size sweep resume at \$(date --iso-8601=seconds)"
echo "Repo: ${REPO_ROOT}"
echo "Output root: \${OUTPUT_ROOT}"
echo "Extra args: \${EXTRA_ARGS}"

bash "${SCRIPT_DIR}/run_pool_size_sweep_slurm.sh"
EOF

chmod +x "${SUBMIT_SCRIPT}"

echo "Queueing resume submitter after current job ${CURRENT_JOB_ID}"
echo "Submitter script: ${SUBMIT_SCRIPT}"

sbatch --dependency="afterany:${CURRENT_JOB_ID}" "${SUBMIT_SCRIPT}"
