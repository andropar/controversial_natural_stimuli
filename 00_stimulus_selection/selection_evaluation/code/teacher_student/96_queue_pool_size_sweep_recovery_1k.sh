#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_STAMP="${RUN_STAMP:-20260617_161534}"
RUN_TAG="${RUN_TAG:-pool_size_sweep_new_methods}"
MODEL_SETS="${MODEL_SETS:-sota,training_objective,architecture,dataset}"
METHODS="${METHODS:-raw_only_mean_min,raw_only_mean_min_no_attenuation,sub01_only_mean_min,sub01_only_mean_min_no_attenuation,raw_enc_w05_mean_min,raw_enc_w05_mean_min_no_attenuation}"
POOL_SIZES="${POOL_SIZES:-1k,10k,50k,100k,250k,500k,1M,5M,10M}"
TARGET_SIZE="${TARGET_SIZE:-100}"

FEATURE_RESULTS_ROOT="${FEATURE_RESULTS_ROOT:-${REPO_ROOT}/00_stimulus_selection/feature_method_sweep/results}"
TEACHER_ROOT="${TEACHER_ROOT:-/ptmp/rothj/controversial_natural_stimuli/00_stimulus_selection/selection_evaluation/feature_method_sweep_recovery/teacher_student}"
REFIT_SIZE="${REFIT_SIZE:-1000}"
REFIT_VAL_SIZE="${REFIT_VAL_SIZE:-200}"
MAX_REFIT_POOL_SIZE="${MAX_REFIT_POOL_SIZE:-10000}"
N_NOISE_SAMPLES="${N_NOISE_SAMPLES:-20}"
NOISE_MULTS="${NOISE_MULTS:-0.1,0.143844988829,0.206913808111,0.297635144163,0.428133239872,0.615848211066,0.88586679041,1,1.2742749857,1.83298071083,2.63665089873,3,3.79269019073,5,5.45559478117,7.84759970351,10}"
NOISE_MULTS_LABEL="${NOISE_MULTS_LABEL:-snr0p1to10}"
N_RANDOM_SUBSETS="${N_RANDOM_SUBSETS:-100}"
N_RANDOM_IMAGES="${N_RANDOM_IMAGES:-100000}"
N_REFIT_REPEATS="${N_REFIT_REPEATS:-3}"
TRACKS="${TRACKS:-raw,sub-01,sub-03,sub-05,sub-06,sub-07}"
TEACHER_CHUNK_SIZE="${TEACHER_CHUNK_SIZE:-auto}"
EVAL_REFIT_MODE="${EVAL_REFIT_MODE:-independent}"
RDM_CALIBRATION_COMPARISON="${RDM_CALIBRATION_COMPARISON:-clean_to_noisy}"
CSTIMS_PATH_CONFIG="${CSTIMS_PATH_CONFIG:-raven}"

OUT_RUN="${OUT_RUN:-${TEACHER_ROOT}/results/${RUN_TAG}_${RUN_STAMP}_refit${REFIT_SIZE}}"
STAGED_SELECTION_ROOT="${STAGED_SELECTION_ROOT:-${OUT_RUN}/payloads}"
RESULTS_NAME="${RESULTS_NAME:-teacher_student_independent_refit_refit${REFIT_SIZE}_rdm_score_spearman_response_empcal_${NOISE_MULTS_LABEL}_ns${N_NOISE_SAMPLES}_rand${N_RANDOM_SUBSETS}_rr${N_REFIT_REPEATS}_fastgpu}"
LOG_DIR="${LOG_DIR:-${TEACHER_ROOT}/logs/${RUN_TAG}_${RUN_STAMP}_refit${REFIT_SIZE}}"
QUEUE_DIR="${QUEUE_DIR:-${LOG_DIR}/queue}"
ARRAY_LOG_DIR="${ARRAY_LOG_DIR:-${LOG_DIR}/array}"
MANIFEST="${MANIFEST:-${QUEUE_DIR}/payload_manifest.tsv}"
CACHE_JOBS="${CACHE_JOBS:-${QUEUE_DIR}/cache_jobs.tsv}"
MERGE_JOBS="${MERGE_JOBS:-${QUEUE_DIR}/merge_jobs.tsv}"

PYTHON="${PYTHON:-/u/rothj/conda-envs/deepjuice/bin/python}"
SCRIPT="${SCRIPT:-${SCRIPT_DIR}/01_compute_independent_refit_rdm_recovery.py}"
RANDOM_FEATURE_DIR="${RANDOM_FEATURE_DIR:-${REPO_ROOT}/shared/cache_or_heavy/natural_pool_subset_100k_seed42}"
ENCODING_ROOT="${ENCODING_ROOT:-/u/rothj/cstims/experiments/encoding_fitting/results/encoding_20251222_141301}"

GPU_PARTITION="${GPU_PARTITION:-gpu}"
GPU_QOS="${GPU_QOS:-g0001}"
GPU_ACCOUNT="${GPU_ACCOUNT:-mnpf_gpu}"
GPUS_PER_CACHE_JOB="${GPUS_PER_CACHE_JOB:-4}"
CACHE_MAX_PROCS="${CACHE_MAX_PROCS:-${GPUS_PER_CACHE_JOB}}"
CACHE_GROUPS="${CACHE_GROUPS:-32}"
CACHE_CPUS_PER_TASK="${CACHE_CPUS_PER_TASK:-32}"
CACHE_MEM="${CACHE_MEM:-240000}"
CACHE_TIME="${CACHE_TIME:-1-00:00:00}"
MERGE_PARTITION="${MERGE_PARTITION:-${GPU_PARTITION}}"
MERGE_ACCOUNT="${MERGE_ACCOUNT:-${GPU_ACCOUNT}}"
MERGE_QOS="${MERGE_QOS:-}"
MERGE_GRES="${MERGE_GRES:-gpu:a100:1}"
MERGE_CPUS_PER_TASK="${MERGE_CPUS_PER_TASK:-8}"
MERGE_GROUPS="${MERGE_GROUPS:-8}"
MERGE_MAX_PROCS="${MERGE_MAX_PROCS:-4}"
MERGE_MEM="${MERGE_MEM:-64G}"
MERGE_TIME="${MERGE_TIME:-04:00:00}"
PREP_PARTITION="${PREP_PARTITION:-${GPU_PARTITION}}"
PREP_ACCOUNT="${PREP_ACCOUNT:-${GPU_ACCOUNT}}"
PREP_QOS="${PREP_QOS:-}"
PREP_GRES="${PREP_GRES:-gpu:a100:1}"
PREP_MEM="${PREP_MEM:-32G}"
PREP_TIME="${PREP_TIME:-04:00:00}"
SUMMARY_PARTITION="${SUMMARY_PARTITION:-${GPU_PARTITION}}"
SUMMARY_ACCOUNT="${SUMMARY_ACCOUNT:-${GPU_ACCOUNT}}"
SUMMARY_QOS="${SUMMARY_QOS:-}"
SUMMARY_GRES="${SUMMARY_GRES:-gpu:a100:1}"
SUMMARY_CPUS_PER_TASK="${SUMMARY_CPUS_PER_TASK:-4}"
SUMMARY_MEM="${SUMMARY_MEM:-16G}"
SUMMARY_TIME="${SUMMARY_TIME:-01:00:00}"

BUILD_NPY_CACHE="${BUILD_NPY_CACHE:-1}"
SUBMIT="${SUBMIT:-1}"
MODE="${MODE:-submit}"

export LD_LIBRARY_PATH="/u/rothj/conda-envs/deepjuice/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

mkdir -p "${OUT_RUN}" "${STAGED_SELECTION_ROOT}" "${LOG_DIR}" "${QUEUE_DIR}" "${ARRAY_LOG_DIR}"

write_env_exports() {
  local script_path="$1"
  for name in \
    REPO_ROOT TS RUN_STAMP RUN_TAG MODEL_SETS METHODS POOL_SIZES TARGET_SIZE \
    FEATURE_RESULTS_ROOT TEACHER_ROOT REFIT_SIZE REFIT_VAL_SIZE \
    MAX_REFIT_POOL_SIZE N_NOISE_SAMPLES NOISE_MULTS NOISE_MULTS_LABEL \
    N_RANDOM_SUBSETS N_RANDOM_IMAGES N_REFIT_REPEATS TRACKS \
    TEACHER_CHUNK_SIZE EVAL_REFIT_MODE \
    RDM_CALIBRATION_COMPARISON OUT_RUN \
    STAGED_SELECTION_ROOT RESULTS_NAME LOG_DIR QUEUE_DIR ARRAY_LOG_DIR \
    MANIFEST CACHE_JOBS MERGE_JOBS PYTHON SCRIPT RANDOM_FEATURE_DIR ENCODING_ROOT \
    CSTIMS_PATH_CONFIG \
    OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS PYTHONUNBUFFERED \
    GPUS_PER_CACHE_JOB CACHE_MAX_PROCS MERGE_MAX_PROCS CACHE_JOB_START \
    CACHE_JOB_END MERGE_JOB_START MERGE_JOB_END; do
    printf 'export %s=%q\n' "${name}" "${!name-}" >> "${script_path}"
  done
}

stage_payloads() {
  "${PYTHON}" - "${FEATURE_RESULTS_ROOT}" "${RUN_STAMP}" "${RUN_TAG}" \
    "${MODEL_SETS}" "${METHODS}" "${POOL_SIZES}" "${TARGET_SIZE}" \
    "${STAGED_SELECTION_ROOT}" "${MANIFEST}" <<'PY'
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

feature_root = Path(sys.argv[1])
run_stamp = sys.argv[2]
run_tag = sys.argv[3]
model_sets = [x.strip() for x in sys.argv[4].split(",") if x.strip()]
methods = [x.strip() for x in sys.argv[5].split(",") if x.strip()]
pool_tokens = [x.strip() for x in sys.argv[6].split(",") if x.strip()]
target_size = int(sys.argv[7])
staged_root = Path(sys.argv[8])
manifest_path = Path(sys.argv[9])


def parse_count(token: str) -> int:
    value = token.strip().lower().replace("_", "")
    if value.endswith("k"):
        return int(float(value[:-1]) * 1_000)
    if value.endswith("m"):
        return int(float(value[:-1]) * 1_000_000)
    return int(float(value))


def pool_dir_name(count: int) -> str:
    return f"pool_{count:09d}"


staged_root.mkdir(parents=True, exist_ok=True)
manifest_path.parent.mkdir(parents=True, exist_ok=True)
rows = []
errors = []
for model_set in model_sets:
    output_root = feature_root / f"{model_set}_{run_tag}_{run_stamp}"
    if not output_root.exists():
        errors.append(f"missing output root: {output_root}")
        continue
    for pool_token in pool_tokens:
        pool_size = parse_count(pool_token)
        pool_dir = output_root / pool_dir_name(pool_size)
        for method_id in methods:
            payload_dir = pool_dir / "payloads" / method_id
            selected_path = payload_dir / "selected_indices.npy"
            payload_path = payload_dir / "selected_stimuli_data.pkl"
            required = [
                selected_path,
                payload_path,
                payload_dir / "method_config.json",
                payload_dir / "selection_trace.csv",
                payload_dir / "selected_image_records.csv",
            ]
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                errors.extend(f"missing: {path}" for path in missing)
                continue
            try:
                n_selected = int(np.load(selected_path, mmap_mode="r").shape[0])
            except Exception as exc:
                errors.append(f"could not read {selected_path}: {exc}")
                continue
            if n_selected < target_size:
                errors.append(f"short: {selected_path}: {n_selected}/{target_size}")
                continue
            try:
                with payload_path.open("rb") as f:
                    payload = pickle.load(f)
                n_teachers = int(len(payload["model_names"]))
            except Exception as exc:
                errors.append(f"could not read {payload_path}: {exc}")
                continue
            eval_id = f"{model_set}__{pool_dir.name}__{method_id}"
            link = staged_root / eval_id
            target = payload_dir.resolve()
            if link.is_symlink() or not link.exists():
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(target)
            elif link.resolve() != target:
                errors.append(f"refusing to replace nonmatching staged path: {link}")
                continue
            rows.append(
                {
                    "eval_id": eval_id,
                    "model_set": model_set,
                    "pool_dir": pool_dir.name,
                    "pool_size": str(pool_size),
                    "method_id": method_id,
                    "payload_path": str(target),
                    "n_teachers": str(n_teachers),
                }
            )

if errors:
    for err in errors[:80]:
        print(err, file=sys.stderr)
    if len(errors) > 80:
        print(f"... {len(errors) - 80} more errors", file=sys.stderr)
    raise SystemExit(1)

with manifest_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "eval_id",
            "model_set",
            "pool_dir",
            "pool_size",
            "method_id",
            "payload_path",
            "n_teachers",
        ],
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"staged_payloads={len(rows)}")
print(f"manifest={manifest_path}")
PY
}

write_job_files() {
  "${PYTHON}" - "${MANIFEST}" "${CACHE_JOBS}" "${MERGE_JOBS}" \
    "${N_REFIT_REPEATS}" "${TEACHER_CHUNK_SIZE}" <<'PY'
import csv
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
cache_jobs_path = Path(sys.argv[2])
merge_jobs_path = Path(sys.argv[3])
n_refit_repeats = int(sys.argv[4])
chunk_arg = sys.argv[5]

with manifest_path.open() as f:
    manifest_rows = list(csv.DictReader(f, delimiter="\t"))


def teacher_ranges(n_teachers: int) -> list[str]:
    if chunk_arg == "auto":
        chunk_size = 3 if n_teachers <= 6 else 5
    else:
        chunk_size = int(chunk_arg)
    out = []
    start = 0
    while start < n_teachers:
        end = min(start + chunk_size - 1, n_teachers - 1)
        out.append(str(start) if start == end else f"{start}-{end}")
        start = end + 1
    return out


cache_jobs_path.parent.mkdir(parents=True, exist_ok=True)
with cache_jobs_path.open("w", newline="") as f:
    writer = csv.writer(f, delimiter="\t", lineterminator="\n")
    for row in manifest_rows:
        ranges = teacher_ranges(int(row["n_teachers"]))
        for refit_repeat in range(n_refit_repeats):
            for teacher_indices in ranges:
                writer.writerow([row["eval_id"], refit_repeat, teacher_indices])

with merge_jobs_path.open("w", newline="") as f:
    writer = csv.writer(f, delimiter="\t", lineterminator="\n")
    for row in manifest_rows:
        writer.writerow([row["eval_id"]])

print(f"cache_jobs={sum(1 for _ in cache_jobs_path.open())}")
print(f"merge_jobs={sum(1 for _ in merge_jobs_path.open())}")
PY
}

submit_sbatch() {
  if [[ "${SUBMIT}" == "0" ]]; then
    printf 'DRY_RUN %q' sbatch
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  sbatch --parsable "$@"
}

submit_prep_job() {
  local script_path="${QUEUE_DIR}/build_random_npy_cache_${TS}.slurm.sh"
  local log_path="${QUEUE_DIR}/build_random_npy_cache_${TS}.%j.log"
  cat > "${script_path}" <<EOF
#!/bin/bash
#SBATCH --job-name=ts_build_npy_cache
#SBATCH --partition=${PREP_PARTITION}
EOF
  if [[ -n "${PREP_ACCOUNT}" ]]; then
    printf '#SBATCH --account=%s\n' "${PREP_ACCOUNT}" >> "${script_path}"
  fi
  if [[ -n "${PREP_QOS}" ]]; then
    printf '#SBATCH --qos=%s\n' "${PREP_QOS}" >> "${script_path}"
  fi
  if [[ -n "${PREP_GRES}" ]]; then
    printf '#SBATCH --gres=%s\n' "${PREP_GRES}" >> "${script_path}"
  fi
  cat >> "${script_path}" <<EOF
#SBATCH --time=${PREP_TIME}
#SBATCH --mem=${PREP_MEM}
#SBATCH --output=${log_path}
#SBATCH --error=${log_path}

set -euo pipefail
cd "${REPO_ROOT}"
EOF
  write_env_exports "${script_path}"
  cat >> "${script_path}" <<'EOF'
"${PYTHON}" - "${REPO_ROOT}" "${MANIFEST}" "${RANDOM_FEATURE_DIR}" <<'PY'
import csv
import pickle
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
random_feature_dir = Path(sys.argv[3])
sys.path.insert(0, str(repo_root / "src"))

from cstims.evaluation.random_features import ensure_npy_feature_cache

models = set()
with manifest_path.open() as f:
    for row in csv.DictReader(f, delimiter="\t"):
        with (Path(row["payload_path"]) / "selected_stimuli_data.pkl").open("rb") as pf:
            payload = pickle.load(pf)
        models.update(payload["model_names"])

models = sorted(models)
print(f"building/checking npy cache for {len(models)} models")
paths = ensure_npy_feature_cache(random_feature_dir, models)
print(f"npy_cache_files={len(paths)}")
PY
EOF
  chmod +x "${script_path}"
  submit_sbatch "${script_path}"
}

submit_cache_array() {
  local dependency="${1:-}"
  local n_jobs
  n_jobs="$(wc -l < "${CACHE_JOBS}")"
  if (( n_jobs < 1 )); then
    echo "No cache jobs in ${CACHE_JOBS}" >&2
    exit 2
  fi
  local n_groups="${CACHE_GROUPS}"
  if (( n_groups > n_jobs )); then
    n_groups="${n_jobs}"
  fi
  local group_size=$(((n_jobs + n_groups - 1) / n_groups))
  local job_ids=()
  local start=1
  local group_idx=0
  while (( start <= n_jobs )); do
    local end=$((start + group_size - 1))
    if (( end > n_jobs )); then
      end="${n_jobs}"
    fi
    local script_path="${QUEUE_DIR}/cache_group_${TS}_${group_idx}_${start}_${end}.slurm.sh"
    local log_path="${ARRAY_LOG_DIR}/cache_group_${TS}_${group_idx}_${start}_${end}.%j.log"
    cat > "${script_path}" <<EOF
#!/bin/bash
#SBATCH --job-name=ts_pool1k_c${group_idx}
#SBATCH --partition=${GPU_PARTITION}
#SBATCH --qos=${GPU_QOS}
#SBATCH --account=${GPU_ACCOUNT}
#SBATCH --gres=gpu:a100:${GPUS_PER_CACHE_JOB}
#SBATCH --cpus-per-task=${CACHE_CPUS_PER_TASK}
#SBATCH --mem=${CACHE_MEM}
#SBATCH --time=${CACHE_TIME}
#SBATCH --output=${log_path}
#SBATCH --error=${log_path}

set -euo pipefail
cd "${REPO_ROOT}"
EOF
    CACHE_JOB_START="${start}" CACHE_JOB_END="${end}" write_env_exports "${script_path}"
    cat >> "${script_path}" <<'EOF'
run_cache_job() {
  set -euo pipefail
  local eval_id="$1"
  local refit_repeat="$2"
  local teacher_indices="$3"
  local out_dir="${OUT_RUN}/${RESULTS_NAME}/${eval_id}"
  local safe_indices="${teacher_indices//,/plus}"
  safe_indices="${safe_indices//-/to}"
  local log="${LOG_DIR}/cache_${TS}_${eval_id}_${safe_indices}_rr${refit_repeat}.log"
  local -a noise_mult_args=()
  if [[ -n "${NOISE_MULTS}" ]]; then
    noise_mult_args=(--noise-mults "${NOISE_MULTS}")
  fi
  {
    echo "job_start $(date -Is) eval_id=${eval_id} refit_repeat=${refit_repeat} teachers=${teacher_indices}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    "${PYTHON}" -u "${SCRIPT}" \
      --model-set "${eval_id}" \
      --selection-root "${STAGED_SELECTION_ROOT}" \
      --tracks "${TRACKS}" \
      --random-feature-dir "${RANDOM_FEATURE_DIR}" \
      --encoding-root "${ENCODING_ROOT}" \
      --n-random-images "${N_RANDOM_IMAGES}" \
      --refit-pool-size "${REFIT_SIZE}" \
      --refit-val-size "${REFIT_VAL_SIZE}" \
      --max-refit-pool-size "${MAX_REFIT_POOL_SIZE}" \
      --n-refit-repeats "${N_REFIT_REPEATS}" \
      --refit-repeat-indices "${refit_repeat}" \
      --n-random-subsets "${N_RANDOM_SUBSETS}" \
      --n-noise-samples "${N_NOISE_SAMPLES}" \
      "${noise_mult_args[@]}" \
      --eval-noise-mode response \
      --fit-noise-calibration rdm_empirical \
      --rdm-calibration-comparison "${RDM_CALIBRATION_COMPARISON}" \
      --eval-refit-mode "${EVAL_REFIT_MODE}" \
      --calibration-images 100 \
      --calibration-noise-samples 2 \
      --calibration-max-iter 8 \
      --corr-type spearman \
      --encoding-device cuda \
      --encoding-batch-size 1024 \
      --shared-encodings \
      --teacher-indices "${teacher_indices}" \
      --fast-gpu-batch \
      --cache-only \
      --output-dir "${out_dir}"
    echo "job_done $(date -Is) eval_id=${eval_id} refit_repeat=${refit_repeat} teachers=${teacher_indices}"
  } > "${log}" 2>&1
}
export -f run_cache_job

echo "cache_group_start $(date -Is) lines=${CACHE_JOB_START}-${CACHE_JOB_END}"
sed -n "${CACHE_JOB_START},${CACHE_JOB_END}p" "${CACHE_JOBS}" |
  xargs --process-slot-var=JOB_SLOT -P "${CACHE_MAX_PROCS}" -n 3 bash -c '
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES=$((JOB_SLOT % GPUS_PER_CACHE_JOB))
    run_cache_job "$@"
  ' _
echo "cache_group_done $(date -Is) lines=${CACHE_JOB_START}-${CACHE_JOB_END}"
EOF
    chmod +x "${script_path}"
    local job_id
    if [[ -n "${dependency}" ]]; then
      job_id="$(submit_sbatch "--dependency=afterok:${dependency}" "${script_path}")"
    else
      job_id="$(submit_sbatch "${script_path}")"
    fi
    job_ids+=("${job_id}")
    start=$((end + 1))
    group_idx=$((group_idx + 1))
  done
  (IFS=:; echo "${job_ids[*]}")
}

submit_merge_array() {
  local dependency="${1:?cache dependency required}"
  local n_jobs
  n_jobs="$(wc -l < "${MERGE_JOBS}")"
  if (( n_jobs < 1 )); then
    echo "No merge jobs in ${MERGE_JOBS}" >&2
    exit 2
  fi
  local n_groups="${MERGE_GROUPS}"
  if (( n_groups > n_jobs )); then
    n_groups="${n_jobs}"
  fi
  local group_size=$(((n_jobs + n_groups - 1) / n_groups))
  local job_ids=()
  local start=1
  local group_idx=0
  while (( start <= n_jobs )); do
    local end=$((start + group_size - 1))
    if (( end > n_jobs )); then
      end="${n_jobs}"
    fi
    local script_path="${QUEUE_DIR}/merge_group_${TS}_${group_idx}_${start}_${end}.slurm.sh"
    local log_path="${ARRAY_LOG_DIR}/merge_group_${TS}_${group_idx}_${start}_${end}.%j.log"
    cat > "${script_path}" <<EOF
#!/bin/bash
#SBATCH --job-name=ts_pool1k_m${group_idx}
#SBATCH --partition=${MERGE_PARTITION}
EOF
    if [[ -n "${MERGE_ACCOUNT}" ]]; then
      printf '#SBATCH --account=%s\n' "${MERGE_ACCOUNT}" >> "${script_path}"
    fi
    if [[ -n "${MERGE_QOS}" ]]; then
      printf '#SBATCH --qos=%s\n' "${MERGE_QOS}" >> "${script_path}"
    fi
    if [[ -n "${MERGE_GRES}" ]]; then
      printf '#SBATCH --gres=%s\n' "${MERGE_GRES}" >> "${script_path}"
    fi
    if [[ -n "${MERGE_CPUS_PER_TASK}" ]]; then
      printf '#SBATCH --cpus-per-task=%s\n' "${MERGE_CPUS_PER_TASK}" >> "${script_path}"
    fi
    cat >> "${script_path}" <<EOF
#SBATCH --time=${MERGE_TIME}
#SBATCH --mem=${MERGE_MEM}
#SBATCH --output=${log_path}
#SBATCH --error=${log_path}

set -euo pipefail
cd "${REPO_ROOT}"
EOF
    MERGE_JOB_START="${start}" MERGE_JOB_END="${end}" write_env_exports "${script_path}"
    cat >> "${script_path}" <<'EOF'
run_merge_job() {
  set -euo pipefail
  local eval_id="$1"
  local out_dir="${OUT_RUN}/${RESULTS_NAME}/${eval_id}"
  local -a noise_mult_args=()
  if [[ -n "${NOISE_MULTS}" ]]; then
    noise_mult_args=(--noise-mults "${NOISE_MULTS}")
  fi
  echo "merge_start $(date -Is) eval_id=${eval_id}"
  "${PYTHON}" -u "${SCRIPT}" \
    --model-set "${eval_id}" \
    --selection-root "${STAGED_SELECTION_ROOT}" \
    --tracks "${TRACKS}" \
    --random-feature-dir "${RANDOM_FEATURE_DIR}" \
    --encoding-root "${ENCODING_ROOT}" \
    --n-random-images "${N_RANDOM_IMAGES}" \
    --refit-pool-size "${REFIT_SIZE}" \
    --refit-val-size "${REFIT_VAL_SIZE}" \
    --max-refit-pool-size "${MAX_REFIT_POOL_SIZE}" \
    --n-refit-repeats "${N_REFIT_REPEATS}" \
    --n-random-subsets "${N_RANDOM_SUBSETS}" \
    --n-noise-samples "${N_NOISE_SAMPLES}" \
    "${noise_mult_args[@]}" \
    --eval-noise-mode response \
    --fit-noise-calibration rdm_empirical \
    --rdm-calibration-comparison "${RDM_CALIBRATION_COMPARISON}" \
    --eval-refit-mode "${EVAL_REFIT_MODE}" \
    --calibration-images 100 \
    --calibration-noise-samples 2 \
    --calibration-max-iter 8 \
    --corr-type spearman \
    --encoding-device cuda \
    --shared-encodings \
    --fast-gpu-batch \
    --merge-only \
    --output-dir "${out_dir}"
  echo "merge_done $(date -Is) eval_id=${eval_id}"
}
export -f run_merge_job

echo "merge_group_start $(date -Is) lines=${MERGE_JOB_START}-${MERGE_JOB_END}"
sed -n "${MERGE_JOB_START},${MERGE_JOB_END}p" "${MERGE_JOBS}" |
  cut -f1 |
  xargs -P "${MERGE_MAX_PROCS}" -n 1 bash -c 'set -euo pipefail; run_merge_job "$@"' _
echo "merge_group_done $(date -Is) lines=${MERGE_JOB_START}-${MERGE_JOB_END}"
EOF
    chmod +x "${script_path}"
    local job_id
    job_id="$(submit_sbatch "--dependency=afterok:${dependency}" "${script_path}")"
    job_ids+=("${job_id}")
    start=$((end + 1))
    group_idx=$((group_idx + 1))
  done
  (IFS=:; echo "${job_ids[*]}")
}

submit_summary_job() {
  local dependency="${1:?merge dependency required}"
  local script_path="${QUEUE_DIR}/summarize_${TS}.slurm.sh"
  local log_path="${QUEUE_DIR}/summarize_${TS}.%j.log"
  cat > "${script_path}" <<EOF
#!/bin/bash
#SBATCH --job-name=ts_pool1k_summary
#SBATCH --partition=${SUMMARY_PARTITION}
EOF
  if [[ -n "${SUMMARY_ACCOUNT}" ]]; then
    printf '#SBATCH --account=%s\n' "${SUMMARY_ACCOUNT}" >> "${script_path}"
  fi
  if [[ -n "${SUMMARY_QOS}" ]]; then
    printf '#SBATCH --qos=%s\n' "${SUMMARY_QOS}" >> "${script_path}"
  fi
  if [[ -n "${SUMMARY_GRES}" ]]; then
    printf '#SBATCH --gres=%s\n' "${SUMMARY_GRES}" >> "${script_path}"
  fi
  if [[ -n "${SUMMARY_CPUS_PER_TASK}" ]]; then
    printf '#SBATCH --cpus-per-task=%s\n' "${SUMMARY_CPUS_PER_TASK}" >> "${script_path}"
  fi
  cat >> "${script_path}" <<EOF
#SBATCH --time=${SUMMARY_TIME}
#SBATCH --mem=${SUMMARY_MEM}
#SBATCH --output=${log_path}
#SBATCH --error=${log_path}

set -euo pipefail
cd "${REPO_ROOT}"
EOF
  write_env_exports "${script_path}"
  printf 'export MODE=summarize\n' >> "${script_path}"
  printf 'bash %q\n' "${BASH_SOURCE[0]}" >> "${script_path}"
  chmod +x "${script_path}"
  submit_sbatch "--dependency=afterok:${dependency}" "${script_path}"
}

summarize_outputs() {
  "${PYTHON}" - "${MANIFEST}" "${OUT_RUN}" "${RESULTS_NAME}" <<'PY'
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

manifest_path = Path(sys.argv[1])
out_run = Path(sys.argv[2])
results_name = sys.argv[3]

frames = []
missing = []
with manifest_path.open() as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
for row in rows:
    path = out_run / results_name / row["eval_id"] / "discriminability.csv"
    if not path.exists():
        missing.append(str(path))
        continue
    df = pd.read_csv(path)
    if df.empty:
        missing.append(f"empty: {path}")
        continue
    df["eval_id"] = row["eval_id"]
    df["selection_model_set"] = row["model_set"]
    df["pool_dir"] = row["pool_dir"]
    df["pool_size"] = int(row["pool_size"])
    df["selection_method_id"] = row["method_id"]
    frames.append(df)

summary_dir = out_run / results_name
summary_dir.mkdir(parents=True, exist_ok=True)
missing_path = summary_dir / "missing_discriminability.txt"
missing_path.write_text("\n".join(missing) + ("\n" if missing else ""))
if not frames:
    raise SystemExit("No discriminability CSVs were available to summarize")

combined = pd.concat(frames, ignore_index=True)
combined_path = summary_dir / "combined_discriminability.csv"
combined.to_csv(combined_path, index=False)

emp = combined[np.isclose(combined["noise_mult"].astype(float), 1.0)].copy()
emp_path = summary_dir / "empirical_snr_discriminability.csv"
emp.to_csv(emp_path, index=False)

keys = [
    "selection_model_set",
    "pool_size",
    "selection_method_id",
    "track",
    "subset_type",
]
metric = (
    emp.groupby(keys, as_index=False, dropna=False)
    .agg(
        recovery_accuracy=("recovery_accuracy", "mean"),
        recovery_accuracy_sem=("recovery_accuracy_sem", "mean"),
        mean_margin=("mean_margin", "mean"),
        n_units=("n_units", "sum"),
        n_models=("n_models", "max"),
        n_refit_repeats=("n_refit_repeats", "max"),
    )
    .sort_values(keys)
)
metric_path = summary_dir / "empirical_snr_by_pool_method_track.csv"
metric.to_csv(metric_path, index=False)

print(f"combined_rows={len(combined)}")
print(f"missing_outputs={len(missing)}")
print(f"wrote={combined_path}")
print(f"wrote={emp_path}")
print(f"wrote={metric_path}")
PY
}

run_submit() {
  echo "queue_start $(date -Is)"
  echo "RUN_STAMP=${RUN_STAMP}"
  echo "MODEL_SETS=${MODEL_SETS}"
  echo "OUT_RUN=${OUT_RUN}"
  echo "RESULTS_NAME=${RESULTS_NAME}"
  echo "NOISE_MULTS=${NOISE_MULTS}"
  stage_payloads
  write_job_files
  local prep_job_id=""
  if [[ "${BUILD_NPY_CACHE}" == "1" ]]; then
    prep_job_id="$(submit_prep_job)"
    echo "prep_job_id=${prep_job_id}"
  fi
  local cache_job_id
  cache_job_id="$(submit_cache_array "${prep_job_id}")"
  echo "cache_job_id=${cache_job_id}"
  local merge_job_id
  merge_job_id="$(submit_merge_array "${cache_job_id}")"
  echo "merge_job_id=${merge_job_id}"
  local summary_job_id
  summary_job_id="$(submit_summary_job "${merge_job_id}")"
  echo "summary_job_id=${summary_job_id}"
  echo "queue_done $(date -Is)"
}

case "${MODE}" in
  stage)
    stage_payloads
    ;;
  write_jobs)
    write_job_files
    ;;
  summarize)
    summarize_outputs
    ;;
  submit)
    run_submit
    ;;
  *)
    echo "Unknown MODE=${MODE}" >&2
    exit 2
    ;;
esac
