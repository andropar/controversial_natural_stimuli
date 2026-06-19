#!/bin/bash
#
# Submit the dense layer sweep on Raven as one GPU Slurm-array task per
# (subject, model), then merge stream parts and build the layer-selection tables.
#
# Fast default:
#   bash 05_controls_and_supplementary/model_scope_followups/layer_sweep/code/queue_dense_layer_sweep_raven_slurm.sh
#
# Useful overrides:
#   MAX_CONCURRENT=60 LAYERS_PER_CHUNK=64 bash .../queue_dense_layer_sweep_raven_slurm.sh
#   MEM=125000 bash .../queue_dense_layer_sweep_raven_slurm.sh
#   MODELS=torchvision_resnet50_imagenet1k_v1 SUBJECTS=sub-01 DRY_RUN=1 bash .../queue_dense_layer_sweep_raven_slurm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAYER_SWEEP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ANALYSIS_DIR="${SCRIPT_DIR}/analysis"
REPO_ROOT="$(cd "${LAYER_SWEEP_ROOT}/../../.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/u/rothj/conda-envs/deepjuice/bin/python" ]]; then
    PYTHON_BIN="/u/rothj/conda-envs/deepjuice/bin/python"
  elif [[ -x "/u/rothj/miniforge3/bin/python" ]]; then
    PYTHON_BIN="/u/rothj/miniforge3/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${LAYER_SWEEP_ROOT}/logs/raven_dense_stream_${RUN_STAMP}}"
PART_ROOT="${PART_ROOT:-${LAYER_SWEEP_ROOT}/results/stream_parts_raven_dense_${RUN_STAMP}}"
OUT_CSV="${OUT_CSV:-${LAYER_SWEEP_ROOT}/results/wrsa_dense_layer_sweep.csv}"
SHARED_OUT_CSV="${SHARED_OUT_CSV:-${LAYER_SWEEP_ROOT}/results/wrsa_dense_shared_layer_sweep.csv}"
MANIFEST="${RUN_ROOT}/subject_model_manifest.tsv"

CSTIMS_PATH_ENV="${CSTIMS_PATH_ENV:-raven}"
LAYER_SET="${LAYER_SET:-dense}"
SUBJECTS="${SUBJECTS:-sub-01,sub-03,sub-05,sub-06,sub-07}"
MODELS="${MODELS:-}"
BATCH_CANDIDATES="${BATCH_CANDIDATES:-1,2,4,8,16,32}"
LAYERS_PER_CHUNK="${LAYERS_PER_CHUNK:-auto}"
MAX_LAYERS_PER_CHUNK="${MAX_LAYERS_PER_CHUNK:-128}"
MAX_FEATURE_GB_PER_CHUNK="${MAX_FEATURE_GB_PER_CHUNK:-24}"
FIT_BACKEND="${FIT_BACKEND:-gpu}"
GPU_FIT_DTYPE="${GPU_FIT_DTYPE:-float64}"
N_FIT_JOBS="${N_FIT_JOBS:-1}"
N_SCORE_JOBS="${N_SCORE_JOBS:-3}"
N_VICCO_BOOT="${N_VICCO_BOOT:-1000}"
N_SHARED_BOOT="${N_SHARED_BOOT:-1000}"
BOOTSTRAP_N="${BOOTSTRAP_N:-100}"
EXTRACT_PREFETCH_WORKERS="${EXTRACT_PREFETCH_WORKERS:-2}"
DEEPVISION_LOAD_JOBS="${DEEPVISION_LOAD_JOBS:-1}"
STREAM_ENCODING_ROOT="${STREAM_ENCODING_ROOT:-}"

MAX_CONCURRENT="${MAX_CONCURRENT:-100}"
GPU_DEPENDENCY="${GPU_DEPENDENCY:-}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
CONSTRAINT="${CONSTRAINT:-gpu}"
GRES="${GRES:-gpu:a100:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-18}"
MEM="${MEM:-64000}"
MERGE_TIME_LIMIT="${MERGE_TIME_LIMIT:-02:00:00}"
MERGE_CPUS_PER_TASK="${MERGE_CPUS_PER_TASK:-8}"
MERGE_MEM="${MERGE_MEM:-64000}"
PARTITION="${PARTITION:-}"
ACCOUNT="${ACCOUNT:-${SLURM_ACCOUNT:-}}"
QOS="${QOS:-}"
DRY_RUN="${DRY_RUN:-0}"
SUBMIT_MERGE="${SUBMIT_MERGE:-1}"
SUBMIT_CSTIM_ENCODING_CV="${SUBMIT_CSTIM_ENCODING_CV:-0}"
CSTIM_ENCODING_CV_QUEUE="${CSTIM_ENCODING_CV_QUEUE:-${REPO_ROOT}/05_controls_and_supplementary/cstim_encoding_cv/code/queue_cstim_encoding_cv_raven_slurm.sh}"

mkdir -p "${RUN_ROOT}/logs" "${PART_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${SCRIPT_DIR}:${ANALYSIS_DIR}:${PYTHONPATH:-}"
export CSTIMS_PATH_ENV

"${PYTHON_BIN}" - "${MANIFEST}" "${SUBJECTS}" "${MODELS}" "${LAYER_SET}" "${SCRIPT_DIR}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

manifest = Path(sys.argv[1])
subjects_arg = sys.argv[2]
models_arg = sys.argv[3]
layer_set = sys.argv[4]
code_root = sys.argv[5]

sys.path.insert(0, code_root)
from layers_config import get_layer_set  # noqa: E402

subjects = [item.strip() for item in subjects_arg.split(",") if item.strip()]
layer_specs = get_layer_set(layer_set)
models = (
    [item.strip() for item in models_arg.split(",") if item.strip()]
    if models_arg.strip()
    else list(layer_specs)
)
unknown = [model for model in models if model not in layer_specs]
if unknown:
    raise SystemExit(f"Unknown models for layer set {layer_set}: {unknown}")

manifest.parent.mkdir(parents=True, exist_ok=True)
with manifest.open("w") as f:
    for subject in subjects:
        for model in models:
            f.write(f"{subject}\t{model}\t{len(layer_specs[model])}\n")
PY

N_TASKS="$(wc -l < "${MANIFEST}" | tr -d ' ')"
if [[ "${N_TASKS}" -le 0 ]]; then
  echo "No subject/model tasks in ${MANIFEST}" >&2
  exit 2
fi

WORKER_SCRIPT="${RUN_ROOT}/dense_stream_worker.slurm.sh"
{
  printf '#!/bin/bash -l\n'
  printf 'set -euo pipefail\n\n'
  printf 'PYTHON_BIN=%q\n' "${PYTHON_BIN}"
  printf 'ANALYSIS_DIR=%q\n' "${ANALYSIS_DIR}"
  printf 'LAYER_SWEEP_ROOT=%q\n' "${LAYER_SWEEP_ROOT}"
  printf 'MANIFEST=%q\n' "${MANIFEST}"
  printf 'PART_ROOT=%q\n' "${PART_ROOT}"
  printf 'CSTIMS_PATH_ENV=%q\n' "${CSTIMS_PATH_ENV}"
  printf 'LAYER_SET=%q\n' "${LAYER_SET}"
  printf 'BATCH_CANDIDATES=%q\n' "${BATCH_CANDIDATES}"
  printf 'LAYERS_PER_CHUNK=%q\n' "${LAYERS_PER_CHUNK}"
  printf 'MAX_LAYERS_PER_CHUNK=%q\n' "${MAX_LAYERS_PER_CHUNK}"
  printf 'MAX_FEATURE_GB_PER_CHUNK=%q\n' "${MAX_FEATURE_GB_PER_CHUNK}"
  printf 'FIT_BACKEND=%q\n' "${FIT_BACKEND}"
  printf 'GPU_FIT_DTYPE=%q\n' "${GPU_FIT_DTYPE}"
  printf 'N_FIT_JOBS=%q\n' "${N_FIT_JOBS}"
  printf 'N_SCORE_JOBS=%q\n' "${N_SCORE_JOBS}"
  printf 'N_VICCO_BOOT=%q\n' "${N_VICCO_BOOT}"
  printf 'N_SHARED_BOOT=%q\n' "${N_SHARED_BOOT}"
  printf 'BOOTSTRAP_N=%q\n' "${BOOTSTRAP_N}"
  printf 'EXTRACT_PREFETCH_WORKERS=%q\n' "${EXTRACT_PREFETCH_WORKERS}"
  printf 'DEEPVISION_LOAD_JOBS=%q\n' "${DEEPVISION_LOAD_JOBS}"
  printf 'STREAM_ENCODING_ROOT=%q\n\n' "${STREAM_ENCODING_ROOT}"
  cat <<'SLURM'
export CSTIMS_PATH_ENV
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

if [[ -n "${MODULES:-}" ]]; then
  module purge
  # shellcheck disable=SC2086
  module load ${MODULES}
fi

CONDA_PREFIX_GUESS="$(cd "$(dirname "${PYTHON_BIN}")/.." 2>/dev/null && pwd || true)"
CONDA_LIB="${CONDA_LIB:-${CONDA_PREFIX_GUESS}/lib}"
if [[ -d "${CONDA_LIB}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"
fi

line="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${MANIFEST}")"
if [[ -z "${line}" ]]; then
  echo "No manifest line for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi
IFS=$'\t' read -r subject model n_layers <<< "${line}"
progress_log="${PART_ROOT}/progress_${subject}_${model}.jsonl"

echo "[start] $(date --iso-8601=seconds) task=${SLURM_ARRAY_TASK_ID} subject=${subject} model=${model} layers=${n_layers}"
cmd=(
  "${PYTHON_BIN}" "${ANALYSIS_DIR}/07_fit_encodings_layer_sweep.py"
  --subject "${subject}"
  --models "${model}"
  --mode stream-model
  --layer-set "${LAYER_SET}"
  --batch-size auto
  --batch-candidates "${BATCH_CANDIDATES}"
  --layers-per-chunk "${LAYERS_PER_CHUNK}"
  --max-layers-per-chunk "${MAX_LAYERS_PER_CHUNK}"
  --max-feature-gb-per-chunk "${MAX_FEATURE_GB_PER_CHUNK}"
  --fit-backend "${FIT_BACKEND}"
  --gpu-fit-dtype "${GPU_FIT_DTYPE}"
  --n-fit-jobs "${N_FIT_JOBS}"
  --n-score-jobs "${N_SCORE_JOBS}"
  --n-vicco-boot "${N_VICCO_BOOT}"
  --n-shared-boot "${N_SHARED_BOOT}"
  --bootstrap-n "${BOOTSTRAP_N}"
  --extract-prefetch-workers "${EXTRACT_PREFETCH_WORKERS}"
  --deepvision-load-jobs "${DEEPVISION_LOAD_JOBS}"
  --stream-part-root "${PART_ROOT}"
  --progress-log "${progress_log}"
)
if [[ -n "${STREAM_ENCODING_ROOT}" ]]; then
  cmd+=(--stream-encoding-root "${STREAM_ENCODING_ROOT}")
fi
printf '[cmd]'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
echo "[done] $(date --iso-8601=seconds) task=${SLURM_ARRAY_TASK_ID} subject=${subject} model=${model}"
SLURM
} > "${WORKER_SCRIPT}"
chmod +x "${WORKER_SCRIPT}"

MERGE_SCRIPT="${RUN_ROOT}/dense_stream_merge.slurm.sh"
{
  printf '#!/bin/bash -l\n'
  printf 'set -euo pipefail\n\n'
  printf 'PYTHON_BIN=%q\n' "${PYTHON_BIN}"
  printf 'ANALYSIS_DIR=%q\n' "${ANALYSIS_DIR}"
  printf 'LAYER_SWEEP_ROOT=%q\n' "${LAYER_SWEEP_ROOT}"
  printf 'PART_ROOT=%q\n' "${PART_ROOT}"
  printf 'OUT_CSV=%q\n' "${OUT_CSV}"
  printf 'SHARED_OUT_CSV=%q\n' "${SHARED_OUT_CSV}"
  printf 'CSTIMS_PATH_ENV=%q\n' "${CSTIMS_PATH_ENV}"
  printf 'LAYER_SET=%q\n\n' "${LAYER_SET}"
  printf 'SUBJECTS=%q\n' "${SUBJECTS}"
  printf 'MODELS=%q\n\n' "${MODELS}"
  cat <<'SLURM'
export CSTIMS_PATH_ENV
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONDA_PREFIX_GUESS="$(cd "$(dirname "${PYTHON_BIN}")/.." 2>/dev/null && pwd || true)"
CONDA_LIB="${CONDA_LIB:-${CONDA_PREFIX_GUESS}/lib}"
if [[ -d "${CONDA_LIB}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"
fi

echo "[merge] $(date --iso-8601=seconds) part_root=${PART_ROOT}"
"${PYTHON_BIN}" "${ANALYSIS_DIR}/07_fit_encodings_layer_sweep.py" \
  --mode merge-stream \
  --layer-set "${LAYER_SET}" \
  --subject "${SUBJECTS}" \
  --stream-part-root "${PART_ROOT}" \
  --out-csv "${OUT_CSV}" \
  --shared-out-csv "${SHARED_OUT_CSV}" \
  ${MODELS:+--models ${MODELS//,/ }}

"${PYTHON_BIN}" "${ANALYSIS_DIR}/14_build_mrsa_layer_selection_tables.py" \
  --layer-set "${LAYER_SET}" \
  --wrsa-csv "${OUT_CSV}" \
  --shared-csv "${SHARED_OUT_CSV}"
echo "[done] $(date --iso-8601=seconds)"
SLURM
} > "${MERGE_SCRIPT}"
chmod +x "${MERGE_SCRIPT}"

GPU_SBATCH_ARGS=(
  --parsable
  --array="0-$((N_TASKS - 1))%${MAX_CONCURRENT}"
  --job-name="dense_layer_stream"
  --time="${TIME_LIMIT}"
  --constraint="${CONSTRAINT}"
  --gres="${GRES}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --mem="${MEM}"
  --output="${RUN_ROOT}/logs/%x.%A_%a.out"
  --error="${RUN_ROOT}/logs/%x.%A_%a.err"
)
if [[ -n "${PARTITION}" ]]; then
  GPU_SBATCH_ARGS+=(--partition="${PARTITION}")
fi
if [[ -n "${ACCOUNT}" ]]; then
  GPU_SBATCH_ARGS+=(--account="${ACCOUNT}")
fi
if [[ -n "${QOS}" ]]; then
  GPU_SBATCH_ARGS+=(--qos="${QOS}")
fi
if [[ -n "${GPU_DEPENDENCY}" ]]; then
  GPU_SBATCH_ARGS+=(--dependency="${GPU_DEPENDENCY}")
fi

MERGE_SBATCH_ARGS=(
  --parsable
  --job-name="dense_layer_merge"
  --time="${MERGE_TIME_LIMIT}"
  --cpus-per-task="${MERGE_CPUS_PER_TASK}"
  --mem="${MERGE_MEM}"
  --output="${RUN_ROOT}/logs/%x.%j.out"
  --error="${RUN_ROOT}/logs/%x.%j.err"
)
if [[ -n "${PARTITION}" ]]; then
  MERGE_SBATCH_ARGS+=(--partition="${PARTITION}")
fi
if [[ -n "${ACCOUNT}" ]]; then
  MERGE_SBATCH_ARGS+=(--account="${ACCOUNT}")
fi
if [[ -n "${QOS}" ]]; then
  MERGE_SBATCH_ARGS+=(--qos="${QOS}")
fi

echo "Manifest: ${MANIFEST} (${N_TASKS} subject/model tasks)"
echo "Part root: ${PART_ROOT}"
echo "Worker: ${WORKER_SCRIPT}"
echo "Merge: ${MERGE_SCRIPT}"
printf 'GPU sbatch: sbatch'
printf ' %q' "${GPU_SBATCH_ARGS[@]}" "${WORKER_SCRIPT}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  if [[ "${SUBMIT_MERGE}" == "1" ]]; then
    printf 'Merge sbatch: sbatch'
    printf ' %q' "${MERGE_SBATCH_ARGS[@]}"
    printf ' --dependency=%q' "afterok:<gpu_job_id>"
    printf ' %q' "${MERGE_SCRIPT}"
    printf '\n'
    if [[ "${SUBMIT_CSTIM_ENCODING_CV}" == "1" ]]; then
      printf 'CSTIM encoding-CV queue: LAYER_MERGE_JOB_ID=%q PYTHON_BIN=%q CSTIMS_PATH_ENV=%q SUBJECTS=%q MODELS=%q bash %q\n' \
        "<merge_job_id>" "${PYTHON_BIN}" "${CSTIMS_PATH_ENV}" "${SUBJECTS}" "${MODELS}" "${CSTIM_ENCODING_CV_QUEUE}"
    fi
  fi
  echo "DRY_RUN=1; not submitting."
  exit 0
fi

GPU_JOB_ID="$(sbatch "${GPU_SBATCH_ARGS[@]}" "${WORKER_SCRIPT}")"
echo "Submitted GPU array job: ${GPU_JOB_ID}"

if [[ "${SUBMIT_MERGE}" == "1" ]]; then
  MERGE_SBATCH_ARGS+=(--dependency="afterok:${GPU_JOB_ID}")
  printf 'Merge sbatch: sbatch'
  printf ' %q' "${MERGE_SBATCH_ARGS[@]}" "${MERGE_SCRIPT}"
  printf '\n'
  MERGE_JOB_ID="$(sbatch "${MERGE_SBATCH_ARGS[@]}" "${MERGE_SCRIPT}")"
  echo "Submitted merge/table job: ${MERGE_JOB_ID}"
  if [[ "${SUBMIT_CSTIM_ENCODING_CV}" == "1" ]]; then
    if [[ ! -x "${CSTIM_ENCODING_CV_QUEUE}" ]]; then
      echo "CSTIM encoding-CV queue script is not executable: ${CSTIM_ENCODING_CV_QUEUE}" >&2
      exit 2
    fi
    echo "Submitting CSTIM encoding-CV pipeline after merge job ${MERGE_JOB_ID}"
    LAYER_MERGE_JOB_ID="${MERGE_JOB_ID}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      CSTIMS_PATH_ENV="${CSTIMS_PATH_ENV}" \
      SUBJECTS="${SUBJECTS}" \
      MODELS="${MODELS}" \
      PARTITION="${PARTITION}" \
      ACCOUNT="${ACCOUNT}" \
      QOS="${QOS}" \
      bash "${CSTIM_ENCODING_CV_QUEUE}"
  fi
fi
