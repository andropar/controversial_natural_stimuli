#!/bin/bash
#
# Queue the canonical CSTIM encoding-CV follow-up on Raven.
#
# Typical use after submitting the dense layer sweep:
#   LAYER_MERGE_JOB_ID=12345678 \
#     bash 05_controls_and_supplementary/cstim_encoding_cv/code/queue_cstim_encoding_cv_raven_slurm.sh
#
# The dependency can also be passed directly:
#   DEPENDENCY=afterok:12345678 bash .../queue_cstim_encoding_cv_raven_slurm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CV_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CV_ROOT}/../.." && pwd)"
LAYER_SWEEP_ROOT="${REPO_ROOT}/05_controls_and_supplementary/model_scope_followups/layer_sweep"
LAYER_CODE_DIR="${LAYER_SWEEP_ROOT}/code"
LAYER_ANALYSIS_DIR="${LAYER_CODE_DIR}/analysis"

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
RUN_ROOT="${RUN_ROOT:-${CV_ROOT}/logs/raven_cstim_encoding_cv_${RUN_STAMP}}"
CACHE_MANIFEST="${RUN_ROOT}/cache_models.txt"

CSTIMS_PATH_ENV="${CSTIMS_PATH_ENV:-raven}"
DEFAULT_SUBJECTS="sub-01,sub-03,sub-05,sub-06,sub-07"
SUBJECTS="${SUBJECTS:-${DEFAULT_SUBJECTS}}"
MODELS="${MODELS:-}"

CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-4}"
CACHE_LAYERS_PER_CHUNK="${CACHE_LAYERS_PER_CHUNK:-32}"
CACHE_PROGRESS_EVERY="${CACHE_PROGRESS_EVERY:-1024}"
CACHE_OVERWRITE="${CACHE_OVERWRITE:-1}"

FIXED_WEIGHTS="${FIXED_WEIGHTS:-0,0.25,0.5,1,2,4,8,16,32,47}"
N_VICCO_BOOT="${N_VICCO_BOOT:-1000}"
OVERWRITE_SCORE="${OVERWRITE_SCORE:-1}"
OVERWRITE_ALPHA="${OVERWRITE_ALPHA:-1}"
RUN_FIGURES="${RUN_FIGURES:-1}"
RUN_WEIGHT0_CHECK="${RUN_WEIGHT0_CHECK:-1}"

GPU_MAX_CONCURRENT="${GPU_MAX_CONCURRENT:-20}"
GPU_TIME_LIMIT="${GPU_TIME_LIMIT:-12:00:00}"
GPU_CONSTRAINT="${GPU_CONSTRAINT:-gpu}"
GPU_GRES="${GPU_GRES:-gpu:a100:1}"
GPU_CPUS_PER_TASK="${GPU_CPUS_PER_TASK:-12}"
GPU_MEM="${GPU_MEM:-64000}"

FIXED_TIME_LIMIT="${FIXED_TIME_LIMIT:-12:00:00}"
FIXED_CPUS_PER_TASK="${FIXED_CPUS_PER_TASK:-18}"
FIXED_MEM="${FIXED_MEM:-64000}"

PARTITION="${PARTITION:-}"
ACCOUNT="${ACCOUNT:-${SLURM_ACCOUNT:-}}"
QOS="${QOS:-}"
DRY_RUN="${DRY_RUN:-0}"

DEPENDENCY="${DEPENDENCY:-}"
if [[ -n "${LAYER_MERGE_JOB_ID:-}" ]]; then
  DEPENDENCY="afterok:${LAYER_MERGE_JOB_ID}"
fi

mkdir -p "${RUN_ROOT}/logs"

export CSTIMS_PATH_ENV
export PYTHONPATH="${REPO_ROOT}/src:${SCRIPT_DIR}:${CV_ROOT}/code/analysis:${LAYER_CODE_DIR}:${LAYER_ANALYSIS_DIR}:${PYTHONPATH:-}"

"${PYTHON_BIN}" - "${CACHE_MANIFEST}" "${MODELS}" "${LAYER_CODE_DIR}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

cache_manifest = Path(sys.argv[1])
models_arg = sys.argv[2]
layer_code_dir = sys.argv[3]

sys.path.insert(0, layer_code_dir)
from layers_config import get_layer_set  # noqa: E402

layer_specs = get_layer_set("dense")
models = (
    [item.strip() for item in models_arg.split(",") if item.strip()]
    if models_arg.strip()
    else list(layer_specs)
)
unknown = [model for model in models if model not in layer_specs]
if unknown:
    raise SystemExit(f"Unknown dense-layer models: {unknown}")

cache_manifest.parent.mkdir(parents=True, exist_ok=True)
cache_manifest.write_text("\n".join(models) + "\n")
PY

N_CACHE_TASKS="$(wc -l < "${CACHE_MANIFEST}" | tr -d ' ')"
if [[ "${N_CACHE_TASKS}" -le 0 ]]; then
  echo "Empty cache manifest: ${CACHE_MANIFEST}" >&2
  exit 2
fi

add_common_sbatch_args() {
  local -n arr_ref="$1"
  if [[ -n "${PARTITION}" ]]; then
    arr_ref+=(--partition="${PARTITION}")
  fi
  if [[ -n "${ACCOUNT}" ]]; then
    arr_ref+=(--account="${ACCOUNT}")
  fi
  if [[ -n "${QOS}" ]]; then
    arr_ref+=(--qos="${QOS}")
  fi
}

CACHE_WORKER="${RUN_ROOT}/cache_features_worker.slurm.sh"
{
  printf '#!/bin/bash -l\n'
  printf 'set -euo pipefail\n\n'
  printf 'PYTHON_BIN=%q\n' "${PYTHON_BIN}"
  printf 'CV_ROOT=%q\n' "${CV_ROOT}"
  printf 'REPO_ROOT=%q\n' "${REPO_ROOT}"
  printf 'LAYER_SWEEP_ROOT=%q\n' "${LAYER_SWEEP_ROOT}"
  printf 'CACHE_MANIFEST=%q\n' "${CACHE_MANIFEST}"
  printf 'CSTIMS_PATH_ENV=%q\n' "${CSTIMS_PATH_ENV}"
  printf 'DEFAULT_SUBJECTS=%q\n' "${DEFAULT_SUBJECTS}"
  printf 'SUBJECTS=%q\n' "${SUBJECTS}"
  printf 'CACHE_BATCH_SIZE=%q\n' "${CACHE_BATCH_SIZE}"
  printf 'CACHE_LAYERS_PER_CHUNK=%q\n' "${CACHE_LAYERS_PER_CHUNK}"
  printf 'CACHE_PROGRESS_EVERY=%q\n' "${CACHE_PROGRESS_EVERY}"
  printf 'CACHE_OVERWRITE=%q\n\n' "${CACHE_OVERWRITE}"
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
export PYTHONPATH="${REPO_ROOT}/src:${CV_ROOT}/code:${CV_ROOT}/code/analysis:${LAYER_SWEEP_ROOT}/code:${LAYER_SWEEP_ROOT}/code/analysis:${PYTHONPATH:-}"

CONDA_PREFIX_GUESS="$(cd "$(dirname "${PYTHON_BIN}")/.." 2>/dev/null && pwd || true)"
CONDA_LIB="${CONDA_LIB:-${CONDA_PREFIX_GUESS}/lib}"
if [[ -d "${CONDA_LIB}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"
fi

selection_csv="${LAYER_SWEEP_ROOT}/results/mrsa_dense_layer_selection_transfer.csv"
if [[ ! -s "${selection_csv}" ]]; then
  echo "Missing layer-selection table: ${selection_csv}" >&2
  exit 2
fi

model="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${CACHE_MANIFEST}")"
if [[ -z "${model}" ]]; then
  echo "No cache manifest line for task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

cd "${CV_ROOT}"
cmd=(
  "${PYTHON_BIN}" code/analysis/01_cache_selected_layer_srp5920_features.py
  --models "${model}"
  --batch-size "${CACHE_BATCH_SIZE}"
  --device cuda:0
  --layers-per-chunk "${CACHE_LAYERS_PER_CHUNK}"
  --progress-every "${CACHE_PROGRESS_EVERY}"
)
if [[ "${CACHE_OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi
if [[ "${SUBJECTS}" != "${DEFAULT_SUBJECTS}" && "${SUBJECTS}" != "all" ]]; then
  if [[ "${SUBJECTS}" == *,* ]]; then
    echo "CSTIM encoding-CV cache supports SUBJECTS=all/default or one subject; got ${SUBJECTS}" >&2
    exit 2
  fi
  cmd+=(--subject "${SUBJECTS}")
fi
printf '[cache-cmd]'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
SLURM
} > "${CACHE_WORKER}"
chmod +x "${CACHE_WORKER}"

FIXED_WORKER="${RUN_ROOT}/fixed_alpha.slurm.sh"
{
  printf '#!/bin/bash -l\n'
  printf 'set -euo pipefail\n\n'
  printf 'PYTHON_BIN=%q\n' "${PYTHON_BIN}"
  printf 'CV_ROOT=%q\n' "${CV_ROOT}"
  printf 'REPO_ROOT=%q\n' "${REPO_ROOT}"
  printf 'LAYER_SWEEP_ROOT=%q\n' "${LAYER_SWEEP_ROOT}"
  printf 'CSTIMS_PATH_ENV=%q\n' "${CSTIMS_PATH_ENV}"
  printf 'DEFAULT_SUBJECTS=%q\n' "${DEFAULT_SUBJECTS}"
  printf 'SUBJECTS=%q\n' "${SUBJECTS}"
  printf 'MODELS=%q\n' "${MODELS}"
  printf 'FIXED_WEIGHTS=%q\n' "${FIXED_WEIGHTS}"
  printf 'N_VICCO_BOOT=%q\n' "${N_VICCO_BOOT}"
  printf 'OVERWRITE_SCORE=%q\n' "${OVERWRITE_SCORE}"
  printf 'OVERWRITE_ALPHA=%q\n' "${OVERWRITE_ALPHA}"
  printf 'RUN_FIGURES=%q\n' "${RUN_FIGURES}"
  printf 'RUN_WEIGHT0_CHECK=%q\n\n' "${RUN_WEIGHT0_CHECK}"
  cat <<'SLURM'
export CSTIMS_PATH_ENV
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="${REPO_ROOT}/src:${CV_ROOT}/code:${CV_ROOT}/code/analysis:${LAYER_SWEEP_ROOT}/code:${LAYER_SWEEP_ROOT}/code/analysis:${PYTHONPATH:-}"

CONDA_PREFIX_GUESS="$(cd "$(dirname "${PYTHON_BIN}")/.." 2>/dev/null && pwd || true)"
CONDA_LIB="${CONDA_LIB:-${CONDA_PREFIX_GUESS}/lib}"
if [[ -d "${CONDA_LIB}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"
fi

cd "${CV_ROOT}"
score_cmd=(
  "${PYTHON_BIN}" code/analysis/02_score_target_adaptation_srp5920_per_voxel_alpha.py
  --weights "${FIXED_WEIGHTS}"
  --n-vicco-boot "${N_VICCO_BOOT}"
)
if [[ "${OVERWRITE_SCORE}" == "1" ]]; then
  score_cmd+=(--overwrite)
fi
if [[ "${OVERWRITE_ALPHA}" == "1" ]]; then
  score_cmd+=(--overwrite-alpha)
fi
if [[ "${SUBJECTS}" != "${DEFAULT_SUBJECTS}" && "${SUBJECTS}" != "all" ]]; then
  if [[ "${SUBJECTS}" == *,* ]]; then
    echo "CSTIM encoding-CV scorer supports SUBJECTS=all/default or one subject; got ${SUBJECTS}" >&2
    exit 2
  fi
  score_cmd+=(--subject "${SUBJECTS}")
fi
if [[ -n "${MODELS}" ]]; then
  IFS=',' read -r -a model_args <<< "${MODELS}"
  score_cmd+=(--models "${model_args[@]}")
fi
printf '[fixed-cmd]'
printf ' %q' "${score_cmd[@]}"
printf '\n'
"${score_cmd[@]}"

if [[ "${RUN_FIGURES}" == "1" ]]; then
  rm -f figures/target_adaptation_*.pdf figures/png/target_adaptation_*.png
  "${PYTHON_BIN}" code/figures/01_plot_target_weight_trajectories.py
  "${PYTHON_BIN}" code/figures/02_plot_weight0_to_best_cstim_grid.py
fi

if [[ "${RUN_WEIGHT0_CHECK}" == "1" ]]; then
  "${PYTHON_BIN}" - <<'PY'
import pandas as pd

scores = pd.read_csv("results/target_adaptation_weighted_scores.csv")
w0 = scores[scores["target_weight"].astype(float).eq(0.0)].copy()
check = w0[w0["eval_target"].isin(["cstim_loso", "vicco_heldout"])].copy()
summary = (
    check.groupby("eval_target")
    .agg(
        mean_score=("mrsa_loso", "mean"),
        mean_original=("original_best_shared_mrsa", "mean"),
        mean_delta=("delta_vs_original", "mean"),
        max_abs_delta=("delta_vs_original", lambda x: x.abs().max()),
        n=("delta_vs_original", "size"),
    )
    .reset_index()
)
out = "results/target_adaptation_weight0_repro_check.csv"
summary.to_csv(out, index=False)
print(summary.to_string(index=False))
print(f"wrote {out}")
PY
fi
SLURM
} > "${FIXED_WORKER}"
chmod +x "${FIXED_WORKER}"

CACHE_SBATCH_ARGS=(
  --parsable
  --array="0-$((N_CACHE_TASKS - 1))%${GPU_MAX_CONCURRENT}"
  --job-name="cstim_cv_cache"
  --time="${GPU_TIME_LIMIT}"
  --constraint="${GPU_CONSTRAINT}"
  --gres="${GPU_GRES}"
  --cpus-per-task="${GPU_CPUS_PER_TASK}"
  --mem="${GPU_MEM}"
  --output="${RUN_ROOT}/logs/%x.%A_%a.out"
  --error="${RUN_ROOT}/logs/%x.%A_%a.err"
)
add_common_sbatch_args CACHE_SBATCH_ARGS
if [[ -n "${DEPENDENCY}" ]]; then
  CACHE_SBATCH_ARGS+=(--dependency="${DEPENDENCY}")
fi

FIXED_SBATCH_ARGS=(
  --parsable
  --job-name="cstim_cv_fixed"
  --time="${FIXED_TIME_LIMIT}"
  --cpus-per-task="${FIXED_CPUS_PER_TASK}"
  --mem="${FIXED_MEM}"
  --output="${RUN_ROOT}/logs/%x.%j.out"
  --error="${RUN_ROOT}/logs/%x.%j.err"
)
add_common_sbatch_args FIXED_SBATCH_ARGS

echo "Python: ${PYTHON_BIN}"
echo "Cache manifest: ${CACHE_MANIFEST} (${N_CACHE_TASKS} models)"
echo "Subjects: ${SUBJECTS}"
echo "Run root: ${RUN_ROOT}"
if [[ -n "${DEPENDENCY}" ]]; then
  echo "Initial dependency: ${DEPENDENCY}"
fi
printf 'Cache sbatch: sbatch'
printf ' %q' "${CACHE_SBATCH_ARGS[@]}" "${CACHE_WORKER}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Fixed sbatch: sbatch'
  printf ' %q' "${FIXED_SBATCH_ARGS[@]}"
  printf ' --dependency=%q' "afterok:<cache_job_id>"
  printf ' %q' "${FIXED_WORKER}"
  printf '\n'
  echo "DRY_RUN=1; not submitting."
  exit 0
fi

CACHE_JOB_ID="$(sbatch "${CACHE_SBATCH_ARGS[@]}" "${CACHE_WORKER}")"
echo "Submitted cache GPU array job: ${CACHE_JOB_ID}"

FIXED_SBATCH_ARGS+=(--dependency="afterok:${CACHE_JOB_ID}")
FIXED_JOB_ID="$(sbatch "${FIXED_SBATCH_ARGS[@]}" "${FIXED_WORKER}")"
echo "Submitted fixed-alpha CPU job: ${FIXED_JOB_ID}"
