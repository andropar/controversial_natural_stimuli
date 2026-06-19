#!/bin/bash
#
# Queue the CSTIM encoding-CV follow-up on Raven.
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
  if [[ -x "/u/rothj/miniforge3/bin/python" ]]; then
    PYTHON_BIN="/u/rothj/miniforge3/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${CV_ROOT}/logs/raven_cstim_encoding_cv_${RUN_STAMP}}"
CACHE_MANIFEST="${RUN_ROOT}/cache_models.txt"
REFIT_MANIFEST="${RUN_ROOT}/refit_subject_model_manifest.tsv"

CSTIMS_PATH_ENV="${CSTIMS_PATH_ENV:-raven}"
SUBJECTS="${SUBJECTS:-sub-01,sub-03,sub-05,sub-06,sub-07}"
MODELS="${MODELS:-}"

CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-4}"
CACHE_LAYERS_PER_CHUNK="${CACHE_LAYERS_PER_CHUNK:-32}"
CACHE_PROGRESS_EVERY="${CACHE_PROGRESS_EVERY:-1024}"
CACHE_OVERWRITE="${CACHE_OVERWRITE:-1}"

FIXED_WEIGHTS="${FIXED_WEIGHTS:-0,0.25,0.5,1,2,4,8,16,32,47}"
REFIT_WEIGHTS="${REFIT_WEIGHTS:-0,0.25,0.5,1,2,4,8,16,32,47,4700}"
N_VICCO_BOOT="${N_VICCO_BOOT:-1000}"
REFIT_SHARD_PREFIX="${REFIT_SHARD_PREFIX:-target_adaptation_full_refit_all_weights_by_model_raven_${RUN_STAMP}}"

GPU_MAX_CONCURRENT="${GPU_MAX_CONCURRENT:-20}"
REFIT_MAX_CONCURRENT="${REFIT_MAX_CONCURRENT:-100}"

GPU_TIME_LIMIT="${GPU_TIME_LIMIT:-12:00:00}"
GPU_CONSTRAINT="${GPU_CONSTRAINT:-gpu}"
GPU_GRES="${GPU_GRES:-gpu:a100:1}"
GPU_CPUS_PER_TASK="${GPU_CPUS_PER_TASK:-12}"
GPU_MEM="${GPU_MEM:-64000}"

FIXED_TIME_LIMIT="${FIXED_TIME_LIMIT:-12:00:00}"
FIXED_CPUS_PER_TASK="${FIXED_CPUS_PER_TASK:-18}"
FIXED_MEM="${FIXED_MEM:-64000}"

REFIT_TIME_LIMIT="${REFIT_TIME_LIMIT:-12:00:00}"
REFIT_CPUS_PER_TASK="${REFIT_CPUS_PER_TASK:-8}"
REFIT_MEM="${REFIT_MEM:-64000}"

MERGE_TIME_LIMIT="${MERGE_TIME_LIMIT:-02:00:00}"
MERGE_CPUS_PER_TASK="${MERGE_CPUS_PER_TASK:-8}"
MERGE_MEM="${MERGE_MEM:-64000}"

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

"${PYTHON_BIN}" - "${CACHE_MANIFEST}" "${REFIT_MANIFEST}" "${SUBJECTS}" "${MODELS}" "${LAYER_CODE_DIR}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

cache_manifest = Path(sys.argv[1])
refit_manifest = Path(sys.argv[2])
subjects_arg = sys.argv[3]
models_arg = sys.argv[4]
layer_code_dir = sys.argv[5]

sys.path.insert(0, layer_code_dir)
from layers_config import get_layer_set  # noqa: E402

subjects = [item.strip() for item in subjects_arg.split(",") if item.strip()]
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

with refit_manifest.open("w") as f:
    for subject in subjects:
        for model in models:
            f.write(f"{subject}\t{model}\n")
PY

N_CACHE_TASKS="$(wc -l < "${CACHE_MANIFEST}" | tr -d ' ')"
N_REFIT_TASKS="$(wc -l < "${REFIT_MANIFEST}" | tr -d ' ')"
if [[ "${N_CACHE_TASKS}" -le 0 || "${N_REFIT_TASKS}" -le 0 ]]; then
  echo "Empty manifest(s): cache=${N_CACHE_TASKS} refit=${N_REFIT_TASKS}" >&2
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
  printf 'FIXED_WEIGHTS=%q\n' "${FIXED_WEIGHTS}"
  printf 'N_VICCO_BOOT=%q\n\n' "${N_VICCO_BOOT}"
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
"${PYTHON_BIN}" code/analysis/02_score_target_adaptation_fixed_alpha.py \
  --weights "${FIXED_WEIGHTS}" \
  --n-vicco-boot "${N_VICCO_BOOT}" \
  --overwrite \
  --overwrite-alpha
SLURM
} > "${FIXED_WORKER}"
chmod +x "${FIXED_WORKER}"

REFIT_WORKER="${RUN_ROOT}/refit_alpha_worker.slurm.sh"
{
  printf '#!/bin/bash -l\n'
  printf 'set -euo pipefail\n\n'
  printf 'PYTHON_BIN=%q\n' "${PYTHON_BIN}"
  printf 'CV_ROOT=%q\n' "${CV_ROOT}"
  printf 'REPO_ROOT=%q\n' "${REPO_ROOT}"
  printf 'LAYER_SWEEP_ROOT=%q\n' "${LAYER_SWEEP_ROOT}"
  printf 'REFIT_MANIFEST=%q\n' "${REFIT_MANIFEST}"
  printf 'CSTIMS_PATH_ENV=%q\n' "${CSTIMS_PATH_ENV}"
  printf 'REFIT_WEIGHTS=%q\n' "${REFIT_WEIGHTS}"
  printf 'N_VICCO_BOOT=%q\n' "${N_VICCO_BOOT}"
  printf 'REFIT_SHARD_PREFIX=%q\n\n' "${REFIT_SHARD_PREFIX}"
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

safe_name() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '_'
}

line="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${REFIT_MANIFEST}")"
if [[ -z "${line}" ]]; then
  echo "No refit manifest line for task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi
IFS=$'\t' read -r subject model <<< "${line}"
safe_subject="$(safe_name "${subject}")"
safe_model="$(safe_name "${model}")"
stem="${REFIT_SHARD_PREFIX}_${safe_subject}_${safe_model}"

cd "${CV_ROOT}"
"${PYTHON_BIN}" code/analysis/03_score_target_adaptation_refit_alpha.py \
  --subject "${subject}" \
  --models "${model}" \
  --weights "${REFIT_WEIGHTS}" \
  --n-vicco-boot "${N_VICCO_BOOT}" \
  --output-stem "${stem}" \
  --overwrite
SLURM
} > "${REFIT_WORKER}"
chmod +x "${REFIT_WORKER}"

MERGE_WORKER="${RUN_ROOT}/merge_refit_and_figures.slurm.sh"
{
  printf '#!/bin/bash -l\n'
  printf 'set -euo pipefail\n\n'
  printf 'PYTHON_BIN=%q\n' "${PYTHON_BIN}"
  printf 'CV_ROOT=%q\n' "${CV_ROOT}"
  printf 'REPO_ROOT=%q\n' "${REPO_ROOT}"
  printf 'LAYER_SWEEP_ROOT=%q\n' "${LAYER_SWEEP_ROOT}"
  printf 'REFIT_SHARD_PREFIX=%q\n' "${REFIT_SHARD_PREFIX}"
  printf 'N_REFIT_TASKS=%q\n' "${N_REFIT_TASKS}"
  printf 'RUN_ROOT=%q\n' "${RUN_ROOT}"
  printf 'CSTIMS_PATH_ENV=%q\n' "${CSTIMS_PATH_ENV}"
  printf 'REFIT_WEIGHTS=%q\n' "${REFIT_WEIGHTS}"
  printf 'FIXED_WEIGHTS=%q\n' "${FIXED_WEIGHTS}"
  printf 'N_VICCO_BOOT=%q\n\n' "${N_VICCO_BOOT}"
  cat <<'SLURM'
export CSTIMS_PATH_ENV
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export CV_ROOT RUN_ROOT REFIT_SHARD_PREFIX N_REFIT_TASKS REFIT_WEIGHTS FIXED_WEIGHTS N_VICCO_BOOT
export PYTHONPATH="${REPO_ROOT}/src:${CV_ROOT}/code:${CV_ROOT}/code/analysis:${LAYER_SWEEP_ROOT}/code:${LAYER_SWEEP_ROOT}/code/analysis:${PYTHONPATH:-}"

CONDA_PREFIX_GUESS="$(cd "$(dirname "${PYTHON_BIN}")/.." 2>/dev/null && pwd || true)"
CONDA_LIB="${CONDA_LIB:-${CONDA_PREFIX_GUESS}/lib}"
if [[ -d "${CONDA_LIB}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"
fi

cd "${CV_ROOT}"
"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

cv_root = Path(os.environ.get("CV_ROOT", ".")).resolve()
sys.path.insert(0, str(cv_root / "code"))
sys.path.insert(0, str(cv_root / "code" / "analysis"))

prefix = os.environ["REFIT_SHARD_PREFIX"]
expected = int(os.environ["N_REFIT_TASKS"])
run_root = Path(os.environ["RUN_ROOT"])
shard_dir = Path("results") / "03_refit_alpha" / "by_model" / "all_weights"
score_paths = sorted(shard_dir.glob(f"{prefix}_*_scores.csv"))
if len(score_paths) != expected:
    raise SystemExit(
        f"Expected {expected} refit shard score files for prefix {prefix}, "
        f"found {len(score_paths)} in {shard_dir}"
    )

frames = [pd.read_csv(path) for path in score_paths]
scores = pd.concat(frames, ignore_index=True)

script_path = Path("code/analysis/03_score_target_adaptation_refit_alpha.py")
spec = importlib.util.spec_from_file_location("_refit_alpha", script_path)
if spec is None or spec.loader is None:
    raise ImportError(script_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

scores = mod.sort_scores(scores)
weights = scores["target_weight"].astype(float).to_numpy()
is_w4700 = np.isclose(weights, 4700.0)
normal_scores = mod.sort_scores(scores.loc[~is_w4700].copy())
w4700_scores = mod.sort_scores(scores.loc[is_w4700].copy())

outputs = [
    ("target_adaptation_full_refit_all_weights_plus4700", scores, "combined refit grid plus 4700"),
    ("target_adaptation_full_refit_all_weights", normal_scores, "combined refit grid"),
    ("target_adaptation_full_refit_w4700", w4700_scores, "4700-only refit grid"),
]
for stem, frame, label in outputs:
    score_csv, summary_csv, meta_json = mod.output_paths(stem)
    score_csv.parent.mkdir(parents=True, exist_ok=True)
    mod.atomic_write_csv(frame, score_csv)
    mod.write_summary(frame, summary_csv)
    meta = {
        "mode": "raven_subject_model_sharded_merge",
        "label": label,
        "shard_prefix": prefix,
        "n_shards": len(score_paths),
        "expected_shards": expected,
        "shard_score_files": [str(path) for path in score_paths],
        "n_rows": int(len(frame)),
        "n_models": int(frame["model"].nunique()) if not frame.empty else 0,
        "n_subjects": int(frame["subject"].nunique()) if not frame.empty else 0,
        "weights": os.environ["REFIT_WEIGHTS"],
        "fixed_weights": os.environ["FIXED_WEIGHTS"],
        "n_vicco_boot": int(os.environ["N_VICCO_BOOT"]),
        "log_dir": str(run_root),
        "score_csv": str(score_csv),
        "summary_csv": str(summary_csv),
    }
    meta_json.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {len(frame)} rows -> {score_csv}", flush=True)
    print(f"Wrote summary -> {summary_csv}", flush=True)
    print(f"Wrote metadata -> {meta_json}", flush=True)
PY

"${PYTHON_BIN}" code/figures/02_fixed_alpha_plot_brain_alignment_grid.py --weight 2
"${PYTHON_BIN}" code/figures/02_fixed_alpha_plot_cstim_vicco_scatter.py --weight 2 --model-set all_models
"${PYTHON_BIN}" code/figures/03_refit_alpha_plot_weight_trajectories.py
"${PYTHON_BIN}" code/figures/03_refit_alpha_plot_weight0_to_best_cstim_grid.py
SLURM
} > "${MERGE_WORKER}"
chmod +x "${MERGE_WORKER}"

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

REFIT_SBATCH_ARGS=(
  --parsable
  --array="0-$((N_REFIT_TASKS - 1))%${REFIT_MAX_CONCURRENT}"
  --job-name="cstim_cv_refit"
  --time="${REFIT_TIME_LIMIT}"
  --cpus-per-task="${REFIT_CPUS_PER_TASK}"
  --mem="${REFIT_MEM}"
  --output="${RUN_ROOT}/logs/%x.%A_%a.out"
  --error="${RUN_ROOT}/logs/%x.%A_%a.err"
)
add_common_sbatch_args REFIT_SBATCH_ARGS

MERGE_SBATCH_ARGS=(
  --parsable
  --job-name="cstim_cv_merge"
  --time="${MERGE_TIME_LIMIT}"
  --cpus-per-task="${MERGE_CPUS_PER_TASK}"
  --mem="${MERGE_MEM}"
  --output="${RUN_ROOT}/logs/%x.%j.out"
  --error="${RUN_ROOT}/logs/%x.%j.err"
)
add_common_sbatch_args MERGE_SBATCH_ARGS

echo "Cache manifest: ${CACHE_MANIFEST} (${N_CACHE_TASKS} models)"
echo "Refit manifest: ${REFIT_MANIFEST} (${N_REFIT_TASKS} subject/model tasks)"
echo "Run root: ${RUN_ROOT}"
if [[ -n "${DEPENDENCY}" ]]; then
  echo "Initial dependency: ${DEPENDENCY}"
fi
printf 'Cache sbatch: sbatch'
printf ' %q' "${CACHE_SBATCH_ARGS[@]}" "${CACHE_WORKER}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Fixed sbatch: sbatch'
  printf ' %q' "${FIXED_SBATCH_ARGS[@]}" "${FIXED_WORKER}"
  printf '\n'
  printf 'Refit sbatch: sbatch'
  printf ' %q' "${REFIT_SBATCH_ARGS[@]}" "${REFIT_WORKER}"
  printf '\n'
  printf 'Merge sbatch: sbatch'
  printf ' %q' "${MERGE_SBATCH_ARGS[@]}" "${MERGE_WORKER}"
  printf '\n'
  echo "DRY_RUN=1; not submitting."
  exit 0
fi

CACHE_JOB_ID="$(sbatch "${CACHE_SBATCH_ARGS[@]}" "${CACHE_WORKER}")"
echo "Submitted cache GPU array job: ${CACHE_JOB_ID}"

FIXED_SBATCH_ARGS+=(--dependency="afterok:${CACHE_JOB_ID}")
FIXED_JOB_ID="$(sbatch "${FIXED_SBATCH_ARGS[@]}" "${FIXED_WORKER}")"
echo "Submitted fixed-alpha CPU job: ${FIXED_JOB_ID}"

REFIT_SBATCH_ARGS+=(--dependency="afterok:${FIXED_JOB_ID}")
REFIT_JOB_ID="$(sbatch "${REFIT_SBATCH_ARGS[@]}" "${REFIT_WORKER}")"
echo "Submitted refit-alpha CPU array job: ${REFIT_JOB_ID}"

MERGE_SBATCH_ARGS+=(--dependency="afterok:${REFIT_JOB_ID}")
MERGE_JOB_ID="$(sbatch "${MERGE_SBATCH_ARGS[@]}" "${MERGE_WORKER}")"
echo "Submitted merge/figure CPU job: ${MERGE_JOB_ID}"
