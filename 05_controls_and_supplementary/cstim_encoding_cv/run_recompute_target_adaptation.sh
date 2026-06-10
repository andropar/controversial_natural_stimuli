#!/usr/bin/env bash
set -euo pipefail

cd /data/home_roth/cstims_share/05_controls_and_supplementary/cstim_encoding_cv

PY=/data/home_roth/miniforge3/bin/python
GPU_DEVICE=${GPU_DEVICE:-cuda:6}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOG_DIR="logs/recompute_target_adaptation_${RUN_ID}"
mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_DIR}/run.log") 2>&1

echo "[start] $(date)"
echo "[cwd] $(pwd)"
echo "[python] ${PY}"
echo "[gpu] ${GPU_DEVICE}"
echo "[log] ${LOG_DIR}/run.log"

export LD_LIBRARY_PATH=/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_COMPILE_THREADS=1

echo "[phase] regenerate selected-layer SRP5920 caches with dense-chunk context"
"${PY}" code/analysis/00_fill_selected_layer_srp5920_cache.py \
  --batch-size 4 \
  --device "${GPU_DEVICE}" \
  --layers-per-chunk 32 \
  --progress-every 1024 \
  --overwrite

echo "[phase] recompute target-adaptation scores"
"${PY}" code/analysis/04_compute_target_adaptation_srp5920_voxelalpha.py \
  --weights 0,0.25,0.5,1,2,4,8 \
  --n-vicco-boot 1000 \
  --overwrite \
  --overwrite-alpha

echo "[phase] rebuild figures"
"${PY}" code/figures/04_plot_target_adaptation_trajectory.py
"${PY}" code/figures/05_plot_target_adaptation_condition_comparisons.py \
  --weight 2 \
  --model-set all_models
"${PY}" code/figures/05_plot_target_adaptation_condition_comparisons.py \
  --weight 2 \
  --model-set all_models \
  --label-models
"${PY}" code/figures/06_plot_target_adaptation_brain_alignment_style_allpoints.py \
  --weight 2

echo "[phase] verify weight-0 equality against source layer-sweep table"
"${PY}" - <<'PY'
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

echo "[done] $(date)"
