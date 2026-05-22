#!/usr/bin/env python3
"""Fit paper-layer encoding models in the full-data ROI-union voxel space."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV


THIS = Path(__file__).resolve()
RERUN_ROOT = THIS.parents[1]
SHARE_ROOT = THIS.parents[3]
HELPERS = SHARE_ROOT / "shared" / "code" / "paper_helpers"
sys.path.insert(0, str(HELPERS))

import config  # noqa: E402


SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
UNIQUE_CACHE = RERUN_ROOT / "results" / "deepvision_unique_cache"
CSTIM_CACHE = RERUN_ROOT / "results" / "brain_data_cache"
OUT_ROOT = RERUN_ROOT / "results" / "encoding_models" / "paper_layer"
FEATURE_ROOTS = [
    SHARE_ROOT
    / "01_brain_model_alignment"
    / "results"
    / "encoding_models"
    / "subject_unique_encoding_models"
    / "runs",
    SHARE_ROOT
    / "01_brain_model_alignment"
    / "inputs"
    / "encoding_models"
    / "subject_unique_encoding_models"
    / "runs",
]
ALPHAS = np.logspace(np.log10(0.1), np.log10(1e7), 20)


def layer_safe(layer: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", layer.replace(".", "_")).strip("_")


def model_layer_map() -> dict[str, str]:
    df = pd.read_csv(config.MODEL_LIST_CSV)
    return {r["model"]: r["layer"] for _, r in df.iterrows()}


def feature_path(subject: str, model: str, layer: str) -> Path:
    safe = layer_safe(layer)
    candidates = []
    for root in FEATURE_ROOTS:
        candidates.extend(root.glob(f"*/{subject}_{model}.layer{safe}/features.npz"))
    candidates = sorted(candidates)
    if not candidates:
        raise FileNotFoundError(f"No features for {subject}/{model}/layer{safe}")
    return candidates[-1]


def roi_union(subject: str):
    z = np.load(CSTIM_CACHE / subject / "voxel_metadata.npz", allow_pickle=True)
    names = [str(x) for x in z["roi_names"]]
    union = np.zeros_like(z["visual_mask"], dtype=bool)
    roi_masks = {}
    for name in names:
        mask = z[f"roi_{name}"].astype(bool)
        roi_masks[name] = mask
        union |= mask
    return union, roi_masks


def fit_one(subject: str, model: str, layer: str, overwrite: bool):
    out_dir = OUT_ROOT / subject / f"{model}.layer{layer_safe(layer)}"
    out_path = out_dir / "encoding_model.npz"
    if out_path.exists() and not overwrite:
        print(f"[cached] {subject} {model} {layer}", flush=True)
        return

    fp = feature_path(subject, model, layer)
    X = np.load(fp, allow_pickle=True)["features"].astype(np.float32)
    union, roi_masks = roi_union(subject)
    unique_z = np.load(UNIQUE_CACHE / subject / "unique_betas_averaged.npz", allow_pickle=True)
    Y_loaded = unique_z["betas"]
    voxel_space = str(unique_z["voxel_space"]) if "voxel_space" in unique_z.files else "brain_mask"
    if voxel_space == "roi_union":
        cached_union = unique_z["union_mask"].astype(bool)
        if not np.array_equal(cached_union, union):
            raise RuntimeError(f"{subject}: unique cache union mask does not match cstim cache")
        Y = Y_loaded.T.astype(np.float32)
    elif voxel_space == "brain_mask":
        Y = Y_loaded[union, :].T.astype(np.float32)
    else:
        raise RuntimeError(f"{subject}: unknown unique cache voxel_space={voxel_space}")

    x_mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_scale = X.std(axis=0, dtype=np.float64).astype(np.float32)
    x_scale = np.maximum(x_scale, 1e-6)
    Xz = ((X - x_mean) / x_scale).astype(np.float32)

    y_mean = Y.mean(axis=0, dtype=np.float64).astype(np.float32)
    y_scale = Y.std(axis=0, dtype=np.float64).astype(np.float32)
    y_scale = np.maximum(y_scale, 1e-6)
    Yz = ((Y - y_mean) / y_scale).astype(np.float32)

    print(f"[fit] {subject} {model} {layer}: X={Xz.shape}, Y={Yz.shape}", flush=True)
    reg = RidgeCV(alphas=ALPHAS, fit_intercept=True, alpha_per_target=True, gcv_mode="auto")
    reg.fit(Xz, Yz)

    roi_payload = {}
    union_idx = np.where(union)[0]
    for name, mask in roi_masks.items():
        roi_payload[f"roi_{name}"] = np.isin(union_idx, np.where(mask)[0])

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        weights=reg.coef_.T.astype(np.float32),
        intercept=reg.intercept_.astype(np.float32),
        feature_mean=x_mean,
        feature_scale=x_scale,
        response_mean=y_mean,
        response_scale=y_scale,
        alphas=np.asarray(reg.alpha_, dtype=np.float32),
        alpha_grid=ALPHAS.astype(np.float32),
        union_mask=union,
        roi_names=np.asarray(sorted(roi_masks)),
        model=np.asarray(model),
        subject=np.asarray(subject),
        layer=np.asarray(layer),
        feature_path=np.asarray(str(fp)),
        protocol=np.asarray("full_data_rerun_ridgecv_gcv_v1"),
        **roi_payload,
    )
    print(f"[wrote] {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="all")
    parser.add_argument("--models", default="all", help="comma-separated model names or all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    subjects = SUBJECTS if args.subject == "all" else [args.subject]
    layers = model_layer_map()
    models = sorted(config.MODEL_SETS["all_models"]) if args.models == "all" else args.models.split(",")
    for subject in subjects:
        for model in models:
            fit_one(subject, model, layers[model], args.overwrite)


if __name__ == "__main__":
    main()
