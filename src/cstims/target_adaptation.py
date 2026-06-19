"""Shared helpers for CSTIM target-adaptation analyses."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from cstims.sampling import bootstrap_sample_indices

CSTIM_SETS = ["all_models", "architecture", "dataset", "sota", "training_objective"]
BASELINE_SET = "vicco"
PANEL_ORDER = ["all_models", "sota", "training_objective", "architecture", "dataset"]
MODEL_SET_TITLES = {
    "training_objective": "Train. Objective",
    "sota": "State of the Art",
    "architecture": "Architecture",
    "dataset": "Dataset",
    "all_models": "All Models",
}
SHORT_MODEL_NAMES = {
    "torchvision_vgg16_imagenet1k_v1": "VGG-16",
    "torchvision_resnet50_imagenet1k_v1": "ResNet-50",
    "torchvision_convnext_base_imagenet1k_v1": "ConvNeXt-B",
    "torchvision_vit_l_16_imagenet1k_v1": "ViT-L/16",
    "cornet_s": "CORnet-S",
    "vissl_resnet50_supervised": "Supervised",
    "vissl_resnet50_barlowtwins": "BarlowTwins",
    "vissl_resnet50_mocov2": "MoCoV2",
    "vicreg_resnet50": "VICReg",
    "robustness_imagenet_l2_eps3": "Robust-L2",
    "slip_vit_l_slip": "SLIP",
    "slip_vit_l_simclr": "SimCLR-ViT",
    "timm_vit_large_patch14_clip_224_laion2b": "CLIP-L2B",
    "dinov2_vitl14": "DINOv2",
    "openclip_vit_so400m_14_siglip_webli": "SigLIP",
    "openclip_vit_l_14_quickgelu_metaclip_400m": "MC-400M",
    "openclip_vit_l_14_quickgelu_metaclip_fullcc": "MC-Full",
    "timm_vit_large_patch14_clip_224_dfn2b": "DFN-2B",
    "timm_vit_large_patch14_clip_quickgelu_224_openai": "CLIP-OAI",
    "openclip_vit_l_14_laion400m_e31": "CLIP-L400",
}

RESPONSE_ZSCORE_EPS = 1e-6
FEATURE_ZSCORE_EPS = 1e-6


@dataclass(frozen=True)
class Selection:
    subject: str
    model: str
    display_name: str
    layer: str


def sanitize_layer_name(layer: str) -> str:
    return (
        str(layer)
        .replace(".", "_")
        .replace(":", "_")
        .replace("[", "_")
        .replace("]", "_")
        .replace("/", "_")
    )


def stable_seed(*parts: str, base: int = 20260607) -> int:
    text = "::".join(str(p) for p in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return (int.from_bytes(digest, "little") + int(base)) % (2**31 - 1)


def stable_layer_seed(model: str, layer: str, *, base_seed: int) -> int:
    """Match the layer-sweep convention for per-model/per-layer SRP seeds."""
    return stable_seed(model, layer, base=base_seed)


def parse_weights(spec: str) -> list[float]:
    values = [float(chunk.strip()) for chunk in spec.split(",") if chunk.strip()]
    if not values:
        raise ValueError("No target weights provided")
    return values


def load_best_shared_selections(selection_csv: Path) -> pd.DataFrame:
    """Load one best-on-shared selected layer per subject/model pair."""
    df = pd.read_csv(selection_csv)
    rows = df[
        df["selection_rule"].eq("best_on_shared")
        & df["selection_model_set"].eq("deepvision_shared")
        & df["eval_target"].eq("shared")
    ].copy()
    columns = [
        "subject",
        "model",
        "display_name",
        "selected_layer",
        "selected_layer_index",
        "selected_layer_frac",
        "selection_mrsa",
    ]
    rows = rows[[col for col in columns if col in rows.columns]].drop_duplicates(
        ["subject", "model"]
    )
    return rows.rename(columns={"selected_layer": "layer"})


def load_original_reference(
    selection_csv: Path,
    *,
    baseline_set: str = BASELINE_SET,
) -> pd.DataFrame:
    """Load source-table mRSA scores for selected CSTIM and baseline rows."""
    df = pd.read_csv(selection_csv)
    rows = df[
        df["selection_rule"].eq("best_on_shared")
        & df["selection_model_set"].eq("deepvision_shared")
        & df["eval_target"].isin(["cstim", baseline_set])
    ].copy()
    out = rows[
        [
            "subject",
            "model",
            "eval_target",
            "eval_model_set",
            "mrsa_mean",
            "mrsa_sem",
            "selected_layer",
        ]
    ].rename(
        columns={
            "eval_model_set": "model_set",
            "mrsa_mean": "original_best_shared_mrsa",
            "mrsa_sem": "original_best_shared_mrsa_sem",
            "selected_layer": "layer",
        }
    )
    out.loc[out["eval_target"].eq(baseline_set), "model_set"] = baseline_set
    return out


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_savez_compressed(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def zscore_targets_by_voxel(
    targets: np.ndarray,
    *,
    eps: float = RESPONSE_ZSCORE_EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = targets.mean(axis=0, dtype=np.float64)
    std = targets.std(axis=0, dtype=np.float64, ddof=0)
    std = np.maximum(std, eps)
    standardized = (targets - mean) / std
    return standardized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def zscore_features_from_deepvision(
    features: np.ndarray,
    *,
    eps: float = FEATURE_ZSCORE_EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features.std(axis=0, dtype=np.float64, ddof=0).astype(np.float32)
    scale = np.maximum(scale, eps)
    standardized = (features - mean) / scale
    return standardized.astype(np.float32), mean, scale


def apply_feature_zscore(
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return ((features - mean) / scale).astype(np.float32)


def layer_sweep_eval_design(
    features_z: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    *,
    eps: float = FEATURE_ZSCORE_EPS,
) -> np.ndarray:
    """Evaluation design that mirrors the dense layer-sweep stream scorer.

    The stream scorer stores raw-space ridge weights but then standardizes
    evaluation features again before prediction. The target-adaptation scorer
    works in standardized primal/dual space, so this transform reproduces that
    stream prediction convention for already-standardized features.
    """
    scale = np.maximum(np.asarray(feature_scale, dtype=np.float64), eps)
    mean_over_scale = np.asarray(feature_mean, dtype=np.float64) / scale
    return np.asarray(features_z, dtype=np.float64) / scale[None, :] - mean_over_scale[None, :]


def rdm_corr(features: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(np.asarray(features, dtype=np.float64))
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    rdm = 1.0 - corr
    np.fill_diagonal(rdm, 0.0)
    return rdm


def rdm_upper_vec(rdm: np.ndarray) -> np.ndarray:
    return rdm[np.triu_indices(rdm.shape[0], k=1)]


def rank_vector(vec: np.ndarray) -> np.ndarray:
    return stats.rankdata(vec, method="average").astype(np.float32)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xm = x - x.mean()
    ym = y - y.mean()
    den = float(np.sqrt(np.dot(xm, xm) * np.dot(ym, ym)))
    if den <= 0:
        return float("nan")
    return float(np.dot(xm, ym) / den)


def rsa_spearman(pred: np.ndarray, brain: np.ndarray) -> float:
    pr = rdm_corr(pred)
    br = rdm_corr(brain)
    idx = np.triu_indices(pr.shape[0], k=1)
    val = stats.spearmanr(pr[idx], br[idx]).statistic
    return float(val) if np.isfinite(val) else np.nan


def rsa_spearman_bootstrap_mean(
    pred: np.ndarray,
    *,
    boot: list[np.ndarray],
    brain_ranks: list[np.ndarray],
) -> tuple[float, float]:
    vals = []
    for idx, brain_rank in zip(boot, brain_ranks):
        pred_rank = rank_vector(rdm_upper_vec(rdm_corr(pred[idx])))
        vals.append(pearson_r(pred_rank, brain_rank))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    sem = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan
    return float(vals.mean()), sem


def sem(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return 0.0
    return float(vals.std(ddof=1) / np.sqrt(len(vals)))


def mean_ci(vals: np.ndarray) -> tuple[float, float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    mean = float(vals.mean())
    if len(vals) <= 1:
        return mean, mean, mean
    err = 1.96 * sem(vals)
    return mean, mean - err, mean + err
