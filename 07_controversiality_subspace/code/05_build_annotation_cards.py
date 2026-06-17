#!/usr/bin/env python
"""Build contrastive residual-neighborhood annotation cards.

This uses cached feature arrays and local image files only. It does not import
torch, load image models, or touch the GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_APP_ROOT = REPO_ROOT / "07_controversiality_subspace" / "annotation_app"
DEFAULT_SELECTED_ROOT = (
    REPO_ROOT / "00_stimulus_selection" / "results" / "selected_stimuli"
)
DEFAULT_FEATURE_CACHE = (
    REPO_ROOT
    / "shared"
    / "cache_or_heavy"
    / "cstim_paper_feature_cache"
    / "feature_cache"
    / "vicco"
)
DEFAULT_IMAGE_ROOTS = (
    Path("/data/labshare/_stachelschwein/SSD/jroth/final_cstims_hdf5_files"),
    REPO_ROOT / "external_data" / "final_cstims_hdf5_files",
)
DEFAULT_BRAIN_DATA = (
    REPO_ROOT
    / "01_brain_model_alignment"
    / "cache_or_heavy"
    / "cstim_brain_response_cache"
    / "data"
)

SUBJECTS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")

SELECTED_CONDITIONS = (
    "all_models",
    "architecture",
    "dataset",
    "sota",
    "training_objective",
)

CUE_CATEGORIES = [
    {"id": "object_category", "label": "Object/category"},
    {"id": "shape_parts", "label": "Shape / parts / silhouette"},
    {"id": "texture_material", "label": "Texture / material / surface pattern"},
    {"id": "color_brightness", "label": "Color / brightness / contrast"},
    {"id": "scene_context", "label": "Scene / background / context"},
    {"id": "layout_viewpoint", "label": "Spatial layout / viewpoint / composition"},
    {"id": "function_action", "label": "Function / action / affordance"},
    {"id": "style_artifacts", "label": "Style / image statistics / artifacts"},
    {"id": "abstract_semantic", "label": "Abstract semantic association"},
    {"id": "no_clear_other", "label": "No clear commonality / other"},
]

CONTRAST_OPTIONS = [
    {"id": "same_reason", "label": "No, mostly same reason"},
    {"id": "partly_different", "label": "Partly different"},
    {"id": "clearly_different", "label": "Clearly different"},
    {"id": "unclear_group", "label": "One or both groups do not make sense"},
]


@dataclass(frozen=True)
class Config:
    app_root: Path
    selected_root: Path
    feature_cache: Path
    image_root: Path | None
    conditions: tuple[str, ...]
    cards_per_selected_condition: int
    vicco_cards: int
    n_neighbors: int
    seed: int
    profile_normalization: str
    thumbnail_size: int
    include_random_controls: bool
    include_consensus_controls: bool
    brain_data: Path
    include_brain_bins: bool
    brain_bin_threshold: float
    brain_consistency_sd: float


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-root", type=Path, default=DEFAULT_APP_ROOT)
    parser.add_argument("--selected-root", type=Path, default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument(
        "--conditions",
        type=str,
        default=",".join((*SELECTED_CONDITIONS, "vicco")),
    )
    parser.add_argument("--cards-per-selected-condition", type=int, default=100)
    parser.add_argument("--vicco-cards", type=int, default=100)
    parser.add_argument("--n-neighbors", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument(
        "--profile-normalization",
        choices=("rankz", "zscore"),
        default="rankz",
    )
    parser.add_argument("--thumbnail-size", type=int, default=360)
    parser.add_argument("--brain-data", type=Path, default=DEFAULT_BRAIN_DATA)
    parser.add_argument(
        "--no-random-controls",
        action="store_true",
        help="Only emit real residual-neighborhood cards.",
    )
    parser.add_argument(
        "--include-consensus-controls",
        action="store_true",
        help="Also emit consensus-neighborhood control cards.",
    )
    parser.add_argument(
        "--no-brain-bins",
        action="store_true",
        help="Do not compute subject-averaged brain similarity bins for image pairs.",
    )
    parser.add_argument("--brain-bin-threshold", type=float, default=0.75)
    parser.add_argument("--brain-consistency-sd", type=float, default=1.0)
    args = parser.parse_args()
    conditions = tuple(x.strip() for x in args.conditions.split(",") if x.strip())
    image_root = args.image_root or find_image_root()
    return Config(
        app_root=args.app_root,
        selected_root=args.selected_root,
        feature_cache=args.feature_cache,
        image_root=image_root,
        conditions=conditions,
        cards_per_selected_condition=args.cards_per_selected_condition,
        vicco_cards=args.vicco_cards,
        n_neighbors=args.n_neighbors,
        seed=args.seed,
        profile_normalization=args.profile_normalization,
        thumbnail_size=args.thumbnail_size,
        include_random_controls=not args.no_random_controls,
        include_consensus_controls=args.include_consensus_controls,
        brain_data=args.brain_data,
        include_brain_bins=not args.no_brain_bins,
        brain_bin_threshold=args.brain_bin_threshold,
        brain_consistency_sd=args.brain_consistency_sd,
    )


def find_image_root() -> Path:
    for root in DEFAULT_IMAGE_ROOTS:
        if (root / "shared_vicco").exists() and (root / "all_models").exists():
            return root
    return DEFAULT_IMAGE_ROOTS[0]


def physical_folder(condition: str) -> str:
    if condition == "architecture":
        return "dataset"
    if condition == "dataset":
        return "architecture"
    if condition == "vicco":
        return "shared_vicco"
    return condition


def image_files_for_condition(cfg: Config, condition: str) -> list[Path]:
    if cfg.image_root is None:
        raise FileNotFoundError("No image root configured.")
    image_dir = cfg.image_root / physical_folder(condition)
    files = sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not files:
        raise FileNotFoundError(f"No image files found for {condition}: {image_dir}")
    return files


def load_payload_models(cfg: Config, condition: str) -> list[str]:
    if condition == "vicco":
        return sorted(
            p.parent.name
            for p in cfg.feature_cache.glob("*/vicco.npy")
            if p.is_file()
        )
    payload_path = cfg.selected_root / condition / "selected_stimuli_data.pkl"
    with payload_path.open("rb") as f:
        payload = pickle.load(f)
    payload_models = set(payload.get("selected_features_raw", {}).keys())
    cache_models = {
        p.parent.name
        for p in cfg.feature_cache.glob(f"*/{condition}.npy")
        if p.is_file()
    }
    return sorted(payload_models & cache_models)


def load_condition_features(
    cfg: Config, condition: str, models: list[str]
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for model in models:
        path = cfg.feature_cache / model / f"{condition}.npy"
        if path.exists():
            out[model] = np.asarray(np.load(path), dtype=np.float32)
    if len(out) < 2:
        raise RuntimeError(
            f"Need at least two cached models for {condition}; found {len(out)}."
        )
    return out


def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def normalize_profiles(sim: np.ndarray, method: str) -> np.ndarray:
    sim = np.asarray(sim, dtype=np.float32)
    if method == "zscore":
        mu = sim.mean(axis=1, keepdims=True)
        sd = sim.std(axis=1, keepdims=True)
        return (sim - mu) / np.maximum(sd, 1e-6)
    if method == "rankz":
        order = np.argsort(sim, axis=1)
        ranks = np.empty(order.shape, dtype=np.float32)
        row_ids = np.arange(sim.shape[0])[:, None]
        ranks[row_ids, order] = np.arange(sim.shape[1], dtype=np.float32)
        mu = ranks.mean(axis=1, keepdims=True)
        sd = ranks.std(axis=1, keepdims=True)
        return (ranks - mu) / np.maximum(sd, 1e-6)
    raise ValueError(f"Unknown profile normalization: {method}")


def finite_mean_square(x: np.ndarray, axis: int) -> np.ndarray:
    valid = np.isfinite(x)
    numerator = np.where(valid, x * x, 0.0).sum(axis=axis)
    denominator = valid.sum(axis=axis)
    return numerator / np.maximum(denominator, 1)


def finite_rms_diff(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if not np.any(valid):
        return float("nan")
    diff = a[valid] - b[valid]
    return float(np.sqrt(np.mean(diff * diff)))


def compute_rdm_correlation(features: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(features)
    rdm = 1.0 - corr
    np.fill_diagonal(rdm, 0.0)
    return rdm


def zscore_condensed_to_square(vec: np.ndarray, n: int) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    mu = np.nanmean(vec)
    sd = np.nanstd(vec, ddof=0)
    vals = np.zeros_like(vec) if sd <= 0 or not np.isfinite(sd) else (vec - mu) / sd
    mat = np.zeros((n, n), dtype=np.float32)
    tri = np.triu_indices(n, k=1)
    mat[tri] = vals.astype(np.float32)
    return mat + mat.T


def brain_source_index(condition: str, stim_idx: int) -> int:
    if condition == "vicco":
        return int(stim_idx) - 1
    return int(stim_idx)


def load_subject_brain_zmat(
    cfg: Config, condition: str, subject: str, n_images: int
) -> np.ndarray | None:
    data_dir = cfg.brain_data / subject
    betas_path = data_dir / "cstim_betas_averaged.npz"
    voxel_path = data_dir / "voxel_metadata.npz"
    stim_path = data_dir / "cstim_stimulus_info.csv"
    if not (betas_path.exists() and voxel_path.exists() and stim_path.exists()):
        return None
    import pandas as pd

    betas_data = np.load(betas_path, allow_pickle=True)
    voxel_data = np.load(voxel_path, allow_pickle=True)
    stim_info = pd.read_csv(stim_path)
    group_info = stim_info[stim_info["group"] == condition].copy()
    if group_info.empty:
        return None

    stim_keys = list(betas_data["stim_keys"])
    key_to_idx = {str(key): idx for idx, key in enumerate(stim_keys)}
    rows = []
    for row in group_info.itertuples(index=False):
        stim_key = str(row.stim_key)
        if stim_key not in key_to_idx:
            continue
        source_idx = brain_source_index(condition, int(row.stim_idx))
        if source_idx < 0 or source_idx >= n_images:
            continue
        rows.append((source_idx, key_to_idx[stim_key]))
    if len(rows) < 3:
        return None

    source_indices = np.asarray([r[0] for r in rows], dtype=np.int64)
    brain_indices = np.asarray([r[1] for r in rows], dtype=np.int64)
    hlvis_mask = voxel_data["hlvis_mask"]
    beta_hlvis = betas_data["betas"][hlvis_mask, :]
    rdm = compute_rdm_correlation(beta_hlvis[:, brain_indices].T)
    tri = np.triu_indices(rdm.shape[0], k=1)
    zmat_order = zscore_condensed_to_square(rdm[tri], rdm.shape[0])
    zmat = np.full((n_images, n_images), np.nan, dtype=np.float32)
    for local_i, source_i in enumerate(source_indices):
        for local_j, source_j in enumerate(source_indices):
            zmat[int(source_i), int(source_j)] = zmat_order[local_i, local_j]
    return zmat


def load_brain_pair_summary(
    cfg: Config, condition: str, n_images: int
) -> dict[str, Any]:
    if not cfg.include_brain_bins:
        return {"available": False, "reason": "disabled"}
    subject_mats = []
    subjects = []
    for subject in SUBJECTS:
        zmat = load_subject_brain_zmat(cfg, condition, subject, n_images)
        if zmat is not None:
            subject_mats.append(zmat)
            subjects.append(subject)
    if not subject_mats:
        return {"available": False, "reason": "no_subject_matrices"}
    stack = np.stack(subject_mats, axis=0)
    valid = np.isfinite(stack)
    count = valid.sum(axis=0).astype(np.int16)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(valid, stack, 0.0).sum(axis=0) / np.maximum(count, 1)
    sd = np.full((n_images, n_images), np.nan, dtype=np.float32)
    enough = count > 1
    if np.any(enough):
        centered = np.where(valid, stack - mean[None, :, :], 0.0)
        sd[enough] = np.sqrt(
            (centered * centered).sum(axis=0)[enough] / np.maximum(count[enough] - 1, 1)
        )
    mean[count == 0] = np.nan
    return {
        "available": True,
        "subjects": subjects,
        "mean": mean.astype(np.float32),
        "sd": sd,
        "count": count,
    }


def brain_bin(cfg: Config, mean_brain_z: float | None) -> str:
    if mean_brain_z is None or not math.isfinite(float(mean_brain_z)):
        return "unknown"
    if mean_brain_z <= -cfg.brain_bin_threshold:
        return "similar"
    if mean_brain_z >= cfg.brain_bin_threshold:
        return "dissimilar"
    return "neutral"


def brain_pair_info(
    cfg: Config,
    brain_summary: dict[str, Any],
    source_i: int,
    source_j: int,
) -> dict[str, Any]:
    if not brain_summary.get("available"):
        return {
            "bin": "unknown",
            "mean_brain_z": None,
            "sd_brain_z": None,
            "n_subjects": 0,
            "subject_consistent": None,
        }
    i, j = sorted((int(source_i), int(source_j)))
    count = int(brain_summary["count"][i, j])
    mean_z = none_if_nan(float(brain_summary["mean"][i, j]))
    sd_z = none_if_nan(float(brain_summary["sd"][i, j]))
    return {
        "bin": brain_bin(cfg, mean_z),
        "mean_brain_z": mean_z,
        "sd_brain_z": sd_z,
        "n_subjects": count,
        "subject_consistent": (
            bool(sd_z < cfg.brain_consistency_sd) if sd_z is not None and count > 1 else None
        ),
    }


def summarize_card_brain_pairs(card: dict[str, Any]) -> dict[str, Any]:
    counts = {"similar": 0, "neutral": 0, "dissimilar": 0, "unknown": 0}
    values = []
    consistent_known = 0
    known = 0
    for group in card.get("groups", []):
        for anchor in group.get("anchors", []):
            pair = anchor.get("brain_pair") or {}
            bin_name = pair.get("bin") or "unknown"
            counts[bin_name if bin_name in counts else "unknown"] += 1
            if pair.get("mean_brain_z") is not None:
                values.append(float(pair["mean_brain_z"]))
                known += 1
                if pair.get("subject_consistent") is True:
                    consistent_known += 1
    known_counts = {k: v for k, v in counts.items() if k != "unknown"}
    if sum(known_counts.values()) == 0:
        dominant = "unknown"
    else:
        dominant = max(known_counts.items(), key=lambda item: item[1])[0]
    return {
        "counts": counts,
        "dominant_bin": dominant,
        "known_pairs": known,
        "total_pairs": int(sum(counts.values())),
        "mean_brain_z": none_if_nan(float(np.mean(values))) if values else None,
        "median_brain_z": none_if_nan(float(np.median(values))) if values else None,
        "consistent_known_pairs": int(consistent_known),
    }


def compute_residual_geometry(
    cfg: Config,
    features: dict[str, np.ndarray],
    query_indices: np.ndarray,
    anchor_indices: np.ndarray,
) -> dict[str, Any]:
    models = sorted(features)
    profiles = []
    raw_sims = []
    for model in models:
        feats = normalize_rows(features[model])
        query = feats[query_indices]
        anchors = feats[anchor_indices]
        sim = query @ anchors.T
        for q, source_idx in enumerate(query_indices):
            hits = np.flatnonzero(anchor_indices == source_idx)
            if hits.size:
                sim[q, hits] = -2.0
        profiles.append(normalize_profiles(sim, cfg.profile_normalization))
        raw_sims.append(sim.astype(np.float32, copy=False))
    profile = np.stack(profiles, axis=1)
    raw_sim = np.stack(raw_sims, axis=1)
    consensus = profile.mean(axis=1, keepdims=True)
    residual = profile - consensus
    for q, source_idx in enumerate(query_indices):
        hits = np.flatnonzero(anchor_indices == source_idx)
        if hits.size:
            residual[q, :, hits] = np.nan
            profile[q, :, hits] = np.nan
            raw_sim[q, :, hits] = np.nan
    c_local = finite_mean_square(residual, axis=(1, 2))
    model_energy = finite_mean_square(residual, axis=2)
    consensus_profile = np.nanmean(profile, axis=1)
    return {
        "models": models,
        "profiles": profile,
        "raw_sims": raw_sim,
        "residuals": residual,
        "consensus_profile": consensus_profile,
        "c_local": c_local,
        "model_energy": model_energy,
    }


def top_indices(values: np.ndarray, n: int, exclude: set[int] | None = None) -> list[int]:
    exclude = exclude or set()
    order = np.argsort(np.nan_to_num(values, nan=-np.inf))[::-1]
    out = []
    for idx in order:
        idx_int = int(idx)
        if idx_int in exclude:
            continue
        value = values[idx_int]
        if not np.isfinite(value):
            continue
        out.append(idx_int)
        if len(out) >= n:
            break
    return out


def strongest_pair(residuals: np.ndarray, models: list[str], q: int) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            distance = finite_rms_diff(residuals[q, i], residuals[q, j])
            if best is None or distance > best["residual_profile_distance"]:
                best = {
                    "model_i": models[i],
                    "model_j": models[j],
                    "model_i_index": i,
                    "model_j_index": j,
                    "residual_profile_distance": distance,
                }
    if best is None:
        raise RuntimeError("No model pair found.")
    return best


def make_anchor_entry(
    condition: str,
    anchor_source_index: int,
    anchor_rank: int,
    asset: str,
    residual_value: float | None = None,
    raw_similarity: float | None = None,
    consensus_similarity: float | None = None,
    brain_pair: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "condition": condition,
        "source_index": int(anchor_source_index),
        "rank": int(anchor_rank),
        "asset": asset,
        "positive_residual": none_if_nan(residual_value),
        "raw_similarity": none_if_nan(raw_similarity),
        "consensus_profile_value": none_if_nan(consensus_similarity),
        "brain_pair": brain_pair
        or {
            "bin": "unknown",
            "mean_brain_z": None,
            "sd_brain_z": None,
            "n_subjects": 0,
            "subject_consistent": None,
        },
    }


def none_if_nan(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def save_thumbnail(src: Path, dest: Path, size: int) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        img = Image.new("RGB", (size, size), color=(235, 235, 235))
        draw = ImageDraw.Draw(img)
        draw.text((12, size // 2 - 8), "missing", fill=(70, 70, 70))
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), color=(250, 250, 248))
    offset = ((size - img.width) // 2, (size - img.height) // 2)
    canvas.paste(img, offset)
    canvas.save(dest, format="JPEG", quality=88, optimize=True)


def prepare_assets(
    cfg: Config, condition: str, image_files: list[Path]
) -> dict[int, str]:
    asset_root = cfg.app_root / "static" / "stimuli" / condition
    asset_map: dict[int, str] = {}
    for index, src in enumerate(image_files):
        dest = asset_root / f"{index:04d}.jpg"
        save_thumbnail(src, dest, cfg.thumbnail_size)
        asset_map[index] = f"static/stimuli/{condition}/{index:04d}.jpg"
    return asset_map


def choose_query_indices(
    cfg: Config,
    condition: str,
    n_images: int,
    c_local: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if condition == "vicco":
        n = min(cfg.vicco_cards, n_images)
        return np.sort(rng.choice(n_images, size=n, replace=False)).astype(np.int64)
    n = min(cfg.cards_per_selected_condition, n_images)
    if n >= n_images:
        return np.arange(n_images, dtype=np.int64)
    order = np.argsort(c_local)[::-1][:n]
    return np.sort(order).astype(np.int64)


def card_base(
    cfg: Config,
    condition: str,
    card_type: str,
    query_source_index: int,
    query_rank_index: int,
    c_local: float,
    central_asset: str,
    hidden: dict[str, Any],
) -> dict[str, Any]:
    return {
        "card_id": f"{condition}__{card_type}__{query_source_index:04d}",
        "condition": condition,
        "condition_label": "VICCO baseline" if condition == "vicco" else condition,
        "card_type": card_type,
        "query_source_index": int(query_source_index),
        "query_rank_index": int(query_rank_index),
        "local_residual_geometry": none_if_nan(c_local),
        "central": {
            "asset": central_asset,
            "source_index": int(query_source_index),
        },
        "groups": [],
        "hidden": hidden,
    }


def residual_card(
    cfg: Config,
    condition: str,
    q: int,
    source_index: int,
    models: list[str],
    anchor_indices: np.ndarray,
    asset_map: dict[int, str],
    geometry: dict[str, Any],
    brain_summary: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    pair = strongest_pair(geometry["residuals"], models, q)
    model_indices = [pair["model_i_index"], pair["model_j_index"]]
    groups = []
    for model_index in model_indices:
        local_anchor_positions = top_indices(
            geometry["residuals"][q, model_index], cfg.n_neighbors
        )
        anchors = []
        for rank, local_anchor_pos in enumerate(local_anchor_positions, start=1):
            anchor_source_index = int(anchor_indices[local_anchor_pos])
            anchors.append(
                make_anchor_entry(
                    condition=condition,
                    anchor_source_index=anchor_source_index,
                    anchor_rank=rank,
                    asset=asset_map[anchor_source_index],
                    residual_value=geometry["residuals"][q, model_index, local_anchor_pos],
                    raw_similarity=geometry["raw_sims"][q, model_index, local_anchor_pos],
                    consensus_similarity=geometry["consensus_profile"][q, local_anchor_pos],
                    brain_pair=brain_pair_info(
                        cfg, brain_summary, source_index, anchor_source_index
                    ),
                )
            )
        groups.append(
            {
                "display_name": "",
                "group_index": 0,
                "hidden_model": models[model_index],
                "hidden_model_index": int(model_index),
                "anchors": anchors,
            }
        )
    if rng.random() < 0.5:
        groups.reverse()
    for idx, group in enumerate(groups, start=1):
        group["display_name"] = f"Group {idx}"
        group["group_index"] = idx
    hidden = {
        "model_i": pair["model_i"],
        "model_j": pair["model_j"],
        "residual_profile_distance": none_if_nan(pair["residual_profile_distance"]),
        "group_order_randomized": True,
        "anchor_set": "condition_local",
        "n_anchor_images": int(len(anchor_indices)),
    }
    card = card_base(
        cfg,
        condition,
        "real_residual",
        source_index,
        q,
        geometry["c_local"][q],
        asset_map[source_index],
        hidden,
    )
    card["groups"] = groups
    card["brain_pair_summary"] = summarize_card_brain_pairs(card)
    return card


def random_control_card(
    cfg: Config,
    condition: str,
    q: int,
    source_index: int,
    anchor_indices: np.ndarray,
    asset_map: dict[int, str],
    geometry: dict[str, Any],
    brain_summary: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    valid = [int(i) for i in anchor_indices if int(i) != int(source_index)]
    groups = []
    for group_idx in (1, 2):
        chosen = rng.choice(valid, size=min(cfg.n_neighbors, len(valid)), replace=False)
        anchors = [
            make_anchor_entry(
                condition=condition,
                anchor_source_index=int(anchor_source_index),
                anchor_rank=rank,
                asset=asset_map[int(anchor_source_index)],
                brain_pair=brain_pair_info(
                    cfg, brain_summary, source_index, int(anchor_source_index)
                ),
            )
            for rank, anchor_source_index in enumerate(chosen, start=1)
        ]
        groups.append(
            {
                "display_name": f"Group {group_idx}",
                "group_index": group_idx,
                "hidden_model": None,
                "hidden_model_index": None,
                "anchors": anchors,
            }
        )
    hidden = {
        "control": "random_anchor_groups",
        "anchor_set": "condition_local",
        "n_anchor_images": int(len(anchor_indices)),
    }
    card = card_base(
        cfg,
        condition,
        "random_anchor_control",
        source_index,
        q,
        geometry["c_local"][q],
        asset_map[source_index],
        hidden,
    )
    card["groups"] = groups
    card["brain_pair_summary"] = summarize_card_brain_pairs(card)
    return card


def consensus_control_card(
    cfg: Config,
    condition: str,
    q: int,
    source_index: int,
    anchor_indices: np.ndarray,
    asset_map: dict[int, str],
    geometry: dict[str, Any],
    brain_summary: dict[str, Any],
) -> dict[str, Any]:
    top = top_indices(geometry["consensus_profile"][q], cfg.n_neighbors * 2)
    first = top[: cfg.n_neighbors]
    second = top[cfg.n_neighbors : cfg.n_neighbors * 2]
    groups = []
    for group_idx, local_positions in enumerate((first, second), start=1):
        anchors = []
        for rank, local_anchor_pos in enumerate(local_positions, start=1):
            anchor_source_index = int(anchor_indices[local_anchor_pos])
            anchors.append(
                make_anchor_entry(
                    condition=condition,
                    anchor_source_index=anchor_source_index,
                    anchor_rank=rank,
                    asset=asset_map[anchor_source_index],
                    consensus_similarity=geometry["consensus_profile"][q, local_anchor_pos],
                    brain_pair=brain_pair_info(
                        cfg, brain_summary, source_index, anchor_source_index
                    ),
                )
            )
        groups.append(
            {
                "display_name": f"Group {group_idx}",
                "group_index": group_idx,
                "hidden_model": None,
                "hidden_model_index": None,
                "anchors": anchors,
            }
        )
    hidden = {
        "control": "consensus_neighborhood_split",
        "anchor_set": "condition_local",
        "n_anchor_images": int(len(anchor_indices)),
    }
    card = card_base(
        cfg,
        condition,
        "consensus_control",
        source_index,
        q,
        geometry["c_local"][q],
        asset_map[source_index],
        hidden,
    )
    card["groups"] = groups
    card["brain_pair_summary"] = summarize_card_brain_pairs(card)
    return card


def build_condition_cards(
    cfg: Config, condition: str, rng: np.random.Generator
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_files = image_files_for_condition(cfg, condition)
    models = load_payload_models(cfg, condition)
    features = load_condition_features(cfg, condition, models)
    n_images = len(image_files)
    for model, array in features.items():
        if array.shape[0] != n_images:
            raise RuntimeError(
                f"{condition}/{model} feature rows ({array.shape[0]}) "
                f"do not match image count ({n_images})."
            )
    anchor_indices = np.arange(n_images, dtype=np.int64)
    all_query_indices = np.arange(n_images, dtype=np.int64)
    geometry_all = compute_residual_geometry(
        cfg, features, all_query_indices, anchor_indices
    )
    brain_summary = load_brain_pair_summary(cfg, condition, n_images)
    query_indices = choose_query_indices(
        cfg, condition, n_images, geometry_all["c_local"], rng
    )
    query_pos_lookup = {int(source_idx): idx for idx, source_idx in enumerate(all_query_indices)}
    asset_map = prepare_assets(cfg, condition, image_files)
    cards = []
    for source_index in query_indices:
        q = query_pos_lookup[int(source_index)]
        cards.append(
            residual_card(
                cfg,
                condition,
                q,
                int(source_index),
                geometry_all["models"],
                anchor_indices,
                asset_map,
                geometry_all,
                brain_summary,
                rng,
            )
        )
        if cfg.include_random_controls:
            cards.append(
                random_control_card(
                    cfg,
                    condition,
                    q,
                    int(source_index),
                    anchor_indices,
                    asset_map,
                    geometry_all,
                    brain_summary,
                    rng,
                )
            )
        if cfg.include_consensus_controls:
            cards.append(
                consensus_control_card(
                    cfg,
                    condition,
                    q,
                    int(source_index),
                    anchor_indices,
                    asset_map,
                    geometry_all,
                    brain_summary,
                )
            )
    summary = {
        "condition": condition,
        "n_images": n_images,
        "n_query_images": int(len(query_indices)),
        "n_cards": int(len(cards)),
        "models": geometry_all["models"],
        "mean_c_local": float(np.nanmean(geometry_all["c_local"])),
        "median_c_local": float(np.nanmedian(geometry_all["c_local"])),
        "max_c_local": float(np.nanmax(geometry_all["c_local"])),
        "brain_pair_bins_available": bool(brain_summary.get("available")),
        "brain_pair_subjects": brain_summary.get("subjects", []),
    }
    return cards, summary


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    cfg = parse_args()
    data_dir = cfg.app_root / "data"
    static_dir = cfg.app_root / "static"
    data_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    cards: list[dict[str, Any]] = []
    summaries = []
    for condition in cfg.conditions:
        condition_cards, summary = build_condition_cards(cfg, condition, rng)
        cards.extend(condition_cards)
        summaries.append(summary)
        print(
            f"{condition}: {summary['n_cards']} cards, "
            f"{summary['n_query_images']} query images, "
            f"{len(summary['models'])} models",
            flush=True,
        )
    rng.shuffle(cards)
    manifest = {
        "schema_version": "contrastive_residual_neighbor_cards_v1",
        "cue_categories": CUE_CATEGORIES,
        "contrast_options": CONTRAST_OPTIONS,
        "cards": cards,
        "metadata": {
            "config": {
                **asdict(cfg),
                "app_root": str(cfg.app_root),
                "selected_root": str(cfg.selected_root),
                "feature_cache": str(cfg.feature_cache),
                "image_root": str(cfg.image_root) if cfg.image_root else None,
                "brain_data": str(cfg.brain_data),
                "conditions": list(cfg.conditions),
            },
            "condition_summaries": summaries,
            "brain_pair_bins": {
                "similar": f"mean_brain_z <= {-cfg.brain_bin_threshold:g}",
                "neutral": (
                    f"{-cfg.brain_bin_threshold:g} < mean_brain_z "
                    f"< {cfg.brain_bin_threshold:g}"
                ),
                "dissimilar": f"mean_brain_z >= {cfg.brain_bin_threshold:g}",
                "subject_consistent": f"sd_brain_z < {cfg.brain_consistency_sd:g}",
            },
            "residual_definition": (
                "For each condition, each model's similarity profile from the "
                "central image to the condition-local anchor set is rank/z "
                "normalized, averaged across models to form a consensus profile, "
                "and residualized by subtracting that consensus."
            ),
            "annotation_note": (
                "The app hides model identities from annotators; hidden_model "
                "metadata is retained in cards.json and annotation exports."
            ),
        },
    }
    write_json(data_dir / "cards.json", manifest)
    if not (data_dir / "annotations.jsonl").exists():
        (data_dir / "annotations.jsonl").write_text("", encoding="utf-8")
    print(f"Wrote {len(cards)} cards to {data_dir / 'cards.json'}")


if __name__ == "__main__":
    main()
