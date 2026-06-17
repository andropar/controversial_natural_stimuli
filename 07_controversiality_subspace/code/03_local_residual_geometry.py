#!/usr/bin/env python
"""Local residual geometry analysis for frozen selected stimuli.

For each query image, every model gets a similarity profile to a shared anchor
set. Profiles are rank- or z-normalized within model/query, then decomposed into
consensus plus model residuals. High residual energy means the query image has a
model-dependent local neighborhood.

This script uses cached feature arrays only. It does not load image models or
touch the GPU.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTED_PKL = (
    REPO_ROOT
    / "00_stimulus_selection/results/selected_stimuli/all_models/selected_stimuli_data.pkl"
)
DEFAULT_RANDOM_CACHE = REPO_ROOT / "shared/cache_or_heavy/natural_pool_subset_100k_seed42"
DEFAULT_LOCAL_SAMPLE = Path(
    "/data/home_roth/_stachelschwein/rsa_based_selection/data/LAION_natural_sample"
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1]

LOCAL_FEATURE_FILES = {
    "vissl_resnet50_supervised": "vissl_resnet50_supervised_layer-1.npy",
    "vissl_resnet50_mocov2": "vissl_resnet50_mocov2_layer-1.npy",
    "vicreg_resnet50": "vicreg_resnet50_layer-1.npy",
    "slip_vit_l_slip": "slip_vit_l_slip_layer-1.npy",
    "slip_vit_l_simclr": "slip_vit_l_simclr_layer-1.npy",
    "dinov2_vitl14": "dinov2_vitl14_layer-1.npy",
    "openclip_vit_so400m_14_siglip_webli": "openclip_vit_so400m_14_siglip_webli_layer-2.npy",
    "torchvision_resnet50_imagenet1k_v1": "torchvision_resnet50_imagenet1k_v1_layer-2.npy",
    "torchvision_alexnet_imagenet1k_v1": "torchvision_alexnet_imagenet1k_v1_layer-2.npy",
    "robustness_imagenet_l2_eps3": "robustness_imagenet_l2_eps3_layer-2.npy",
}

DEFAULT_VISUAL_MODELS = (
    "torchvision_resnet50_imagenet1k_v1",
    "dinov2_vitl14",
    "slip_vit_l_slip",
    "openclip_vit_so400m_14_siglip_webli",
    "vicreg_resnet50",
    "robustness_imagenet_l2_eps3",
)


@dataclass(frozen=True)
class Config:
    selected_pkl: Path
    random_cache: Path
    local_sample: Path
    output_root: Path
    seed: int
    anchor_size: int
    n_random: int
    profile_normalization: str
    top_selected_panels: int
    nearest_per_model: int
    residual_anchors_per_side: int
    visual_models: tuple[str, ...]


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-pkl", type=Path, default=DEFAULT_SELECTED_PKL)
    parser.add_argument("--random-cache", type=Path, default=DEFAULT_RANDOM_CACHE)
    parser.add_argument("--local-sample", type=Path, default=DEFAULT_LOCAL_SAMPLE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anchor-size", type=int, default=1000)
    parser.add_argument("--n-random", type=int, default=1000)
    parser.add_argument(
        "--profile-normalization",
        choices=["rankz", "zscore"],
        default="rankz",
    )
    parser.add_argument("--top-selected-panels", type=int, default=3)
    parser.add_argument("--nearest-per-model", type=int, default=5)
    parser.add_argument("--residual-anchors-per-side", type=int, default=3)
    parser.add_argument(
        "--visual-models",
        type=str,
        default=",".join(DEFAULT_VISUAL_MODELS),
    )
    args = parser.parse_args()
    values = vars(args)
    values["visual_models"] = tuple(
        name.strip() for name in values["visual_models"].split(",") if name.strip()
    )
    return Config(**values)


def load_selected_payload(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


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
    raise ValueError(f"Unknown normalization method: {method}")


def model_cache_path(random_cache: Path, model_name: str) -> Path:
    return random_cache / f"{model_name}.npz"


def load_cache_subset(
    cache_file: Path,
    anchor_rows: np.ndarray,
    random_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache_file) as data:
        features = data["features"]
        anchors = np.asarray(features[anchor_rows], dtype=np.float32)
        random_queries = np.asarray(features[random_rows], dtype=np.float32)
    return anchors, random_queries


def compute_local_residual_geometry(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    payload = load_selected_payload(cfg.selected_pkl)
    selected_features = payload["selected_features_raw"]
    selected_indices = np.asarray(payload["selected_global_indices"], dtype=np.int64)
    selected_records = payload["selected_image_records"]
    sampled_global = np.load(cfg.random_cache / "_sampled_indices.npy")

    cache_models = {
        p.stem for p in cfg.random_cache.glob("*.npz") if p.name != "manifest.npz"
    }
    model_names = [
        name
        for name in selected_features
        if name in cache_models and model_cache_path(cfg.random_cache, name).exists()
    ]
    if not model_names:
        raise RuntimeError("No overlapping models between selected payload and random cache.")

    all_rows = np.arange(len(sampled_global), dtype=np.int64)
    anchor_rows = rng.choice(all_rows, size=min(cfg.anchor_size, len(all_rows)), replace=False)
    remaining = np.setdiff1d(all_rows, anchor_rows, assume_unique=False)
    random_rows = rng.choice(remaining, size=min(cfg.n_random, len(remaining)), replace=False)

    n_selected = len(selected_indices)
    n_random = len(random_rows)
    n_queries = n_selected + n_random
    n_models = len(model_names)
    n_anchors = len(anchor_rows)
    profiles = np.empty((n_queries, n_models, n_anchors), dtype=np.float32)

    query_rows = []
    for pos, (global_idx, record) in enumerate(zip(selected_indices, selected_records)):
        query_rows.append(
            {
                "query_id": pos,
                "condition": "selected",
                "selected_position": pos,
                "global_index": int(global_idx),
                "random_cache_row": np.nan,
                "image_name": record.get("image_name", ""),
            }
        )
    for i, row in enumerate(random_rows, start=n_selected):
        query_rows.append(
            {
                "query_id": i,
                "condition": "random",
                "selected_position": np.nan,
                "global_index": int(sampled_global[row]),
                "random_cache_row": int(row),
                "image_name": "",
            }
        )
    query_frame = pd.DataFrame(query_rows)

    for model_pos, model_name in enumerate(model_names):
        selected_model = np.asarray(selected_features[model_name], dtype=np.float32)
        anchors, random_model = load_cache_subset(
            model_cache_path(cfg.random_cache, model_name),
            anchor_rows,
            random_rows,
        )
        if selected_model.shape[1] != anchors.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch for {model_name}: "
                f"selected={selected_model.shape[1]}, anchors={anchors.shape[1]}"
            )
        query_model = np.vstack([selected_model, random_model])
        sim = normalize_rows(query_model) @ normalize_rows(anchors).T
        profiles[:, model_pos, :] = normalize_profiles(sim, cfg.profile_normalization)
        print(
            f"processed model {model_pos + 1}/{n_models}: {model_name}",
            flush=True,
        )

    consensus = profiles.mean(axis=1, keepdims=True)
    residuals = profiles - consensus
    scores = (residuals * residuals).mean(axis=(1, 2))
    model_scores = (residuals * residuals).mean(axis=2)

    score_frame = query_frame.copy()
    score_frame["local_residual_geometry"] = scores

    model_rows = []
    for query_id in range(n_queries):
        for model_pos, model_name in enumerate(model_names):
            model_rows.append(
                {
                    "query_id": query_id,
                    "condition": score_frame.loc[query_id, "condition"],
                    "global_index": int(score_frame.loc[query_id, "global_index"]),
                    "model": model_name,
                    "model_residual_energy": float(model_scores[query_id, model_pos]),
                }
            )
    model_frame = pd.DataFrame(model_rows)

    anchor_frame = pd.DataFrame(
        {
            "anchor_position": np.arange(n_anchors, dtype=np.int64),
            "random_cache_row": anchor_rows,
            "global_index": sampled_global[anchor_rows],
        }
    )

    return score_frame, model_frame, anchor_frame


def summarize_scores(score_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, grp in score_frame.groupby("condition", observed=True):
        rows.append(
            {
                "condition": condition,
                "n": len(grp),
                "mean": grp["local_residual_geometry"].mean(),
                "std": grp["local_residual_geometry"].std(ddof=1),
                "median": grp["local_residual_geometry"].median(),
                "q25": grp["local_residual_geometry"].quantile(0.25),
                "q75": grp["local_residual_geometry"].quantile(0.75),
            }
        )
    selected = score_frame.loc[
        score_frame["condition"] == "selected", "local_residual_geometry"
    ].to_numpy()
    random = score_frame.loc[
        score_frame["condition"] == "random", "local_residual_geometry"
    ].to_numpy()
    if len(selected) and len(random):
        rows.append(
            {
                "condition": "selected_vs_random",
                "n": len(selected),
                "mean": selected.mean() - random.mean(),
                "std": np.nan,
                "median": selected.mean(),
                "q25": random.mean(),
                "q75": float((selected[:, None] > random[None, :]).mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_distribution(score_frame: pd.DataFrame, fig_dir: Path) -> None:
    selected = score_frame.loc[
        score_frame["condition"] == "selected", "local_residual_geometry"
    ].to_numpy()
    random = score_frame.loc[
        score_frame["condition"] == "random", "local_residual_geometry"
    ].to_numpy()
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    parts = ax.violinplot([random, selected], positions=[0, 1], widths=0.75, showextrema=False)
    for body, color in zip(parts["bodies"], ["#9E9E9E", "#4C78A8"]):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.45)
    rng = np.random.default_rng(0)
    ax.scatter(
        rng.normal(0, 0.045, size=len(random)),
        random,
        s=7,
        alpha=0.25,
        color="#777777",
        edgecolor="none",
    )
    ax.scatter(
        rng.normal(1, 0.045, size=len(selected)),
        selected,
        s=12,
        alpha=0.65,
        color="#4C78A8",
        edgecolor="none",
    )
    ax.plot([-0.2, 0.2], [np.median(random), np.median(random)], color="black")
    ax.plot([0.8, 1.2], [np.median(selected), np.median(selected)], color="black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["random", "selected"])
    ax.set_ylabel("local residual geometry")
    ax.set_title("Model-dependent local neighborhoods")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "local_residual_geometry_distribution.pdf")
    plt.close(fig)


def plot_model_residuals(model_frame: pd.DataFrame, fig_dir: Path) -> None:
    summary = (
        model_frame.groupby(["condition", "model"], observed=True)["model_residual_energy"]
        .mean()
        .reset_index()
    )
    pivot = summary.pivot(index="model", columns="condition", values="model_residual_energy")
    pivot = pivot.sort_values("selected", ascending=False)
    fig, ax = plt.subplots(figsize=(7.8, max(4, 0.23 * len(pivot))))
    y = np.arange(len(pivot))
    ax.barh(y + 0.18, pivot.get("selected", np.nan), height=0.35, label="selected", color="#4C78A8")
    ax.barh(y - 0.18, pivot.get("random", np.nan), height=0.35, label="random", color="#9E9E9E")
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("mean residual energy")
    ax.set_title("Which models drive local residual geometry?")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "local_residual_geometry_model_residuals.pdf")
    plt.close(fig)


def load_square_image(path: Path, size: int = 128) -> Image.Image:
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        img = Image.new("RGB", (size, size), color=(230, 230, 230))
    img.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), color=(255, 255, 255))
    offset = ((size - img.width) // 2, (size - img.height) // 2)
    canvas.paste(img, offset)
    return canvas


def selected_image_path(cfg: Config, global_index: int) -> Path:
    selected_dir = cfg.selected_pkl.parent / "validated_images"
    return selected_dir / f"{global_index}.png"


def local_image_path(cfg: Config, image_index: int) -> Path:
    return cfg.local_sample / "images" / f"{image_index}.jpg"


def visual_model_files(cfg: Config, selected_features: dict[str, np.ndarray]) -> list[tuple[str, Path]]:
    out = []
    feature_dir = cfg.local_sample / "features"
    for model_name in cfg.visual_models:
        if model_name not in selected_features or model_name not in LOCAL_FEATURE_FILES:
            continue
        path = feature_dir / LOCAL_FEATURE_FILES[model_name]
        if not path.exists():
            continue
        local = np.load(path, mmap_mode="r")
        if local.shape[1] == selected_features[model_name].shape[1]:
            out.append((model_name, path))
    return out


def plot_nearest_anchor_panels(
    cfg: Config,
    score_frame: pd.DataFrame,
    selected_features: dict[str, np.ndarray],
    fig_dir: Path,
) -> None:
    visual_models = visual_model_files(cfg, selected_features)
    if not visual_models:
        return
    top = (
        score_frame[score_frame["condition"] == "selected"]
        .sort_values("local_residual_geometry", ascending=False)
        .head(cfg.top_selected_panels)
    )
    for _, row in top.iterrows():
        selected_pos = int(row["selected_position"])
        global_index = int(row["global_index"])
        n_cols = cfg.nearest_per_model + 1
        n_rows = len(visual_models)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.55 * n_cols, 1.65 * n_rows))
        axes = np.asarray(axes).reshape(n_rows, n_cols)
        target = load_square_image(selected_image_path(cfg, global_index))
        for r, (model_name, feature_path) in enumerate(visual_models):
            local = np.load(feature_path, mmap_mode="r")
            local_norm = normalize_rows(np.asarray(local, dtype=np.float32))
            query = normalize_rows(selected_features[model_name][selected_pos : selected_pos + 1])
            sim = (query @ local_norm.T).ravel()
            nearest = np.argsort(sim)[::-1][: cfg.nearest_per_model]
            axes[r, 0].imshow(target)
            axes[r, 0].set_title(f"{model_name}\nquery", fontsize=6)
            axes[r, 0].axis("off")
            for c, image_idx in enumerate(nearest, start=1):
                axes[r, c].imshow(load_square_image(local_image_path(cfg, int(image_idx))))
                axes[r, c].set_title(f"{int(image_idx)}\n{sim[image_idx]:.2f}", fontsize=6)
                axes[r, c].axis("off")
        fig.suptitle(
            f"Nearest local anchors by model: selected {global_index} "
            f"(C={row['local_residual_geometry']:.3f})",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(fig_dir / f"local_neighbors_selected_{global_index}.pdf")
        plt.close(fig)


def plot_residual_anchor_panel(
    cfg: Config,
    score_frame: pd.DataFrame,
    selected_features: dict[str, np.ndarray],
    fig_dir: Path,
) -> None:
    visual_models = visual_model_files(cfg, selected_features)
    if not visual_models:
        return
    row = (
        score_frame[score_frame["condition"] == "selected"]
        .sort_values("local_residual_geometry", ascending=False)
        .iloc[0]
    )
    selected_pos = int(row["selected_position"])
    global_index = int(row["global_index"])
    profiles = []
    model_names = []
    sims_by_model = {}
    for model_name, feature_path in visual_models:
        local = np.load(feature_path, mmap_mode="r")
        local_norm = normalize_rows(np.asarray(local, dtype=np.float32))
        query = normalize_rows(selected_features[model_name][selected_pos : selected_pos + 1])
        sim = (query @ local_norm.T).astype(np.float32)
        profiles.append(normalize_profiles(sim, cfg.profile_normalization).ravel())
        sims_by_model[model_name] = sim.ravel()
        model_names.append(model_name)
    prof = np.stack(profiles, axis=0)
    residual = prof - prof.mean(axis=0, keepdims=True)
    n_each = cfg.residual_anchors_per_side
    n_cols = 1 + 2 * n_each
    n_rows = len(model_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.55 * n_cols, 1.65 * n_rows))
    axes = np.asarray(axes).reshape(n_rows, n_cols)
    target = load_square_image(selected_image_path(cfg, global_index))
    for r, model_name in enumerate(model_names):
        axes[r, 0].imshow(target)
        axes[r, 0].set_title(f"{model_name}\nquery", fontsize=6)
        axes[r, 0].axis("off")
        pos_idx = np.argsort(residual[r])[::-1][:n_each]
        neg_idx = np.argsort(residual[r])[:n_each]
        for c, image_idx in enumerate(pos_idx, start=1):
            axes[r, c].imshow(load_square_image(local_image_path(cfg, int(image_idx))))
            axes[r, c].set_title(f"closer\n{int(image_idx)}", fontsize=6)
            axes[r, c].axis("off")
        for c, image_idx in enumerate(neg_idx, start=1 + n_each):
            axes[r, c].imshow(load_square_image(local_image_path(cfg, int(image_idx))))
            axes[r, c].set_title(f"farther\n{int(image_idx)}", fontsize=6)
            axes[r, c].axis("off")
    fig.suptitle(
        f"Residual anchors for selected {global_index}: closer/farther than consensus",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / f"local_residual_anchor_panel_selected_{global_index}.pdf")
    plt.close(fig)


def main() -> None:
    cfg = parse_args()
    results_dir = cfg.output_root / "results"
    fig_dir = cfg.output_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    score_frame, model_frame, anchor_frame = compute_local_residual_geometry(cfg)
    score_frame.to_csv(results_dir / "local_residual_geometry_scores.csv", index=False)
    model_frame.to_csv(results_dir / "local_residual_geometry_model_residuals.csv", index=False)
    anchor_frame.to_csv(results_dir / "local_residual_geometry_anchors.csv", index=False)
    summary = summarize_scores(score_frame)
    summary.to_csv(results_dir / "local_residual_geometry_summary.csv", index=False)

    metadata = {
        "config": {
            **asdict(cfg),
            "selected_pkl": str(cfg.selected_pkl),
            "random_cache": str(cfg.random_cache),
            "local_sample": str(cfg.local_sample),
            "output_root": str(cfg.output_root),
            "visual_models": list(cfg.visual_models),
        },
        "score_definition": "mean_m,anchor (profile_m - model_mean_profile)^2",
        "profile_normalization": cfg.profile_normalization,
    }
    with (results_dir / "local_residual_geometry_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    plot_distribution(score_frame, fig_dir)
    plot_model_residuals(model_frame, fig_dir)

    payload = load_selected_payload(cfg.selected_pkl)
    plot_nearest_anchor_panels(cfg, score_frame, payload["selected_features_raw"], fig_dir)
    plot_residual_anchor_panel(cfg, score_frame, payload["selected_features_raw"], fig_dir)

    print("local residual geometry summary")
    print(summary.to_string(index=False))
    print("Wrote results to", results_dir)
    print("Wrote figures to", fig_dir)


if __name__ == "__main__":
    main()

