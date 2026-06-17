#!/usr/bin/env python
"""Contrastive residual-neighbor panels for high local-residual images.

For each high-C_local selected image, this script finds anchors that are
unusually similar for each model relative to the cross-model consensus. These
are positive residual neighbors, not ordinary nearest neighbors.

The script uses cached local features and local images only. It does not load
image models or touch the GPU.
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
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTED_PKL = (
    REPO_ROOT
    / "00_stimulus_selection/results/selected_stimuli/all_models/selected_stimuli_data.pkl"
)
DEFAULT_LOCAL_SAMPLE = Path(
    "/data/home_roth/_stachelschwein/rsa_based_selection/data/LAION_natural_sample"
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORE_CSV = DEFAULT_OUTPUT_ROOT / "results/local_residual_geometry_scores.csv"

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
    "robustness_imagenet_l2_eps3",
)


@dataclass(frozen=True)
class Config:
    selected_pkl: Path
    local_sample: Path
    output_root: Path
    score_csv: Path
    top_n: int
    n_neighbors: int
    max_models: int
    profile_normalization: str
    visual_models: tuple[str, ...]
    image_size: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-pkl", type=Path, default=DEFAULT_SELECTED_PKL)
    parser.add_argument("--local-sample", type=Path, default=DEFAULT_LOCAL_SAMPLE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--score-csv", type=Path, default=DEFAULT_SCORE_CSV)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--max-models", type=int, default=5)
    parser.add_argument(
        "--profile-normalization",
        choices=["rankz", "zscore"],
        default="rankz",
    )
    parser.add_argument(
        "--visual-models",
        type=str,
        default=",".join(DEFAULT_VISUAL_MODELS),
    )
    parser.add_argument("--image-size", type=int, default=112)
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
    raise ValueError(f"Unknown profile normalization: {method}")


def selected_image_path(cfg: Config, global_index: int) -> Path:
    return cfg.selected_pkl.parent / "validated_images" / f"{global_index}.png"


def local_image_path(cfg: Config, image_index: int) -> Path:
    return cfg.local_sample / "images" / f"{image_index}.jpg"


def load_square_image(path: Path, size: int) -> Image.Image:
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        img = Image.new("RGB", (size, size), color=(235, 235, 235))
        draw = ImageDraw.Draw(img)
        draw.text((8, size // 2 - 8), "missing", fill=(80, 80, 80))
    img.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), color=(255, 255, 255))
    offset = ((size - img.width) // 2, (size - img.height) // 2)
    canvas.paste(img, offset)
    return canvas


def matched_visual_models(cfg: Config, selected_features: dict[str, np.ndarray]) -> list[tuple[str, Path]]:
    feature_dir = cfg.local_sample / "features"
    out = []
    for model_name in cfg.visual_models:
        if model_name not in selected_features:
            continue
        filename = LOCAL_FEATURE_FILES.get(model_name)
        if not filename:
            continue
        path = feature_dir / filename
        if not path.exists():
            continue
        local = np.load(path, mmap_mode="r")
        if local.shape[1] == selected_features[model_name].shape[1]:
            out.append((model_name, path))
    if len(out) < 2:
        raise RuntimeError("Need at least two visual models with matching local features.")
    return out


def residual_profile_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
    if denom <= 1e-12:
        return np.nan
    return float(np.dot(a, b) / denom)


def compute_profiles(
    cfg: Config,
    payload: dict,
    top_selected: pd.DataFrame,
    visual_models: list[tuple[str, Path]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    positions = top_selected["selected_position"].to_numpy(dtype=np.int64)
    selected_features = payload["selected_features_raw"]
    profiles = []
    raw_sims = []
    model_names = []
    for model_pos, (model_name, feature_path) in enumerate(visual_models):
        local = np.load(feature_path, mmap_mode="r")
        local_norm = normalize_rows(np.asarray(local, dtype=np.float32))
        query = normalize_rows(np.asarray(selected_features[model_name][positions], dtype=np.float32))
        sim = query @ local_norm.T
        profiles.append(normalize_profiles(sim, cfg.profile_normalization))
        raw_sims.append(sim.astype(np.float32, copy=False))
        model_names.append(model_name)
        print(
            f"computed local profiles for model {model_pos + 1}/{len(visual_models)}: {model_name}",
            flush=True,
        )
    return np.stack(profiles, axis=1), np.stack(raw_sims, axis=1), model_names


def build_neighbor_tables(
    cfg: Config,
    top_selected: pd.DataFrame,
    profiles: np.ndarray,
    raw_sims: np.ndarray,
    model_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    consensus = profiles.mean(axis=1, keepdims=True)
    residuals = profiles - consensus
    model_energy = (residuals * residuals).mean(axis=2)

    neighbor_rows = []
    model_rows = []
    pair_rows = []
    for q, (_, image_row) in enumerate(top_selected.iterrows()):
        c_local = float(image_row["local_residual_geometry"])
        selected_position = int(image_row["selected_position"])
        global_index = int(image_row["global_index"])

        for m, model_name in enumerate(model_names):
            model_rows.append(
                {
                    "selected_position": selected_position,
                    "global_index": global_index,
                    "local_residual_geometry": c_local,
                    "model": model_name,
                    "visual_model_residual_energy": float(model_energy[q, m]),
                    "manual_neighborhood_label": "",
                    "manual_cue_notes": "",
                }
            )

            top_anchor_idx = np.argsort(residuals[q, m])[::-1][: cfg.n_neighbors]
            for rank, anchor_idx in enumerate(top_anchor_idx, start=1):
                neighbor_rows.append(
                    {
                        "selected_position": selected_position,
                        "global_index": global_index,
                        "local_residual_geometry": c_local,
                        "model": model_name,
                        "visual_model_residual_energy": float(model_energy[q, m]),
                        "anchor_rank": rank,
                        "anchor_image_index": int(anchor_idx),
                        "positive_residual": float(residuals[q, m, anchor_idx]),
                        "normalized_similarity_profile": float(profiles[q, m, anchor_idx]),
                        "raw_cosine_similarity": float(raw_sims[q, m, anchor_idx]),
                        "manual_anchor_label": "",
                    }
                )

        for i, model_i in enumerate(model_names):
            for j in range(i + 1, len(model_names)):
                model_j = model_names[j]
                corr = residual_profile_correlation(residuals[q, i], residuals[q, j])
                dist = float(np.sqrt(np.mean((residuals[q, i] - residuals[q, j]) ** 2)))
                pair_rows.append(
                    {
                        "selected_position": selected_position,
                        "global_index": global_index,
                        "local_residual_geometry": c_local,
                        "model_i": model_i,
                        "model_j": model_j,
                        "residual_profile_correlation": corr,
                        "residual_profile_distance": dist,
                        "model_i_residual_energy": float(model_energy[q, i]),
                        "model_j_residual_energy": float(model_energy[q, j]),
                        "manual_contrast_label": "",
                        "manual_contrast_notes": "",
                    }
                )

    neighbors = pd.DataFrame(neighbor_rows)
    model_summary = pd.DataFrame(model_rows)
    pair_summary = pd.DataFrame(pair_rows)
    strongest_pairs = (
        pair_summary.sort_values(
            ["selected_position", "residual_profile_distance"],
            ascending=[True, False],
        )
        .groupby("selected_position", observed=True)
        .head(1)
        .reset_index(drop=True)
    )
    return neighbors, model_summary, strongest_pairs


def models_for_panel(
    model_summary: pd.DataFrame,
    strongest_pair: pd.Series,
    selected_position: int,
    max_models: int,
) -> list[str]:
    rows = model_summary[model_summary["selected_position"] == selected_position]
    ordered = list(
        rows.sort_values("visual_model_residual_energy", ascending=False)["model"]
    )
    first = [str(strongest_pair["model_i"]), str(strongest_pair["model_j"])]
    out = []
    for name in first + ordered:
        if name not in out:
            out.append(name)
        if len(out) >= max_models:
            break
    return out


def plot_one_panel(
    cfg: Config,
    image_row: pd.Series,
    neighbor_frame: pd.DataFrame,
    model_summary: pd.DataFrame,
    strongest_pair: pd.Series,
    output_path: Path | None = None,
    pdf: PdfPages | None = None,
) -> None:
    selected_position = int(image_row["selected_position"])
    global_index = int(image_row["global_index"])
    c_local = float(image_row["local_residual_geometry"])
    panel_models = models_for_panel(
        model_summary, strongest_pair, selected_position, cfg.max_models
    )
    n_rows = len(panel_models)
    n_cols = cfg.n_neighbors + 1
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.35 * n_cols, 1.55 * n_rows + 0.5),
        squeeze=False,
    )
    query_img = load_square_image(selected_image_path(cfg, global_index), cfg.image_size)
    pair_models = {str(strongest_pair["model_i"]), str(strongest_pair["model_j"])}

    for r, model_name in enumerate(panel_models):
        model_row = model_summary[
            (model_summary["selected_position"] == selected_position)
            & (model_summary["model"] == model_name)
        ].iloc[0]
        prefix = "*" if model_name in pair_models else ""
        axes[r, 0].imshow(query_img)
        axes[r, 0].set_title(
            f"{prefix}{model_name}\nenergy={model_row['visual_model_residual_energy']:.3f}",
            fontsize=6,
        )
        axes[r, 0].axis("off")
        anchors = neighbor_frame[
            (neighbor_frame["selected_position"] == selected_position)
            & (neighbor_frame["model"] == model_name)
        ].sort_values("anchor_rank")
        for c, (_, anchor_row) in enumerate(anchors.iterrows(), start=1):
            image_idx = int(anchor_row["anchor_image_index"])
            axes[r, c].imshow(load_square_image(local_image_path(cfg, image_idx), cfg.image_size))
            axes[r, c].set_title(
                f"{image_idx}\nres={anchor_row['positive_residual']:.2f}",
                fontsize=6,
            )
            axes[r, c].axis("off")

    fig.suptitle(
        "Positive residual neighbors: "
        f"selected {global_index}, C_local={c_local:.3f}; "
        f"strongest contrast {strongest_pair['model_i']} vs {strongest_pair['model_j']}",
        fontsize=9,
    )
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path)
    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)


def plot_panels(
    cfg: Config,
    top_selected: pd.DataFrame,
    neighbors: pd.DataFrame,
    model_summary: pd.DataFrame,
    strongest_pairs: pd.DataFrame,
    fig_dir: Path,
) -> None:
    individual_dir = fig_dir / "contrastive_residual_neighbors"
    individual_dir.mkdir(parents=True, exist_ok=True)
    combined_path = fig_dir / "contrastive_residual_neighbors_top20.pdf"
    strongest_by_pos = {
        int(row["selected_position"]): row for _, row in strongest_pairs.iterrows()
    }
    with PdfPages(combined_path) as pdf:
        for _, image_row in top_selected.iterrows():
            selected_position = int(image_row["selected_position"])
            global_index = int(image_row["global_index"])
            out_path = individual_dir / f"contrastive_residual_neighbors_{global_index}.pdf"
            plot_one_panel(
                cfg,
                image_row,
                neighbors,
                model_summary,
                strongest_by_pos[selected_position],
                output_path=out_path,
                pdf=pdf,
            )


def main() -> None:
    cfg = parse_args()
    results_dir = cfg.output_root / "results"
    fig_dir = cfg.output_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    payload = load_selected_payload(cfg.selected_pkl)
    score_frame = pd.read_csv(cfg.score_csv)
    top_selected = (
        score_frame[score_frame["condition"] == "selected"]
        .sort_values("local_residual_geometry", ascending=False)
        .head(cfg.top_n)
        .copy()
    )
    top_selected["selected_position"] = top_selected["selected_position"].astype(int)
    top_selected.to_csv(
        results_dir / "contrastive_residual_top_selected.csv",
        index=False,
    )

    visual_models = matched_visual_models(cfg, payload["selected_features_raw"])
    profiles, raw_sims, model_names = compute_profiles(
        cfg, payload, top_selected, visual_models
    )
    neighbors, model_summary, strongest_pairs = build_neighbor_tables(
        cfg, top_selected, profiles, raw_sims, model_names
    )

    neighbors.to_csv(
        results_dir / "contrastive_residual_neighbors.csv",
        index=False,
    )
    model_summary.to_csv(
        results_dir / "contrastive_residual_model_label_template.csv",
        index=False,
    )
    strongest_pairs.to_csv(
        results_dir / "contrastive_residual_strongest_model_pairs.csv",
        index=False,
    )

    plot_panels(cfg, top_selected, neighbors, model_summary, strongest_pairs, fig_dir)

    metadata = {
        "config": {
            **asdict(cfg),
            "selected_pkl": str(cfg.selected_pkl),
            "local_sample": str(cfg.local_sample),
            "output_root": str(cfg.output_root),
            "score_csv": str(cfg.score_csv),
            "visual_models": list(cfg.visual_models),
        },
        "actual_visual_models": model_names,
        "residual_definition": "rank/z-normalized model profile minus cross-model consensus profile over local 10k anchors",
        "labeling_note": "Manual label columns are intentionally blank; inspect panels first, then annotate cue hypotheses.",
    }
    with (results_dir / "contrastive_residual_neighbors_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print("contrastive residual-neighbor strongest pairs")
    print(
        strongest_pairs[
            [
                "selected_position",
                "global_index",
                "local_residual_geometry",
                "model_i",
                "model_j",
                "residual_profile_distance",
                "residual_profile_correlation",
            ]
        ].head(20).to_string(index=False)
    )
    print("Wrote results to", results_dir)
    print("Wrote figures to", fig_dir)


if __name__ == "__main__":
    main()

