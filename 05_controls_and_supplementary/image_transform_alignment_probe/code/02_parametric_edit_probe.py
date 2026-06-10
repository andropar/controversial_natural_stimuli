#!/usr/bin/env python3
"""Interpretable parametric edit probe for a single counterfactual pair.

This script evaluates deterministic, low-dimensional edits of the target image
before trying unconstrained pixels or diffusion.  It asks whether simple visual
factors move ResNet50's anchor-target distance toward the brain-derived target.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy.spatial.distance import pdist


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ANALYSIS_DIR / "results"
FIGURES_DIR = ANALYSIS_DIR / "figures"
IMAGE_OUT_DIR = ANALYSIS_DIR / "optimized_images"

HELPER_DIR = ROOT / "shared" / "code" / "paper_helpers"
sys.path.insert(0, str(HELPER_DIR))
import config  # noqa: E402


PAIR_SUBJECT = (
    ROOT
    / "05_controls_and_supplementary"
    / "stimulus_and_pair_diagnostics"
    / "pair_level_brain_placement"
    / "results"
    / "pair_level_brain_placement.csv"
)
IMAGE_DIR = (
    ROOT
    / "00_stimulus_selection"
    / "decision_checks"
    / "selection_evaluation"
    / "results"
    / "all_models"
    / "images"
)

MODEL_NAME = "torchvision_resnet50_imagenet1k_v1"
SUBJECT = "sub-06"
ANCHOR_IDX = 22
TARGET_IDX = 46

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass(frozen=True)
class Calibration:
    raw_mean: float
    raw_std: float
    model_z_distribution: np.ndarray
    cached_original_z: float
    brain_target_z: float
    brain_target_quantile: float
    quantile_target_model_z: float


def read_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def preprocess_pil_to_224_tensor(image: Image.Image) -> torch.Tensor:
    image = TF.resize(image, 256, interpolation=TF.InterpolationMode.BILINEAR)
    image = TF.center_crop(image, [224, 224])
    return TF.to_tensor(image).unsqueeze(0)


def normalize_tensor(image_01: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=image_01.device, dtype=image_01.dtype)
    std = IMAGENET_STD.to(device=image_01.device, dtype=image_01.dtype)
    return (image_01 - mean) / std


def tensor_to_uint8_image(image_01: torch.Tensor) -> np.ndarray:
    arr = image_01.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (arr * 255).round().astype(np.uint8)


def build_resnet50(device: torch.device) -> torch.nn.Module:
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet50(weights=weights)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.no_grad()
def extract_flatten_features(model: torch.nn.Module, image_01: torch.Tensor) -> torch.Tensor:
    feats = {}

    def hook(_module, _inputs, output):
        feats["value"] = output

    handle = model.avgpool.register_forward_hook(hook)
    try:
        model(normalize_tensor(image_01))
    finally:
        handle.remove()
    return torch.flatten(feats["value"], 1)


def load_calibration(subject: str, anchor_idx: int, target_idx: int) -> Calibration:
    with open(config.SELECTION_PAYLOAD, "rb") as f:
        payload = pickle.load(f)

    model_features = np.asarray(payload["selected_features_raw"][MODEL_NAME])
    raw_distances = pdist(model_features, metric="cosine")
    model_z_distribution = (raw_distances - raw_distances.mean()) / raw_distances.std(ddof=0)

    anchor = model_features[anchor_idx]
    target = model_features[target_idx]
    raw_original = 1.0 - float(np.dot(anchor, target) / (np.linalg.norm(anchor) * np.linalg.norm(target)))
    cached_original_z = (raw_original - raw_distances.mean()) / raw_distances.std(ddof=0)

    pair_subject = pd.read_csv(PAIR_SUBJECT)
    selected = pair_subject[
        (pair_subject["subject"] == subject)
        & (pair_subject["img_i"] == anchor_idx)
        & (pair_subject["img_j"] == target_idx)
    ]
    if selected.empty:
        raise RuntimeError(f"No pair-level brain target for {subject}, pair {anchor_idx}-{target_idx}")
    brain_target_z = float(selected.iloc[0]["brain_z"])

    subject_brain_z = pair_subject[pair_subject["subject"] == subject]["brain_z"].dropna().to_numpy()
    brain_quantile = float(np.mean(subject_brain_z <= brain_target_z))
    quantile_target_model_z = float(np.quantile(model_z_distribution, brain_quantile))

    return Calibration(
        raw_mean=float(raw_distances.mean()),
        raw_std=float(raw_distances.std(ddof=0)),
        model_z_distribution=model_z_distribution,
        cached_original_z=float(cached_original_z),
        brain_target_z=brain_target_z,
        brain_target_quantile=brain_quantile,
        quantile_target_model_z=quantile_target_model_z,
    )


def model_z_distance(
    anchor_feat_unit: torch.Tensor,
    candidate_feat: torch.Tensor,
    calibration: Calibration,
) -> tuple[float, float]:
    candidate_unit = F.normalize(candidate_feat, dim=1)
    cosine_similarity = (anchor_feat_unit * candidate_unit).sum(dim=1)
    raw_distance = 1.0 - cosine_similarity
    z_distance = (raw_distance - calibration.raw_mean) / calibration.raw_std
    return float(raw_distance.cpu()), float(z_distance.cpu())


def blend_grayscale(image: Image.Image, alpha: float) -> Image.Image:
    gray = ImageOps.grayscale(image).convert("RGB")
    return Image.blend(image, gray, alpha)


def gamma_adjust(image: Image.Image, gamma: float) -> Image.Image:
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = np.clip(arr, 0, 1) ** gamma
    return Image.fromarray((arr * 255).round().astype(np.uint8), mode="RGB")


def center_zoom(image: Image.Image, zoom: float) -> Image.Image:
    if zoom <= 1:
        return image.copy()
    width, height = image.size
    new_w = max(1, int(round(width / zoom)))
    new_h = max(1, int(round(height / zoom)))
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    cropped = image.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((width, height), Image.Resampling.LANCZOS)


def edge_blend(image: Image.Image, alpha: float) -> Image.Image:
    gray_edges = ImageOps.grayscale(image).filter(ImageFilter.FIND_EDGES)
    edge_rgb = ImageOps.autocontrast(gray_edges).convert("RGB")
    return Image.blend(image, edge_rgb, alpha)


def make_edit_grid(image: Image.Image) -> list[dict]:
    edits = [{"family": "original", "parameter": "none", "value": 0.0, "image": image.copy()}]

    for alpha in [0.25, 0.5, 0.75, 1.0]:
        edits.append({"family": "grayscale_mix", "parameter": "alpha", "value": alpha, "image": blend_grayscale(image, alpha)})
    for radius in [0.75, 1.5, 3.0, 5.0, 8.0]:
        edits.append({"family": "gaussian_blur", "parameter": "radius", "value": radius, "image": image.filter(ImageFilter.GaussianBlur(radius=radius))})
    for factor in [0.0, 0.25, 0.5, 0.75, 1.25, 1.5, 2.0]:
        edits.append({"family": "saturation", "parameter": "factor", "value": factor, "image": ImageEnhance.Color(image).enhance(factor)})
    for factor in [0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0]:
        edits.append({"family": "contrast", "parameter": "factor", "value": factor, "image": ImageEnhance.Contrast(image).enhance(factor)})
    for factor in [0.65, 0.8, 0.9, 1.1, 1.2, 1.35]:
        edits.append({"family": "brightness", "parameter": "factor", "value": factor, "image": ImageEnhance.Brightness(image).enhance(factor)})
    for gamma in [0.6, 0.75, 0.9, 1.1, 1.3, 1.6]:
        edits.append({"family": "gamma", "parameter": "gamma", "value": gamma, "image": gamma_adjust(image, gamma)})
    for zoom in [1.05, 1.1, 1.2, 1.35, 1.5]:
        edits.append({"family": "center_zoom", "parameter": "zoom", "value": zoom, "image": center_zoom(image, zoom)})
    for alpha in [0.1, 0.2, 0.35, 0.5]:
        edits.append({"family": "edge_blend", "parameter": "alpha", "value": alpha, "image": edge_blend(image, alpha)})

    return edits


def pareto_mask(frame: pd.DataFrame, x_col: str, y_col: str) -> np.ndarray:
    values = frame[[x_col, y_col]].to_numpy()
    mask = np.ones(len(values), dtype=bool)
    for idx, point in enumerate(values):
        other = np.delete(values, idx, axis=0)
        dominated = np.any(
            (other[:, 0] <= point[0])
            & (other[:, 1] <= point[1])
            & ((other[:, 0] < point[0]) | (other[:, 1] < point[1]))
        )
        mask[idx] = not dominated
    return mask


def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anchor_path = IMAGE_DIR / f"image_{ANCHOR_IDX:04d}.png"
    target_path = IMAGE_DIR / f"image_{TARGET_IDX:04d}.png"

    anchor_image = read_rgb(anchor_path)
    target_image = read_rgb(target_path)
    calibration = load_calibration(SUBJECT, ANCHOR_IDX, TARGET_IDX)
    model = build_resnet50(device)

    anchor_01 = preprocess_pil_to_224_tensor(anchor_image).to(device)
    target_01 = preprocess_pil_to_224_tensor(target_image).to(device)
    anchor_feat = extract_flatten_features(model, anchor_01)
    anchor_feat_unit = F.normalize(anchor_feat, dim=1)
    target_feat = extract_flatten_features(model, target_01)
    _, live_original_z = model_z_distance(anchor_feat_unit, target_feat, calibration)

    records = []
    edit_images = []
    for edit in make_edit_grid(target_image):
        edited_01 = preprocess_pil_to_224_tensor(edit["image"]).to(device)
        candidate_feat = extract_flatten_features(model, edited_01)
        raw_distance, z_distance = model_z_distance(anchor_feat_unit, candidate_feat, calibration)
        pixel_mse = float(F.mse_loss(edited_01, target_01).cpu())
        pixel_rmse = float(np.sqrt(pixel_mse))
        pixel_l1 = float((edited_01 - target_01).abs().mean().cpu())
        records.append(
            {
                "model": MODEL_NAME,
                "subject": SUBJECT,
                "anchor_idx": ANCHOR_IDX,
                "target_idx": TARGET_IDX,
                "family": edit["family"],
                "parameter": edit["parameter"],
                "value": edit["value"],
                "raw_model_distance": raw_distance,
                "z_model_distance": z_distance,
                "live_original_z": live_original_z,
                "cached_original_z": calibration.cached_original_z,
                "brain_target_z": calibration.brain_target_z,
                "brain_target_quantile": calibration.brain_target_quantile,
                "quantile_target_model_z": calibration.quantile_target_model_z,
                "abs_error_raw_z_target": abs(z_distance - calibration.brain_target_z),
                "abs_error_quantile_target": abs(z_distance - calibration.quantile_target_model_z),
                "raw_z_improvement": abs(live_original_z - calibration.brain_target_z) - abs(z_distance - calibration.brain_target_z),
                "quantile_improvement": abs(live_original_z - calibration.quantile_target_model_z) - abs(z_distance - calibration.quantile_target_model_z),
                "pixel_mse": pixel_mse,
                "pixel_rmse": pixel_rmse,
                "pixel_l1": pixel_l1,
            }
        )
        edit_images.append((edit, edited_01.detach().cpu()))

    frame = pd.DataFrame(records)
    frame["pareto_quantile"] = pareto_mask(frame, "pixel_rmse", "abs_error_quantile_target")
    frame = frame.sort_values(["abs_error_quantile_target", "pixel_rmse"]).reset_index(drop=True)
    out_csv = RESULTS_DIR / f"resnet50_pair_{ANCHOR_IDX:04d}_{TARGET_IDX:04d}_{SUBJECT}_parametric_grid.csv"
    frame.to_csv(out_csv, index=False)

    image_lookup = {
        (edit["family"], edit["parameter"], float(edit["value"])): tensor
        for edit, tensor in edit_images
    }
    selected_for_export = frame.head(8).copy()
    for _, row in selected_for_export.iterrows():
        key = (row["family"], row["parameter"], float(row["value"]))
        img = Image.fromarray(tensor_to_uint8_image(image_lookup[key]))
        safe_value = str(row["value"]).replace(".", "p")
        path = IMAGE_OUT_DIR / (
            f"parametric_pair_{ANCHOR_IDX:04d}_{TARGET_IDX:04d}_{SUBJECT}_"
            f"{row['family']}_{row['parameter']}_{safe_value}.png"
        )
        img.save(path)

    make_figure(
        frame=frame,
        image_lookup=image_lookup,
        anchor_01=anchor_01.detach().cpu(),
        target_01=target_01.detach().cpu(),
        figure_path=FIGURES_DIR / f"resnet50_pair_{ANCHOR_IDX:04d}_{TARGET_IDX:04d}_{SUBJECT}_parametric_probe.png",
    )

    print(f"Wrote {out_csv}")
    print(
        frame[
            [
                "family",
                "parameter",
                "value",
                "z_model_distance",
                "quantile_target_model_z",
                "abs_error_quantile_target",
                "pixel_rmse",
                "quantile_improvement",
                "pareto_quantile",
            ]
        ]
        .head(12)
        .to_string(index=False)
    )


def make_figure(
    frame: pd.DataFrame,
    image_lookup: dict[tuple[str, str, float], torch.Tensor],
    anchor_01: torch.Tensor,
    target_01: torch.Tensor,
    figure_path: Path,
) -> None:
    best = frame.iloc[0]
    pareto = frame[frame["pareto_quantile"]].sort_values("pixel_rmse")
    if len(pareto) > 1:
        low_cost = pareto.iloc[min(1, len(pareto) - 1)]
    else:
        low_cost = best

    best_key = (best["family"], best["parameter"], float(best["value"]))
    low_key = (low_cost["family"], low_cost["parameter"], float(low_cost["value"]))

    best_img = image_lookup[best_key]
    low_img = image_lookup[low_key]
    delta = (best_img - target_01).abs()

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.95], hspace=0.35, wspace=0.18)

    panels = [
        ("anchor", anchor_01),
        ("target", target_01),
        (f"best parametric\n{best['family']}={best['value']}", best_img),
        ("abs change\nbest-target", delta),
    ]
    for idx, (title, tensor) in enumerate(panels):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(tensor_to_uint8_image(tensor))
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    ax = fig.add_subplot(gs[1, :2])
    families = list(dict.fromkeys(frame["family"].tolist()))
    cmap = plt.get_cmap("tab10")
    for fam_idx, family in enumerate(families):
        subset = frame[frame["family"] == family]
        ax.scatter(
            subset["pixel_rmse"],
            subset["abs_error_quantile_target"],
            s=32,
            color=cmap(fam_idx % 10),
            label=family,
            alpha=0.85,
        )
    pareto = frame[frame["pareto_quantile"]].sort_values("pixel_rmse")
    ax.plot(pareto["pixel_rmse"], pareto["abs_error_quantile_target"], color="black", lw=1.2, label="Pareto")
    ax.set_xlabel("pixel RMSE to original target")
    ax.set_ylabel("abs error to quantile-calibrated target")
    ax.set_title("Alignment/cost tradeoff")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = fig.add_subplot(gs[1, 2:])
    ordered = frame.sort_values("z_model_distance")
    colors = ["#999999"] * len(ordered)
    ax.scatter(np.arange(len(ordered)), ordered["z_model_distance"], color=colors, s=24)
    ax.axhline(frame["live_original_z"].iloc[0], color="#555555", lw=1.2, ls=":", label="original model z")
    ax.axhline(frame["brain_target_z"].iloc[0], color="#d62728", lw=1.2, ls="--", label="raw brain z")
    ax.axhline(frame["quantile_target_model_z"].iloc[0], color="#1f77b4", lw=1.2, ls="--", label="quantile model target")
    ax.set_xlabel("parametric edit candidates sorted by model z")
    ax.set_ylabel("ResNet50 z-distance")
    ax.set_title("Distance range reached by simple edits")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Parametric edit probe before unconstrained optimization", fontsize=12, fontweight="bold")
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
