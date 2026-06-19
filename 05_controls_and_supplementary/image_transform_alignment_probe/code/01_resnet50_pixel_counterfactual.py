#!/usr/bin/env python3
"""Quick-and-dirty single-pair ResNet50 counterfactual image probe.

This script keeps one image fixed as an anchor and optimizes the other image in
pixel space so that ResNet50's feature distance to the anchor approaches a
subject-specific brain z-distance target.

The result is intentionally exploratory.  It is a feasibility check for the
counterfactual objective, not a natural-image editing method.
"""

from __future__ import annotations

import argparse
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
from PIL import Image
from scipy.spatial.distance import pdist


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ANALYSIS_DIR / "results"
FIGURES_DIR = ANALYSIS_DIR / "figures"
OPTIMIZED_DIR = ANALYSIS_DIR / "optimized_images"

HELPER_DIR = ROOT / "src"
sys.path.insert(0, str(HELPER_DIR))
from cstims import constants, paths


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
ANCHOR_IDX = 22
TARGET_IDX = 46
SUBJECT = "sub-06"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass(frozen=True)
class Calibration:
    mean: float
    std: float
    cached_original_z: float


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


def total_variation(image_01: torch.Tensor) -> torch.Tensor:
    dx = (image_01[:, :, :, 1:] - image_01[:, :, :, :-1]).abs().mean()
    dy = (image_01[:, :, 1:, :] - image_01[:, :, :-1, :]).abs().mean()
    return dx + dy


def load_brain_target(subject: str, img_i: int, img_j: int) -> float:
    frame = pd.read_csv(PAIR_SUBJECT)
    row = frame[
        (frame["subject"] == subject)
        & (frame["img_i"] == img_i)
        & (frame["img_j"] == img_j)
    ]
    if row.empty:
        raise RuntimeError(f"No brain target for {subject}, pair {img_i}-{img_j}")
    return float(row.iloc[0]["brain_z"])


def load_cached_calibration(model_name: str, img_i: int, img_j: int) -> Calibration:
    with open(paths.selected_stimuli_payload(), "rb") as f:
        payload = pickle.load(f)
    features = np.asarray(payload["selected_features_raw"][model_name])
    values = pdist(features, metric="cosine")
    mu = float(values.mean())
    sd = float(values.std(ddof=0))

    # pdist order is awkward; compute the pair directly for the original z.
    a = features[img_i]
    b = features[img_j]
    raw = 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return Calibration(mean=mu, std=sd, cached_original_z=(raw - mu) / sd)


def build_resnet50(device: torch.device) -> torch.nn.Module:
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet50(weights=weights)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def extract_flatten_features(model: torch.nn.Module, image_01: torch.Tensor) -> torch.Tensor:
    normalized = normalize_tensor(image_01)
    feats = {}

    def hook(_module, _inputs, output):
        feats["value"] = output

    handle = model.avgpool.register_forward_hook(hook)
    try:
        model(normalized)
    finally:
        handle.remove()
    if "value" not in feats:
        raise RuntimeError("avgpool hook did not capture features")
    return torch.flatten(feats["value"], 1)


def feature_z_distance(
    anchor_feat_unit: torch.Tensor,
    candidate_feat: torch.Tensor,
    calibration: Calibration,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate_unit = F.normalize(candidate_feat, dim=1)
    cosine_similarity = (anchor_feat_unit * candidate_unit).sum(dim=1)
    distance = 1.0 - cosine_similarity
    z_distance = (distance - calibration.mean) / calibration.std
    return distance.squeeze(0), z_distance.squeeze(0)


def save_snapshot(path: Path, tensor_01: torch.Tensor) -> None:
    Image.fromarray(tensor_to_uint8_image(tensor_01)).save(path)


def run_optimization(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    anchor_path = IMAGE_DIR / f"image_{args.anchor_idx:04d}.png"
    target_path = IMAGE_DIR / f"image_{args.target_idx:04d}.png"
    anchor_01 = preprocess_pil_to_224_tensor(read_rgb(anchor_path)).to(device)
    target_01 = preprocess_pil_to_224_tensor(read_rgb(target_path)).to(device)

    brain_target_z = load_brain_target(args.subject, args.anchor_idx, args.target_idx)
    calibration = load_cached_calibration(MODEL_NAME, args.anchor_idx, args.target_idx)
    run_suffix = f"_{args.run_label}" if args.run_label else ""

    model = build_resnet50(device)
    with torch.no_grad():
        anchor_feat = extract_flatten_features(model, anchor_01)
        target_feat = extract_flatten_features(model, target_01)
        anchor_feat_unit = F.normalize(anchor_feat, dim=1)
        original_raw_distance, original_live_z = feature_z_distance(
            anchor_feat_unit, target_feat, calibration
        )

    # Logit parameterization keeps pixels in [0, 1] without post-step clipping.
    eps = 1e-4
    init = target_01.detach().clamp(eps, 1 - eps)
    parameter = torch.logit(init).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([parameter], lr=args.lr)

    records = []
    snapshot_steps = sorted(set([0, args.steps // 4, args.steps // 2, args.steps, *args.snapshot_steps]))
    snapshot_paths = {}

    def current_image() -> torch.Tensor:
        return torch.sigmoid(parameter)

    for step in range(args.steps + 1):
        image_01 = current_image()
        candidate_feat = extract_flatten_features(model, image_01)
        raw_distance, z_distance = feature_z_distance(anchor_feat_unit, candidate_feat, calibration)

        alignment_loss = (z_distance - brain_target_z).pow(2)
        identity_loss = F.mse_loss(image_01, target_01)
        tv_loss = total_variation(image_01)
        loss = (
            args.alignment_weight * alignment_loss
            + args.identity_weight * identity_loss
            + args.tv_weight * tv_loss
        )

        if step in snapshot_steps:
            out = OPTIMIZED_DIR / f"resnet50_pair_{args.anchor_idx:04d}_{args.target_idx:04d}_{args.subject}{run_suffix}_step_{step:04d}.png"
            save_snapshot(out, image_01)
            snapshot_paths[step] = out

        records.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "alignment_loss": float(alignment_loss.detach().cpu()),
                "identity_mse": float(identity_loss.detach().cpu()),
                "tv": float(tv_loss.detach().cpu()),
                "raw_model_distance": float(raw_distance.detach().cpu()),
                "z_model_distance": float(z_distance.detach().cpu()),
                "target_brain_z": brain_target_z,
                "cached_original_z": calibration.cached_original_z,
                "live_original_z": float(original_live_z.detach().cpu()),
                "live_original_raw_distance": float(original_raw_distance.detach().cpu()),
            }
        )

        if step == args.steps:
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    trajectory = pd.DataFrame(records)
    trajectory_path = RESULTS_DIR / f"resnet50_pair_{args.anchor_idx:04d}_{args.target_idx:04d}_{args.subject}{run_suffix}_trajectory.csv"
    trajectory.to_csv(trajectory_path, index=False)

    final_path = OPTIMIZED_DIR / f"resnet50_pair_{args.anchor_idx:04d}_{args.target_idx:04d}_{args.subject}{run_suffix}_optimized.png"
    save_snapshot(final_path, current_image())

    metadata = pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "run_label": args.run_label,
                "subject": args.subject,
                "anchor_idx": args.anchor_idx,
                "target_idx": args.target_idx,
                "anchor_path": str(anchor_path),
                "target_path": str(target_path),
                "optimized_path": str(final_path),
                "trajectory_path": str(trajectory_path),
                "brain_target_z": brain_target_z,
                "cached_original_z": calibration.cached_original_z,
                "live_original_z": float(original_live_z.detach().cpu()),
                "final_z": float(trajectory.iloc[-1]["z_model_distance"]),
                "final_alignment_abs_error": abs(float(trajectory.iloc[-1]["z_model_distance"]) - brain_target_z),
                "original_alignment_abs_error": abs(float(original_live_z.detach().cpu()) - brain_target_z),
                "calibration_pair_distance_mean": calibration.mean,
                "calibration_pair_distance_std": calibration.std,
                "steps": args.steps,
                "lr": args.lr,
                "identity_weight": args.identity_weight,
                "tv_weight": args.tv_weight,
                "alignment_weight": args.alignment_weight,
            }
        ]
    )
    metadata_path = RESULTS_DIR / f"resnet50_pair_{args.anchor_idx:04d}_{args.target_idx:04d}_{args.subject}{run_suffix}_summary.csv"
    metadata.to_csv(metadata_path, index=False)

    figure_path = FIGURES_DIR / f"resnet50_pair_{args.anchor_idx:04d}_{args.target_idx:04d}_{args.subject}{run_suffix}_quick_dirty_probe.png"
    make_figure(
        args=args,
        anchor_01=anchor_01,
        target_01=target_01,
        final_01=current_image(),
        trajectory=trajectory,
        snapshot_paths=snapshot_paths,
        figure_path=figure_path,
    )

    print("Wrote:")
    print(f"  {trajectory_path}")
    print(f"  {metadata_path}")
    print(f"  {final_path}")
    print(f"  {figure_path}")
    print("\nSummary:")
    print(metadata.to_string(index=False))


def make_figure(
    args: argparse.Namespace,
    anchor_01: torch.Tensor,
    target_01: torch.Tensor,
    final_01: torch.Tensor,
    trajectory: pd.DataFrame,
    snapshot_paths: dict[int, Path],
    figure_path: Path,
) -> None:
    anchor_img = tensor_to_uint8_image(anchor_01)
    target_img = tensor_to_uint8_image(target_01)
    final_img = tensor_to_uint8_image(final_01)
    delta = np.abs(final_img.astype(np.int16) - target_img.astype(np.int16)).astype(np.uint8)

    fig = plt.figure(figsize=(12.5, 7.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.85], hspace=0.35, wspace=0.18)

    panels = [
        ("anchor A\nimage_0022", anchor_img),
        ("original target B\nimage_0046", target_img),
        ("optimized target B'", final_img),
        ("absolute change\n|B' - B|", delta),
    ]
    for idx, (title, img) in enumerate(panels):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    ax = fig.add_subplot(gs[1, :2])
    ax.plot(trajectory["step"], trajectory["z_model_distance"], color="#1f77b4", lw=2, label="ResNet50 z-distance")
    ax.axhline(trajectory["target_brain_z"].iloc[0], color="#d62728", lw=1.5, ls="--", label=f"{args.subject} brain z")
    ax.axhline(trajectory["live_original_z"].iloc[0], color="#555555", lw=1.0, ls=":", label="original target")
    ax.set_xlabel("optimization step")
    ax.set_ylabel("z-scored pair distance")
    ax.set_title("Counterfactual objective")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[1, 2:])
    ax.plot(trajectory["step"], trajectory["alignment_loss"], color="#9467bd", lw=2, label="alignment loss")
    ax.plot(trajectory["step"], trajectory["identity_mse"], color="#2ca02c", lw=2, label="identity MSE")
    ax.plot(trajectory["step"], trajectory["tv"], color="#ff7f0e", lw=2, label="TV")
    ax.set_xlabel("optimization step")
    ax.set_title("Loss components")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Quick dirty pixel-space ResNet50 counterfactual: floral wall anchor vs stone wall target",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-idx", type=int, default=ANCHOR_IDX)
    parser.add_argument("--target-idx", type=int, default=TARGET_IDX)
    parser.add_argument("--subject", default=SUBJECT)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--identity-weight", type=float, default=8.0)
    parser.add_argument("--tv-weight", type=float, default=0.04)
    parser.add_argument("--alignment-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label", default="", help="Optional suffix for output filenames")
    parser.add_argument("--snapshot-steps", type=int, nargs="*", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    run_optimization(parse_args())
