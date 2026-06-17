#!/usr/bin/env python3
"""Low-frequency residual counterfactual for the ResNet50 pair.

This is a constrained middle ground between interpretable parametric edits and
unconstrained pixel optimization.  The target image is modified only by an
upsampled low-resolution residual with bounded amplitude.  This suppresses the
high-frequency texture artifacts that made the direct pixel run too permissive.
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
IMAGE_OUT_DIR = ANALYSIS_DIR / "optimized_images"

HELPER_DIR = ROOT / "src"
sys.path.insert(0, str(HELPER_DIR))
from cstims.paper import config  # noqa: E402


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


def total_variation(image_01: torch.Tensor) -> torch.Tensor:
    dx = (image_01[:, :, :, 1:] - image_01[:, :, :, :-1]).abs().mean()
    dy = (image_01[:, :, 1:, :] - image_01[:, :, :-1, :]).abs().mean()
    return dx + dy


def build_resnet50(device: torch.device) -> torch.nn.Module:
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet50(weights=weights)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


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
        cached_original_z=float(cached_original_z),
        brain_target_z=brain_target_z,
        brain_target_quantile=brain_quantile,
        quantile_target_model_z=quantile_target_model_z,
    )


def z_distance_tensor(
    anchor_feat_unit: torch.Tensor,
    candidate_feat: torch.Tensor,
    calibration: Calibration,
) -> torch.Tensor:
    candidate_unit = F.normalize(candidate_feat, dim=1)
    cosine_similarity = (anchor_feat_unit * candidate_unit).sum(dim=1)
    raw_distance = 1.0 - cosine_similarity
    return ((raw_distance - calibration.raw_mean) / calibration.raw_std).squeeze(0)


def make_counterfactual(
    target_01: torch.Tensor,
    residual_param: torch.Tensor,
    max_delta: float,
    residual_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    low_residual = max_delta * torch.tanh(residual_param)
    if residual_mode == "luminance":
        low_residual = low_residual.expand(-1, 3, -1, -1)
    residual = F.interpolate(
        low_residual,
        size=target_01.shape[-2:],
        mode="bicubic",
        align_corners=False,
    )
    image = (target_01 + residual).clamp(0, 1)
    return image, residual


def run(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    calibration = load_calibration(args.subject, args.anchor_idx, args.target_idx)
    target_z = (
        calibration.quantile_target_model_z
        if args.target_mode == "quantile"
        else calibration.brain_target_z
    )

    anchor_path = IMAGE_DIR / f"image_{args.anchor_idx:04d}.png"
    target_path = IMAGE_DIR / f"image_{args.target_idx:04d}.png"
    anchor_01 = preprocess_pil_to_224_tensor(read_rgb(anchor_path)).to(device)
    target_01 = preprocess_pil_to_224_tensor(read_rgb(target_path)).to(device)

    model = build_resnet50(device)
    with torch.no_grad():
        anchor_feat = extract_flatten_features(model, anchor_01)
        anchor_feat_unit = F.normalize(anchor_feat, dim=1)
        target_feat = extract_flatten_features(model, target_01)
        original_z = float(z_distance_tensor(anchor_feat_unit, target_feat, calibration).cpu())

    residual_param = torch.zeros(
        (
            1,
            1 if args.residual_mode == "luminance" else 3,
            args.residual_size,
            args.residual_size,
        ),
        device=device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([residual_param], lr=args.lr)

    records = []
    best = {"abs_error": float("inf"), "step": 0, "image": target_01.detach().cpu()}
    for step in range(args.steps + 1):
        candidate_01, residual = make_counterfactual(
            target_01,
            residual_param,
            args.max_delta,
            args.residual_mode,
        )
        candidate_feat = extract_flatten_features(model, candidate_01)
        z_model = z_distance_tensor(anchor_feat_unit, candidate_feat, calibration)

        alignment_loss = (z_model - target_z).pow(2)
        pixel_mse = F.mse_loss(candidate_01, target_01)
        residual_l2 = residual.pow(2).mean()
        tv = total_variation(candidate_01)
        loss = (
            alignment_loss
            + args.identity_weight * pixel_mse
            + args.residual_weight * residual_l2
            + args.tv_weight * tv
        )

        abs_error = abs(float(z_model.detach().cpu()) - target_z)
        if abs_error < best["abs_error"]:
            best = {
                "abs_error": abs_error,
                "step": step,
                "image": candidate_01.detach().cpu(),
                "residual": residual.detach().cpu(),
                "z_model": float(z_model.detach().cpu()),
            }

        records.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "alignment_loss": float(alignment_loss.detach().cpu()),
                "pixel_mse": float(pixel_mse.detach().cpu()),
                "pixel_rmse": float(torch.sqrt(pixel_mse).detach().cpu()),
                "residual_l2": float(residual_l2.detach().cpu()),
                "tv": float(tv.detach().cpu()),
                "z_model_distance": float(z_model.detach().cpu()),
                "target_z": target_z,
                "target_mode": args.target_mode,
                "original_z": original_z,
                "brain_target_z": calibration.brain_target_z,
                "quantile_target_model_z": calibration.quantile_target_model_z,
                "abs_error": abs_error,
            }
        )

        if step == args.steps:
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    run_label = args.run_label or (
        f"lowfreq_{args.target_mode}_{args.residual_mode}_r{args.residual_size}_d{str(args.max_delta).replace('.', 'p')}"
    )
    stem = f"resnet50_pair_{args.anchor_idx:04d}_{args.target_idx:04d}_{args.subject}_{run_label}"

    trajectory = pd.DataFrame(records)
    trajectory_path = RESULTS_DIR / f"{stem}_trajectory.csv"
    trajectory.to_csv(trajectory_path, index=False)

    final_image, final_residual = make_counterfactual(
        target_01,
        residual_param,
        args.max_delta,
        args.residual_mode,
    )
    final_path = IMAGE_OUT_DIR / f"{stem}_final.png"
    best_path = IMAGE_OUT_DIR / f"{stem}_best.png"
    Image.fromarray(tensor_to_uint8_image(final_image.detach().cpu())).save(final_path)
    Image.fromarray(tensor_to_uint8_image(best["image"])).save(best_path)

    final_row = trajectory.iloc[-1]
    summary = pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "subject": args.subject,
                "anchor_idx": args.anchor_idx,
                "target_idx": args.target_idx,
                "run_label": run_label,
                "target_mode": args.target_mode,
                "brain_target_z": calibration.brain_target_z,
                "quantile_target_model_z": calibration.quantile_target_model_z,
                "target_z": target_z,
                "cached_original_z": calibration.cached_original_z,
                "live_original_z": original_z,
                "final_z": float(final_row["z_model_distance"]),
                "best_z": best["z_model"],
                "final_abs_error": abs(float(final_row["z_model_distance"]) - target_z),
                "best_abs_error": best["abs_error"],
                "original_abs_error": abs(original_z - target_z),
                "final_pixel_rmse": float(final_row["pixel_rmse"]),
                "best_step": best["step"],
                "steps": args.steps,
                "lr": args.lr,
                "residual_size": args.residual_size,
                "residual_mode": args.residual_mode,
                "max_delta": args.max_delta,
                "identity_weight": args.identity_weight,
                "residual_weight": args.residual_weight,
                "tv_weight": args.tv_weight,
                "trajectory_path": str(trajectory_path),
                "final_path": str(final_path),
                "best_path": str(best_path),
            }
        ]
    )
    summary_path = RESULTS_DIR / f"{stem}_summary.csv"
    summary.to_csv(summary_path, index=False)

    make_figure(
        anchor_01=anchor_01.detach().cpu(),
        target_01=target_01.detach().cpu(),
        best_image=best["image"],
        best_residual=best["residual"],
        trajectory=trajectory,
        summary=summary.iloc[0],
        figure_path=FIGURES_DIR / f"{stem}.png",
    )

    print(f"Wrote {trajectory_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {best_path}")
    print(f"Wrote {FIGURES_DIR / f'{stem}.png'}")
    print(summary.to_string(index=False))


def make_figure(
    anchor_01: torch.Tensor,
    target_01: torch.Tensor,
    best_image: torch.Tensor,
    best_residual: torch.Tensor,
    trajectory: pd.DataFrame,
    summary: pd.Series,
    figure_path: Path,
) -> None:
    residual_vis = (best_residual / (2 * float(summary["max_delta"])) + 0.5).clamp(0, 1)
    delta = (best_image - target_01).abs()

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.95], hspace=0.35, wspace=0.18)

    panels = [
        ("anchor", anchor_01),
        ("target", target_01),
        ("best lowfreq", best_image),
        ("abs change", delta),
    ]
    for idx, (title, tensor) in enumerate(panels):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(tensor_to_uint8_image(tensor))
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    ax = fig.add_subplot(gs[1, :2])
    ax.plot(trajectory["step"], trajectory["z_model_distance"], color="#1f77b4", lw=2, label="ResNet50 z")
    ax.axhline(summary["target_z"], color="#d62728", lw=1.4, ls="--", label="target z")
    ax.axhline(summary["live_original_z"], color="#555555", lw=1.2, ls=":", label="original z")
    ax.set_xlabel("optimization step")
    ax.set_ylabel("z-scored distance")
    ax.set_title("Low-frequency residual objective")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(tensor_to_uint8_image(residual_vis))
    ax.set_title("residual\nscaled display", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 3])
    ax.plot(trajectory["step"], trajectory["pixel_rmse"], color="#2ca02c", lw=2)
    ax.set_xlabel("step")
    ax.set_ylabel("pixel RMSE")
    ax.set_title("edit cost")

    fig.suptitle("Low-frequency residual counterfactual", fontsize=12, fontweight="bold")
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-idx", type=int, default=ANCHOR_IDX)
    parser.add_argument("--target-idx", type=int, default=TARGET_IDX)
    parser.add_argument("--subject", default=SUBJECT)
    parser.add_argument("--target-mode", choices=["quantile", "raw"], default="quantile")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.06)
    parser.add_argument("--residual-size", type=int, default=32)
    parser.add_argument("--residual-mode", choices=["rgb", "luminance"], default="rgb")
    parser.add_argument("--max-delta", type=float, default=0.45)
    parser.add_argument("--identity-weight", type=float, default=2.0)
    parser.add_argument("--residual-weight", type=float, default=0.5)
    parser.add_argument("--tv-weight", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
