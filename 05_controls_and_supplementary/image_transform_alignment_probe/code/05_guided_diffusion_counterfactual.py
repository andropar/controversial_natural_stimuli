#!/usr/bin/env python3
"""Gradient-guided diffusion counterfactual for a ResNet50/brain pair target.

This is a real guided-denoising loop, not candidate reranking.  Starting from
the target image via Stable Diffusion img2img/DDIM, each denoising step:

1. predicts the clean-image latent x0 from the current latent,
2. decodes x0 through the VAE,
3. computes a ResNet50 representational-distance loss to the fixed anchor,
4. backpropagates that loss to the current diffusion latent, and
5. applies a normalized latent guidance step before the DDIM update.

The edited image still has no measured brain response; this is a model
counterfactual constrained by the diffusion prior and image identity penalties.
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
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
from PIL import Image
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
DEFAULT_SD15_SNAPSHOT = Path(
    "/data/home_roth/_stachelschwein/.cache/huggingface/hub/"
    "models--runwayml--stable-diffusion-v1-5/snapshots/"
    "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
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


def preprocess_for_resnet(image_01: torch.Tensor) -> torch.Tensor:
    resized = F.interpolate(image_01, size=(256, 256), mode="bilinear", align_corners=False)
    top = (256 - 224) // 2
    cropped = resized[:, :, top : top + 224, top : top + 224]
    mean = IMAGENET_MEAN.to(device=cropped.device, dtype=cropped.dtype)
    std = IMAGENET_STD.to(device=cropped.device, dtype=cropped.dtype)
    return (cropped - mean) / std


def preprocess_pil_to_resnet_tensor(image: Image.Image) -> torch.Tensor:
    # Keep the raw image here; extract_resnet_features applies the same
    # resize-to-256 and center-crop-to-224 transform to both original and
    # diffusion-decoded images.
    return TF.to_tensor(image).unsqueeze(0)


def resize_pil_square(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize((size, size), Image.Resampling.LANCZOS)


def tensor_to_uint8_image(image_01: torch.Tensor) -> np.ndarray:
    arr = image_01.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return (arr * 255).round().astype(np.uint8)


def total_variation(image_01: torch.Tensor) -> torch.Tensor:
    dx = (image_01[:, :, :, 1:] - image_01[:, :, :, :-1]).abs().mean()
    dy = (image_01[:, :, 1:, :] - image_01[:, :, :-1, :]).abs().mean()
    return dx + dy


def stabilize_latents(latents: torch.Tensor, clip_value: float) -> torch.Tensor:
    latents = torch.nan_to_num(latents, nan=0.0, posinf=clip_value, neginf=-clip_value)
    if clip_value > 0:
        latents = latents.clamp(-clip_value, clip_value)
    return latents


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


def build_resnet50(device: torch.device) -> torch.nn.Module:
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def extract_resnet_features(model: torch.nn.Module, image_01: torch.Tensor) -> torch.Tensor:
    feats = {}

    def hook(_module, _inputs, output):
        feats["value"] = output

    handle = model.avgpool.register_forward_hook(hook)
    try:
        model(preprocess_for_resnet(image_01.float()))
    finally:
        handle.remove()
    return torch.flatten(feats["value"], 1)


def z_distance_tensor(
    anchor_feat_unit: torch.Tensor,
    candidate_feat: torch.Tensor,
    calibration: Calibration,
) -> torch.Tensor:
    candidate_unit = F.normalize(candidate_feat, dim=1)
    cosine_similarity = (anchor_feat_unit * candidate_unit).sum(dim=1)
    raw_distance = 1.0 - cosine_similarity
    return ((raw_distance - calibration.raw_mean) / calibration.raw_std).squeeze(0)


def load_sd_pipeline(model_path: Path, device: torch.device) -> StableDiffusionImg2ImgPipeline:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    for module in [pipe.unet, pipe.vae, pipe.text_encoder]:
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)
    return pipe


def encode_prompt(
    pipe: StableDiffusionImg2ImgPipeline,
    prompt: str,
    negative_prompt: str,
    device: torch.device,
    do_cfg: bool,
) -> torch.Tensor:
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=negative_prompt,
    )
    if do_cfg:
        return torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    return prompt_embeds


def decode_latents(pipe: StableDiffusionImg2ImgPipeline, latents: torch.Tensor) -> torch.Tensor:
    decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    return (decoded / 2 + 0.5).clamp(0, 1)


def prepare_img2img_latents(
    pipe: StableDiffusionImg2ImgPipeline,
    init_image: Image.Image,
    num_steps: int,
    strength: float,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    pipe.scheduler.set_timesteps(num_steps, device=device)
    init_timestep = min(int(num_steps * strength), num_steps)
    t_start = max(num_steps - init_timestep, 0)
    timesteps = pipe.scheduler.timesteps[t_start:]
    latent_timestep = timesteps[:1].repeat(1)
    image_tensor = pipe.image_processor.preprocess(init_image).to(device=device, dtype=torch.float16)
    latents = pipe.prepare_latents(
        image_tensor,
        latent_timestep,
        batch_size=1,
        num_images_per_prompt=1,
        dtype=torch.float16,
        device=device,
        generator=generator,
    )
    return latents, timesteps


def predict_noise(
    pipe: StableDiffusionImg2ImgPipeline,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    do_cfg = guidance_scale > 1.0
    latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
    with torch.no_grad():
        noise_pred = pipe.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
    if do_cfg:
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
    return noise_pred.detach()


def score_predicted_x0(
    pipe: StableDiffusionImg2ImgPipeline,
    resnet: torch.nn.Module,
    anchor_feat_unit: torch.Tensor,
    calibration: Calibration,
    target_z: float,
    target_01: torch.Tensor,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    guidance_scale: float,
    extra_step_kwargs: dict,
) -> tuple[torch.Tensor, object, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict clean x0 from a diffusion latent and score it with the RSA loss."""
    noise_pred = predict_noise(pipe, latents, timestep, prompt_embeds, guidance_scale)
    step_preview = pipe.scheduler.step(
        noise_pred,
        timestep,
        latents,
        **extra_step_kwargs,
        return_dict=True,
    )
    pred_x0_latents = step_preview.pred_original_sample
    pred_image_01 = decode_latents(pipe, pred_x0_latents)
    candidate_feat = extract_resnet_features(resnet, pred_image_01)
    z_model = z_distance_tensor(anchor_feat_unit, candidate_feat, calibration)

    alignment_loss = (z_model - target_z).pow(2)
    identity_loss = F.mse_loss(pred_image_01.float(), target_01.float())
    tv_loss = total_variation(pred_image_01.float())
    return noise_pred, step_preview, pred_image_01, z_model, alignment_loss, identity_loss, tv_loss, candidate_feat


def run(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    calibration = load_calibration(args.subject, args.anchor_idx, args.target_idx)
    target_z = calibration.brain_target_z if args.target_mode == "raw" else calibration.quantile_target_model_z

    anchor_path = IMAGE_DIR / f"image_{args.anchor_idx:04d}.png"
    target_path = IMAGE_DIR / f"image_{args.target_idx:04d}.png"
    anchor_pil = resize_pil_square(read_rgb(anchor_path), args.resolution)
    target_pil = resize_pil_square(read_rgb(target_path), args.resolution)
    target_01 = TF.to_tensor(target_pil).unsqueeze(0).to(device)

    resnet = build_resnet50(device)
    with torch.no_grad():
        anchor_resnet_01 = preprocess_pil_to_resnet_tensor(read_rgb(anchor_path)).to(device)
        target_resnet_01 = preprocess_pil_to_resnet_tensor(read_rgb(target_path)).to(device)
        anchor_feat = extract_resnet_features(resnet, anchor_resnet_01)
        anchor_feat_unit = F.normalize(anchor_feat, dim=1)
        target_feat = extract_resnet_features(resnet, target_resnet_01)
        original_z = float(z_distance_tensor(anchor_feat_unit, target_feat, calibration).cpu())

    pipe = load_sd_pipeline(args.model_path, device)
    do_cfg = args.guidance_scale > 1.0
    prompt_embeds = encode_prompt(pipe, args.prompt, args.negative_prompt, device, do_cfg)
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta=args.eta)
    latents, timesteps = prepare_img2img_latents(
        pipe=pipe,
        init_image=target_pil,
        num_steps=args.num_steps,
        strength=args.strength,
        generator=generator,
        device=device,
    )
    latents = stabilize_latents(latents, args.latent_clip_value)

    records = []
    best = {"abs_error": float("inf"), "image": target_01.detach().cpu(), "step_index": -1, "z": original_z}
    snapshot_indices = set(args.snapshot_indices)

    for step_index, timestep in enumerate(timesteps):
        guided_latents = stabilize_latents(latents.detach(), args.latent_clip_value)
        grad_rms_value = np.nan
        update_rms_value = np.nan
        guidance_factor = np.nan

        for _inner_index in range(args.inner_guidance_steps):
            guided_latents = guided_latents.detach().requires_grad_(True)
            (
                _noise_pred,
                _step_preview,
                pred_image_01,
                z_model,
                alignment_loss,
                identity_loss,
                tv_loss,
                candidate_feat,
            ) = score_predicted_x0(
                pipe,
                resnet,
                anchor_feat_unit,
                calibration,
                target_z,
                target_01,
                guided_latents,
                timestep,
                prompt_embeds,
                args.guidance_scale,
                extra_step_kwargs,
            )
            loss = (
                args.alignment_weight * alignment_loss
                + args.identity_weight * identity_loss
                + args.tv_weight * tv_loss
            )

            grad = torch.autograd.grad(loss, guided_latents)[0]
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_rms = grad.flatten(1).pow(2).mean(dim=1).sqrt().view(-1, 1, 1, 1).clamp_min(1e-8)
            abs_error_for_step = abs(float(z_model.detach().cpu()) - target_z)
            guidance_factor = 1.0
            if args.guidance_error_scale > 0:
                guidance_factor = min(1.0, abs_error_for_step / args.guidance_error_scale)
            update = args.guidance_step_size * guidance_factor * grad / grad_rms
            guided_latents = stabilize_latents(
                guided_latents - update,
                args.latent_clip_value,
            )
            grad_rms_value = float(grad_rms.mean().detach().cpu())
            update_rms_value = float(update.flatten(1).pow(2).mean(dim=1).sqrt().mean().detach().cpu())

            del pred_image_01, candidate_feat, loss, grad, update, _noise_pred, _step_preview
            torch.cuda.empty_cache()

        guided_latents = stabilize_latents(guided_latents.detach(), args.latent_clip_value)
        with torch.no_grad():
            noise_pred, step_preview, pred_image_01, z_model, alignment_loss, identity_loss, tv_loss, candidate_feat = score_predicted_x0(
                pipe,
                resnet,
                anchor_feat_unit,
                calibration,
                target_z,
                target_01,
                guided_latents,
                timestep,
                prompt_embeds,
                args.guidance_scale,
                extra_step_kwargs,
            )
            logged_loss = (
                args.alignment_weight * alignment_loss
                + args.identity_weight * identity_loss
                + args.tv_weight * tv_loss
            )
            latents = pipe.scheduler.step(
                noise_pred,
                timestep,
                guided_latents,
                **extra_step_kwargs,
                return_dict=True,
            ).prev_sample
            latents = stabilize_latents(latents, args.latent_clip_value)

        abs_error = abs(float(z_model.detach().cpu()) - target_z)
        pixel_rmse = float(torch.sqrt(identity_loss.detach()).cpu())
        if abs_error < best["abs_error"]:
            best = {
                "abs_error": abs_error,
                "image": pred_image_01.detach().cpu(),
                "step_index": step_index,
                "z": float(z_model.detach().cpu()),
                "pixel_rmse": pixel_rmse,
            }

        if step_index in snapshot_indices:
            snapshot_path = IMAGE_OUT_DIR / (
                f"guided_diffusion_step_{step_index:03d}_"
                f"{args.target_mode}_seed{args.seed}.png"
            )
            Image.fromarray(tensor_to_uint8_image(pred_image_01)).save(snapshot_path)

        records.append(
            {
                "step_index": step_index,
                "timestep": int(timestep.detach().cpu()),
                "loss": float(logged_loss.detach().cpu()),
                "alignment_loss": float(alignment_loss.detach().cpu()),
                "identity_mse": float(identity_loss.detach().cpu()),
                "pixel_rmse": pixel_rmse,
                "tv": float(tv_loss.detach().cpu()),
                "z_model_distance": float(z_model.detach().cpu()),
                "target_z": target_z,
                "target_mode": args.target_mode,
                "abs_error": abs_error,
                "original_z": original_z,
                "brain_target_z": calibration.brain_target_z,
                "quantile_target_model_z": calibration.quantile_target_model_z,
                "grad_rms": grad_rms_value,
                "update_rms": update_rms_value,
                "guidance_factor": guidance_factor,
                "inner_guidance_steps": args.inner_guidance_steps,
            }
        )

        del pred_image_01, step_preview, candidate_feat, noise_pred, guided_latents
        torch.cuda.empty_cache()

    with torch.no_grad():
        final_image_01 = decode_latents(pipe, latents.detach())
        final_feat = extract_resnet_features(resnet, final_image_01)
        final_z = float(z_distance_tensor(anchor_feat_unit, final_feat, calibration).cpu())
        final_rmse = float(torch.sqrt(F.mse_loss(final_image_01.float(), target_01.float())).cpu())

    run_label = args.run_label or f"guided_diffusion_{args.target_mode}_seed{args.seed}"
    stem = f"resnet50_pair_{args.anchor_idx:04d}_{args.target_idx:04d}_{args.subject}_{run_label}"
    trajectory_path = RESULTS_DIR / f"{stem}_trajectory.csv"
    pd.DataFrame(records).to_csv(trajectory_path, index=False)

    best_path = IMAGE_OUT_DIR / f"{stem}_best_predx0.png"
    final_path = IMAGE_OUT_DIR / f"{stem}_final.png"
    Image.fromarray(tensor_to_uint8_image(best["image"])).save(best_path)
    Image.fromarray(tensor_to_uint8_image(final_image_01)).save(final_path)

    summary = pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "subject": args.subject,
                "anchor_idx": args.anchor_idx,
                "target_idx": args.target_idx,
                "run_label": run_label,
                "target_mode": args.target_mode,
                "target_z": target_z,
                "brain_target_z": calibration.brain_target_z,
                "quantile_target_model_z": calibration.quantile_target_model_z,
                "cached_original_z": calibration.cached_original_z,
                "live_original_z": original_z,
                "best_z": best["z"],
                "best_abs_error": best["abs_error"],
                "best_pixel_rmse": best.get("pixel_rmse", np.nan),
                "best_step_index": best["step_index"],
                "final_z": final_z,
                "final_abs_error": abs(final_z - target_z),
                "final_pixel_rmse": final_rmse,
                "prompt": args.prompt,
                "negative_prompt": args.negative_prompt,
                "num_steps": args.num_steps,
                "strength": args.strength,
                "guidance_scale": args.guidance_scale,
                "guidance_step_size": args.guidance_step_size,
                "guidance_error_scale": args.guidance_error_scale,
                "inner_guidance_steps": args.inner_guidance_steps,
                "latent_clip_value": args.latent_clip_value,
                "identity_weight": args.identity_weight,
                "tv_weight": args.tv_weight,
                "seed": args.seed,
                "trajectory_path": str(trajectory_path),
                "best_path": str(best_path),
                "final_path": str(final_path),
            }
        ]
    )
    summary_path = RESULTS_DIR / f"{stem}_summary.csv"
    summary.to_csv(summary_path, index=False)

    make_figure(
        anchor=anchor_pil,
        target=target_pil,
        best_image=best["image"],
        final_image=final_image_01.detach().cpu(),
        trajectory=pd.DataFrame(records),
        summary=summary.iloc[0],
        figure_path=FIGURES_DIR / f"{stem}.png",
    )

    print(f"Wrote {trajectory_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {best_path}")
    print(f"Wrote {final_path}")
    print(f"Wrote {FIGURES_DIR / f'{stem}.png'}")
    print(summary.to_string(index=False))


def make_figure(
    anchor: Image.Image,
    target: Image.Image,
    best_image: torch.Tensor,
    final_image: torch.Tensor,
    trajectory: pd.DataFrame,
    summary: pd.Series,
    figure_path: Path,
) -> None:
    target_tensor = TF.to_tensor(target).unsqueeze(0)
    delta = (best_image - target_tensor).abs()

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.95], hspace=0.35, wspace=0.18)
    panels = [
        ("anchor", TF.to_tensor(anchor).unsqueeze(0)),
        ("source target", target_tensor),
        ("best guided x0", best_image),
        ("abs change", delta),
    ]
    for idx, (title, tensor) in enumerate(panels):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(tensor_to_uint8_image(tensor))
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    ax = fig.add_subplot(gs[1, :2])
    ax.plot(trajectory["step_index"], trajectory["z_model_distance"], color="#1f77b4", lw=2, label="pred-x0 ResNet50 z")
    ax.axhline(summary["target_z"], color="#d62728", lw=1.4, ls="--", label="target z")
    ax.axhline(summary["live_original_z"], color="#555555", lw=1.2, ls=":", label="original z")
    ax.set_xlabel("denoising step")
    ax.set_ylabel("z-scored distance")
    ax.set_title("Guided diffusion alignment")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(tensor_to_uint8_image(final_image))
    ax.set_title("final sample", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 3])
    ax.plot(trajectory["step_index"], trajectory["pixel_rmse"], color="#2ca02c", lw=2)
    ax.set_xlabel("denoising step")
    ax.set_ylabel("pixel RMSE")
    ax.set_title("identity cost")

    fig.suptitle("Gradient-guided diffusion counterfactual", fontsize=12, fontweight="bold")
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_SD15_SNAPSHOT)
    parser.add_argument("--anchor-idx", type=int, default=ANCHOR_IDX)
    parser.add_argument("--target-idx", type=int, default=TARGET_IDX)
    parser.add_argument("--subject", default=SUBJECT)
    parser.add_argument("--target-mode", choices=["raw", "quantile"], default="quantile")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--strength", type=float, default=0.55)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--guidance-step-size", type=float, default=0.08)
    parser.add_argument("--guidance-error-scale", type=float, default=1.0)
    parser.add_argument("--inner-guidance-steps", type=int, default=1)
    parser.add_argument("--latent-clip-value", type=float, default=10.0)
    parser.add_argument("--alignment-weight", type=float, default=1.0)
    parser.add_argument("--identity-weight", type=float, default=1.0)
    parser.add_argument("--tv-weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-label", default="")
    parser.add_argument(
        "--prompt",
        default="a realistic natural photograph of a light beige stone tile wall, neutral lighting, detailed masonry texture",
    )
    parser.add_argument(
        "--negative-prompt",
        default="flowers, colorful stains, neon colors, rainbow, psychedelic, painting, illustration, text, artifacts, low quality",
    )
    parser.add_argument("--snapshot-indices", type=int, nargs="*", default=[0, 5, 10, 20, 29])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
