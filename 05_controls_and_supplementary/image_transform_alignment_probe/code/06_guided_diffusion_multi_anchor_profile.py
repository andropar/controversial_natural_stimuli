#!/usr/bin/env python3
"""Multi-anchor gradient-guided diffusion counterfactual.

This extends the single-pair guided diffusion prototype from
`05_guided_diffusion_counterfactual.py` to an RSA-style distance profile.

For a fixed target image B, choose K anchor images A_i.  At each diffusion
denoising step, decode the predicted clean image B', compute ResNet50 distances
from B' to the anchors, and backpropagate a profile loss to the current
diffusion latent:

    d_M(A_i, B') ~= target_i

The default target_i is quantile calibrated: each subject-specific brain
distance z_B(A_i, B) is converted to the same empirical quantile in the live
ResNet50 distance distribution.  This avoids direct raw brain/model z matching.
"""

from __future__ import annotations

import argparse
import importlib.util
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
import torchvision.transforms.functional as TF
from PIL import Image


SINGLE_PAIR_SCRIPT = Path(__file__).with_name("05_guided_diffusion_counterfactual.py")
_spec = importlib.util.spec_from_file_location("single_pair_guided_diffusion", SINGLE_PAIR_SCRIPT)
single = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = single
_spec.loader.exec_module(single)

ROOT = single.ROOT
ANALYSIS_DIR = single.ANALYSIS_DIR
RESULTS_DIR = single.RESULTS_DIR
FIGURES_DIR = single.FIGURES_DIR
IMAGE_OUT_DIR = single.IMAGE_OUT_DIR
PAIR_SUBJECT = single.PAIR_SUBJECT
IMAGE_DIR = single.IMAGE_DIR
DEFAULT_SD15_SNAPSHOT = single.DEFAULT_SD15_SNAPSHOT
MODEL_NAME = single.MODEL_NAME
SUBJECT = single.SUBJECT
TARGET_IDX = single.TARGET_IDX
DEFAULT_INCLUDE_ANCHOR_IDX = single.ANCHOR_IDX


@dataclass(frozen=True)
class ProfileCalibration:
    raw_mean: float
    raw_std: float
    model_z_distribution: np.ndarray


@dataclass(frozen=True)
class ProfileMetrics:
    train_rmse: float
    train_mae: float
    train_corr: float
    holdout_rmse: float
    holdout_mae: float
    holdout_corr: float


def pearson_corr_tensor(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.float()
    y = y.float()
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    return (x_centered * y_centered).sum() / (
        x_centered.norm() * y_centered.norm()
    ).clamp_min(1e-8)


def pearson_corr_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denom)


def load_live_resnet_features(
    resnet: torch.nn.Module,
    device: torch.device,
    image_count: int,
    cache_path: Path | None,
    force_recompute: bool,
) -> torch.Tensor:
    if cache_path is not None and cache_path.exists() and not force_recompute:
        features = np.load(cache_path)
        if features.shape[0] >= image_count:
            return torch.from_numpy(features[:image_count]).to(device=device, dtype=torch.float32)

    features = []
    with torch.no_grad():
        for image_idx in range(image_count):
            image_path = IMAGE_DIR / f"image_{image_idx:04d}.png"
            image_01 = single.preprocess_pil_to_resnet_tensor(single.read_rgb(image_path)).to(device)
            feat = single.extract_resnet_features(resnet, image_01)
            features.append(feat.detach().cpu())
    feature_tensor = torch.cat(features, dim=0).float()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, feature_tensor.numpy())

    return feature_tensor.to(device=device)


def make_profile_calibration(live_features: torch.Tensor) -> ProfileCalibration:
    with torch.no_grad():
        unit = F.normalize(live_features.float(), dim=1)
        raw_matrix = 1.0 - (unit @ unit.T)
    raw_np = raw_matrix.detach().cpu().numpy()
    upper = raw_np[np.triu_indices(raw_np.shape[0], k=1)]
    model_z_distribution = (upper - upper.mean()) / upper.std(ddof=0)
    return ProfileCalibration(
        raw_mean=float(upper.mean()),
        raw_std=float(upper.std(ddof=0)),
        model_z_distribution=model_z_distribution,
    )


def select_profile_anchors(
    subject: str,
    target_idx: int,
    n_train: int,
    n_holdout: int,
    include_anchor_idx: int | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    pair_subject = pd.read_csv(PAIR_SUBJECT)
    subject_rows = pair_subject[pair_subject["subject"] == subject].copy()
    subject_brain_z = subject_rows["brain_z"].dropna().to_numpy()

    profile = subject_rows[(subject_rows["img_i"] == target_idx) | (subject_rows["img_j"] == target_idx)].copy()
    profile["anchor_idx"] = np.where(
        profile["img_i"].to_numpy() == target_idx,
        profile["img_j"].to_numpy(),
        profile["img_i"].to_numpy(),
    ).astype(int)
    profile = profile[profile["anchor_idx"] != target_idx].sort_values("brain_z").reset_index(drop=True)

    total = n_train + n_holdout
    if total > len(profile):
        raise ValueError(f"Requested {total} anchors, but only {len(profile)} are available")

    fixed = pd.DataFrame()
    if include_anchor_idx is not None:
        fixed = profile[profile["anchor_idx"] == include_anchor_idx]
        if fixed.empty:
            raise ValueError(f"Requested include anchor {include_anchor_idx}, but it was not found")

    remaining = profile[~profile["anchor_idx"].isin(fixed["anchor_idx"])]
    n_needed = total - len(fixed)
    positions = np.linspace(0, len(remaining) - 1, n_needed).round().astype(int)
    selected = pd.concat([fixed, remaining.iloc[positions]], ignore_index=True)
    selected = selected.drop_duplicates("anchor_idx").copy()

    if len(selected) < total:
        extra = remaining[~remaining["anchor_idx"].isin(selected["anchor_idx"])]
        selected = pd.concat([selected, extra.head(total - len(selected))], ignore_index=True)

    selected = selected.sort_values("brain_z").reset_index(drop=True)
    selected["split"] = "train"
    holdout_candidates = selected.copy()
    if include_anchor_idx is not None:
        holdout_candidates = holdout_candidates[holdout_candidates["anchor_idx"] != include_anchor_idx]
    holdout_positions = np.linspace(0, len(holdout_candidates) - 1, n_holdout).round().astype(int)
    holdout_anchors = set(holdout_candidates.iloc[holdout_positions]["anchor_idx"].astype(int).tolist())
    selected.loc[selected["anchor_idx"].isin(holdout_anchors), "split"] = "holdout"

    if (selected["split"] == "train").sum() != n_train:
        raise RuntimeError("Anchor split construction produced the wrong train count")
    if (selected["split"] == "holdout").sum() != n_holdout:
        raise RuntimeError("Anchor split construction produced the wrong holdout count")

    return selected, subject_brain_z


def add_profile_targets(
    selected: pd.DataFrame,
    subject_brain_z: np.ndarray,
    calibration: ProfileCalibration,
    target_mode: str,
) -> pd.DataFrame:
    selected = selected.copy()
    quantiles = []
    target_model_z = []
    for brain_z in selected["brain_z"].to_numpy(dtype=float):
        quantile = float(np.mean(subject_brain_z <= brain_z))
        quantiles.append(quantile)
        if target_mode == "quantile":
            target_model_z.append(float(np.quantile(calibration.model_z_distribution, quantile)))
        else:
            target_model_z.append(float(brain_z))
    selected["brain_quantile"] = quantiles
    selected["target_model_z"] = target_model_z
    return selected


def z_profile_tensor(
    anchor_feat_unit: torch.Tensor,
    candidate_feat: torch.Tensor,
    calibration: ProfileCalibration,
) -> torch.Tensor:
    candidate_unit = F.normalize(candidate_feat.float(), dim=1)
    raw_distance = 1.0 - (anchor_feat_unit.float() @ candidate_unit.squeeze(0))
    return (raw_distance - calibration.raw_mean) / calibration.raw_std


def profile_metrics(
    profile_z: np.ndarray,
    target_z: np.ndarray,
    train_mask: np.ndarray,
    holdout_mask: np.ndarray,
) -> ProfileMetrics:
    def split_metrics(mask: np.ndarray) -> tuple[float, float, float]:
        if mask.sum() == 0:
            return float("nan"), float("nan"), float("nan")
        diff = profile_z[mask] - target_z[mask]
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))
        corr = pearson_corr_np(profile_z[mask], target_z[mask])
        return rmse, mae, corr

    train_rmse, train_mae, train_corr = split_metrics(train_mask)
    holdout_rmse, holdout_mae, holdout_corr = split_metrics(holdout_mask)
    return ProfileMetrics(
        train_rmse=train_rmse,
        train_mae=train_mae,
        train_corr=train_corr,
        holdout_rmse=holdout_rmse,
        holdout_mae=holdout_mae,
        holdout_corr=holdout_corr,
    )


def score_predicted_x0_profile(
    pipe,
    resnet: torch.nn.Module,
    anchor_feat_unit: torch.Tensor,
    calibration: ProfileCalibration,
    target_01: torch.Tensor,
    target_profile_z: torch.Tensor,
    train_mask: torch.Tensor,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    guidance_scale: float,
    extra_step_kwargs: dict,
    mse_weight: float,
    corr_weight: float,
) -> tuple:
    noise_pred = single.predict_noise(pipe, latents, timestep, prompt_embeds, guidance_scale)
    step_preview = pipe.scheduler.step(
        noise_pred,
        timestep,
        latents,
        **extra_step_kwargs,
        return_dict=True,
    )
    pred_image_01 = single.decode_latents(pipe, step_preview.pred_original_sample)
    candidate_feat = single.extract_resnet_features(resnet, pred_image_01)
    model_profile_z = z_profile_tensor(anchor_feat_unit, candidate_feat, calibration)

    train_model_z = model_profile_z[train_mask]
    train_target_z = target_profile_z[train_mask]
    profile_mse = F.mse_loss(train_model_z.float(), train_target_z.float())
    profile_corr = pearson_corr_tensor(train_model_z, train_target_z)
    profile_corr_loss = 1.0 - profile_corr
    alignment_loss = mse_weight * profile_mse + corr_weight * profile_corr_loss
    identity_loss = F.mse_loss(pred_image_01.float(), target_01.float())
    tv_loss = single.total_variation(pred_image_01.float())

    return (
        noise_pred,
        step_preview,
        pred_image_01,
        candidate_feat,
        model_profile_z,
        alignment_loss,
        profile_mse,
        profile_corr,
        identity_loss,
        tv_loss,
    )


def image_count_from_pair_table() -> int:
    pair_subject = pd.read_csv(PAIR_SUBJECT, usecols=["img_i", "img_j"])
    return int(max(pair_subject["img_i"].max(), pair_subject["img_j"].max()) + 1)


def run(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    target_path = IMAGE_DIR / f"image_{args.target_idx:04d}.png"
    target_pil = single.resize_pil_square(single.read_rgb(target_path), args.resolution)
    target_01 = TF.to_tensor(target_pil).unsqueeze(0).to(device)

    resnet = single.build_resnet50(device)
    image_count = image_count_from_pair_table()
    live_features = load_live_resnet_features(
        resnet=resnet,
        device=device,
        image_count=image_count,
        cache_path=args.feature_cache,
        force_recompute=args.force_recompute_features,
    )
    calibration = make_profile_calibration(live_features)

    selected, subject_brain_z = select_profile_anchors(
        subject=args.subject,
        target_idx=args.target_idx,
        n_train=args.n_train_anchors,
        n_holdout=args.n_holdout_anchors,
        include_anchor_idx=args.include_anchor_idx,
    )
    selected = add_profile_targets(selected, subject_brain_z, calibration, args.target_mode)

    anchor_indices = selected["anchor_idx"].astype(int).to_numpy(copy=True)
    train_mask_np = (selected["split"].to_numpy() == "train")
    holdout_mask_np = ~train_mask_np
    target_profile_np = selected["target_model_z"].to_numpy(dtype=np.float32)

    with torch.no_grad():
        live_unit = F.normalize(live_features.float(), dim=1)
        anchor_feat_unit = live_unit[anchor_indices].to(device)
        original_target_feat = live_features[args.target_idx : args.target_idx + 1].to(device)
        original_profile_z_tensor = z_profile_tensor(anchor_feat_unit, original_target_feat, calibration)

    original_profile_np = original_profile_z_tensor.detach().cpu().numpy()
    original_metrics = profile_metrics(
        original_profile_np,
        target_profile_np,
        train_mask_np,
        holdout_mask_np,
    )
    selected["original_live_model_z"] = original_profile_np
    selected["original_abs_error"] = np.abs(original_profile_np - target_profile_np)

    pipe = single.load_sd_pipeline(args.model_path, device)
    do_cfg = args.guidance_scale > 1.0
    prompt_embeds = single.encode_prompt(pipe, args.prompt, args.negative_prompt, device, do_cfg)
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta=args.eta)
    latents, timesteps = single.prepare_img2img_latents(
        pipe=pipe,
        init_image=target_pil,
        num_steps=args.num_steps,
        strength=args.strength,
        generator=generator,
        device=device,
    )
    latents = single.stabilize_latents(latents, args.latent_clip_value)

    target_profile_t = torch.from_numpy(target_profile_np).to(device=device, dtype=torch.float32)
    train_mask_t = torch.from_numpy(train_mask_np).to(device=device, dtype=torch.bool)

    records = []
    snapshot_indices = set(args.snapshot_indices)
    best = {
        "train_rmse": float("inf"),
        "image": target_01.detach().cpu(),
        "step_index": -1,
        "profile_z": original_profile_np,
        "pixel_rmse": 0.0,
    }

    for step_index, timestep in enumerate(timesteps):
        guided_latents = single.stabilize_latents(latents.detach(), args.latent_clip_value)
        grad_rms_value = np.nan
        update_rms_value = np.nan
        guidance_factor = np.nan

        for _inner_idx in range(args.inner_guidance_steps):
            guided_latents = guided_latents.detach().requires_grad_(True)
            (
                _noise_pred,
                _step_preview,
                pred_image_01,
                candidate_feat,
                model_profile_z,
                alignment_loss,
                profile_mse,
                profile_corr,
                identity_loss,
                tv_loss,
            ) = score_predicted_x0_profile(
                pipe=pipe,
                resnet=resnet,
                anchor_feat_unit=anchor_feat_unit,
                calibration=calibration,
                target_01=target_01,
                target_profile_z=target_profile_t,
                train_mask=train_mask_t,
                latents=guided_latents,
                timestep=timestep,
                prompt_embeds=prompt_embeds,
                guidance_scale=args.guidance_scale,
                extra_step_kwargs=extra_step_kwargs,
                mse_weight=args.profile_mse_weight,
                corr_weight=args.profile_corr_weight,
            )
            loss = (
                args.alignment_weight * alignment_loss
                + args.identity_weight * identity_loss
                + args.tv_weight * tv_loss
            )
            grad = torch.autograd.grad(loss, guided_latents)[0]
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_rms = grad.flatten(1).pow(2).mean(dim=1).sqrt().view(-1, 1, 1, 1).clamp_min(1e-8)

            train_rmse_for_step = float(torch.sqrt(profile_mse.detach()).cpu())
            guidance_factor = 1.0
            if args.guidance_error_scale > 0:
                guidance_factor = min(1.0, train_rmse_for_step / args.guidance_error_scale)
            update = args.guidance_step_size * guidance_factor * grad / grad_rms
            guided_latents = single.stabilize_latents(
                guided_latents - update,
                args.latent_clip_value,
            )
            grad_rms_value = float(grad_rms.mean().detach().cpu())
            update_rms_value = float(update.flatten(1).pow(2).mean(dim=1).sqrt().mean().detach().cpu())

            del pred_image_01, candidate_feat, model_profile_z, loss, grad, update, _noise_pred, _step_preview
            torch.cuda.empty_cache()

        guided_latents = single.stabilize_latents(guided_latents.detach(), args.latent_clip_value)
        with torch.no_grad():
            (
                noise_pred,
                _step_preview,
                pred_image_01,
                candidate_feat,
                model_profile_z,
                alignment_loss,
                profile_mse,
                profile_corr,
                identity_loss,
                tv_loss,
            ) = score_predicted_x0_profile(
                pipe=pipe,
                resnet=resnet,
                anchor_feat_unit=anchor_feat_unit,
                calibration=calibration,
                target_01=target_01,
                target_profile_z=target_profile_t,
                train_mask=train_mask_t,
                latents=guided_latents,
                timestep=timestep,
                prompt_embeds=prompt_embeds,
                guidance_scale=args.guidance_scale,
                extra_step_kwargs=extra_step_kwargs,
                mse_weight=args.profile_mse_weight,
                corr_weight=args.profile_corr_weight,
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
            latents = single.stabilize_latents(latents, args.latent_clip_value)

        model_profile_np = model_profile_z.detach().cpu().numpy()
        metrics = profile_metrics(model_profile_np, target_profile_np, train_mask_np, holdout_mask_np)
        pixel_rmse = float(torch.sqrt(identity_loss.detach()).cpu())

        if metrics.train_rmse < best["train_rmse"]:
            best = {
                "train_rmse": metrics.train_rmse,
                "image": pred_image_01.detach().cpu(),
                "step_index": step_index,
                "profile_z": model_profile_np,
                "pixel_rmse": pixel_rmse,
                "metrics": metrics,
            }

        if step_index in snapshot_indices:
            snapshot_path = IMAGE_OUT_DIR / (
                f"guided_profile_step_{step_index:03d}_{args.target_mode}_seed{args.seed}.png"
            )
            Image.fromarray(single.tensor_to_uint8_image(pred_image_01)).save(snapshot_path)

        records.append(
            {
                "step_index": step_index,
                "timestep": int(timestep.detach().cpu()),
                "loss": float(logged_loss.detach().cpu()),
                "alignment_loss": float(alignment_loss.detach().cpu()),
                "profile_mse": float(profile_mse.detach().cpu()),
                "profile_corr": float(profile_corr.detach().cpu()),
                "train_rmse": metrics.train_rmse,
                "train_mae": metrics.train_mae,
                "train_corr": metrics.train_corr,
                "holdout_rmse": metrics.holdout_rmse,
                "holdout_mae": metrics.holdout_mae,
                "holdout_corr": metrics.holdout_corr,
                "identity_mse": float(identity_loss.detach().cpu()),
                "pixel_rmse": pixel_rmse,
                "tv": float(tv_loss.detach().cpu()),
                "original_train_rmse": original_metrics.train_rmse,
                "original_holdout_rmse": original_metrics.holdout_rmse,
                "grad_rms": grad_rms_value,
                "update_rms": update_rms_value,
                "guidance_factor": guidance_factor,
            }
        )

        del pred_image_01, candidate_feat, model_profile_z, noise_pred, guided_latents, _step_preview
        torch.cuda.empty_cache()

    with torch.no_grad():
        final_image_01 = single.decode_latents(pipe, latents.detach())
        final_feat = single.extract_resnet_features(resnet, final_image_01)
        final_profile_z_tensor = z_profile_tensor(anchor_feat_unit, final_feat, calibration)
        final_profile_np = final_profile_z_tensor.detach().cpu().numpy()
        final_metrics = profile_metrics(final_profile_np, target_profile_np, train_mask_np, holdout_mask_np)
        final_rmse = float(torch.sqrt(F.mse_loss(final_image_01.float(), target_01.float())).cpu())

    best_metrics = best.get(
        "metrics",
        profile_metrics(best["profile_z"], target_profile_np, train_mask_np, holdout_mask_np),
    )

    selected["best_model_z"] = best["profile_z"]
    selected["best_abs_error"] = np.abs(best["profile_z"] - target_profile_np)
    selected["final_model_z"] = final_profile_np
    selected["final_abs_error"] = np.abs(final_profile_np - target_profile_np)

    run_label = args.run_label or f"guided_profile_{args.target_mode}_seed{args.seed}"
    stem = f"resnet50_target_{args.target_idx:04d}_{args.subject}_{run_label}"
    trajectory_path = RESULTS_DIR / f"{stem}_trajectory.csv"
    anchor_profile_path = RESULTS_DIR / f"{stem}_anchor_profile.csv"
    summary_path = RESULTS_DIR / f"{stem}_summary.csv"
    best_path = IMAGE_OUT_DIR / f"{stem}_best_predx0.png"
    final_path = IMAGE_OUT_DIR / f"{stem}_final.png"
    figure_path = FIGURES_DIR / f"{stem}.png"

    pd.DataFrame(records).to_csv(trajectory_path, index=False)
    selected.to_csv(anchor_profile_path, index=False)
    Image.fromarray(single.tensor_to_uint8_image(best["image"])).save(best_path)
    Image.fromarray(single.tensor_to_uint8_image(final_image_01)).save(final_path)

    summary = pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "subject": args.subject,
                "target_idx": args.target_idx,
                "run_label": run_label,
                "target_mode": args.target_mode,
                "n_train_anchors": int(train_mask_np.sum()),
                "n_holdout_anchors": int(holdout_mask_np.sum()),
                "anchor_indices": " ".join(str(x) for x in anchor_indices),
                "train_anchor_indices": " ".join(str(x) for x in anchor_indices[train_mask_np]),
                "holdout_anchor_indices": " ".join(str(x) for x in anchor_indices[holdout_mask_np]),
                "original_train_rmse": original_metrics.train_rmse,
                "original_train_mae": original_metrics.train_mae,
                "original_train_corr": original_metrics.train_corr,
                "original_holdout_rmse": original_metrics.holdout_rmse,
                "original_holdout_mae": original_metrics.holdout_mae,
                "original_holdout_corr": original_metrics.holdout_corr,
                "best_step_index": best["step_index"],
                "best_train_rmse": best_metrics.train_rmse,
                "best_train_mae": best_metrics.train_mae,
                "best_train_corr": best_metrics.train_corr,
                "best_holdout_rmse": best_metrics.holdout_rmse,
                "best_holdout_mae": best_metrics.holdout_mae,
                "best_holdout_corr": best_metrics.holdout_corr,
                "best_pixel_rmse": best.get("pixel_rmse", np.nan),
                "final_train_rmse": final_metrics.train_rmse,
                "final_train_mae": final_metrics.train_mae,
                "final_train_corr": final_metrics.train_corr,
                "final_holdout_rmse": final_metrics.holdout_rmse,
                "final_holdout_mae": final_metrics.holdout_mae,
                "final_holdout_corr": final_metrics.holdout_corr,
                "final_pixel_rmse": final_rmse,
                "num_steps": args.num_steps,
                "strength": args.strength,
                "guidance_scale": args.guidance_scale,
                "guidance_step_size": args.guidance_step_size,
                "guidance_error_scale": args.guidance_error_scale,
                "inner_guidance_steps": args.inner_guidance_steps,
                "latent_clip_value": args.latent_clip_value,
                "profile_mse_weight": args.profile_mse_weight,
                "profile_corr_weight": args.profile_corr_weight,
                "alignment_weight": args.alignment_weight,
                "identity_weight": args.identity_weight,
                "tv_weight": args.tv_weight,
                "seed": args.seed,
                "prompt": args.prompt,
                "negative_prompt": args.negative_prompt,
                "trajectory_path": str(trajectory_path),
                "anchor_profile_path": str(anchor_profile_path),
                "best_path": str(best_path),
                "final_path": str(final_path),
                "figure_path": str(figure_path),
            }
        ]
    )
    summary.to_csv(summary_path, index=False)

    make_figure(
        target=target_pil,
        best_image=best["image"],
        final_image=final_image_01.detach().cpu(),
        trajectory=pd.DataFrame(records),
        profile_table=selected,
        summary=summary.iloc[0],
        figure_path=figure_path,
    )

    print(f"Wrote {trajectory_path}")
    print(f"Wrote {anchor_profile_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {best_path}")
    print(f"Wrote {final_path}")
    print(f"Wrote {figure_path}")
    print(summary.to_string(index=False))


def make_figure(
    target: Image.Image,
    best_image: torch.Tensor,
    final_image: torch.Tensor,
    trajectory: pd.DataFrame,
    profile_table: pd.DataFrame,
    summary: pd.Series,
    figure_path: Path,
) -> None:
    target_tensor = TF.to_tensor(target).unsqueeze(0)
    delta = (best_image - target_tensor).abs()

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05], hspace=0.35, wspace=0.25)

    panels = [
        ("source target", target_tensor),
        ("best guided x0", best_image),
        ("final sample", final_image),
        ("abs change", delta),
    ]
    for idx, (title, tensor) in enumerate(panels):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(single.tensor_to_uint8_image(tensor))
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(trajectory["step_index"], trajectory["train_rmse"], color="#1f77b4", lw=2, label="train")
    ax.plot(trajectory["step_index"], trajectory["holdout_rmse"], color="#ff7f0e", lw=2, label="held out")
    ax.axhline(summary["original_train_rmse"], color="#1f77b4", lw=1, ls=":")
    ax.axhline(summary["original_holdout_rmse"], color="#ff7f0e", lw=1, ls=":")
    ax.set_xlabel("denoising step")
    ax.set_ylabel("profile RMSE")
    ax.set_title("Profile error")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(trajectory["step_index"], trajectory["train_corr"], color="#1f77b4", lw=2, label="train")
    ax.plot(trajectory["step_index"], trajectory["holdout_corr"], color="#ff7f0e", lw=2, label="held out")
    ax.axhline(summary["original_train_corr"], color="#1f77b4", lw=1, ls=":")
    ax.axhline(summary["original_holdout_corr"], color="#ff7f0e", lw=1, ls=":")
    ax.set_xlabel("denoising step")
    ax.set_ylabel("profile r")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Profile correlation")

    ax = fig.add_subplot(gs[1, 2])
    split_colors = profile_table["split"].map({"train": "#1f77b4", "holdout": "#ff7f0e"}).to_numpy()
    target_z = profile_table["target_model_z"].to_numpy()
    ax.scatter(target_z, profile_table["original_live_model_z"], c="#999999", s=22, label="original", alpha=0.75)
    ax.scatter(target_z, profile_table["best_model_z"], c=split_colors, s=30, label="best", alpha=0.9)
    lim_min = float(np.nanmin([target_z.min(), profile_table["original_live_model_z"].min(), profile_table["best_model_z"].min()])) - 0.25
    lim_max = float(np.nanmax([target_z.max(), profile_table["original_live_model_z"].max(), profile_table["best_model_z"].max()])) + 0.25
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="#444444", lw=1, ls="--")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("target model z")
    ax.set_ylabel("observed model z")
    ax.set_title("Anchor profile")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[1, 3])
    ax.plot(trajectory["step_index"], trajectory["pixel_rmse"], color="#2ca02c", lw=2)
    ax.set_xlabel("denoising step")
    ax.set_ylabel("pixel RMSE")
    ax.set_title("Identity cost")

    fig.suptitle("Multi-anchor gradient-guided diffusion counterfactual", fontsize=12, fontweight="bold")
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_SD15_SNAPSHOT)
    parser.add_argument("--target-idx", type=int, default=TARGET_IDX)
    parser.add_argument("--subject", default=SUBJECT)
    parser.add_argument("--target-mode", choices=["raw", "quantile"], default="quantile")
    parser.add_argument("--n-train-anchors", type=int, default=16)
    parser.add_argument("--n-holdout-anchors", type=int, default=8)
    parser.add_argument("--include-anchor-idx", type=int, default=DEFAULT_INCLUDE_ANCHOR_IDX)
    parser.add_argument("--feature-cache", type=Path, default=RESULTS_DIR / "resnet50_live_features_all_models.npy")
    parser.add_argument("--force-recompute-features", action="store_true")
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--guidance-step-size", type=float, default=0.04)
    parser.add_argument("--guidance-error-scale", type=float, default=0.9)
    parser.add_argument("--inner-guidance-steps", type=int, default=2)
    parser.add_argument("--latent-clip-value", type=float, default=10.0)
    parser.add_argument("--profile-mse-weight", type=float, default=1.0)
    parser.add_argument("--profile-corr-weight", type=float, default=0.15)
    parser.add_argument("--alignment-weight", type=float, default=1.0)
    parser.add_argument("--identity-weight", type=float, default=0.2)
    parser.add_argument("--tv-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2)
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
    parser.add_argument("--snapshot-indices", type=int, nargs="*", default=[0, 5, 9, 12])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
