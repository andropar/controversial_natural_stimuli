#!/usr/bin/env python3
"""Compute in-silico recovery curves using VICCO baseline images.

This supplementary control mirrors the selection-evaluation discriminability
analysis, but replaces the natural-pool random comparison with subsamples from
the curated VICCO baseline image set. Outputs are written under
05_controls_and_supplementary so the primary selection-evaluation results stay
unchanged.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid


SCRIPT = Path(__file__).resolve()
SHARE = SCRIPT.parents[4]
ANALYSIS_DIR = SHARE / "00_stimulus_selection" / "decision_checks" / "selection_evaluation" / "code" / "analysis"
HELPERS_DIR = SHARE / "src"
SRC_DIR = SHARE / "src"
for path in (SRC_DIR, HELPERS_DIR, ANALYSIS_DIR):
    sys.path.insert(0, str(path))

from cstims import constants, paths
import utils as eval_utils  # noqa: E402
from cstims.evaluation.computation import (  # noqa: E402
    compute_all_rdms,
    compute_clean_correlation_matrix,
    compute_correlation_at_target_noise,
)


def _load_disc_module():
    path = ANALYSIS_DIR / "02_compute_discriminability.py"
    spec = importlib.util.spec_from_file_location("selection_eval_discriminability", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


disc = _load_disc_module()

MODEL_SET_ORDER = ["all_models", "sota", "training_objective", "architecture", "dataset"]
ENCODING_TRACKS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
VICCO_CACHE = (
    SHARE
    / "05_controls_and_supplementary"
    / "cstim_encoding_cv"
    / "cache_or_heavy"
    / "selected_layer_features_srp5920"
    / "features"
)
VICCO_RAW_CACHE = (
    SHARE
    / "shared"
    / "cache_or_heavy"
    / "cstim_paper_feature_cache"
    / "feature_cache"
    / "vicco"
)
RESULTS = SCRIPT.parents[1] / "results"


def layer_lookup() -> dict[str, str]:
    model_list = Path(paths.model_list_csv())
    if not model_list.exists():
        return {}
    df = pd.read_csv(model_list)
    return dict(zip(df["model"], df["layer"]))


def load_vicco_features(model_names: list[str]) -> dict[str, np.ndarray]:
    layers = layer_lookup()
    feats = {}
    for model in model_names:
        raw_path = VICCO_RAW_CACHE / model / "vicco.npy"
        if raw_path.exists():
            feats[model] = np.asarray(np.load(raw_path), dtype=np.float32)
            continue

        layer = layers.get(model)
        if layer is None:
            continue
        path = VICCO_CACHE / model / "vicco.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as z:
            if layer in z.files:
                key = layer
            else:
                candidates = [
                    key
                    for key in z.files
                    if not key.startswith("_")
                    and key != "image_filenames"
                    and getattr(z[key], "ndim", 0) >= 2
                ]
                if not candidates:
                    continue
                key = candidates[0]
            feats[model] = z[key].astype(np.float32)
    return feats


def make_subsamples(n_total: int, n_sample: int, n_subsets: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.sort(rng.choice(n_total, n_sample, replace=False)) for _ in range(n_subsets)]


def load_selection_payload(model_set: str) -> dict[str, Any]:
    path = SHARE / "00_stimulus_selection" / "results" / "selected_stimuli" / model_set / "selected_stimuli_data.pkl"
    with path.open("rb") as f:
        return pickle.load(f)


def encoding_root_map() -> dict[str, Path]:
    from cstims import paths
    UNIQUE_ENCODING_DIRS = paths.unique_encoding_dirs()

    return {k: v for k, v in UNIQUE_ENCODING_DIRS.items()}


def selected_for_track(
    payload: dict,
    track: dict,
    device: torch.device,
    selection_variant: str,
    encoding_cache: dict[str, Any],
    use_unique_encodings: bool,
) -> dict[str, torch.Tensor]:
    track_type = track.get("type", "identity")
    track_name = track["name"]
    if track_type == "identity":
        return eval_utils._load_selected_identity(
            payload, track_name, device, selection_variant
        )

    encoding_name = track.get("encoding_name") or track_name
    root_map = encoding_root_map() if use_unique_encodings else None
    raw_selected = eval_utils._load_selected_identity(
        payload, "raw", device, selection_variant
    )
    params = eval_utils._ensure_encoding_params(
        payload,
        [encoding_name],
        device,
        encoding_cache,
        encoding_root_map=root_map,
    )
    encoded = eval_utils.encode_batch_for_all_encodings(
        raw_selected, {encoding_name: params[encoding_name]}
    )
    return encoded[encoding_name]


def baseline_arrays_for_track(
    payload: dict,
    track: dict,
    vicco_raw: dict[str, np.ndarray],
    device: torch.device,
    encoding_cache: dict[str, Any],
    use_unique_encodings: bool,
    batch_size: int,
) -> dict[str, np.ndarray]:
    model_names = payload["model_names"]
    if track.get("type", "identity") == "identity":
        return {m: vicco_raw[m] for m in model_names if m in vicco_raw}

    encoding_name = track.get("encoding_name") or track["name"]
    root_map = encoding_root_map() if use_unique_encodings else None
    params = eval_utils._ensure_encoding_params(
        payload,
        [encoding_name],
        device,
        encoding_cache,
        encoding_root_map=root_map,
    )
    encoded_batches: dict[str, list[np.ndarray]] = {m: [] for m in model_names if m in vicco_raw}
    n_total = next(iter(vicco_raw.values())).shape[0]
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        batch = {
            m: torch.tensor(vicco_raw[m][start:end], device=device, dtype=torch.float32)
            for m in encoded_batches
        }
        encoded = eval_utils.encode_batch_for_all_encodings(
            batch,
            {encoding_name: params[encoding_name]},
        )
        for model in encoded_batches:
            encoded_batches[model].append(encoded[encoding_name][model].detach().cpu().numpy())
        del batch, encoded
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {m: np.concatenate(parts, axis=0).astype(np.float32) for m, parts in encoded_batches.items()}


def compute_track(
    payload: dict,
    track: dict,
    device: torch.device,
    selected_features: dict[str, torch.Tensor],
    baseline_features: dict[str, np.ndarray],
    subsamples: list[np.ndarray],
    noise_level_multipliers: np.ndarray,
    n_noise_samples: int,
    metric: str,
    corr_type: str,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, dict, dict]:
    track_name = track["name"]
    model_names = [m for m in payload["model_names"] if m in selected_features and m in baseline_features]
    selected_features = {m: selected_features[m].to(device=device, dtype=torch.float32) for m in model_names}
    n_selected = next(iter(selected_features.values())).shape[0]
    target_nc = payload.get("config", {}).get("noise_ceiling_target", 0.46)

    precomputed_noise = eval_utils.get_track_noise_variances(payload, track_name)
    if precomputed_noise is not None:
        noise_stds = torch.stack(
            [torch.tensor(precomputed_noise[m], device=device, dtype=torch.float32).sqrt() for m in model_names]
        ).unsqueeze(1)
        noise_info = {m: float(noise_stds[i].item()) for i, m in enumerate(model_names)}
    else:
        baseline_tensors = {
            m: torch.tensor(baseline_features[m], device=device, dtype=torch.float32)
            for m in model_names
        }
        noise_params = disc.calibrate_noise_parameters(
            features=baseline_tensors,
            model_names=model_names,
            metrics=[metric],
            target_nc=target_nc,
            device=device,
            mode="analytical",
        )
        noise_stds = noise_params.get_noise_stds(metric)
        noise_info = {m: float(noise_stds[i].item()) for i, m in enumerate(model_names)}
        del baseline_tensors

    selected_rdms = compute_all_rdms(selected_features, [metric])
    selected_corr = compute_clean_correlation_matrix(selected_rdms[metric], corr_type)
    selected_corr_noised = compute_correlation_at_target_noise(
        selected_rdms[metric],
        noise_stds,
        corr_type,
        n_noise_samples,
    )

    random_rdms = []
    for idx in subsamples:
        baseline_subset = {
            m: torch.tensor(baseline_features[m][idx], device=device, dtype=torch.float32)
            for m in model_names
        }
        random_rdms.append(compute_all_rdms(baseline_subset, [metric]))
        del baseline_subset

    selected_seed = (hash(track_name) & 0xFFFFFFFF) ^ 0xC57A1EE
    selected_discrim, selected_boot = disc.compute_discriminability_by_noise_level_with_bootstrap(
        rdms=selected_rdms[metric],
        noise_stds=noise_stds,
        n_noise_samples=n_noise_samples,
        noise_level_multipliers=noise_level_multipliers,
        corr_type=corr_type,
        n_bootstrap=n_bootstrap,
        seed=selected_seed,
    )

    n_levels = len(noise_level_multipliers)
    random_boot_all = np.empty((len(random_rdms), n_levels, n_bootstrap), dtype=np.float64)
    random_rows = []
    for subset_idx, random_rdm in enumerate(tqdm(random_rdms, desc=f"{track_name} baseline subsets", leave=False)):
        subset_discrim, subset_boot = disc.compute_discriminability_by_noise_level_with_bootstrap(
            rdms=random_rdm[metric],
            noise_stds=noise_stds,
            n_noise_samples=n_noise_samples,
            noise_level_multipliers=noise_level_multipliers,
            corr_type=corr_type,
            n_bootstrap=n_bootstrap,
            seed=selected_seed + 1 + subset_idx,
        )
        random_boot_all[subset_idx] = subset_boot
        for level_idx, level_results in enumerate(subset_discrim):
            random_rows.append(
                {
                    "subset_idx": subset_idx,
                    "noise_mult": noise_level_multipliers[level_idx],
                    "error_prob": float(level_results.get("non_parametric_multiclass_error_prob", 0)),
                    "error_prob_mc_std": float(subset_boot[level_idx].std(ddof=1)),
                    "error_prob_mc_ci_lo": float(np.quantile(subset_boot[level_idx], 0.025)),
                    "error_prob_mc_ci_hi": float(np.quantile(subset_boot[level_idx], 0.975)),
                }
            )

    discrim_rows = []
    for level_idx, level_results in enumerate(selected_discrim):
        noise_mult = noise_level_multipliers[level_idx]
        discrim_rows.append(
            {
                "track": track_name,
                "track_type": track.get("type", "identity"),
                "metric": metric,
                "corr_type": corr_type,
                "noise_mult": noise_mult,
                "noise_ceiling": disc.multiplier_to_noise_ceiling(noise_mult, target_nc),
                "subset_type": "selected",
                "error_prob": float(level_results.get("non_parametric_multiclass_error_prob", 0)),
                "error_prob_std": np.nan,
                "error_prob_mc_std": float(selected_boot[level_idx].std(ddof=1)),
                "error_prob_mc_ci_lo": float(np.quantile(selected_boot[level_idx], 0.025)),
                "error_prob_mc_ci_hi": float(np.quantile(selected_boot[level_idx], 0.975)),
                "mean_offdiag_corr": np.nan,
            }
        )

    random_df = pd.DataFrame(random_rows)
    for noise_mult, group in random_df.groupby("noise_mult"):
        discrim_rows.append(
            {
                "track": track_name,
                "track_type": track.get("type", "identity"),
                "metric": metric,
                "corr_type": corr_type,
                "noise_mult": noise_mult,
                "noise_ceiling": disc.multiplier_to_noise_ceiling(noise_mult, target_nc),
                "subset_type": "random",
                "error_prob": float(group["error_prob"].mean()),
                "error_prob_std": float(group["error_prob"].std(ddof=1)),
                "error_prob_mc_std": float(group["error_prob_mc_std"].mean()),
                "error_prob_mc_ci_lo": float(group["error_prob_mc_ci_lo"].mean()),
                "error_prob_mc_ci_hi": float(group["error_prob_mc_ci_hi"].mean()),
                "mean_offdiag_corr": np.nan,
            }
        )

    discrim_df = pd.DataFrame(discrim_rows)
    selected_data = discrim_df[discrim_df["subset_type"] == "selected"]
    random_data = discrim_df[discrim_df["subset_type"] == "random"]
    selected_auc = disc.compute_auc(selected_data["noise_mult"].values, selected_data["error_prob"].values)
    random_auc = disc.compute_auc(random_data["noise_mult"].values, random_data["error_prob"].values)
    discrim_df["auc"] = discrim_df["subset_type"].map({"selected": selected_auc, "random": random_auc})

    x_levels = np.array(noise_level_multipliers, dtype=float)
    sort_idx = np.argsort(x_levels)
    x_sorted = x_levels[sort_idx]

    def auc_from_curve(y: np.ndarray) -> float:
        return disc.compute_auc(x_sorted, y[sort_idx])

    selected_auc_boot = np.array([auc_from_curve(selected_boot[:, b]) for b in range(selected_boot.shape[1])])
    random_per_subset_auc = np.array(
        [
            auc_from_curve(
                random_df[random_df["subset_idx"] == subset_idx].sort_values("noise_mult")["error_prob"].values
            )
            for subset_idx in sorted(random_df["subset_idx"].unique())
        ]
    )
    random_auc_boot = np.array(
        [auc_from_curve(random_boot_all[:, :, b].mean(axis=0)) for b in range(random_boot_all.shape[2])]
    )
    n_le = int(np.sum(random_per_subset_auc <= selected_auc))
    auc_info = {
        "selected_auc": float(selected_auc),
        "selected_auc_mc_std": float(selected_auc_boot.std(ddof=1)),
        "selected_auc_mc_ci_lo": float(np.quantile(selected_auc_boot, 0.025)),
        "selected_auc_mc_ci_hi": float(np.quantile(selected_auc_boot, 0.975)),
        "random_auc_mean": float(random_per_subset_auc.mean()),
        "random_auc_subset_std": float(random_per_subset_auc.std(ddof=1)),
        "random_auc_subset_ci_lo": float(np.quantile(random_per_subset_auc, 0.025)),
        "random_auc_subset_ci_hi": float(np.quantile(random_per_subset_auc, 0.975)),
        "random_auc_mc_std": float(random_auc_boot.std(ddof=1)),
        "p_value_empirical": float((n_le + 1) / (len(random_per_subset_auc) + 1)),
        "z_score": float(
            (selected_auc - random_per_subset_auc.mean()) / random_per_subset_auc.std(ddof=1)
        )
        if random_per_subset_auc.std(ddof=1) > 0
        else np.nan,
    }
    correlation_info = {
        "model_names": model_names,
        "selected_clean": selected_corr.cpu().numpy().tolist(),
        "selected_noised": selected_corr_noised.cpu().numpy().tolist(),
    }
    return discrim_df, correlation_info, noise_info, auc_info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-sets", default=",".join(MODEL_SET_ORDER))
    parser.add_argument("--tracks", default="raw," + ",".join(ENCODING_TRACKS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-baseline-subsets", type=int, default=20)
    parser.add_argument("--n-noise-samples", type=int, default=100)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--n-baseline", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--which-selection", choices=["final", "greedy", "best_raw_combined"], default="final")
    parser.add_argument("--unique-encodings", action="store_true", default=True)
    parser.add_argument("--shared-encodings", action="store_false", dest="unique_encodings")
    parser.add_argument("--metric", default=None)
    parser.add_argument("--corr-type", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    model_sets = [m.strip() for m in args.model_sets.split(",") if m.strip()]
    requested_tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
    RESULTS.mkdir(parents=True, exist_ok=True)

    for model_set in tqdm(model_sets, desc="Model sets"):
        payload = load_selection_payload(model_set)
        config_payload = payload.get("config", {})
        metric = args.metric or config_payload.get("metric", "cosine")
        corr_type = args.corr_type or config_payload.get("corr_type", "spearman")
        tracks = [t for t in eval_utils.get_all_tracks_for_evaluation(payload) if t["name"] in requested_tracks]
        model_names = payload["model_names"]
        vicco_raw = load_vicco_features(model_names)
        missing = sorted(set(model_names) - set(vicco_raw))
        if missing:
            print(f"[{model_set}] missing VICCO features for {len(missing)} models")
        n_total = next(iter(vicco_raw.values())).shape[0]
        subsamples = make_subsamples(n_total, args.n_baseline, args.n_baseline_subsets, args.seed)
        noise_mults = disc.get_default_noise_level_multipliers()
        encoding_cache: dict[str, Any] = {}

        out_dir = RESULTS / f"{model_set}_baseline_boot"
        out_dir.mkdir(parents=True, exist_ok=True)
        all_discrim, all_auc, all_noise, corr_rows = [], [], [], []
        for track in tracks:
            print(f"[{model_set}] track={track['name']}")
            selected = selected_for_track(
                payload,
                track,
                device,
                args.which_selection,
                encoding_cache,
                args.unique_encodings,
            )
            baseline = baseline_arrays_for_track(
                payload,
                track,
                vicco_raw,
                device,
                encoding_cache,
                args.unique_encodings,
                args.batch_size,
            )
            disc_df, corr_info, noise_info, auc_info = compute_track(
                payload,
                track,
                device,
                selected,
                baseline,
                subsamples,
                noise_mults,
                args.n_noise_samples,
                metric,
                corr_type,
                args.n_bootstrap,
            )
            all_discrim.append(disc_df)
            all_auc.append({"track": track["name"], **auc_info})
            for model, noise_std in noise_info.items():
                all_noise.append({"track": track["name"], "model": model, "noise_std": noise_std})
            for matrix_type in ["selected_clean", "selected_noised"]:
                matrix = corr_info[matrix_type]
                for i, mi in enumerate(corr_info["model_names"]):
                    for j, mj in enumerate(corr_info["model_names"]):
                        corr_rows.append(
                            {
                                "track": track["name"],
                                "matrix_type": matrix_type,
                                "model_i": mi,
                                "model_j": mj,
                                "correlation": matrix[i][j],
                            }
                        )
            del selected, baseline
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        pd.concat(all_discrim, ignore_index=True).to_csv(out_dir / "discriminability.csv", index=False)
        pd.DataFrame(all_auc).to_csv(out_dir / "auc_significance.csv", index=False)
        pd.DataFrame(all_noise).to_csv(out_dir / "noise_calibration.csv", index=False)
        pd.DataFrame(corr_rows).to_csv(out_dir / "correlation_matrices.csv", index=False)
        print(f"[{model_set}] saved {out_dir}")


if __name__ == "__main__":
    main()
