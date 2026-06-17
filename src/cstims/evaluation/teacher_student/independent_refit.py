#!/usr/bin/env python3
"""Independent-refit teacher/student recovery for noisy-by-clean model sets.

This uses an independent natural-image refit pool to fit candidate readouts, then
evaluates model recovery on selected and random held-out stimulus sets.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cstims.evaluation.teacher_student import recovery as base


SCRIPT = Path(__file__).resolve()

DEFAULT_RANDOM_FEATURE_DIR = (
    base.ROOT / "shared" / "cache_or_heavy" / "natural_pool_subset_100k_seed42"
)
DATA_SUFFIX = "_teacher_student_independent_refit_1k"


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def standardize_from_train(
    train: np.ndarray,
    *others: np.ndarray,
    scale_by_sqrt_features: bool = False,
) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    out = [(train - mean) / scale]
    out.extend((arr - mean) / scale for arr in others)
    if scale_by_sqrt_features:
        denom = math.sqrt(train.shape[1])
        out = [arr / denom for arr in out]
    return tuple(np.asarray(arr, dtype=np.float32) for arr in out)


def flat_corr(pred: np.ndarray, target: np.ndarray) -> float:
    x = pred.reshape(-1).astype(np.float64)
    y = target.reshape(-1).astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if denom <= 0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def ridge_ops_for_eval_sets(
    x_train: np.ndarray,
    x_val: np.ndarray,
    eval_sets: dict[str, np.ndarray],
    alphas: list[float],
) -> dict[float, tuple[np.ndarray, dict[str, np.ndarray]]]:
    kernel = (x_train @ x_train.T).astype(np.float64)
    k_val = (x_val @ x_train.T).astype(np.float64)
    eye = np.eye(kernel.shape[0], dtype=np.float64)
    out = {}
    for alpha in alphas:
        inv = np.linalg.inv(kernel + float(alpha) * eye)
        eval_ops = {
            key: np.asarray((x_eval @ x_train.T).astype(np.float64) @ inv, dtype=np.float32)
            for key, x_eval in eval_sets.items()
        }
        out[float(alpha)] = (np.asarray(k_val @ inv, dtype=np.float32), eval_ops)
    return out


def init_stats() -> dict[str, Any]:
    return {
        "correct_sum": 0.0,
        "n_units": 0,
        "pairwise_dominance_sum": 0.0,
        "pairwise_margin_sum": 0.0,
        "n_pairwise": 0,
    }


def metric_row_from_stats(
    *,
    model_set: str,
    track: dict,
    subset_type: str,
    subset_idx: int,
    noise_mult: float,
    base_noise_ceiling: float,
    stats: dict[str, Any],
    n_models: int,
    n_noise_samples: int,
    refit_pool_size: int,
    refit_train_n: int,
    refit_val_n: int,
) -> dict[str, Any]:
    n_units = max(int(stats["n_units"]), 1)
    n_pairwise = max(int(stats["n_pairwise"]), 1)
    recovery_accuracy = float(stats["correct_sum"] / n_units)
    error_prob = 1.0 - recovery_accuracy
    err_std = math.sqrt(error_prob * (1.0 - error_prob) / n_units)
    pairwise_dominance = float(stats["pairwise_dominance_sum"] / n_pairwise)
    return {
        "model_set": model_set,
        "recovery_orientation": "teacher_student_independent_refit",
        "track": track["name"],
        "track_type": track.get("type", "identity"),
        "metric": "fitted_prediction_independent_refit",
        "corr_type": "pearson_flat",
        "noise_mult": float(noise_mult),
        "noise_ceiling": base.multiplier_to_noise_ceiling(noise_mult, base_noise_ceiling),
        "subset_type": subset_type,
        "subset_idx": subset_idx,
        "recovery_accuracy": recovery_accuracy,
        "error_prob": error_prob,
        "error_prob_std": err_std,
        "error_prob_mc_std": err_std,
        "error_prob_mc_ci_lo": max(0.0, error_prob - 1.96 * err_std),
        "error_prob_mc_ci_hi": min(1.0, error_prob + 1.96 * err_std),
        "pairwise_dominance": pairwise_dominance,
        "pairwise_error_prob": 1.0 - pairwise_dominance,
        "mean_margin": float(stats["pairwise_margin_sum"] / n_pairwise),
        "n_units": n_units,
        "n_pairwise": n_pairwise,
        "n_models": n_models,
        "n_splits": 1,
        "n_noise_samples": n_noise_samples,
        "base_noise_ceiling": base_noise_ceiling,
        "refit_pool_size": refit_pool_size,
        "refit_train_n": refit_train_n,
        "refit_val_n": refit_val_n,
    }


def run_track(
    *,
    model_set: str,
    track: dict,
    selected_raw: dict[str, np.ndarray],
    selected_target: dict[str, np.ndarray],
    random_raw_union: dict[str, np.ndarray],
    random_target_union: dict[str, np.ndarray],
    refit_positions: np.ndarray,
    random_subset_positions: list[np.ndarray],
    model_names: list[str],
    alphas: list[float],
    noise_mults: np.ndarray,
    n_noise_samples: int,
    refit_train_n: int,
    refit_val_n: int,
    base_noise_ceiling: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(refit_positions))
    train_pos = refit_positions[perm[:refit_train_n]]
    val_pos = refit_positions[perm[refit_train_n : refit_train_n + refit_val_n]]
    if len(val_pos) != refit_val_n:
        raise ValueError("Refit validation split is shorter than requested")

    eval_raw: dict[str, dict[str, np.ndarray]] = {"selected|0": selected_raw}
    eval_target: dict[str, dict[str, np.ndarray]] = {"selected|0": selected_target}
    eval_meta: dict[str, tuple[str, int]] = {"selected|0": ("selected", 0)}
    for subset_idx, pos in enumerate(random_subset_positions):
        key = f"random|{subset_idx}"
        eval_raw[key] = {model: random_raw_union[model][pos] for model in model_names}
        eval_target[key] = {model: random_target_union[model][pos] for model in model_names}
        eval_meta[key] = ("random", subset_idx)

    candidate_ops = {}
    for candidate in model_names:
        x = random_raw_union[candidate]
        eval_x = {key: raw_by_model[candidate] for key, raw_by_model in eval_raw.items()}
        standardized = standardize_from_train(
            x[train_pos],
            x[val_pos],
            *eval_x.values(),
            scale_by_sqrt_features=True,
        )
        x_train = standardized[0]
        x_val = standardized[1]
        eval_x_std = dict(zip(eval_x.keys(), standardized[2:]))
        candidate_ops[candidate] = ridge_ops_for_eval_sets(
            x_train,
            x_val,
            eval_x_std,
            alphas,
        )

    stats = {
        (key, float(noise_mult)): init_stats()
        for key in eval_raw
        for noise_mult in noise_mults
    }
    confusion = {
        (key, float(noise_mult)): np.zeros(
            (len(model_names), len(model_names)), dtype=np.int64
        )
        for key in eval_raw
        for noise_mult in noise_mults
    }
    recovery_rows: list[dict[str, Any]] = []

    for teacher_idx, teacher in enumerate(model_names):
        clean_y = random_target_union[teacher]
        eval_y = {key: target_by_model[teacher] for key, target_by_model in eval_target.items()}
        standardized_y = standardize_from_train(
            clean_y[train_pos],
            clean_y[val_pos],
            *eval_y.values(),
        )
        y_train_clean = standardized_y[0]
        y_val_clean = standardized_y[1]
        eval_y_clean = dict(zip(eval_y.keys(), standardized_y[2:]))
        offdiag = np.ones(len(model_names), dtype=bool)
        offdiag[teacher_idx] = False

        for noise_mult in noise_mults:
            noise_mult = float(noise_mult)
            noise_std = base.noise_std_from_multiplier(noise_mult, base_noise_ceiling)
            noise_ceiling = base.multiplier_to_noise_ceiling(noise_mult, base_noise_ceiling)
            for noise_sample_idx in range(n_noise_samples):
                y_train = y_train_clean + rng.normal(
                    0.0,
                    noise_std,
                    y_train_clean.shape,
                ).astype(np.float32)
                y_val = y_val_clean + rng.normal(
                    0.0,
                    noise_std,
                    y_val_clean.shape,
                ).astype(np.float32)
                eval_y_noisy = {
                    key: y_clean + rng.normal(0.0, noise_std, y_clean.shape).astype(np.float32)
                    for key, y_clean in eval_y_clean.items()
                }

                scores_by_eval = {
                    key: np.full(len(model_names), np.nan, dtype=np.float32)
                    for key in eval_raw
                }
                best_alpha_by_candidate = {}
                for candidate_idx, candidate in enumerate(model_names):
                    best_alpha = float(alphas[0])
                    best_val_score = -np.inf
                    for alpha, (val_op, _) in candidate_ops[candidate].items():
                        val_score = flat_corr(val_op @ y_train, y_val)
                        if np.isfinite(val_score) and val_score > best_val_score:
                            best_alpha = float(alpha)
                            best_val_score = val_score
                    best_alpha_by_candidate[candidate] = best_alpha
                    _, eval_ops = candidate_ops[candidate][best_alpha]
                    for key, eval_op in eval_ops.items():
                        scores_by_eval[key][candidate_idx] = flat_corr(
                            eval_op @ y_train,
                            eval_y_noisy[key],
                        )

                for key, scores in scores_by_eval.items():
                    scores = np.nan_to_num(scores, nan=-np.inf)
                    recovered_idx = int(np.argmax(scores))
                    subset_type, subset_idx = eval_meta[key]
                    current = stats[(key, noise_mult)]
                    current["correct_sum"] += float(recovered_idx == teacher_idx)
                    current["n_units"] += 1
                    margins = scores[teacher_idx] - scores[offdiag]
                    dominance = (
                        (margins > 0).astype(np.float64)
                        + 0.5 * np.isclose(margins, 0.0).astype(np.float64)
                    )
                    current["pairwise_dominance_sum"] += float(dominance.sum())
                    current["pairwise_margin_sum"] += float(margins.sum())
                    current["n_pairwise"] += int(margins.size)
                    confusion[(key, noise_mult)][teacher_idx, recovered_idx] += 1
                    recovery_rows.append(
                        {
                            "model_set": model_set,
                            "track": track["name"],
                            "track_type": track.get("type", "identity"),
                            "subset_type": subset_type,
                            "subset_idx": subset_idx,
                            "teacher_model": teacher,
                            "recovered_model": model_names[recovered_idx],
                            "recovered_correct": bool(recovered_idx == teacher_idx),
                            "best_test_score": float(scores[recovered_idx]),
                            "teacher_self_test_score": float(scores[teacher_idx]),
                            "noise_mult": noise_mult,
                            "noise_ceiling": noise_ceiling,
                            "relative_snr": np.inf if noise_mult <= 0 else 1.0 / noise_mult,
                            "noise_sample_idx": noise_sample_idx,
                            "best_alpha_recovered": best_alpha_by_candidate[
                                model_names[recovered_idx]
                            ],
                        }
                    )

    subset_rows = []
    pairwise_subset_rows = []
    for (key, noise_mult), current in stats.items():
        subset_type, subset_idx = eval_meta[key]
        row = metric_row_from_stats(
            model_set=model_set,
            track=track,
            subset_type=subset_type,
            subset_idx=subset_idx,
            noise_mult=float(noise_mult),
            base_noise_ceiling=base_noise_ceiling,
            stats=current,
            n_models=len(model_names),
            n_noise_samples=n_noise_samples,
            refit_pool_size=len(refit_positions),
            refit_train_n=refit_train_n,
            refit_val_n=refit_val_n,
        )
        subset_rows.append(row)
        pairwise_subset_rows.append(
            {
                **row,
                "pairwise_dominance_subset_std": np.nan,
                "pairwise_dominance_mc_std": math.sqrt(
                    row["pairwise_dominance"]
                    * (1.0 - row["pairwise_dominance"])
                    / max(row["n_pairwise"], 1)
                ),
                "pairwise_dominance_mc_ci_lo": np.nan,
                "pairwise_dominance_mc_ci_hi": np.nan,
                "mean_margin_subset_std": np.nan,
                "mean_margin_mc_std": np.nan,
                "mean_margin_mc_ci_lo": np.nan,
                "mean_margin_mc_ci_hi": np.nan,
                "random_feature_source": "",
            }
        )

    confusion_rows = []
    for (key, noise_mult), matrix in confusion.items():
        subset_type, subset_idx = eval_meta[key]
        for teacher_idx, teacher in enumerate(model_names):
            total = int(matrix[teacher_idx].sum())
            for recovered_idx, recovered in enumerate(model_names):
                count = int(matrix[teacher_idx, recovered_idx])
                confusion_rows.append(
                    {
                        "model_set": model_set,
                        "track": track["name"],
                        "track_type": track.get("type", "identity"),
                        "subset_type": subset_type,
                        "subset_idx": subset_idx,
                        "noise_mult": float(noise_mult),
                        "noise_ceiling": base.multiplier_to_noise_ceiling(
                            float(noise_mult), base_noise_ceiling
                        ),
                        "teacher_model": teacher,
                        "recovered_model": recovered,
                        "count": count,
                        "proportion": count / total if total else np.nan,
                    }
                )

    return subset_rows, pairwise_subset_rows, recovery_rows + confusion_rows


def run_model_set(
    model_set: str,
    args: argparse.Namespace,
    encoding_root_map: dict[str, Path] | None,
) -> None:
    payload = base.apply_env_paths(
        base.eval_utils.load_selection_payload(args.selection_root / model_set),
        args.env,
    )
    original_models = list(payload["model_names"])
    available_models = base.available_random_models(args.random_feature_dir, original_models)
    missing = sorted(set(original_models) - set(available_models))
    if missing:
        print(
            f"[{model_set}] using {len(available_models)}/{len(original_models)} "
            f"models available in {args.random_feature_dir}; excluded: {missing}"
        )
    if len(available_models) < 2:
        raise RuntimeError(f"{model_set}: need at least two available models")
    payload = base.filter_payload_to_models(payload, available_models)

    config_payload = payload.get("config", {})
    target_nc = float(args.noise_ceiling or config_payload.get("noise_ceiling_target", 0.46))
    roi_subset = args.roi_subset or config_payload.get("encoding_roi_subset", "hlvis")
    tracks = [
        track
        for track in base.eval_utils.get_all_tracks_for_evaluation(payload)
        if track["name"] in args.tracks
    ]
    out_dir = args.output_root / f"{model_set}{DATA_SUFFIX}"
    out_dir.mkdir(parents=True, exist_ok=True)
    random_feature_source = f"local_random_pool:{args.random_feature_dir}"
    pd.DataFrame(
        {
            "model": original_models,
            "included": [model in available_models for model in original_models],
            "reason": [
                "included"
                if model in available_models
                else "excluded_missing_random_pool_feature"
                for model in original_models
            ],
            "random_feature_source": random_feature_source,
        }
    ).to_csv(out_dir / "model_roster.csv", index=False)

    metadata = {
        "model_set": model_set,
        "selection_root": str(args.selection_root),
        "output_dir": str(out_dir),
        "recovery_orientation": "teacher_student_independent_refit",
        "random_feature_source": random_feature_source,
        "tracks": [track["name"] for track in tracks],
        "model_names": available_models,
        "excluded_models": missing,
        "n_random_images": args.n_random_images,
        "refit_pool_size": args.refit_pool_size,
        "refit_train_n": args.refit_pool_size - args.refit_val_size,
        "refit_val_n": args.refit_val_size,
        "n_random_subsets": args.n_random_subsets,
        "n_noise_samples": args.n_noise_samples,
        "alphas": args.alphas,
        "noise_multipliers": args.noise_mults.tolist(),
        "base_noise_ceiling": target_nc,
        "roi_subset": roi_subset,
        "which_selection": args.which_selection,
        "seed": args.seed,
    }
    with (out_dir / "teacher_student_independent_refit_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[{model_set}] loading selected raw features")
    selected_raw = base.selected_raw_from_payload(
        payload, available_models, args.which_selection
    )
    print(f"[{model_set}] loading random raw features")
    random_raw = base.load_random_raw_features(
        payload,
        available_models,
        args.n_random_images,
        args.random_feature_dir,
    )
    n_available = min(arr.shape[0] for arr in random_raw.values())
    n_selected = next(iter(selected_raw.values())).shape[0]
    if args.refit_pool_size + n_selected > n_available:
        raise ValueError(f"{model_set}: not enough random images for refit/eval")
    rng = np.random.default_rng(args.seed + base.stable_seed(model_set, "independent_refit"))
    refit_indices = rng.choice(n_available, size=args.refit_pool_size, replace=False)
    remaining = np.setdiff1d(np.arange(n_available), refit_indices, assume_unique=False)
    random_subset_indices = [
        rng.choice(remaining, size=n_selected, replace=False)
        for _ in range(args.n_random_subsets)
    ]
    union_indices = np.unique(np.concatenate([refit_indices, *random_subset_indices]))
    union_lookup = {int(idx): pos for pos, idx in enumerate(union_indices)}
    refit_positions = np.asarray(
        [union_lookup[int(idx)] for idx in refit_indices], dtype=np.int64
    )
    random_subset_positions = [
        np.asarray([union_lookup[int(idx)] for idx in subset], dtype=np.int64)
        for subset in random_subset_indices
    ]
    random_raw_union = {model: arr[union_indices] for model, arr in random_raw.items()}

    all_subset_rows: list[dict[str, Any]] = []
    all_pairwise_subset_rows: list[dict[str, Any]] = []
    all_recovery_rows: list[dict[str, Any]] = []
    all_confusion_rows: list[dict[str, Any]] = []
    encoding_params_cache: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    selected_encoded_cache: dict[str, dict[str, np.ndarray]] = {}
    random_encoded_cache: dict[str, dict[str, np.ndarray]] = {}
    encoding_device = torch.device(args.encoding_device)

    for track_idx, track in enumerate(tracks):
        track_name = track["name"]
        print(f"\n[{model_set}] track {track_idx + 1}/{len(tracks)}: {track_name}")
        if track.get("type", "identity") == "identity":
            selected_target = selected_raw
            random_target_union = random_raw_union
        else:
            enc_name = track.get("encoding_name") or track_name
            if enc_name not in encoding_params_cache:
                enc_root = (encoding_root_map or {}).get(
                    enc_name,
                    Path(config_payload.get("paths", {}).get("encoding_root", "")),
                )
                print(f"  Loading encoding params from {enc_root}")
                encoding_params_cache[enc_name] = base.load_encoding_params(
                    enc_root,
                    base.MODEL_LIST_CSV,
                    available_models,
                    enc_name,
                    roi_subset=roi_subset,
                )
            if enc_name not in selected_encoded_cache:
                print("  Encoding selected raw features")
                selected_encoded_cache[enc_name] = base.encode_raw_features(
                    selected_raw,
                    encoding_params_cache[enc_name],
                    device=encoding_device,
                    batch_size=args.encoding_batch_size,
                )
            if enc_name not in random_encoded_cache:
                print(f"  Encoding random union ({len(union_indices)} images)")
                random_encoded_cache[enc_name] = base.encode_raw_features(
                    random_raw_union,
                    encoding_params_cache[enc_name],
                    device=encoding_device,
                    batch_size=args.encoding_batch_size,
                )
            selected_target = selected_encoded_cache[enc_name]
            random_target_union = random_encoded_cache[enc_name]

        subset_rows, pairwise_rows, detail_rows = run_track(
            model_set=model_set,
            track=track,
            selected_raw=selected_raw,
            selected_target=selected_target,
            random_raw_union=random_raw_union,
            random_target_union=random_target_union,
            refit_positions=refit_positions,
            random_subset_positions=random_subset_positions,
            model_names=available_models,
            alphas=args.alphas,
            noise_mults=args.noise_mults,
            n_noise_samples=args.n_noise_samples,
            refit_train_n=args.refit_pool_size - args.refit_val_size,
            refit_val_n=args.refit_val_size,
            base_noise_ceiling=target_nc,
            seed=args.seed + base.stable_seed(model_set, track_name, "independent_refit"),
        )
        all_subset_rows.extend(subset_rows)
        all_pairwise_subset_rows.extend(pairwise_rows)
        all_recovery_rows.extend(row for row in detail_rows if "recovered_correct" in row)
        all_confusion_rows.extend(row for row in detail_rows if "recovered_correct" not in row)

        pd.DataFrame(all_subset_rows).to_csv(
            out_dir / "teacher_student_subset_curves.csv", index=False
        )
        pd.DataFrame(all_pairwise_subset_rows).to_csv(
            out_dir / "teacher_student_pairwise_subset_curves.csv", index=False
        )
        pd.DataFrame(all_recovery_rows).to_csv(
            out_dir / "teacher_student_recoveries.csv", index=False
        )
        pd.DataFrame(all_confusion_rows).to_csv(
            out_dir / "teacher_student_confusion_matrix.csv", index=False
        )
        gc.collect()
        if encoding_device.type == "cuda":
            torch.cuda.empty_cache()

    subset_df = pd.DataFrame(all_subset_rows)
    subset_df.to_csv(out_dir / "teacher_student_subset_curves.csv", index=False)
    curves = base.aggregate_curve_rows(subset_df, random_feature_source=random_feature_source)
    curves.to_csv(out_dir / "discriminability.csv", index=False)
    pairwise_subset_df = pd.DataFrame(all_pairwise_subset_rows)
    pairwise_curves = base.aggregate_pairwise_rows(
        pairwise_subset_df,
        random_feature_source=random_feature_source,
    )
    pairwise_curves.to_csv(out_dir / "pairwise_margin.csv", index=False)
    auc_df, pairwise_auc_df = base.auc_rows_from_curves(curves, pairwise_curves)
    auc_df.to_csv(out_dir / "auc_significance.csv", index=False)
    pairwise_auc_df.to_csv(out_dir / "pairwise_auc.csv", index=False)
    pd.DataFrame(all_recovery_rows).to_csv(
        out_dir / "teacher_student_recoveries.csv", index=False
    )
    pd.DataFrame(all_confusion_rows).to_csv(
        out_dir / "teacher_student_confusion_matrix.csv", index=False
    )
    print(f"[{model_set}] saved {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-sets", default=",".join(base.MODEL_SET_ORDER))
    parser.add_argument("--tracks", default="raw," + ",".join(base.ENCODING_TRACKS))
    parser.add_argument("--selection-root", type=Path, default=base.SELECTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=base.DEFAULT_RESULTS)
    parser.add_argument("--env", choices=base.eval_utils.VALID_ENVS, default=None)
    parser.add_argument("--random-feature-dir", type=Path, default=DEFAULT_RANDOM_FEATURE_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-random-images", type=int, default=100000)
    parser.add_argument("--refit-pool-size", type=int, default=1000)
    parser.add_argument("--refit-val-size", type=int, default=200)
    parser.add_argument("--n-random-subsets", type=int, default=20)
    parser.add_argument("--n-noise-samples", type=int, default=1)
    parser.add_argument("--alphas", default="0.001,0.01,0.1,1,10,100")
    parser.add_argument("--noise-ceiling", type=float, default=None)
    parser.add_argument("--noise-mults", default=None)
    parser.add_argument(
        "--which-selection",
        choices=["final", "greedy", "best_raw_combined"],
        default="final",
    )
    parser.add_argument("--unique-encodings", action="store_true", default=True)
    parser.add_argument("--shared-encodings", action="store_false", dest="unique_encodings")
    parser.add_argument("--encoding-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--encoding-batch-size", type=int, default=256)
    parser.add_argument("--roi-subset", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_sets = [item.strip() for item in args.model_sets.split(",") if item.strip()]
    args.tracks = [item.strip() for item in args.tracks.split(",") if item.strip()]
    args.selection_root = args.selection_root.resolve()
    args.output_root = args.output_root.resolve()
    args.random_feature_dir = args.random_feature_dir.resolve()
    args.alphas = parse_float_list(args.alphas)
    if args.noise_mults:
        args.noise_mults = np.asarray(parse_float_list(args.noise_mults), dtype=np.float64)
    else:
        args.noise_mults = np.asarray(base.disc.get_default_noise_level_multipliers(), dtype=np.float64)
    if not args.random_feature_dir.exists():
        raise FileNotFoundError(f"Random feature directory not found: {args.random_feature_dir}")
    if args.encoding_device == "cuda" and not torch.cuda.is_available():
        args.encoding_device = "cpu"
    if args.encoding_device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    encoding_root_map = None
    if args.unique_encodings:
        encoding_root_map = {
            key: Path(value) for key, value in base.paper_config.UNIQUE_ENCODING_DIRS.items()
        }
        print(f"Using unique encoding roots: {list(encoding_root_map)}")
    print(f"Using random-feature cache: {args.random_feature_dir}")
    for model_set in tqdm(args.model_sets, desc="Model sets"):
        run_model_set(model_set, args, encoding_root_map)


if __name__ == "__main__":
    main()
