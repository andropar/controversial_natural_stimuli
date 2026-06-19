#!/usr/bin/env python3
"""Profile individual steps in teacher/student RDM recovery.

The goal is not to replace a full cProfile run.  This script times the named
computational blocks that make up the production recovery script, using the
same package functions and data paths wherever possible.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPT = Path(__file__).resolve()
ROOT = next(p for p in SCRIPT.parents if (p / "src" / "cstims").exists())
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cstims import paths  # noqa: E402
from cstims.evaluation.constants import get_default_noise_level_multipliers  # noqa: E402
from cstims.evaluation.io import load_payload  # noqa: E402
from cstims.evaluation.noise_calibration import (  # noqa: E402
    calibrate_response_noise_for_rdm_reliability,
    multiplier_to_noise_ceiling,
    rdm_noise_std_from_clean,
    response_noise_std_from_mode,
)
from cstims.evaluation.payload import filter_payload_to_models  # noqa: E402
from cstims.evaluation.random_features import (  # noqa: E402
    available_random_models,
    load_random_feature_cache,
)
from cstims.evaluation.ridge import (  # noqa: E402
    IndependentRidgeOps,
    build_independent_ridge_ops,
    ridge_eval_augmented_loo_ops,
    ridge_ops_for_eval_sets,
    standardize_from_train,
)
from cstims.evaluation.teacher_student import (  # noqa: E402
    build_eval_raw_and_meta,
    detect_equivalent_models,
    eval_keys_for_ops,
    load_encoding_params_for_models,
    parse_csv_list,
    parse_float_list,
    parse_index_list,
    predict_eval_with_targetwise_alphas,
    select_targetwise_alpha_indices,
    stable_seed,
)
from cstims.evaluation.track_loading import (  # noqa: E402
    encode_raw_feature_arrays,
    load_selected_raw_features,
)
from cstims.rdm import calculate_correlation_value, get_rdm_vector_np  # noqa: E402


@dataclass
class TimingRecord:
    label: str
    seconds: float
    track: str
    teacher: str
    candidate: str
    noise_mult: float | None
    noise_sample_idx: int | None
    detail: str


class StepProfiler:
    def __init__(self) -> None:
        self.records: list[TimingRecord] = []

    @contextmanager
    def timed(
        self,
        label: str,
        *,
        track: str = "",
        teacher: str = "",
        candidate: str = "",
        noise_mult: float | None = None,
        noise_sample_idx: int | None = None,
        detail: str = "",
    ):
        start = perf_counter()
        try:
            yield
        finally:
            self.records.append(
                TimingRecord(
                    label=label,
                    seconds=perf_counter() - start,
                    track=track,
                    teacher=teacher,
                    candidate=candidate,
                    noise_mult=noise_mult,
                    noise_sample_idx=noise_sample_idx,
                    detail=detail,
                )
            )

    def detail_frame(self) -> pd.DataFrame:
        return pd.DataFrame([record.__dict__ for record in self.records])

    def summary_frame(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame()
        grouped: dict[str, list[float]] = defaultdict(list)
        for record in self.records:
            grouped[record.label].append(record.seconds)
        rows = []
        for label, vals in grouped.items():
            arr = np.asarray(vals, dtype=np.float64)
            rows.append(
                {
                    "function": label,
                    "calls": int(arr.size),
                    "completion_time_s": float(arr.sum()),
                    "mean_call_s": float(arr.mean()),
                    "max_call_s": float(arr.max()),
                }
            )
        return pd.DataFrame(rows).sort_values("completion_time_s", ascending=False)


def print_markdown_table(df: pd.DataFrame, *, max_rows: int | None = None) -> None:
    if df.empty:
        print("No timings recorded.")
        return
    if max_rows is not None:
        df = df.head(max_rows)
    print("| function | calls | completion time (s) | mean call (s) | max call (s) |")
    print("|---|---:|---:|---:|---:|")
    for row in df.itertuples(index=False):
        print(
            f"| {row.function} | {row.calls} | "
            f"{row.completion_time_s:.3f} | {row.mean_call_s:.4f} | {row.max_call_s:.4f} |"
        )


def parse_tracks(value: str) -> list[dict[str, str]]:
    tracks = []
    for track_name in parse_csv_list(value):
        if track_name == "raw":
            tracks.append({"name": "raw", "type": "identity"})
        else:
            tracks.append(
                {"name": track_name, "type": "encoding", "encoding_name": track_name}
            )
    return tracks


def profile_build_candidate_ops(
    *,
    profiler: StepProfiler,
    random_raw_union: dict[str, np.ndarray],
    eval_raw: dict[str, dict[str, np.ndarray]],
    train_pos: np.ndarray,
    val_pos: np.ndarray,
    base_fit_pos: np.ndarray | None,
    model_names: list[str],
    alphas: list[float],
    eval_refit_mode: str,
) -> dict[
    str,
    IndependentRidgeOps | dict[float, tuple[np.ndarray, dict[str, np.ndarray]]],
]:
    candidate_ops = {}
    for candidate_idx, candidate in enumerate(model_names):
        print(
            f"candidate ridge ops {candidate_idx + 1}/{len(model_names)}: {candidate}",
            flush=True,
        )
        x = random_raw_union[candidate]
        eval_x = {key: raw_by_model[candidate] for key, raw_by_model in eval_raw.items()}
        if eval_refit_mode == "independent":
            with profiler.timed(
                "candidate_features_standardize",
                candidate=candidate,
            ):
                standardized = standardize_from_train(
                    x[train_pos],
                    x[val_pos],
                    *eval_x.values(),
                    scale_by_sqrt_features=True,
                )
            eval_x_std = dict(zip(eval_x.keys(), standardized[2:]))
            with profiler.timed(
                "candidate_ridge_factorization",
                candidate=candidate,
                detail=f"n_train={standardized[0].shape[0]},n_features={standardized[0].shape[1]}",
            ):
                candidate_ops[candidate] = build_independent_ridge_ops(
                    standardized[0],
                    standardized[1],
                    eval_x_std,
                    alphas,
                )
        elif eval_refit_mode == "eval_augmented_loo":
            if base_fit_pos is None:
                raise ValueError("base_fit_pos is required for eval_augmented_loo")
            with profiler.timed(
                "candidate_features_standardize",
                candidate=candidate,
            ):
                standardized = standardize_from_train(
                    x[train_pos],
                    x[val_pos],
                    x[base_fit_pos],
                    *eval_x.values(),
                    scale_by_sqrt_features=True,
                )
            x_train = standardized[0]
            x_val = standardized[1]
            x_base = standardized[2]
            eval_x_std = dict(zip(eval_x.keys(), standardized[3:]))
            with profiler.timed("candidate_validation_ridge_ops", candidate=candidate):
                val_ops = ridge_ops_for_eval_sets(x_train, x_val, {}, alphas)
            with profiler.timed("candidate_eval_augmented_loo_ops", candidate=candidate):
                loo_ops = ridge_eval_augmented_loo_ops(x_base, eval_x_std, alphas)
            candidate_ops[candidate] = {
                float(alpha): (val_ops[float(alpha)][0], loo_ops[float(alpha)])
                for alpha in alphas
            }
        else:
            raise ValueError(f"Unsupported eval_refit_mode: {eval_refit_mode}")
    return candidate_ops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-set", default="all_models")
    parser.add_argument("--selection-root", type=Path, default=paths.selected_stimuli_root())
    parser.add_argument(
        "--random-feature-dir",
        type=Path,
        default=paths.find_share_root()
        / "shared"
        / "cache_or_heavy"
        / "natural_pool_subset_100k_seed42",
    )
    parser.add_argument("--encoding-root", type=Path, default=paths.shared_encoding_root())
    parser.add_argument("--tracks", default="raw,sub-01")
    parser.add_argument("--n-random-images", type=int, default=100000)
    parser.add_argument("--refit-pool-size", type=int, default=100)
    parser.add_argument("--refit-val-size", type=int, default=20)
    parser.add_argument("--max-refit-pool-size", type=int, default=10000)
    parser.add_argument("--n-random-subsets", type=int, default=100)
    parser.add_argument("--n-noise-samples", type=int, default=1)
    parser.add_argument("--noise-mults", default="1")
    parser.add_argument("--noise-ceiling", type=float, default=None)
    parser.add_argument("--alphas", default="0.001,0.01,0.1,1,10,100")
    parser.add_argument("--metric", default=None)
    parser.add_argument("--corr-type", choices=["pearson", "spearman"], default="spearman")
    parser.add_argument("--eval-noise-mode", choices=["rdm", "response"], default="response")
    parser.add_argument(
        "--fit-noise-calibration",
        choices=["response", "rdm_analytic", "rdm_empirical"],
        default="rdm_empirical",
    )
    parser.add_argument(
        "--eval-refit-mode",
        choices=["independent", "eval_augmented_loo"],
        default="independent",
    )
    parser.add_argument("--calibration-images", type=int, default=100)
    parser.add_argument("--calibration-noise-samples", type=int, default=2)
    parser.add_argument("--calibration-max-iter", type=int, default=8)
    parser.add_argument("--target-dim", type=int, default=None)
    parser.add_argument("--teacher-indices", default="0")
    parser.add_argument("--encoding-device", default="cuda")
    parser.add_argument("--encoding-batch-size", type=int, default=1024)
    parser.add_argument("--unique-encodings", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT.parent / "profile_reports",
    )
    parser.add_argument("--summary-rows", type=int, default=50)
    args = parser.parse_args()

    profiler = StepProfiler()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with profiler.timed("load_payload_and_filter_models"):
        payload = load_payload(args.selection_root / args.model_set)
        model_names = list(payload["model_names"])
        available = available_random_models(args.random_feature_dir, model_names)
        model_names = available
        payload = filter_payload_to_models(payload, model_names)
        metric = args.metric or payload.get("config", {}).get("metric", "cosine")
        base_noise_ceiling = float(
            args.noise_ceiling
            or payload.get("config", {}).get("noise_ceiling_target", 0.46)
        )

    teacher_indices = parse_index_list(args.teacher_indices, len(model_names))
    if teacher_indices is None:
        teacher_idx_list = list(range(len(model_names)))
    else:
        teacher_idx_list = sorted(teacher_indices)
    tracks = parse_tracks(args.tracks)
    alphas = [float(x) for x in parse_csv_list(args.alphas)]
    noise_mults = (
        np.asarray(parse_float_list(args.noise_mults), dtype=np.float64)
        if args.noise_mults
        else np.asarray(get_default_noise_level_multipliers(), dtype=np.float64)
    )

    with profiler.timed("load_selected_raw_features"):
        selected_raw = load_selected_raw_features(
            payload,
            model_names=model_names,
            selection_variant="final",
        )

    with profiler.timed("load_random_feature_cache"):
        random_raw = load_random_feature_cache(
            random_feature_dir=args.random_feature_dir,
            model_names=model_names,
            n_random=args.n_random_images,
            view_name="raw",
        )

    n_available = min(arr.shape[0] for arr in random_raw.values())
    n_selected = next(iter(selected_raw.values())).shape[0]
    max_refit_pool_size = int(args.max_refit_pool_size or args.refit_pool_size)
    if max_refit_pool_size < args.refit_pool_size:
        raise ValueError("--max-refit-pool-size cannot be smaller than --refit-pool-size")
    if max_refit_pool_size + args.n_random_subsets * n_selected > n_available:
        raise ValueError("Not enough random images for disjoint refit and random eval pools")

    with profiler.timed("sample_refit_and_random_eval_indices"):
        repeat_rng = np.random.default_rng(
            args.seed + stable_seed(args.model_set, "profile", "refit_repeat", 0)
        )
        natural_pool_order = repeat_rng.permutation(n_available)
        refit_indices = natural_pool_order[: args.refit_pool_size]
        random_eval_pool = natural_pool_order[
            max_refit_pool_size : max_refit_pool_size
            + args.n_random_subsets * n_selected
        ]
        random_subset_indices = [
            random_eval_pool[
                subset_idx * n_selected : (subset_idx + 1) * n_selected
            ]
            for subset_idx in range(args.n_random_subsets)
        ]
        union_indices = np.unique(np.concatenate([refit_indices, *random_subset_indices]))
        union_lookup = {int(idx): pos for pos, idx in enumerate(union_indices)}
        refit_positions = np.asarray(
            [union_lookup[int(idx)] for idx in refit_indices],
            dtype=np.int64,
        )
        random_subset_positions = [
            np.asarray([union_lookup[int(idx)] for idx in subset], dtype=np.int64)
            for subset in random_subset_indices
        ]
        split_rng = np.random.default_rng(
            args.seed + stable_seed(args.model_set, "profile", "refit_split")
        )
        refit_perm = split_rng.permutation(len(refit_positions))
        train_pos = refit_positions[refit_perm[: args.refit_pool_size - args.refit_val_size]]
        val_pos = refit_positions[
            refit_perm[
                args.refit_pool_size
                - args.refit_val_size : args.refit_pool_size
            ]
        ]
        base_fit_pos = (
            np.concatenate([train_pos, val_pos])
            if args.eval_refit_mode == "eval_augmented_loo"
            else None
        )

    with profiler.timed("slice_random_feature_union"):
        random_raw_union = {
            model: arr[union_indices] for model, arr in random_raw.items()
        }

    with profiler.timed("build_eval_raw_and_meta"):
        eval_raw, eval_meta = build_eval_raw_and_meta(
            selected_raw=selected_raw,
            random_raw_union=random_raw_union,
            random_subset_positions=random_subset_positions,
            model_names=model_names,
        )

    candidate_ops = profile_build_candidate_ops(
        profiler=profiler,
        random_raw_union=random_raw_union,
        eval_raw=eval_raw,
        train_pos=train_pos,
        val_pos=val_pos,
        base_fit_pos=base_fit_pos,
        model_names=model_names,
        alphas=alphas,
        eval_refit_mode=args.eval_refit_mode,
    )

    with profiler.timed("detect_equivalent_models"):
        equivalence_labels = detect_equivalent_models(random_raw_union, model_names)

    encoding_device = torch.device(
        args.encoding_device
        if args.encoding_device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    encoding_root_map = (
        {key: Path(value) for key, value in paths.unique_encoding_dirs().items()}
        if args.unique_encodings
        else None
    )
    enc_cache: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    selected_enc_cache: dict[str, dict[str, np.ndarray]] = {}
    random_enc_cache: dict[str, dict[str, np.ndarray]] = {}

    for track in tracks:
        track_name = track["name"]
        print(f"profile track: {track_name}", flush=True)
        if track.get("type") == "identity":
            with profiler.timed("track_prepare_raw_targets", track=track_name):
                selected_target = selected_raw
                random_target_union = random_raw_union
        else:
            enc_name = track.get("encoding_name") or track_name
            enc_root = (
                encoding_root_map.get(enc_name, args.encoding_root)
                if encoding_root_map is not None
                else args.encoding_root
            )
            if enc_name not in enc_cache:
                with profiler.timed("load_encoding_params", track=track_name):
                    enc_cache[enc_name] = load_encoding_params_for_models(
                        enc_root,
                        model_names,
                        enc_name,
                        roi_subset=payload.get("config", {}).get(
                            "encoding_roi_subset",
                            "hlvis",
                        ),
                        device=encoding_device,
                    )
            if enc_name not in selected_enc_cache:
                with profiler.timed("encode_selected_targets", track=track_name):
                    selected_enc_cache[enc_name] = encode_raw_feature_arrays(
                        selected_raw,
                        enc_name,
                        enc_cache[enc_name],
                        device=encoding_device,
                        batch_size=args.encoding_batch_size,
                    )
            if enc_name not in random_enc_cache:
                with profiler.timed("encode_random_union_targets", track=track_name):
                    random_enc_cache[enc_name] = encode_raw_feature_arrays(
                        random_raw_union,
                        enc_name,
                        enc_cache[enc_name],
                        device=encoding_device,
                        batch_size=args.encoding_batch_size,
                    )
            selected_target = selected_enc_cache[enc_name]
            random_target_union = random_enc_cache[enc_name]

        with profiler.timed("build_eval_target_views", track=track_name):
            eval_target: dict[str, dict[str, np.ndarray]] = {"selected|0": selected_target}
            for subset_idx, pos in enumerate(random_subset_positions):
                key = f"random|{subset_idx}"
                eval_target[key] = {
                    model: random_target_union[model][pos] for model in model_names
                }

        for teacher_idx in teacher_idx_list:
            teacher = model_names[teacher_idx]
            print(f"  profile teacher {teacher_idx + 1}/{len(model_names)}: {teacher}", flush=True)
            teacher_rng = np.random.default_rng(
                args.seed + stable_seed(track_name, teacher, "profile_teacher_noise")
            )

            with profiler.timed(
                "teacher_extract_clean_targets",
                track=track_name,
                teacher=teacher,
            ):
                clean_y = random_target_union[teacher]
                eval_y = {
                    key: target_by_model[teacher]
                    for key, target_by_model in eval_target.items()
                }
                target_cols = None
                if args.target_dim is not None and 0 < args.target_dim < clean_y.shape[1]:
                    target_rng = np.random.default_rng(
                        args.seed + stable_seed(track_name, teacher, "target_cols")
                    )
                    target_cols = np.sort(
                        target_rng.choice(
                            clean_y.shape[1],
                            size=args.target_dim,
                            replace=False,
                        )
                    )
                if target_cols is not None:
                    clean_y = clean_y[:, target_cols]
                    eval_y = {key: y[:, target_cols] for key, y in eval_y.items()}

            with profiler.timed(
                "teacher_target_standardize",
                track=track_name,
                teacher=teacher,
            ):
                if args.eval_refit_mode == "eval_augmented_loo":
                    standardized_y = standardize_from_train(
                        clean_y[train_pos],
                        clean_y[val_pos],
                        clean_y[base_fit_pos],
                        *eval_y.values(),
                    )
                    y_train_clean = standardized_y[0]
                    y_val_clean = standardized_y[1]
                    y_base_fit_clean = standardized_y[2]
                    eval_y_clean = dict(zip(eval_y.keys(), standardized_y[3:]))
                else:
                    standardized_y = standardize_from_train(
                        clean_y[train_pos],
                        clean_y[val_pos],
                        *eval_y.values(),
                    )
                    y_train_clean = standardized_y[0]
                    y_val_clean = standardized_y[1]
                    y_base_fit_clean = None
                    eval_y_clean = dict(zip(eval_y.keys(), standardized_y[2:]))

            with profiler.timed(
                "clean_eval_rdm_all_sets",
                track=track_name,
                teacher=teacher,
                detail=f"n_eval_sets={len(eval_y_clean)}",
            ):
                clean_eval_rdms = {
                    key: get_rdm_vector_np(y_clean, metric)
                    for key, y_clean in eval_y_clean.items()
                }

            with profiler.timed(
                "calibration_subset_select",
                track=track_name,
                teacher=teacher,
            ):
                if (
                    args.calibration_images > 0
                    and args.calibration_images < y_train_clean.shape[0]
                ):
                    calib_rng = np.random.default_rng(
                        args.seed + stable_seed(track_name, teacher, "calibration_subset")
                    )
                    calib_idx = np.sort(
                        calib_rng.choice(
                            y_train_clean.shape[0],
                            size=args.calibration_images,
                            replace=False,
                        )
                    )
                    y_calib_clean = y_train_clean[calib_idx]
                else:
                    y_calib_clean = y_train_clean

            teacher_equiv_label = int(equivalence_labels[teacher_idx])
            off_equiv = np.asarray(
                [label != teacher_equiv_label for label in equivalence_labels],
                dtype=bool,
            )

            for noise_mult in noise_mults:
                noise_mult = float(noise_mult)
                noise_ceiling = multiplier_to_noise_ceiling(
                    noise_mult,
                    base_noise_ceiling,
                )
                with profiler.timed(
                    "calibrate_response_noise_for_rdm_reliability",
                    track=track_name,
                    teacher=teacher,
                    noise_mult=noise_mult,
                ):
                    if args.fit_noise_calibration == "rdm_empirical":
                        cal_rng = np.random.default_rng(
                            args.seed
                            + stable_seed(track_name, teacher, noise_mult, "rdm_empirical")
                        )
                        response_noise_std, _achieved = (
                            calibrate_response_noise_for_rdm_reliability(
                                y_calib_clean,
                                target_reliability=noise_ceiling,
                                metric=metric,
                                corr_type=args.corr_type,
                                rng=cal_rng,
                                n_samples=args.calibration_noise_samples,
                                max_iter=args.calibration_max_iter,
                            )
                        )
                    else:
                        response_noise_std = response_noise_std_from_mode(
                            noise_mult,
                            base_noise_ceiling,
                            args.fit_noise_calibration,
                        )

                for noise_sample_idx in range(args.n_noise_samples):
                    with profiler.timed(
                        "generate_noisy_train_val_targets",
                        track=track_name,
                        teacher=teacher,
                        noise_mult=noise_mult,
                        noise_sample_idx=noise_sample_idx,
                    ):
                        y_train = y_train_clean + teacher_rng.normal(
                            0.0,
                            response_noise_std,
                            y_train_clean.shape,
                        ).astype(np.float32)
                        y_val = y_val_clean + teacher_rng.normal(
                            0.0,
                            response_noise_std,
                            y_val_clean.shape,
                        ).astype(np.float32)
                        if args.eval_refit_mode == "eval_augmented_loo":
                            if y_base_fit_clean is None:
                                raise RuntimeError("Missing base fit targets")
                            if (
                                y_base_fit_clean.shape[0]
                                == y_train_clean.shape[0] + y_val_clean.shape[0]
                            ):
                                y_base_fit = np.concatenate([y_train, y_val], axis=0)
                            else:
                                y_base_fit = y_base_fit_clean + teacher_rng.normal(
                                    0.0,
                                    response_noise_std,
                                    y_base_fit_clean.shape,
                                ).astype(np.float32)
                        else:
                            y_base_fit = None

                    eval_y_fit = {}
                    noisy_teacher_rdms = {}
                    with profiler.timed(
                        "generate_noisy_eval_rdms_all_sets",
                        track=track_name,
                        teacher=teacher,
                        noise_mult=noise_mult,
                        noise_sample_idx=noise_sample_idx,
                        detail=f"n_eval_sets={len(eval_y_clean)}",
                    ):
                        for key, clean_rdm in clean_eval_rdms.items():
                            if args.eval_noise_mode == "rdm":
                                std = rdm_noise_std_from_clean(
                                    clean_rdm,
                                    base_noise_ceiling,
                                    noise_mult,
                                )
                                noisy_teacher_rdms[key] = clean_rdm + teacher_rng.normal(
                                    0.0,
                                    std,
                                    clean_rdm.shape,
                                ).astype(np.float32)
                            elif args.eval_noise_mode == "response":
                                y_eval_noisy = eval_y_clean[key] + teacher_rng.normal(
                                    0.0,
                                    response_noise_std,
                                    eval_y_clean[key].shape,
                                ).astype(np.float32)
                                noisy_teacher_rdms[key] = get_rdm_vector_np(
                                    y_eval_noisy,
                                    metric,
                                )
                                if args.eval_refit_mode == "eval_augmented_loo":
                                    eval_y_fit[key] = y_eval_noisy
                            else:
                                raise ValueError(args.eval_noise_mode)

                    scores_by_eval = {
                        key: np.full(len(model_names), np.nan, dtype=np.float32)
                        for key in eval_meta
                    }
                    for candidate_idx, candidate in enumerate(model_names):
                        with profiler.timed(
                            "select_targetwise_alpha",
                            track=track_name,
                            teacher=teacher,
                            candidate=candidate,
                            noise_mult=noise_mult,
                            noise_sample_idx=noise_sample_idx,
                        ):
                            alpha_values, best_alpha_idx, coefficient_cache = (
                                select_targetwise_alpha_indices(
                                    candidate_ops[candidate],
                                    y_train,
                                    y_val,
                                )
                            )
                        for key in eval_keys_for_ops(
                            candidate_ops[candidate],
                            alpha_values,
                        ):
                            with profiler.timed(
                                "predict_eval_responses",
                                track=track_name,
                                teacher=teacher,
                                candidate=candidate,
                                noise_mult=noise_mult,
                                noise_sample_idx=noise_sample_idx,
                            ):
                                pred = predict_eval_with_targetwise_alphas(
                                    alpha_ops=candidate_ops[candidate],
                                    alpha_values=alpha_values,
                                    best_alpha_idx=best_alpha_idx,
                                    eval_key=key,
                                    y_train=y_train,
                                    eval_refit_mode=args.eval_refit_mode,
                                    y_base_fit=y_base_fit,
                                    eval_y_fit=eval_y_fit.get(key) if eval_y_fit else None,
                                    coefficient_cache=coefficient_cache,
                                )
                            with profiler.timed(
                                "prediction_rdm",
                                track=track_name,
                                teacher=teacher,
                                candidate=candidate,
                                noise_mult=noise_mult,
                                noise_sample_idx=noise_sample_idx,
                            ):
                                pred_rdm = get_rdm_vector_np(pred, metric)
                            with profiler.timed(
                                "rdm_spearman_score",
                                track=track_name,
                                teacher=teacher,
                                candidate=candidate,
                                noise_mult=noise_mult,
                                noise_sample_idx=noise_sample_idx,
                            ):
                                scores_by_eval[key][candidate_idx] = (
                                    calculate_correlation_value(
                                        pred_rdm,
                                        noisy_teacher_rdms[key],
                                        args.corr_type,
                                    )
                                )

                    with profiler.timed(
                        "recover_argmax_and_margin",
                        track=track_name,
                        teacher=teacher,
                        noise_mult=noise_mult,
                        noise_sample_idx=noise_sample_idx,
                    ):
                        for scores in scores_by_eval.values():
                            scores = np.nan_to_num(scores, nan=-np.inf)
                            recovered_idx = int(np.argmax(scores))
                            competitor_scores = scores[off_equiv]
                            _teacher_margin = (
                                float(np.min(scores[teacher_idx] - competitor_scores))
                                if len(competitor_scores)
                                else float("nan")
                            )
                            _recovered_correct = (
                                int(equivalence_labels[recovered_idx])
                                == teacher_equiv_label
                            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"teacher_student_step_profile_{args.model_set}_"
        f"{timestamp}_pid{os.getpid()}"
    )
    detail_path = args.output_dir / f"{stem}_detail.csv"
    summary_path = args.output_dir / f"{stem}_summary.csv"
    metadata_path = args.output_dir / f"{stem}_metadata.json"
    detail = profiler.detail_frame()
    summary = profiler.summary_frame()
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata = {
        "model_set": args.model_set,
        "tracks": tracks,
        "model_names": model_names,
        "teacher_indices": teacher_idx_list,
        "metric": metric,
        "corr_type": args.corr_type,
        "eval_noise_mode": args.eval_noise_mode,
        "fit_noise_calibration": args.fit_noise_calibration,
        "eval_refit_mode": args.eval_refit_mode,
        "n_selected": int(n_selected),
        "n_random_subsets": int(args.n_random_subsets),
        "n_noise_samples": int(args.n_noise_samples),
        "noise_mults": noise_mults.tolist(),
        "refit_pool_size": int(args.refit_pool_size),
        "refit_train_n": int(args.refit_pool_size - args.refit_val_size),
        "refit_val_n": int(args.refit_val_size),
        "max_refit_pool_size": int(max_refit_pool_size),
        "output_detail_csv": str(detail_path),
        "output_summary_csv": str(summary_path),
    }
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print()
    print(f"Detail CSV: {detail_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Metadata: {metadata_path}")
    print()
    print_markdown_table(summary, max_rows=args.summary_rows)


if __name__ == "__main__":
    main()
