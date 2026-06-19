#!/usr/bin/env python3
"""Compute teacher/student independent-refit recovery scored in RDM space."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT = Path(__file__).resolve()
ROOT = next(p for p in SCRIPT.parents if (p / "src" / "cstims").exists())
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cstims import paths  # noqa: E402
from cstims.evaluation.constants import get_default_noise_level_multipliers  # noqa: E402
from cstims.evaluation.io import load_payload  # noqa: E402
from cstims.evaluation.payload import filter_payload_to_models  # noqa: E402
from cstims.evaluation.random_features import (  # noqa: E402
    available_random_models,
    ensure_npy_feature_cache,
    load_random_feature_cache,
)
from cstims.evaluation.teacher_student import (  # noqa: E402
    DEFAULT_RESULTS,
    build_candidate_ops,
    build_eval_raw_and_meta,
    completed_tracks_from_detail,
    detect_equivalent_models,
    load_encoding_params_for_models,
    load_existing_detail,
    merge_complete_cache_tracks,
    parse_csv_list,
    parse_float_list,
    parse_index_list,
    run_track_rdm_recovery,
    safe_name,
    stable_seed,
    write_outputs,
)
from cstims.evaluation.track_loading import (  # noqa: E402
    encode_raw_feature_arrays,
    load_selected_raw_features,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-set", default="sota")
    parser.add_argument("--selection-root", type=Path, default=paths.selected_stimuli_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--random-feature-dir", type=Path, required=True)
    parser.add_argument(
        "--encoding-root",
        type=Path,
        default=paths.shared_encoding_root(),
    )
    parser.add_argument("--tracks", default="sub-01")
    parser.add_argument("--n-random-images", type=int, default=100000)
    parser.add_argument("--refit-pool-size", type=int, default=1000)
    parser.add_argument("--refit-val-size", type=int, default=200)
    parser.add_argument(
        "--max-refit-pool-size",
        type=int,
        default=None,
        help=(
            "Reserve this many natural-pool images before drawing random eval "
            "subsets. Use the largest refit size in a size sweep so random eval "
            "subsets are identical across refit sizes."
        ),
    )
    parser.add_argument("--n-refit-repeats", type=int, default=1)
    parser.add_argument(
        "--refit-repeat-indices",
        default=None,
        help="Comma/range list of zero-based refit-pool repeat indices to compute.",
    )
    parser.add_argument("--n-random-subsets", type=int, default=20)
    parser.add_argument("--n-noise-samples", type=int, default=1)
    parser.add_argument(
        "--batch-noise-samples",
        action="store_true",
        help="Batch all noise samples for each teacher/noise level in response-noise mode.",
    )
    parser.add_argument(
        "--fast-gpu-batch",
        action="store_true",
        help=(
            "Convenience flag for the fastest current path: batched noise samples, "
            "GPU alpha selection, GPU eval prediction, GPU eval noise, and CUDA RDM scoring."
        ),
    )
    parser.add_argument(
        "--build-random-npy-cache",
        action="store_true",
        help="Create uncompressed .npy random-feature caches before loading.",
    )
    parser.add_argument("--noise-mults", default=None)
    parser.add_argument("--noise-ceiling", type=float, default=None)
    parser.add_argument("--alphas", default="0.001,0.01,0.1,1,10,100")
    parser.add_argument("--metric", default=None)
    parser.add_argument("--corr-type", choices=["pearson", "spearman"], default="spearman")
    parser.add_argument("--eval-noise-mode", choices=["rdm", "response"], default="rdm")
    parser.add_argument(
        "--fit-noise-calibration",
        choices=["response", "rdm_analytic", "rdm_empirical"],
        default="response",
    )
    parser.add_argument(
        "--eval-refit-mode",
        choices=["independent", "eval_augmented_loo", "eval_augmented_nested_loo"],
        default="independent",
        help=(
            "independent fits readouts only on the refit pool; "
            "eval_augmented_loo fits final readouts on the refit pool plus each "
            "eval set, using kernel-ridge leave-one-out predictions for eval images; "
            "eval_augmented_nested_loo additionally selects alpha inside each "
            "eval set with the outer eval image left out."
        ),
    )
    parser.add_argument("--calibration-images", type=int, default=100)
    parser.add_argument("--calibration-noise-samples", type=int, default=2)
    parser.add_argument("--calibration-max-iter", type=int, default=8)
    parser.add_argument("--rdm-device", default="cpu")
    parser.add_argument(
        "--gpu-alpha-batch",
        action="store_true",
        help="Select target-wise alphas on the CUDA RDM device in batched mode.",
    )
    parser.add_argument(
        "--gpu-predict-batch",
        action="store_true",
        help="Compute batched eval predictions on the CUDA RDM device.",
    )
    parser.add_argument(
        "--gpu-eval-noise-batch",
        action="store_true",
        help="Generate noisy eval responses on the CUDA RDM device.",
    )
    parser.add_argument("--target-dim", type=int, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--teacher-indices",
        default=None,
        help="Comma/range list of zero-based teacher indices to compute, e.g. 0-4,9.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only write per-teacher cache shards; do not update aggregate CSVs.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge complete tracks from existing detail CSV and teacher cache shards.",
    )
    parser.add_argument(
        "--write-detail-output",
        action="store_true",
        help=(
            "When used with --merge-only, also write the full per-teacher/per-subset "
            "teacher_student_recoveries.csv. Normal plots only require the summary CSVs."
        ),
    )
    parser.add_argument(
        "--teacher-workers",
        type=int,
        default=1,
        help=(
            "Compute teachers in parallel inside this process. The default serial "
            "path is retained for exact legacy behavior. Values >1 currently "
            "support CPU scalar scoring only, not CUDA/batched evaluation."
        ),
    )
    parser.add_argument("--encoding-device", default="cuda")
    parser.add_argument("--encoding-batch-size", type=int, default=1024)
    parser.add_argument(
        "--unique-encodings",
        action="store_true",
        default=False,
        help="Use track-specific subject-unique encoding roots from paper config.",
    )
    parser.add_argument(
        "--shared-encodings",
        action="store_false",
        dest="unique_encodings",
        help="Use the single --encoding-root for all encoding tracks.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.fast_gpu_batch:
        args.batch_noise_samples = True
        args.gpu_alpha_batch = True
        args.gpu_predict_batch = True
        args.gpu_eval_noise_batch = True
        args.rdm_device = "cuda"

    if args.n_refit_repeats < 1:
        raise ValueError("--n-refit-repeats must be at least 1")
    if args.teacher_workers < 1:
        raise ValueError("--teacher-workers must be at least 1")
    if args.rdm_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--rdm-device cuda requested but CUDA is unavailable")
    if args.gpu_alpha_batch and args.rdm_device != "cuda":
        raise ValueError("--gpu-alpha-batch requires --rdm-device cuda")
    if args.gpu_predict_batch and args.rdm_device != "cuda":
        raise ValueError("--gpu-predict-batch requires --rdm-device cuda")
    if args.gpu_eval_noise_batch and args.rdm_device != "cuda":
        raise ValueError("--gpu-eval-noise-batch requires --rdm-device cuda")
    if args.batch_noise_samples and args.eval_noise_mode != "response":
        print(
            "--batch-noise-samples only applies to --eval-noise-mode response; "
            "legacy RDM-noise evaluation will use the scalar path.",
            flush=True,
        )
        args.batch_noise_samples = False
        args.gpu_alpha_batch = False
        args.gpu_predict_batch = False
        args.gpu_eval_noise_batch = False
    if args.batch_noise_samples and (
        not args.gpu_alpha_batch or not args.gpu_predict_batch
    ):
        raise ValueError(
            "--batch-noise-samples currently requires --gpu-alpha-batch and "
            "--gpu-predict-batch"
        )
    if args.batch_noise_samples and args.rdm_device != "cuda":
        raise ValueError("--batch-noise-samples currently requires --rdm-device cuda")
    if args.teacher_workers > 1 and (
        args.batch_noise_samples
        or args.rdm_device == "cuda"
        or args.gpu_alpha_batch
        or args.gpu_predict_batch
        or args.gpu_eval_noise_batch
    ):
        raise ValueError(
            "--teacher-workers > 1 currently supports CPU scalar scoring only; "
            "use --teacher-workers 1 for CUDA/batched evaluation"
        )
    rdm_device = torch.device(args.rdm_device)
    max_refit_pool_size = int(args.max_refit_pool_size or args.refit_pool_size)
    if max_refit_pool_size < args.refit_pool_size:
        raise ValueError("--max-refit-pool-size cannot be smaller than --refit-pool-size")
    if args.refit_val_size < 1 or args.refit_val_size >= args.refit_pool_size:
        raise ValueError("--refit-val-size must be between 1 and refit_pool_size - 1")
    refit_repeat_indices_set = parse_index_list(
        args.refit_repeat_indices,
        args.n_refit_repeats,
    )
    refit_repeat_indices = (
        sorted(refit_repeat_indices_set)
        if refit_repeat_indices_set is not None
        else list(range(args.n_refit_repeats))
    )

    payload = load_payload(args.selection_root / args.model_set)
    model_names = list(payload["model_names"])
    if args.random_feature_dir is not None:
        available = available_random_models(args.random_feature_dir, model_names)
        model_names = available
        payload = filter_payload_to_models(payload, model_names)
    if args.build_random_npy_cache:
        print(
            f"Building mmap random-feature cache for {len(model_names)} models",
            flush=True,
        )
        ensure_npy_feature_cache(args.random_feature_dir, model_names)

    metric = args.metric or payload.get("config", {}).get("metric", "cosine")
    noise_mults = (
        np.asarray(parse_float_list(args.noise_mults), dtype=np.float64)
        if args.noise_mults
        else np.asarray(get_default_noise_level_multipliers(), dtype=np.float64)
    )
    alphas = [float(x) for x in parse_csv_list(args.alphas)]
    alpha_selection = (
        "targetwise_eval_nested_loo_pearson"
        if args.eval_refit_mode == "eval_augmented_nested_loo"
        else "targetwise_validation_pearson"
    )
    teacher_indices = parse_index_list(args.teacher_indices, len(model_names))
    target_nc = float(args.noise_ceiling or payload.get("config", {}).get("noise_ceiling_target", 0.46))
    tracks = []
    for track_name in parse_csv_list(args.tracks):
        if track_name == "raw":
            tracks.append({"name": "raw", "type": "identity"})
        else:
            tracks.append({"name": track_name, "type": "encoding", "encoding_name": track_name})

    if args.output_dir is None:
        if args.eval_refit_mode == "independent":
            default_name = (
                f"{args.model_set}_teacher_student_independent_refit_1k_rdm_recovery_"
                f"{args.eval_noise_mode}"
            )
        elif args.eval_refit_mode == "eval_augmented_loo":
            default_name = (
                f"{args.model_set}_teacher_student_eval_augmented_loo_refit_1k_"
                f"rdm_recovery_{args.eval_noise_mode}"
            )
        else:
            default_name = (
                f"{args.model_set}_teacher_student_eval_augmented_nested_loo_refit_1k_"
                f"rdm_recovery_{args.eval_noise_mode}"
            )
        out_dir = DEFAULT_RESULTS / default_name
    else:
        out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (out_dir / "_teacher_cache")
    model_cache_dir = cache_dir / safe_name(args.model_set)
    encoding_root_map = None
    if args.unique_encodings:
        encoding_root_map = {
            key: Path(value) for key, value in paths.unique_encoding_dirs().items()
        }
    encoding_roots_metadata = (
        {key: str(value) for key, value in encoding_root_map.items()}
        if encoding_root_map is not None
        else {"shared": str(args.encoding_root)}
    )

    if args.merge_only:
        selected_indices = payload.get("selected_global_indices", [])
        n_selected_metadata = (
            int(len(selected_indices))
            if selected_indices is not None
            else int(payload.get("config", {}).get("target_size", 0))
        )
        metadata = {
            "model_set": args.model_set,
            "model_names": model_names,
            "tracks": tracks,
            "metric": metric,
            "corr_type": args.corr_type,
            "alpha_selection": alpha_selection,
            "eval_noise_mode": args.eval_noise_mode,
            "fit_noise_calibration": args.fit_noise_calibration,
            "eval_refit_mode": args.eval_refit_mode,
            "calibration_images": args.calibration_images,
            "calibration_noise_samples": args.calibration_noise_samples,
            "calibration_max_iter": args.calibration_max_iter,
            "target_dim": args.target_dim,
            "random_feature_dir": str(args.random_feature_dir),
            "encoding_roots": encoding_roots_metadata,
            "n_random_images": args.n_random_images,
            "n_selected_eval_images": n_selected_metadata,
            "random_eval_subset_size": n_selected_metadata,
            "refit_pool_size": args.refit_pool_size,
            "max_refit_pool_size": max_refit_pool_size,
            "refit_train_n": args.refit_pool_size - args.refit_val_size,
            "refit_val_n": args.refit_val_size,
            "n_refit_repeats": args.n_refit_repeats,
            "computed_refit_repeat_indices": refit_repeat_indices,
            "n_random_subsets": args.n_random_subsets,
            "n_noise_samples": args.n_noise_samples,
            "teacher_workers": int(args.teacher_workers),
            "batch_noise_samples": bool(args.batch_noise_samples),
            "fast_gpu_batch": bool(args.fast_gpu_batch),
            "rdm_device": str(rdm_device),
            "gpu_alpha_batch": bool(args.gpu_alpha_batch),
            "gpu_predict_batch": bool(args.gpu_predict_batch),
            "gpu_eval_noise_batch": bool(args.gpu_eval_noise_batch),
            "base_noise_ceiling": target_nc,
            "noise_mults": noise_mults.tolist(),
            "seed": args.seed,
            "resume_enabled": not args.no_resume,
            "note": (
                "Held-out recovery is scored as corr(RDM(candidate prediction), "
                "noisy RDM(teacher response)). eval-augmented modes use the refit "
                "pool plus each eval set for final readouts, with leave-one-out "
                "predictions for eval images. eval_augmented_nested_loo also "
                "selects alpha inside the eval set with the outer eval image held out."
            ),
        }
        metadata_path = out_dir / "metadata.json"
        tmp_metadata_path = out_dir / f".metadata.{os.getpid()}.tmp"
        with tmp_metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2)
        tmp_metadata_path.replace(metadata_path)
        complete = merge_complete_cache_tracks(
            out_dir=out_dir,
            cache_dir=model_cache_dir,
            tracks=tracks,
            model_names=model_names,
            noise_mults=noise_mults,
            n_noise_samples=args.n_noise_samples,
            n_eval_sets=1 + args.n_random_subsets,
            refit_repeat_indices=refit_repeat_indices,
            write_detail=args.write_detail_output,
        )
        print(
            "Merged complete tracks: "
            + (", ".join(sorted(complete)) if complete else "none"),
            flush=True,
        )
        return

    selected_raw = load_selected_raw_features(
        payload,
        model_names=model_names,
        selection_variant="final",
    )
    random_raw = load_random_feature_cache(
        random_feature_dir=args.random_feature_dir,
        model_names=model_names,
        n_random=args.n_random_images,
        view_name="raw",
    )
    n_available = min(arr.shape[0] for arr in random_raw.values())
    n_selected = next(iter(selected_raw.values())).shape[0]
    if max_refit_pool_size + args.n_random_subsets * n_selected > n_available:
        raise ValueError("Not enough random images for disjoint refit and random eval pools")

    metadata = {
        "model_set": args.model_set,
        "model_names": model_names,
        "tracks": tracks,
        "metric": metric,
        "corr_type": args.corr_type,
        "alpha_selection": alpha_selection,
        "eval_noise_mode": args.eval_noise_mode,
        "fit_noise_calibration": args.fit_noise_calibration,
        "eval_refit_mode": args.eval_refit_mode,
        "calibration_images": args.calibration_images,
        "calibration_noise_samples": args.calibration_noise_samples,
        "calibration_max_iter": args.calibration_max_iter,
        "target_dim": args.target_dim,
        "random_feature_dir": str(args.random_feature_dir),
        "encoding_roots": encoding_roots_metadata,
        "n_random_images": args.n_random_images,
        "n_selected_eval_images": int(n_selected),
        "random_eval_subset_size": int(n_selected),
        "refit_pool_size": args.refit_pool_size,
        "max_refit_pool_size": max_refit_pool_size,
        "refit_train_n": args.refit_pool_size - args.refit_val_size,
        "refit_val_n": args.refit_val_size,
        "n_refit_repeats": args.n_refit_repeats,
        "computed_refit_repeat_indices": refit_repeat_indices,
        "n_random_subsets": args.n_random_subsets,
        "n_noise_samples": args.n_noise_samples,
        "teacher_workers": int(args.teacher_workers),
        "batch_noise_samples": bool(args.batch_noise_samples),
        "fast_gpu_batch": bool(args.fast_gpu_batch),
        "rdm_device": str(rdm_device),
        "gpu_alpha_batch": bool(args.gpu_alpha_batch),
        "gpu_predict_batch": bool(args.gpu_predict_batch),
        "gpu_eval_noise_batch": bool(args.gpu_eval_noise_batch),
        "base_noise_ceiling": target_nc,
        "noise_mults": noise_mults.tolist(),
        "seed": args.seed,
        "resume_enabled": not args.no_resume,
        "note": (
            "Held-out recovery is scored as corr(RDM(candidate prediction), noisy "
            "RDM(teacher response)). eval-augmented modes use the refit pool plus "
            "each eval set for final readouts, with leave-one-out predictions for "
            "eval images. eval_augmented_nested_loo also selects alpha inside the "
            "eval set with the outer eval image held out."
        ),
    }
    metadata_path = out_dir / "metadata.json"
    tmp_metadata_path = out_dir / f".metadata.{os.getpid()}.tmp"
    with tmp_metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    tmp_metadata_path.replace(metadata_path)

    all_rows: list[dict[str, Any]] = []
    completed_tracks: set[str] = set()
    if not args.no_resume and not args.cache_only:
        existing_detail = load_existing_detail(out_dir)
        if not existing_detail.empty:
            completed_tracks = completed_tracks_from_detail(
                existing_detail,
                tracks=tracks,
                n_models=len(model_names),
                n_noise_levels=len(noise_mults),
                n_noise_samples=args.n_noise_samples,
                n_eval_sets=1 + args.n_random_subsets,
                n_refit_repeats=len(refit_repeat_indices),
            )
            if completed_tracks:
                print(
                    "Resume: preserving completed tracks from existing detail CSV: "
                    + ", ".join(sorted(completed_tracks)),
                    flush=True,
                )
                all_rows.extend(existing_detail.to_dict("records"))

    remaining_tracks = [track for track in tracks if track["name"] not in completed_tracks]
    if not remaining_tracks:
        if all_rows:
            write_outputs(out_dir, all_rows, len(model_names))
        print(f"All requested tracks already complete in {out_dir}", flush=True)
        return

    enc_cache: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    selected_enc_cache: dict[str, dict[str, np.ndarray]] = {}
    encoding_device = torch.device(
        args.encoding_device if args.encoding_device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    for refit_repeat_idx in refit_repeat_indices:
        print(
            f"[{args.model_set}] refit repeat "
            f"{refit_repeat_idx + 1}/{args.n_refit_repeats}",
            flush=True,
        )
        repeat_rng = np.random.default_rng(
            args.seed
            + stable_seed(
                args.model_set,
                "rdm_recovery",
                "refit_repeat",
                refit_repeat_idx,
            )
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
        random_raw_union = {model: arr[union_indices] for model, arr in random_raw.items()}
        eval_raw, eval_meta = build_eval_raw_and_meta(
            selected_raw=selected_raw,
            random_raw_union=random_raw_union,
            random_subset_positions=random_subset_positions,
            model_names=model_names,
        )
        random_enc_cache: dict[str, dict[str, np.ndarray]] = {}
        repeat_cache_dir = model_cache_dir / f"refit_repeat_{refit_repeat_idx:03d}"
        split_seed = args.seed + stable_seed(
            args.model_set,
            "rdm_recovery",
            "refit_split",
            "refit_repeat",
            refit_repeat_idx,
        )
        split_rng = np.random.default_rng(split_seed)
        refit_perm = split_rng.permutation(len(refit_positions))
        train_pos = refit_positions[refit_perm[: args.refit_pool_size - args.refit_val_size]]
        val_pos = refit_positions[
            refit_perm[
                args.refit_pool_size
                - args.refit_val_size : args.refit_pool_size
            ]
        ]
        if len(val_pos) != args.refit_val_size:
            raise ValueError("Refit validation split is shorter than requested")
        base_fit_pos = (
            np.concatenate([train_pos, val_pos])
            if args.eval_refit_mode in {"eval_augmented_loo", "eval_augmented_nested_loo"}
            else None
        )
        print(
            f"[{args.model_set}] repeat {refit_repeat_idx + 1}/"
            f"{args.n_refit_repeats}: building shared candidate ridge operators",
            flush=True,
        )
        candidate_ops = build_candidate_ops(
            random_raw_union=random_raw_union,
            eval_raw=eval_raw,
            refit_positions=refit_positions,
            train_pos=train_pos,
            val_pos=val_pos,
            base_fit_pos=base_fit_pos,
            model_names=model_names,
            alphas=alphas,
            eval_refit_mode=args.eval_refit_mode,
        )
        equivalence_labels = detect_equivalent_models(random_raw_union, model_names)
        if len(set(equivalence_labels)) < len(equivalence_labels):
            groups: dict[int, list[str]] = {}
            for label, name in zip(equivalence_labels, model_names):
                groups.setdefault(int(label), []).append(name)
            dup_groups = [names for names in groups.values() if len(names) > 1]
            print(f"  equivalent model groups: {dup_groups}", flush=True)

        for track_idx, track in enumerate(tracks):
            print(
                f"[{args.model_set}] repeat {refit_repeat_idx + 1}/"
                f"{args.n_refit_repeats}, track {track_idx + 1}/{len(tracks)}: "
                f"{track['name']}",
                flush=True,
            )
            if track["name"] in completed_tracks:
                print(
                    f"  skipping completed track from resume cache: {track['name']}",
                    flush=True,
                )
                continue
            track_seed = args.seed + stable_seed(
                args.model_set,
                track["name"],
                "rdm_recovery",
                "refit_repeat",
                refit_repeat_idx,
            )
            if track.get("type") == "identity":
                selected_target = selected_raw
                random_target_union = random_raw_union
            else:
                enc_name = track.get("encoding_name") or track["name"]
                enc_root = (
                    encoding_root_map.get(enc_name, args.encoding_root)
                    if encoding_root_map is not None
                    else args.encoding_root
                )
                if enc_name not in enc_cache:
                    enc_cache[enc_name] = load_encoding_params_for_models(
                        enc_root,
                        model_names,
                        enc_name,
                        roi_subset=payload.get("config", {}).get("encoding_roi_subset", "hlvis"),
                        device=encoding_device,
                    )
                if enc_name not in selected_enc_cache:
                    selected_enc_cache[enc_name] = encode_raw_feature_arrays(
                        selected_raw,
                        enc_name,
                        enc_cache[enc_name],
                        device=encoding_device,
                        batch_size=args.encoding_batch_size,
                    )
                if enc_name not in random_enc_cache:
                    random_enc_cache[enc_name] = encode_raw_feature_arrays(
                        random_raw_union,
                        enc_name,
                        enc_cache[enc_name],
                        device=encoding_device,
                        batch_size=args.encoding_batch_size,
                    )
                selected_target = selected_enc_cache[enc_name]
                random_target_union = random_enc_cache[enc_name]

            rows = run_track_rdm_recovery(
                model_set=args.model_set,
                track=track,
                refit_repeat_idx=refit_repeat_idx,
                selected_target=selected_target,
                random_target_union=random_target_union,
                random_subset_positions=random_subset_positions,
                train_pos=train_pos,
                val_pos=val_pos,
                base_fit_pos=base_fit_pos,
                eval_meta=eval_meta,
                candidate_ops=candidate_ops,
                model_names=model_names,
                equivalence_labels=equivalence_labels,
                alphas=alphas,
                noise_mults=noise_mults,
                n_noise_samples=args.n_noise_samples,
                refit_train_n=args.refit_pool_size - args.refit_val_size,
                refit_val_n=args.refit_val_size,
                base_noise_ceiling=target_nc,
                metric=metric,
                corr_type=args.corr_type,
                eval_noise_mode=args.eval_noise_mode,
                fit_noise_calibration=args.fit_noise_calibration,
                eval_refit_mode=args.eval_refit_mode,
                calibration_images=args.calibration_images,
                calibration_noise_samples=args.calibration_noise_samples,
                calibration_max_iter=args.calibration_max_iter,
                target_dim=args.target_dim,
                teacher_cache_dir=repeat_cache_dir,
                teacher_indices=teacher_indices,
                seed=track_seed,
                batch_noise_samples=args.batch_noise_samples,
                rdm_device=rdm_device,
                gpu_alpha_batch=args.gpu_alpha_batch,
                gpu_predict_batch=args.gpu_predict_batch,
                gpu_eval_noise_batch=args.gpu_eval_noise_batch,
                teacher_workers=args.teacher_workers,
            )
            all_rows.extend(rows)
            if not args.cache_only:
                write_outputs(out_dir, all_rows, len(model_names))

    if args.cache_only:
        print(f"Wrote teacher cache shards under {model_cache_dir}", flush=True)
    else:
        print(f"Wrote {out_dir}", flush=True)



if __name__ == "__main__":
    main()
