#!/usr/bin/env python3
"""Greedy selection with eval-set-augmented teacher/student refit scoring.

This is an experimental selector for the question:

    Which images preserve teacher recovery after each candidate model is allowed
    to refit a readout on an independent refit set plus the selected images?

The expensive objective is only evaluated on a shortlist.  By default that
shortlist is formed with the existing attenuated fixed-RDM sub-01 objective plus
some random exploration candidates.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[3]
SRC_DIR = ROOT / "src"
for path in (SRC_DIR, SCRIPT.parent):
    sys.path.insert(0, str(path))

from cstims.data_loader import load_natural_features_with_metadata, max_images_for_ram  # noqa: E402
from cstims.evaluation.teacher_student import stable_seed  # noqa: E402

from feature_method_sweep import (  # noqa: E402
    calibrate_noise_by_track,
    exclude_failed_indices,
    filter_record_to_dict,
    load_encoding_params_for_sweep,
    load_env_paths,
    load_existing_filter_records,
    load_layer_names,
    load_model_set,
    load_npz_pool_features,
    make_image_filter,
    mark_filter_failures,
    save_manifest,
    save_runtime_progress,
    select_initial_indices,
)
from refit_robust import (  # noqa: E402
    DEFAULT_ALPHA_TARGET_BATCH_SIZE,
    DEFAULT_SCORE_TARGET_BATCH_SIZE,
    NUMBA_AVAILABLE,
    build_fit_context,
    build_noise_states,
    build_proxy_runtime,
    build_refit_splits,
    bound_natural_feature_pool,
    choose_target_columns,
    encode_indices_modelwise_cached,
    format_seconds,
    load_existing_indices,
    load_resume_state,
    make_method,
    parse_csv_floats,
    proxy_scores_for_pool,
    resolve_base_kernel_precompute,
    save_filter_records,
    score_shortlist_refit_robust,
    topk_shortlist,
)


MODEL_LIST_CSV = ROOT / "00_stimulus_selection" / "resources" / "model_list.csv"

def run_selection(args: argparse.Namespace) -> Path:
    """Run greedy refit-robust selection and write resumable payload files."""
    paths = load_env_paths(args.env)
    local_encoding_root = (
        ROOT
        / "01_brain_model_alignment"
        / "results"
        / "encoding_models"
        / "shared_subject_encoding_models"
        / "encoding_20251222_141301"
    )
    encoding_root = Path(paths.get("encoding_root", ""))
    if not encoding_root.exists() and local_encoding_root.exists():
        paths["encoding_root"] = str(local_encoding_root)
        print(f"Using local encoding_root: {local_encoding_root}", flush=True)
    model_set_name, model_names = load_model_set(args.model_set)

    model_list_csv = Path(paths.get("model_list_csv") or MODEL_LIST_CSV)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    payload_root = output_root / "payloads"
    method = make_method(args.method_id, args.track)
    method_dir = payload_root / args.method_id
    method_dir.mkdir(parents=True, exist_ok=True)
    save_manifest([method], payload_root)

    pool_feature_dir = args.pool_feature_dir
    max_images_arg = args.max_images
    pool_records_by_index: list[dict[str, Any]] | None = None
    pool_info: dict[str, Any] | None = None
    if pool_feature_dir is not None:
        if not args.disable_image_filter:
            raise ValueError(
                "Image filtering matches feature_method_sweep.py and requires the "
                "natural LAION feature loader. Do not pass --pool-feature-dir for "
                "filtered refit runs; use --disable-image-filter for explicit "
                "unfiltered .npz-pool runs."
            )
        pool_feature_dir = Path(pool_feature_dir).resolve()
        max_images = int(max_images_arg) if max_images_arg is not None else None
        raw_features_np, pool_records_by_index, pool_info = load_npz_pool_features(
            pool_feature_dir=pool_feature_dir,
            model_names=model_names,
            max_images=max_images,
        )
        raw_shard_slices = []
        pool_size = int(next(iter(raw_features_np.values())).shape[0])
    else:
        layer_names = load_layer_names(model_list_csv, model_names)
        if max_images_arg is not None:
            max_images = int(max_images_arg)
        else:
            max_images = max_images_for_ram(
                subset_root=Path(paths["subset_root"]),
                model_names=model_names,
                max_ram_bytes=int(args.max_ram_gb * 1024**3),
                model_csv=model_list_csv,
            )
        print(f"Loading {max_images} natural-pool images for model_set={model_set_name}")
        raw_features_np, raw_shard_slices = load_natural_features_with_metadata(
            subset_root=Path(paths["subset_root"]),
            preprocessed_dir=Path(paths["preprocessed_dirs"]["raw"]),
            model_names=model_names,
            layer_names=layer_names,
            max_images=max_images,
            model_csv=model_list_csv,
        )
        raw_features_np, pool_size, pool_info = bound_natural_feature_pool(
            raw_features_np,
            model_names,
            max_images=max_images,
        )
    if pool_size <= args.init_size:
        raise ValueError(f"Candidate pool too small: pool_size={pool_size}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    encoding_params = load_encoding_params_for_sweep(
        paths=paths,
        model_list_csv=model_list_csv,
        encoding_names=[args.track],
        device=device,
        roi_subset=args.encoding_roi_subset,
        shared_encodings=not args.unique_encodings,
    )
    if args.track not in encoding_params:
        raise RuntimeError(f"Missing encoding params for {args.track}")

    image_filter, image_filter_config = make_image_filter(
        args=args,
        paths=paths,
        raw_shard_slices=raw_shard_slices,
        output_root=output_root,
    )
    print(f"Image filter enabled: {image_filter is not None}", flush=True)
    if image_filter is not None:
        print(f"Image filter config: {image_filter_config}", flush=True)

    existing_selected = load_existing_indices(method_dir) if args.resume else None
    if existing_selected is not None:
        if len(existing_selected) < args.init_size:
            raise ValueError(
                f"Cannot resume {method_dir}: selected_indices.npy has only "
                f"{len(existing_selected)} entries, expected at least init_size={args.init_size}"
            )
        initial_indices = [int(x) for x in existing_selected[: args.init_size]]
        initial_filter_records: list[dict[str, Any]] = []
        initialization_source = "resume_existing_selected_indices"
    else:
        rng = np.random.default_rng(args.seed)
        initial_array, initial_filter_records_raw = select_initial_indices(
            rng=rng,
            initial_pool_size=pool_size,
            init_size=args.init_size,
            image_filter=image_filter,
        )
        initial_indices = [int(x) for x in initial_array.tolist()]
        initial_filter_records = [
            {
                **record,
                "method_id": args.method_id,
                "pool_size": pool_size,
            }
            for record in initial_filter_records_raw
        ]
        initialization_source = (
            "filtered_random_order" if image_filter is not None else "random_choice"
        )

    var_noise_by_track = calibrate_noise_by_track(
        raw_features_np=raw_features_np,
        model_names=model_names,
        track_specs=list(method.tracks),
        encoding_params=encoding_params,
        metric=args.metric,
        corr_type=args.corr_type,
        target_nc=args.noise_ceiling,
        seed=args.seed,
        device=device,
        calib_n_examples=args.proxy_noise_calib_examples,
        n_repeats=args.proxy_noise_calib_repeats,
    )
    if args.no_proxy_attenuation:
        var_noise_by_track = {args.track: {model: 0.0 for model in model_names}}

    base_indices, train_indices, val_indices, refit_order = build_refit_splits(
        pool_size=pool_size,
        selected_initial=initial_indices,
        refit_pool_size=args.refit_pool_size,
        refit_val_size=args.refit_val_size,
        seed=args.seed,
        exclude_refit_from_selection=args.exclude_refit_from_selection,
    )
    refit_lookup = {int(idx): pos for pos, idx in enumerate(base_indices)}
    refit_train_pos = np.asarray([refit_lookup[int(idx)] for idx in train_indices], dtype=np.int64)
    refit_val_pos = np.asarray([refit_lookup[int(idx)] for idx in val_indices], dtype=np.int64)

    target_cols = choose_target_columns(
        raw_features_np=raw_features_np,
        model_names=model_names,
        track=args.track,
        encoding_params=encoding_params,
        device=device,
        target_dim=args.target_dim,
        seed=args.seed,
    )
    print(
        f"Encoding refit targets for {args.track}; "
        f"target_dim={len(target_cols) if target_cols is not None else 'all'}",
        flush=True,
    )
    alphas = parse_csv_floats(args.alphas)
    precompute_base_kernels, base_kernel_precompute = resolve_base_kernel_precompute(
        requested=args.precompute_base_kernels,
        pool_size=pool_size,
        refit_pool_size=args.refit_pool_size,
        model_names=model_names,
        max_ram_gb=args.max_ram_gb,
    )
    target_cache_dir = args.noise_cache_dir / "clean_targets" if args.noise_cache_dir else None
    fit_context = build_fit_context(
        raw_features_np=raw_features_np,
        model_names=model_names,
        track=args.track,
        encoding_params=encoding_params,
        device=device,
        target_cols=target_cols,
        train_indices=train_indices,
        val_indices=val_indices,
        base_indices=base_indices,
        refit_train_pos=refit_train_pos,
        refit_val_pos=refit_val_pos,
        alphas=alphas,
        base_noise_ceiling=args.noise_ceiling,
        noise_mult=args.noise_mult,
        fit_noise_calibration=args.fit_noise_calibration,
        rdm_calibration_comparison=args.rdm_calibration_comparison,
        metric=args.metric,
        corr_type=args.corr_type,
        seed=args.seed,
        calibration_images=args.calibration_images,
        calibration_noise_samples=args.calibration_noise_samples,
        calibration_max_iter=args.calibration_max_iter,
        precompute_base_kernels=precompute_base_kernels,
        kernel_batch_size=args.kernel_batch_size,
        target_cache_dir=target_cache_dir,
        target_batch_size=args.alpha_target_batch_size,
    )
    print("Precomputing noisy refit targets and target-wise alpha choices", flush=True)
    noise_states = build_noise_states(
        fit_context=fit_context,
        model_names=model_names,
        seed=args.seed + stable_seed(args.method_id),
        alphas=alphas,
        n_noise_samples=args.n_noise_samples,
        alpha_target_batch_size=args.alpha_target_batch_size,
        noise_cache_dir=args.noise_cache_dir,
    )

    resume_state = (
        load_resume_state(
            method_dir=method_dir,
            target_size=args.target_size,
            init_size=args.init_size,
            pool_size=pool_size,
        )
        if args.resume
        else None
    )
    if resume_state is not None:
        selected_indices, trace_rows, candidate_rows = resume_state
        if selected_indices[: args.init_size] != list(initial_indices):
            raise ValueError(
                "Resume initial indices differ from the current run initialization; "
                "use the same --seed/--init-size/filter settings as the original "
                "run or start a new --output-root."
            )
        print(
            f"Resuming {args.method_id}: n_selected={len(selected_indices)}/"
            f"{args.target_size}, completed_iterations={len(trace_rows)}",
            flush=True,
        )
        filter_records = (
            load_existing_filter_records(method_dir)
            if image_filter is not None
            else []
        )
    else:
        selected_indices = list(initial_indices)
        trace_rows: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        filter_records = initial_filter_records
    if image_filter is not None:
        mark_filter_failures(image_filter, filter_records)

    run_config = {
        "refit_robust_selection_version": "self_initialized_feature_method",
        "script": str(SCRIPT),
        "argv": sys.argv,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method_name": args.method_id,
        "model_set_name": model_set_name,
        "model_names": model_names,
        "paths": paths,
        "feature_method_sweep": True,
        "refit_robust_selection": True,
        "target_size": args.target_size,
        "init_size": args.init_size,
        "initial_indices": selected_indices[: args.init_size],
        "initialization_source": initialization_source,
        "seed": args.seed,
        "metric": args.metric,
        "corr_type": args.corr_type,
        "track": args.track,
        "candidate_pool_size": pool_size,
        "pool_feature_dir": str(pool_feature_dir) if pool_feature_dir is not None else None,
        "pool_info": pool_info,
        "max_ram_gb": args.max_ram_gb,
        "max_loaded_images": pool_size,
        "image_filter": image_filter_config,
        "refit_pool_size": args.refit_pool_size,
        "refit_val_size": args.refit_val_size,
        "refit_train_n": args.refit_pool_size - args.refit_val_size,
        "noise_mult": args.noise_mult,
        "noise_ceiling_target": args.noise_ceiling,
        "proxy_noise_calib_examples": args.proxy_noise_calib_examples,
        "proxy_noise_calib_repeats": args.proxy_noise_calib_repeats,
        "proxy_attenuation_disabled": bool(args.no_proxy_attenuation),
        "fit_noise_calibration": args.fit_noise_calibration,
        "rdm_calibration_comparison": args.rdm_calibration_comparison,
        "n_noise_samples": args.n_noise_samples,
        "alphas": alphas,
        "target_dim": len(target_cols) if target_cols is not None else None,
        "top_k_proxy": args.top_k_proxy,
        "random_shortlist": args.random_shortlist,
        "teacher_aggregation": args.teacher_aggregation,
        "refit_objective": args.refit_objective,
        "exclude_refit_from_selection": args.exclude_refit_from_selection,
        "precompute_base_kernels": precompute_base_kernels,
        "base_kernel_precompute": base_kernel_precompute,
        "kernel_batch_size": args.kernel_batch_size,
        "alpha_target_batch_size": args.alpha_target_batch_size,
        "score_target_batch_size": args.score_target_batch_size,
        "noise_cache_dir": str(args.noise_cache_dir) if args.noise_cache_dir else None,
        "target_cache_dir": str(target_cache_dir) if target_cache_dir else None,
        "refit_score_workers": args.refit_score_workers,
        "refit_indices": base_indices.tolist(),
        "refit_train_indices": train_indices.tolist(),
        "refit_val_indices": val_indices.tolist(),
        "feature_method_spec": asdict(method),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2, default=str)

    checkpoint_runtime = build_proxy_runtime(
        method=method,
        selected_indices=selected_indices,
        raw_features_np=raw_features_np,
        model_names=model_names,
        encoding_params=encoding_params,
        var_noise_by_track=var_noise_by_track,
        metric=args.metric,
        device=device,
        pool_size=pool_size,
    )
    checkpoint_runtime.trace_rows = trace_rows
    checkpoint_runtime.scores_combined = [
        float(row["score_combined"]) for row in trace_rows if row.get("score_combined") is not None
    ]
    checkpoint_runtime.scores_per_track_history[args.track] = list(
        checkpoint_runtime.scores_combined
    )
    checkpoint_runtime.filter_records = filter_records
    save_runtime_progress(
        checkpoint_runtime,
        payload_root,
        raw_features_np,
        raw_shard_slices=raw_shard_slices,
        model_names=model_names,
        run_config=run_config,
        pool_records_by_index=pool_records_by_index,
    )

    start_total = time.monotonic()
    while len(selected_indices) < args.target_size:
        step_start = time.monotonic()
        step = len(selected_indices) - args.init_size + 1
        runtime = build_proxy_runtime(
            method=method,
            selected_indices=selected_indices,
            raw_features_np=raw_features_np,
            model_names=model_names,
            encoding_params=encoding_params,
            var_noise_by_track=var_noise_by_track,
            metric=args.metric,
            device=device,
            pool_size=pool_size,
        )
        if args.exclude_refit_from_selection:
            runtime.pool_mask[base_indices] = False
        if image_filter is not None:
            exclude_failed_indices(runtime, image_filter)
        proxy_scores = proxy_scores_for_pool(
            runtime=runtime,
            raw_features_np=raw_features_np,
            model_names=model_names,
            encoding_params=encoding_params,
            track=args.track,
            metric=args.metric,
            corr_type="correlation",
            device=device,
            batch_size=args.proxy_batch_size,
        )
        shortlist = topk_shortlist(
            proxy_scores=proxy_scores,
            pool_mask=runtime.pool_mask,
            top_k=args.top_k_proxy,
            random_k=args.random_shortlist,
            rng=np.random.default_rng(
                args.seed + stable_seed(args.method_id, "shortlist", step)
            ),
        )
        print(
            f"[refit-robust] step {step}: selected={len(selected_indices)}, "
            f"shortlist={len(shortlist)}, proxy_best={float(np.max(proxy_scores[shortlist])):.4f}",
            flush=True,
        )

        eval_indices_for_encoding = np.unique(
            np.concatenate([np.asarray(selected_indices, dtype=np.int64), shortlist])
        )
        eval_cache_dir = (
            args.noise_cache_dir / "eval_targets" / f"step_{step:03d}"
            if args.noise_cache_dir
            else None
        )
        encoded_eval_pool = encode_indices_modelwise_cached(
            raw_features_np=raw_features_np,
            model_names=model_names,
            indices=eval_indices_for_encoding,
            track=args.track,
            encoding_params=encoding_params,
            device=device,
            target_cols=target_cols,
            cache_dir=eval_cache_dir,
            cache_prefix=f"eval_step_{step:03d}",
        )
        encoded_pos = {int(idx): pos for pos, idx in enumerate(eval_indices_for_encoding)}

        best: dict[str, Any] | None = None
        score_seed = args.seed + stable_seed(args.method_id, step)
        try:
            scored_rows, backend_timing = score_shortlist_refit_robust(
                shortlist=shortlist,
                selected_indices=selected_indices,
                encoded_eval_pool=encoded_eval_pool,
                encoded_pos=encoded_pos,
                raw_features_np=raw_features_np,
                fit_context=fit_context,
                noise_states=noise_states,
                model_names=model_names,
                alphas=alphas,
                metric=args.metric,
                corr_type=args.corr_type,
                noise_mult=args.noise_mult,
                base_noise_ceiling=args.noise_ceiling,
                seed=score_seed,
                aggregate_teachers=args.teacher_aggregation,
                objective=args.refit_objective,
                workers=args.refit_score_workers,
                score_target_batch_size=args.score_target_batch_size,
            )
        finally:
            if eval_cache_dir is not None:
                del encoded_eval_pool
                shutil.rmtree(eval_cache_dir, ignore_errors=True)
        print(
            "[refit-robust] "
            f"step {step}: backend={backend_timing.get('backend')} "
            f"objective={args.refit_objective} "
            f"workers={backend_timing.get('workers', 1)} "
            f"cache={format_seconds(float(backend_timing.get('cache_seconds', 0.0)))} "
            f"warmup={format_seconds(float(backend_timing.get('warmup_seconds', 0.0)))} "
            f"score={format_seconds(float(backend_timing.get('score_seconds', 0.0)))} "
            f"min_delta={float(backend_timing.get('minimum_delta', np.nan)):.3e}",
            flush=True,
        )

        for rank, row in enumerate(scored_rows, start=1):
            candidate_idx = int(row["candidate_index"])
            row["iteration"] = step
            row["shortlist_rank"] = rank
            row["proxy_score"] = float(proxy_scores[int(candidate_idx)])
            candidate_rows.append(row)
        if not scored_rows:
            raise RuntimeError("Shortlist was empty")
        ranked_rows = sorted(
            scored_rows,
            key=lambda row: (float(row["score"]), float(row["score_tie_breaker"])),
            reverse=True,
        )
        filter_attempts = 0
        filter_selected_passed = None
        filter_selected_reason = None
        if image_filter is not None:
            before = len(image_filter.filter_records)
            ranked_indices = np.asarray(
                [int(row["candidate_index"]) for row in ranked_rows],
                dtype=np.int64,
            )
            ranked_scores = np.asarray(
                [float(row["score"]) for row in ranked_rows],
                dtype=np.float32,
            )
            selected_idx, _filter_score, filter_attempts = image_filter.select_first_valid(
                ranked_indices,
                ranked_scores,
                candidate_scores_per_track={
                    "refit_accuracy": np.asarray(
                        [float(row["score_recovery_accuracy"]) for row in ranked_rows],
                        dtype=np.float32,
                    ),
                    "refit_margin": np.asarray(
                        [float(row["score_margin"]) for row in ranked_rows],
                        dtype=np.float32,
                    ),
                    "proxy": np.asarray(
                        [float(row["proxy_score"]) for row in ranked_rows],
                        dtype=np.float32,
                    ),
                },
                phase="greedy",
                iteration=step,
            )
            new_filter_records = [
                {
                    **filter_record_to_dict(
                        record,
                        method_id=args.method_id,
                        pool_size=pool_size,
                    )
                }
                for record in image_filter.filter_records[before:]
            ]
            filter_records.extend(new_filter_records)
            selected_filter_record = next(
                (
                    record
                    for record in new_filter_records
                    if int(record["global_idx"]) == int(selected_idx)
                ),
                None,
            )
            if selected_filter_record is not None:
                filter_selected_passed = bool(selected_filter_record["passed"])
                filter_selected_reason = selected_filter_record["reason"]
            if not args.allow_filter_fallback and not bool(filter_selected_passed):
                raise RuntimeError(
                    "Image filter did not find a passing refit-robust candidate within "
                    f"{image_filter.config.max_attempts_per_iteration} attempts "
                    f"at iteration={step}. Increase --filter-max-attempts-per-iteration "
                    "or pass --allow-filter-fallback for diagnostic runs."
                )
            best_matches = [
                row for row in ranked_rows if int(row["candidate_index"]) == int(selected_idx)
            ]
            if not best_matches:
                raise RuntimeError(
                    f"Image filter returned idx={selected_idx}, which was not in the scored shortlist"
                )
            best = best_matches[0]
        else:
            best = ranked_rows[0]

        selected_indices.append(int(best["candidate_index"]))
        trace_row = {
            "iteration": step,
            "n_selected": len(selected_indices),
            "selected_index": int(best["candidate_index"]),
            "score_combined": float(best["score"]),
            "score_objective": args.refit_objective,
            "score_tie_breaker": float(best["score_tie_breaker"]),
            "score_refit_accuracy": float(best["score_recovery_accuracy"]),
            "score_refit_margin": float(best["score_margin"]),
            "score_refit_margin_mean": float(best["teacher_margin_mean"]),
            "score_refit_margin_min": float(best["teacher_margin_min"]),
            "teacher_self_score_mean": float(best["teacher_self_score_mean"]),
            "teacher_other_score_mean": float(best["teacher_other_score_mean"]),
            "recovery_accuracy": float(best["recovery_accuracy"]),
            "teacher_majority_recovery_accuracy": float(best["teacher_majority_recovery_accuracy"]),
            "proxy_score": float(best["proxy_score"]),
            "shortlist_size": int(len(shortlist)),
            "method_id": args.method_id,
            "within": args.refit_objective,
            "across": args.teacher_aggregation,
            "filter_attempts": int(filter_attempts),
            "filter_selected_passed": filter_selected_passed,
            "filter_selected_reason": filter_selected_reason,
            "elapsed_seconds": float(time.monotonic() - step_start),
        }
        trace_rows.append(trace_row)
        print(
            f"[refit-robust] step {step}: selected idx={trace_row['selected_index']} "
            f"score={trace_row['score_combined']:.4f} "
            f"acc={trace_row['score_refit_accuracy']:.3f} "
            f"margin={trace_row['score_refit_margin']:.4f} "
            f"proxy={trace_row['proxy_score']:.4f} "
            f"elapsed={format_seconds(trace_row['elapsed_seconds'])}",
            flush=True,
        )

        np.save(method_dir / "selected_indices.npy", np.asarray(selected_indices, dtype=np.int64))
        pd.DataFrame(trace_rows).to_csv(method_dir / "selection_trace.csv", index=False)
        pd.DataFrame(candidate_rows).to_csv(method_dir / "candidate_scores.csv", index=False)
        save_filter_records(method_dir, filter_records)

    final_runtime = build_proxy_runtime(
        method=method,
        selected_indices=selected_indices,
        raw_features_np=raw_features_np,
        model_names=model_names,
        encoding_params=encoding_params,
        var_noise_by_track=var_noise_by_track,
        metric=args.metric,
        device=device,
        pool_size=pool_size,
    )
    final_runtime.trace_rows = trace_rows
    final_runtime.scores_combined = [float(row["score_combined"]) for row in trace_rows]
    final_runtime.scores_per_track_history[args.track] = [
        float(row["score_combined"]) for row in trace_rows
    ]
    final_runtime.filter_records = filter_records
    save_runtime_progress(
        final_runtime,
        payload_root,
        raw_features_np,
        raw_shard_slices=raw_shard_slices,
        model_names=model_names,
        run_config=run_config,
        pool_records_by_index=pool_records_by_index,
    )
    pd.DataFrame(candidate_rows).to_csv(method_dir / "candidate_scores.csv", index=False)
    with (method_dir / "refit_robust_summary.json").open("w") as f:
        json.dump(
            {
                "selected_indices": selected_indices,
                "n_selected": len(selected_indices),
                "elapsed_seconds": time.monotonic() - start_total,
                "elapsed": format_seconds(time.monotonic() - start_total),
            },
            f,
            indent=2,
        )
    return method_dir


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments for a refit-robust run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--method-id", default="sub01_eval_augmented_loo_refit_robust")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--env", default="raven")
    parser.add_argument("--model-set", default="sota")
    parser.add_argument("--pool-feature-dir", type=Path, default=None)
    parser.add_argument("--max-ram-gb", type=float, default=300.0)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--track", default="sub-01")
    parser.add_argument("--encoding-roi-subset", default="hlvis")
    parser.add_argument("--unique-encodings", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--corr-type", choices=["spearman"], default="spearman")
    parser.add_argument("--target-size", type=int, default=6)
    parser.add_argument("--init-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refit-pool-size", type=int, default=200)
    parser.add_argument("--refit-val-size", type=int, default=40)
    parser.add_argument("--exclude-refit-from-selection", action="store_true", default=True)
    parser.add_argument("--allow-refit-selection-overlap", dest="exclude_refit_from_selection", action="store_false")
    parser.add_argument("--top-k-proxy", type=int, default=8)
    parser.add_argument("--random-shortlist", type=int, default=0)
    parser.add_argument("--proxy-batch-size", type=int, default=2048)
    parser.add_argument("--proxy-noise-calib-examples", type=int, default=1000)
    parser.add_argument("--proxy-noise-calib-repeats", type=int, default=100)
    parser.add_argument("--no-proxy-attenuation", action="store_true")
    parser.add_argument("--alphas", default="0.001,0.01,0.1,1,10,100")
    parser.add_argument("--noise-mult", type=float, default=1.0)
    parser.add_argument("--noise-ceiling", type=float, default=0.46)
    parser.add_argument(
        "--fit-noise-calibration",
        choices=["response", "rdm_analytic", "rdm_empirical"],
        default="rdm_empirical",
    )
    parser.add_argument(
        "--rdm-calibration-comparison",
        choices=["noisy_to_noisy", "clean_to_noisy"],
        default="clean_to_noisy",
        help=(
            "Empirical RDM calibration target for --fit-noise-calibration rdm_empirical."
        ),
    )
    parser.add_argument("--calibration-images", type=int, default=100)
    parser.add_argument("--calibration-noise-samples", type=int, default=2)
    parser.add_argument("--calibration-max-iter", type=int, default=8)
    parser.add_argument("--n-noise-samples", type=int, default=1)
    parser.add_argument(
        "--target-dim",
        type=int,
        default=0,
        help="Target-response dimensions for selection; <=0 uses the full encoding target.",
    )
    parser.add_argument("--teacher-aggregation", choices=["mean", "min"], default="mean")
    parser.add_argument(
        "--refit-objective",
        choices=["accuracy_margin", "margin"],
        default="accuracy_margin",
        help=(
            "accuracy_margin selects by recovery accuracy and uses the RDM "
            "margin as a tie-breaker. margin keeps the old margin-only objective."
        ),
    )
    parser.add_argument("--precompute-base-kernels", action="store_true", default=True)
    parser.add_argument(
        "--no-precompute-base-kernels",
        dest="precompute_base_kernels",
        action="store_false",
    )
    parser.add_argument("--kernel-batch-size", type=int, default=4096)
    parser.add_argument(
        "--alpha-target-batch-size",
        type=int,
        default=DEFAULT_ALPHA_TARGET_BATCH_SIZE,
        help="Target columns per chunk when selecting validation alphas.",
    )
    parser.add_argument(
        "--score-target-batch-size",
        type=int,
        default=DEFAULT_SCORE_TARGET_BATCH_SIZE,
        help="Target columns per chunk when scoring refit predictions.",
    )
    parser.add_argument(
        "--noise-cache-dir",
        type=Path,
        default=None,
        help="Directory for disk-backed noisy base target caches.",
    )
    parser.add_argument(
        "--refit-score-workers",
        type=int,
        default=1,
        help=(
            "Forked candidate workers for shortlist scoring. Set native BLAS "
            "threads to 1 when using more than one worker."
        ),
    )
    parser.add_argument(
        "--disable-image-filter",
        action="store_true",
        help="Disable final-run image filtering. Required for arbitrary .npz feature pools.",
    )
    parser.add_argument("--filter-min-resolution", type=int, default=1000)
    parser.add_argument("--filter-natural-prob-threshold", type=float, default=0.85)
    parser.add_argument("--filter-download-timeout", type=float, default=10.0)
    parser.add_argument("--filter-max-attempts-per-iteration", type=int, default=1000)
    parser.add_argument("--filter-parallel-batch-size", type=int, default=1)
    parser.add_argument("--filter-classifier-path", type=Path, default=None)
    parser.add_argument("--disable-filter-image-save", action="store_true")
    parser.add_argument(
        "--allow-filter-fallback",
        action="store_true",
        help=(
            "Allow selecting the top scored candidate when none passes within the "
            "configured filter attempt window. Off by default for final runs."
        ),
    )
    args = parser.parse_args()
    if args.kernel_batch_size <= 0:
        raise ValueError("--kernel-batch-size must be positive")
    if args.alpha_target_batch_size <= 0:
        raise ValueError("--alpha-target-batch-size must be positive")
    if args.score_target_batch_size <= 0:
        raise ValueError("--score-target-batch-size must be positive")
    if args.refit_score_workers <= 0:
        raise ValueError("--refit-score-workers must be positive")
    if args.proxy_noise_calib_examples <= 0:
        raise ValueError("--proxy-noise-calib-examples must be positive")
    if args.proxy_noise_calib_repeats <= 0:
        raise ValueError("--proxy-noise-calib-repeats must be positive")
    if not NUMBA_AVAILABLE:
        raise RuntimeError("refit-robust selection requires numba")
    if args.metric != "cosine" or args.corr_type != "spearman":
        raise ValueError("refit-robust selection currently supports only cosine/Spearman")
    if args.output_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_root = (
            SCRIPT.parents[1]
            / "results"
            / f"{args.model_set}_refit_robust_{stamp}"
        )
    return args


def main() -> None:
    """CLI entry point for refit-robust selection."""
    args = parse_args()
    method_dir = run_selection(args)
    print(f"Done. Payload: {method_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
