#!/usr/bin/env python3
"""Independent-refit teacher/student recovery for feature-method sweep.

Candidate readouts are fitted on an independent natural-image refit pool, then
evaluated on selected and random image sets.  This keeps the selected/random
stimuli out of readout training.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
BASE_SCRIPT = SCRIPT.parent / "compute_teacher_student_recovery.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("feature_sweep_teacher_student", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


DEFAULT_RANDOM_FEATURE_DIR = (
    base.ROOT / "shared" / "cache_or_heavy" / "natural_pool_subset_100k_seed42"
)


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
            name: np.asarray((x_eval @ x_train.T).astype(np.float64) @ inv, dtype=np.float32)
            for name, x_eval in eval_sets.items()
        }
        out[float(alpha)] = (np.asarray(k_val @ inv, dtype=np.float32), eval_ops)
    return out


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


def build_eval_sets(
    *,
    method_id: str,
    target_track: str,
    selected_raw: dict[str, np.ndarray],
    selected_target: dict[str, np.ndarray],
    random_raw_union: dict[str, np.ndarray],
    random_target_union: dict[str, np.ndarray],
    random_subset_positions: list[np.ndarray],
    model_names: list[str],
    include_random: bool = True,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
    raw_sets: dict[str, dict[str, np.ndarray]] = {
        f"{method_id}|selected|0": selected_raw,
    }
    target_sets: dict[str, dict[str, np.ndarray]] = {
        f"{method_id}|selected|0": selected_target,
    }
    metadata = {
        f"{method_id}|selected|0": {
            "method_id": method_id,
            "subset_type": "selected",
            "subset_idx": 0,
            "target_track": target_track,
        }
    }
    if include_random:
        for subset_idx, pos in enumerate(random_subset_positions):
            key = f"{method_id}|random|{subset_idx}"
            raw_sets[key] = {model: random_raw_union[model][pos] for model in model_names}
            target_sets[key] = {model: random_target_union[model][pos] for model in model_names}
            metadata[key] = {
                "method_id": method_id,
                "subset_type": "random",
                "subset_idx": subset_idx,
                "target_track": target_track,
            }
    return raw_sets, target_sets, metadata


def run_track(
    *,
    methods: list[str],
    payloads: dict[str, dict],
    target_track: str,
    target_space: str,
    model_names: list[str],
    random_raw_union: dict[str, np.ndarray],
    refit_positions: np.ndarray,
    random_subset_positions: list[np.ndarray],
    encoding_params: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    alphas: list[float],
    noise_mults: list[float],
    n_noise_samples: int,
    refit_train_n: int,
    refit_val_n: int,
    base_noise_ceiling: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    if refit_train_n + refit_val_n > len(refit_positions):
        raise ValueError("refit_train_n + refit_val_n exceeds refit pool size")
    perm = rng.permutation(len(refit_positions))
    train_pos = refit_positions[perm[:refit_train_n]]
    val_pos = refit_positions[perm[refit_train_n : refit_train_n + refit_val_n]]

    if target_space == "encoded":
        if encoding_params is None:
            raise ValueError("encoding_params required for encoded target space")
        random_target_union = base.encode_raw_features(random_raw_union, encoding_params)
    else:
        random_target_union = random_raw_union

    refit_raw = {model: random_raw_union[model] for model in model_names}
    refit_target = {model: random_target_union[model] for model in model_names}

    all_raw_eval: dict[str, dict[str, np.ndarray]] = {}
    all_target_eval: dict[str, dict[str, np.ndarray]] = {}
    eval_metadata: dict[str, dict[str, Any]] = {}
    random_owner_method = methods[0]
    for method_idx, method_id in enumerate(methods):
        payload = payloads[method_id]
        if target_space == "encoded":
            selected_raw, selected_target = base.selected_arrays_from_payload(
                payload,
                target_track,
                model_names,
                encoding_params,
            )
        else:
            selected_raw = base.selected_raw_from_payload(payload, model_names)
            selected_target = selected_raw
        raw_sets, target_sets, metadata = build_eval_sets(
            method_id=method_id,
            target_track=target_track,
            selected_raw=selected_raw,
            selected_target=selected_target,
            random_raw_union=random_raw_union,
            random_target_union=random_target_union,
            random_subset_positions=random_subset_positions,
            model_names=model_names,
            include_random=(method_idx == 0),
        )
        all_raw_eval.update(raw_sets)
        all_target_eval.update(target_sets)
        eval_metadata.update(metadata)

    score_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []

    candidate_ops = {}
    for candidate in model_names:
        x = refit_raw[candidate]
        eval_x = {key: raw_by_model[candidate] for key, raw_by_model in all_raw_eval.items()}
        standardized = standardize_from_train(
            x[train_pos],
            x[val_pos],
            *eval_x.values(),
            scale_by_sqrt_features=True,
        )
        x_train = standardized[0]
        x_val = standardized[1]
        eval_standardized = dict(zip(eval_x.keys(), standardized[2:]))
        candidate_ops[candidate] = ridge_ops_for_eval_sets(
            x_train,
            x_val,
            eval_standardized,
            alphas,
        )

    for teacher in model_names:
        clean_y = refit_target[teacher]
        eval_y_clean = {
            key: target_by_model[teacher] for key, target_by_model in all_target_eval.items()
        }
        standardized_y = standardize_from_train(
            clean_y[train_pos],
            clean_y[val_pos],
            *eval_y_clean.values(),
        )
        y_train_clean = standardized_y[0]
        y_val_clean = standardized_y[1]
        eval_y_clean_std = dict(zip(eval_y_clean.keys(), standardized_y[2:]))

        for noise_mult in noise_mults:
            noise_std = base.noise_std_from_multiplier(noise_mult, base_noise_ceiling)
            effective_noise_ceiling = base.multiplier_to_noise_ceiling(
                noise_mult,
                base_noise_ceiling,
            )
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
                eval_y = {
                    key: y_clean + rng.normal(0.0, noise_std, y_clean.shape).astype(np.float32)
                    for key, y_clean in eval_y_clean_std.items()
                }

                candidate_eval_scores: dict[str, dict[str, float]] = {
                    key: {} for key in all_raw_eval
                }
                for candidate in model_names:
                    best_alpha = float(alphas[0])
                    best_val_score = -np.inf
                    for alpha, (val_op, _) in candidate_ops[candidate].items():
                        val_score = base.flat_corr(val_op @ y_train, y_val)
                        if np.isfinite(val_score) and val_score > best_val_score:
                            best_val_score = val_score
                            best_alpha = float(alpha)

                    _, eval_ops = candidate_ops[candidate][best_alpha]
                    for key, eval_op in eval_ops.items():
                        score = base.flat_corr(eval_op @ y_train, eval_y[key])
                        candidate_eval_scores[key][candidate] = score
                        meta = eval_metadata[key]
                        score_rows.append(
                            {
                                **meta,
                                "teacher_model": teacher,
                                "candidate_model": candidate,
                                "best_alpha": best_alpha,
                                "val_score": best_val_score,
                                "test_score": score,
                                "noise_mult": noise_mult,
                                "relative_snr": np.inf if noise_mult <= 0 else 1.0 / noise_mult,
                                "base_noise_ceiling": base_noise_ceiling,
                                "effective_noise_ceiling": effective_noise_ceiling,
                                "noise_std": noise_std,
                                "noise_sample_idx": noise_sample_idx,
                                "refit_pool_size": len(refit_positions),
                                "refit_train_n": refit_train_n,
                                "refit_val_n": refit_val_n,
                            }
                        )

                for key, scores in candidate_eval_scores.items():
                    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
                    recovered, best_score = ordered[0]
                    meta = eval_metadata[key]
                    recovery_rows.append(
                        {
                            **meta,
                            "teacher_model": teacher,
                            "recovered_model": recovered,
                            "recovered_correct": bool(recovered == teacher),
                            "best_test_score": float(best_score),
                            "teacher_self_test_score": float(scores[teacher]),
                            "noise_mult": float(noise_mult),
                            "relative_snr": np.inf if noise_mult <= 0 else 1.0 / float(noise_mult),
                            "base_noise_ceiling": base_noise_ceiling,
                            "effective_noise_ceiling": effective_noise_ceiling,
                            "noise_sample_idx": noise_sample_idx,
                            "refit_pool_size": len(refit_positions),
                            "refit_train_n": refit_train_n,
                            "refit_val_n": refit_val_n,
                        }
                    )

    if len(methods) > 1:
        for rows in (score_rows, recovery_rows):
            random_rows = [
                row
                for row in rows
                if row["method_id"] == random_owner_method
                and row["subset_type"] == "random"
            ]
            for method_id in methods[1:]:
                for row in random_rows:
                    copy = dict(row)
                    copy["method_id"] = method_id
                    rows.append(copy)

    return score_rows, recovery_rows


def summarize_recovery(recovery: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["method_id", "subset_type", "target_track", "noise_mult"]
    for group_key, group in recovery.groupby(keys, sort=False):
        method_id, subset_type, target_track, noise_mult = group_key
        unit = (
            group.groupby(
                ["subset_idx", "noise_sample_idx", "teacher_model"],
                as_index=False,
            )
            .agg(recovered_correct=("recovered_correct", "mean"))
        )
        acc = unit["recovered_correct"].astype(float)
        rows.append(
            {
                "method_id": method_id,
                "method_label": base.METHOD_LABELS.get(method_id, method_id),
                "subset_type": subset_type,
                "target_track": target_track,
                "noise_mult": float(noise_mult),
                "relative_snr": np.inf if float(noise_mult) <= 0 else 1.0 / float(noise_mult),
                "base_noise_ceiling": float(group["base_noise_ceiling"].iloc[0]),
                "effective_noise_ceiling": float(group["effective_noise_ceiling"].iloc[0]),
                "recovery_accuracy": float(acc.mean()),
                "recovery_accuracy_sd": float(acc.std(ddof=1)) if len(acc) > 1 else np.nan,
                "recovery_accuracy_sem": float(acc.std(ddof=1) / np.sqrt(len(acc)))
                if len(acc) > 1
                else np.nan,
                "n_units": int(len(acc)),
                "n_subsets": int(group["subset_idx"].nunique()),
                "n_noise_samples": int(group["noise_sample_idx"].nunique()),
                "n_teachers": int(group["teacher_model"].nunique()),
                "refit_pool_size": int(group["refit_pool_size"].iloc[0]),
                "refit_train_n": int(group["refit_train_n"].iloc[0]),
                "refit_val_n": int(group["refit_val_n"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def plot_curves(summary: pd.DataFrame, out_dir: Path, methods: list[str], target_space: str) -> list[Path]:
    if summary.empty:
        return []
    plot_df = base.aggregate_for_plot(summary)
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(7.8, 4.2), constrained_layout=True)
    for method_id in methods:
        method_df = plot_df[plot_df["method_id"] == method_id]
        color = base.METHOD_COLORS.get(method_id, "#777777")
        for subset_type, linestyle, alpha, linewidth in [
            ("selected", "-", 0.95, 2.1),
            ("random", "--", 0.55, 1.5),
        ]:
            sub = method_df[method_df["subset_type"] == subset_type].sort_values("relative_snr")
            if sub.empty:
                continue
            ax.plot(
                sub["relative_snr"],
                sub["recovery_accuracy"],
                color=color,
                linestyle=linestyle,
                alpha=alpha,
                linewidth=linewidth,
                label=base.METHOD_LABELS.get(method_id, method_id) if subset_type == "selected" else None,
            )
            sem = sub["recovery_accuracy_sem"].to_numpy(dtype=float)
            x = sub["relative_snr"].to_numpy(dtype=float)
            y = sub["recovery_accuracy"].to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(sem)
            if valid.any():
                ax.fill_between(
                    x[valid],
                    np.clip(y[valid] - sem[valid], 0.0, 1.0),
                    np.clip(y[valid] + sem[valid], 0.0, 1.0),
                    color=color,
                    alpha=0.10 if subset_type == "selected" else 0.05,
                    linewidth=0,
                )
    ax.axhline(1.0 / 6.0, color="#555555", linestyle=":", linewidth=0.9, alpha=0.7)
    ax.axvline(1.0, color="#444444", linestyle="-", linewidth=0.8, alpha=0.35)
    ax.set_xscale("log")
    ax.set_xlim(0.009, 11.0)
    ax.set_xticks([0.01, 0.1, 1.0, 10.0])
    ax.set_xticklabels(["0.01", "0.1", "1", "10"])
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Relative SNR")
    ax.set_ylabel("Teacher recovery accuracy")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.7)
    ax.set_title(f"Independent-refit teacher/student recovery ({target_space}, 1k refit)")
    ax.legend(frameon=False, loc="lower right", fontsize=7, ncol=2)

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if target_space == "encoded" else f"_{target_space}_space"
    pdf = out_dir / f"teacher_student_independent_refit_1k_curves{suffix}.pdf"
    png = out_dir / f"teacher_student_independent_refit_1k_curves{suffix}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=base.DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--fig-dir", type=Path, default=base.SWEEP_ROOT / "figures")
    parser.add_argument("--methods", default=",".join(base.DEFAULT_METHODS))
    parser.add_argument("--target-space", choices=["encoded", "raw"], default="encoded")
    parser.add_argument("--target-track", default=None)
    parser.add_argument("--target-tracks", default=None)
    parser.add_argument("--random-feature-dir", type=Path, default=DEFAULT_RANDOM_FEATURE_DIR)
    parser.add_argument("--encoding-root", type=Path, default=base.DEFAULT_ENCODING_ROOT)
    parser.add_argument("--n-random-images", type=int, default=100000)
    parser.add_argument("--refit-pool-size", type=int, default=1000)
    parser.add_argument("--refit-val-size", type=int, default=200)
    parser.add_argument("--n-random-subsets", type=int, default=20)
    parser.add_argument("--n-noise-samples", type=int, default=5)
    parser.add_argument("--alphas", default="0.001,0.01,0.1,1,10,100")
    parser.add_argument("--noise-ceiling", type=float, default=None)
    parser.add_argument("--noise-mults", default=",".join(str(x) for x in base.DEFAULT_NOISE_MULTS))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = (
        args.out_dir
        or (
            run_dir
            / (
                "teacher_student_independent_refit_1k"
                if args.target_space == "encoded"
                else "teacher_student_independent_refit_1k_raw_space"
            )
        )
    ).resolve()
    fig_dir = args.fig_dir.resolve()
    random_feature_dir = args.random_feature_dir.resolve()
    encoding_root = args.encoding_root.resolve()
    methods = base.parse_csv_list(args.methods)
    if args.target_space == "raw":
        target_tracks = ["raw"]
    else:
        target_tracks = (
            base.parse_csv_list(args.target_tracks)
            or base.parse_csv_list(args.target_track)
            or ["sub-01"]
        )
    alphas = [float(x) for x in base.parse_csv_list(args.alphas)]
    noise_mults = [float(x) for x in base.parse_csv_list(args.noise_mults)]
    out_dir.mkdir(parents=True, exist_ok=True)

    first_payload = base.load_payload(run_dir, methods[0])
    model_names = list(first_payload["model_names"])
    payloads = {method: base.load_payload(run_dir, method) for method in methods}
    for method_id, payload in payloads.items():
        if list(payload["model_names"]) != model_names:
            raise ValueError(f"Model order mismatch in {method_id}")
    noise_ceiling = (
        float(args.noise_ceiling)
        if args.noise_ceiling is not None
        else float(first_payload.get("config", {}).get("noise_ceiling_target", 0.46))
    )

    print(f"Models: {model_names}")
    print(f"Target space: {args.target_space}")
    print(f"Target tracks: {target_tracks}; base noise ceiling: {noise_ceiling:g}")
    print(f"Random cache: {random_feature_dir}")
    random_raw = base.load_random_raw_features(
        random_feature_dir,
        model_names,
        args.n_random_images,
    )
    n_available = min(arr.shape[0] for arr in random_raw.values())
    n_selected = next(iter(first_payload["selected_features_raw"].values())).shape[0]
    rng = np.random.default_rng(args.seed)
    if args.refit_pool_size + args.n_random_subsets * n_selected > n_available:
        raise ValueError("Not enough random images for disjoint refit and random eval pools")
    refit_indices = rng.choice(n_available, size=args.refit_pool_size, replace=False)
    remaining = np.setdiff1d(np.arange(n_available), refit_indices, assume_unique=False)
    random_subset_indices = [
        rng.choice(remaining, size=n_selected, replace=False)
        for _ in range(args.n_random_subsets)
    ]
    union_indices = np.unique(np.concatenate([refit_indices, *random_subset_indices]))
    union_lookup = {int(idx): pos for pos, idx in enumerate(union_indices)}
    refit_positions = np.asarray([union_lookup[int(idx)] for idx in refit_indices], dtype=np.int64)
    random_subset_positions = [
        np.asarray([union_lookup[int(idx)] for idx in subset], dtype=np.int64)
        for subset in random_subset_indices
    ]
    random_raw_union = {model: arr[union_indices] for model, arr in random_raw.items()}

    metadata = {
        "run_dir": str(run_dir),
        "methods": methods,
        "target_space": args.target_space,
        "target_tracks": target_tracks,
        "model_names": model_names,
        "random_feature_dir": str(random_feature_dir),
        "encoding_root": str(encoding_root),
        "n_random_images": args.n_random_images,
        "refit_pool_size": args.refit_pool_size,
        "refit_train_n": args.refit_pool_size - args.refit_val_size,
        "refit_val_n": args.refit_val_size,
        "n_random_subsets": args.n_random_subsets,
        "n_noise_samples": args.n_noise_samples,
        "alphas": alphas,
        "base_noise_ceiling": noise_ceiling,
        "noise_mults": noise_mults,
        "seed": args.seed,
        "note": (
            "Independent-refit teacher/student analysis: readouts are fitted on "
            "a 1k natural-image pool from the 100k cache and evaluated on held-out "
            "selected/random image sets."
        ),
    }
    with (out_dir / "teacher_student_independent_refit_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    all_score_rows: list[dict[str, Any]] = []
    all_recovery_rows: list[dict[str, Any]] = []
    for target_idx, target_track in enumerate(target_tracks):
        print(f"Target {target_idx + 1}/{len(target_tracks)}: {target_track}")
        if args.target_space == "encoded":
            encoding_params = base.load_encoding_params(
                encoding_root,
                base.MODEL_LIST_CSV,
                model_names,
                target_track,
            )
        else:
            encoding_params = None
        score_rows, recovery_rows = run_track(
            methods=methods,
            payloads=payloads,
            target_track=target_track,
            target_space=args.target_space,
            model_names=model_names,
            random_raw_union=random_raw_union,
            refit_positions=refit_positions,
            random_subset_positions=random_subset_positions,
            encoding_params=encoding_params,
            alphas=alphas,
            noise_mults=noise_mults,
            n_noise_samples=args.n_noise_samples,
            refit_train_n=args.refit_pool_size - args.refit_val_size,
            refit_val_n=args.refit_val_size,
            base_noise_ceiling=noise_ceiling,
            seed=args.seed + target_idx * 100000,
        )
        all_score_rows.extend(score_rows)
        all_recovery_rows.extend(recovery_rows)
        pd.DataFrame(all_score_rows).to_csv(out_dir / "teacher_student_scores.csv", index=False)
        pd.DataFrame(all_recovery_rows).to_csv(out_dir / "teacher_student_recoveries.csv", index=False)

    scores = pd.DataFrame(all_score_rows)
    recovery = pd.DataFrame(all_recovery_rows)
    summary = summarize_recovery(recovery)
    auc_summary = base.summarize_auc(summary)
    confusion = base.confusion_table(recovery, model_names)
    scores.to_csv(out_dir / "teacher_student_scores.csv", index=False)
    recovery.to_csv(out_dir / "teacher_student_recoveries.csv", index=False)
    summary.to_csv(out_dir / "teacher_student_recovery_summary.csv", index=False)
    auc_summary.to_csv(out_dir / "teacher_student_recovery_auc_summary.csv", index=False)
    confusion.to_csv(out_dir / "teacher_student_confusion_matrix.csv", index=False)
    plot_paths = plot_curves(summary, fig_dir, methods, args.target_space)
    for path in [
        out_dir / "teacher_student_scores.csv",
        out_dir / "teacher_student_recoveries.csv",
        out_dir / "teacher_student_recovery_summary.csv",
        out_dir / "teacher_student_recovery_auc_summary.csv",
        out_dir / "teacher_student_confusion_matrix.csv",
        *plot_paths,
    ]:
        print(path)


if __name__ == "__main__":
    main()
