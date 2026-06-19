#!/usr/bin/env python3
"""CPU teacher/student alignment panel check.

This keeps the previous teacher/student setup but writes the full
teacher-by-candidate alignment scores needed for a Fig. 2A style
selected-vs-random panel. Runtime and precision are controlled by the command
line settings, so the same script can run a tiny smoke test or a larger check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve()
ROOT = next(p for p in SCRIPT.parents if (p / "src" / "cstims").exists())
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cstims import paths  # noqa: E402
from cstims.evaluation.io import load_payload  # noqa: E402
from cstims.evaluation.noise_calibration import (  # noqa: E402
    multiplier_to_noise_ceiling,
    response_noise_std_from_mode,
)
from cstims.evaluation.payload import filter_payload_to_models  # noqa: E402
from cstims.evaluation.random_features import (  # noqa: E402
    available_random_models,
    load_random_feature_cache,
)
from cstims.evaluation.teacher_student import (  # noqa: E402
    build_candidate_ops,
    build_eval_raw_and_meta,
    detect_equivalent_models,
    parse_csv_list,
    parse_float_list,
    predict_eval_with_targetwise_alphas,
    safe_name,
    select_targetwise_alpha_indices,
    stable_seed,
)
from cstims.evaluation.ridge import standardize_from_train  # noqa: E402
from cstims.evaluation.track_loading import load_selected_raw_features  # noqa: E402
from cstims.rdm import calculate_correlation_value, get_rdm_vector_np  # noqa: E402


DEFAULT_OUT = (
    paths.find_share_root()
    / "00_stimulus_selection"
    / "selection_evaluation"
    / "final_stimuli_recovery"
    / "teacher_student"
    / "results"
    / "teacher_alignment_panel_pilot"
)

COLOR_RANDOM = "#0072B2"
COLOR_SELECTED = "#D55E00"
MODEL_SHORT = {
    "torchvision_vgg16_imagenet1k_v1": "VGG-16",
    "torchvision_resnet50_imagenet1k_v1": "ResNet-50",
    "torchvision_convnext_base_imagenet1k_v1": "ConvNeXt-B",
    "torchvision_vit_l_16_imagenet1k_v1": "ViT-L/16",
    "cornet_s": "CORnet-S",
    "vissl_resnet50_supervised": "Supervised",
    "vissl_resnet50_barlowtwins": "BarlowTwins",
    "vissl_resnet50_mocov2": "MoCoV2",
    "vicreg_resnet50": "VICReg",
    "robustness_imagenet_l2_eps3": "Robust-L2",
    "slip_vit_l_slip": "SLIP",
    "slip_vit_l_simclr": "SimCLR-ViT",
    "timm_vit_large_patch14_clip_224_laion2b": "CLIP-L2B",
    "dinov2_vitl14": "DINOv2",
    "openclip_vit_so400m_14_siglip_webli": "SigLIP",
    "openclip_vit_l_14_quickgelu_metaclip_400m": "MC-400M",
    "openclip_vit_l_14_quickgelu_metaclip_fullcc": "MC-Full",
    "timm_vit_large_patch14_clip_224_dfn2b": "DFN-2B",
    "timm_vit_large_patch14_clip_quickgelu_224_openai": "CLIP-OAI",
    "openclip_vit_l_14_laion400m_e31": "CLIP-L400",
}


def draw_box(ax, x: float, vals: np.ndarray, *, color: str, width: float = 0.24) -> None:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    lo, hi = float(vals.min()), float(vals.max())
    ax.add_patch(
        plt.Rectangle(
            (x - width / 2, q1),
            width,
            q3 - q1,
            facecolor=color,
            edgecolor=color,
            alpha=0.18,
            linewidth=0.8,
            zorder=2,
        )
    )
    ax.vlines(x, lo, hi, colors=color, linewidth=0.75, zorder=2)
    ax.hlines([lo, med, hi], x - width * 0.32, x + width * 0.32, colors=color, linewidth=0.75, zorder=3)
    ax.scatter(
        np.full(len(vals), x),
        vals,
        s=5,
        color=color,
        alpha=0.28,
        edgecolors="none",
        zorder=4,
    )


def model_panel_table(scores: pd.DataFrame) -> pd.DataFrame:
    selected = scores[scores["subset_type"].eq("selected")].copy()
    random = scores[scores["subset_type"].eq("random")].copy()
    random = (
        random.groupby(
            [
                "eval_refit_mode",
                "teacher_model",
                "candidate_model",
                "noise_sample_idx",
                "refit_repeat_idx",
            ],
            as_index=False,
        )["alignment_score"]
        .mean()
        .assign(subset_type="random")
    )
    selected = selected[
        [
            "eval_refit_mode",
            "teacher_model",
            "candidate_model",
            "noise_sample_idx",
            "refit_repeat_idx",
            "subset_type",
            "alignment_score",
        ]
    ]
    return pd.concat([selected, random], ignore_index=True)


def median_pairwise_diff(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    diff = np.abs(values[:, None] - values[None, :])
    return float(np.median(diff[np.triu_indices_from(diff, k=1)]))


def plot_panel(scores: pd.DataFrame, out_dir: Path) -> None:
    panel = model_panel_table(scores)
    modes = list(dict.fromkeys(panel["eval_refit_mode"].tolist()))
    fig, axes = plt.subplots(
        1,
        len(modes),
        figsize=(max(8.0, 4.5 * len(modes)), 3.6),
        constrained_layout=True,
        sharey=True,
    )
    if len(modes) == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        sub = panel[panel["eval_refit_mode"].eq(mode)].copy()
        selected_mean = (
            sub[sub["subset_type"].eq("selected")]
            .groupby("candidate_model")["alignment_score"]
            .mean()
            .sort_values(ascending=False)
        )
        order = selected_mean.index.tolist()
        offset = 0.18
        x = np.arange(len(order))
        model_means: dict[str, list[float]] = {"random": [], "selected": []}
        for i, model in enumerate(order):
            block = sub[sub["candidate_model"].eq(model)]
            random_vals = block[block["subset_type"].eq("random")]
            selected_vals = block[block["subset_type"].eq("selected")]
            draw_box(
                ax,
                x[i] - offset,
                random_vals["alignment_score"].to_numpy(),
                color=COLOR_RANDOM,
            )
            draw_box(
                ax,
                x[i] + offset,
                selected_vals["alignment_score"].to_numpy(),
                color=COLOR_SELECTED,
            )
            model_means["random"].append(float(random_vals["alignment_score"].mean()))
            model_means["selected"].append(float(selected_vals["alignment_score"].mean()))
            paired = (
                block.pivot_table(
                    index=["teacher_model", "noise_sample_idx", "refit_repeat_idx"],
                    columns="subset_type",
                    values="alignment_score",
                    aggfunc="mean",
                )
                .dropna()
            )
            for _idx, row in paired.iterrows():
                ax.plot(
                    [x[i] - offset, x[i] + offset],
                    [row["random"], row["selected"]],
                    color="#777777",
                    linewidth=0.25,
                    alpha=0.20,
                    zorder=1,
                )

        range_x = len(order) + 0.7
        for color, condition, xpos in [
            (COLOR_RANDOM, "random", range_x - 0.10),
            (COLOR_SELECTED, "selected", range_x + 0.10),
        ]:
            vals = np.asarray(model_means[condition], dtype=float)
            lo, med, hi = np.nanmin(vals), np.nanmedian(vals), np.nanmax(vals)
            ax.vlines(xpos, lo, hi, colors=color, linewidth=0.95, zorder=5)
            ax.hlines([lo, med, hi], xpos - 0.045, xpos + 0.045, colors=color, linewidth=0.95, zorder=6)

        rand_spread = median_pairwise_diff(np.asarray(model_means["random"]))
        sel_spread = median_pairwise_diff(np.asarray(model_means["selected"]))
        ax.text(
            0.01,
            0.98,
            f"spread random={rand_spread:.3f}\nspread selected={sel_spread:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7,
        )
        ax.set_title(mode.replace("_", " "), fontsize=9, fontweight="bold")
        ax.set_xticks([*x, range_x])
        ax.set_xticklabels(
            [MODEL_SHORT.get(model, model) for model in order] + ["range"],
            rotation=58,
            ha="right",
            fontsize=6,
        )
        ax.grid(axis="y", color="#ECECEC", linewidth=0.55, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Teacher alignment (Spearman rho)")
    handles = [
        plt.Line2D([0], [0], color=COLOR_RANDOM, linewidth=2, label="Random"),
        plt.Line2D([0], [0], color=COLOR_SELECTED, linewidth=2, label="Selected"),
    ]
    axes[0].legend(handles=handles, frameon=False, loc="lower left", fontsize=7)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "teacher_alignment_panel_pilot.png", dpi=220)
    fig.savefig(out_dir / "teacher_alignment_panel_pilot.pdf")
    plt.close(fig)


def compute_mode(
    *,
    mode: str,
    model_set: str,
    selected_raw: dict[str, np.ndarray],
    random_raw: dict[str, np.ndarray],
    model_names: list[str],
    metric: str,
    corr_type: str,
    alphas: list[float],
    n_random_subsets: int,
    refit_pool_size: int,
    refit_val_size: int,
    max_refit_pool_size: int,
    n_noise_samples: int,
    noise_mult: float,
    base_noise_ceiling: float,
    target_dim: int | None,
    exclude_teacher_candidates: bool,
    seed: int,
) -> list[dict[str, Any]]:
    n_selected = next(iter(selected_raw.values())).shape[0]
    n_available = min(arr.shape[0] for arr in random_raw.values())
    if max_refit_pool_size + n_random_subsets * n_selected > n_available:
        raise ValueError("Not enough random images for requested pilot split")

    repeat_rng = np.random.default_rng(seed + stable_seed(model_set, "alignment_panel_pilot"))
    natural_pool_order = repeat_rng.permutation(n_available)
    refit_indices = natural_pool_order[:refit_pool_size]
    random_eval_pool = natural_pool_order[
        max_refit_pool_size : max_refit_pool_size + n_random_subsets * n_selected
    ]
    random_subset_indices = [
        random_eval_pool[subset_idx * n_selected : (subset_idx + 1) * n_selected]
        for subset_idx in range(n_random_subsets)
    ]
    union_indices = np.unique(np.concatenate([refit_indices, *random_subset_indices]))
    union_lookup = {int(idx): pos for pos, idx in enumerate(union_indices)}
    refit_positions = np.asarray([union_lookup[int(idx)] for idx in refit_indices], dtype=np.int64)
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
    split_rng = np.random.default_rng(seed + stable_seed(model_set, mode, "pilot_refit_split"))
    refit_perm = split_rng.permutation(len(refit_positions))
    train_pos = refit_positions[refit_perm[: refit_pool_size - refit_val_size]]
    val_pos = refit_positions[refit_perm[refit_pool_size - refit_val_size : refit_pool_size]]
    base_fit_pos = np.concatenate([train_pos, val_pos]) if mode == "eval_augmented_loo" else None

    print(f"[{mode}] building candidate ops", flush=True)
    candidate_ops = build_candidate_ops(
        random_raw_union=random_raw_union,
        eval_raw=eval_raw,
        refit_positions=refit_positions,
        train_pos=train_pos,
        val_pos=val_pos,
        base_fit_pos=base_fit_pos,
        model_names=model_names,
        alphas=alphas,
        eval_refit_mode=mode,
    )
    equivalence_labels = detect_equivalent_models(random_raw_union, model_names)

    eval_target: dict[str, dict[str, np.ndarray]] = {"selected|0": selected_raw}
    for subset_idx, pos in enumerate(random_subset_positions):
        key = f"random|{subset_idx}"
        eval_target[key] = {model: random_raw_union[model][pos] for model in model_names}

    rows: list[dict[str, Any]] = []
    noise_ceiling = multiplier_to_noise_ceiling(noise_mult, base_noise_ceiling)
    for teacher_idx, teacher in enumerate(model_names):
        print(f"[{mode}] teacher {teacher_idx + 1}/{len(model_names)} {teacher}", flush=True)
        teacher_rng = np.random.default_rng(seed + stable_seed(model_set, mode, teacher, "pilot_noise"))
        clean_y = random_raw_union[teacher]
        eval_y = {key: target_by_model[teacher] for key, target_by_model in eval_target.items()}
        if target_dim is not None and 0 < target_dim < clean_y.shape[1]:
            target_rng = np.random.default_rng(seed + stable_seed(model_set, mode, teacher, "target_cols"))
            target_cols = np.sort(target_rng.choice(clean_y.shape[1], size=target_dim, replace=False))
            clean_y = clean_y[:, target_cols]
            eval_y = {key: y[:, target_cols] for key, y in eval_y.items()}

        if mode == "eval_augmented_loo":
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

        response_noise_std = response_noise_std_from_mode(
            noise_mult,
            base_noise_ceiling,
            "response",
        )
        for noise_sample_idx in range(n_noise_samples):
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
            if mode == "eval_augmented_loo":
                if y_base_fit_clean is None:
                    raise RuntimeError("Missing eval-augmented base targets")
                if y_base_fit_clean.shape[0] == y_train_clean.shape[0] + y_val_clean.shape[0]:
                    y_base_fit = np.concatenate([y_train, y_val], axis=0)
                else:
                    y_base_fit = y_base_fit_clean + teacher_rng.normal(
                        0.0,
                        response_noise_std,
                        y_base_fit_clean.shape,
                    ).astype(np.float32)
                eval_y_fit = {
                    key: y_clean
                    + teacher_rng.normal(0.0, response_noise_std, y_clean.shape).astype(np.float32)
                    for key, y_clean in eval_y_clean.items()
                }
            else:
                y_base_fit = None
                eval_y_fit = {}

            noisy_teacher_rdms = {}
            for key, y_clean in eval_y_clean.items():
                y_eval_noisy = y_clean + teacher_rng.normal(
                    0.0,
                    response_noise_std,
                    y_clean.shape,
                ).astype(np.float32)
                noisy_teacher_rdms[key] = get_rdm_vector_np(y_eval_noisy, metric)

            for candidate_idx, candidate in enumerate(model_names):
                if (
                    exclude_teacher_candidates
                    and equivalence_labels[candidate_idx] == equivalence_labels[teacher_idx]
                ):
                    continue
                alpha_values, best_alpha_idx, coefficient_cache = select_targetwise_alpha_indices(
                    candidate_ops[candidate],
                    y_train,
                    y_val,
                )
                for key in eval_y_clean:
                    pred = predict_eval_with_targetwise_alphas(
                        alpha_ops=candidate_ops[candidate],
                        alpha_values=alpha_values,
                        best_alpha_idx=best_alpha_idx,
                        eval_key=key,
                        y_train=y_train,
                        eval_refit_mode=mode,
                        y_base_fit=y_base_fit,
                        eval_y_fit=eval_y_fit.get(key) if eval_y_fit else None,
                        coefficient_cache=coefficient_cache,
                    )
                    pred_rdm = get_rdm_vector_np(pred, metric)
                    score = calculate_correlation_value(
                        pred_rdm,
                        noisy_teacher_rdms[key],
                        corr_type,
                    )
                    subset_type, subset_idx = eval_meta[key]
                    rows.append(
                        {
                            "model_set": model_set,
                            "eval_refit_mode": mode,
                            "track": "raw",
                            "metric": metric,
                            "corr_type": corr_type,
                            "subset_type": subset_type,
                            "subset_idx": int(subset_idx),
                            "teacher_model": teacher,
                            "candidate_model": candidate,
                            "teacher_idx": int(teacher_idx),
                            "candidate_idx": int(candidate_idx),
                            "teacher_equivalence_label": int(equivalence_labels[teacher_idx]),
                            "candidate_equivalence_label": int(equivalence_labels[candidate_idx]),
                            "alignment_score": float(score),
                            "noise_mult": float(noise_mult),
                            "noise_ceiling": float(noise_ceiling),
                            "noise_sample_idx": int(noise_sample_idx),
                            "refit_repeat_idx": 0,
                            "refit_pool_size": int(refit_pool_size),
                            "refit_train_n": int(refit_pool_size - refit_val_size),
                            "refit_val_n": int(refit_val_size),
                            "target_dim": int(target_dim or clean_y.shape[1]),
                            "response_noise_std": float(response_noise_std),
                            "exclude_teacher_candidates": bool(exclude_teacher_candidates),
                        }
                    )
    return rows


def write_summary(scores: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    panel = model_panel_table(scores)
    rows = []
    for (mode, subset_type), block in panel.groupby(["eval_refit_mode", "subset_type"]):
        model_means = block.groupby("candidate_model")["alignment_score"].mean().to_numpy(dtype=float)
        rows.append(
            {
                "eval_refit_mode": mode,
                "subset_type": subset_type,
                "mean_alignment": float(block["alignment_score"].mean()),
                "sem_alignment": float(block["alignment_score"].std(ddof=1) / np.sqrt(len(block)))
                if len(block) > 1
                else np.nan,
                "model_spread_median_pairwise": median_pairwise_diff(model_means),
                "n_rows": int(len(block)),
                "n_teachers": int(block["teacher_model"].nunique()),
                "n_candidates": int(block["candidate_model"].nunique()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "teacher_alignment_panel_pilot_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-set", default="all_models")
    parser.add_argument("--selection-root", type=Path, default=paths.selected_stimuli_root())
    parser.add_argument(
        "--random-feature-dir",
        type=Path,
        default=Path("shared/cache_or_heavy/natural_pool_subset_100k_seed42"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-random-images", type=int, default=100000)
    parser.add_argument("--refit-pool-size", type=int, default=200)
    parser.add_argument("--refit-val-size", type=int, default=50)
    parser.add_argument("--max-refit-pool-size", type=int, default=1000)
    parser.add_argument("--n-random-subsets", type=int, default=3)
    parser.add_argument("--n-noise-samples", type=int, default=1)
    parser.add_argument("--noise-mult", type=float, default=1.0)
    parser.add_argument("--target-dim", type=int, default=128)
    parser.add_argument(
        "--exclude-teacher-candidates",
        action="store_true",
        help=(
            "Leave-one-model-out panel: when a model is the teacher, exclude "
            "candidate models with the same equivalence label from scoring."
        ),
    )
    parser.add_argument("--alphas", default="0.001,0.01,0.1,1,10,100")
    parser.add_argument("--metric", default=None)
    parser.add_argument("--corr-type", choices=["pearson", "spearman"], default="spearman")
    parser.add_argument("--modes", default="independent,eval_augmented_loo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = load_payload(args.selection_root / args.model_set)
    model_names = list(payload["model_names"])
    available = available_random_models(args.random_feature_dir, model_names)
    model_names = available
    payload = filter_payload_to_models(payload, model_names)
    metric = args.metric or payload.get("config", {}).get("metric", "cosine")
    base_noise_ceiling = float(payload.get("config", {}).get("noise_ceiling_target", 0.46))
    alphas = parse_float_list(args.alphas)
    modes = parse_csv_list(args.modes)

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

    all_rows: list[dict[str, Any]] = []
    for mode in modes:
        if mode not in {"independent", "eval_augmented_loo"}:
            raise ValueError(f"Unsupported mode: {mode}")
        rows = compute_mode(
            mode=mode,
            model_set=args.model_set,
            selected_raw=selected_raw,
            random_raw=random_raw,
            model_names=model_names,
            metric=metric,
            corr_type=args.corr_type,
            alphas=alphas,
            n_random_subsets=args.n_random_subsets,
            refit_pool_size=args.refit_pool_size,
            refit_val_size=args.refit_val_size,
            max_refit_pool_size=args.max_refit_pool_size,
            n_noise_samples=args.n_noise_samples,
            noise_mult=args.noise_mult,
            base_noise_ceiling=base_noise_ceiling,
            target_dim=args.target_dim,
            exclude_teacher_candidates=args.exclude_teacher_candidates,
            seed=args.seed,
        )
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(
            args.output_dir / "teacher_alignment_panel_pilot_scores.csv",
            index=False,
        )

    scores = pd.DataFrame(all_rows)
    summary = write_summary(scores, args.output_dir)
    plot_panel(scores, args.output_dir)
    metadata = {
        "model_set": args.model_set,
        "model_names": model_names,
        "eval_refit_modes": modes,
        "track": "raw",
        "metric": metric,
        "corr_type": args.corr_type,
        "noise_mult": args.noise_mult,
        "base_noise_ceiling": base_noise_ceiling,
        "refit_pool_size": args.refit_pool_size,
        "refit_val_size": args.refit_val_size,
        "max_refit_pool_size": args.max_refit_pool_size,
        "n_random_subsets": args.n_random_subsets,
        "n_noise_samples": args.n_noise_samples,
        "target_dim": args.target_dim,
        "exclude_teacher_candidates": bool(args.exclude_teacher_candidates),
        "alphas": alphas,
        "seed": args.seed,
        "runtime_seconds": time.time() - t0,
        "note": "CPU teacher-alignment panel check; interpret precision from the run settings.",
    }
    (args.output_dir / "teacher_alignment_panel_pilot_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {len(scores)} rows -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
