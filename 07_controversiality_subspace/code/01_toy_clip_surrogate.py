#!/usr/bin/env python
"""CPU-only toy controversiality-subspace analysis.

This script intentionally uses cached feature arrays only. It does not import
PyTorch or load any image models.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_SAMPLE_ROOT = Path(
    "/data/home_roth/_stachelschwein/rsa_based_selection/data/LAION_natural_sample"
)
DEFAULT_ANALYSIS_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    sample_root: Path
    output_root: Path
    seed: int
    train_fraction: float
    set_size: int
    init_size: int
    n_train_sets: int
    n_eval_sets: int
    n_random_eval_sets: int
    top_k: int
    n_rank_contexts: int
    rank_context_size: int
    n_marginal_contexts: int
    marginal_context_size: int
    diversity_weights: tuple[float, ...]
    negative_pos_ratio: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--set-size", type=int, default=20)
    parser.add_argument("--init-size", type=int, default=3)
    parser.add_argument("--n-train-sets", type=int, default=10)
    parser.add_argument("--n-eval-sets", type=int, default=10)
    parser.add_argument("--n-random-eval-sets", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--n-rank-contexts", type=int, default=10)
    parser.add_argument("--rank-context-size", type=int, default=10)
    parser.add_argument("--n-marginal-contexts", type=int, default=50)
    parser.add_argument("--marginal-context-size", type=int, default=10)
    parser.add_argument("--diversity-weights", type=str, default="0.25,0.5,1.0,2.0")
    parser.add_argument("--negative-pos-ratio", type=int, default=10)
    args = parser.parse_args()
    values = vars(args)
    values["diversity_weights"] = tuple(
        float(x) for x in values["diversity_weights"].split(",") if x.strip()
    )
    return Config(**values)


def l2_normalize(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def load_features(sample_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_root = sample_root / "features"
    resnet = np.load(
        feature_root / "torchvision_resnet50_imagenet1k_v1_layer-2.npy",
        mmap_mode="r",
    )
    dino = np.load(feature_root / "dinov2_vitl14_layer-1.npy", mmap_mode="r")
    clip = np.load(
        feature_root / "openclip_vit_l_14_quickgelu_metaclip_400m_layer0.npy",
        mmap_mode="r",
    )
    return l2_normalize(resnet), l2_normalize(dino), np.asarray(clip, dtype=np.float32)


def rdm_vector(norm_features: np.ndarray, indices: np.ndarray) -> np.ndarray:
    x = norm_features[indices]
    sim = x @ x.T
    tri = np.triu_indices(len(indices), k=1)
    return (1.0 - sim[tri]).astype(np.float64, copy=False)


def controversiality(
    norm_a: np.ndarray,
    norm_b: np.ndarray,
    indices: np.ndarray,
) -> float:
    a = rdm_vector(norm_a, indices)
    b = rdm_vector(norm_b, indices)
    if len(a) < 2:
        return np.nan
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
    if denom <= 1e-12:
        return np.nan
    return float(1.0 - (np.dot(a, b) / denom))


def score_candidates(
    norm_a: np.ndarray,
    norm_b: np.ndarray,
    selected: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """Score C(selected union candidate) for all candidates."""
    selected = np.asarray(selected, dtype=np.int64)
    candidates = np.asarray(candidates, dtype=np.int64)
    if len(selected) < 2:
        raise ValueError("Need at least two selected images for candidate scoring.")

    base_a = rdm_vector(norm_a, selected)
    base_b = rdm_vector(norm_b, selected)
    base_n = len(base_a)
    k = len(selected)
    n = base_n + k

    cand_a = 1.0 - (norm_a[candidates] @ norm_a[selected].T)
    cand_b = 1.0 - (norm_b[candidates] @ norm_b[selected].T)
    cand_a = cand_a.astype(np.float64, copy=False)
    cand_b = cand_b.astype(np.float64, copy=False)

    sum_a = base_a.sum() + cand_a.sum(axis=1)
    sum_b = base_b.sum() + cand_b.sum(axis=1)
    sum_a2 = np.dot(base_a, base_a) + (cand_a * cand_a).sum(axis=1)
    sum_b2 = np.dot(base_b, base_b) + (cand_b * cand_b).sum(axis=1)
    sum_ab = np.dot(base_a, base_b) + (cand_a * cand_b).sum(axis=1)

    cov = sum_ab - (sum_a * sum_b / n)
    var_a = sum_a2 - (sum_a * sum_a / n)
    var_b = sum_b2 - (sum_b * sum_b / n)
    denom = np.sqrt(np.maximum(var_a * var_b, 1e-24))
    corr = cov / denom
    return (1.0 - corr).astype(np.float32, copy=False)


def greedy_select(
    norm_a: np.ndarray,
    norm_b: np.ndarray,
    pool_indices: np.ndarray,
    rng: np.random.Generator,
    set_size: int,
    init_size: int,
    run_label: str,
) -> tuple[np.ndarray, list[dict[str, float | int | str]]]:
    selected = rng.choice(pool_indices, size=init_size, replace=False).astype(np.int64)
    selected_set = set(int(x) for x in selected)
    trace: list[dict[str, float | int | str]] = []

    initial_score = controversiality(norm_a, norm_b, selected)
    for pos, idx in enumerate(selected):
        trace.append(
            {
                "run": run_label,
                "step": pos,
                "image_index": int(idx),
                "phase": "init",
                "score": float(initial_score),
            }
        )

    while len(selected) < set_size:
        candidates = np.array(
            [idx for idx in pool_indices if int(idx) not in selected_set],
            dtype=np.int64,
        )
        scores = score_candidates(norm_a, norm_b, selected, candidates)
        best_pos = int(np.nanargmax(scores))
        best_idx = int(candidates[best_pos])
        selected = np.append(selected, best_idx)
        selected_set.add(best_idx)
        trace.append(
            {
                "run": run_label,
                "step": int(len(selected) - 1),
                "image_index": best_idx,
                "phase": "greedy",
                "score": float(scores[best_pos]),
            }
        )

    return selected, trace


def sample_random_scores(
    norm_a: np.ndarray,
    norm_b: np.ndarray,
    pool_indices: np.ndarray,
    rng: np.random.Generator,
    n_sets: int,
    set_size: int,
    label: str,
) -> list[dict[str, float | int | str]]:
    rows = []
    for i in range(n_sets):
        subset = rng.choice(pool_indices, size=set_size, replace=False)
        rows.append(
            {
                "condition": label,
                "replicate": i,
                "controversiality": controversiality(norm_a, norm_b, subset),
            }
        )
    return rows


def balanced_negative_sample(
    rng: np.random.Generator,
    candidates: np.ndarray,
    n_pos: int,
    ratio: int,
) -> np.ndarray:
    n_neg = min(len(candidates), max(n_pos * ratio, n_pos))
    return rng.choice(candidates, size=n_neg, replace=False)


def train_surrogate(
    clip_features: np.ndarray,
    selected_by_run: list[np.ndarray],
    train_pool: np.ndarray,
    rng: np.random.Generator,
    negative_pos_ratio: int,
) -> tuple[object, dict[str, float], np.ndarray]:
    n_runs = len(selected_by_run)
    if n_runs > 1:
        n_fit_runs = min(n_runs - 1, max(1, int(np.floor(n_runs * 0.8))))
    else:
        n_fit_runs = 1
    fit_pos = np.unique(np.concatenate(selected_by_run[:n_fit_runs]))
    eval_pos = (
        np.unique(np.concatenate(selected_by_run[n_fit_runs:]))
        if n_fit_runs < n_runs
        else np.array([], dtype=np.int64)
    )
    fit_pos_set = set(int(x) for x in fit_pos)
    all_pos_set = set(int(x) for x in np.unique(np.concatenate(selected_by_run)))

    fit_neg_pool = np.array(
        [idx for idx in train_pool if int(idx) not in fit_pos_set],
        dtype=np.int64,
    )
    fit_neg = balanced_negative_sample(
        rng, fit_neg_pool, len(fit_pos), negative_pos_ratio
    )

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=0,
        ),
    )
    fit_idx = np.concatenate([fit_pos, fit_neg])
    fit_y = np.concatenate([np.ones(len(fit_pos)), np.zeros(len(fit_neg))])
    clf.fit(clip_features[fit_idx], fit_y)

    metrics: dict[str, float] = {
        "n_fit_positive": float(len(fit_pos)),
        "n_fit_negative": float(len(fit_neg)),
    }
    if len(eval_pos) > 0:
        eval_neg_pool = np.array(
            [idx for idx in train_pool if int(idx) not in all_pos_set],
            dtype=np.int64,
        )
        eval_neg = balanced_negative_sample(
            rng, eval_neg_pool, len(eval_pos), negative_pos_ratio
        )
        eval_idx = np.concatenate([eval_pos, eval_neg])
        eval_y = np.concatenate([np.ones(len(eval_pos)), np.zeros(len(eval_neg))])
        eval_score = clf.decision_function(clip_features[eval_idx])
        metrics.update(
            {
                "n_eval_positive": float(len(eval_pos)),
                "n_eval_negative": float(len(eval_neg)),
                "heldout_seed_roc_auc": float(roc_auc_score(eval_y, eval_score)),
                "heldout_seed_average_precision": float(
                    average_precision_score(eval_y, eval_score)
                ),
            }
        )

    all_pos = np.unique(np.concatenate(selected_by_run))
    all_pos_set = set(int(x) for x in all_pos)
    all_neg_pool = np.array(
        [idx for idx in train_pool if int(idx) not in all_pos_set],
        dtype=np.int64,
    )
    all_neg = balanced_negative_sample(
        rng, all_neg_pool, len(all_pos), negative_pos_ratio
    )
    final_clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=0,
        ),
    )
    final_idx = np.concatenate([all_pos, all_neg])
    final_y = np.concatenate([np.ones(len(all_pos)), np.zeros(len(all_neg))])
    final_clf.fit(clip_features[final_idx], final_y)
    metrics.update(
        {
            "n_final_positive": float(len(all_pos)),
            "n_final_negative": float(len(all_neg)),
        }
    )
    return final_clf, metrics, all_pos


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if denom <= 1e-12:
        return np.nan
    return float(np.dot(x, y) / denom)


def compute_contextual_marginal_labels(
    norm_a: np.ndarray,
    norm_b: np.ndarray,
    pool_indices: np.ndarray,
    rng: np.random.Generator,
    n_contexts: int,
    context_size: int,
) -> pd.DataFrame:
    """Average exact marginal gain for every image across random contexts."""
    pool_indices = np.asarray(pool_indices, dtype=np.int64)
    labels = np.zeros(len(pool_indices), dtype=np.float64)
    counts = np.zeros(len(pool_indices), dtype=np.float64)
    index_to_pos = np.full(norm_a.shape[0], -1, dtype=np.int64)
    index_to_pos[pool_indices] = np.arange(len(pool_indices), dtype=np.int64)

    for context_id in range(n_contexts):
        context = rng.choice(pool_indices, size=context_size, replace=False)
        base_score = controversiality(norm_a, norm_b, context)
        context_set = set(int(x) for x in context)
        candidates = np.array(
            [idx for idx in pool_indices if int(idx) not in context_set],
            dtype=np.int64,
        )
        scores = score_candidates(norm_a, norm_b, context, candidates)
        pos = index_to_pos[candidates]
        labels[pos] += scores.astype(np.float64) - base_score
        counts[pos] += 1.0
        print(
            f"finished marginal-label context {context_id + 1}/{n_contexts}",
            flush=True,
        )

    labels = labels / np.maximum(counts, 1.0)
    return pd.DataFrame(
        {
            "image_index": pool_indices,
            "marginal_gain": labels,
            "n_contexts_observed": counts.astype(int),
        }
    )


def train_marginal_surrogate(
    clip_features: np.ndarray,
    labels: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[object, dict[str, float]]:
    indices = labels["image_index"].to_numpy(dtype=np.int64)
    y = labels["marginal_gain"].to_numpy(dtype=np.float64)
    perm = rng.permutation(len(indices))
    n_fit = max(1, int(round(0.8 * len(indices))))
    fit_pos = perm[:n_fit]
    eval_pos = perm[n_fit:]

    alphas = np.logspace(-4, 4, 17)
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    model.fit(clip_features[indices[fit_pos]], y[fit_pos])
    eval_pred = model.predict(clip_features[indices[eval_pos]])
    eval_y = y[eval_pos]

    metrics = {
        "n_labels": float(len(indices)),
        "n_fit": float(len(fit_pos)),
        "n_eval": float(len(eval_pos)),
        "label_mean": float(np.mean(y)),
        "label_std": float(np.std(y)),
        "eval_r2": float(r2_score(eval_y, eval_pred)),
        "eval_mae": float(mean_absolute_error(eval_y, eval_pred)),
        "eval_pearson": safe_corr(eval_y, eval_pred),
        "eval_spearman": float(
            pd.Series(eval_y).corr(pd.Series(eval_pred), method="spearman")
        ),
        "ridge_alpha": float(model.named_steps["ridgecv"].alpha_),
    }

    final_model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    final_model.fit(clip_features[indices], y)
    metrics["final_ridge_alpha"] = float(final_model.named_steps["ridgecv"].alpha_)
    return final_model, metrics


def evaluate_acquisition(
    norm_a: np.ndarray,
    norm_b: np.ndarray,
    test_pool: np.ndarray,
    surrogate_scores: np.ndarray,
    rng: np.random.Generator,
    cfg: Config,
    score_prefix: str = "clip",
    score_label: str = "CLIP",
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = sample_random_scores(
        norm_a,
        norm_b,
        test_pool,
        rng,
        cfg.n_random_eval_sets,
        cfg.set_size,
        "random",
    )

    ranked_test = test_pool[np.argsort(surrogate_scores[test_pool])[::-1]]
    top_direct = ranked_test[: cfg.set_size]
    rows.append(
        {
            "condition": f"{score_prefix}_top20",
            "replicate": 0,
            "controversiality": controversiality(norm_a, norm_b, top_direct),
        }
    )

    top_k_pool = ranked_test[: min(cfg.top_k, len(ranked_test))]
    top_k_sets = []
    for i in range(cfg.n_eval_sets):
        selected, _ = greedy_select(
            norm_a,
            norm_b,
            top_k_pool,
            rng,
            cfg.set_size,
            cfg.init_size,
            f"clip_top_k_exact_{i}",
        )
        top_k_sets.append(selected)
        rows.append(
            {
                "condition": f"{score_prefix}_top_k_exact",
                "replicate": i,
                "controversiality": controversiality(norm_a, norm_b, selected),
            }
        )
        print(
            f"finished {score_label} top-K exact eval set {i + 1}/{cfg.n_eval_sets}",
            flush=True,
        )

    random_k_sets = []
    random_k_size = min(cfg.top_k, len(test_pool))
    for i in range(cfg.n_eval_sets):
        random_k_pool = rng.choice(test_pool, size=random_k_size, replace=False)
        selected, _ = greedy_select(
            norm_a,
            norm_b,
            random_k_pool,
            rng,
            cfg.set_size,
            cfg.init_size,
            f"random_k_exact_{i}",
        )
        random_k_sets.append(selected)
        rows.append(
            {
                "condition": "random_k_exact",
                "replicate": i,
                "controversiality": controversiality(norm_a, norm_b, selected),
            }
        )
        print(f"finished random-K exact eval set {i + 1}/{cfg.n_eval_sets}", flush=True)

    exact_sets = []
    for i in range(cfg.n_eval_sets):
        selected, _ = greedy_select(
            norm_a,
            norm_b,
            test_pool,
            rng,
            cfg.set_size,
            cfg.init_size,
            f"exact_full_{i}",
        )
        exact_sets.append(selected)
        rows.append(
            {
                "condition": "exact_full",
                "replicate": i,
                "controversiality": controversiality(norm_a, norm_b, selected),
            }
        )
        print(f"finished full exact eval set {i + 1}/{cfg.n_eval_sets}", flush=True)

    sets = {
        f"{score_prefix}_top20": top_direct,
        f"{score_prefix}_top_k_exact_first": top_k_sets[0],
        "random_k_exact_first": random_k_sets[0],
        "exact_full_first": exact_sets[0],
    }
    return pd.DataFrame(rows), sets


def format_weight(weight: float) -> str:
    return str(weight).replace(".", "p").replace("-", "m")


def mmr_select_by_score(
    clip_features: np.ndarray,
    candidate_indices: np.ndarray,
    surrogate_scores: np.ndarray,
    n_select: int,
    diversity_weight: float,
) -> np.ndarray:
    """Select high-score but CLIP-diverse images with greedy MMR."""
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    n_select = min(n_select, len(candidate_indices))
    x = l2_normalize(clip_features[candidate_indices])
    raw_score = surrogate_scores[candidate_indices].astype(np.float64)
    score = (raw_score - raw_score.mean()) / (raw_score.std() + 1e-8)

    selected_positions: list[int] = []
    available = np.ones(len(candidate_indices), dtype=bool)
    max_sim = np.zeros(len(candidate_indices), dtype=np.float32)

    for _ in range(n_select):
        if not selected_positions:
            utility = score.copy()
        else:
            utility = score - diversity_weight * max_sim
        utility[~available] = -np.inf
        best_pos = int(np.argmax(utility))
        selected_positions.append(best_pos)
        available[best_pos] = False
        sim = x @ x[best_pos]
        max_sim = np.maximum(max_sim, sim.astype(np.float32, copy=False))

    return candidate_indices[np.array(selected_positions, dtype=np.int64)]


def evaluate_diverse_marginal_acquisition(
    norm_a: np.ndarray,
    norm_b: np.ndarray,
    clip_features: np.ndarray,
    test_pool: np.ndarray,
    marginal_scores: np.ndarray,
    rng: np.random.Generator,
    cfg: Config,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    sets: dict[str, np.ndarray] = {}
    for weight in cfg.diversity_weights:
        weight_label = format_weight(weight)
        direct = mmr_select_by_score(
            clip_features,
            test_pool,
            marginal_scores,
            cfg.set_size,
            weight,
        )
        direct_condition = f"marginal_diverse_top20_w{weight_label}"
        rows.append(
            {
                "condition": direct_condition,
                "replicate": 0,
                "diversity_weight": weight,
                "controversiality": controversiality(norm_a, norm_b, direct),
            }
        )
        sets[direct_condition] = direct

        diverse_k_pool = mmr_select_by_score(
            clip_features,
            test_pool,
            marginal_scores,
            min(cfg.top_k, len(test_pool)),
            weight,
        )
        exact_condition = f"marginal_diverse_top_k_exact_w{weight_label}"
        exact_sets = []
        for i in range(cfg.n_eval_sets):
            selected, _ = greedy_select(
                norm_a,
                norm_b,
                diverse_k_pool,
                rng,
                cfg.set_size,
                cfg.init_size,
                f"{exact_condition}_{i}",
            )
            exact_sets.append(selected)
            rows.append(
                {
                    "condition": exact_condition,
                    "replicate": i,
                    "diversity_weight": weight,
                    "controversiality": controversiality(norm_a, norm_b, selected),
                }
            )
            print(
                f"finished marginal diverse top-K exact w={weight:g} "
                f"eval set {i + 1}/{cfg.n_eval_sets}",
                flush=True,
            )
        sets[f"{exact_condition}_first"] = exact_sets[0]

    return pd.DataFrame(rows), sets


def compute_rank_curve(
    norm_a: np.ndarray,
    norm_b: np.ndarray,
    test_pool: np.ndarray,
    surrogate_scores: np.ndarray,
    rng: np.random.Generator,
    n_contexts: int,
    context_size: int,
) -> pd.DataFrame:
    marginal = np.zeros(len(test_pool), dtype=np.float64)
    counts = np.zeros(len(test_pool), dtype=np.float64)
    pool_pos = {int(idx): pos for pos, idx in enumerate(test_pool)}

    for _ in range(n_contexts):
        context = rng.choice(test_pool, size=context_size, replace=False)
        base_score = controversiality(norm_a, norm_b, context)
        context_set = set(int(x) for x in context)
        candidates = np.array(
            [idx for idx in test_pool if int(idx) not in context_set],
            dtype=np.int64,
        )
        scores = score_candidates(norm_a, norm_b, context, candidates)
        for idx, score in zip(candidates, scores):
            pos = pool_pos[int(idx)]
            marginal[pos] += float(score - base_score)
            counts[pos] += 1.0

    marginal = marginal / np.maximum(counts, 1.0)
    score = surrogate_scores[test_pool]
    quantiles = pd.qcut(score, q=10, labels=False, duplicates="drop")
    frame = pd.DataFrame(
        {
            "image_index": test_pool,
            "surrogate_score": score,
            "actual_contextual_marginal_gain": marginal,
            "surrogate_decile": quantiles,
        }
    )
    grouped = (
        frame.groupby("surrogate_decile", observed=True)
        .agg(
            n=("image_index", "size"),
            surrogate_score_mean=("surrogate_score", "mean"),
            actual_mean=("actual_contextual_marginal_gain", "mean"),
            actual_sem=(
                "actual_contextual_marginal_gain",
                lambda x: float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0,
            ),
        )
        .reset_index()
    )
    return grouped


def condition_order(score_prefix: str) -> list[str]:
    return [
        "random",
        f"{score_prefix}_top20",
        "random_k_exact",
        f"{score_prefix}_top_k_exact",
        "exact_full",
    ]


def plot_selection_works(selection_summary: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    order = ["random_train", "exact_train"]
    positions = np.arange(len(order))
    for x, cond in zip(positions, order):
        vals = selection_summary.loc[
            selection_summary["condition"] == cond, "controversiality"
        ].to_numpy()
        ax.scatter(
            np.full(len(vals), x) + np.linspace(-0.08, 0.08, len(vals)),
            vals,
            s=22,
            alpha=0.75,
        )
        ax.plot([x - 0.18, x + 0.18], [np.mean(vals), np.mean(vals)], color="black")
    ax.set_xticks(positions)
    ax.set_xticklabels(["random", "exact greedy"])
    ax.set_ylabel("1 - corr(RDM ResNet50, RDM DINOv2)")
    ax.set_title("Toy selection objective")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "toy_selection_works.pdf")
    plt.close(fig)


def plot_classifier_metrics(metrics: dict[str, float], fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    labels = ["ROC AUC", "Average precision"]
    vals = [
        metrics.get("heldout_seed_roc_auc", np.nan),
        metrics.get("heldout_seed_average_precision", np.nan),
    ]
    ax.bar(labels, vals, color=["#4C78A8", "#F58518"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("held-out selection-seed score")
    ax.set_title("CLIP surrogate classifier")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "clip_surrogate_classifier.pdf")
    plt.close(fig)


def plot_acquisition(
    eval_summary: pd.DataFrame,
    fig_dir: Path,
    filename: str,
    score_prefix: str,
    score_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    order = condition_order(score_prefix)
    colors = {
        "random": "#9E9E9E",
        f"{score_prefix}_top20": "#E45756",
        "random_k_exact": "#B279A2",
        f"{score_prefix}_top_k_exact": "#72B7B2",
        "exact_full": "#4C78A8",
    }
    for x, cond in enumerate(order):
        vals = eval_summary.loc[
            eval_summary["condition"] == cond, "controversiality"
        ].to_numpy()
        if len(vals) == 0:
            continue
        jitter = np.linspace(-0.1, 0.1, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(
            np.full(len(vals), x) + jitter,
            vals,
            s=22,
            alpha=0.75,
            color=colors[cond],
            edgecolor="none",
        )
        ax.plot([x - 0.2, x + 0.2], [np.mean(vals), np.mean(vals)], color="black")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(
        [
            "random",
            f"{score_label} top 20",
            "random-K + exact",
            f"{score_label} top-K + exact",
            "exact full",
        ],
        rotation=15,
    )
    ax.set_ylabel("actual controversiality")
    ax.set_title("Held-out acquisition")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / filename)
    plt.close(fig)


def plot_rank_curve(rank_curve: pd.DataFrame, fig_dir: Path, filename: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = rank_curve["surrogate_decile"].to_numpy()
    y = rank_curve["actual_mean"].to_numpy()
    yerr = rank_curve["actual_sem"].to_numpy()
    ax.errorbar(x, y, yerr=yerr, marker="o", lw=1.5, color="#4C78A8")
    ax.set_xlabel("CLIP surrogate score decile")
    ax.set_ylabel("mean actual marginal gain")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / filename)
    plt.close(fig)


def plot_top_bottom_grid(
    sample_root: Path,
    test_pool: np.ndarray,
    surrogate_scores: np.ndarray,
    fig_dir: Path,
    filename: str,
    title: str,
    n_each: int = 10,
) -> None:
    ranked = test_pool[np.argsort(surrogate_scores[test_pool])]
    bottom = ranked[:n_each]
    top = ranked[-n_each:][::-1]
    indices = np.concatenate([top, bottom])
    labels = ["high"] * len(top) + ["low"] * len(bottom)

    n_cols = 5
    n_rows = int(np.ceil(len(indices) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 3.2 * n_rows / 2))
    axes = np.asarray(axes).reshape(-1)
    for ax, idx, label in zip(axes, indices, labels):
        image_path = sample_root / "images" / f"{int(idx)}.jpg"
        try:
            with Image.open(image_path) as img:
                ax.imshow(img.convert("RGB"))
        except Exception:
            ax.text(0.5, 0.5, f"missing\n{idx}", ha="center", va="center")
        ax.set_title(f"{label} {int(idx)}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(indices) :]:
        ax.axis("off")
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(fig_dir / filename)
    plt.close(fig)


def plot_diverse_acquisition(
    diverse_eval: pd.DataFrame,
    marginal_eval: pd.DataFrame,
    fig_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))

    references = (
        marginal_eval.groupby("condition")["controversiality"]
        .mean()
        .to_dict()
    )
    ref_specs = [
        ("random", "random", "#9E9E9E"),
        ("random_k_exact", "random-K + exact", "#B279A2"),
        ("marginal_top_k_exact", "marginal top-K + exact", "#72B7B2"),
        ("exact_full", "full exact", "#4C78A8"),
    ]
    for key, label, color in ref_specs:
        if key in references:
            ax.axhline(references[key], color=color, lw=1.2, alpha=0.75, label=label)

    weights = sorted(diverse_eval["diversity_weight"].dropna().unique())
    direct_means = []
    exact_means = []
    exact_sems = []
    for weight in weights:
        wdf = diverse_eval[diverse_eval["diversity_weight"] == weight]
        direct = wdf[wdf["condition"].str.contains("diverse_top20")]
        exact = wdf[wdf["condition"].str.contains("diverse_top_k_exact")]
        direct_means.append(direct["controversiality"].mean())
        exact_means.append(exact["controversiality"].mean())
        exact_sems.append(
            exact["controversiality"].std(ddof=1) / np.sqrt(len(exact))
            if len(exact) > 1
            else 0.0
        )

    ax.plot(weights, direct_means, marker="o", color="#E45756", label="diverse top 20")
    ax.errorbar(
        weights,
        exact_means,
        yerr=exact_sems,
        marker="o",
        color="#54A24B",
        label="diverse top-K + exact",
    )
    ax.set_xscale("log")
    ax.set_xlabel("MMR diversity weight")
    ax.set_ylabel("actual controversiality")
    ax.set_title("Marginal surrogate with CLIP diversity")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "marginal_diverse_acquisition_eval.pdf")
    plt.close(fig)


def save_selected_sets(
    selected_by_run: list[np.ndarray],
    output_path: Path,
    split: str,
) -> None:
    rows = []
    for run_id, indices in enumerate(selected_by_run):
        for pos, idx in enumerate(indices):
            rows.append(
                {
                    "split": split,
                    "run": run_id,
                    "position": pos,
                    "image_index": int(idx),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    cfg = parse_args()
    results_dir = cfg.output_root / "results"
    fig_dir = cfg.output_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    norm_resnet, norm_dino, clip = load_features(cfg.sample_root)
    n_images = norm_resnet.shape[0]
    all_indices = np.arange(n_images, dtype=np.int64)
    shuffled = rng.permutation(all_indices)
    n_train = int(round(cfg.train_fraction * n_images))
    train_pool = np.sort(shuffled[:n_train])
    test_pool = np.sort(shuffled[n_train:])

    selected_by_run = []
    trace_rows = []
    train_summary_rows = sample_random_scores(
        norm_resnet,
        norm_dino,
        train_pool,
        rng,
        cfg.n_train_sets,
        cfg.set_size,
        "random_train",
    )
    for run_id in range(cfg.n_train_sets):
        selected, trace = greedy_select(
            norm_resnet,
            norm_dino,
            train_pool,
            rng,
            cfg.set_size,
            cfg.init_size,
            f"train_exact_{run_id}",
        )
        selected_by_run.append(selected)
        trace_rows.extend(trace)
        train_summary_rows.append(
            {
                "condition": "exact_train",
                "replicate": run_id,
                "controversiality": controversiality(norm_resnet, norm_dino, selected),
            }
        )
        print(f"finished train exact set {run_id + 1}/{cfg.n_train_sets}", flush=True)

    save_selected_sets(
        selected_by_run,
        results_dir / "toy_selected_sets.csv",
        split="train",
    )
    pd.DataFrame(trace_rows).to_csv(results_dir / "toy_selection_trace.csv", index=False)
    selection_summary = pd.DataFrame(train_summary_rows)
    selection_summary.to_csv(results_dir / "toy_selection_summary.csv", index=False)

    surrogate, metrics, all_selected = train_surrogate(
        clip,
        selected_by_run,
        train_pool,
        rng,
        cfg.negative_pos_ratio,
    )
    surrogate_scores = surrogate.decision_function(clip).astype(np.float32)
    pd.DataFrame([metrics]).to_csv(results_dir / "clip_surrogate_metrics.csv", index=False)
    pd.DataFrame(
        {
            "image_index": all_indices,
            "split": np.where(np.isin(all_indices, train_pool), "train", "test"),
            "selected_training_positive": np.isin(all_indices, all_selected),
            "surrogate_score": surrogate_scores,
        }
    ).to_csv(results_dir / "clip_surrogate_scores.csv", index=False)

    marginal_labels = compute_contextual_marginal_labels(
        norm_resnet,
        norm_dino,
        train_pool,
        rng,
        cfg.n_marginal_contexts,
        cfg.marginal_context_size,
    )
    marginal_labels.to_csv(results_dir / "marginal_surrogate_train_labels.csv", index=False)
    marginal_surrogate, marginal_metrics = train_marginal_surrogate(
        clip,
        marginal_labels,
        rng,
    )
    marginal_scores = marginal_surrogate.predict(clip).astype(np.float32)
    pd.DataFrame([marginal_metrics]).to_csv(
        results_dir / "marginal_surrogate_metrics.csv",
        index=False,
    )
    label_lookup = marginal_labels.set_index("image_index")["marginal_gain"]
    pd.DataFrame(
        {
            "image_index": all_indices,
            "split": np.where(np.isin(all_indices, train_pool), "train", "test"),
            "train_marginal_gain": [
                label_lookup.get(int(idx), np.nan) for idx in all_indices
            ],
            "marginal_surrogate_score": marginal_scores,
        }
    ).to_csv(results_dir / "marginal_surrogate_scores.csv", index=False)

    eval_summary, exemplar_sets = evaluate_acquisition(
        norm_resnet,
        norm_dino,
        test_pool,
        surrogate_scores,
        rng,
        cfg,
        score_prefix="clip",
        score_label="CLIP classifier",
    )
    eval_summary.to_csv(results_dir / "surrogate_acquisition_eval.csv", index=False)
    exemplar_rows = []
    for condition, indices in exemplar_sets.items():
        for pos, idx in enumerate(indices):
            exemplar_rows.append(
                {"condition": condition, "position": pos, "image_index": int(idx)}
            )
    pd.DataFrame(exemplar_rows).to_csv(
        results_dir / "surrogate_acquisition_exemplar_sets.csv",
        index=False,
    )

    marginal_eval_summary, marginal_exemplar_sets = evaluate_acquisition(
        norm_resnet,
        norm_dino,
        test_pool,
        marginal_scores,
        rng,
        cfg,
        score_prefix="marginal",
        score_label="marginal regressor",
    )
    marginal_eval_summary.to_csv(
        results_dir / "marginal_surrogate_acquisition_eval.csv",
        index=False,
    )
    marginal_exemplar_rows = []
    for condition, indices in marginal_exemplar_sets.items():
        for pos, idx in enumerate(indices):
            marginal_exemplar_rows.append(
                {"condition": condition, "position": pos, "image_index": int(idx)}
            )
    pd.DataFrame(marginal_exemplar_rows).to_csv(
        results_dir / "marginal_surrogate_acquisition_exemplar_sets.csv",
        index=False,
    )

    diverse_eval_summary, diverse_exemplar_sets = evaluate_diverse_marginal_acquisition(
        norm_resnet,
        norm_dino,
        clip,
        test_pool,
        marginal_scores,
        rng,
        cfg,
    )
    diverse_eval_summary.to_csv(
        results_dir / "marginal_diverse_acquisition_eval.csv",
        index=False,
    )
    diverse_exemplar_rows = []
    for condition, indices in diverse_exemplar_sets.items():
        for pos, idx in enumerate(indices):
            diverse_exemplar_rows.append(
                {"condition": condition, "position": pos, "image_index": int(idx)}
            )
    pd.DataFrame(diverse_exemplar_rows).to_csv(
        results_dir / "marginal_diverse_acquisition_exemplar_sets.csv",
        index=False,
    )

    rank_curve = compute_rank_curve(
        norm_resnet,
        norm_dino,
        test_pool,
        surrogate_scores,
        rng,
        cfg.n_rank_contexts,
        cfg.rank_context_size,
    )
    rank_curve.to_csv(results_dir / "surrogate_rank_curve.csv", index=False)
    marginal_rank_curve = compute_rank_curve(
        norm_resnet,
        norm_dino,
        test_pool,
        marginal_scores,
        rng,
        cfg.n_rank_contexts,
        cfg.rank_context_size,
    )
    marginal_rank_curve.to_csv(
        results_dir / "marginal_surrogate_rank_curve.csv",
        index=False,
    )

    metadata = {
        "config": {
            **asdict(cfg),
            "sample_root": str(cfg.sample_root),
            "output_root": str(cfg.output_root),
        },
        "n_images": int(n_images),
        "n_train": int(len(train_pool)),
        "n_test": int(len(test_pool)),
        "objective_model_a": "torchvision_resnet50_imagenet1k_v1_layer-2",
        "objective_model_b": "dinov2_vitl14_layer-1",
        "surrogate_feature_space": "openclip_vit_l_14_quickgelu_metaclip_400m_layer0",
    }
    with (results_dir / "run_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    plot_selection_works(selection_summary, fig_dir)
    plot_classifier_metrics(metrics, fig_dir)
    plot_acquisition(
        eval_summary,
        fig_dir,
        "surrogate_acquisition_eval.pdf",
        score_prefix="clip",
        score_label="CLIP classifier",
    )
    plot_acquisition(
        marginal_eval_summary,
        fig_dir,
        "marginal_surrogate_acquisition_eval.pdf",
        score_prefix="marginal",
        score_label="marginal regressor",
    )
    plot_diverse_acquisition(diverse_eval_summary, marginal_eval_summary, fig_dir)
    plot_rank_curve(
        rank_curve,
        fig_dir,
        "surrogate_rank_curve.pdf",
        "Classifier score vs actual marginal gain",
    )
    plot_rank_curve(
        marginal_rank_curve,
        fig_dir,
        "marginal_surrogate_rank_curve.pdf",
        "Marginal regressor score vs actual marginal gain",
    )
    plot_top_bottom_grid(
        cfg.sample_root,
        test_pool,
        surrogate_scores,
        fig_dir,
        "top_bottom_clip_direction.pdf",
        "Top and bottom held-out images along CLIP classifier",
    )
    plot_top_bottom_grid(
        cfg.sample_root,
        test_pool,
        marginal_scores,
        fig_dir,
        "top_bottom_marginal_direction.pdf",
        "Top and bottom held-out images along marginal regressor",
    )

    print("Wrote results to", results_dir)
    print("Wrote figures to", fig_dir)
    print("selected-vs-random classifier metrics")
    print(pd.DataFrame([metrics]).to_string(index=False))
    print("marginal regressor metrics")
    print(pd.DataFrame([marginal_metrics]).to_string(index=False))
    print("selected-vs-random classifier acquisition")
    print(eval_summary.groupby("condition")["controversiality"].agg(["mean", "std", "count"]))
    print("marginal regressor acquisition")
    print(
        marginal_eval_summary.groupby("condition")["controversiality"].agg(
            ["mean", "std", "count"]
        )
    )
    print("marginal regressor with CLIP diversity")
    print(
        diverse_eval_summary.groupby(["condition", "diversity_weight"])[
            "controversiality"
        ].agg(["mean", "std", "count"])
    )


if __name__ == "__main__":
    main()
