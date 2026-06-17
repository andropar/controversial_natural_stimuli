#!/usr/bin/env python
"""Pair-level CLIP probe for toy controversiality.

This analysis asks whether CLIP pair features can predict where two objective
models disagree about pairwise image distances. It uses cached NumPy features
only and does not load image models or touch the GPU.
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
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
    n_train_pairs: int
    n_test_pairs: int
    pair_pool_size: int
    pair_batch_size: int
    top_k: int
    set_size: int
    init_size: int
    n_eval_sets: int
    n_random_eval_sets: int
    ridge_alphas: tuple[float, ...]


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--n-train-pairs", type=int, default=50000)
    parser.add_argument("--n-test-pairs", type=int, default=30000)
    parser.add_argument("--pair-pool-size", type=int, default=1000)
    parser.add_argument("--pair-batch-size", type=int, default=50000)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--set-size", type=int, default=20)
    parser.add_argument("--init-size", type=int, default=3)
    parser.add_argument("--n-eval-sets", type=int, default=10)
    parser.add_argument("--n-random-eval-sets", type=int, default=100)
    parser.add_argument("--ridge-alphas", type=str, default="0.1,1,10,100,1000,10000")
    args = parser.parse_args()
    values = vars(args)
    values["ridge_alphas"] = tuple(
        float(x) for x in values["ridge_alphas"].split(",") if x.strip()
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
    return l2_normalize(resnet), l2_normalize(dino), l2_normalize(clip)


def sample_pairs(
    pool_indices: np.ndarray,
    n_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    a = rng.choice(pool_indices, size=n_pairs, replace=True)
    b = rng.choice(pool_indices, size=n_pairs, replace=True)
    same = a == b
    while same.any():
        b[same] = rng.choice(pool_indices, size=int(same.sum()), replace=True)
        same = a == b
    swap = a > b
    a2 = a.copy()
    a[swap] = b[swap]
    b[swap] = a2[swap]
    return a.astype(np.int64), b.astype(np.int64)


def cosine_distance_for_pairs(
    norm_features: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
) -> np.ndarray:
    sim = (norm_features[idx_a] * norm_features[idx_b]).sum(axis=1)
    return (1.0 - sim).astype(np.float32, copy=False)


def pair_distances(
    norm_a: np.ndarray,
    norm_b: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dist_a = cosine_distance_for_pairs(norm_a, idx_a, idx_b)
    dist_b = cosine_distance_for_pairs(norm_b, idx_a, idx_b)
    return dist_a, dist_b


def standardized_pair_targets(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    mean_a: float,
    std_a: float,
    mean_b: float,
    std_b: float,
) -> tuple[np.ndarray, np.ndarray]:
    z_a = (dist_a - mean_a) / max(std_a, 1e-8)
    z_b = (dist_b - mean_b) / max(std_b, 1e-8)
    signed_delta = z_a - z_b
    abs_delta = np.abs(signed_delta)
    return signed_delta.astype(np.float32), abs_delta.astype(np.float32)


def pair_features(
    norm_clip: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
) -> np.ndarray:
    x = norm_clip[idx_a]
    y = norm_clip[idx_b]
    return np.concatenate([np.abs(x - y), x * y], axis=1).astype(np.float32)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if denom <= 1e-12:
        return np.nan
    return float(np.dot(x, y) / denom)


def fit_ridge_with_validation(
    x: np.ndarray,
    y: np.ndarray,
    alphas: tuple[float, ...],
    rng: np.random.Generator,
) -> tuple[StandardScaler, Ridge, dict[str, float]]:
    perm = rng.permutation(len(y))
    n_fit = int(round(0.8 * len(y)))
    fit_idx = perm[:n_fit]
    val_idx = perm[n_fit:]

    scaler = StandardScaler()
    x_fit = scaler.fit_transform(x[fit_idx])
    x_val = scaler.transform(x[val_idx])
    y_fit = y[fit_idx]
    y_val = y[val_idx]

    best_alpha = None
    best_r2 = -np.inf
    best_pred = None
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(x_fit, y_fit)
        pred = model.predict(x_val)
        score = r2_score(y_val, pred)
        if score > best_r2:
            best_r2 = score
            best_alpha = alpha
            best_pred = pred

    final_scaler = StandardScaler()
    x_all = final_scaler.fit_transform(x)
    final_model = Ridge(alpha=float(best_alpha))
    final_model.fit(x_all, y)

    metrics = {
        "n_pairs": float(len(y)),
        "n_fit": float(len(fit_idx)),
        "n_val": float(len(val_idx)),
        "alpha": float(best_alpha),
        "val_r2": float(best_r2),
        "val_mae": float(mean_absolute_error(y_val, best_pred)),
        "val_pearson": safe_corr(y_val, best_pred),
        "val_spearman": float(pd.Series(y_val).corr(pd.Series(best_pred), method="spearman")),
    }
    return final_scaler, final_model, metrics


def predict_pairs(
    scaler: StandardScaler,
    model: Ridge,
    norm_clip: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    preds = np.empty(len(idx_a), dtype=np.float32)
    for start in range(0, len(idx_a), batch_size):
        end = min(start + batch_size, len(idx_a))
        x = pair_features(norm_clip, idx_a[start:end], idx_b[start:end])
        preds[start:end] = model.predict(scaler.transform(x)).astype(np.float32)
    return preds


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
    return float(1.0 - safe_corr(a, b))


def score_candidates(
    norm_a: np.ndarray,
    norm_b: np.ndarray,
    selected: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    selected = np.asarray(selected, dtype=np.int64)
    candidates = np.asarray(candidates, dtype=np.int64)
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
    corr = cov / np.sqrt(np.maximum(var_a * var_b, 1e-24))
    return (1.0 - corr).astype(np.float32, copy=False)


def exact_greedy_select(
    norm_a: np.ndarray,
    norm_b: np.ndarray,
    pool_indices: np.ndarray,
    rng: np.random.Generator,
    set_size: int,
    init_size: int,
) -> np.ndarray:
    selected = rng.choice(pool_indices, size=init_size, replace=False).astype(np.int64)
    selected_set = set(int(x) for x in selected)
    while len(selected) < set_size:
        candidates = np.array(
            [idx for idx in pool_indices if int(idx) not in selected_set],
            dtype=np.int64,
        )
        scores = score_candidates(norm_a, norm_b, selected, candidates)
        best_idx = int(candidates[int(np.nanargmax(scores))])
        selected = np.append(selected, best_idx)
        selected_set.add(best_idx)
    return selected


def build_predicted_pair_matrix(
    scaler: StandardScaler,
    model: Ridge,
    norm_clip: np.ndarray,
    pool_indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    n = len(pool_indices)
    matrix = np.zeros((n, n), dtype=np.float32)
    tri_i, tri_j = np.triu_indices(n, k=1)
    for start in range(0, len(tri_i), batch_size):
        end = min(start + batch_size, len(tri_i))
        idx_a = pool_indices[tri_i[start:end]]
        idx_b = pool_indices[tri_j[start:end]]
        pred = predict_pairs(scaler, model, norm_clip, idx_a, idx_b, batch_size)
        matrix[tri_i[start:end], tri_j[start:end]] = pred
        matrix[tri_j[start:end], tri_i[start:end]] = pred
    return matrix


def predicted_pair_greedy(
    pool_indices: np.ndarray,
    score_matrix: np.ndarray,
    rng: np.random.Generator,
    set_size: int,
    init_size: int,
) -> np.ndarray:
    n = len(pool_indices)
    selected = list(rng.choice(np.arange(n), size=init_size, replace=False))
    available = np.ones(n, dtype=bool)
    available[selected] = False
    while len(selected) < set_size:
        gains = score_matrix[:, selected].sum(axis=1)
        gains[~available] = -np.inf
        best_pos = int(np.argmax(gains))
        selected.append(best_pos)
        available[best_pos] = False
    return pool_indices[np.array(selected, dtype=np.int64)]


def evaluate_set_construction(
    norm_a: np.ndarray,
    norm_b: np.ndarray,
    pool_indices: np.ndarray,
    score_matrix: np.ndarray,
    rng: np.random.Generator,
    cfg: Config,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    exemplars = {}
    for i in range(cfg.n_random_eval_sets):
        selected = rng.choice(pool_indices, size=cfg.set_size, replace=False)
        rows.append(
            {
                "condition": "random",
                "replicate": i,
                "controversiality": controversiality(norm_a, norm_b, selected),
            }
        )

    for i in range(cfg.n_eval_sets):
        selected = predicted_pair_greedy(
            pool_indices,
            score_matrix,
            rng,
            cfg.set_size,
            cfg.init_size,
        )
        if i == 0:
            exemplars["pair_clip_greedy_first"] = selected
        rows.append(
            {
                "condition": "pair_clip_greedy",
                "replicate": i,
                "controversiality": controversiality(norm_a, norm_b, selected),
            }
        )

    degree = score_matrix.sum(axis=1) / max(1, len(pool_indices) - 1)
    top_k_pool = pool_indices[np.argsort(degree)[::-1][: min(cfg.top_k, len(pool_indices))]]
    for i in range(cfg.n_eval_sets):
        selected = exact_greedy_select(
            norm_a,
            norm_b,
            top_k_pool,
            rng,
            cfg.set_size,
            cfg.init_size,
        )
        if i == 0:
            exemplars["pair_clip_top_k_exact_first"] = selected
        rows.append(
            {
                "condition": "pair_clip_top_k_exact",
                "replicate": i,
                "controversiality": controversiality(norm_a, norm_b, selected),
            }
        )

    random_k_size = min(cfg.top_k, len(pool_indices))
    for i in range(cfg.n_eval_sets):
        random_k_pool = rng.choice(pool_indices, size=random_k_size, replace=False)
        selected = exact_greedy_select(
            norm_a,
            norm_b,
            random_k_pool,
            rng,
            cfg.set_size,
            cfg.init_size,
        )
        if i == 0:
            exemplars["random_k_exact_first"] = selected
        rows.append(
            {
                "condition": "random_k_exact",
                "replicate": i,
                "controversiality": controversiality(norm_a, norm_b, selected),
            }
        )

    for i in range(cfg.n_eval_sets):
        selected = exact_greedy_select(
            norm_a,
            norm_b,
            pool_indices,
            rng,
            cfg.set_size,
            cfg.init_size,
        )
        if i == 0:
            exemplars["exact_full_first"] = selected
        rows.append(
            {
                "condition": "exact_full",
                "replicate": i,
                "controversiality": controversiality(norm_a, norm_b, selected),
            }
        )
    return pd.DataFrame(rows), exemplars


def pair_rank_curve(frame: pd.DataFrame, pred_col: str, true_col: str) -> pd.DataFrame:
    decile = pd.qcut(frame[pred_col], q=10, labels=False, duplicates="drop")
    out = frame.assign(pred_decile=decile)
    return (
        out.groupby("pred_decile", observed=True)
        .agg(
            n=(true_col, "size"),
            pred_mean=(pred_col, "mean"),
            true_mean=(true_col, "mean"),
            true_sem=(
                true_col,
                lambda x: float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0,
            ),
        )
        .reset_index()
    )


def plot_pair_scatter(test_predictions: pd.DataFrame, fig_dir: Path) -> None:
    rng = np.random.default_rng(0)
    sample = test_predictions.sample(
        n=min(6000, len(test_predictions)), random_state=int(rng.integers(1_000_000))
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    axes[0].scatter(sample["signed_delta"], sample["pred_signed_delta"], s=3, alpha=0.18)
    axes[0].set_xlabel("true signed delta")
    axes[0].set_ylabel("predicted signed delta")
    axes[0].set_title("Signed disagreement")
    axes[1].scatter(sample["abs_delta"], sample["pred_abs_delta"], s=3, alpha=0.18)
    axes[1].set_xlabel("true abs delta")
    axes[1].set_ylabel("predicted abs delta")
    axes[1].set_title("Absolute disagreement")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "pair_clip_prediction_scatter.pdf")
    plt.close(fig)


def plot_pair_rank_curve(curve: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.errorbar(
        curve["pred_decile"],
        curve["true_mean"],
        yerr=curve["true_sem"],
        marker="o",
        color="#4C78A8",
    )
    ax.set_xlabel("predicted abs disagreement decile")
    ax.set_ylabel("mean true abs disagreement")
    ax.set_title("Pair-level CLIP rank curve")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "pair_clip_abs_rank_curve.pdf")
    plt.close(fig)


def plot_set_eval(eval_summary: pd.DataFrame, fig_dir: Path) -> None:
    order = [
        "random",
        "pair_clip_greedy",
        "random_k_exact",
        "pair_clip_top_k_exact",
        "exact_full",
    ]
    colors = {
        "random": "#9E9E9E",
        "pair_clip_greedy": "#E45756",
        "random_k_exact": "#B279A2",
        "pair_clip_top_k_exact": "#72B7B2",
        "exact_full": "#4C78A8",
    }
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for x, cond in enumerate(order):
        vals = eval_summary.loc[eval_summary["condition"] == cond, "controversiality"].to_numpy()
        jitter = np.linspace(-0.1, 0.1, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(
            np.full(len(vals), x) + jitter,
            vals,
            s=20,
            alpha=0.75,
            color=colors[cond],
            edgecolor="none",
        )
        ax.plot([x - 0.2, x + 0.2], [np.mean(vals), np.mean(vals)], color="black")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(
        ["random", "pair CLIP greedy", "random-K + exact", "pair CLIP top-K + exact", "exact full"],
        rotation=15,
    )
    ax.set_ylabel("actual controversiality")
    ax.set_title("Pair-level CLIP set construction")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "pair_clip_set_acquisition_eval.pdf")
    plt.close(fig)


def plot_top_pair_grid(
    sample_root: Path,
    pairs: pd.DataFrame,
    fig_dir: Path,
    pred_col: str,
    filename: str,
    title: str,
    n_pairs: int = 10,
) -> None:
    top = pairs.sort_values(pred_col, ascending=False).head(n_pairs)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(4.8, 1.8 * n_pairs))
    for row_id, (_, row) in enumerate(top.iterrows()):
        for col, key in enumerate(["image_i", "image_j"]):
            ax = axes[row_id, col]
            idx = int(row[key])
            image_path = sample_root / "images" / f"{idx}.jpg"
            try:
                with Image.open(image_path) as img:
                    ax.imshow(img.convert("RGB"))
            except Exception:
                ax.text(0.5, 0.5, f"missing\n{idx}", ha="center", va="center")
            ax.set_title(
                f"{idx} | pred={row[pred_col]:.3f} true_abs={row['abs_delta']:.3f}",
                fontsize=7,
            )
            ax.axis("off")
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(fig_dir / filename)
    plt.close(fig)


def main() -> None:
    cfg = parse_args()
    results_dir = cfg.output_root / "results"
    fig_dir = cfg.output_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    norm_resnet, norm_dino, norm_clip = load_features(cfg.sample_root)
    n_images = norm_resnet.shape[0]
    all_indices = np.arange(n_images, dtype=np.int64)
    shuffled = rng.permutation(all_indices)
    n_train = int(round(cfg.train_fraction * n_images))
    train_pool = np.sort(shuffled[:n_train])
    test_pool = np.sort(shuffled[n_train:])

    train_i, train_j = sample_pairs(train_pool, cfg.n_train_pairs, rng)
    test_i, test_j = sample_pairs(test_pool, cfg.n_test_pairs, rng)
    train_dist_resnet, train_dist_dino = pair_distances(
        norm_resnet, norm_dino, train_i, train_j
    )
    test_dist_resnet, test_dist_dino = pair_distances(
        norm_resnet, norm_dino, test_i, test_j
    )
    train_resnet_mean = float(train_dist_resnet.mean())
    train_resnet_std = float(train_dist_resnet.std())
    train_dino_mean = float(train_dist_dino.mean())
    train_dino_std = float(train_dist_dino.std())
    train_signed, train_abs = standardized_pair_targets(
        train_dist_resnet,
        train_dist_dino,
        train_resnet_mean,
        train_resnet_std,
        train_dino_mean,
        train_dino_std,
    )
    test_signed, test_abs = standardized_pair_targets(
        test_dist_resnet,
        test_dist_dino,
        train_resnet_mean,
        train_resnet_std,
        train_dino_mean,
        train_dino_std,
    )
    x_train = pair_features(norm_clip, train_i, train_j)

    signed_scaler, signed_model, signed_val_metrics = fit_ridge_with_validation(
        x_train, train_signed, cfg.ridge_alphas, rng
    )
    abs_scaler, abs_model, abs_val_metrics = fit_ridge_with_validation(
        x_train, train_abs, cfg.ridge_alphas, rng
    )
    pred_signed = predict_pairs(
        signed_scaler,
        signed_model,
        norm_clip,
        test_i,
        test_j,
        cfg.pair_batch_size,
    )
    pred_abs = predict_pairs(
        abs_scaler,
        abs_model,
        norm_clip,
        test_i,
        test_j,
        cfg.pair_batch_size,
    )

    pair_metrics = {
        "signed_test_r2": float(r2_score(test_signed, pred_signed)),
        "signed_test_mae": float(mean_absolute_error(test_signed, pred_signed)),
        "signed_test_pearson": safe_corr(test_signed, pred_signed),
        "signed_test_spearman": float(
            pd.Series(test_signed).corr(pd.Series(pred_signed), method="spearman")
        ),
        "abs_test_r2": float(r2_score(test_abs, pred_abs)),
        "abs_test_mae": float(mean_absolute_error(test_abs, pred_abs)),
        "abs_test_pearson": safe_corr(test_abs, pred_abs),
        "abs_test_spearman": float(
            pd.Series(test_abs).corr(pd.Series(pred_abs), method="spearman")
        ),
    }
    for key, value in signed_val_metrics.items():
        pair_metrics[f"signed_val_{key}"] = value
    for key, value in abs_val_metrics.items():
        pair_metrics[f"abs_val_{key}"] = value
    pd.DataFrame([pair_metrics]).to_csv(
        results_dir / "pair_clip_disagreement_metrics.csv",
        index=False,
    )

    test_predictions = pd.DataFrame(
        {
            "image_i": test_i,
            "image_j": test_j,
            "signed_delta": test_signed,
            "abs_delta": test_abs,
            "resnet_distance": test_dist_resnet,
            "dino_distance": test_dist_dino,
            "pred_signed_delta": pred_signed,
            "pred_abs_delta": pred_abs,
        }
    )
    test_predictions.to_csv(
        results_dir / "pair_clip_test_predictions.csv",
        index=False,
    )

    abs_curve = pair_rank_curve(test_predictions, "pred_abs_delta", "abs_delta")
    abs_curve.to_csv(results_dir / "pair_clip_abs_rank_curve.csv", index=False)

    pair_pool = np.sort(
        rng.choice(test_pool, size=min(cfg.pair_pool_size, len(test_pool)), replace=False)
    )
    score_matrix = build_predicted_pair_matrix(
        abs_scaler,
        abs_model,
        norm_clip,
        pair_pool,
        cfg.pair_batch_size,
    )
    degree = score_matrix.sum(axis=1) / max(1, len(pair_pool) - 1)
    pd.DataFrame(
        {
            "image_index": pair_pool,
            "predicted_abs_disagreement_degree": degree,
        }
    ).to_csv(results_dir / "pair_clip_pool_image_scores.csv", index=False)

    set_eval, exemplars = evaluate_set_construction(
        norm_resnet,
        norm_dino,
        pair_pool,
        score_matrix,
        rng,
        cfg,
    )
    set_eval.to_csv(results_dir / "pair_clip_set_acquisition_eval.csv", index=False)
    exemplar_rows = []
    for condition, indices in exemplars.items():
        for pos, idx in enumerate(indices):
            exemplar_rows.append(
                {"condition": condition, "position": pos, "image_index": int(idx)}
            )
    pd.DataFrame(exemplar_rows).to_csv(
        results_dir / "pair_clip_set_exemplar_sets.csv",
        index=False,
    )

    metadata = {
        "config": {
            **asdict(cfg),
            "sample_root": str(cfg.sample_root),
            "output_root": str(cfg.output_root),
        },
        "n_images": int(n_images),
        "n_train_images": int(len(train_pool)),
        "n_test_images": int(len(test_pool)),
        "objective_model_a": "torchvision_resnet50_imagenet1k_v1_layer-2",
        "objective_model_b": "dinov2_vitl14_layer-1",
        "pair_surrogate_feature_space": "openclip_vit_l_14_quickgelu_metaclip_400m_layer0",
        "pair_feature_definition": "concat(abs(clip_i - clip_j), clip_i * clip_j)",
        "pair_target_definition": "zscore_train(resnet_distance) - zscore_train(dino_distance)",
        "train_resnet_distance_mean": train_resnet_mean,
        "train_resnet_distance_std": train_resnet_std,
        "train_dino_distance_mean": train_dino_mean,
        "train_dino_distance_std": train_dino_std,
    }
    with (results_dir / "pair_clip_run_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    plot_pair_scatter(test_predictions, fig_dir)
    plot_pair_rank_curve(abs_curve, fig_dir)
    plot_set_eval(set_eval, fig_dir)
    plot_top_pair_grid(
        cfg.sample_root,
        test_predictions,
        fig_dir,
        "pred_abs_delta",
        "top_predicted_abs_disagreement_pairs.pdf",
        "Top predicted pairwise ResNet-vs-DINO disagreement",
    )

    print("pair-level CLIP metrics")
    print(pd.DataFrame([pair_metrics]).to_string(index=False))
    print("pair-level set acquisition")
    print(set_eval.groupby("condition")["controversiality"].agg(["mean", "std", "count"]))
    print("Wrote results to", results_dir)
    print("Wrote figures to", fig_dir)


if __name__ == "__main__":
    main()
