#!/usr/bin/env python3
"""Teacher/student recovery with fitted linear readouts.

This is a closer analogue to flexible model-recovery simulations than the
standard noisy-RDM evaluator.  An encoded model is treated as the teacher,
continuous noisy brain-response targets are sampled from its encoded features at
one or more calibrated noise levels, and every candidate model is refit with
ridge regression from raw features to those synthetic targets.  Recovery
succeeds when the teacher model has the best held-out prediction score after
this refit.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
SWEEP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = SWEEP_ROOT / "results" / "sota_20260611_112941"
DEFAULT_RANDOM_FEATURE_DIR = ROOT / "shared" / "cache_or_heavy" / "natural_pool_subset_10k"
DEFAULT_ENCODING_ROOT = (
    ROOT
    / "01_brain_model_alignment"
    / "results"
    / "encoding_models"
    / "shared_subject_encoding_models"
    / "encoding_20251222_141301"
)
MODEL_LIST_CSV = ROOT / "00_stimulus_selection" / "resources" / "model_list.csv"

DEFAULT_METHODS = (
    "paper_effective_identity_sub01_mean_min_no_attenuation",
    "raw_enc_w05_mean_min",
    "raw_only_mean_min",
    "sub01_only_mean_min",
)
ENCODING_TRACKS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")
DEFAULT_NOISE_MULTS = tuple(
    float(x)
    for x in np.sort(
        np.unique(
            np.concatenate(
                [
                    np.logspace(-1, 2, 20),
                    np.array([0.1, 1.0, 3.0, 5.0, 10.0]),
                ]
            )
        )
    )
)

METHOD_LABELS = {
    "paper_effective_identity_sub01_mean_min_no_attenuation": "Sub-01 only (no attenuation)",
    "raw_enc_w05_mean_min": "Intended (Raw + enc, mean/min)",
    "raw_only_mean_min": "Raw features only",
    "sub01_only_mean_min": "Sub-01 only (current)",
}

METHOD_COLORS = {
    "paper_effective_identity_sub01_mean_min_no_attenuation": "#9C755F",
    "raw_enc_w05_mean_min": "#59A14F",
    "raw_only_mean_min": "#4C78A8",
    "sub01_only_mean_min": "#F28E2B",
}

MODEL_LABELS = {
    "slip_vit_l_slip": "SLIP",
    "slip_vit_l_simclr": "SimCLR-ViT",
    "timm_vit_large_patch14_clip_224_laion2b": "CLIP-L2B",
    "dinov2_vitl14": "DINOv2",
    "openclip_vit_so400m_14_siglip_webli": "SigLIP",
    "torchvision_convnext_base_imagenet1k_v1": "ConvNeXt-B",
}


def parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def sanitize_layer_name(layer: str | int) -> str:
    return (
        str(layer)
        .strip()
        .replace(".", "_")
        .replace(":", "_")
        .replace("[", "_")
        .replace("]", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def load_model_layers(model_list_csv: Path) -> dict[str, str]:
    table = pd.read_csv(model_list_csv)
    return dict(zip(table["model"], table["layer"]))


def load_npz_feature_array(path: Path, n_images: int | None = None) -> np.ndarray:
    with np.load(path, allow_pickle=True) as z:
        if "features" in z.files:
            arr = z["features"]
        else:
            keys = [
                key
                for key in z.files
                if not key.startswith("_") and getattr(z[key], "ndim", 0) >= 2
            ]
            if not keys:
                raise ValueError(f"No feature array found in {path}")
            arr = z[keys[0]]
        arr = np.asarray(arr, dtype=np.float32)
    if n_images is not None:
        arr = arr[:n_images]
    return arr


def load_random_raw_features(
    random_feature_dir: Path,
    model_names: list[str],
    n_images: int,
) -> dict[str, np.ndarray]:
    out = {}
    for model in model_names:
        path = random_feature_dir / f"{model}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing random feature cache: {path}")
        out[model] = load_npz_feature_array(path, n_images)
    return out


def load_encoding_params(
    encoding_root: Path,
    model_list_csv: Path,
    model_names: list[str],
    target_track: str,
    *,
    roi_subset: str = "hlvis",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    layers = load_model_layers(model_list_csv)
    params = {}
    roi_mask = None
    for model in model_names:
        if model not in layers:
            raise KeyError(f"Model not in {model_list_csv}: {model}")
        layer_safe = sanitize_layer_name(layers[model])
        path = encoding_root / f"{target_track}_{model}.layer{layer_safe}" / "encoding_model.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing encoding model: {path}")
        with np.load(path) as z:
            weights = np.asarray(z["weights"], dtype=np.float32)
            bias = np.asarray(z["intercept"], dtype=np.float32)
            roi_key = f"roi_{roi_subset}"
            if roi_subset and roi_key in z:
                if roi_mask is None:
                    roi_mask = np.asarray(z[roi_key], dtype=bool)
                weights = weights[:, roi_mask]
                bias = bias[roi_mask]
        params[model] = (weights, bias)
    return params


def encode_raw_features(
    raw_features: dict[str, np.ndarray],
    encoding_params: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    encoded = {}
    for model, features in raw_features.items():
        weights, bias = encoding_params[model]
        encoded[model] = np.asarray(features @ weights + bias, dtype=np.float32)
    return encoded


def load_payload(run_dir: Path, method_id: str) -> dict:
    for root in (run_dir / "cross_eval_full_tracks" / "payloads", run_dir / "payloads"):
        path = root / method_id / "selected_stimuli_data.pkl"
        if path.exists():
            with path.open("rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"No payload found for method {method_id} under {run_dir}")


def selected_arrays_from_payload(
    payload: dict,
    target_track: str,
    model_names: list[str],
    encoding_params: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    raw = payload.get("selected_features_raw")
    encoded_all = payload.get("selected_features_by_encoding") or {}
    if raw is None:
        raw = payload.get("selected_features")
    if raw is None:
        raise KeyError("Payload has no selected raw features")
    raw_out = {model: np.asarray(raw[model], dtype=np.float32) for model in model_names}
    if target_track in encoded_all:
        encoded = encoded_all[target_track]
        encoded_out = {
            model: np.asarray(encoded[model], dtype=np.float32) for model in model_names
        }
    else:
        encoded_out = encode_raw_features(raw_out, encoding_params)
    return raw_out, encoded_out


def selected_raw_from_payload(
    payload: dict,
    model_names: list[str],
) -> dict[str, np.ndarray]:
    raw = payload.get("selected_features_raw")
    if raw is None:
        raw = payload.get("selected_features")
    if raw is None:
        raise KeyError("Payload has no selected raw features")
    return {model: np.asarray(raw[model], dtype=np.float32) for model in model_names}


def standardize_train_apply(
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


def multiplier_to_noise_ceiling(noise_mult: float, nc_base: float) -> float:
    if noise_mult <= 0:
        return 1.0
    if nc_base <= 0 or nc_base >= 1:
        return nc_base
    term = noise_mult * noise_mult * (1.0 / (nc_base * nc_base) - 1.0)
    return float(1.0 / math.sqrt(1.0 + term))


def noise_std_from_multiplier(noise_mult: float, nc_base: float) -> float:
    if noise_mult <= 0 or nc_base <= 0 or nc_base >= 1:
        return 0.0
    return float(noise_mult * math.sqrt(1.0 / (nc_base * nc_base) - 1.0))


def ridge_predict_dual(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    alpha: float,
) -> np.ndarray:
    kernel = x_train @ x_train.T
    kernel.flat[:: kernel.shape[0] + 1] += float(alpha)
    dual = np.linalg.solve(kernel.astype(np.float64), y_train.astype(np.float64))
    return np.asarray((x_eval @ x_train.T).astype(np.float64) @ dual, dtype=np.float32)


def ridge_prediction_operators(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    alphas: list[float],
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Precompute linear maps from train targets to val/test predictions."""
    kernel = (x_train @ x_train.T).astype(np.float64)
    k_val = (x_val @ x_train.T).astype(np.float64)
    k_test = (x_test @ x_train.T).astype(np.float64)
    eye = np.eye(kernel.shape[0], dtype=np.float64)
    operators = {}
    for alpha in alphas:
        inv = np.linalg.inv(kernel + float(alpha) * eye)
        operators[float(alpha)] = (
            np.asarray(k_val @ inv, dtype=np.float32),
            np.asarray(k_test @ inv, dtype=np.float32),
        )
    return operators


def split_indices(
    n_items: int,
    rng: np.random.Generator,
    train_n: int,
    val_n: int,
    test_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_n + val_n + test_n > n_items:
        raise ValueError(
            f"Split sizes ({train_n}+{val_n}+{test_n}) exceed n_items={n_items}"
        )
    perm = rng.permutation(n_items)
    train = perm[:train_n]
    val = perm[train_n : train_n + val_n]
    test = perm[train_n + val_n : train_n + val_n + test_n]
    return train, val, test


def run_subset_recovery(
    *,
    method_id: str,
    subset_type: str,
    subset_idx: int,
    target_track: str,
    raw_by_model: dict[str, np.ndarray],
    target_by_model: dict[str, np.ndarray],
    model_names: list[str],
    n_splits: int,
    n_noise_samples: int,
    train_n: int,
    val_n: int,
    test_n: int,
    alphas: list[float],
    base_noise_ceiling: float,
    noise_mults: list[float],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    n_items = next(iter(raw_by_model.values())).shape[0]

    score_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []

    for split_idx in range(n_splits):
        train_idx, val_idx, test_idx = split_indices(n_items, rng, train_n, val_n, test_n)

        standardized_x = {}
        ridge_ops = {}
        for candidate in model_names:
            x = raw_by_model[candidate]
            standardized_x[candidate] = standardize_train_apply(
                x[train_idx],
                x[val_idx],
                x[test_idx],
                scale_by_sqrt_features=True,
            )
            ridge_ops[candidate] = ridge_prediction_operators(
                *standardized_x[candidate],
                alphas=alphas,
            )

        for teacher in model_names:
            clean_y = target_by_model[teacher]
            y_train_clean, y_val_clean, y_test_clean = standardize_train_apply(
                clean_y[train_idx],
                clean_y[val_idx],
                clean_y[test_idx],
            )
            for noise_mult in noise_mults:
                noise_std = noise_std_from_multiplier(noise_mult, base_noise_ceiling)
                effective_noise_ceiling = multiplier_to_noise_ceiling(
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
                    y_test = y_test_clean + rng.normal(
                        0.0,
                        noise_std,
                        y_test_clean.shape,
                    ).astype(np.float32)

                    noise_sample_rows = []
                    for candidate in model_names:
                        best_alpha = float(alphas[0])
                        best_val_score = -np.inf
                        best_test_pred = None
                        for alpha, (val_op, test_op) in ridge_ops[candidate].items():
                            pred_val = val_op @ y_train
                            val_score = flat_corr(pred_val, y_val)
                            if np.isfinite(val_score) and val_score > best_val_score:
                                best_val_score = val_score
                                best_alpha = float(alpha)
                                best_test_pred = test_op @ y_train
                        if best_test_pred is None:
                            _, test_op = ridge_ops[candidate][best_alpha]
                            best_test_pred = test_op @ y_train
                        pred_test = best_test_pred
                        test_score = flat_corr(pred_test, y_test)
                        row = {
                            "method_id": method_id,
                            "subset_type": subset_type,
                            "subset_idx": subset_idx,
                            "target_track": target_track,
                            "split_idx": split_idx,
                            "noise_sample_idx": noise_sample_idx,
                            "teacher_model": teacher,
                            "candidate_model": candidate,
                            "best_alpha": best_alpha,
                            "val_score": best_val_score,
                            "test_score": test_score,
                            "noise_mult": noise_mult,
                            "relative_snr": np.inf if noise_mult <= 0 else 1.0 / noise_mult,
                            "base_noise_ceiling": base_noise_ceiling,
                            "effective_noise_ceiling": effective_noise_ceiling,
                            "noise_std": noise_std,
                            "n_items": n_items,
                            "train_n": train_n,
                            "val_n": val_n,
                            "test_n": test_n,
                        }
                        noise_sample_rows.append(row)
                        score_rows.append(row)

                    group = pd.DataFrame(noise_sample_rows)
                    best = group.sort_values(
                        ["test_score", "candidate_model"], ascending=[False, True]
                    ).iloc[0]
                    recovery_rows.append(
                        {
                            "method_id": method_id,
                            "subset_type": subset_type,
                            "subset_idx": subset_idx,
                            "target_track": target_track,
                            "split_idx": split_idx,
                            "noise_sample_idx": noise_sample_idx,
                            "teacher_model": teacher,
                            "recovered_model": best["candidate_model"],
                            "recovered_correct": bool(best["candidate_model"] == teacher),
                            "best_test_score": float(best["test_score"]),
                            "teacher_self_test_score": float(
                                group[group["candidate_model"] == teacher]["test_score"].iloc[0]
                            ),
                            "noise_mult": float(noise_mult),
                            "relative_snr": np.inf if noise_mult <= 0 else 1.0 / float(noise_mult),
                            "base_noise_ceiling": base_noise_ceiling,
                            "effective_noise_ceiling": float(best["effective_noise_ceiling"]),
                            "n_items": n_items,
                        }
                    )

    return score_rows, recovery_rows


def summarize_recovery(recovery: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["method_id", "subset_type", "target_track", "noise_mult"]
    for group_key, group in recovery.groupby(keys, sort=False):
        method_id, subset_type, target_track, noise_mult = group_key
        unit_cols = ["subset_idx", "split_idx", "teacher_model"]
        if "noise_sample_idx" in group.columns:
            unit_cols.insert(2, "noise_sample_idx")
        unit = (
            group.groupby(unit_cols, as_index=False)
            .agg(recovered_correct=("recovered_correct", "mean"))
        )
        acc = unit["recovered_correct"].astype(float)
        rows.append(
            {
                "method_id": method_id,
                "method_label": METHOD_LABELS.get(method_id, method_id),
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
                "n_splits": int(group["split_idx"].nunique()),
                "n_teachers": int(group["teacher_model"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def compute_log_noise_auc(noise_mult: pd.Series, values: pd.Series) -> float:
    x = np.asarray(noise_mult, dtype=float)
    y = np.asarray(values, dtype=float)
    valid = np.isfinite(x) & (x > 0) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        return float("nan")
    order = np.argsort(x)
    x_log = np.log10(x[order])
    y_sorted = y[order]
    span = x_log[-1] - x_log[0]
    auc = float(np.trapezoid(y_sorted, x_log))
    return auc / span if span > 0 else auc


def summarize_auc(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["method_id", "method_label", "subset_type", "target_track"]
    for group_key, group in summary.groupby(keys, sort=False):
        method_id, method_label, subset_type, target_track = group_key
        rows.append(
            {
                "method_id": method_id,
                "method_label": method_label,
                "subset_type": subset_type,
                "target_track": target_track,
                "recovery_accuracy_auc": compute_log_noise_auc(
                    group["noise_mult"],
                    group["recovery_accuracy"],
                ),
                "n_noise_levels": int(group["noise_mult"].nunique()),
                "noise_mult_min": float(group["noise_mult"].min()),
                "noise_mult_max": float(group["noise_mult"].max()),
            }
        )
    return pd.DataFrame(rows)


def confusion_table(recovery: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in recovery.groupby(
        ["method_id", "subset_type", "target_track", "noise_mult"],
        sort=False,
    ):
        method_id, subset_type, target_track, noise_mult = keys
        counts = (
            group.groupby(["teacher_model", "recovered_model"])
            .size()
            .rename("count")
            .reset_index()
        )
        total_by_teacher = counts.groupby("teacher_model")["count"].transform("sum")
        counts["proportion"] = counts["count"] / total_by_teacher
        present = {
            (row.teacher_model, row.recovered_model)
            for row in counts.itertuples(index=False)
        }
        rows.extend(
            counts.assign(
                method_id=method_id,
                subset_type=subset_type,
                target_track=target_track,
                noise_mult=float(noise_mult),
                relative_snr=np.inf if float(noise_mult) <= 0 else 1.0 / float(noise_mult),
                effective_noise_ceiling=float(group["effective_noise_ceiling"].iloc[0]),
            ).to_dict("records")
        )
        for teacher in model_names:
            for recovered in model_names:
                if (teacher, recovered) not in present:
                    rows.append(
                        {
                            "method_id": method_id,
                            "subset_type": subset_type,
                            "target_track": target_track,
                            "noise_mult": float(noise_mult),
                            "relative_snr": np.inf if float(noise_mult) <= 0 else 1.0 / float(noise_mult),
                            "effective_noise_ceiling": float(group["effective_noise_ceiling"].iloc[0]),
                            "teacher_model": teacher,
                            "recovered_model": recovered,
                            "count": 0,
                            "proportion": 0.0,
                        }
                    )
    return pd.DataFrame(rows)


def aggregate_for_plot(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["method_id", "method_label", "subset_type", "noise_mult"]
    for keys, group in summary.groupby(group_cols, sort=False):
        method_id, method_label, subset_type, noise_mult = keys
        values = group["recovery_accuracy"].astype(float)
        rows.append(
            {
                "method_id": method_id,
                "method_label": method_label,
                "subset_type": subset_type,
                "noise_mult": float(noise_mult),
                "relative_snr": np.inf if float(noise_mult) <= 0 else 1.0 / float(noise_mult),
                "effective_noise_ceiling": float(group["effective_noise_ceiling"].mean()),
                "recovery_accuracy": float(values.mean()),
                "recovery_accuracy_sem": float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else float(group["recovery_accuracy_sem"].mean()),
                "n_target_tracks": int(group["target_track"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def plot_summary(
    summary: pd.DataFrame,
    out_dir: Path,
    methods: list[str],
    *,
    target_space: str = "encoded",
) -> list[Path]:
    if summary.empty:
        return []
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    plot_df = aggregate_for_plot(summary)
    empirical_mult = float(plot_df.iloc[(plot_df["noise_mult"] - 1.0).abs().argmin()]["noise_mult"])
    plot_df = plot_df[np.isclose(plot_df["noise_mult"], empirical_mult)].copy()
    plot_df["_order"] = plot_df["method_id"].map({m: i for i, m in enumerate(methods)}).fillna(999)
    plot_df = plot_df.sort_values(["_order", "subset_type"])
    x = np.arange(len(methods))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.8, 3.8), constrained_layout=True)
    for offset, subset_type, alpha in [(-width / 2, "selected", 0.92), (width / 2, "random", 0.45)]:
        sub = plot_df[plot_df["subset_type"] == subset_type]
        values = []
        errs = []
        colors = []
        for method_id in methods:
            row = sub[sub["method_id"] == method_id]
            values.append(float(row["recovery_accuracy"].iloc[0]) if not row.empty else np.nan)
            errs.append(float(row["recovery_accuracy_sem"].iloc[0]) if not row.empty else np.nan)
            colors.append(METHOD_COLORS.get(method_id, "#777777"))
        ax.bar(
            x + offset,
            values,
            width,
            yerr=errs,
            color=colors,
            alpha=alpha,
            edgecolor="#222222" if subset_type == "random" else "none",
            linewidth=0.6,
            capsize=2,
            label=subset_type.capitalize(),
        )
    ax.axhline(1.0 / 6.0, color="#555555", linestyle="--", linewidth=0.8, alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [METHOD_LABELS.get(method_id, method_id) for method_id in methods],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Teacher recovery accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_title(f"Teacher/student fitted recovery ({target_space}) at noise x{empirical_mult:g}")

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if target_space == "encoded" else f"_{target_space}_space"
    pdf = out_dir / f"teacher_student_recovery_summary{suffix}.pdf"
    png = out_dir / f"teacher_student_recovery_summary{suffix}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def plot_recovery_curves(
    summary: pd.DataFrame,
    out_dir: Path,
    methods: list[str],
    *,
    target_space: str = "encoded",
) -> list[Path]:
    if summary.empty:
        return []
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    plot_df = aggregate_for_plot(summary)
    fig, ax = plt.subplots(figsize=(7.8, 4.2), constrained_layout=True)
    for method_id in methods:
        method_df = plot_df[plot_df["method_id"] == method_id]
        if method_df.empty:
            continue
        color = METHOD_COLORS.get(method_id, "#777777")
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
                label=METHOD_LABELS.get(method_id, method_id) if subset_type == "selected" else None,
            )
            sem = sub["recovery_accuracy_sem"].to_numpy(dtype=float)
            if np.isfinite(sem).any():
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
    ax.set_title(f"Teacher/student fitted-recovery curves ({target_space})")

    from matplotlib.lines import Line2D

    method_handles = [
        Line2D([0], [0], color=METHOD_COLORS.get(method, "#777777"), linewidth=2.0,
               label=METHOD_LABELS.get(method, method))
        for method in methods
        if method in set(plot_df["method_id"])
    ]
    style_handles = [
        Line2D([0], [0], color="#222222", linestyle="-", linewidth=1.8, label="Selected"),
        Line2D([0], [0], color="#222222", linestyle="--", linewidth=1.4, alpha=0.65, label="Random"),
    ]
    ax.legend(
        handles=method_handles + style_handles,
        frameon=False,
        loc="lower right",
        fontsize=7,
        ncol=2,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if target_space == "encoded" else f"_{target_space}_space"
    pdf = out_dir / f"teacher_student_recovery_curves{suffix}.pdf"
    png = out_dir / f"teacher_student_recovery_curves{suffix}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--fig-dir", type=Path, default=SWEEP_ROOT / "figures")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--target-space", choices=["encoded", "raw"], default="encoded")
    parser.add_argument("--target-track", default=None, help="Backward-compatible single target track.")
    parser.add_argument("--target-tracks", default=None)
    parser.add_argument("--random-feature-dir", type=Path, default=DEFAULT_RANDOM_FEATURE_DIR)
    parser.add_argument("--encoding-root", type=Path, default=DEFAULT_ENCODING_ROOT)
    parser.add_argument("--n-random-images", type=int, default=500)
    parser.add_argument("--n-random-subsets", type=int, default=3)
    parser.add_argument("--n-splits", type=int, default=8)
    parser.add_argument("--n-noise-samples", type=int, default=1)
    parser.add_argument("--train-n", type=int, default=60)
    parser.add_argument("--val-n", type=int, default=20)
    parser.add_argument("--test-n", type=int, default=20)
    parser.add_argument("--alphas", default="0.001,0.01,0.1,1,10,100")
    parser.add_argument("--noise-ceiling", type=float, default=None)
    parser.add_argument("--noise-mults", default=",".join(str(x) for x in DEFAULT_NOISE_MULTS))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    default_out_name = (
        "teacher_student_recovery"
        if args.target_space == "encoded"
        else "teacher_student_recovery_raw_space"
    )
    out_dir = (args.out_dir or (run_dir / default_out_name)).resolve()
    fig_dir = args.fig_dir.resolve()
    random_feature_dir = args.random_feature_dir.resolve()
    encoding_root = args.encoding_root.resolve()
    methods = parse_csv_list(args.methods)
    if args.target_space == "raw":
        target_tracks = ["raw"]
    else:
        target_tracks = parse_csv_list(args.target_tracks) or parse_csv_list(args.target_track) or ["sub-01"]
    alphas = [float(x) for x in parse_csv_list(args.alphas)]
    noise_mults = [float(x) for x in parse_csv_list(args.noise_mults)]
    out_dir.mkdir(parents=True, exist_ok=True)

    first_payload = load_payload(run_dir, methods[0])
    model_names = list(first_payload["model_names"])
    noise_ceiling = (
        float(args.noise_ceiling)
        if args.noise_ceiling is not None
        else float(first_payload.get("config", {}).get("noise_ceiling_target", 0.46))
    )

    print(f"Models: {model_names}")
    print(f"Target space: {args.target_space}")
    print(f"Target tracks: {target_tracks}; base noise ceiling: {noise_ceiling:g}")
    print(f"Noise multipliers: {noise_mults}")
    print(f"Loading random raw features from {random_feature_dir}")
    random_raw = load_random_raw_features(random_feature_dir, model_names, args.n_random_images)
    encoding_params_by_track = {}
    random_target_by_track = {}
    if args.target_space == "encoded":
        for target_track in target_tracks:
            print(f"Loading {target_track} encoding params from {encoding_root}")
            encoding_params_by_track[target_track] = load_encoding_params(
                encoding_root,
                MODEL_LIST_CSV,
                model_names,
                target_track,
            )
            print(f"Encoding random features for {target_track}")
            random_target_by_track[target_track] = encode_raw_features(
                random_raw,
                encoding_params_by_track[target_track],
            )
    else:
        random_target_by_track["raw"] = random_raw

    all_score_rows: list[dict[str, Any]] = []
    all_recovery_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    max_available = min(arr.shape[0] for arr in random_raw.values())
    n_selected = next(iter(load_payload(run_dir, methods[0])["selected_features_raw"].values())).shape[0]
    random_subset_indices = [
        rng.choice(max_available, size=n_selected, replace=False)
        for _ in range(args.n_random_subsets)
    ]

    metadata = {
        "run_dir": str(run_dir),
        "methods": methods,
        "target_space": args.target_space,
        "target_tracks": target_tracks,
        "model_names": model_names,
        "random_feature_dir": str(random_feature_dir),
        "encoding_root": str(encoding_root),
        "n_random_images": args.n_random_images,
        "n_random_subsets": args.n_random_subsets,
        "n_splits": args.n_splits,
        "n_noise_samples": args.n_noise_samples,
        "train_n": args.train_n,
        "val_n": args.val_n,
        "test_n": args.test_n,
        "alphas": alphas,
        "base_noise_ceiling": noise_ceiling,
        "noise_mults": noise_mults,
        "seed": args.seed,
        "random_subset_indices_reused_across_methods": True,
        "random_split_and_noise_seeds_reused_across_methods": True,
        "note": (
            "Continuous-response teacher/student pilot: teacher targets are "
            "encoded features or raw features plus calibrated Gaussian noise "
            "over one or more noise multipliers; candidates are refit from raw "
            "features with ridge regression."
        ),
    }
    with (out_dir / "teacher_student_recovery_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    for method_idx, method_id in enumerate(methods):
        print(f"[{method_idx + 1}/{len(methods)}] {method_id}")
        payload = load_payload(run_dir, method_id)
        if list(payload["model_names"]) != model_names:
            raise ValueError(f"Model order mismatch in {method_id}")
        for target_idx, target_track in enumerate(target_tracks):
            print(f"  target {target_idx + 1}/{len(target_tracks)}: {target_track}")
            if args.target_space == "encoded":
                selected_raw, selected_target = selected_arrays_from_payload(
                    payload,
                    target_track,
                    model_names,
                    encoding_params_by_track[target_track],
                )
            else:
                selected_raw = selected_raw_from_payload(payload, model_names)
                selected_target = selected_raw

            score_rows, recovery_rows = run_subset_recovery(
                method_id=method_id,
                subset_type="selected",
                subset_idx=0,
                target_track=target_track,
                raw_by_model=selected_raw,
                target_by_model=selected_target,
                model_names=model_names,
                n_splits=args.n_splits,
                n_noise_samples=args.n_noise_samples,
                train_n=args.train_n,
                val_n=args.val_n,
                test_n=args.test_n,
                alphas=alphas,
                base_noise_ceiling=noise_ceiling,
                noise_mults=noise_mults,
                seed=args.seed + target_idx * 100000 + method_idx * 10000,
            )
            all_score_rows.extend(score_rows)
            all_recovery_rows.extend(recovery_rows)

            for subset_idx, sample_idx in enumerate(random_subset_indices):
                random_raw_subset = {model: random_raw[model][sample_idx] for model in model_names}
                random_target_subset = {
                    model: random_target_by_track[target_track][model][sample_idx]
                    for model in model_names
                }
                score_rows, recovery_rows = run_subset_recovery(
                    method_id=method_id,
                    subset_type="random",
                    subset_idx=subset_idx,
                    target_track=target_track,
                    raw_by_model=random_raw_subset,
                    target_by_model=random_target_subset,
                    model_names=model_names,
                    n_splits=args.n_splits,
                    n_noise_samples=args.n_noise_samples,
                    train_n=args.train_n,
                    val_n=args.val_n,
                    test_n=args.test_n,
                    alphas=alphas,
                    base_noise_ceiling=noise_ceiling,
                    noise_mults=noise_mults,
                    seed=args.seed + target_idx * 100000 + 50000 + subset_idx,
                )
                all_score_rows.extend(score_rows)
                all_recovery_rows.extend(recovery_rows)

        pd.DataFrame(all_score_rows).to_csv(out_dir / "teacher_student_scores.csv", index=False)
        pd.DataFrame(all_recovery_rows).to_csv(out_dir / "teacher_student_recoveries.csv", index=False)

    scores = pd.DataFrame(all_score_rows)
    recovery = pd.DataFrame(all_recovery_rows)
    summary = summarize_recovery(recovery)
    auc_summary = summarize_auc(summary)
    confusion = confusion_table(recovery, model_names)
    scores.to_csv(out_dir / "teacher_student_scores.csv", index=False)
    recovery.to_csv(out_dir / "teacher_student_recoveries.csv", index=False)
    summary.to_csv(out_dir / "teacher_student_recovery_summary.csv", index=False)
    auc_summary.to_csv(out_dir / "teacher_student_recovery_auc_summary.csv", index=False)
    confusion.to_csv(out_dir / "teacher_student_confusion_matrix.csv", index=False)

    paths = []
    paths.extend(plot_summary(summary, fig_dir, methods, target_space=args.target_space))
    paths.extend(plot_recovery_curves(summary, fig_dir, methods, target_space=args.target_space))
    for path in [
        out_dir / "teacher_student_scores.csv",
        out_dir / "teacher_student_recoveries.csv",
        out_dir / "teacher_student_recovery_summary.csv",
        out_dir / "teacher_student_recovery_auc_summary.csv",
        out_dir / "teacher_student_confusion_matrix.csv",
        *paths,
    ]:
        print(path)


if __name__ == "__main__":
    main()
