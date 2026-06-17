#!/usr/bin/env python3
"""Mixed-RSA correlation-vs-cosine distance robustness.

The earlier robustness script recomputed fixed RSA only. This companion
script recomputes mixed RSA from cached CSTIM features by projecting features
through the existing evaluation encodings and then scoring predicted-voxel
RDMs with either correlation or cosine distance.

Outputs:
  04_alignment_robustness/results/mixed_distance_metric_robustness.csv
  04_alignment_robustness/results/mixed_distance_metric_rank_summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

STAGE = Path(__file__).resolve().parents[1]
SHARE_ROOT = STAGE.parent
PAPER_HELPERS = SHARE_ROOT / "src"
sys.path.insert(0, str(PAPER_HELPERS))
sys.path.insert(0, str(SHARE_ROOT / "src"))

from cstims.paper import config  # noqa: E402
from cstims.paper.utils import bootstrap_sample_indices, compute_rdm_correlation, compute_rsa_score, load_encoding_model  # noqa: E402


MODEL_SETS = ["all_models", "sota", "training_objective", "architecture", "dataset"]


def compute_rdm_cosine(patterns: np.ndarray) -> np.ndarray:
    x = np.asarray(patterns, dtype=np.float64)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    x = x / denom
    sim = x @ x.T
    rdm = 1.0 - sim
    np.fill_diagonal(rdm, 0.0)
    return rdm


def predict_hlvis(features: np.ndarray, encoding: dict) -> np.ndarray:
    x = features.astype(np.float64, copy=True)
    mean = encoding["feature_mean"]
    scale = encoding["feature_scale"]
    if mean is not None and np.any(mean != 0):
        x -= mean
    if scale is not None and np.any(scale != 1):
        x /= scale + 1e-8
    roi = encoding["roi_hlvis"].astype(bool)
    return x @ encoding["weights"][:, roi] + encoding["intercept"][roi]


def load_subject_brain_data(subject: str, n_bootstrap: int) -> dict | None:
    data_dir = config.get_subject_data_dir(subject)
    betas_path = data_dir / "cstim_betas_averaged.npz"
    if not betas_path.exists():
        return None
    betas_data = np.load(betas_path, allow_pickle=True)
    voxel_data = np.load(data_dir / "voxel_metadata.npz", allow_pickle=True)
    stim_info = pd.read_csv(data_dir / "cstim_stimulus_info.csv")

    hlvis = voxel_data["hlvis_mask"]
    betas = betas_data["betas"][hlvis, :]
    stim_keys = betas_data["stim_keys"]
    stim_key_to_idx = {k: i for i, k in enumerate(stim_keys)}

    group_indices = {}
    group_file_idx = {}
    for group, gdf in stim_info.groupby("group"):
        keys = gdf["stim_key"].values
        group_indices[group] = np.array([stim_key_to_idx[k] for k in keys])
        idx = gdf["stim_idx"].values.astype(int)
        group_file_idx[group] = idx - 1 if group == "vicco" else idx

    n_vicco = len(group_indices.get("vicco", []))
    n_vicco_sample = min(100, n_vicco)
    vicco_bootstrap = bootstrap_sample_indices(
        n_vicco, n_vicco_sample, n_bootstrap=n_bootstrap, seed=0
    )
    return {
        "betas_hlvis": betas,
        "group_indices": group_indices,
        "group_file_idx": group_file_idx,
        "vicco_bootstrap": vicco_bootstrap,
        "n_vicco_sample": n_vicco_sample,
    }


def load_features(model: str) -> dict[str, np.ndarray]:
    path = config.CSTIM_FEATURE_CACHE / f"{model}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as z:
        return {k: z[k].astype(np.float32) for k in z.files}


def score_one(brain_patterns: np.ndarray, pred_patterns: np.ndarray) -> tuple[float, float]:
    brain_rdm = compute_rdm_correlation(brain_patterns)
    pred_corr = compute_rdm_correlation(pred_patterns)
    pred_cos = compute_rdm_cosine(pred_patterns)
    return (
        compute_rsa_score(pred_corr, brain_rdm, method="spearman"),
        compute_rsa_score(pred_cos, brain_rdm, method="spearman"),
    )


def build_rank_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subject, model_set, stimulus_type, bootstrap_idx), grp in df.groupby(
        ["subject", "model_set", "stimulus_type", "bootstrap_idx"]
    ):
        if grp["model"].nunique() < 3:
            continue
        rho, p = spearmanr(grp["mixed_rsa_correlation"], grp["mixed_rsa_cosine"])
        rows.append(
            {
                "subject": subject,
                "model_set": model_set,
                "stimulus_type": stimulus_type,
                "bootstrap_idx": bootstrap_idx,
                "n_models": int(grp["model"].nunique()),
                "rank_rho": float(rho),
                "p_spearman": float(p),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=config.SUBJECTS)
    parser.add_argument("--model-sets", nargs="+", default=MODEL_SETS)
    parser.add_argument("--n-bootstrap", type=int, default=10)
    parser.add_argument("--max-models", type=int, default=None)
    args = parser.parse_args()

    data_dir = config.ROBUSTNESS_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for subject in args.subjects:
        sdata = load_subject_brain_data(subject, n_bootstrap=args.n_bootstrap)
        if sdata is None:
            continue
        model_union = sorted(set(m for ms in args.model_sets for m in config.MODEL_SETS[ms]))
        if args.max_models is not None:
            model_union = model_union[: args.max_models]
        for model in tqdm(model_union, desc=subject):
            try:
                feats_by_group = load_features(model)
                encoding = load_encoding_model(model, subject)
            except Exception as exc:
                print(f"skip {subject} {model}: {exc}")
                continue

            for model_set in args.model_sets:
                if model not in config.MODEL_SETS[model_set] or model_set not in sdata["group_indices"]:
                    continue
                file_idx = sdata["group_file_idx"][model_set]
                brain_idx = sdata["group_indices"][model_set]
                pred = predict_hlvis(feats_by_group[model_set][file_idx], encoding)
                brain = sdata["betas_hlvis"][:, brain_idx].T
                corr_score, cos_score = score_one(brain, pred)
                rows.append(
                    {
                        "subject": subject,
                        "model_set": model_set,
                        "model": model,
                        "display_name": config.MODEL_DISPLAY_NAMES.get(model, model),
                        "stimulus_type": "controversial",
                        "bootstrap_idx": 0,
                        "n_stimuli": len(brain_idx),
                        "mixed_rsa_correlation": corr_score,
                        "mixed_rsa_cosine": cos_score,
                    }
                )

                if "vicco" not in feats_by_group or "vicco" not in sdata["group_indices"]:
                    continue
                vicco_file_idx = sdata["group_file_idx"]["vicco"]
                pred_vicco_all = predict_hlvis(feats_by_group["vicco"][vicco_file_idx], encoding)
                for boot_idx, subset in enumerate(sdata["vicco_bootstrap"]):
                    brain_idx_v = sdata["group_indices"]["vicco"][subset]
                    brain_v = sdata["betas_hlvis"][:, brain_idx_v].T
                    pred_v = pred_vicco_all[subset]
                    corr_v, cos_v = score_one(brain_v, pred_v)
                    rows.append(
                        {
                            "subject": subject,
                            "model_set": model_set,
                            "model": model,
                            "display_name": config.MODEL_DISPLAY_NAMES.get(model, model),
                            "stimulus_type": "vicco",
                            "bootstrap_idx": boot_idx,
                            "n_stimuli": len(subset),
                            "mixed_rsa_correlation": corr_v,
                            "mixed_rsa_cosine": cos_v,
                        }
                    )

    df = pd.DataFrame(rows)
    out = data_dir / "mixed_distance_metric_robustness.csv"
    df.to_csv(out, index=False)
    rank_df = build_rank_summary(df)
    rank_out = data_dir / "mixed_distance_metric_rank_summary.csv"
    rank_df.to_csv(rank_out, index=False)
    print(f"saved {out} ({len(df)} rows)")
    print(f"saved {rank_out} ({len(rank_df)} rows)")


if __name__ == "__main__":
    main()
