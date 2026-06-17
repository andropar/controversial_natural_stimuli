#!/usr/bin/env python3
"""
Compute PPCA log-likelihood for all stimuli relative to each subject's
encoding model training distribution — in both feature space and prediction space.

For each model × subject:
  1. Load the 4712 unique training image features from the encoding folder
  2. Z-score using encoding model normalization params (feature_mean / feature_scale)
  3. Fit PCA on training features (feature space)
  4. Project z-scored features through encoding weights → hlvis predictions
  5. Fit PCA on training predictions (prediction space)
  6. Compute PPCA log-likelihood for:
       - All 4712 training stimuli  (establishes per-model×subject reference distribution)
       - All cstim groups: architecture, training_objective, sota, dataset, all_models
       - Vicco baseline
  7. Z-score all logliks using training distribution mean/std (per model × subject)
     so that loglik_*_z = 0 means "as likely as the average training image"

The distinction between feature-space and prediction-space OOD matters:
a stimulus can be OOD in feature space but produce a normal-range prediction
if the OOD feature dimensions have near-zero encoding weights. Prediction-space
OOD is more directly relevant to brain alignment quality.

Output: results/pca_loglik.csv
  Columns: model, subject, stimulus_group, stimulus_idx,
           loglik_feature_raw, loglik_pred_raw,
           loglik_feature_z, loglik_pred_z,
           k_feature, k_pred            (number of PCs used, at VAR_THRESHOLD=0.95)

Usage:
    python 01_compute_pca_loglik.py                         # all models × subjects
    python 01_compute_pca_loglik.py --subjects sub-01       # single subject
    python 01_compute_pca_loglik.py --models cornet_s dinov2_vitl14
"""

import argparse
import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1]
SHARE_ROOT = STAGE.parents[2]
sys.path.insert(0, str(SHARE_ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm

from cstims.paper import config
from cstims.paper.utils import get_encoding_folder, load_encoding_model

# =============================================================================
# Constants
# =============================================================================

ALL_MODELS = config.MODEL_SETS["all_models"]
SUBJECTS = config.SUBJECTS
CSTIM_GROUPS = [
    "architecture", "training_objective", "sota", "dataset", "all_models", "vicco"
]
DATA_DIR = config.OOD_DATA_DIR
VAR_THRESHOLD = 0.95   # fraction of variance explained for choosing k
MIN_K_PRED   = 20     # minimum k for prediction space (prevents PPCA breakdown
                      # for near-rank-deficient models like VGG-16, ConvNeXt-B)


# =============================================================================
# Core math
# =============================================================================

def find_k(pca: PCA, threshold: float = VAR_THRESHOLD) -> int:
    """Smallest k such that the top-k PCs explain >= threshold of variance."""
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, threshold) + 1)
    return min(k, len(pca.components_))


def ppca_loglik(pca: PCA, X: np.ndarray, k: int) -> np.ndarray:
    """
    Per-sample log-likelihood under a PPCA model.

    Signal subspace: top-k PCs with eigenvalues λ_1..λ_k.
    Noise: isotropic σ² = mean of remaining eigenvalues.

    Args:
        pca: fitted PCA (pca.mean_ used for centering)
        X:   (n, d) features — NOT pre-centered
        k:   number of signal PCs

    Returns:
        (n,) log-likelihood per sample
    """
    d = X.shape[1]
    eigenvalues = pca.explained_variance_[:k]

    if k < len(pca.explained_variance_):
        sigma2 = float(np.mean(pca.explained_variance_[k:]))
    else:
        sigma2 = float(pca.noise_variance_)
    sigma2 = max(sigma2, 1e-10)

    centered = X - pca.mean_
    Z = centered @ pca.components_[:k].T       # (n, k) scores
    recon = Z @ pca.components_[:k]            # (n, d) reconstruction
    residual = centered - recon

    mahal_signal = np.sum(Z ** 2 / eigenvalues[None, :], axis=1)
    mahal_noise = np.sum(residual ** 2, axis=1) / sigma2

    log_norm = (
        (d / 2) * np.log(2 * np.pi)
        + 0.5 * np.sum(np.log(eigenvalues))
        + ((d - k) / 2) * np.log(sigma2)
    )

    return -log_norm - 0.5 * (mahal_signal + mahal_noise)


# =============================================================================
# Per model × subject computation
# =============================================================================

def process_model_subject(model: str, subject: str) -> list:
    """
    Compute PPCA log-likelihoods for one model × subject pair.

    Returns a list of row dicts (one per stimulus).
    Returns [] if training features or encoding model are missing.
    """
    enc_folder = get_encoding_folder(subject, model)
    features_path = enc_folder / "features.npz"
    if not features_path.exists():
        print(f"  SKIP {model}/{subject}: features.npz not found at {features_path}")
        return []

    # Load encoding model (weights, intercept, normalization, hlvis mask)
    try:
        enc = load_encoding_model(model, subject)
    except FileNotFoundError as e:
        print(f"  SKIP {model}/{subject}: {e}")
        return []

    feature_mean  = enc["feature_mean"]                        # (d,)
    feature_scale = enc["feature_scale"]                       # (d,)
    hlvis_mask    = enc["roi_hlvis"].astype(bool)              # (n_all_voxels,)
    W_hlvis       = enc["weights"][:, hlvis_mask]              # (d, n_hlvis)
    b_hlvis       = enc["intercept"][hlvis_mask]               # (n_hlvis,)

    # Load and z-score training features
    X_train   = np.load(features_path)["features"].astype(np.float32)  # (4712, d)
    n_train, d = X_train.shape
    if n_train < 100:
        print(f"  WARNING {model}/{subject}: only {n_train} training images — expected ~4712")

    X_train_z = (X_train - feature_mean) / (feature_scale + 1e-8)      # (4712, d)

    # Training predictions in hlvis voxel space
    P_train = X_train_z @ W_hlvis + b_hlvis                             # (4712, n_hlvis)

    # ---- Fit PCAs ----
    # Effective rank of X_train_z is min(n_train, d) = d (since d < n_train here).
    # Effective rank of P_train = X_train_z @ W_hlvis is also ≤ d.
    n_hlvis = int(hlvis_mask.sum())
    n_components = min(n_train - 1, d)
    n_components_pred = min(n_train - 1, d, n_hlvis)

    pca_feat = PCA(n_components=n_components)
    pca_feat.fit(X_train_z)
    k_feat = find_k(pca_feat)

    pca_pred = PCA(n_components=n_components_pred)
    pca_pred.fit(P_train)
    k_pred = max(find_k(pca_pred), MIN_K_PRED)

    # ---- Training log-likelihoods (reference distribution) ----
    ll_feat_train = ppca_loglik(pca_feat, X_train_z, k_feat)
    ll_pred_train = ppca_loglik(pca_pred, P_train,   k_pred)

    mu_feat  = float(np.mean(ll_feat_train))
    sig_feat = float(np.std(ll_feat_train))
    mu_pred  = float(np.mean(ll_pred_train))
    sig_pred = float(np.std(ll_pred_train))

    # Sorted training logliks for percentile computation
    # percentile = fraction of training stimuli with LOWER loglik than the query
    # (higher percentile = more in-distribution)
    train_sorted_feat = np.sort(ll_feat_train)
    train_sorted_pred = np.sort(ll_pred_train)

    def pct(train_sorted, values):
        """Percentile of each value within the training distribution (0–100)."""
        return np.searchsorted(train_sorted, values) / len(train_sorted) * 100

    def make_row(group, idx, ll_f, ll_p):
        return {
            "model":              model,
            "subject":            subject,
            "stimulus_group":     group,
            "stimulus_idx":       idx,
            "loglik_feature_raw": float(ll_f),
            "loglik_pred_raw":    float(ll_p),
            "loglik_feature_z":   (float(ll_f) - mu_feat) / (sig_feat + 1e-10),
            "loglik_pred_z":      (float(ll_p) - mu_pred) / (sig_pred + 1e-10),
            "k_feature":          k_feat,
            "k_pred":             k_pred,
        }

    rows = []

    pct_feat_train = pct(train_sorted_feat, ll_feat_train)
    pct_pred_train = pct(train_sorted_pred, ll_pred_train)
    for i in range(n_train):
        r = make_row("training", i, ll_feat_train[i], ll_pred_train[i])
        r["pct_feature"] = float(pct_feat_train[i])
        r["pct_pred"]    = float(pct_pred_train[i])
        rows.append(r)

    # ---- cstim + vicco groups ----
    cache_path = config.CSTIM_FEATURE_CACHE / f"{model}.npz"
    if not cache_path.exists():
        print(f"  WARNING: cstim feature cache missing for {model} — skipping test stimuli")
        return rows

    cache = np.load(cache_path)
    for group in CSTIM_GROUPS:
        if group not in cache:
            continue

        X_group   = cache[group].astype(np.float32)                   # (n_group, d)
        X_group_z = (X_group - feature_mean) / (feature_scale + 1e-8)
        P_group   = X_group_z @ W_hlvis + b_hlvis

        ll_feat = ppca_loglik(pca_feat, X_group_z, k_feat)
        ll_pred = ppca_loglik(pca_pred, P_group,   k_pred)
        pct_feat_group = pct(train_sorted_feat, ll_feat)
        pct_pred_group = pct(train_sorted_pred, ll_pred)

        for i in range(len(X_group)):
            r = make_row(group, i, ll_feat[i], ll_pred[i])
            r["pct_feature"] = float(pct_feat_group[i])
            r["pct_pred"]    = float(pct_pred_group[i])
            rows.append(r)

    return rows


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute PPCA log-likelihoods (feature + prediction space) for OOD analysis"
    )
    parser.add_argument("--models",   nargs="+", default=None,
                        help="Subset of model names to run (default: all)")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Subset of subjects to run (default: all)")
    args = parser.parse_args()

    models   = args.models   or ALL_MODELS
    subjects = args.subjects or SUBJECTS

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for model in tqdm(models, desc="Models"):
        for subject in tqdm(subjects, desc=f"  {model}", leave=False):
            rows = process_model_subject(model, subject)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_path = DATA_DIR / "pca_loglik.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df):,} rows → {out_path}")

    # Quick sanity summary
    train_df = df[df["stimulus_group"] == "training"]
    print(f"\nTraining loglik_feature_z (should be mean≈0, std≈1 per model×subject):")
    check = (
        train_df.groupby(["model", "subject"])["loglik_feature_z"]
        .agg(["mean", "std"])
        .round(3)
    )
    print(check.to_string())

    print(f"\nMean loglik_feature_z by group (across all models × subjects):")
    print(
        df.groupby("stimulus_group")["loglik_feature_z"]
        .mean()
        .sort_values()
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
