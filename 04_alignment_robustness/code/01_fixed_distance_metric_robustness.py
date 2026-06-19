#!/usr/bin/env python3
"""
Distance metric robustness check: recompute fRSA with cosine distance
and compare to the standard correlation-distance fRSA scores.

Cosine distance = 1 - cosine_similarity(x_i, x_j)
Correlation distance = 1 - pearson_r(x_i, x_j)  (= cosine on mean-centered vectors)

The two differ only by whether features are mean-centered per stimulus before
computing the dot product. For high-dimensional neural network features the
difference is typically small.

Usage:
    python 04_alignment_robustness/code/01_fixed_distance_metric_robustness.py

Outputs:
    04_alignment_robustness/results/distance_metric_robustness.csv
"""

import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1]
SHARE_ROOT = STAGE.parent
PAPER_HELPERS = SHARE_ROOT / "src"
sys.path.insert(0, str(PAPER_HELPERS))
sys.path.insert(0, str(SHARE_ROOT / "src"))

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from tqdm import tqdm

from cstims import constants, paths
from cstims.cache import load_cstim_brain_cache
from cstims.rdm import compute_rdm_correlation, compute_rsa_score
from cstims.sampling import bootstrap_sample_indices
from cstims.feature_extraction.universal_extractor import UniversalFeatureExtractor


# -- Helper functions (duplicated from 02_compute_crsa.py to avoid import issues) --

def load_model_config(model_name: str) -> dict:
    """Load layer, aggregation, and source config for a model."""
    df = pd.read_csv(paths.model_list_csv())
    row = df[df["model"] == model_name].iloc[0]
    return {
        "layer": row["layer"],
        "aggregation": row["aggregation"],
        "source": row["source"],
    }


def load_images(group: str) -> list:
    """Load stimulus images for a group (with architecture/dataset folder swap)."""
    folder_group = group
    if group == "architecture":
        folder_group = "dataset"
    elif group == "dataset":
        folder_group = "architecture"

    if folder_group == "vicco":
        img_dir = paths.cstim_hdf5_root() / "shared_vicco"
    else:
        img_dir = paths.cstim_hdf5_root() / folder_group

    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    return [Image.open(f).convert("RGB") for f in img_files]


def extract_features(model_name: str, images: list, target_size: int = 224) -> np.ndarray:
    """Extract features from images using a model."""
    config = load_model_config(model_name)
    extractor = UniversalFeatureExtractor(
        model_name=model_name,
        layer=config["layer"],
        aggregation=config["aggregation"],
        source=config["source"],
    )
    import torch
    batch_size = 32
    feats_list = []
    for start in range(0, len(images), batch_size):
        batch_images = images[start:start + batch_size]
        tensors = [extractor.preprocess(img) for img in batch_images]
        batch = torch.stack(tensors).to(extractor.device)
        with torch.no_grad():
            feats = extractor.extract(batch)
        if isinstance(feats, torch.Tensor):
            feats = feats.detach().cpu().numpy()
        feats = np.asarray(feats).reshape(len(batch_images), -1).astype(np.float32)
        feats_list.append(feats)
    return np.concatenate(feats_list, axis=0)


def load_subject_brain_data(subject: str) -> dict:
    """Load brain data for a subject. Returns None if not available."""
    cache = load_cstim_brain_cache(subject, missing_ok=True)
    if cache is None:
        return None
    data = cache.as_legacy_group_dict()
    group_indices = data["group_indices"]
    n_vicco = len(group_indices.get("vicco", []))
    n_vicco_sample = min(100, n_vicco) if n_vicco > 0 else 0
    vicco_bootstrap = bootstrap_sample_indices(n_vicco, n_vicco_sample, n_bootstrap=10, seed=0) if n_vicco > 0 else []
    data["vicco_bootstrap"] = vicco_bootstrap
    data["n_vicco_sample"] = n_vicco_sample
    return data


# -- Cosine distance RDM --

def compute_rdm_cosine(features: np.ndarray) -> np.ndarray:
    """Compute RDM using cosine distance (1 - cosine_similarity)."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    features_normed = features / norms
    cos_sim = features_normed @ features_normed.T
    rdm = 1 - cos_sim
    np.fill_diagonal(rdm, 0)
    return rdm


# -- Main --

def main():
    print("=== Distance Metric Robustness Check ===")
    print("Comparing correlation-distance vs cosine-distance fRSA\n")

    # Load brain data for all subjects
    subject_data = {}
    for subject in constants.SUBJECTS:
        data = load_subject_brain_data(subject)
        if data is None:
            print(f"  {subject}: no brain data, skipping")
            continue
        subject_data[subject] = data
        print(f"  {subject}: {data['n_hlvis']} hlvis voxels")

    if not subject_data:
        print("No subjects with data found.")
        return

    # Pre-load vicco images
    print("\nLoading vicco images...")
    all_vicco_images = load_images("vicco")
    print(f"  {len(all_vicco_images)} vicco images")

    results = []

    for model_set, models in constants.MODEL_SETS.items():
        print(f"\n{'='*60}")
        print(f"Model set: {model_set} ({len(models)} models)")
        print(f"{'='*60}")

        cstim_images = load_images(model_set)

        for model in tqdm(models, desc="Models"):
            display_name = constants.MODEL_DISPLAY_NAMES.get(model, model)

            # Extract features once
            all_images = cstim_images + all_vicco_images
            all_features = extract_features(model, all_images)

            n_cstim = len(cstim_images)
            features_cstim = all_features[:n_cstim]
            features_all_vicco = all_features[n_cstim:]

            for subject, sdata in subject_data.items():
                if model_set not in sdata["group_indices"]:
                    continue

                betas = sdata["betas_hlvis"]
                cstim_brain_idx = sdata["group_indices"][model_set]
                cstim_file_idx = sdata["group_stim_idx"][model_set]

                subj_features_cstim = features_cstim[cstim_file_idx]

                # Brain RDM (same for both metrics)
                brain_rdm_cstim = compute_rdm_correlation(betas[:, cstim_brain_idx].T)

                # Correlation-distance fRSA
                model_rdm_corr = compute_rdm_correlation(subj_features_cstim)
                crsa_corr = compute_rsa_score(model_rdm_corr, brain_rdm_cstim, method="spearman")

                # Cosine-distance fRSA
                model_rdm_cos = compute_rdm_cosine(subj_features_cstim)
                crsa_cos = compute_rsa_score(model_rdm_cos, brain_rdm_cstim, method="spearman")

                results.append({
                    "subject": subject,
                    "model_set": model_set,
                    "model": model,
                    "display_name": display_name,
                    "stimulus_type": "controversial",
                    "bootstrap_idx": 0,
                    "crsa_correlation": crsa_corr,
                    "crsa_cosine": crsa_cos,
                })

                # Vicco bootstrap
                vicco_file_idx = sdata["group_stim_idx"]["vicco"]
                subj_features_vicco = features_all_vicco[vicco_file_idx]

                for boot_idx, vicco_subset_idx in enumerate(sdata["vicco_bootstrap"]):
                    vicco_brain_idx = sdata["group_indices"]["vicco"][vicco_subset_idx]
                    brain_rdm_vicco = compute_rdm_correlation(betas[:, vicco_brain_idx].T)

                    feat_vicco = subj_features_vicco[vicco_subset_idx]
                    model_rdm_corr_v = compute_rdm_correlation(feat_vicco)
                    crsa_corr_v = compute_rsa_score(model_rdm_corr_v, brain_rdm_vicco, method="spearman")

                    model_rdm_cos_v = compute_rdm_cosine(feat_vicco)
                    crsa_cos_v = compute_rsa_score(model_rdm_cos_v, brain_rdm_vicco, method="spearman")

                    results.append({
                        "subject": subject,
                        "model_set": model_set,
                        "model": model,
                        "display_name": display_name,
                        "stimulus_type": "vicco",
                        "bootstrap_idx": boot_idx,
                        "crsa_correlation": crsa_corr_v,
                        "crsa_cosine": crsa_cos_v,
                    })

    df = pd.DataFrame(results)
    out_path = paths.robustness_data_dir() / "distance_metric_robustness.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(df)} rows)")

    # Quick summary
    rho, p = spearmanr(df["crsa_correlation"], df["crsa_cosine"])
    print(f"\nOverall Spearman rho: {rho:.4f} (p = {p:.2e})")
    print(f"Mean absolute difference: {(df['crsa_correlation'] - df['crsa_cosine']).abs().mean():.4f}")

    # Per model set
    print("\nPer model set:")
    for ms in df["model_set"].unique():
        sub = df[df["model_set"] == ms]
        r, p = spearmanr(sub["crsa_correlation"], sub["crsa_cosine"])
        print(f"  {ms}: rho = {r:.4f}, p = {p:.2e}, mean |diff| = {(sub['crsa_correlation'] - sub['crsa_cosine']).abs().mean():.4f}")


if __name__ == "__main__":
    main()
