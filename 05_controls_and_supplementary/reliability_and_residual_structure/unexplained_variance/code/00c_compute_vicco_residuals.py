#!/usr/bin/env python3
"""
Compute residual brain RDM for vicco baseline stimuli.

Same pipeline as 01_subject_consistency.py but for the 292-image vicco set.
Uses all 292 images (no bootstrap) for maximum stability.

Outputs:
    data/vicco_residuals.npz      - per-subject residual + brain_z vectors (42486,)
    data/vicco_consistency.csv    - pairwise inter-subject residual correlations

Usage:
    python 00c_compute_vicco_residuals.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import zscore, spearmanr
from sklearn.linear_model import RidgeCV
from tqdm import tqdm
from PIL import Image

_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))
import config
from utils import compute_rdm_correlation, load_encoding_model, predict_voxel_responses

from cstims.feature_extraction.universal_extractor import UniversalFeatureExtractor

SUBJECTS = config.SUBJECTS  # all 5
DATA_DIR = Path(__file__).resolve().parent / "results"
VICCO_IMG_DIR = config.CSTIM_HDF5_ROOT / "shared_vicco"
N_VICCO = 292


def upper_tri(rdm):
    idx = np.triu_indices(rdm.shape[0], k=1)
    return rdm[idx]


def load_model_config(model_name):
    df = pd.read_csv(config.MODEL_LIST_CSV)
    row = df[df["model"] == model_name].iloc[0]
    return {"layer": row["layer"], "aggregation": row["aggregation"], "source": row["source"]}


def extract_features(model_name, images):
    import torch
    cfg = load_model_config(model_name)
    extractor = UniversalFeatureExtractor(
        model_name=model_name, layer=cfg["layer"],
        aggregation=cfg["aggregation"], source=cfg["source"],
    )
    feats = []
    for start in range(0, len(images), 32):
        batch = images[start:start + 32]
        t = torch.stack([extractor.preprocess(img) for img in batch]).to(extractor.device)
        with torch.no_grad():
            f = extractor.extract(t)
        if hasattr(f, "detach"):
            f = f.detach().cpu().numpy()
        feats.append(np.asarray(f).reshape(len(batch), -1).astype(np.float32))
    return np.concatenate(feats, axis=0)


def load_subject_vicco_betas(subject):
    """Return (n_voxels_hlvis, 292) betas for vicco stimuli."""
    data_dir = config.get_brain_input_dir(subject)
    betas_data = np.load(data_dir / "cstim_betas_averaged.npz", allow_pickle=True)
    voxel_data = np.load(data_dir / "voxel_metadata.npz", allow_pickle=True)
    stim_info = pd.read_csv(data_dir / "cstim_stimulus_info.csv")

    hlvis_mask = voxel_data["hlvis_mask"]
    betas_hlvis = betas_data["betas"][hlvis_mask, :]
    stim_keys = betas_data["stim_keys"]
    stim_key_to_idx = {k: i for i, k in enumerate(stim_keys)}

    vicco_rows = stim_info[stim_info["group"] == "vicco"].sort_values("stim_idx")
    brain_idx = np.array([stim_key_to_idx[k] for k in vicco_rows["stim_key"].values])
    file_idx = vicco_rows["stim_idx"].values - 1  # 1-based → 0-based

    return betas_hlvis[:, brain_idx], file_idx


def main():
    all_models = []
    for model_set, models in config.MODEL_SETS.items():
        all_models.extend(models)
    # Deduplicate preserving order, exclude VICReg (excluded in archive too)
    seen = set()
    model_list = []
    for m in all_models:
        if m not in seen and m != "vicreg_resnet50":
            seen.add(m)
            model_list.append(m)
    print(f"Models: {len(model_list)}")

    # Load vicco images once
    img_files = sorted(list(VICCO_IMG_DIR.glob("*.jpg")) + list(VICCO_IMG_DIR.glob("*.png")))
    assert len(img_files) == N_VICCO, f"Expected {N_VICCO} images, got {len(img_files)}"
    vicco_images = [Image.open(f).convert("RGB") for f in img_files]
    print(f"Loaded {len(vicco_images)} vicco images")

    # Extract features + compute predicted RDMs per model
    # (features are subject-independent; encoding models are per-subject)
    print("\nExtracting features and computing predicted RDMs per subject...")

    # Load brain betas per subject
    subject_betas = {}
    subject_file_idx = {}
    for subj in SUBJECTS:
        try:
            betas, file_idx = load_subject_vicco_betas(subj)
            subject_betas[subj] = betas
            subject_file_idx[subj] = file_idx
            print(f"  {subj}: {betas.shape[1]} vicco stimuli, {betas.shape[0]} hlvis voxels")
        except Exception as e:
            print(f"  {subj}: FAILED loading betas — {e}")

    # Per-subject: brain RDM + predicted RDMs
    subject_brain_vecs = {}
    subject_model_vecs = {subj: {} for subj in subject_betas}

    for subj, betas in subject_betas.items():
        brain_rdm = compute_rdm_correlation(betas.T)  # (292, 292)
        subject_brain_vecs[subj] = upper_tri(brain_rdm)

    for model in tqdm(model_list, desc="Models"):
        try:
            features_all = extract_features(model, vicco_images)  # (292, d)
        except Exception as e:
            print(f"  FAILED extracting {model}: {e}")
            continue

        for subj in subject_betas:
            try:
                file_idx = subject_file_idx[subj]
                subj_features = features_all[file_idx]
                encoding = load_encoding_model(model, subj)
                predicted = predict_voxel_responses(subj_features, encoding)
                pred_hlvis = predicted[:, encoding["roi_hlvis"]]
                pred_rdm = compute_rdm_correlation(pred_hlvis)
                subject_model_vecs[subj][model] = upper_tri(pred_rdm)
            except Exception as e:
                print(f"  FAILED {model}/{subj}: {e}")

    # Ridge regression per subject
    print("\nFitting ridge regression per subject...")
    residuals = {}
    brain_zs = {}
    r2s = {}

    for subj in subject_betas:
        brain_z = zscore(subject_brain_vecs[subj])
        model_names = sorted(subject_model_vecs[subj].keys())
        if not model_names:
            print(f"  {subj}: no model predictions, skipping")
            continue
        X = np.column_stack([zscore(subject_model_vecs[subj][m]) for m in model_names])
        reg = RidgeCV(alphas=np.logspace(-2, 4, 20), fit_intercept=True)
        reg.fit(X, brain_z)
        residual = brain_z - reg.predict(X)
        r2 = reg.score(X, brain_z)
        residuals[subj] = residual
        brain_zs[subj] = brain_z
        r2s[subj] = r2
        print(f"  {subj}: R²={r2:.3f}, residual std={residual.std():.3f}")

    # Pairwise inter-subject correlations
    subj_list = list(residuals.keys())
    rows = []
    print("\nPairwise residual correlations (Spearman):")
    for i in range(len(subj_list)):
        for j in range(i + 1, len(subj_list)):
            s1, s2 = subj_list[i], subj_list[j]
            rho_res, p_res = spearmanr(residuals[s1], residuals[s2])
            rho_brain, p_brain = spearmanr(brain_zs[s1], brain_zs[s2])
            print(f"  {s1} vs {s2}: residual rho={rho_res:.3f} (p={p_res:.2e}), "
                  f"brain rho={rho_brain:.3f} (p={p_brain:.2e})")
            rows.append({
                "subject_1": s1, "subject_2": s2,
                "residual_rho": rho_res, "residual_p": p_res,
                "brain_rho": rho_brain, "brain_p": p_brain,
                "r2_s1": r2s[s1], "r2_s2": r2s[s2],
            })

    pd.DataFrame(rows).to_csv(DATA_DIR / "vicco_consistency.csv", index=False)
    print("Saved: data/vicco_consistency.csv")

    np.savez(DATA_DIR / "vicco_residuals.npz",
             **{f"residual_{s}": residuals[s] for s in residuals},
             **{f"brain_z_{s}": brain_zs[s] for s in brain_zs},
             r2_scores=np.array([r2s[s] for s in subj_list]),
             subjects=np.array(subj_list))
    print("Saved: data/vicco_residuals.npz")
    print("Done.")


if __name__ == "__main__":
    main()
