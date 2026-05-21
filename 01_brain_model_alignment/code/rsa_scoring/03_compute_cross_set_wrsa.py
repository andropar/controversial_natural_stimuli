#!/usr/bin/env python3
"""
Compute mRSA scores for ALL models on ALL stimulus sets (cross-set evaluation).

The standard wrsa_transfer_scores.csv only evaluates each model on its own
set's stimuli. This script evaluates every model in all_models on every
controversial stimulus group, enabling the question: do out-of-set models
also score higher on controlled stimuli, or only in-set models?

Usage:
    python 03_compute_cross_set_wrsa.py                   # All subjects
    python 03_compute_cross_set_wrsa.py --subject sub-01  # Single subject

Outputs per subject (in data/{subject}/):
    cross_set_wrsa_scores.csv
"""

import argparse
import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import (
    MODEL_SETS, MODEL_DISPLAY_NAMES, CSTIM_HDF5_ROOT, MODEL_LIST_CSV,
    get_brain_input_dir, RSA_DATA_DIR, PROJECT_ROOT,
)
from utils import (
    compute_rdm_correlation,
    compute_rsa_score,
    load_encoding_model,
    predict_voxel_responses,
    parse_subject_arg,
)

from cstims.feature_extraction.universal_extractor import UniversalFeatureExtractor


GROUPS = ["all_models", "architecture", "dataset", "sota", "training_objective"]
ALL_MODELS = MODEL_SETS["all_models"]


def load_model_config(model_name: str) -> dict:
    df = pd.read_csv(MODEL_LIST_CSV)
    row = df[df["model"] == model_name].iloc[0]
    return {"layer": row["layer"], "aggregation": row["aggregation"], "source": row["source"]}


def load_images(group: str) -> list:
    """Load stimulus images for a group (preserving architecture/dataset folder swap)."""
    folder_group = group
    if group == "architecture":
        folder_group = "dataset"
    elif group == "dataset":
        folder_group = "architecture"
    img_dir = CSTIM_HDF5_ROOT / folder_group
    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    return [Image.open(f).convert("RGB") for f in img_files]


def extract_features(model_name: str, images: list) -> np.ndarray:
    import torch
    cfg = load_model_config(model_name)
    extractor = UniversalFeatureExtractor(
        model_name=model_name,
        layer=cfg["layer"],
        aggregation=cfg["aggregation"],
        source=cfg["source"],
    )
    feats_list = []
    for start in range(0, len(images), 32):
        batch_images = images[start:start + 32]
        tensors = [extractor.preprocess(img) for img in batch_images]
        import torch
        batch = torch.stack(tensors).to(extractor.device)
        with torch.no_grad():
            feats = extractor.extract(batch)
        if isinstance(feats, torch.Tensor):
            feats = feats.detach().cpu().numpy()
        feats_list.append(np.asarray(feats).reshape(len(batch_images), -1).astype(np.float32))
    return np.concatenate(feats_list, axis=0)


def load_subject_brain_data(subject: str) -> dict:
    data_dir = get_brain_input_dir(subject)
    betas_path = data_dir / "cstim_betas_averaged.npz"
    if not betas_path.exists():
        return None

    betas_data = np.load(betas_path, allow_pickle=True)
    voxel_data = np.load(data_dir / "voxel_metadata.npz", allow_pickle=True)
    stim_info  = pd.read_csv(data_dir / "cstim_stimulus_info.csv")

    hlvis_mask   = voxel_data["hlvis_mask"]
    betas_hlvis  = betas_data["betas"][hlvis_mask, :]
    stim_keys    = betas_data["stim_keys"]
    stim_key_to_idx = {k: i for i, k in enumerate(stim_keys)}

    group_indices = {}
    group_stim_idx = {}
    for group in stim_info["group"].unique():
        mask = stim_info["group"] == group
        keys = stim_info.loc[mask, "stim_key"].values
        group_indices[group]  = np.array([stim_key_to_idx[k] for k in keys])
        idx = stim_info.loc[mask, "stim_idx"].values
        group_stim_idx[group] = idx - 1 if group == "vicco" else idx

    return {
        "betas_hlvis":   betas_hlvis,
        "group_indices": group_indices,
        "group_stim_idx": group_stim_idx,
        "n_hlvis":       int(hlvis_mask.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="all")
    args = parser.parse_args()

    subjects = parse_subject_arg(args.subject)
    print(f"Subjects: {subjects}")

    subject_data = {}
    for subject in subjects:
        data = load_subject_brain_data(subject)
        if data is None:
            print(f"  {subject}: no brain data, skipping")
            continue
        subject_data[subject] = data
        print(f"  {subject}: {data['n_hlvis']} hlvis voxels")

    if not subject_data:
        return

    # Output paths — fresh run
    out_paths = {}
    for subject in subject_data:
        p = RSA_DATA_DIR / subject / "cross_set_wrsa_scores.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            p.unlink()
        out_paths[subject] = p

    # Outer loop: stimulus groups (load images once per group)
    for group in GROUPS:
        print(f"\n{'='*60}\nGroup: {group}\n{'='*60}")
        images = load_images(group)
        print(f"  {len(images)} images")

        # Middle loop: models (extract features once per model per group)
        for model in tqdm(ALL_MODELS, desc="Models"):
            display_name = MODEL_DISPLAY_NAMES.get(model, model)
            features = extract_features(model, images)  # (n_images, n_feats)

            # Inner loop: subjects
            for subject, sdata in subject_data.items():
                if group not in sdata["group_indices"]:
                    continue
                try:
                    encoding    = load_encoding_model(model, subject)
                    encoding_hlvis = encoding["roi_hlvis"]
                    betas       = sdata["betas_hlvis"]
                    brain_idx   = sdata["group_indices"][group]
                    file_idx    = sdata["group_stim_idx"][group]

                    subj_feats  = features[file_idx]
                    pred        = predict_voxel_responses(subj_feats, encoding)
                    pred_rdm    = compute_rdm_correlation(pred[:, encoding_hlvis])
                    brain_rdm   = compute_rdm_correlation(betas[:, brain_idx].T)
                    score       = compute_rsa_score(pred_rdm, brain_rdm, method="spearman")

                    row = {
                        "subject":        subject,
                        "stimulus_group": group,
                        "model":          model,
                        "display_name":   display_name,
                        "in_set":         model in MODEL_SETS.get(group, []),
                        "n_stimuli":      len(brain_idx),
                        "wrsa_transfer":  score,
                    }
                    pd.DataFrame([row]).to_csv(
                        out_paths[subject], mode="a",
                        header=not out_paths[subject].exists() or out_paths[subject].stat().st_size == 0,
                        index=False,
                    )
                except Exception as e:
                    print(f"  Error {model} / {subject}: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
