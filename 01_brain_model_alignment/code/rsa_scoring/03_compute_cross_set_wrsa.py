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
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))

_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from cstims import paths
from cstims.cache import load_cstim_brain_cache
from cstims.constants import MODEL_SETS, MODEL_DISPLAY_NAMES
CSTIM_HDF5_ROOT = paths.cstim_hdf5_root()
MODEL_LIST_CSV = paths.model_list_csv()
get_brain_input_dir = paths.get_brain_input_dir
RSA_DATA_DIR = paths.rsa_data_dir()
PROJECT_ROOT = paths.project_root()
from cstims.rdm import compute_rdm_correlation, compute_rsa_score
from cstims.subjects import parse_subject_arg
from cstims.paper.utils import load_encoding_model, predict_voxel_responses

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
    cache = load_cstim_brain_cache(subject, missing_ok=True)
    if cache is None:
        return None
    return cache.as_legacy_group_dict()


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
