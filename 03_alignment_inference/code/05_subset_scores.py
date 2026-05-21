#!/usr/bin/env python3
"""
Compute RSA scores (fRSA + mRSA) using only the first K controversial stimuli.

Tests whether smaller stimulus subsets produce similar brain alignment patterns
to the full 100-stimulus set. Uses the first K stimuli from greedy selection
(stim_idx 0..K-1) and K-sized Vicco bootstrap samples for baseline.

Usage:
    python 11_compute_subset_scores.py                    # Default K=20
    python 11_compute_subset_scores.py --max-stim 15      # K=15
    python 11_compute_subset_scores.py --subject sub-01   # Single subject

Outputs:
    data/subset_scores_K{max_stim}.csv   - All scores in one file
"""

import argparse
import sys
from pathlib import Path

# Setup imports from cstim_paper root
_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))  # project root for cstims

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import config
from utils import (
    compute_rdm_correlation,
    compute_rsa_score,
    bootstrap_sample_indices,
    get_encoding_folder,
    parse_subject_arg,
    load_encoding_model,
    predict_voxel_responses,
)

from cstims.feature_extraction.universal_extractor import UniversalFeatureExtractor


def load_model_config(model_name: str) -> dict:
    df = pd.read_csv(config.MODEL_LIST_CSV)
    row = df[df["model"] == model_name].iloc[0]
    return {
        "layer": row["layer"],
        "aggregation": row["aggregation"],
        "source": row["source"],
    }


def load_images(group: str) -> list:
    folder_group = group
    if group == "architecture":
        folder_group = "dataset"
    elif group == "dataset":
        folder_group = "architecture"

    if folder_group == "vicco":
        img_dir = config.CSTIM_HDF5_ROOT / "shared_vicco"
    else:
        img_dir = config.CSTIM_HDF5_ROOT / folder_group

    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    return [Image.open(f).convert("RGB") for f in img_files]


def extract_features(model_name: str, images: list, target_size: int = 224) -> np.ndarray:
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


def load_subject_brain_data(subject: str, max_stim: int) -> dict:
    """Load brain data with subset indexing."""
    data_dir = config.get_subject_data_dir(subject)
    betas_path = data_dir / "cstim_betas_averaged.npz"
    if not betas_path.exists():
        return None

    betas_data = np.load(betas_path, allow_pickle=True)
    voxel_data = np.load(data_dir / "voxel_metadata.npz", allow_pickle=True)
    stim_info = pd.read_csv(data_dir / "cstim_stimulus_info.csv")

    hlvis_mask = voxel_data["hlvis_mask"]
    betas_hlvis = betas_data["betas"][hlvis_mask, :]
    stim_keys = betas_data["stim_keys"]
    stim_key_to_idx = {k: i for i, k in enumerate(stim_keys)}

    available_groups = sorted(stim_info["group"].unique().tolist())

    # Full indices (same as original scripts)
    group_indices = {}
    group_stim_idx = {}
    # Subset indices (first K stimuli by stim_idx)
    group_indices_subset = {}
    group_stim_idx_subset = {}

    for group in available_groups:
        mask = stim_info["group"] == group
        group_df = stim_info[mask].sort_values("stim_idx")
        keys = group_df["stim_key"].values
        idx = group_df["stim_idx"].values

        brain_idx = np.array([stim_key_to_idx[k] for k in keys])
        file_idx = idx - 1 if group == "vicco" else idx

        group_indices[group] = brain_idx
        group_stim_idx[group] = file_idx

        if group != "vicco":
            # Take first max_stim stimuli (by stim_idx, which = greedy order)
            k = min(max_stim, len(brain_idx))
            group_indices_subset[group] = brain_idx[:k]
            group_stim_idx_subset[group] = file_idx[:k]

    # Vicco bootstrap: use max_stim-sized samples
    n_vicco = len(group_indices.get("vicco", []))
    n_vicco_sample = min(max_stim, n_vicco) if n_vicco > 0 else 0
    vicco_bootstrap = bootstrap_sample_indices(
        n_vicco, n_vicco_sample, n_bootstrap=10, seed=0
    ) if n_vicco > 0 else []

    return {
        "betas_hlvis": betas_hlvis,
        "group_indices": group_indices,
        "group_stim_idx": group_stim_idx,
        "group_indices_subset": group_indices_subset,
        "group_stim_idx_subset": group_stim_idx_subset,
        "available_groups": available_groups,
        "vicco_bootstrap": vicco_bootstrap,
        "n_vicco_sample": n_vicco_sample,
        "n_hlvis": int(hlvis_mask.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-stim", type=int, default=20,
                        help="Number of stimuli to use from greedy selection (default: 20)")
    parser.add_argument("--subject", default="all")
    args = parser.parse_args()

    max_stim = args.max_stim
    subjects = parse_subject_arg(args.subject)
    print(f"Subset size: {max_stim} stimuli")
    print(f"Subjects: {subjects}")

    # Load brain data
    subject_data = {}
    for subject in subjects:
        data = load_subject_brain_data(subject, max_stim)
        if data is None:
            print(f"  {subject}: no brain data, skipping")
            continue
        subject_data[subject] = data
        print(f"  {subject}: {data['n_hlvis']} hlvis voxels")

    if not subject_data:
        print("No subjects found.")
        return

    results = []

    # Pre-load vicco images
    print("\nLoading vicco images...")
    all_vicco_images = load_images("vicco")
    print(f"  {len(all_vicco_images)} vicco images")

    for model_set, models in config.MODEL_SETS.items():
        print(f"\n{'='*60}")
        print(f"Model set: {model_set} ({len(models)} models)")
        print(f"{'='*60}")

        cstim_images = load_images(model_set)

        for model in tqdm(models, desc="Models"):
            display_name = config.MODEL_DISPLAY_NAMES.get(model, model)

            # Extract features for all images
            all_images = cstim_images + all_vicco_images
            all_features = extract_features(model, all_images)

            n_cstim = len(cstim_images)
            model_features_cstim = all_features[:n_cstim]
            model_features_all_vicco = all_features[n_cstim:]

            for subject, sdata in subject_data.items():
                if model_set not in sdata["group_indices_subset"]:
                    continue

                betas = sdata["betas_hlvis"]
                cstim_brain_idx = sdata["group_indices_subset"][model_set]
                cstim_file_idx = sdata["group_stim_idx_subset"][model_set]
                actual_k = len(cstim_brain_idx)

                # Subset model features
                subj_features_cstim = model_features_cstim[cstim_file_idx]

                # --- fRSA ---
                model_rdm = compute_rdm_correlation(subj_features_cstim)
                brain_rdm = compute_rdm_correlation(betas[:, cstim_brain_idx].T)
                crsa = compute_rsa_score(model_rdm, brain_rdm, method="spearman")

                # --- mRSA ---
                wrsa = np.nan
                try:
                    encoding = load_encoding_model(model, subject)
                    pred = predict_voxel_responses(subj_features_cstim, encoding)
                    pred_rdm = compute_rdm_correlation(pred[:, encoding["roi_hlvis"]])
                    wrsa = compute_rsa_score(pred_rdm, brain_rdm, method="spearman")
                except Exception as e:
                    pass  # Some models may not have encoding models

                results.append({
                    "subject": subject,
                    "model_set": model_set,
                    "model": model,
                    "display_name": display_name,
                    "stimulus_type": "controversial",
                    "bootstrap_idx": 0,
                    "n_stimuli": actual_k,
                    "crsa": crsa,
                    "wrsa_transfer": wrsa,
                })

                # --- Vicco bootstrap ---
                vicco_file_idx = sdata["group_stim_idx"]["vicco"]
                subj_features_vicco = model_features_all_vicco[vicco_file_idx]

                for boot_idx, vicco_subset in enumerate(sdata["vicco_bootstrap"]):
                    vicco_brain_idx = sdata["group_indices"]["vicco"][vicco_subset]
                    brain_rdm_v = compute_rdm_correlation(betas[:, vicco_brain_idx].T)

                    feat_v = subj_features_vicco[vicco_subset]
                    model_rdm_v = compute_rdm_correlation(feat_v)
                    crsa_v = compute_rsa_score(model_rdm_v, brain_rdm_v, method="spearman")

                    wrsa_v = np.nan
                    try:
                        pred_v = predict_voxel_responses(feat_v, encoding)
                        pred_rdm_v = compute_rdm_correlation(pred_v[:, encoding["roi_hlvis"]])
                        wrsa_v = compute_rsa_score(pred_rdm_v, brain_rdm_v, method="spearman")
                    except Exception:
                        pass

                    results.append({
                        "subject": subject,
                        "model_set": model_set,
                        "model": model,
                        "display_name": display_name,
                        "stimulus_type": "vicco",
                        "bootstrap_idx": boot_idx,
                        "n_stimuli": sdata["n_vicco_sample"],
                        "crsa": crsa_v,
                        "wrsa_transfer": wrsa_v,
                    })

    df = pd.DataFrame(results)
    out_path = config.STATS_DATA_DIR / f"subset_scores_K{max_stim}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY (controversial, cross-subject mean, K={max_stim})")
    print(f"{'='*60}")
    cstim = df[df["stimulus_type"] == "controversial"]
    for method, col in [("mRSA", "wrsa_transfer"), ("fRSA", "crsa")]:
        print(f"\n  {method}:")
        sub_mean = cstim.groupby(["model_set", "model"])[col].mean()
        for ms in ["sota", "training_objective", "architecture", "dataset", "all_models"]:
            ms_data = sub_mean.loc[ms].sort_values(ascending=False)
            top = ms_data.index[0]
            top_name = config.MODEL_DISPLAY_NAMES.get(top, top)
            print(f"    {ms:25s}: top={top_name:15s} r={ms_data.iloc[0]:.3f}")


if __name__ == "__main__":
    main()
