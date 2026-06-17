#!/usr/bin/env python3
"""
Compute brain + encoding-predicted RDMs for subjects missing from the archive.

The archive at experiments/archive/simulationdiffs_to_braindiffs/data/ has
rdms_sub-01/03/05/06.npz, but sub-06 is missing all_models and sub-07 is absent
entirely. This script computes the missing data and saves it in the same format
so all 5 subjects can be used in downstream analyses.

Outputs:
    /home/jroth/rsa_based_selection/experiments/archive/simulationdiffs_to_braindiffs/data/rdms_sub-06.npz  (updated, all_models added)
    /home/jroth/rsa_based_selection/experiments/archive/simulationdiffs_to_braindiffs/data/rdms_sub-07.npz  (new)

Usage:
    python 00b_compute_missing_rdms.py [--subject sub-06|sub-07|all]
"""

import argparse
import sys
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))
import numpy as np
from tqdm import tqdm
from PIL import Image

_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))
from cstims.paper import config
from cstims.paper.utils import (
    compute_rdm_correlation,
    load_encoding_model,
    predict_voxel_responses,
    parse_subject_arg,
    load_model_layer_mapping,
)
from cstims.feature_extraction.universal_extractor import UniversalFeatureExtractor

ARCHIVE_DIR = Path(__file__).resolve().parents[3] / "archive/simulationdiffs_to_braindiffs/data"
CSTIM_HDF5_ROOT = config.CSTIM_HDF5_ROOT

# Only the all_models group is missing (sub-06 has the rest; sub-07 has nothing)
MISSING = {
    "sub-06": ["all_models"],
    "sub-07": list(config.MODEL_SETS.keys()),
}


def load_model_config(model_name):
    import pandas as pd
    df = pd.read_csv(config.MODEL_LIST_CSV)
    row = df[df["model"] == model_name].iloc[0]
    return {"layer": row["layer"], "aggregation": row["aggregation"], "source": row["source"]}


def load_images(group):
    """Load stimulus images for a group from the HDF5 image root."""
    folder_group = group
    if group == "architecture":
        folder_group = "dataset"
    elif group == "dataset":
        folder_group = "architecture"
    img_dir = CSTIM_HDF5_ROOT / folder_group
    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    return [Image.open(f).convert("RGB") for f in img_files]


def extract_features(model_name, images):
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
        batch = images[start:start + 32]
        tensors = [extractor.preprocess(img) for img in batch]
        t = __import__("torch").stack(tensors).to(extractor.device)
        with __import__("torch").no_grad():
            feats = extractor.extract(t)
        if hasattr(feats, "detach"):
            feats = feats.detach().cpu().numpy()
        feats_list.append(np.asarray(feats).reshape(len(batch), -1).astype(np.float32))
    return np.concatenate(feats_list, axis=0)


def load_subject_brain_data(subject):
    import pandas as pd
    from utils import bootstrap_sample_indices
    data_dir = config.get_brain_input_dir(subject)
    betas_data = np.load(data_dir / "cstim_betas_averaged.npz", allow_pickle=True)
    voxel_data = np.load(data_dir / "voxel_metadata.npz", allow_pickle=True)
    stim_info = pd.read_csv(data_dir / "cstim_stimulus_info.csv")

    hlvis_mask = voxel_data["hlvis_mask"]
    betas_hlvis = betas_data["betas"][hlvis_mask, :]
    stim_keys = betas_data["stim_keys"]
    stim_key_to_idx = {k: i for i, k in enumerate(stim_keys)}

    group_indices = {}
    group_stim_idx = {}
    for group in stim_info["group"].unique():
        mask = stim_info["group"] == group
        keys = stim_info.loc[mask, "stim_key"].values
        group_indices[group] = np.array([stim_key_to_idx[k] for k in keys])
        idx = stim_info.loc[mask, "stim_idx"].values
        group_stim_idx[group] = idx - 1 if group == "vicco" else idx

    return {
        "betas_hlvis": betas_hlvis,
        "group_indices": group_indices,
        "group_stim_idx": group_stim_idx,
        "n_hlvis": int(hlvis_mask.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="all")
    args = parser.parse_args()

    if args.subject == "all":
        subjects = ["sub-06", "sub-07"]
    else:
        subjects = [args.subject]

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    for subject in subjects:
        missing_groups = MISSING.get(subject, [])
        if not missing_groups:
            print(f"{subject}: nothing missing, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {subject}: missing groups = {missing_groups}")
        print(f"{'='*60}")

        # Load existing RDMs if file exists (to preserve other groups)
        out_path = ARCHIVE_DIR / f"rdms_{subject}.npz"
        rdm_dict = {}
        if out_path.exists():
            existing = np.load(out_path, allow_pickle=True)
            rdm_dict = dict(existing)
            print(f"  Loaded {len(rdm_dict)} existing RDMs from {out_path.name}")

        sdata = load_subject_brain_data(subject)
        print(f"  {subject}: {sdata['n_hlvis']} hlvis voxels")

        for model_set in missing_groups:
            models = config.MODEL_SETS[model_set]
            EXCLUDED = {"vicreg_resnet50"}

            print(f"\n  --- {model_set} ({len(models)} models) ---")

            if model_set not in sdata["group_indices"]:
                print(f"  {model_set} not in stimulus info for {subject}, skipping")
                continue

            # Brain RDM
            brain_idx = sdata["group_indices"][model_set]
            brain_rdm = compute_rdm_correlation(sdata["betas_hlvis"][:, brain_idx].T)
            rdm_dict[f"brain__{model_set}"] = brain_rdm
            print(f"  Computed brain RDM: shape={brain_rdm.shape}")

            cstim_images = load_images(model_set)

            for model in tqdm(models, desc=f"  {model_set} models"):
                if model in EXCLUDED:
                    continue
                key = f"pred__{model_set}__{model}"
                if key in rdm_dict:
                    print(f"    {model}: already exists, skipping")
                    continue
                try:
                    features = extract_features(model, cstim_images)
                    file_idx = sdata["group_stim_idx"][model_set]
                    subj_features = features[file_idx]

                    encoding = load_encoding_model(model, subject)
                    predicted = predict_voxel_responses(subj_features, encoding)
                    pred_hlvis = predicted[:, encoding["roi_hlvis"]]

                    rdm_dict[key] = compute_rdm_correlation(pred_hlvis)
                except Exception as e:
                    print(f"    FAILED {model}: {e}")

        np.savez_compressed(out_path, **rdm_dict)
        print(f"\nSaved {len(rdm_dict)} RDMs to {out_path}")


if __name__ == "__main__":
    main()
