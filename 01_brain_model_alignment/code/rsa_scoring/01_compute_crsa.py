#!/usr/bin/env python3
"""
Compute classical RSA (fRSA) scores: direct model RDM vs brain RDM correlation.

Usage:
    python 02_compute_crsa.py                  # All subjects
    python 02_compute_crsa.py --subject sub-05  # Single subject

Model features are extracted ONCE (stimuli are identical across subjects).
Only brain RDMs change per subject.

Outputs per subject (in data/{subject}/):
    crsa_scores.csv - fRSA scores for all models, stimulus sets, and bootstrap samples
"""

import argparse
import sys
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))

# Setup imports from cstim_paper root
_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))  # project root for cstims

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
get_subject_data_dir = paths.get_subject_data_dir
RSA_DATA_DIR = paths.rsa_data_dir()
PROJECT_ROOT = paths.project_root()
from cstims.rdm import compute_rdm_correlation, compute_rsa_score
from cstims.sampling import bootstrap_sample_indices
from cstims.subjects import parse_subject_arg

from cstims.feature_extraction.universal_extractor import UniversalFeatureExtractor


ALL_GROUPS = ["vicco", "all_models", "architecture", "dataset", "sota", "training_objective"]

N_VICCO_BOOTSTRAPS = 1000


def load_model_config(model_name: str) -> dict:
    """Load layer, aggregation, and source config for a model."""
    df = pd.read_csv(MODEL_LIST_CSV)
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
        img_dir = CSTIM_HDF5_ROOT / "shared_vicco"
    else:
        img_dir = CSTIM_HDF5_ROOT / folder_group

    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    images = [Image.open(f).convert("RGB") for f in img_files]
    return images


def extract_features(model_name: str, images: list) -> np.ndarray:
    """Extract features from PIL images using the model's own preprocessing.

    Matches the pipeline in fit_encoding_hydra.py: preprocess each PIL image
    with the model-specific transform, stack tensors, then extract.
    """
    import torch
    config = load_model_config(model_name)
    extractor = UniversalFeatureExtractor(
        model_name=model_name,
        layer=config["layer"],
        aggregation=config["aggregation"],
        source=config["source"],
    )
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
    vicco_bootstrap = bootstrap_sample_indices(n_vicco, n_vicco_sample, n_bootstrap=N_VICCO_BOOTSTRAPS, seed=0) if n_vicco > 0 else []
    data["vicco_bootstrap"] = vicco_bootstrap
    data["n_vicco_sample"] = n_vicco_sample
    return data


def main():
    parser = argparse.ArgumentParser(description="Compute fRSA scores")
    parser.add_argument("--subject", default="all",
                        help="Subject ID (e.g. sub-05) or 'all' (default: all)")
    args = parser.parse_args()

    subjects = parse_subject_arg(args.subject)
    print(f"Processing subjects: {subjects}")

    # Load brain data for all subjects upfront
    subject_data = {}
    for subject in subjects:
        data = load_subject_brain_data(subject)
        if data is None:
            print(f"  {subject}: no brain data, skipping")
            continue
        subject_data[subject] = data
        print(f"  {subject}: {data['n_hlvis']} hlvis voxels, groups: {data['available_groups']}")

    if not subject_data:
        print("No subjects with data found.")
        return

    # Results per subject (accumulated, saved incrementally per model_set)
    results = {s: [] for s in subject_data}
    # Track what's been saved so we can append
    saved_counts = {s: 0 for s in subject_data}

    # Clear any existing output files (fresh run)
    for subject in subject_data:
        out_path = (RSA_DATA_DIR / subject / "crsa_scores.csv")
        if out_path.exists():
            out_path.unlink()

    # Pre-load all vicco images once
    print("\nPre-loading all vicco images...")
    all_vicco_images = load_images("vicco")
    print(f"Loaded {len(all_vicco_images)} vicco images")

    # Outer loop: model sets and models (extract features ONCE)
    for model_set, models in MODEL_SETS.items():
        print(f"\n{'='*60}")
        print(f"Model set: {model_set} ({len(models)} models)")
        print(f"{'='*60}")

        # Load controversial images for this set
        cstim_images = load_images(model_set)

        for model in tqdm(models, desc="Models"):
            display_name = MODEL_DISPLAY_NAMES.get(model, model)

            # Extract features ONCE for cstim + all vicco
            all_images = cstim_images + all_vicco_images
            all_features = extract_features(model, all_images)

            n_cstim = len(cstim_images)
            model_features_cstim = all_features[:n_cstim]
            model_features_all_vicco = all_features[n_cstim:]

            # Inner loop: subjects (only brain RDMs change, but may use stimulus subsets)
            for subject, sdata in subject_data.items():
                if model_set not in sdata["group_indices"]:
                    continue

                betas = sdata["betas_hlvis"]
                cstim_brain_idx = sdata["group_indices"][model_set]
                cstim_file_idx = sdata["group_stim_idx"][model_set]

                # Subset model features to this subject's available stimuli
                subj_model_features_cstim = model_features_cstim[cstim_file_idx]
                model_rdm_cstim = compute_rdm_correlation(subj_model_features_cstim)

                # Brain RDM for controversial stimuli
                brain_rdm_cstim = compute_rdm_correlation(betas[:, cstim_brain_idx].T)
                crsa_cstim = compute_rsa_score(model_rdm_cstim, brain_rdm_cstim, method="spearman")

                results[subject].append({
                    "subject": subject,
                    "model_set": model_set,
                    "model": model,
                    "display_name": display_name,
                    "stimulus_type": "controversial",
                    "bootstrap_idx": 0,
                    "n_stimuli": len(cstim_brain_idx),
                    "crsa": crsa_cstim,
                })

                # Vicco bootstrap samples
                vicco_file_idx = sdata["group_stim_idx"]["vicco"]
                subj_model_features_vicco = model_features_all_vicco[vicco_file_idx]

                for boot_idx, vicco_subset_idx in enumerate(sdata["vicco_bootstrap"]):
                    vicco_brain_idx = sdata["group_indices"]["vicco"][vicco_subset_idx]
                    brain_rdm_vicco = compute_rdm_correlation(betas[:, vicco_brain_idx].T)

                    model_features_vicco = subj_model_features_vicco[vicco_subset_idx]
                    model_rdm_vicco = compute_rdm_correlation(model_features_vicco)
                    crsa_vicco = compute_rsa_score(model_rdm_vicco, brain_rdm_vicco, method="spearman")

                    results[subject].append({
                        "subject": subject,
                        "model_set": model_set,
                        "model": model,
                        "display_name": display_name,
                        "stimulus_type": "vicco",
                        "bootstrap_idx": boot_idx,
                        "n_stimuli": sdata["n_vicco_sample"],
                        "crsa": crsa_vicco,
                    })

        # Save incrementally after each model_set
        for subject, rows in results.items():
            new_rows = rows[saved_counts[subject]:]
            if not new_rows:
                continue
            df_new = pd.DataFrame(new_rows)
            out_path = (RSA_DATA_DIR / subject / "crsa_scores.csv")
            header = not out_path.exists()
            df_new.to_csv(out_path, mode="a", header=header, index=False)
            saved_counts[subject] = len(rows)
            print(f"  Saved {len(new_rows)} rows for {subject} → {out_path}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
