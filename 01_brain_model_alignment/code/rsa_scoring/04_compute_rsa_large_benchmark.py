#!/usr/bin/env python3
"""
Compute CRSA and wRSA-transfer scores for all 125 large-benchmark models.

Unlike 02_compute_wrsa_transfer.py (which uses MODEL_SETS), every model here
is evaluated on ALL 6 stimulus sets (architecture, dataset, sota,
training_objective, all_models, vicco), regardless of which set it "belongs to".

Encoding models must be pre-fitted with:
    python fit_encoding_hydra.py paths=iris_large benchmark.image_set=unique

Usage:
    python 04_compute_rsa_large_benchmark.py --encoding-root /SSD/jroth/deepvision_encoding_models_large/runs/YYYYMMDD_HHMMSS
    python 04_compute_rsa_large_benchmark.py --encoding-root ... --subject sub-05
    python 04_compute_rsa_large_benchmark.py --encoding-root ... --resume

Outputs:
    01_brain_model_alignment/results/rsa_scores/rsa_large_benchmark_scores.csv
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
CSTIM_HDF5_ROOT = paths.cstim_hdf5_root()
get_brain_input_dir = paths.get_brain_input_dir
RSA_DATA_DIR = paths.rsa_data_dir()
from cstims.rdm import compute_rdm_correlation, compute_rsa_score
from cstims.sampling import bootstrap_sample_indices
from cstims.subjects import parse_subject_arg
from cstims.paper.utils import predict_voxel_responses

from cstims.feature_extraction.universal_extractor import UniversalFeatureExtractor


# All stimulus sets to evaluate every model on
ALL_GROUPS = ["architecture", "dataset", "sota", "training_objective", "all_models", "vicco"]

# model_list_large.csv location: 00_stimulus_selection/resources/model_list_large.csv
MODEL_LIST_LARGE = (
    Path(__file__).resolve().parents[3]
    / "00_stimulus_selection"
    / "resources"
    / "model_list_large.csv"
)


def _sanitize_layer(layer: str) -> str:
    """Match fit_encoding_hydra.py's sanitize_layer_name exactly."""
    return (
        str(layer)
        .replace(".", "_")
        .replace(":", "_")
        .replace("[", "_")
        .replace("]", "_")
        .replace("/", "_")
    )


def load_subject_brain_data(subject: str) -> dict:
    """Load brain data for a subject across all groups. Returns None if unavailable."""
    cache = load_cstim_brain_cache(subject, missing_ok=True)
    if cache is None:
        return None
    data = cache.as_legacy_group_dict()
    group_indices = data["group_indices"]
    n_vicco = len(group_indices.get("vicco", []))
    n_vicco_sample = min(100, n_vicco) if n_vicco > 0 else 0
    vicco_bootstrap = (
        bootstrap_sample_indices(n_vicco, n_vicco_sample, n_bootstrap=1000, seed=0)
        if n_vicco > 0 else []
    )
    data["vicco_bootstrap"] = vicco_bootstrap
    data["n_vicco_sample"] = n_vicco_sample
    return data


def load_images(group: str) -> list:
    """Load stimulus images for a group (respects architecture/dataset swap)."""
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
    return [Image.open(f).convert("RGB") for f in img_files]


def extract_features(model_name: str, images: list, model_df: pd.DataFrame,
                     device: str = "cuda:0") -> np.ndarray:
    """Extract features from PIL images using the model's own preprocessing."""
    import torch
    import concurrent.futures
    row = model_df[model_df["model"] == model_name].iloc[0]
    extractor = UniversalFeatureExtractor(
        model_name=model_name,
        layer=row["layer"],
        aggregation=row["aggregation"],
        source=row["source"],
        device=device,
    )
    batch_size = 32
    feats_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        for start in range(0, len(images), batch_size):
            batch_images = images[start:start + batch_size]
            tensors = list(pool.map(extractor.preprocess, batch_images))
            batch = torch.stack(tensors).to(extractor.device)
            # Retry with half batch on OOM
            current_bs = len(batch_images)
            while True:
                try:
                    with torch.no_grad():
                        feats = extractor.extract(batch)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    current_bs = max(1, current_bs // 2)
                    sub_feats = []
                    for s in range(0, len(batch_images), current_bs):
                        sub = batch[s:s + current_bs]
                        with torch.no_grad():
                            sub_feats.append(extractor.extract(sub))
                        torch.cuda.empty_cache()
                    feats = torch.cat(sub_feats, dim=0)
                    break
            if isinstance(feats, torch.Tensor):
                feats = feats.detach().cpu().numpy()
            feats = np.asarray(feats).reshape(len(batch_images), -1).astype(np.float32)
            feats_list.append(feats)
    return np.concatenate(feats_list, axis=0)


def load_large_encoding_model(model_name: str, subject: str, encoding_root: Path,
                               model_df: pd.DataFrame) -> dict:
    """Load encoding model from the large-benchmark run directory."""
    row = model_df[model_df["model"] == model_name].iloc[0]
    layer_safe = _sanitize_layer(row["layer"])
    folder_name = f"{subject}_{model_name}.layer{layer_safe}"
    npz_path = encoding_root / folder_name / "encoding_model.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Encoding model not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    return {
        "weights": data["weights"],
        "intercept": data["intercept"],
        "feature_mean": data["feature_mean"],
        "feature_scale": data["feature_scale"],
        "roi_hlvis": data["roi_hlvis"],
    }


def main():
    parser = argparse.ArgumentParser(description="Compute CRSA + wRSA for large benchmark models")
    parser.add_argument("--encoding-root", required=True,
                        help="Path to encoding model run dir, e.g. /SSD/jroth/deepvision_encoding_models_large/runs/20260403_120000")
    parser.add_argument("--subject", default="all",
                        help="Subject ID (e.g. sub-05) or 'all' (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Append to existing output file instead of overwriting")
    parser.add_argument("--device", default="cuda:0",
                        help="Device for feature extraction (default: cuda:0)")
    args = parser.parse_args()

    encoding_root = Path(args.encoding_root)
    if not encoding_root.exists():
        raise FileNotFoundError(f"Encoding root not found: {encoding_root}")

    subjects = parse_subject_arg(args.subject)
    print(f"Processing subjects: {subjects}")
    print(f"Encoding root: {encoding_root}")

    # Load model config once
    model_df = pd.read_csv(MODEL_LIST_LARGE)
    models = model_df["model"].tolist()
    print(f"Models to process: {len(models)}")

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

    out_path = RSA_DATA_DIR / "rsa_large_benchmark_scores.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_models = set()
    if args.resume and out_path.exists():
        existing = pd.read_csv(out_path)
        done_models = set(existing["model"].unique())
        print(f"Resuming: {len(done_models)} models already done")
    elif not args.resume and out_path.exists():
        out_path.unlink()
        print("Cleared existing output file (use --resume to append)")

    # Pre-load all group images once
    print("\nPre-loading stimulus images...")
    group_images = {}
    for group in ALL_GROUPS:
        group_images[group] = load_images(group)
        print(f"  {group}: {len(group_images[group])} images")

    # Build group slices for combined feature extraction
    group_slices = {}
    offset = 0
    for group in ALL_GROUPS:
        n = len(group_images[group])
        group_slices[group] = (offset, offset + n)
        offset += n
    all_images_combined = []
    for group in ALL_GROUPS:
        all_images_combined.extend(group_images[group])

    # Main loop: one forward pass per model for all images
    for model in tqdm(models, desc="Models"):
        if model in done_models:
            continue

        try:
            all_features = extract_features(model, all_images_combined, model_df, device=args.device)
        except Exception as ex:
            print(f"\n  SKIP (feature extraction failed) {model}: {ex}")
            continue

        model_rows = []

        for subject, sdata in subject_data.items():
            try:
                encoding = load_large_encoding_model(model, subject, encoding_root, model_df)
            except FileNotFoundError:
                print(f"\n  SKIP (no encoding) {model} / {subject}")
                continue

            encoding_hlvis = encoding["roi_hlvis"]
            betas = sdata["betas_hlvis"]

            for group in ALL_GROUPS:
                if group not in sdata["group_indices"]:
                    continue

                s, e = group_slices[group]
                model_features_group = all_features[s:e]

                if group == "vicco":
                    vicco_file_idx = sdata["group_stim_idx"]["vicco"]
                    subj_features_vicco = model_features_group[vicco_file_idx]

                    # Precompute predictions for ALL vicco stimuli once, then
                    # subset per bootstrap. Avoids redoing the features->voxels
                    # matmul 1000× per (model, subject). Same for brain-side
                    # subsetting (no matmul there — already an indexing op).
                    pred_all_vicco = predict_voxel_responses(subj_features_vicco, encoding)
                    pred_all_vicco_hlvis = pred_all_vicco[:, encoding_hlvis]

                    for boot_idx, vicco_subset_idx in enumerate(sdata["vicco_bootstrap"]):
                        vicco_brain_idx = sdata["group_indices"]["vicco"][vicco_subset_idx]
                        brain_rdm = compute_rdm_correlation(betas[:, vicco_brain_idx].T)
                        feat_subset = subj_features_vicco[vicco_subset_idx]

                        model_rdm = compute_rdm_correlation(feat_subset)
                        crsa = compute_rsa_score(model_rdm, brain_rdm, method="spearman")

                        pred_rdm = compute_rdm_correlation(pred_all_vicco_hlvis[vicco_subset_idx])
                        wrsa = compute_rsa_score(pred_rdm, brain_rdm, method="spearman")

                        model_rows.append({
                            "subject": subject,
                            "model": model,
                            "group": group,
                            "stimulus_type": "vicco",
                            "bootstrap_idx": boot_idx,
                            "n_stimuli": sdata["n_vicco_sample"],
                            "crsa": crsa,
                            "wrsa_transfer": wrsa,
                        })
                else:
                    brain_idx = sdata["group_indices"][group]
                    file_idx = sdata["group_stim_idx"][group]
                    subj_features = model_features_group[file_idx]

                    brain_rdm = compute_rdm_correlation(betas[:, brain_idx].T)

                    model_rdm = compute_rdm_correlation(subj_features)
                    crsa = compute_rsa_score(model_rdm, brain_rdm, method="spearman")

                    pred = predict_voxel_responses(subj_features, encoding)
                    pred_rdm = compute_rdm_correlation(pred[:, encoding_hlvis])
                    wrsa = compute_rsa_score(pred_rdm, brain_rdm, method="spearman")

                    model_rows.append({
                        "subject": subject,
                        "model": model,
                        "group": group,
                        "stimulus_type": "controversial",
                        "bootstrap_idx": 0,
                        "n_stimuli": len(brain_idx),
                        "crsa": crsa,
                        "wrsa_transfer": wrsa,
                    })

        if model_rows:
            df_new = pd.DataFrame(model_rows)
            write_header = not out_path.exists()
            df_new.to_csv(out_path, mode="a", header=write_header, index=False)

        # Explicit cleanup to prevent memory accumulation across models.
        # Without this, large models (ViT-H, etc.) were leaking RAM — the
        # process grew to 40 GB by model 42/125 and started swapping, causing
        # severe slowdown (87s/model → 1000s/model).
        try:
            del all_features, encoding, pred_all_vicco, pred_all_vicco_hlvis
        except NameError:
            pass
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nDone. Results at: {out_path}")


if __name__ == "__main__":
    main()
