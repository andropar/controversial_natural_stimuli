#!/usr/bin/env python3
"""
Recompute eval pipeline (discriminability + correlation matrices) with unique encodings.

1. Extract features for any models missing from LAION sample
2. For each model set: load payload, encode selected + random features with
   unique per-subject encodings, compute discriminability and correlation matrices

Usage:
    python recompute_with_unique_encodings.py                    # all model sets
    python recompute_with_unique_encodings.py --model-set sota   # single set
    python recompute_with_unique_encodings.py --skip-extraction  # if features already extracted
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.spatial.distance import pdist
from tqdm import tqdm

STAGE = Path(__file__).resolve().parents[2]
SHARE_ROOT = STAGE.parents[1]
sys.path.insert(0, str(SHARE_ROOT / "src"))
sys.path.insert(0, str(SHARE_ROOT / "src"))
sys.path.insert(0, str(SHARE_ROOT))

from cstims import paths
from cstims.constants import MODEL_SETS
UNIQUE_ENCODING_DIRS = paths.unique_encoding_dirs()
MODEL_LIST_CSV = paths.model_list_csv()
from cstims.encoding.linear import load_encoding_params_by_encoding, encode_batch_for_all_encodings
from cstims.feature_extraction.universal_extractor import UniversalFeatureExtractor

# ── Paths ────────────────────────────────────────────────────────────────────
SELECTION_ROOT = SHARE_ROOT / "00_stimulus_selection" / "results" / "selected_stimuli"
LAION_SAMPLE_DIR = SHARE_ROOT / "external_data" / "LAION_natural_sample"
LAION_FEATURES_DIR = LAION_SAMPLE_DIR / "features"
LAION_IMAGES_DIR = LAION_SAMPLE_DIR / "images"
OUTPUT_ROOT = STAGE / "results"

ALL_MODEL_SETS = ["all_models", "architecture", "dataset", "sota", "training_objective"]
ENCODING_NAMES = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]

N_RANDOM_IMAGES = 1000
N_RANDOM_SUBSETS = 10
N_NOISE_SAMPLES = 50
TARGET_NC = 0.46


# ── Feature extraction ───────────────────────────────────────────────────────

def get_model_feature_path(model_name: str) -> Path:
    """Find the feature .npy file for a model (handles naming variations)."""
    # Try exact match first
    for f in LAION_FEATURES_DIR.glob(f"{model_name}*.npy"):
        return f
    return None


def extract_missing_features(model_names: list[str], device: torch.device) -> dict[str, Path]:
    """Extract features for models not yet in the LAION sample features dir."""
    model_df = pd.read_csv(MODEL_LIST_CSV)
    paths = {}

    for model_name in model_names:
        existing = get_model_feature_path(model_name)
        if existing is not None:
            paths[model_name] = existing
            continue

        print(f"  Extracting features for {model_name}...")
        row = model_df[model_df["model"] == model_name].iloc[0]

        extractor = UniversalFeatureExtractor(
            model_name=model_name,
            layer=row["layer"],
            aggregation=row["aggregation"],
            source=row["source"],
        )

        # Load images
        image_files = sorted(LAION_IMAGES_DIR.glob("*.jpg"))[:N_RANDOM_IMAGES]
        from PIL import Image
        all_feats = []
        batch_size = 32
        for start in tqdm(range(0, len(image_files), batch_size), desc=f"  {model_name}"):
            batch_imgs = []
            for f in image_files[start:start + batch_size]:
                try:
                    img = Image.open(f).convert("RGB")
                    batch_imgs.append(img)
                except Exception:
                    continue
            if not batch_imgs:
                continue
            tensors = [extractor.preprocess(img) for img in batch_imgs]
            batch = torch.stack(tensors).to(device)
            with torch.no_grad():
                feats = extractor.extract(batch)
            if isinstance(feats, torch.Tensor):
                feats = feats.detach().cpu().numpy()
            feats = np.asarray(feats).reshape(len(batch_imgs), -1).astype(np.float32)
            all_feats.append(feats)

        features = np.concatenate(all_feats, axis=0)
        out_path = LAION_FEATURES_DIR / f"{model_name}_extracted.npy"
        np.save(out_path, features)
        print(f"  Saved {features.shape} to {out_path}")
        paths[model_name] = out_path

        # Free GPU
        del extractor
        torch.cuda.empty_cache()

    return paths


def extract_random_features(model_names: list[str], n_images: int,
                            device: torch.device) -> dict[str, np.ndarray]:
    """Extract random baseline features from LAION sample images at correct layers."""
    model_df = pd.read_csv(MODEL_LIST_CSV)
    image_files = sorted(LAION_IMAGES_DIR.glob("*.jpg"))[:n_images]
    from PIL import Image

    # Pre-load images once
    print(f"  Loading {len(image_files)} random images...")
    images = []
    for f in image_files:
        try:
            images.append(Image.open(f).convert("RGB"))
        except Exception:
            continue

    features = {}
    for model_name in model_names:
        # Check cache first (correct layer)
        cache_path = LAION_FEATURES_DIR / f"{model_name}_random_{n_images}.npy"
        if cache_path.exists():
            features[model_name] = np.load(cache_path).astype(np.float32)
            continue

        row = model_df[model_df["model"] == model_name].iloc[0]
        extractor = UniversalFeatureExtractor(
            model_name=model_name, layer=row["layer"],
            aggregation=row["aggregation"], source=row["source"],
        )

        all_feats = []
        batch_size = 32
        for start in range(0, len(images), batch_size):
            batch_imgs = images[start:start + batch_size]
            tensors = [extractor.preprocess(img) for img in batch_imgs]
            batch = torch.stack(tensors).to(device)
            with torch.no_grad():
                feats = extractor.extract(batch)
            if isinstance(feats, torch.Tensor):
                feats = feats.detach().cpu().numpy()
            all_feats.append(np.asarray(feats).reshape(len(batch_imgs), -1).astype(np.float32))

        arr = np.concatenate(all_feats, axis=0)
        np.save(cache_path, arr)
        features[model_name] = arr
        print(f"    {model_name}: {arr.shape}")

        del extractor
        torch.cuda.empty_cache()

    return features


# ── RDM / correlation computation ────────────────────────────────────────────

def compute_rdm_vector(features: np.ndarray) -> np.ndarray:
    return pdist(features, metric="correlation")


def compute_correlation_matrix(model_rdms: dict[str, np.ndarray]) -> pd.DataFrame:
    models = sorted(model_rdms.keys())
    rows = []
    for mi in models:
        for mj in models:
            r, _ = stats.spearmanr(model_rdms[mi], model_rdms[mj])
            rows.append({"model_i": mi, "model_j": mj, "correlation": r})
    return pd.DataFrame(rows)


# ── Noise calibration ────────────────────────────────────────────────────────

def calibrate_noise_std(features: np.ndarray, target_nc: float = 0.46,
                        n_iter: int = 20) -> float:
    """Binary search for noise std that gives target split-half RDM correlation."""
    n_stim, n_feat = features.shape
    lo, hi = 0.0, float(np.std(features)) * 5

    for _ in range(n_iter):
        mid = (lo + hi) / 2
        noise1 = np.random.randn(n_stim, n_feat).astype(np.float32) * mid
        noise2 = np.random.randn(n_stim, n_feat).astype(np.float32) * mid
        rdm1 = compute_rdm_vector(features + noise1)
        rdm2 = compute_rdm_vector(features + noise2)
        r, _ = stats.spearmanr(rdm1, rdm2)
        if r > target_nc:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2


# ── Discriminability computation ─────────────────────────────────────────────

def compute_discriminability(
    selected_features: dict[str, np.ndarray],
    random_features: dict[str, np.ndarray],
    n_random_subsets: int,
    n_noise_samples: int,
    target_nc: float,
    noise_multipliers: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute discriminability metrics and correlation matrices."""
    model_names = sorted(selected_features.keys())
    n_selected = next(iter(selected_features.values())).shape[0]

    # Calibrate noise per model using random features
    # Calibrate noise on selected features (consistent with old pipeline)
    print("  Calibrating noise...")
    noise_stds = {}
    for model in model_names:
        noise_stds[model] = calibrate_noise_std(selected_features[model], target_nc)

    # Compute clean RDMs for selected stimuli
    print("  Computing selected RDMs...")
    selected_rdms = {m: compute_rdm_vector(selected_features[m]) for m in model_names}

    # Correlation matrix (selected, clean)
    corr_df = compute_correlation_matrix(selected_rdms)
    corr_df["matrix_type"] = "selected_clean"

    # Compute discriminability across noise levels
    print("  Computing discriminability...")
    discrim_rows = []

    for mult in tqdm(noise_multipliers, desc="  Noise levels"):
        # Selected stimuli
        correct_selected = 0
        total_selected = 0
        for _ in range(n_noise_samples):
            # Add noise to each model's features, compute noisy RDMs
            noisy_rdms = {}
            for m in model_names:
                noise = np.random.randn(*selected_features[m].shape).astype(np.float32)
                noise *= noise_stds[m] * mult
                noisy_rdms[m] = compute_rdm_vector(selected_features[m] + noise)

            # Model identification: for each noisy RDM, find best match among clean
            for m_noisy in model_names:
                best_match = max(model_names,
                                 key=lambda m_clean: stats.spearmanr(noisy_rdms[m_noisy],
                                                                      selected_rdms[m_clean])[0])
                if best_match == m_noisy:
                    correct_selected += 1
                total_selected += 1

        error_selected = 1 - correct_selected / total_selected

        # Random baseline (average over subsets)
        n_random = next(iter(random_features.values())).shape[0]
        error_randoms = []
        for sub_idx in range(min(n_random_subsets, 10)):  # limit for speed
            # Sample n_selected random images
            rng = np.random.default_rng(sub_idx)
            idx = rng.choice(n_random, size=n_selected, replace=False)
            rand_feats = {m: random_features[m][idx] for m in model_names}
            rand_rdms = {m: compute_rdm_vector(rand_feats[m]) for m in model_names}

            correct = 0
            total = 0
            for _ in range(n_noise_samples):
                noisy = {}
                for m in model_names:
                    noise = np.random.randn(*rand_feats[m].shape).astype(np.float32)
                    noise *= noise_stds[m] * mult
                    noisy[m] = compute_rdm_vector(rand_feats[m] + noise)

                for m_noisy in model_names:
                    best = max(model_names,
                               key=lambda mc: stats.spearmanr(noisy[m_noisy], rand_rdms[mc])[0])
                    if best == m_noisy:
                        correct += 1
                    total += 1
            error_randoms.append(1 - correct / total)

        discrim_rows.append({
            "noise_multiplier": mult,
            "error_selected": error_selected,
            "error_random_mean": np.mean(error_randoms),
            "error_random_std": np.std(error_randoms),
            "n_models": len(model_names),
            "n_selected": n_selected,
        })

    return pd.DataFrame(discrim_rows), corr_df


# ── Main ─────────────────────────────────────────────────────────────────────

def process_model_set(model_set: str, device: torch.device):
    print(f"\n{'='*60}")
    print(f"Model set: {model_set}")
    print(f"{'='*60}")

    # Load payload
    payload_path = SELECTION_ROOT / model_set / "selected_stimuli_data.pkl"
    print(f"Loading payload...")
    with open(payload_path, "rb") as f:
        payload = pickle.load(f)

    model_names = payload["model_names"]
    print(f"  {len(model_names)} models")

    raw_features = payload.get("selected_features_raw") or payload.get("selected_features")

    # Extract random baseline features at correct layers
    print("Extracting random baseline features...")
    random_features = extract_random_features(model_names, N_RANDOM_IMAGES, device)

    output_dir = OUTPUT_ROOT / f"{model_set}_unique"
    output_dir.mkdir(parents=True, exist_ok=True)

    noise_multipliers = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0])

    all_corr_rows = []
    all_discrim_rows = []

    # === Raw track ===
    print("\n--- Raw track ---")
    raw_selected = {m: np.asarray(raw_features[m], dtype=np.float32) for m in model_names}
    discrim_df, corr_df = compute_discriminability(
        raw_selected, random_features, N_RANDOM_SUBSETS, N_NOISE_SAMPLES,
        TARGET_NC, noise_multipliers,
    )
    corr_df["track"] = "raw"
    discrim_df["track"] = "raw"
    all_corr_rows.append(corr_df)
    all_discrim_rows.append(discrim_df)

    # === Encoding tracks (unique per-subject) ===
    for enc_name in ENCODING_NAMES:
        print(f"\n--- Encoding track: {enc_name} ---")
        enc_root = UNIQUE_ENCODING_DIRS[enc_name]

        params = load_encoding_params_by_encoding(
            encoding_root=enc_root,
            model_list_csv=MODEL_LIST_CSV,
            encoding_names=[enc_name],
            device=device,
        )

        # Encode selected features
        raw_torch = {m: torch.tensor(np.asarray(raw_features[m], dtype=np.float32), device=device)
                     for m in model_names}
        encoded = encode_batch_for_all_encodings(raw_torch, {enc_name: params[enc_name]})
        enc_selected = {m: v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
                        for m, v in encoded[enc_name].items()}

        # Encode random features in batches
        enc_random = {m: [] for m in model_names}
        batch_size = 200
        n_total = next(iter(random_features.values())).shape[0]
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch = {m: torch.tensor(random_features[m][start:end], device=device)
                     for m in model_names}
            enc_batch = encode_batch_for_all_encodings(batch, {enc_name: params[enc_name]})
            for m in model_names:
                v = enc_batch[enc_name][m]
                enc_random[m].append(v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
        enc_random = {m: np.concatenate(v, axis=0) for m, v in enc_random.items()}

        discrim_df, corr_df = compute_discriminability(
            enc_selected, enc_random, N_RANDOM_SUBSETS, N_NOISE_SAMPLES,
            TARGET_NC, noise_multipliers,
        )
        corr_df["track"] = enc_name
        discrim_df["track"] = enc_name
        all_corr_rows.append(corr_df)
        all_discrim_rows.append(discrim_df)

        del raw_torch, encoded, enc_random
        torch.cuda.empty_cache()

    # Save
    corr_out = pd.concat(all_corr_rows, ignore_index=True)
    corr_out = corr_out[["track", "matrix_type", "model_i", "model_j", "correlation"]]
    corr_out.to_csv(output_dir / "correlation_matrices.csv", index=False)
    print(f"\nSaved correlation_matrices.csv ({len(corr_out)} rows)")

    discrim_out = pd.concat(all_discrim_rows, ignore_index=True)
    discrim_out.to_csv(output_dir / "discriminability.csv", index=False)
    print(f"Saved discriminability.csv ({len(discrim_out)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-set", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--skip-extraction", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    model_sets_to_run = [args.model_set] if args.model_set else ALL_MODEL_SETS

    # Extract missing features first
    if not args.skip_extraction:
        all_models = sorted(set(m for ms in model_sets_to_run for m in MODEL_SETS.get(ms, [])))
        if not all_models:
            # Load from payloads
            for ms in model_sets_to_run:
                payload_path = SELECTION_ROOT / ms / "selected_stimuli_data.pkl"
                with open(payload_path, "rb") as f:
                    payload = pickle.load(f)
                all_models = sorted(set(all_models) | set(payload["model_names"]))

        print(f"Checking features for {len(all_models)} models...")
        extract_missing_features(all_models, device)

    for ms in model_sets_to_run:
        process_model_set(ms, device)

    print("\nAll done!")


if __name__ == "__main__":
    main()
