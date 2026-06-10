#!/usr/bin/env python3
"""
Recompute correlation_matrices.csv with unique per-subject encoding models.

Only needs the selected features (from payload) + unique encoding models.
No LAION shard data needed.

Usage:
    python recompute_correlation_matrices.py --model-set all_models
    python recompute_correlation_matrices.py  # all model sets
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

# Paths
STAGE = Path(__file__).resolve().parents[2]
SHARE_ROOT = STAGE.parents[2]
sys.path.insert(0, str(SHARE_ROOT / "shared" / "code" / "paper_helpers"))
sys.path.insert(0, str(SHARE_ROOT / "src"))
sys.path.insert(0, str(SHARE_ROOT))

from config import UNIQUE_ENCODING_DIRS, MODEL_LIST_CSV
from cstims.encoding.linear import load_encoding_params_by_encoding, encode_batch_for_all_encodings

SELECTION_ROOT = SHARE_ROOT / "00_stimulus_selection" / "results" / "selected_stimuli"
MODEL_SETS = ["all_models", "architecture", "dataset", "sota", "training_objective"]
ENCODING_NAMES = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]


def load_payload(model_set: str) -> dict:
    path = SELECTION_ROOT / model_set / "selected_stimuli_data.pkl"
    print(f"Loading payload: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_rdm(features: np.ndarray, metric: str = "correlation") -> np.ndarray:
    """Compute RDM (upper triangle vector) from features (n_stimuli, n_features)."""
    from scipy.spatial.distance import pdist
    return pdist(features, metric=metric)


def compute_correlation_matrix(
    model_rdms: dict[str, np.ndarray],
    corr_type: str = "spearman",
) -> pd.DataFrame:
    """Compute pairwise correlations between model RDMs."""
    models = sorted(model_rdms.keys())
    rows = []
    for i, mi in enumerate(models):
        for j, mj in enumerate(models):
            if corr_type == "spearman":
                r, _ = stats.spearmanr(model_rdms[mi], model_rdms[mj])
            else:
                r, _ = stats.pearsonr(model_rdms[mi], model_rdms[mj])
            rows.append({"model_i": mi, "model_j": mj, "correlation": r})
    return pd.DataFrame(rows)


def process_model_set(model_set: str, device: torch.device, output_dir: Path):
    payload = load_payload(model_set)
    model_names = payload["model_names"]
    print(f"  {len(model_names)} models")

    # Get raw selected features from payload
    raw_features = payload.get("selected_features_raw") or payload.get("selected_features")
    if raw_features is None:
        print("  ERROR: no raw features in payload")
        return

    # Compute raw RDMs
    print("  Computing raw RDMs...")
    raw_rdms = {}
    for model in model_names:
        feats = raw_features[model]
        if isinstance(feats, torch.Tensor):
            feats = feats.cpu().numpy()
        feats = np.asarray(feats, dtype=np.float32)
        raw_rdms[model] = compute_rdm(feats)

    # Raw correlation matrix
    raw_corr = compute_correlation_matrix(raw_rdms)
    raw_corr["track"] = "raw"
    raw_corr["matrix_type"] = "selected_clean"
    all_rows = [raw_corr]

    # Encoding tracks with unique per-subject encodings
    for enc_name in ENCODING_NAMES:
        print(f"  Encoding track: {enc_name}")
        enc_root = UNIQUE_ENCODING_DIRS[enc_name]

        # Load encoding params
        params = load_encoding_params_by_encoding(
            encoding_root=enc_root,
            model_list_csv=MODEL_LIST_CSV,
            encoding_names=[enc_name],
            device=device,
        )

        # Convert raw features to torch tensors on device
        raw_torch = {
            m: torch.tensor(
                np.asarray(raw_features[m], dtype=np.float32),
                device=device,
            )
            for m in model_names
        }

        # Encode
        encoded = encode_batch_for_all_encodings(
            raw_torch, {enc_name: params[enc_name]}
        )
        enc_features = encoded[enc_name]

        # Compute encoded RDMs
        enc_rdms = {}
        for model in model_names:
            feats = enc_features[model]
            if isinstance(feats, torch.Tensor):
                feats = feats.cpu().numpy()
            feats = np.asarray(feats, dtype=np.float32)
            enc_rdms[model] = compute_rdm(feats)

        enc_corr = compute_correlation_matrix(enc_rdms)
        enc_corr["track"] = enc_name
        enc_corr["matrix_type"] = "selected_clean"
        all_rows.append(enc_corr)

        # Free GPU memory
        del raw_torch, encoded, enc_features
        torch.cuda.empty_cache()

    # Save
    df = pd.concat(all_rows, ignore_index=True)
    df = df[["track", "matrix_type", "model_i", "model_j", "correlation"]]
    out_path = output_dir / "correlation_matrices.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-set", type=str, default=None,
                        help="Single model set to process (default: all)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    model_sets = [args.model_set] if args.model_set else MODEL_SETS

    for ms in model_sets:
        print(f"\n{'='*60}")
        print(f"Model set: {ms}")
        print(f"{'='*60}")
        output_dir = STAGE / "results" / f"{ms}_unique"
        output_dir.mkdir(parents=True, exist_ok=True)
        process_model_set(ms, device, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
