#!/usr/bin/env python3
"""Augment cross-model RDM correlation matrices with random+noise variants.

For each (model_set, track) we compute 4 cross-model correlation matrices:
    - random_clean    — model RDMs on a fixed 100-image vicco subsample
    - selected_clean  — model RDMs on the 100 selected controversial stimuli
    - random_noised   — corr( clean random RDM_i , noised random RDM_j ),
                        averaged over n_noise_samples noise draws
    - selected_noised — same form, on selected RDMs

The "noised" matrices reproduce the discriminability statistic the selection
procedure was actually optimizing against: the calibrated brain noise applied
to the model RDM. Noise stds come from the eval pipeline's noise_calibration.csv.

We use vicco as the random baseline (rather than the 10k natural pool the
original eval used) so that random_clean and random_noised come from the same
stimulus set and are directly comparable.

Outputs per model_set:
    <eval_pipeline_dir>/correlation_matrices_with_random_noised.csv

Usage:
    python compute_correlation_matrices_with_noised.py
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys
from pathlib import Path

# Single-thread BLAS — workload is small per task
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

STAGE = Path(__file__).resolve().parents[2]
SHARE_ROOT = STAGE.parents[1]
sys.path.insert(0, str(SHARE_ROOT / "src"))
sys.path.insert(0, str(SHARE_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cstims.paper import config
from cstims.paper.utils import load_encoding_model, load_model_layer_mapping
from cstims.evaluation.computation import (
    compute_all_rdms,
    compute_clean_correlation_matrix,
    compute_correlation_at_target_noise,
)


VICCO_CACHE = (
    SHARE_ROOT
    / "05_controls_and_supplementary"
    / "model_scope_followups"
    / "layer_sweep"
    / "cache_or_heavy"
    / "features"
)
SUBJECTS = list(config.SUBJECTS)

EVAL_DIRS = {
    ms: config.SELECTION_OUTPUT_ROOT / ms / "eval_pipeline"
    for ms in ["all_models", "sota", "training_objective", "architecture", "dataset"]
}
PAYLOAD_PATHS = {
    ms: config.SELECTION_OUTPUT_ROOT / ms / "selected_stimuli_data.pkl"
    for ms in EVAL_DIRS
}

# Layer-name lookup direct from model_list.csv (NOT load_model_layer_mapping
# which adds a "layer" prefix and underscores — vicco cache uses raw dotted form)
def _layer_lookup() -> dict:
    df = pd.read_csv(config.MODEL_LIST_CSV)
    return dict(zip(df["model"], df["layer"]))


N_NOISE_SAMPLES = 100
N_VICCO_SAMPLE = 100         # stimuli per vicco subsample (matches selected n)
N_VICCO_SUBSAMPLES = 20      # number of random vicco subsamples to average over
VICCO_SEED = 0
DEFAULT_METRIC = None        # None => read from selection/current paper config
DEFAULT_CORR_TYPE = None     # None => read from selection/current paper config


def _stable_seed(*parts: object) -> int:
    """Stable 31-bit seed from semantic parts, independent of Python hash salt."""
    text = "::".join(str(p) for p in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**31)


def load_vicco_features_full(layer_map: dict, model_names: list[str]) -> dict:
    """Load the FULL vicco feature stack per model (no subsampling)."""
    feats = {}
    for m in model_names:
        layer = layer_map.get(m)
        if layer is None:
            continue
        fp = VICCO_CACHE / m / "vicco.npz"
        if not fp.exists():
            continue
        with np.load(fp, allow_pickle=True) as z:
            if layer not in z.files:
                continue
            feats[m] = z[layer].astype(np.float32)
    return feats


def load_natural_pool_features(pool_dir: Path, model_names: list[str]) -> dict:
    """Load the rsynced 10k-image natural-pool subset, one .npz per model."""
    feats = {}
    for m in model_names:
        fp = pool_dir / f"{m}.npz"
        if not fp.exists():
            continue
        with np.load(fp, allow_pickle=True) as z:
            feats[m] = z["features"].astype(np.float32)
    return feats


def make_vicco_subsamples(n_total: int, n_sample: int, n_subsamples: int,
                          seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.sort(rng.choice(n_total, n_sample, replace=False))
            for _ in range(n_subsamples)]


def project_features(features: np.ndarray, subject: str, model: str) -> np.ndarray:
    """Apply encoding model to raw features, returning predicted voxel responses."""
    enc = load_encoding_model(model, subject)
    f = features.astype(np.float64)
    if enc["feature_mean"] is not None and np.any(enc["feature_mean"] != 0):
        f = f - enc["feature_mean"]
    if enc["feature_scale"] is not None and np.any(enc["feature_scale"] != 1):
        f = f / (enc["feature_scale"] + 1e-8)
    return f @ enc["weights"] + enc["intercept"]


def _compute_for_features(
    feat_dict: dict,
    available: list,
    noise_stds: torch.Tensor,
    n_noise_samples: int,
    metric: str,
    corr_type: str,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (clean, noised) cross-model correlation matrices for a feature dict."""
    rdms = compute_all_rdms(feat_dict, [metric])[metric]
    clean = compute_clean_correlation_matrix(rdms, corr_type).numpy()

    if seed is None:
        noised = compute_correlation_at_target_noise(
            rdms, noise_stds, corr_type, n_noise_samples
        ).numpy()
    else:
        devices = []
        if rdms.is_cuda:
            devices = [rdms.device.index if rdms.device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            noised = compute_correlation_at_target_noise(
                rdms, noise_stds, corr_type, n_noise_samples
            ).numpy()
    return clean, noised


def _current_noise_calibration_path(model_set: str, eval_dir: Path) -> Path:
    """Prefer the current paper eval noise calibration, fallback to legacy output."""
    current = config.EVAL_DATA_DIR / f"{model_set}_unique_boot" / "noise_calibration.csv"
    if current.exists():
        return current
    return eval_dir / "noise_calibration.csv"


def process_model_set(
    model_set: str,
    vicco_features_full: dict,
    tracks: list[str],
    n_noise_samples: int,
    vicco_subsamples: list[np.ndarray],
    out_name: str = "correlation_matrices_with_random_noised.csv",
    metric: str | None = DEFAULT_METRIC,
    corr_type: str | None = DEFAULT_CORR_TYPE,
) -> int:
    """Compute random and selected corr matrices for one model_set.

    For the *random* matrices we average over multiple 100-image vicco
    subsamples (the original eval pipeline averaged over many 100-stim subsets
    of a 10k natural pool; here we do the same idea with the 292-image vicco
    set). The *selected* matrices use the fixed 100 selected controversial
    stimuli from the payload (no subsampling).
    """
    eval_dir = EVAL_DIRS[model_set]
    payload_path = PAYLOAD_PATHS[model_set]
    if not payload_path.exists():
        print(f"  [{model_set}] no payload at {payload_path}")
        return 0
    with open(payload_path, "rb") as fp:
        payload = pickle.load(fp)
    selected_features = payload.get("selected_features_raw") or {}
    payload_config = payload.get("config", {})
    metric = metric or payload_config.get("metric", "cosine")
    corr_type = corr_type or payload_config.get("corr_type", "spearman")

    set_models = list(config.MODEL_SETS[model_set])
    available = [m for m in set_models
                 if m in vicco_features_full and m in selected_features]
    if not available:
        print(f"  [{model_set}] no overlap of model_set with vicco+payload")
        return 0
    print(f"  [{model_set}] {len(available)}/{len(set_models)} models available")

    nc_path = _current_noise_calibration_path(model_set, eval_dir)
    if not nc_path.exists():
        print(f"  [{model_set}] missing noise_calibration.csv")
        return 0
    nc = pd.read_csv(nc_path)
    print(f"  [{model_set}] metric={metric}, corr_type={corr_type}, noise={nc_path}")

    rows = []
    for track in tracks:
        # Project selected features once per track
        sel_track = {}
        for m in available:
            sel_raw = np.asarray(selected_features[m], dtype=np.float32)
            if track == "raw":
                sel_track[m] = torch.from_numpy(sel_raw)
            else:
                sel_proj = project_features(sel_raw, track, m).astype(np.float32)
                sel_track[m] = torch.from_numpy(sel_proj)

        # Noise stds vector aligned to `available`
        nt = nc[nc["track"] == track]
        if nt.empty:
            print(f"  [{model_set}/{track}] no noise calibration rows")
            continue
        std_vec = [
            float(nt[nt["model"] == m]["noise_std"].iloc[0])
            if not nt[nt["model"] == m].empty else 0.0
            for m in available
        ]
        noise_stds = torch.tensor(std_vec, dtype=torch.float32).reshape(-1, 1)

        # Selected matrices (no subsampling — fixed 100 selected stims)
        sel_clean, sel_noised = _compute_for_features(
            sel_track,
            available,
            noise_stds,
            n_noise_samples,
            metric,
            corr_type,
            seed=_stable_seed(
                model_set, track, "selected", metric, corr_type, n_noise_samples
            ),
        )

        # Random matrices: average across vicco subsamples
        ran_clean_list, ran_noised_list = [], []
        # Pre-project full vicco features per (model, track) once to avoid
        # re-projecting per subsample.
        if track == "raw":
            vic_proj_full = {m: vicco_features_full[m] for m in available}
        else:
            vic_proj_full = {
                m: project_features(vicco_features_full[m], track, m).astype(np.float32)
                for m in available
            }

        for idx_subset in vicco_subsamples:
            ran_track = {
                m: torch.from_numpy(vic_proj_full[m][idx_subset]) for m in available
            }
            c, n = _compute_for_features(
                ran_track,
                available,
                noise_stds,
                n_noise_samples,
                metric,
                corr_type,
                seed=_stable_seed(
                    model_set,
                    track,
                    "random",
                    len(ran_clean_list),
                    metric,
                    corr_type,
                    n_noise_samples,
                    out_name,
                ),
            )
            ran_clean_list.append(c)
            ran_noised_list.append(n)
        ran_clean = np.mean(np.stack(ran_clean_list, axis=0), axis=0)
        ran_noised = np.mean(np.stack(ran_noised_list, axis=0), axis=0)

        for mt_name, M in [
            ("random_clean",  ran_clean),
            ("selected_clean", sel_clean),
            ("random_noised",  ran_noised),
            ("selected_noised", sel_noised),
        ]:
            for i, mi in enumerate(available):
                for j, mj in enumerate(available):
                    rows.append({
                        "track": track,
                        "matrix_type": mt_name,
                        "model_i": mi,
                        "model_j": mj,
                        "correlation": float(M[i, j]),
                    })

    out_csv = eval_dir / out_name
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"  [{model_set}] saved {len(rows)} rows -> {out_csv.name}")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="raw,sub-01,sub-03,sub-05,sub-06,sub-07",
                    help="Comma-separated track names to process")
    ap.add_argument("--n_noise", type=int, default=N_NOISE_SAMPLES,
                    help="Noise draws averaged for the noised matrices")
    ap.add_argument("--n_subsamples", type=int, default=N_VICCO_SUBSAMPLES,
                    help="Number of subsamples averaged for random matrices")
    ap.add_argument("--n_vicco", type=int, default=N_VICCO_SAMPLE,
                    help="Images per subsample (matches selected n=100)")
    ap.add_argument("--model_sets", default="all_models,sota,training_objective,architecture,dataset")
    ap.add_argument("--natural_pool_dir", type=Path, default=None,
                    help="Path to rsynced 10k natural-pool subset dir (per-model .npz). "
                         "When provided, uses LAION natural pool as the random baseline. "
                         "Output CSV is named correlation_matrices_with_random_noised_pool.csv "
                         "to keep the vicco-based file separate.")
    ap.add_argument("--metric", default=DEFAULT_METRIC,
                    help="RDM distance metric. Default: selection/current paper config.")
    ap.add_argument("--corr_type", default=DEFAULT_CORR_TYPE,
                    help="Cross-model RDM correlation type. Default: selection/current paper config.")
    args = ap.parse_args()

    tracks = [t.strip() for t in args.tracks.split(",")]
    model_sets = [m.strip() for m in args.model_sets.split(",")]

    layer_map = _layer_lookup()
    needed = sorted({m for ms in model_sets for m in config.MODEL_SETS[ms]})

    if args.natural_pool_dir is not None:
        print(f"Loading natural-pool features from {args.natural_pool_dir}")
        random_features = load_natural_pool_features(args.natural_pool_dir, needed)
        random_label = "natural_pool"
        out_name = "correlation_matrices_with_random_noised_pool.csv"
    else:
        print(f"Pre-loading vicco features for {len(needed)} models (curated baseline)")
        random_features = load_vicco_features_full(layer_map, needed)
        random_label = "vicco"
        out_name = "correlation_matrices_with_random_noised.csv"

    if not random_features:
        raise RuntimeError(f"No {random_label} features loaded")
    n_total = next(iter(random_features.values())).shape[0]
    subsamples = make_vicco_subsamples(n_total, args.n_vicco, args.n_subsamples,
                                       VICCO_SEED)
    print(f"Random baseline = {random_label}; loaded {len(random_features)} models "
          f"(n_total={n_total}); {args.n_subsamples} random subsamples of size {args.n_vicco}.")

    for ms in tqdm(model_sets, desc="Model sets"):
        process_model_set(ms, random_features, tracks, args.n_noise, subsamples,
                          out_name=out_name, metric=args.metric,
                          corr_type=args.corr_type)

    print(f"Done. Output: <eval_pipeline>/{out_name}")


if __name__ == "__main__":
    main()
