#!/usr/bin/env python3
"""Sweep noise multipliers and compute the discriminability-margin curve.

For each (model_set, track, noise_mult) we compute:
    - random:   averaged over vicco subsamples; clean RDM_i vs noised RDM_j
    - selected: clean selected RDM_i vs noised selected RDM_j

For each correlation matrix we extract median diag (within-model) and median
off-diag (between-model). Margin = median diag - median off-diag.

This is the same statistic as the existing in-silico discriminability curve,
but in correlation units rather than non-parametric error probability —
trades the binary "did the right model win the contest" question for the
underlying r-space gap.

Outputs per model_set:
    <eval_pipeline_dir>/margin_curve.csv

Usage:
    python compute_margin_curve.py --tracks raw,sub-01
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

STAGE = Path(__file__).resolve().parents[2]
SHARE_ROOT = STAGE.parents[1]
sys.path.insert(0, str(SHARE_ROOT / "shared" / "code" / "paper_helpers"))
sys.path.insert(0, str(SHARE_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config
from utils import load_encoding_model
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

EVAL_DIRS = {
    ms: config.SELECTION_OUTPUT_ROOT / ms / "eval_pipeline"
    for ms in ["all_models", "sota", "training_objective", "architecture", "dataset"]
}
PAYLOAD_PATHS = {
    ms: config.SELECTION_OUTPUT_ROOT / ms / "selected_stimuli_data.pkl"
    for ms in EVAL_DIRS
}


def _layer_lookup() -> dict:
    df = pd.read_csv(config.MODEL_LIST_CSV)
    return dict(zip(df["model"], df["layer"]))


DEFAULT_METRIC = None        # None => read from selection/current paper config
DEFAULT_CORR_TYPE = None     # None => read from selection/current paper config
N_NOISE_SAMPLES = 50          # MC noise draws per noise level (lighter than the matrix script)
N_VICCO_SUBSAMPLES = 10       # random subsamples per noise level
N_VICCO_SAMPLE = 100
VICCO_SEED = 0
DEFAULT_MULTIPLIERS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


def _stable_seed(*parts: object) -> int:
    text = "::".join(str(p) for p in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**31)


def _current_noise_calibration_path(model_set: str, eval_dir: Path) -> Path:
    current = config.EVAL_DATA_DIR / f"{model_set}_unique_boot" / "noise_calibration.csv"
    if current.exists():
        return current
    return eval_dir / "noise_calibration.csv"


def load_vicco_features_full(layer_map: dict, model_names: list[str]) -> dict:
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
    """Load the rsynced 10k-image natural-pool subset, one .npz per model.

    Each .npz has key "features" of shape (n_pool, n_feat). All models share
    the same image indices (saved as ``_sampled_indices.npy``).
    """
    feats = {}
    for m in model_names:
        fp = pool_dir / f"{m}.npz"
        if not fp.exists():
            continue
        with np.load(fp, allow_pickle=True) as z:
            feats[m] = z["features"].astype(np.float32)
    return feats


def make_subsamples(n_total: int, n: int, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.sort(rng.choice(n_total, n, replace=False)) for _ in range(k)]


def project_features(features: np.ndarray, subject: str, model: str) -> np.ndarray:
    enc = load_encoding_model(model, subject)
    f = features.astype(np.float64)
    if enc["feature_mean"] is not None and np.any(enc["feature_mean"] != 0):
        f = f - enc["feature_mean"]
    if enc["feature_scale"] is not None and np.any(enc["feature_scale"] != 1):
        f = f / (enc["feature_scale"] + 1e-8)
    return f @ enc["weights"] + enc["intercept"]


def median_diag_offdiag(M: np.ndarray) -> tuple[float, float]:
    n = M.shape[0]
    diag = float(np.nanmedian(np.diag(M)))
    mask = ~np.eye(n, dtype=bool)
    off = float(np.nanmedian(M[mask]))
    return diag, off


def corr_matrix_at_noise(
    rdms: torch.Tensor,
    noise_stds: torch.Tensor,
    multiplier: float,
    n_noise: int,
    corr_type: str,
    seed: int | None = None,
) -> np.ndarray:
    """Wrapper: when multiplier == 0 just return the clean self-correlation."""
    if multiplier <= 0:
        return compute_clean_correlation_matrix(rdms, corr_type).numpy()
    if seed is None:
        return compute_correlation_at_target_noise(
            rdms, noise_stds * multiplier, corr_type, n_noise
        ).numpy()
    devices = []
    if rdms.is_cuda:
        devices = [rdms.device.index if rdms.device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return compute_correlation_at_target_noise(
            rdms, noise_stds * multiplier, corr_type, n_noise
        ).numpy()


def process_model_set(
    model_set: str,
    vicco_features_full: dict,
    tracks: list[str],
    multipliers: list[float],
    n_noise: int,
    subsamples: list[np.ndarray],
    out_name: str = "margin_curve.csv",
    metric: str | None = DEFAULT_METRIC,
    corr_type: str | None = DEFAULT_CORR_TYPE,
) -> int:
    eval_dir = EVAL_DIRS[model_set]
    payload_path = PAYLOAD_PATHS[model_set]
    if not payload_path.exists():
        print(f"  [{model_set}] no payload"); return 0
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
        print(f"  [{model_set}] no models"); return 0

    nc_path = _current_noise_calibration_path(model_set, eval_dir)
    if not nc_path.exists():
        print(f"  [{model_set}] missing noise_calibration"); return 0
    nc = pd.read_csv(nc_path)
    print(f"  [{model_set}] metric={metric}, corr_type={corr_type}, noise={nc_path}")

    rows = []
    for track in tqdm(tracks, desc=f"  {model_set}", leave=False):
        # Selected features per model on this track
        sel_feats = {}
        for m in available:
            sel = np.asarray(selected_features[m], dtype=np.float32)
            if track == "raw":
                sel_feats[m] = torch.from_numpy(sel)
            else:
                sel_feats[m] = torch.from_numpy(
                    project_features(sel, track, m).astype(np.float32))
        sel_rdms = compute_all_rdms(sel_feats, [metric])[metric]

        # Noise stds aligned to `available`
        nt = nc[nc["track"] == track]
        if nt.empty:
            continue
        std_vec = [
            float(nt[nt["model"] == m]["noise_std"].iloc[0])
            if not nt[nt["model"] == m].empty else 0.0
            for m in available
        ]
        noise_stds = torch.tensor(std_vec, dtype=torch.float32).reshape(-1, 1)

        # Pre-project full vicco features per model on this track
        if track == "raw":
            vic_proj_full = {m: vicco_features_full[m] for m in available}
        else:
            vic_proj_full = {
                m: project_features(vicco_features_full[m], track, m).astype(np.float32)
                for m in available
            }

        # Per-subsample random RDMs (cached so we don't recompute per multiplier)
        random_rdms_per_subsample = []
        for idx in subsamples:
            ran_feats = {m: torch.from_numpy(vic_proj_full[m][idx])
                         for m in available}
            random_rdms_per_subsample.append(
                compute_all_rdms(ran_feats, [metric])[metric]
            )

        for mult in multipliers:
            # Selected
            M_sel = corr_matrix_at_noise(
                sel_rdms,
                noise_stds,
                mult,
                n_noise,
                corr_type,
                seed=_stable_seed(
                    model_set, track, "selected", mult, metric, corr_type, n_noise
                ),
            )
            d_sel, o_sel = median_diag_offdiag(M_sel)
            rows.append({
                "track": track, "condition": "selected", "noise_mult": float(mult),
                "median_diag": d_sel, "median_offdiag": o_sel,
                "margin": d_sel - o_sel,
                "n_models": len(available),
            })
            # Random — average per-subsample summary stats (medians across subsamples)
            ds, os_ = [], []
            for idx_sub, ran_rdm in enumerate(random_rdms_per_subsample):
                M_ran = corr_matrix_at_noise(
                    ran_rdm,
                    noise_stds,
                    mult,
                    n_noise,
                    corr_type,
                    seed=_stable_seed(
                        model_set,
                        track,
                        "random",
                        idx_sub,
                        mult,
                        metric,
                        corr_type,
                        n_noise,
                        out_name,
                    ),
                )
                d, o = median_diag_offdiag(M_ran)
                ds.append(d); os_.append(o)
            d_ran = float(np.mean(ds))
            o_ran = float(np.mean(os_))
            rows.append({
                "track": track, "condition": "random", "noise_mult": float(mult),
                "median_diag": d_ran, "median_offdiag": o_ran,
                "margin": d_ran - o_ran,
                "n_models": len(available),
            })

    out_csv = eval_dir / out_name
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"  [{model_set}] saved {len(rows)} rows -> {out_csv.name}")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="raw,sub-01")
    ap.add_argument("--model_sets", default="all_models,sota,training_objective,architecture,dataset")
    ap.add_argument("--multipliers", default=",".join(str(m) for m in DEFAULT_MULTIPLIERS),
                    help="Comma-separated noise multipliers (0 = clean)")
    ap.add_argument("--n_noise", type=int, default=N_NOISE_SAMPLES)
    ap.add_argument("--n_subsamples", type=int, default=N_VICCO_SUBSAMPLES)
    ap.add_argument("--natural_pool_dir", type=Path, default=None,
                    help="Path to rsynced 10k natural-pool subset dir (per-model .npz). "
                         "When provided, uses LAION natural pool as the random "
                         "baseline instead of vicco. Output CSV is named "
                         "`margin_curve_pool.csv` to keep the vicco-based curve "
                         "in `margin_curve.csv` for comparison.")
    ap.add_argument("--metric", default=DEFAULT_METRIC,
                    help="RDM distance metric. Default: selection/current paper config.")
    ap.add_argument("--corr_type", default=DEFAULT_CORR_TYPE,
                    help="Cross-model RDM correlation type. Default: selection/current paper config.")
    args = ap.parse_args()

    tracks = [t.strip() for t in args.tracks.split(",")]
    model_sets = [m.strip() for m in args.model_sets.split(",")]
    mults = [float(x) for x in args.multipliers.split(",")]

    layer_map = _layer_lookup()
    needed = sorted({m for ms in model_sets for m in config.MODEL_SETS[ms]})

    if args.natural_pool_dir is not None:
        print(f"Loading natural-pool features from {args.natural_pool_dir}")
        random_features = load_natural_pool_features(args.natural_pool_dir, needed)
        random_label = "natural_pool"
    else:
        print(f"Loading vicco features for {len(needed)} models (curated baseline)")
        random_features = load_vicco_features_full(layer_map, needed)
        random_label = "vicco"

    if not random_features:
        raise RuntimeError("No random-baseline features loaded")
    n_total = next(iter(random_features.values())).shape[0]
    subsamples = make_subsamples(n_total, N_VICCO_SAMPLE, args.n_subsamples, VICCO_SEED)
    print(f"Random baseline: {random_label}, n_total={n_total}, "
          f"n_subsamples={args.n_subsamples}, n_noise={args.n_noise}, mults={mults}")

    out_name = ("margin_curve_pool.csv" if random_label == "natural_pool"
                else "margin_curve.csv")
    for ms in tqdm(model_sets, desc="Model sets"):
        process_model_set(ms, random_features, tracks, mults, args.n_noise,
                          subsamples, out_name=out_name, metric=args.metric,
                          corr_type=args.corr_type)
    print(f"Done. Output: <eval_pipeline>/{out_name}")


if __name__ == "__main__":
    main()
