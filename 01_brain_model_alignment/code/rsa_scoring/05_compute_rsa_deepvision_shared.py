#!/usr/bin/env python3
"""Compute cRSA and wRSA-transfer scores on DeepVision *shared* brain data.

Uses the unique-trained per-subject encoding models (UNIQUE_ENCODING_DIRS),
applied out-of-distribution to the DeepVision shared stimulus set, and
compares against actual brain responses to those same shared stimuli.

Bootstrap-of-100 samples for direct comparability with the existing
'vicco' rows in crsa_scores.csv / wrsa_transfer_scores.csv.

Outputs (appends to existing CSVs):
    data/{subject}/crsa_scores.csv             rows with stimulus_type='deepvision_shared'
    data/{subject}/wrsa_transfer_scores.csv    rows with stimulus_type='deepvision_shared'

Usage:
    python 05_compute_rsa_deepvision_shared.py                  # all subjects
    python 05_compute_rsa_deepvision_shared.py --subject sub-05 # single subject
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))

# Limit BLAS thread oversubscription before numpy is imported.
# Workers get OMP_NUM_THREADS=1 via env (set in main before pool creation).
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("DV_BLAS_THREADS", "8"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("DV_BLAS_THREADS", "8"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("DV_BLAS_THREADS", "8"))

_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from tqdm import tqdm

from cstims import paths
from cstims.constants import MODEL_SETS, MODEL_DISPLAY_NAMES
RSA_DATA_DIR = paths.rsa_data_dir()
SHARED_ENCODING_ROOT = paths.shared_encoding_root()
PROJECT_ROOT = paths.project_root()
VOXEL_CACHE_DIR = paths.voxel_cache_dir()
from cstims.rdm import compute_rdm_correlation
from cstims.sampling import bootstrap_sample_indices
from cstims.subjects import parse_subject_arg
from cstims.paper.utils import load_encoding_model


DV_SHARED_CACHE = (
    VOXEL_CACHE_DIR / "deepvision_shared_visual_cve0p20" / "finalinterp"
)
N_BOOTSTRAPS = int(os.environ.get("DV_SHARED_N_BOOTSTRAPS", "1000"))
BOOTSTRAP_N = 100
STIMULUS_TYPE = "deepvision_shared"


def _rdm_upper_vec(rdm: np.ndarray) -> np.ndarray:
    """Upper triangular (k=1) of an (n,n) RDM as a flat vector."""
    n = rdm.shape[0]
    iu = np.triu_indices(n, k=1)
    return rdm[iu]


def _rank_vector(vec: np.ndarray) -> np.ndarray:
    """Rank a 1D vector (average for ties), float32."""
    return rankdata(vec, method="average").astype(np.float32)


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r between two 1D vectors (float32 in)."""
    xm = x - x.mean()
    ym = y - y.mean()
    num = float(np.dot(xm, ym))
    den = float(np.sqrt(np.dot(xm, xm) * np.dot(ym, ym)))
    return num / den if den > 0 else float("nan")


def load_subject_betas_hlvis(subject: str):
    """Load cached DeepVision shared betas, masked to hlvis voxels.

    Returns dict with betas_hlvis (n_hlvis, n_stim), hlvis_mask, n_visual,
    or None if cache is missing.
    """
    p = DV_SHARED_CACHE / subject
    betas_fp = p / "voxel_betas.npy"
    bsa_fp = p / "brain_space_arrays.npz"
    if not betas_fp.exists() or not bsa_fp.exists():
        return None
    betas = np.load(betas_fp).astype(np.float32)  # (n_visual, n_stim)
    bsa = np.load(bsa_fp)
    hlvis_mask = bsa["hlvis_mask"]
    if betas.shape[0] != hlvis_mask.shape[0]:
        raise RuntimeError(
            f"{subject}: betas voxels {betas.shape[0]} != hlvis_mask {hlvis_mask.shape[0]}"
        )
    return {
        "betas_hlvis": betas[hlvis_mask, :],
        "hlvis_mask": hlvis_mask,
        "n_visual": betas.shape[0],
        "n_hlvis": int(hlvis_mask.sum()),
        "n_stim": betas.shape[1],
    }


def load_shared_features(model: str, layer_with_prefix: str) -> np.ndarray:
    """Load cached features for shared stimuli.

    These were extracted during the shared-encoding fit (encoding_20251222_141301)
    and are identical across subjects (same images + deterministic models).
    """
    folder = SHARED_ENCODING_ROOT / f"sub-01_{model}.{layer_with_prefix}"
    return np.load(folder / "features.npz")["features"].astype(np.float32)


def _process_one_model(
    model: str,
    subjects: list,
    bootstrap: list,
    brain_rdm_ranks_per_subject: dict,
    layer_map: dict,
):
    """Compute cRSA & wRSA-transfer scores for one model across subjects+bootstraps.

    `brain_rdm_ranks_per_subject[s]` is a list (one per boot) of pre-ranked
    upper-triangular RDM vectors (Spearman = Pearson on ranks).
    """
    layer_with_prefix = layer_map[model]
    display = MODEL_DISPLAY_NAMES.get(model, model)
    feats = load_shared_features(model, layer_with_prefix)

    # Pre-load encodings, pre-mask weights to hlvis (in float32 for speed)
    encs = {}
    for s in subjects:
        e = load_encoding_model(model, s)
        roi_hlvis = e["roi_hlvis"]
        encs[s] = {
            "weights_hlvis": np.ascontiguousarray(e["weights"][:, roi_hlvis], dtype=np.float32),
            "intercept_hlvis": e["intercept"][roi_hlvis].astype(np.float32),
            "feature_mean": e["feature_mean"].astype(np.float32),
            "feature_scale": e["feature_scale"].astype(np.float32),
        }

    crsa_per_subj = {s: np.empty(len(bootstrap), dtype=np.float32) for s in subjects}
    wrsa_per_subj = {s: np.empty(len(bootstrap), dtype=np.float32) for s in subjects}

    for boot_idx, sub in enumerate(bootstrap):
        f_sub = feats[sub]
        model_rdm = compute_rdm_correlation(f_sub)
        model_vec = _rdm_upper_vec(model_rdm)
        model_ranks = _rank_vector(model_vec)

        for s in subjects:
            brain_ranks = brain_rdm_ranks_per_subject[s][boot_idx]
            crsa_per_subj[s][boot_idx] = _pearson_r(model_ranks, brain_ranks)

            enc = encs[s]
            f_norm = (f_sub - enc["feature_mean"]) / (enc["feature_scale"] + 1e-8)
            pred_hlvis = f_norm @ enc["weights_hlvis"] + enc["intercept_hlvis"]
            pred_rdm = compute_rdm_correlation(pred_hlvis)
            pred_ranks = _rank_vector(_rdm_upper_vec(pred_rdm))
            wrsa_per_subj[s][boot_idx] = _pearson_r(pred_ranks, brain_ranks)

    return model, display, crsa_per_subj, wrsa_per_subj


def main():
    parser = argparse.ArgumentParser(description="cRSA & wRSA-transfer on DeepVision shared brain data")
    parser.add_argument("--subject", default="all", help="Subject ID or 'all' (default)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Process pool size for model-level parallelism")
    args = parser.parse_args()

    subjects = parse_subject_arg(args.subject)
    layer_map = paths.model_layer_mapping()

    # Load + hlvis-mask brain betas for each subject
    print("Loading shared brain data per subject...")
    subject_data = {}
    for s in subjects:
        d = load_subject_betas_hlvis(s)
        if d is None:
            print(f"  {s}: no shared brain cache, skipping")
            continue
        subject_data[s] = d
        print(f"  {s}: visual={d['n_visual']}, hlvis={d['n_hlvis']}, n_stim={d['n_stim']}")
    if not subject_data:
        print("No subjects with cached shared brain data.")
        return

    # All subjects share the same n_stim (1492); single bootstrap set
    n_stim = next(iter(subject_data.values()))["n_stim"]
    if not all(d["n_stim"] == n_stim for d in subject_data.values()):
        raise RuntimeError("Inconsistent n_stim across subjects")
    bootstrap = bootstrap_sample_indices(n_stim, BOOTSTRAP_N, n_bootstrap=N_BOOTSTRAPS, seed=0)

    # Pre-compute brain RDM rank vectors per (subject, boot_idx).
    # Spearman is Pearson on ranks, so caching ranks lets the inner loop avoid
    # scipy.stats.spearmanr overhead.
    print(f"Pre-computing {len(subject_data)} x {N_BOOTSTRAPS} brain RDM rank vectors...")
    brain_rdm_ranks_per_subject = {}
    for s, d in subject_data.items():
        rank_vecs = []
        b = d["betas_hlvis"]
        for sub in tqdm(bootstrap, desc=s, leave=False):
            rdm = compute_rdm_correlation(b[:, sub].T)
            rank_vecs.append(_rank_vector(_rdm_upper_vec(rdm)))
        brain_rdm_ranks_per_subject[s] = rank_vecs

    # Free betas memory once rank vectors are cached
    for d in subject_data.values():
        d.pop("betas_hlvis", None)

    # Models: compute once per unique model (scores don't depend on model_set)
    all_models = sorted({m for ms in MODEL_SETS.values() for m in ms})
    subjects_with_data = list(subject_data.keys())

    # Clear any prior deepvision_shared rows in target CSVs (idempotent re-run)
    for s in subjects_with_data:
        for fname in ("crsa_scores.csv", "wrsa_transfer_scores.csv"):
            fp = RSA_DATA_DIR / s / fname
            if fp.exists():
                df = pd.read_csv(fp)
                if "stimulus_type" in df.columns and (df.stimulus_type == STIMULUS_TYPE).any():
                    df = df[df.stimulus_type != STIMULUS_TYPE]
                    df.to_csv(fp, index=False)
                    print(f"  Cleared existing {STIMULUS_TYPE} rows in {fp}")

    print(f"Processing {len(all_models)} models with {args.workers} workers "
          f"(BLAS threads/worker={os.environ['OMP_NUM_THREADS']})...")
    model_results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(_process_one_model, m, subjects_with_data, bootstrap,
                      brain_rdm_ranks_per_subject, layer_map): m
            for m in all_models
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Models"):
            model, display, crsa, wrsa = fut.result()
            model_results[model] = (display, crsa, wrsa)

    # Emit rows duplicated per (model_set, model) pair to match existing CSV convention
    crsa_rows_by_subj = {s: [] for s in subjects_with_data}
    wrsa_rows_by_subj = {s: [] for s in subjects_with_data}
    for model_set, models in MODEL_SETS.items():
        for model in models:
            display, crsa, wrsa = model_results[model]
            for s in subjects_with_data:
                for boot_idx in range(len(bootstrap)):
                    base = dict(
                        subject=s, model_set=model_set, model=model,
                        display_name=display, stimulus_type=STIMULUS_TYPE,
                        bootstrap_idx=boot_idx, n_stimuli=BOOTSTRAP_N,
                    )
                    crsa_rows_by_subj[s].append({**base, "crsa": float(crsa[s][boot_idx])})
                    wrsa_rows_by_subj[s].append({**base, "wrsa_transfer": float(wrsa[s][boot_idx])})

    crsa_cols = ["subject", "model_set", "model", "display_name", "stimulus_type",
                 "bootstrap_idx", "n_stimuli", "crsa"]
    wrsa_cols = ["subject", "model_set", "model", "display_name", "stimulus_type",
                 "bootstrap_idx", "n_stimuli", "wrsa_transfer"]

    for s in subjects_with_data:
        df_c = pd.DataFrame(crsa_rows_by_subj[s])[crsa_cols]
        out = RSA_DATA_DIR / s / "crsa_scores.csv"
        df_c.to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"  Appended {len(df_c)} cRSA rows -> {out}")

        df_w = pd.DataFrame(wrsa_rows_by_subj[s])[wrsa_cols]
        out = RSA_DATA_DIR / s / "wrsa_transfer_scores.csv"
        df_w.to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"  Appended {len(df_w)} wRSA-transfer rows -> {out}")

    print("\nDone!")


if __name__ == "__main__":
    main()
