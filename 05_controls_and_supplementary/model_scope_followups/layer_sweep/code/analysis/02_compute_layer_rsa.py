#!/usr/bin/env python3
"""Compute fixed RSA per (subject, model, layer, stimulus_set, bootstrap_idx).

Mirrors 02_rsa_scores/01_compute_crsa.py exactly except that we also iterate
over multiple layers per model (and skip wRSA-transfer, since we don't have
encoding models trained at the alternative layers).

For controversial sets, one row per (subject, model, layer, set) using the
subject's available stimuli.
For vicco, N_VICCO_BOOTSTRAPS rows per (subject, model, layer) with subsets of
size 100 (matching main pipeline). N_VICCO_BOOTSTRAPS is 50 by default to keep
runtime reasonable; pass --n-vicco-boot to override.

Output:
    11_layer_sweep/data/fixed_rsa_layer_sweep.csv

Parallelization:
    - Per (subject, model, layer, set): vectorized RDM computation.
    - Across (model, layer, set, bootstrap): a process pool runs all
      bootstrap RDM/RSA computations for vicco in parallel via joblib.
"""

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from config import (
    MODEL_DISPLAY_NAMES, PAPER_ROOT, get_brain_input_dir, SUBJECTS,
)
from utils import (
    compute_rdm_correlation, compute_rsa_score, bootstrap_sample_indices,
    parse_subject_arg,
)
from layers_config import MODEL_LAYERS, STIMULUS_SETS

CACHE_ROOT = LAYER_SWEEP_ROOT / "cache_or_heavy" / "features"
DATA_DIR = LAYER_SWEEP_ROOT / "results"
OUT_CSV = DATA_DIR / "fixed_rsa_layer_sweep.csv"


def load_layer_features(model: str, stimulus_set: str, layer_name: str) -> np.ndarray:
    p = CACHE_ROOT / model / f"{stimulus_set}.npz"
    d = np.load(p, allow_pickle=True)
    return d[layer_name]


def load_subject_brain_data(subject: str, n_vicco_boot: int) -> dict:
    data_dir = get_brain_input_dir(subject)
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

    group_indices, group_stim_idx = {}, {}
    for group in available_groups:
        mask = stim_info["group"] == group
        keys = stim_info.loc[mask, "stim_key"].values
        group_indices[group] = np.array([stim_key_to_idx[k] for k in keys])
        idx = stim_info.loc[mask, "stim_idx"].values
        group_stim_idx[group] = idx - 1 if group == "vicco" else idx

    n_vicco = len(group_indices.get("vicco", []))
    n_vicco_sample = min(100, n_vicco) if n_vicco > 0 else 0
    vicco_boot = bootstrap_sample_indices(n_vicco, n_vicco_sample,
                                          n_bootstrap=n_vicco_boot, seed=0) if n_vicco > 0 else []
    return {
        "betas_hlvis": betas_hlvis,
        "group_indices": group_indices,
        "group_stim_idx": group_stim_idx,
        "available_groups": available_groups,
        "vicco_bootstrap": vicco_boot,
        "n_vicco_sample": n_vicco_sample,
        "n_hlvis": int(hlvis_mask.sum()),
    }


def _vicco_boot_one(boot_idx, vicco_subset_idx, betas, vicco_brain_idx_full,
                    feats_vicco_subj, n_vicco_sample):
    vicco_brain_idx = vicco_brain_idx_full[vicco_subset_idx]
    brain_rdm = compute_rdm_correlation(betas[:, vicco_brain_idx].T)
    feats = feats_vicco_subj[vicco_subset_idx]
    model_rdm = compute_rdm_correlation(feats)
    score = compute_rsa_score(model_rdm, brain_rdm, method="spearman")
    return {"bootstrap_idx": boot_idx, "n_stimuli": n_vicco_sample, "rsa": score}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="all")
    parser.add_argument("--n-vicco-boot", type=int, default=50,
                        help="Number of vicco bootstrap subsets per (model, layer, subject)")
    parser.add_argument("--n-jobs", type=int, default=8,
                        help="Joblib workers for vicco bootstrap parallelism")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    subjects = parse_subject_arg(args.subject)
    print(f"Subjects: {subjects}")
    print(f"vicco bootstraps: {args.n_vicco_boot} (parallel jobs: {args.n_jobs})")

    subject_data = {}
    for s in subjects:
        d = load_subject_brain_data(s, n_vicco_boot=args.n_vicco_boot)
        if d is None:
            print(f"  {s}: no brain data, skipping")
            continue
        subject_data[s] = d
        print(f"  {s}: {d['n_hlvis']} hlvis voxels, groups: {d['available_groups']}")

    if OUT_CSV.exists():
        OUT_CSV.unlink()

    rows = []
    for model in tqdm(MODEL_LAYERS, desc="model"):
        layer_specs = MODEL_LAYERS[model]
        display = MODEL_DISPLAY_NAMES.get(model, model)

        for stim_set in STIMULUS_SETS:
            cache_p = CACHE_ROOT / model / f"{stim_set}.npz"
            if not cache_p.exists():
                print(f"[skip] {model}/{stim_set} missing cache; run 01 first")
                continue
            cached = np.load(cache_p, allow_pickle=True)

            for layer_name, _ in layer_specs:
                feats_full = cached[layer_name]  # (N_images, F)

                for subject, sdata in subject_data.items():
                    if stim_set != "vicco" and stim_set not in sdata["group_indices"]:
                        continue
                    betas = sdata["betas_hlvis"]

                    if stim_set == "vicco":
                        # Vicco: N bootstrap subsets in parallel.
                        if "vicco" not in sdata["group_indices"]:
                            continue
                        vicco_brain_idx_full = sdata["group_indices"]["vicco"]
                        vicco_file_idx = sdata["group_stim_idx"]["vicco"]
                        feats_vicco_subj = feats_full[vicco_file_idx]
                        boots = sdata["vicco_bootstrap"]
                        if args.n_jobs > 1 and len(boots) > 1:
                            res = Parallel(n_jobs=args.n_jobs, prefer="threads")(
                                delayed(_vicco_boot_one)(
                                    bi, idx, betas, vicco_brain_idx_full,
                                    feats_vicco_subj, sdata["n_vicco_sample"],
                                ) for bi, idx in enumerate(boots)
                            )
                        else:
                            res = [
                                _vicco_boot_one(bi, idx, betas, vicco_brain_idx_full,
                                                feats_vicco_subj, sdata["n_vicco_sample"])
                                for bi, idx in enumerate(boots)
                            ]
                        for r in res:
                            rows.append({
                                "subject": subject, "model": model,
                                "display_name": display,
                                "layer": layer_name,
                                "model_set": "vicco",
                                "stimulus_type": "vicco",
                                **r,
                            })
                    else:
                        cstim_brain_idx = sdata["group_indices"][stim_set]
                        cstim_file_idx = sdata["group_stim_idx"][stim_set]
                        feats_cstim = feats_full[cstim_file_idx]
                        model_rdm = compute_rdm_correlation(feats_cstim)
                        brain_rdm = compute_rdm_correlation(betas[:, cstim_brain_idx].T)
                        score = compute_rsa_score(model_rdm, brain_rdm, method="spearman")
                        rows.append({
                            "subject": subject, "model": model,
                            "display_name": display,
                            "layer": layer_name,
                            "model_set": stim_set,
                            "stimulus_type": "controversial",
                            "bootstrap_idx": 0,
                            "n_stimuli": len(cstim_brain_idx),
                            "rsa": score,
                        })

        # Save incrementally per model.
        df = pd.DataFrame(rows)
        df.to_csv(OUT_CSV, index=False)
        print(f"  saved {len(df)} rows -> {OUT_CSV}")

    print(f"\nFinal: {len(rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
