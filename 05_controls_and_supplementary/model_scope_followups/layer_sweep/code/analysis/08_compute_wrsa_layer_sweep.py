#!/usr/bin/env python3
"""Compute wRSA-transfer at every (subject, model, sweep-layer, set, bootstrap).

Loads:
    - Cstim/vicco features per (model, layer) from cache/features/.
    - Per-(subject, model, layer) encoding model from cache/encodings/.
    - Brain betas (hlvis) per subject.

For each (subject, model, layer, set):
    cstim row: predict_voxels(features) -> RDM, brain RDM, spearman
    vicco rows (1000 bootstraps by default, matching 02_compute_wrsa_transfer)

Output:
    results/wrsa_layer_sweep.csv
"""

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from cstims import paths
from cstims.cache import cstim_brain_cache_exists, load_cstim_brain_cache
from cstims.constants import MODEL_DISPLAY_NAMES
PAPER_ROOT = paths.paper_root()
get_brain_input_dir = paths.get_brain_input_dir
from cstims.rdm import compute_rdm_correlation, compute_rsa_score
from cstims.sampling import bootstrap_sample_indices
from cstims.subjects import parse_subject_arg
from layers_config import MODEL_LAYERS, STIMULUS_SETS, get_layer_set

CACHE_FEAT = LAYER_SWEEP_ROOT / "cache_or_heavy" / "features"
CACHE_ENC = LAYER_SWEEP_ROOT / "cache_or_heavy" / "encodings"
DATA_DIR = LAYER_SWEEP_ROOT / "results"
OUT_CSV = DATA_DIR / "wrsa_layer_sweep.csv"
ENCODING_PROTOCOL = "hydra_random_kfold_v1"


def sanitize_layer_name(layer):
    return (str(layer).replace(".", "_").replace(":", "_")
            .replace("[", "_").replace("]", "_").replace("/", "_"))


def encoding_path(subject, model, layer):
    return CACHE_ENC / subject / f"{model}.layer{sanitize_layer_name(layer)}" / "encoding_model.npz"


def load_encoding(path):
    enc = dict(np.load(path, allow_pickle=True))
    protocol = str(np.asarray(enc.get("fit_protocol", "")).item())
    if protocol != ENCODING_PROTOCOL:
        return None
    return enc


def predict(features, enc):
    """Predict with raw-space encoding weights saved by the layer-sweep fitter."""
    x = np.asarray(features, dtype=np.float32)
    weights = np.asarray(enc["weights"], dtype=np.float32)
    intercept = np.asarray(enc["intercept"], dtype=np.float32)
    pred = x @ weights + intercept
    roi = enc.get("roi_hlvis")
    if roi is not None:
        pred = pred[:, np.asarray(roi, dtype=bool)]
    return np.ascontiguousarray(pred, dtype=np.float32)


def load_subject_data(subject, n_vicco_boot):
    cache = load_cstim_brain_cache(subject)
    data = cache.as_legacy_group_dict()
    g_brain = data["group_indices"]
    g_file = data["group_stim_idx"]
    n_vicco = len(g_brain.get("vicco", []))
    n_vicco_sample = min(100, n_vicco) if n_vicco > 0 else 0
    boots = bootstrap_sample_indices(n_vicco, n_vicco_sample, n_bootstrap=n_vicco_boot, seed=0) if n_vicco > 0 else []
    return {
        "betas_hlvis": data["betas_hlvis"],
        "group_indices": g_brain,
        "group_stim_idx": g_file,
        "available_groups": data["available_groups"],
        "vicco_bootstrap": boots,
        "n_vicco_sample": n_vicco_sample,
    }


def vicco_one_boot(boot_idx, idx, betas, vicco_brain_idx_full, pred_full, n):
    vbi = vicco_brain_idx_full[idx]
    brain_rdm = compute_rdm_correlation(betas[:, vbi].T)
    pred_rdm = compute_rdm_correlation(pred_full[idx])
    return {"bootstrap_idx": boot_idx, "n_stimuli": n,
            "rsa": compute_rsa_score(pred_rdm, brain_rdm, "spearman")}


def load_existing_counts(out_csv: Path):
    if not out_csv.exists():
        return {}
    usecols = ["subject", "model", "layer", "model_set", "stimulus_type", "bootstrap_idx"]
    try:
        existing = pd.read_csv(out_csv, usecols=usecols)
    except Exception:
        return {}
    counts = (
        existing.groupby(["subject", "model", "layer", "model_set", "stimulus_type"])["bootstrap_idx"]
        .nunique()
        .to_dict()
    )
    print(f"[resume] found {len(counts)} existing scored groups in {out_csv}", flush=True)
    return counts


def group_complete(existing_counts, subject, model, layer, model_set, stimulus_type, expected_n):
    key = (subject, model, layer, model_set, stimulus_type)
    return existing_counts.get(key, 0) >= expected_n


def append_rows(out_csv: Path, rows):
    if not rows:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        out_csv,
        mode="a",
        header=not out_csv.exists(),
        index=False,
    )


def main():
    global MODEL_LAYERS
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="all")
    parser.add_argument("--layer-set", choices=["configured", "dense"], default="configured",
                        help="Layer inventory to score")
    parser.add_argument("--out-csv", default=None,
                        help="Output CSV. Defaults to wrsa_layer_sweep.csv for configured, "
                             "wrsa_dense_layer_sweep.csv for dense.")
    parser.add_argument("--n-vicco-boot", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=16)
    args = parser.parse_args()
    MODEL_LAYERS = get_layer_set(args.layer_set)
    out_csv = Path(args.out_csv) if args.out_csv else (
        OUT_CSV if args.layer_set == "configured"
        else DATA_DIR / f"wrsa_{args.layer_set}_layer_sweep.csv"
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subjects = parse_subject_arg(args.subject)
    sdata_all = {s: load_subject_data(s, args.n_vicco_boot) for s in subjects
                  if cstim_brain_cache_exists(s)}
    print("Subjects:", list(sdata_all.keys()))

    existing_counts = load_existing_counts(out_csv)
    n_stale_encodings = 0

    for model in tqdm(MODEL_LAYERS, desc="model"):
        model_rows = []
        display = MODEL_DISPLAY_NAMES.get(model, model)
        for stim_set in STIMULUS_SETS:
            cache_p = CACHE_FEAT / model / f"{stim_set}.npz"
            if not cache_p.exists():
                continue
            cached = np.load(cache_p, allow_pickle=True)

            for layer_name, _ in MODEL_LAYERS[model]:
                if layer_name not in cached.files:
                    continue
                feats_full = cached[layer_name]
                if feats_full.ndim != 2:
                    feats_full = feats_full.reshape(feats_full.shape[0], -1)

                for subject, sdata in sdata_all.items():
                    if stim_set == "vicco":
                        expected_n = len(sdata["vicco_bootstrap"])
                        if group_complete(
                            existing_counts, subject, model, layer_name,
                            "vicco", "vicco", expected_n,
                        ):
                            continue
                    else:
                        if stim_set not in sdata["group_indices"]:
                            continue
                        if group_complete(
                            existing_counts, subject, model, layer_name,
                            stim_set, "controversial", 1,
                        ):
                            continue

                    enc_p = encoding_path(subject, model, layer_name)
                    if not enc_p.exists():
                        continue
                    enc = load_encoding(enc_p)
                    if enc is None:
                        n_stale_encodings += 1
                        continue

                    # Predict voxels for all images in this set, then index per stim type.
                    pred_all = predict(feats_full, enc)  # (n_set_images, n_hlvis)
                    betas = sdata["betas_hlvis"]

                    if stim_set == "vicco":
                        vfi = sdata["group_stim_idx"]["vicco"]
                        pred_vicco = pred_all[vfi]
                        vbi_full = sdata["group_indices"]["vicco"]
                        boots = sdata["vicco_bootstrap"]
                        if args.n_jobs > 1 and len(boots) > 1:
                            res = Parallel(n_jobs=args.n_jobs, prefer="threads")(
                                delayed(vicco_one_boot)(bi, idx, betas, vbi_full,
                                                         pred_vicco, sdata["n_vicco_sample"])
                                for bi, idx in enumerate(boots))
                        else:
                            res = [vicco_one_boot(bi, idx, betas, vbi_full,
                                                  pred_vicco, sdata["n_vicco_sample"])
                                   for bi, idx in enumerate(boots)]
                        for r in res:
                            model_rows.append({
                                "subject": subject, "model": model,
                                "display_name": display,
                                "layer": layer_name, "model_set": "vicco",
                                "stimulus_type": "vicco", **r,
                            })
                    else:
                        if stim_set not in sdata["group_indices"]:
                            continue
                        cbi = sdata["group_indices"][stim_set]
                        cfi = sdata["group_stim_idx"][stim_set]
                        pred_subj = pred_all[cfi]
                        pred_rdm = compute_rdm_correlation(pred_subj)
                        brain_rdm = compute_rdm_correlation(betas[:, cbi].T)
                        rsa = compute_rsa_score(pred_rdm, brain_rdm, "spearman")
                        model_rows.append({
                            "subject": subject, "model": model,
                            "display_name": display,
                            "layer": layer_name, "model_set": stim_set,
                            "stimulus_type": "controversial",
                            "bootstrap_idx": 0,
                            "n_stimuli": len(cbi),
                            "rsa": rsa,
                        })

        append_rows(out_csv, model_rows)
        if model_rows:
            for r in model_rows:
                key = (r["subject"], r["model"], r["layer"], r["model_set"], r["stimulus_type"])
                existing_counts[key] = existing_counts.get(key, 0) + 1
            print(f"[append] {model}: {len(model_rows)} new rows -> {out_csv}", flush=True)

    if not out_csv.exists():
        print("\nNo rows written. If encodings already exist, they are likely from "
              "the old layer-sweep protocol; rerun 07_fit_encodings_layer_sweep.py.")
    print(f"\nFinal output -> {out_csv}")
    if n_stale_encodings:
        print(f"Skipped {n_stale_encodings} stale/non-hydra protocol encoding files.")


if __name__ == "__main__":
    main()
