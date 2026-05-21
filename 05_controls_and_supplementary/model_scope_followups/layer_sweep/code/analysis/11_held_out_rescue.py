#!/usr/bin/env python3
"""Cross-validated (unbiased) layer rescue.

For each (subject, model, set):
    Split cstim images into halves A and B with seed=k.
    On half A only:  pick L_k = argmax_layer cstim RSA (using A-subset features
                     and brain RDM).
    On half B only:  evaluate cstim RSA at L_k AND at the paper layer.
    Rescue_k = RSA_at_L_k_on_B - RSA_at_paper_on_B.

Average across K splits to reduce noise. This is the unbiased counterpart of
04_best_layer_rescue.py / 10_mrsa_rescue_summary.py — the layer is chosen on
data disjoint from where it's evaluated.

Output:
    11_layer_sweep/data/held_out_rescue_{frsa,mrsa}.csv
"""

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from config import MODEL_DISPLAY_NAMES, PAPER_ROOT, get_brain_input_dir
from utils import (
    compute_rdm_correlation,
    compute_rsa_score,
    parse_subject_arg,
    predict_voxel_responses,
)
from layers_config import MODEL_LAYERS, MAIN_LAYER

CACHE_FEAT = LAYER_SWEEP_ROOT / "cache_or_heavy" / "features"
CACHE_ENC = LAYER_SWEEP_ROOT / "cache_or_heavy" / "encodings"
DATA_DIR = LAYER_SWEEP_ROOT / "data"
CSTIM_SETS = ["all_models", "architecture", "dataset", "sota", "training_objective"]
ENCODING_PROTOCOL = "hydra_random_kfold_v1"


def sanitize(layer):
    return (str(layer).replace(".", "_").replace(":", "_")
            .replace("[", "_").replace("]", "_").replace("/", "_"))


def load_encoding(path):
    enc = dict(np.load(path, allow_pickle=True))
    protocol = str(np.asarray(enc.get("fit_protocol", "")).item())
    if protocol != ENCODING_PROTOCOL:
        return None
    return enc


def predict(features, enc):
    """Match 02_rsa_scores/02_compute_wrsa_transfer.py prediction semantics."""
    pred = predict_voxel_responses(features, enc)
    roi = enc.get("roi_hlvis")
    if roi is not None:
        pred = pred[:, np.asarray(roi, dtype=bool)]
    return pred


def load_subject_brain(subject):
    d = get_brain_input_dir(subject)
    b = np.load(d / "cstim_betas_averaged.npz", allow_pickle=True)
    v = np.load(d / "voxel_metadata.npz", allow_pickle=True)
    si = pd.read_csv(d / "cstim_stimulus_info.csv")
    hlvis = v["hlvis_mask"]
    betas_hlvis = b["betas"][hlvis, :]
    k2i = {k: i for i, k in enumerate(b["stim_keys"])}
    g_brain, g_file = {}, {}
    for g in si["group"].unique():
        m = si["group"] == g
        g_brain[g] = np.array([k2i[k] for k in si.loc[m, "stim_key"].values])
        idx = si.loc[m, "stim_idx"].values
        g_file[g] = idx - 1 if g == "vicco" else idx
    return betas_hlvis, g_brain, g_file


def held_out_one(subject, model, mset, betas, brain_idx, file_idx,
                 features_per_layer, encodings_per_layer, n_splits, seed):
    """Returns dict with mRSA_paper_holdoutB, mRSA_best_holdoutB, fRSA_..., rescue."""
    n = len(file_idx)
    half = n // 2
    rng = np.random.default_rng(seed)
    paper_layer = MAIN_LAYER[model]
    layers = [n for n, _ in MODEL_LAYERS[model]]

    # Pre-extract per-layer features for this subject (using file_idx)
    feats = {l: features_per_layer[l][file_idx] for l in layers}
    # Pre-predict for mRSA (per layer)
    preds = {}
    for l in layers:
        if l in encodings_per_layer:
            preds[l] = predict(feats[l], encodings_per_layer[l])

    # Brain rdm — depends only on which images we pick
    # We'll recompute on the held-out half each split
    crsa_paper, crsa_best, mrsa_paper, mrsa_best = [], [], [], []
    for k in range(n_splits):
        perm = rng.permutation(n)
        A = perm[:half]
        B = perm[half:]
        # Brain RDMs on B
        b_idx_B = brain_idx[B]
        brain_rdm_B = compute_rdm_correlation(betas[:, b_idx_B].T)
        # Brain RDM on A (for layer selection)
        b_idx_A = brain_idx[A]
        brain_rdm_A = compute_rdm_correlation(betas[:, b_idx_A].T)

        # --- fRSA: choose best on A, evaluate on B
        best_l_f, best_rsa_A = None, -np.inf
        for l in layers:
            f_A = feats[l][A]
            mrdm_A = compute_rdm_correlation(f_A)
            r = compute_rsa_score(mrdm_A, brain_rdm_A, "spearman")
            if r > best_rsa_A:
                best_rsa_A = r
                best_l_f = l
        # Eval on B
        crsa_best.append(compute_rsa_score(
            compute_rdm_correlation(feats[best_l_f][B]), brain_rdm_B, "spearman"))
        crsa_paper.append(compute_rsa_score(
            compute_rdm_correlation(feats[paper_layer][B]), brain_rdm_B, "spearman"))

        # --- mRSA: choose best on A, evaluate on B (only if encodings exist)
        if all(l in preds for l in layers):
            best_l_m, best_rsa_A = None, -np.inf
            for l in layers:
                p_A = preds[l][A]
                mrdm_A = compute_rdm_correlation(p_A)
                r = compute_rsa_score(mrdm_A, brain_rdm_A, "spearman")
                if r > best_rsa_A:
                    best_rsa_A = r
                    best_l_m = l
            mrsa_best.append(compute_rsa_score(
                compute_rdm_correlation(preds[best_l_m][B]), brain_rdm_B, "spearman"))
            mrsa_paper.append(compute_rsa_score(
                compute_rdm_correlation(preds[paper_layer][B]), brain_rdm_B, "spearman"))

    out = {
        "subject": subject, "model": model, "model_set": mset,
        "display_name": MODEL_DISPLAY_NAMES.get(model, model),
        "n_splits": n_splits,
        "frsa_paper_mean": float(np.mean(crsa_paper)),
        "frsa_best_mean": float(np.mean(crsa_best)),
        "frsa_rescue_mean": float(np.mean(np.array(crsa_best) - np.array(crsa_paper))),
        "frsa_rescue_sem": float(np.std(np.array(crsa_best) - np.array(crsa_paper), ddof=1) / np.sqrt(n_splits)),
    }
    if mrsa_best:
        out.update({
            "mrsa_paper_mean": float(np.mean(mrsa_paper)),
            "mrsa_best_mean": float(np.mean(mrsa_best)),
            "mrsa_rescue_mean": float(np.mean(np.array(mrsa_best) - np.array(mrsa_paper))),
            "mrsa_rescue_sem": float(np.std(np.array(mrsa_best) - np.array(mrsa_paper), ddof=1) / np.sqrt(n_splits)),
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="all")
    parser.add_argument("--n-splits", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()

    subjects = parse_subject_arg(args.subject)
    rows = []
    for subject in tqdm(subjects, desc="subj"):
        betas, g_brain, g_file = load_subject_brain(subject)
        for model in MODEL_LAYERS:
            # Load encodings per layer for this (subject, model) once
            enc_per_layer = {}
            for l, _ in MODEL_LAYERS[model]:
                p = CACHE_ENC / subject / f"{model}.layer{sanitize(l)}" / "encoding_model.npz"
                if p.exists():
                    enc = load_encoding(p)
                    if enc is not None:
                        enc_per_layer[l] = enc

            for mset in CSTIM_SETS:
                if mset not in g_brain:
                    continue
                cache_p = CACHE_FEAT / model / f"{mset}.npz"
                if not cache_p.exists():
                    continue
                cached = np.load(cache_p, allow_pickle=True)
                feats_full = {l: cached[l] for l, _ in MODEL_LAYERS[model]}
                rows.append(held_out_one(
                    subject, model, mset,
                    betas, g_brain[mset], g_file[mset],
                    feats_full, enc_per_layer,
                    args.n_splits, args.seed,
                ))

    df = pd.DataFrame(rows)
    out_csv = DATA_DIR / "held_out_rescue.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows -> {out_csv}")
    print()
    print("Set-level held-out rescue (mean ± SEM across subjects × models):")
    for col in ["frsa_rescue_mean", "mrsa_rescue_mean"]:
        if col in df.columns:
            label = col.replace("_rescue_mean", "")
            print(f"\n{label}:")
            agg = (df.groupby("model_set")[col]
                     .agg(["mean", lambda s: s.std(ddof=1) / np.sqrt(len(s)), "size"])
                     .rename(columns={"<lambda_0>": "sem"}))
            print(agg.to_string())


if __name__ == "__main__":
    main()
