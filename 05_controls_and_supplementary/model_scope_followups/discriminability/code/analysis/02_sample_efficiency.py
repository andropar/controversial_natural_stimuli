#!/usr/bin/env python3
"""Sample-efficiency curves: how many model pairs become separable vs n images.

For each n in {10, 20, 40, 60, 80, 100}:
    For each bootstrap b in {0..B-1}:
        Sample n images from the cstim_all_models set OR a random vicco
        100-subset (drawn from 292) — both at sample size n.
        Per (subject, model) compute mRSA / fRSA at this n.
        Per pair (190 pairs from 20 models): paired-t across 5 subjects.
        FDR-correct across pairs.
        Count separated pairs.

Plots:
    n_separated_pairs vs n  (cstim vs vicco; mRSA vs fRSA)
    rank_stability vs n (mean Spearman with full ranking)

Output:
    data/sample_efficiency.csv
    figures/sample_efficiency.{pdf,png}

Uses paper-layer features from 11_layer_sweep/cache/features and existing
per-(subject, model) paper-layer encoding models.
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t as student_t
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from joblib import Parallel, delayed

PROJECT = Path(__file__).resolve().parents[4]
PAPER = PROJECT / "experiments" / "cstim_paper"
LAYER_SWEEP = PAPER / "11_layer_sweep"
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(LAYER_SWEEP))

from config import MODEL_SETS, get_brain_input_dir  # noqa
from utils import load_encoding_model, predict_voxel_responses  # noqa
from layers_config import MAIN_LAYER  # noqa

DATA_DIR = Path(__file__).resolve().parents[1] / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FEAT = LAYER_SWEEP / "cache" / "features"
SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
MODELS = MODEL_SETS["all_models"]   # 20 paper models
N_SIZES = [10, 20, 40, 60, 80, 100]
N_BOOT = 50
ALPHA = 0.05


def paired_t_p(deltas):
    deltas = np.asarray(deltas)
    deltas = deltas[~np.isnan(deltas)]
    if len(deltas) < 3 or np.std(deltas) == 0:
        return np.nan
    t = np.mean(deltas) / (np.std(deltas, ddof=1) / np.sqrt(len(deltas)))
    return float(2 * student_t.sf(abs(t), df=len(deltas) - 1))


def rdm_corr(features):
    """Correlation distance RDM. features: (n, d). Returns (n, n)."""
    c = np.corrcoef(features)
    rdm = 1 - c
    np.fill_diagonal(rdm, 0)
    return rdm


def upper_tri_vec(rdm):
    n = rdm.shape[0]
    iu = np.triu_indices(n, k=1)
    return rdm[iu]


def spearman_rdm(a, b):
    return stats.spearmanr(upper_tri_vec(a), upper_tri_vec(b))[0]


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


def precompute_predictions():
    """For each (subject, model), predict voxel responses on all-models cstim
    and full vicco. Returns:
        pred[(subject, model, set)] : (n_images, n_hlvis)
        feats[(subject, model, set)] : (n_images, n_features) — for fRSA
        brain[subject][set]: (n_hlvis, n_images_for_subject)
        n_per_set[subject][set]: number of images for that subject
    """
    pred = {}
    feats = {}
    brain = {s: {} for s in SUBJECTS}

    print("Precomputing predictions and features...")
    for subject in tqdm(SUBJECTS, desc="subj"):
        betas_hlvis, g_brain, g_file = load_subject_brain(subject)
        for stim_set in ["all_models", "vicco"]:
            if stim_set not in g_brain:
                continue
            brain_idx = g_brain[stim_set]
            file_idx = g_file[stim_set]
            brain[subject][stim_set] = betas_hlvis[:, brain_idx]  # (n_hlvis, n_images)

            for model in MODELS:
                cache_p = CACHE_FEAT / model / f"{stim_set}.npz"
                if not cache_p.exists():
                    continue
                cached = np.load(cache_p, allow_pickle=True)
                paper_layer = MAIN_LAYER[model]
                feat_full = cached[paper_layer]  # (n_total_images, d)
                feat_subj = feat_full[file_idx]
                feats[(subject, model, stim_set)] = feat_subj.astype(np.float32)

                # Encoding model -> predicted voxels at hlvis voxels
                enc = load_encoding_model(model, subject)
                pred_full = predict_voxel_responses(feat_subj, enc)
                pred_hlvis = pred_full[:, enc["roi_hlvis"]]
                pred[(subject, model, stim_set)] = pred_hlvis.astype(np.float32)

    return pred, feats, brain


def compute_rsa_at_subset(per_image_pred_or_feat, brain_per_image, idx,
                          metric):
    """Compute RSA at a given subset of images. Returns scalar Spearman ρ."""
    if metric == "mRSA":
        pred_rdm = rdm_corr(per_image_pred_or_feat[idx])
    else:  # fRSA
        pred_rdm = rdm_corr(per_image_pred_or_feat[idx])
    brain_rdm = rdm_corr(brain_per_image[:, idx].T)
    return spearman_rdm(pred_rdm, brain_rdm)


def run_one_bootstrap(idx_array, set_name, pred, feats, brain, metric):
    """For a single image-subset, compute per-(subject, model) RSA, then
    paired-t per pair across subjects. Returns dict[pair] = p-value."""
    rsa_per = {}  # (subject, model) -> rsa
    for subject in SUBJECTS:
        for model in MODELS:
            key = (subject, model, set_name)
            if key not in feats:
                continue
            if metric == "mRSA":
                if key not in pred:
                    continue
                t = pred[key]
            else:
                t = feats[key]
            brain_per_image = brain[subject].get(set_name)
            if brain_per_image is None:
                continue
            rsa_per[(subject, model)] = compute_rsa_at_subset(
                t, brain_per_image, idx_array, metric)

    pair_p = {}
    for A, B in combinations(MODELS, 2):
        deltas = []
        for s in SUBJECTS:
            a = rsa_per.get((s, A))
            b = rsa_per.get((s, B))
            if a is not None and b is not None:
                deltas.append(a - b)
        if len(deltas) >= 3:
            pair_p[(A, B)] = paired_t_p(np.array(deltas))
    return pair_p, rsa_per


def n_separated_after_fdr(pair_p):
    pairs = list(pair_p.keys())
    pvals = np.array([pair_p[p] for p in pairs])
    valid = ~np.isnan(pvals)
    if valid.sum() == 0:
        return 0
    _, q, _, _ = multipletests(pvals[valid], alpha=ALPHA, method="fdr_bh")
    return int((q < ALPHA).sum())


def rank_correlation_with_full(rsa_per, full_rsa_per):
    """Per subject, Spearman ρ between subset ranking and full-set ranking."""
    rhos = []
    for subject in SUBJECTS:
        x_full = []
        x_sub = []
        for model in MODELS:
            f = full_rsa_per.get((subject, model))
            s = rsa_per.get((subject, model))
            if f is not None and s is not None:
                x_full.append(f)
                x_sub.append(s)
        if len(x_full) >= 5:
            r, _ = stats.spearmanr(x_full, x_sub)
            rhos.append(r)
    return float(np.mean(rhos)) if rhos else np.nan


def main():
    pred, feats, brain = precompute_predictions()
    n_cstim = next(iter(brain[SUBJECTS[0]].get("all_models").shape[1:2]))  # 100
    n_vicco = brain[SUBJECTS[0]]["vicco"].shape[1]  # 292
    print(f"n_cstim_all_models={n_cstim}, n_vicco={n_vicco}")

    # Full-set reference rankings (at n=full) for rank-stability metric
    full_idx_cstim = np.arange(n_cstim)
    full_idx_vicco = np.arange(n_vicco)
    full_rsa_ref = {}  # (set, metric) -> dict[(subj, model)] -> rsa
    for set_name, full_idx in [("all_models", full_idx_cstim),
                                ("vicco", full_idx_vicco)]:
        for metric in ["mRSA", "fRSA"]:
            _, rsa_per = run_one_bootstrap(full_idx, set_name, pred, feats, brain, metric)
            full_rsa_ref[(set_name, metric)] = rsa_per

    rng = np.random.default_rng(42)
    rows = []
    for n in tqdm(N_SIZES, desc="n"):
        for set_name, n_total in [("all_models", n_cstim), ("vicco", n_vicco)]:
            if n > n_total:
                continue
            # Draw N_BOOT random subsets
            for b in range(N_BOOT):
                idx = rng.choice(n_total, size=n, replace=False)
                idx.sort()
                for metric in ["mRSA", "fRSA"]:
                    pair_p, rsa_per = run_one_bootstrap(
                        idx, set_name, pred, feats, brain, metric)
                    n_sep = n_separated_after_fdr(pair_p)
                    rho_rank = rank_correlation_with_full(
                        rsa_per, full_rsa_ref[(set_name, metric)])
                    rows.append({
                        "metric": metric, "set": set_name, "n": n,
                        "boot": b, "n_separated": n_sep,
                        "rank_rho_with_full": rho_rank,
                    })

    df = pd.DataFrame(rows)
    out_path = DATA_DIR / "sample_efficiency.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows -> {out_path}")

    # Summary
    summary = (df.groupby(["metric", "set", "n"], as_index=False)
                 .agg(n_separated_mean=("n_separated", "mean"),
                      n_separated_sem=("n_separated", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
                      rank_rho_mean=("rank_rho_with_full", "mean"),
                      rank_rho_sem=("rank_rho_with_full", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
                      n_boot=("n_separated", "size")))
    summary.to_csv(DATA_DIR / "sample_efficiency_summary.csv", index=False)
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
