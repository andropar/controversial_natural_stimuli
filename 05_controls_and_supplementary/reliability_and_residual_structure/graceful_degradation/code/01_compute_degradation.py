#!/usr/bin/env python3
"""
Graceful-degradation test: does the brain's identification of a best-fit model
remain stable as the brain's effective noise ceiling degrades?

For each subject × stimulus condition (all_models controversial vs vicco baseline):
  - Load 4-rep hlvis betas.
  - For k in {1, 2, 3, 4}: iterate all C(4, k) rep subsets.
    - Average over k reps -> one brain RDM.
    - Effective NC: measured per-subsample via split-half correlation of the
      k-rep RDM against the complementary (4-k)-rep RDM (for k=4, use the
      canonical 2-vs-2 split). No Spearman-Brown extrapolation.
    - Correlate brain RDM with 20 model RDMs (both fRSA and mRSA), record
      argmax = winning model for each RSA type.
  - Tabulate argmax distribution across subsets.

For baseline: each run samples a fresh random 100-image subset of the 292-image
vicco pool (stimulus-count matched to controversial); full k-loop per subsample.

Output:
  data/degradation_results.csv - one row per
    (subject, condition, rsa_type, k, rep_subset, boot_idx)
    with measured per-subsample NC, argmax, full corr vector.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))

import numpy as np
import pandas as pd
from scipy import stats

_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))

from cstims.paper import config
from cstims.paper.utils import (
    compute_rdm_correlation,
    load_encoding_model,
    predict_voxel_responses,
    rdm_to_vector,
)

ALL_MODELS = config.MODEL_SETS["all_models"]
N_VICCO_SUBSAMPLES = 10
VICCO_SEED = 0

OUT_CSV = Path(__file__).resolve().parent / "results" / "degradation_results.csv"


# ------------------------------------------------------------------
# Per-subsample measured NC (no Spearman-Brown extrapolation)
# ------------------------------------------------------------------

def measured_sample_nc(
    betas_4d: np.ndarray, rep_subset: tuple[int, ...]
) -> float:
    """Return split-half correlation between the k-rep RDM and its complement.

    betas_4d: (n_stim, n_vox, 4)
    rep_subset: subset of reps (k of them)

    For k in {1,2,3}: correlate k-rep RDM with (4-k)-rep complement RDM.
    For k = 4: correlate the canonical 2-vs-2 split (rep0,rep2 vs rep1,rep3).
    Returns Spearman correlation of upper-triangular RDM vectors.
    """
    n_reps = betas_4d.shape[2]
    k = len(rep_subset)
    if k == n_reps:
        even = betas_4d[:, :, [0, 2]].mean(axis=2)
        odd = betas_4d[:, :, [1, 3]].mean(axis=2)
    else:
        comp = tuple(r for r in range(n_reps) if r not in rep_subset)
        even = betas_4d[:, :, list(rep_subset)].mean(axis=2)
        odd = betas_4d[:, :, list(comp)].mean(axis=2)
    rdm_e = compute_rdm_correlation(even)
    rdm_o = compute_rdm_correlation(odd)
    r, _ = stats.spearmanr(rdm_to_vector(rdm_e), rdm_to_vector(rdm_o))
    return float(r)


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_subject_betas(subject: str) -> dict:
    data_dir = config.get_subject_data_dir(subject)
    vox = np.load(data_dir / "voxel_metadata.npz", allow_pickle=True)
    hlvis_mask = vox["hlvis_mask"]
    npz = np.load(data_dir / "cstim_betas_by_rep.npz", allow_pickle=True)
    betas_by_key = {k: npz[k][hlvis_mask] for k in npz.files}
    npz.close()
    stim_info = pd.read_csv(data_dir / "cstim_stimulus_info.csv")
    return {
        "betas_by_key": betas_by_key,
        "stim_info": stim_info,
        "n_hlvis": int(hlvis_mask.sum()),
    }


def stack_stim_betas(
    betas_by_key: dict, stim_info: pd.DataFrame, group: str, n_reps_required: int = 4
) -> tuple[np.ndarray, list[int]]:
    rows = stim_info[stim_info["group"] == group].sort_values("stim_idx")
    keys = rows["stim_key"].values
    stim_idx = rows["stim_idx"].values.astype(int)
    if group == "vicco":
        stim_idx = stim_idx - 1
    keep = np.array([betas_by_key[k].shape[1] >= n_reps_required for k in keys])
    n_drop = int((~keep).sum())
    if n_drop:
        print(f"    (dropped {n_drop}/{len(keys)} stimuli with <{n_reps_required} reps)")
    keys = keys[keep]
    stim_idx = stim_idx[keep]
    arr = np.stack(
        [betas_by_key[k][:, :n_reps_required] for k in keys], axis=0
    )
    return arr, stim_idx.tolist()


def load_features(group: str) -> dict[str, np.ndarray]:
    feats = {}
    for model in ALL_MODELS:
        path = config.CSTIM_FEATURE_CACHE / f"{model}.npz"
        d = np.load(path)
        feats[model] = d[group]
    return feats


def feature_rdm_vectors(
    feats: dict[str, np.ndarray], stim_idx: list[int]
) -> dict[str, np.ndarray]:
    """fRSA: RDM vectors on raw features at stim_idx."""
    out = {}
    for m, f in feats.items():
        out[m] = rdm_to_vector(compute_rdm_correlation(f[stim_idx]))
    return out


def predicted_voxel_responses_all_stimuli(
    feats: dict[str, np.ndarray], subject: str
) -> dict[str, np.ndarray]:
    """mRSA support: for each model, pass ALL cached stimuli through the encoder
    and return (n_stim, n_hlvis_voxels) predicted responses restricted to hlvis.
    We subset stim_idx later when computing RDMs per rep-subset.
    """
    out = {}
    for m in ALL_MODELS:
        enc = load_encoding_model(m, subject)
        pred = predict_voxel_responses(feats[m], enc)[:, enc["roi_hlvis"]]
        out[m] = pred.astype(np.float32)
    return out


def predicted_rdm_vectors(
    pred_by_model: dict[str, np.ndarray], stim_idx: list[int]
) -> dict[str, np.ndarray]:
    """mRSA: RDM vectors on predicted voxel responses at stim_idx."""
    out = {}
    for m, p in pred_by_model.items():
        out[m] = rdm_to_vector(compute_rdm_correlation(p[stim_idx]))
    return out


# ------------------------------------------------------------------
# Core loop
# ------------------------------------------------------------------

def process_condition(
    subject: str,
    condition: str,
    group: str,
    betas_4d: np.ndarray,
    rdm_vecs_by_rsa: dict[str, dict[str, np.ndarray]],  # {"fRSA": {model:vec}, "mRSA": ...}
    boot_idx: int,
) -> list[dict]:
    rows = []
    n_reps = betas_4d.shape[2]

    # Pre-pack per-RSA model matrices for fast batched corr
    packed = {}
    for rsa_type, rdms in rdm_vecs_by_rsa.items():
        names = list(rdms.keys())
        mat = np.stack([rdms[m] for m in names], axis=0)  # (M, n_pairs)
        packed[rsa_type] = (names, mat)

    for k in range(1, n_reps + 1):
        for subset in itertools.combinations(range(n_reps), k):
            avg = betas_4d[:, :, list(subset)].mean(axis=2)
            brain_vec = rdm_to_vector(compute_rdm_correlation(avg))
            eff_nc = measured_sample_nc(betas_4d, subset)

            base = {
                "subject": subject,
                "condition": condition,
                "group": group,
                "boot_idx": boot_idx,
                "k_reps": k,
                "rep_subset": "".join(str(x) for x in subset),
                "eff_nc": eff_nc,
            }
            for rsa_type, (names, mat) in packed.items():
                corrs = np.array([
                    stats.spearmanr(brain_vec, m_vec)[0] for m_vec in mat
                ])
                argmax_idx = int(np.nanargmax(corrs))
                row = dict(base)
                row["rsa_type"] = rsa_type
                row["argmax_model"] = names[argmax_idx]
                row["argmax_corr"] = float(corrs[argmax_idx])
                row["second_corr"] = float(np.sort(corrs)[-2])
                row["margin"] = row["argmax_corr"] - row["second_corr"]
                for m, c in zip(names, corrs):
                    row[f"corr_{m}"] = float(c)
                rows.append(row)
    return rows


def main():
    print("Loading cached features (all_models + vicco) for 20 models...")
    feats_all = load_features("all_models")
    feats_vicco = load_features("vicco")

    all_rows = []

    for subject in config.SUBJECTS:
        data_dir = config.get_subject_data_dir(subject)
        if not (data_dir / "cstim_betas_by_rep.npz").exists():
            print(f"  {subject}: no rep betas, skipping")
            continue

        print(f"\n== {subject} ==")
        subj = load_subject_betas(subject)

        # ---- precompute encoder-predicted responses (mRSA) ----
        print("  Loading per-subject encoders and predicting voxel responses...")
        pred_all_by_model = predicted_voxel_responses_all_stimuli(feats_all, subject)
        pred_vicco_by_model = predicted_voxel_responses_all_stimuli(feats_vicco, subject)
        print(f"    all_models pred: {next(iter(pred_all_by_model.values())).shape}")
        print(f"    vicco pred:      {next(iter(pred_vicco_by_model.values())).shape}")

        # ---- controversial (all_models) ----
        betas_c, stim_idx_c = stack_stim_betas(
            subj["betas_by_key"], subj["stim_info"], "all_models"
        )
        rdm_vecs_c = {
            "fRSA": feature_rdm_vectors(feats_all, stim_idx_c),
            "mRSA": predicted_rdm_vectors(pred_all_by_model, stim_idx_c),
        }
        all_rows.extend(process_condition(
            subject, "controversial", "all_models", betas_c, rdm_vecs_c, boot_idx=0
        ))
        print(f"  controversial (all_models): {betas_c.shape}")

        # ---- baseline (vicco), N subsamples of 100 of 292 ----
        betas_v_full, stim_idx_v_full = stack_stim_betas(
            subj["betas_by_key"], subj["stim_info"], "vicco"
        )
        n_vicco = betas_v_full.shape[0]
        rng = np.random.default_rng(VICCO_SEED)
        for b in range(N_VICCO_SUBSAMPLES):
            sub = np.sort(rng.choice(n_vicco, size=100, replace=False))
            betas_v = betas_v_full[sub]
            idx_v = [stim_idx_v_full[i] for i in sub]
            rdm_vecs_v = {
                "fRSA": feature_rdm_vectors(feats_vicco, idx_v),
                "mRSA": predicted_rdm_vectors(pred_vicco_by_model, idx_v),
            }
            all_rows.extend(process_condition(
                subject, "baseline", "vicco", betas_v, rdm_vecs_v, boot_idx=b
            ))
            if b == 0:
                print(f"  baseline (vicco boot0): {betas_v.shape}")

    df = pd.DataFrame(all_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows -> {OUT_CSV}")

    # ---- sanity print: summary per RSA type × condition × k ----
    print("\n--- Summary: top-model win rate (pooled across subjects, boots, subsets) ---")
    for rsa_type in ["fRSA", "mRSA"]:
        print(f"\n[{rsa_type}]")
        for cond in ["controversial", "baseline"]:
            g0 = df[(df["rsa_type"] == rsa_type) & (df["condition"] == cond)]
            print(f"  {cond}:")
            for k in sorted(g0["k_reps"].unique()):
                g = g0[g0["k_reps"] == k]
                counts = g["argmax_model"].value_counts(normalize=True)
                top = counts.index[0] if len(counts) else "n/a"
                frac = counts.iloc[0] if len(counts) else 0.0
                print(f"    k={k}  N={len(g)}  NC~{g['eff_nc'].mean():.3f}  "
                      f"top={top:35s}  freq={frac:.2f}")


if __name__ == "__main__":
    main()
