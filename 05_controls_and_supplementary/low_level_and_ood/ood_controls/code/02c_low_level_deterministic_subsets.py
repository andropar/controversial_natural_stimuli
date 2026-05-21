#!/usr/bin/env python3
"""
Direct test of the original paper claim's low-level half:
"the cstim alignment drop is not (just) due to low-level visual properties."

The vicco bootstrap pool used by 02 cannot reach cstim's mean low-level
distance — bootstrap means cluster narrowly around vicco's overall mean.
This script constructs DETERMINISTIC vicco subsets that DO span (or approach)
cstim's low-level distance, and computes wRSA on them.

Method:
  1. Load full vicco brain RDM (292×292) and full vicco predicted RDM per
     (model × subject) — predicted = encoding-model voxel-space prediction
     from the cstim feature cache's vicco features.
  2. Define vicco subsets of 100 images each:
       - 'top100':       the 100 highest-low-level vicco images (max possible
                          mean low-level distance for n=100)
       - 'bottom100':    the 100 lowest-low-level vicco images
       - 'middle100':    the 100 vicco images centred on the median
       - 'match_<set>':  for each cstim set, a 100-vicco subset whose mean
                          low-level distance is as close as possible to the
                          cstim set's mean (greedy nearest-image construction)
  3. wRSA per (subject, model, subset) = Spearman correlation of the
     upper-triangular sub-blocks of brain_rdm and pred_rdm at the subset's
     image indices.

Output: data/wrsa_low_level_subsets.csv
        columns: subject, model, subset, n_imgs, mean_low, wrsa,
                 cstim_mean_low (for reference: per cstim set ref).

If the cstim drop is just low-level distribution shift, then
'top100' wRSA should approach cstim wRSA. If 'top100' wRSA is well above
cstim wRSA, the drop persists at matched-or-higher low-level distance —
meaning low-level shift alone does not explain the drop.
"""

import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

import config
from utils import (
    load_encoding_model, predict_voxel_responses,
    compute_rdm_correlation, rdm_to_vector,
)

OOD_DIR    = config.OOD_DATA_DIR
PER_IMG    = OOD_DIR / "low_level_robustness_per_image_distances.csv"
OUT_PATH   = OOD_DIR / "wrsa_low_level_subsets.csv"
SUM_PATH   = OOD_DIR / "wrsa_low_level_subsets_summary.csv"

CSTIM_SETS = ["all_models", "architecture", "training_objective", "sota", "dataset"]
ALL_MODELS = config.MODEL_SETS["all_models"]


# --------------------------------------------------------------------------------
# Subset construction
# --------------------------------------------------------------------------------

def build_subsets(per_image_low: np.ndarray, cstim_per_image_per_set: dict,
                  n=100) -> dict:
    """Return {subset_name: (indices, mean_low)}. indices are file_idx into 0..291.

    Subsets:
      bottom100 / top100 / middle100 — extremes and centre of vicco
      match_<set>      — vicco subset whose MEAN low-level distance matches cstim set
                          (greedy nearest-image, narrow spread)
      dist_match_<set> — vicco subset whose FULL DISTRIBUTION (mean + spread + shape)
                          matches cstim set (greedy 1-to-1 quantile pairing)
    """
    cstim_mean_per_set = {s: float(np.mean(d)) for s, d in cstim_per_image_per_set.items()}

    order_by_low = np.argsort(per_image_low)        # ascending
    descending   = order_by_low[::-1]               # descending
    subsets = {}
    subsets["bottom100"] = order_by_low[:n].copy()
    subsets["top100"]    = descending[:n].copy()
    middle_start = (len(per_image_low) - n) // 2
    middle = order_by_low[middle_start:middle_start + n].copy()
    subsets["middle100"] = middle

    # Mean-match: pick n vicco images closest to the target mean. Hits the mean
    # but produces a tight band with much smaller spread than cstim.
    for stim_set, target in cstim_mean_per_set.items():
        diffs = np.abs(per_image_low - target)
        order = np.argsort(diffs)
        candidate = order[:n]
        subsets[f"match_{stim_set}"] = candidate

    # Distribution-shape match: greedy 1-to-1 pairing of cstim quantiles to vicco
    # images. Sort cstim distances ascending; for each cstim value pick the
    # closest unused vicco image. This matches mean, spread, and shape.
    for stim_set, cstim_d in cstim_per_image_per_set.items():
        cstim_sorted = np.sort(np.asarray(cstim_d, float))
        used = np.zeros(len(per_image_low), bool)
        selected = []
        for target in cstim_sorted:
            d = np.abs(per_image_low - target)
            d[used] = np.inf
            i = int(d.argmin())
            used[i] = True
            selected.append(i)
        subsets[f"dist_match_{stim_set}"] = np.array(selected)

    info = {}
    for name, idx in subsets.items():
        info[name] = (idx, float(per_image_low[idx].mean()))
    return info


# --------------------------------------------------------------------------------
# Per-subject brain data
# --------------------------------------------------------------------------------

def load_subject_brain_vicco(subject: str):
    data_dir = config.get_brain_input_dir(subject)
    betas_data = np.load(data_dir / "cstim_betas_averaged.npz", allow_pickle=True)
    voxel_data = np.load(data_dir / "voxel_metadata.npz", allow_pickle=True)
    stim_info = pd.read_csv(data_dir / "cstim_stimulus_info.csv")

    hlvis_mask = voxel_data["hlvis_mask"]
    betas_hlvis = betas_data["betas"][hlvis_mask, :]
    stim_keys = betas_data["stim_keys"]
    stim_key_to_idx = {k: i for i, k in enumerate(stim_keys)}

    vicco = stim_info[stim_info["group"] == "vicco"].sort_values("stim_idx").reset_index(drop=True)
    file_idx = vicco["stim_idx"].astype(int).values - 1   # 0..291
    keys = vicco["stim_key"].values
    brain_idx = np.array([stim_key_to_idx[k] for k in keys])

    return {
        "betas_hlvis": betas_hlvis,
        "vicco_file_idx": file_idx,        # ordered by file index 0..291
        "vicco_brain_idx": brain_idx,      # for indexing betas
    }


# --------------------------------------------------------------------------------
# Full vicco RDM for one (model, subject) — brain and predicted
# --------------------------------------------------------------------------------

def full_vicco_rdms(model: str, subject: str, brain) -> dict | None:
    feat_path = config.CSTIM_FEATURE_CACHE / f"{model}.npz"
    if not feat_path.exists():
        return None
    feats = np.load(feat_path)
    if "vicco" not in feats:
        return None
    X_full = feats["vicco"].astype(np.float32)         # (292, d), file_idx ordered
    if X_full.shape[0] != 292:
        return None

    # Reorder by subject's vicco_file_idx (in case stim_info ordering differs)
    X_subj = X_full[brain["vicco_file_idx"]]            # (292, d)
    try:
        enc = load_encoding_model(model, subject)
    except FileNotFoundError:
        return None
    pred = predict_voxel_responses(X_subj, enc)
    pred_hlvis = pred[:, enc["roi_hlvis"]]
    pred_rdm = compute_rdm_correlation(pred_hlvis)        # (292, 292)

    brain_betas = brain["betas_hlvis"][:, brain["vicco_brain_idx"]]   # (n_voxels, 292)
    brain_rdm = compute_rdm_correlation(brain_betas.T)                 # (292, 292)
    return {"brain_rdm": brain_rdm, "pred_rdm": pred_rdm}


def wrsa_on_subset(rdms: dict, subset_idx: np.ndarray) -> float:
    """Spearman correlation of upper-triangular sub-block."""
    b = rdms["brain_rdm"][np.ix_(subset_idx, subset_idx)]
    p = rdms["pred_rdm"][np.ix_(subset_idx, subset_idx)]
    return float(stats.spearmanr(rdm_to_vector(b), rdm_to_vector(p))[0])


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

def main():
    # 1. Per-image low-level distances for vicco (ordered by file_idx 0..291)
    pi = pd.read_csv(PER_IMG)
    vicco_pi = pi[pi["stim_set"] == "vicco"].sort_values("image_idx").reset_index(drop=True)
    vicco_low = vicco_pi["mahal_distance"].values
    if len(vicco_low) != 292:
        raise RuntimeError(f"Expected 292 vicco distances, got {len(vicco_low)}")

    # cstim per-image distances per set (for both mean-match and dist-shape-match)
    cstim_per_image = {}
    for s in CSTIM_SETS:
        sub = pi[pi["stim_set"] == s].sort_values("image_idx")
        cstim_per_image[s] = sub["mahal_distance"].to_numpy()
    cstim_mean = {s: float(np.mean(d)) for s, d in cstim_per_image.items()}
    print("cstim per-set distance summary (mean, std):")
    for s in CSTIM_SETS:
        d = cstim_per_image[s]
        print(f"  {s:<22s} mean={d.mean():.3f}  std={d.std(ddof=1):.3f}")
    print(f"vicco mean_low = {vicco_low.mean():.3f}, std = {vicco_low.std(ddof=1):.3f}, "
          f"max = {vicco_low.max():.3f}, min = {vicco_low.min():.3f}")

    subsets = build_subsets(vicco_low, cstim_per_image, n=100)
    print("\nSubset means / std:")
    for name, (idx, m) in subsets.items():
        sd = float(vicco_low[idx].std(ddof=1))
        print(f"  {name:<25s} mean_low = {m:.3f}  std = {sd:.3f}")

    # 2. Per (model, subject), compute full vicco RDMs and then wRSA per subset
    rows = []
    print("\nComputing full vicco RDMs and per-subset wRSA...")
    subj_brain = {s: load_subject_brain_vicco(s) for s in config.SUBJECTS}
    for model in tqdm(ALL_MODELS, desc="Models"):
        for subject in config.SUBJECTS:
            brain = subj_brain[subject]
            rdms = full_vicco_rdms(model, subject, brain)
            if rdms is None:
                continue
            for name, (idx, m) in subsets.items():
                w = wrsa_on_subset(rdms, idx)
                if name.startswith("dist_match_"):
                    target_set = name[len("dist_match_"):]
                elif name.startswith("match_"):
                    target_set = name[len("match_"):]
                else:
                    target_set = None
                rows.append({
                    "subject":         subject,
                    "model":           model,
                    "subset":          name,
                    "n_imgs":          len(idx),
                    "mean_low":        m,
                    "wrsa":            w,
                    "target_set":      target_set,
                    "target_low":      cstim_mean.get(target_set, np.nan)
                                        if target_set is not None else np.nan,
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df):,} rows → {OUT_PATH}")

    # 3. Summary: subject- and model-mean per subset
    summary = (df.groupby("subset")
               [["mean_low", "wrsa"]]
               .agg(["mean", "std", "count"]).round(4))
    print("\nPer-subset summary (across subject × model):")
    print(summary.to_string())
    summary.to_csv(SUM_PATH)
    print(f"Saved → {SUM_PATH}")

    # 4. Comparison: cstim wRSA per (subject, model, model_set) vs subset wRSA
    print("\nComparing to cstim wRSA from 02_rsa_scores...")
    wrsa_dfs = []
    for s in config.SUBJECTS:
        p = config.RSA_DATA_DIR / s / "wrsa_transfer_scores.csv"
        if p.exists():
            wrsa_dfs.append(pd.read_csv(p))
    cstim = pd.concat(wrsa_dfs, ignore_index=True)
    cstim = cstim[cstim["stimulus_type"] == "controversial"]
    cstim_summary = (cstim.groupby(["model_set", "model"])
                     ["wrsa_transfer"]
                     .mean().reset_index()
                     .rename(columns={"wrsa_transfer": "cstim_wrsa_mean"}))
    print("cstim_wrsa subject-mean per (model_set, model):")
    print(cstim_summary.head(10).to_string(index=False))

    # For each subset, average across (subject, model) within each cstim model_set
    # by joining each (model_set, model) cstim row with subset rows that include
    # that model.
    comp_rows = []
    for stim_set in CSTIM_SETS + ["all_models_overall"]:
        if stim_set == "all_models_overall":
            models = ALL_MODELS
            cs = cstim[cstim["model_set"] == "all_models"]
        else:
            models = config.MODEL_SETS[stim_set]
            cs = cstim[cstim["model_set"] == stim_set]
        cs_mean_wrsa = cs["wrsa_transfer"].mean()
        sub_df = df[df["model"].isin(models)]
        per_subset = (sub_df.groupby("subset")
                      [["mean_low", "wrsa"]]
                      .mean().reset_index())
        per_subset["cstim_wrsa_mean"] = cs_mean_wrsa
        per_subset["cstim_mean_low"] = cstim_mean.get(stim_set, np.nan)
        per_subset["model_set"] = stim_set
        per_subset["drop_vs_cstim"] = per_subset["cstim_wrsa_mean"] - per_subset["wrsa"]
        comp_rows.append(per_subset)
    comp = pd.concat(comp_rows, ignore_index=True)

    print("\nComparison: subset_wrsa vs cstim_wrsa, per model_set.")
    print("If 'top100' subset_wrsa ≈ cstim_wrsa, low-level shift could explain the drop.")
    print("If 'top100' subset_wrsa >> cstim_wrsa, the drop persists at higher low-level → not explained by low-level shift.\n")
    pivot = comp.pivot_table(index="model_set", columns="subset",
                             values="wrsa", aggfunc="mean").round(3)
    print(pivot.to_string())
    print("\ncstim_wrsa per model_set:")
    print(comp.groupby("model_set")["cstim_wrsa_mean"].first().round(3).to_string())
    print("\ncstim_mean_low per model_set:")
    print(comp.groupby("model_set")["cstim_mean_low"].first().round(3).to_string())

    comp.to_csv(OOD_DIR / "wrsa_low_level_subsets_comparison.csv", index=False)
    print(f"\nSaved → {OOD_DIR / 'wrsa_low_level_subsets_comparison.csv'}")


if __name__ == "__main__":
    main()
