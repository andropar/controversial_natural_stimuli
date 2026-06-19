#!/usr/bin/env python3
"""
Subject consistency of residual brain RDM on all_models controversial stimuli.

Per subject (sub-01, sub-03, sub-05):
  - Load brain RDM and all encoding-predicted model RDMs
  - Ridge regression: regress brain on all model predictions
  - Compute residual vector (4950,)

Then:
  - Correlate residuals pairwise across subjects (Spearman r)
  - Identify image pairs with consistently high |residual| across all 3 subjects
  - Save per-subject residual vectors + consistency metrics

Outputs:
    results/subject_residuals.npz     - per-subject residual vectors (4950,)
    results/subject_consistency.csv   - pairwise residual correlations
    results/consistent_top_pairs.csv  - top pairs with high residual across ALL subjects
    figures/subject_consistency.pdf/png

Usage:
    python 01_subject_consistency.py
"""

import sys
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.stats import zscore, spearmanr
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt

_PAPER = Path(__file__).resolve().parents[1]
SHARE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))
from cstims import constants, paths

SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
RDM_DIR = Path("/home/jroth/rsa_based_selection/experiments/archive/simulationdiffs_to_braindiffs/data")
MANIFEST_PATH = (
    SHARE_ROOT
    / "00_stimulus_selection"
    / "decision_checks"
    / "selection_evaluation"
    / "results"
    / "all_models"
    / "images"
    / "image_manifest.csv"
)

DATA_DIR = Path(__file__).resolve().parent / "results"
FIG_DIR = Path(__file__).resolve().parent / "figures"
N_IMAGES = 100

try:
    from cstims.paper.style_improved import apply_style, DPI, W_SINGLE, W_DOUBLE
    apply_style()
except ImportError:
    DPI = 150
    W_SINGLE = 6
    W_DOUBLE = 12


def upper_tri(rdm):
    idx = np.triu_indices(rdm.shape[0], k=1)
    return rdm[idx]


def load_and_compute_residual(subject):
    """Load brain + model RDMs, run ridge regression, return residual vector."""
    d = np.load(RDM_DIR / f"rdms_{subject}.npz", allow_pickle=True)

    brain_vec = upper_tri(d["brain__all_models"])
    brain_z = zscore(brain_vec)

    model_names = sorted([k.replace("pred__all_models__", "")
                          for k in d.keys() if k.startswith("pred__all_models__")])
    X = np.column_stack([zscore(upper_tri(d[f"pred__all_models__{m}"])) for m in model_names])

    alphas = np.logspace(-2, 4, 20)
    reg = RidgeCV(alphas=alphas, fit_intercept=True)
    reg.fit(X, brain_z)

    residual = brain_z - reg.predict(X)
    r2 = reg.score(X, brain_z)

    return residual, r2, brain_z, model_names


def main():
    print("Computing per-subject residuals...")
    residuals = {}
    brain_zs = {}
    r2s = {}

    for subj in SUBJECTS:
        residual, r2, brain_z, model_names = load_and_compute_residual(subj)
        residuals[subj] = residual
        brain_zs[subj] = brain_z
        r2s[subj] = r2
        print(f"  {subj}: R²={r2:.3f}, residual std={residual.std():.3f}")

    # -------------------------------------------------------------------------
    # 1. Pairwise residual correlations across subjects
    # -------------------------------------------------------------------------
    subj_list = SUBJECTS
    n = len(subj_list)
    consistency_rows = []

    print("\nPairwise residual correlations (Spearman):")
    for i in range(n):
        for j in range(i + 1, n):
            s1, s2 = subj_list[i], subj_list[j]
            rho_res, p_res = spearmanr(residuals[s1], residuals[s2])
            rho_brain, p_brain = spearmanr(brain_zs[s1], brain_zs[s2])
            print(f"  {s1} vs {s2}: residual rho={rho_res:.3f} (p={p_res:.2e}), "
                  f"brain rho={rho_brain:.3f} (p={p_brain:.2e})")
            consistency_rows.append({
                "subject_1": s1, "subject_2": s2,
                "residual_rho": rho_res, "residual_p": p_res,
                "brain_rho": rho_brain, "brain_p": p_brain,
            })

    consistency_df = pd.DataFrame(consistency_rows)
    consistency_df.to_csv(DATA_DIR / "subject_consistency.csv", index=False)
    print("Saved: results/subject_consistency.csv")

    # -------------------------------------------------------------------------
    # 2. Top pairs with consistently high residual across ALL subjects
    # -------------------------------------------------------------------------
    idx_i, idx_j = np.triu_indices(N_IMAGES, k=1)

    manifest = pd.read_csv(MANIFEST_PATH)
    img_names = manifest["image_name"].values if "image_name" in manifest.columns \
        else [f"image_{i:04d}" for i in range(N_IMAGES)]

    pair_df = pd.DataFrame({"img_i": idx_i, "img_j": idx_j})
    abs_residuals = []
    signed_residuals = []
    for subj in SUBJECTS:
        pair_df[f"residual_{subj}"] = residuals[subj]
        abs_residuals.append(np.abs(residuals[subj]))
        signed_residuals.append(residuals[subj])

    # Mean absolute residual across subjects
    pair_df["mean_abs_residual"] = np.mean(abs_residuals, axis=0)
    pair_df["std_abs_residual"] = np.std(abs_residuals, axis=0)
    pair_df["mean_signed_residual"] = np.mean(signed_residuals, axis=0)

    # Consistency: Spearman r between subject residuals at the pair level
    # (already computed above; add per-pair sign consistency)
    # Sign agreement: fraction of subject pairs where sign matches
    signs = np.sign(np.stack(signed_residuals, axis=0))  # (n_subjects, 4950)
    pair_df["sign_consistency"] = (
        (signs[0] == signs[1]).astype(float) +
        (signs[0] == signs[2]).astype(float) +
        (signs[1] == signs[2]).astype(float)
    ) / 3.0  # 1.0 = all subjects agree on sign

    pair_df["img_i_name"] = img_names[idx_i]
    pair_df["img_j_name"] = img_names[idx_j]

    pair_df_sorted = pair_df.sort_values("mean_abs_residual", ascending=False)
    pair_df_sorted.to_csv(DATA_DIR / "consistent_top_pairs.csv", index=False)
    print(f"Saved: data/consistent_top_pairs.csv")

    print(f"\nTop 15 pairs by mean |residual| across subjects:")
    top15 = pair_df_sorted.head(15)[
        ["img_i", "img_j", "img_i_name", "img_j_name",
         "mean_abs_residual", "sign_consistency",
         f"residual_{SUBJECTS[0]}", f"residual_{SUBJECTS[1]}", f"residual_{SUBJECTS[2]}"]
    ]
    print(top15.to_string(index=False))

    # Also save per-subject residual vectors for downstream use
    np.savez(DATA_DIR / "subject_residuals.npz",
             **{f"residual_{s}": residuals[s] for s in SUBJECTS},
             **{f"brain_z_{s}": brain_zs[s] for s in SUBJECTS},
             model_names=np.array(model_names),
             image_names=img_names)
    print("\nSaved: data/subject_residuals.npz")

    # -------------------------------------------------------------------------
    # 3. Figure: residual correlations + per-image residual consistency
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, 4.5))

    # Panel a/b/c: scatter of residuals for each subject pair
    pairs = [(0, 1), (0, 2), (1, 2)]
    for ax, (i, j) in zip(axes, pairs):
        s1, s2 = SUBJECTS[i], SUBJECTS[j]
        r1, r2 = residuals[s1], residuals[s2]
        rho = consistency_df[
            (consistency_df.subject_1 == s1) & (consistency_df.subject_2 == s2)
        ]["residual_rho"].iloc[0]

        ax.scatter(r1, r2, s=2, alpha=0.3, color="#444444", rasterized=True)
        ax.set_xlabel(f"Residual {s1}", fontsize=8)
        ax.set_ylabel(f"Residual {s2}", fontsize=8)
        ax.set_title(f"Spearman ρ = {rho:.3f}", fontsize=9)

        # Reference line
        lim = max(np.abs(r1).max(), np.abs(r2).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=0.8, alpha=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    fig.suptitle("Subject consistency of residual brain RDM\n(all_models controversial stimuli)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "subject_consistency.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "subject_consistency.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/subject_consistency.pdf/png")
    print("Done.")


if __name__ == "__main__":
    main()
