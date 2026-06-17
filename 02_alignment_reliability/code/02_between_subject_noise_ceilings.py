"""
02_between_subject_noise_ceilings.py

Compute Kriegeskorte between-subject noise ceilings for brain RDMs.

Approach (Kriegeskorte et al. 2008 / RSA toolbox):
  - For each subject i, correlate their RDM with:
      Upper bound: mean RDM of ALL subjects (including i)
      Lower bound: mean RDM of all OTHER subjects (leave-one-out)
  - For vicco (292 stimuli), subsample to N_MATCH=100 and bootstrap to match
    sample size with controversial stimuli.
  - Uses hlvis-masked voxels (consistent with RSA score computation).

Outputs:
  02_alignment_reliability/results/between_subject_noise_ceilings.csv
    columns: group, stimulus_type, subject, nc_upper, nc_lower, nc_mid

Usage:
  python 02_alignment_reliability/code/02_between_subject_noise_ceilings.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

STAGE = Path(__file__).resolve().parents[1]
SHARE_ROOT = STAGE.parent
PAPER_HELPERS = SHARE_ROOT / "src"
sys.path.insert(0, str(PAPER_HELPERS))

from cstims.paper import config

SUBJECTS = config.SUBJECTS
BRAIN_DATA_DIR = config.BRAIN_DATA_DIR
DATA_OUT = config.RELIABILITY_DATA_DIR / "between_subject_noise_ceilings.csv"

N_MATCH = 100    # match controversial stimulus count
N_BOOT = 200     # bootstrap iterations for vicco subsampling


def get_rdm_vec(betas: np.ndarray) -> np.ndarray:
    """Correlation-distance RDM upper triangle from (n_voxels, n_stim) betas."""
    normed = betas - betas.mean(axis=0, keepdims=True)
    normed /= (np.linalg.norm(normed, axis=0, keepdims=True) + 1e-12)
    corr = normed.T @ normed
    idx = np.triu_indices(corr.shape[0], k=1)
    return 1.0 - corr[idx]


def kriegeskorte_nc_per_subject(rdm_mat: np.ndarray):
    """
    rdm_mat: (n_subjects, n_pairs)
    Returns upper, lower arrays of length n_subjects.
    """
    n = rdm_mat.shape[0]
    upper, lower = np.zeros(n), np.zeros(n)
    mean_all = rdm_mat.mean(axis=0)
    rdm_sum = rdm_mat.sum(axis=0)
    for i in range(n):
        mean_loo = (rdm_sum - rdm_mat[i]) / (n - 1)
        upper[i] = spearmanr(rdm_mat[i], mean_all).statistic
        lower[i] = spearmanr(rdm_mat[i], mean_loo).statistic
    return upper, lower


def load_betas_hlvis(subject: str, stim_group: str):
    """Load hlvis-masked betas for stimuli matching stim_group."""
    d = np.load(BRAIN_DATA_DIR / subject / "cstim_betas_averaged.npz", allow_pickle=True)
    vm = np.load(BRAIN_DATA_DIR / subject / "voxel_metadata.npz", allow_pickle=True)
    hlvis = vm["hlvis_mask"]
    betas = d["betas"][hlvis].astype(np.float32)
    stim_keys = d["stim_keys"]
    mask = np.array([stim_group in k for k in stim_keys])
    return betas[:, mask]


def compute_nc(stim_group: str, rng: np.random.Generator):
    """
    Returns list of dicts with per-subject NC values.
    Subsamples to N_MATCH if n_stim > N_MATCH (bootstrapped).
    """
    all_betas = [load_betas_hlvis(s, stim_group) for s in SUBJECTS]
    n_stim = all_betas[0].shape[1]

    if n_stim <= N_MATCH:
        rdm_mat = np.stack([get_rdm_vec(b) for b in all_betas])
        upper, lower = kriegeskorte_nc_per_subject(rdm_mat)
        # shape: (n_subjects,)
        return upper, lower

    # Bootstrap: average per-subject NC over N_BOOT subsamples
    boot_upper = np.zeros((N_BOOT, len(SUBJECTS)))
    boot_lower = np.zeros((N_BOOT, len(SUBJECTS)))
    for b in range(N_BOOT):
        idx = rng.choice(n_stim, size=N_MATCH, replace=False)
        rdm_mat = np.stack([get_rdm_vec(b_[:, idx]) for b_ in all_betas])
        u, l = kriegeskorte_nc_per_subject(rdm_mat)
        boot_upper[b] = u
        boot_lower[b] = l
    return boot_upper.mean(axis=0), boot_lower.mean(axis=0)


def main():
    rng = np.random.default_rng(42)
    rows = []

    groups = list(config.MODEL_SETS.keys()) + ["vicco"]

    for group in groups:
        stim_group = "vicco" if group == "vicco" else group
        stimulus_type = "vicco" if group == "vicco" else "controversial"
        print(f"  {group} ({stimulus_type})...", end=" ", flush=True)

        upper, lower = compute_nc(stim_group, rng)

        for i, subj in enumerate(SUBJECTS):
            rows.append({
                "group": group,
                "stimulus_type": stimulus_type,
                "subject": subj,
                "nc_upper": upper[i],
                "nc_lower": lower[i],
                "nc_mid": (upper[i] + lower[i]) / 2,
            })

        n = len(SUBJECTS)
        mid = (upper + lower) / 2
        print(f"upper={upper.mean():.3f}±{upper.std(ddof=1)/np.sqrt(n):.3f}  "
              f"lower={lower.mean():.3f}±{lower.std(ddof=1)/np.sqrt(n):.3f}  "
              f"mid={mid.mean():.3f}±{mid.std(ddof=1)/np.sqrt(n):.3f}")

    df = pd.DataFrame(rows)
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_OUT, index=False)
    print(f"\nSaved: {DATA_OUT}")


if __name__ == "__main__":
    main()
