"""
01_within_subject_noise_ceilings.py

Compute split-half RDM noise ceilings (Spearman-Brown corrected) for each
controversial stimulus set and for vicco baseline.

Approach:
- Split 4 repetitions into odd (reps 1,3) / even (reps 0,2)
- Compute brain RDM from each half (average within half → RDM)
- Correlate the two half-RDMs (Spearman)
- Apply Spearman-Brown correction: r_sb = 2r / (1 + r)
- Compare noise ceilings between controversial and baseline stimuli

Inputs:
  01_brain_model_alignment/cache_or_heavy/brain_data_cache/data/{sub}/cstim_betas_by_rep.npz
  01_brain_model_alignment/cache_or_heavy/brain_data_cache/data/{sub}/cstim_stimulus_info.csv
  01_brain_model_alignment/cache_or_heavy/brain_data_cache/data/{sub}/voxel_metadata.npz

Outputs:
  02_alignment_reliability/results/rdm_noise_ceilings.csv

Usage:
  python 02_alignment_reliability/code/01_within_subject_noise_ceilings.py [--subject sub-05 | --subject all]
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

import sys

STAGE = Path(__file__).resolve().parents[1]
SHARE_ROOT = STAGE.parent
PAPER_HELPERS = SHARE_ROOT / "src"
sys.path.insert(0, str(PAPER_HELPERS))

from cstims.paper import config
from cstims.paper.utils import (
    bootstrap_sample_indices,
    compute_rdm_correlation,
    rdm_to_vector,
)


N_VICCO_SAMPLE = 100
N_VICCO_BOOTSTRAPS = 1000


def load_subject_data(subject: str):
    """Load betas_by_rep, stimulus info, and hlvis mask for a subject.

    Eagerly materialises the npz into an in-memory dict and preslices to
    the hlvis voxels, so the inner bootstrap loop does not repeatedly re-read
    the .npz file (each bootstrap touches ~100 keys, so 1000 bootstraps would
    otherwise re-read ~100k times — observed 610 GB of disk reads).
    """
    data_dir = config.get_subject_data_dir(subject)

    voxel_meta = np.load(data_dir / "voxel_metadata.npz", allow_pickle=True)
    hlvis_mask = voxel_meta["hlvis_mask"]

    _npz = np.load(data_dir / "cstim_betas_by_rep.npz", allow_pickle=True)
    # Materialise once, slice to hlvis voxels, store as a plain dict
    betas_by_rep = {k: _npz[k][hlvis_mask] for k in _npz.files}
    _npz.close()

    stim_info = pd.read_csv(data_dir / "cstim_stimulus_info.csv")

    return betas_by_rep, stim_info, hlvis_mask


def compute_split_half_rdm_nc(
    stim_keys: list[str],
    betas_by_rep: dict,
    hlvis_mask: np.ndarray,
) -> dict:
    """
    Compute split-half RDM noise ceiling for a set of stimuli.

    Split reps into odd (1,3) / even (0,2), average within half,
    compute RDM from each, correlate, and Spearman-Brown correct.

    Returns dict with noise ceiling metrics.
    """
    n_stimuli = len(stim_keys)
    n_voxels = hlvis_mask.sum()

    # Build half-averaged beta matrices: (n_stimuli, n_voxels)
    even_betas = np.zeros((n_stimuli, n_voxels))
    odd_betas = np.zeros((n_stimuli, n_voxels))

    for i, key in enumerate(stim_keys):
        reps_hlvis = betas_by_rep[key]  # (n_hlvis_voxels, n_reps) — already presliced in load_subject_data
        n_reps = reps_hlvis.shape[1]

        if n_reps < 2:
            # Can't split — use same data for both (NC will be inflated)
            even_betas[i] = reps_hlvis[:, 0]
            odd_betas[i] = reps_hlvis[:, 0]
        elif n_reps == 2:
            even_betas[i] = reps_hlvis[:, 0]
            odd_betas[i] = reps_hlvis[:, 1]
        else:
            # Standard split: even indices (0,2,...) vs odd indices (1,3,...)
            even_idx = list(range(0, n_reps, 2))
            odd_idx = list(range(1, n_reps, 2))
            even_betas[i] = reps_hlvis[:, even_idx].mean(axis=1)
            odd_betas[i] = reps_hlvis[:, odd_idx].mean(axis=1)

    # Compute RDMs from each half
    rdm_even = compute_rdm_correlation(even_betas)
    rdm_odd = compute_rdm_correlation(odd_betas)

    # Correlate the two RDMs (Spearman)
    vec_even = rdm_to_vector(rdm_even)
    vec_odd = rdm_to_vector(rdm_odd)

    r_split, p_split = stats.spearmanr(vec_even, vec_odd)

    # Spearman-Brown correction
    r_sb = 2 * r_split / (1 + r_split) if (1 + r_split) != 0 else np.nan

    # Also compute with Pearson for comparison
    r_pearson, _ = stats.pearsonr(vec_even, vec_odd)
    r_sb_pearson = 2 * r_pearson / (1 + r_pearson) if (1 + r_pearson) != 0 else np.nan

    return {
        "n_stimuli": n_stimuli,
        "n_voxels": n_voxels,
        "split_half_r_spearman": r_split,
        "noise_ceiling_spearman": r_sb,
        "split_half_r_pearson": r_pearson,
        "noise_ceiling_pearson": r_sb_pearson,
    }


def process_subject(subject: str) -> list[dict]:
    """Process all stimulus groups for a subject."""
    print(f"\nLoading data for {subject}...")
    betas_by_rep, stim_info, hlvis_mask = load_subject_data(subject)

    results = []
    groups = sorted(stim_info["group"].unique())

    for group in groups:
        if group == "vicco":
            continue  # Handle vicco separately with bootstraps

        group_info = stim_info[stim_info["group"] == group]
        stim_keys = group_info["stim_key"].tolist()

        print(f"  {group}: {len(stim_keys)} stimuli...", end=" ")
        nc = compute_split_half_rdm_nc(stim_keys, betas_by_rep, hlvis_mask)

        results.append({
            "subject": subject,
            "group": group,
            "stimulus_type": "controversial",
            "bootstrap_idx": 0,
            **nc,
        })
        print(f"NC={nc['noise_ceiling_spearman']:.3f}")

    # Vicco: bootstrap samples of 100.
    vicco_info = stim_info[stim_info["group"] == "vicco"]
    vicco_keys = vicco_info["stim_key"].tolist()
    n_vicco = len(vicco_keys)

    print(f"  vicco: {n_vicco} total stimuli, {N_VICCO_BOOTSTRAPS} bootstrap samples of {N_VICCO_SAMPLE}...")
    bootstrap_indices = bootstrap_sample_indices(n_vicco, N_VICCO_SAMPLE, N_VICCO_BOOTSTRAPS, seed=0)

    for bidx, indices in enumerate(bootstrap_indices):
        sample_keys = [vicco_keys[i] for i in indices]
        nc = compute_split_half_rdm_nc(sample_keys, betas_by_rep, hlvis_mask)

        results.append({
            "subject": subject,
            "group": "vicco",
            "stimulus_type": "vicco",
            "bootstrap_idx": bidx,
            **nc,
        })

    vicco_ncs = [r["noise_ceiling_spearman"] for r in results if r["group"] == "vicco"]
    print(f"  vicco NC: mean={np.mean(vicco_ncs):.3f}, "
          f"range=[{np.min(vicco_ncs):.3f}, {np.max(vicco_ncs):.3f}]")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute RDM noise ceilings")
    parser.add_argument("--subject", default="all",
                       help="Subject to process (sub-XX or 'all')")
    args = parser.parse_args()

    subjects = config.SUBJECTS if args.subject == "all" else [args.subject]

    all_results = []
    for subject in subjects:
        results = process_subject(subject)
        all_results.extend(results)

    # Save
    df = pd.DataFrame(all_results)
    output_path = config.RELIABILITY_DATA_DIR / "rdm_noise_ceilings.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Print summary
    print(f"\n{'='*80}")
    print("RDM NOISE CEILING SUMMARY (Spearman-Brown corrected)")
    print(f"{'='*80}")

    for subject in subjects:
        sub_df = df[df["subject"] == subject]
        print(f"\n{subject}:")

        # Controversial groups
        cstim_df = sub_df[sub_df["stimulus_type"] == "controversial"]
        for _, row in cstim_df.iterrows():
            print(f"  {row['group']:25s}: NC = {row['noise_ceiling_spearman']:.3f}")

        # Vicco
        vicco_df = sub_df[sub_df["stimulus_type"] == "vicco"]
        vicco_mean = vicco_df["noise_ceiling_spearman"].mean()
        vicco_std = vicco_df["noise_ceiling_spearman"].std()
        print(f"  {'vicco (mean±std)':25s}: NC = {vicco_mean:.3f} ± {vicco_std:.3f}")


if __name__ == "__main__":
    main()
