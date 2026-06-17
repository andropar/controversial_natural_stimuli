#!/usr/bin/env python3
"""
Load and cache brain data from cstim sessions for one or more subjects.

Usage:
    python 01_load_brain_data.py                  # All subjects
    python 01_load_brain_data.py --subject sub-05  # Single subject

Outputs per subject (in data/{subject}/):
    cstim_betas_averaged.npz - Averaged betas (n_voxels, n_stimuli)
    cstim_betas_by_rep.npz - Per-rep betas for split-half analysis
    cstim_stimulus_info.csv - Stimulus metadata (group, index, etc.)
    voxel_metadata.npz - Voxel indices and ROI masks
"""

import argparse
import sys
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))

# Setup imports from cstim_paper root
_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import zscore
from tqdm import tqdm

from cstims.paper.config import DEEPVISION_ROOT, INPUT_SOURCE, get_brain_input_dir
from cstims.paper.utils import (
    correct_stimulus_label,
    detect_available_sessions,
    parse_stimulus_label,
    parse_subject_arg,
)

CVE_THRESHOLD = 0.2


def get_brain_mask(subject: str) -> np.ndarray:
    """Load brain mask from a session in the subject directory."""
    deriv_root = DEEPVISION_ROOT / "derivatives/functional/1sTR_1pt5mm"
    session_dir = deriv_root / subject / "ses-01"
    mask_files = list(session_dir.glob("*_bold_final_mask.nii.gz"))
    if not mask_files:
        raise FileNotFoundError(f"No brain mask found in {session_dir}")
    return nib.load(str(mask_files[0])).get_fdata().astype(bool)


def load_session_data(subject: str, session: str, n_vox_expected: int) -> tuple:
    """
    Load betas and trial info for a session.

    Returns:
        labels: List of trial labels
        betas: (n_brain_voxels, n_trials) array, z-scored
    """
    glms_root = DEEPVISION_ROOT / "derivatives/functional/1sTR_1pt5mm/glmsingle" / INPUT_SOURCE / subject
    session_dir = glms_root / session
    h5_path = session_dir / "TYPED_FITHRF_GLMDENOISE_RR.hdf5"
    tsv_path = session_dir / "trial_info.tsv"

    if not h5_path.exists() or not tsv_path.exists():
        raise FileNotFoundError(f"Missing files in {session_dir}")

    # Load trial info
    trial_info = pd.read_csv(tsv_path, sep="\t")
    labels = trial_info["label"].tolist()

    # Load betas - already stored as (n_brain_voxels, n_trials)
    with h5py.File(h5_path, "r") as f:
        betas = np.asarray(f["betasmd"]).squeeze().astype(np.float32)

    if betas.shape[0] != n_vox_expected:
        raise ValueError(f"Beta shape mismatch: {betas.shape[0]} vs {n_vox_expected}")

    # Z-score per session (across trials, per voxel)
    betas_z = zscore(betas, axis=1, nan_policy="omit")
    betas_z = np.nan_to_num(betas_z, copy=False)

    return labels, betas_z


def process_subject(subject: str):
    """Load and save brain data for a single subject."""
    data_dir = get_brain_input_dir(subject)
    data_dir.mkdir(parents=True, exist_ok=True)

    sessions = detect_available_sessions(subject)
    if not sessions:
        print(f"  No cstim sessions found for {subject}, skipping.")
        return

    print(f"\nProcessing {subject}")
    print(f"  Sessions: {sessions}")

    # Get brain mask
    print("  Loading brain mask...")
    brain_mask = get_brain_mask(subject)
    n_vox_brain = int(brain_mask.sum())
    volume_shape = brain_mask.shape
    print(f"  Brain mask: {n_vox_brain} voxels, shape {volume_shape}")

    # Load CVE and hlvis masks
    atlas_root = DEEPVISION_ROOT / "derivatives/functional/1sTR_1pt5mm/atlas"
    cve_path = atlas_root / f"cross_validated_effect_{subject}.nii.gz"
    hlvis_path = atlas_root / subject / "hlvis_p20_mask_func.nii.gz"

    print("  Loading ROI masks...")
    cve_vol = nib.load(str(cve_path)).get_fdata()
    cve_vec = cve_vol[brain_mask]
    visual_mask = cve_vec > CVE_THRESHOLD
    print(f"  Visual (CVE > {CVE_THRESHOLD}): {visual_mask.sum()} voxels")

    hlvis_vol = nib.load(str(hlvis_path)).get_fdata()
    hlvis_vec = hlvis_vol[brain_mask] > 0
    hlvis_mask = visual_mask & hlvis_vec
    print(f"  hlvis (visual & hlvis ROI): {hlvis_mask.sum()} voxels")

    # Collect all trials across sessions
    print("  Loading session data...")
    all_trials = []

    for session in tqdm(sessions, desc=f"  {subject} sessions"):
        labels, betas = load_session_data(subject, session, n_vox_brain)
        print(f"    {session}: {len(labels)} trials, {betas.shape[0]} voxels")

        for trial_idx, label in enumerate(labels):
            if label == "blank":
                continue
            all_trials.append({
                "label": label,
                "session": session,
                "trial_idx": trial_idx,
                "beta": betas[:, trial_idx],
            })

    print(f"  Total non-blank trials: {len(all_trials)}")

    # Parse labels and group by stimulus
    stimulus_trials = {}

    for trial in all_trials:
        corrected_label = correct_stimulus_label(trial["label"])
        group, idx = parse_stimulus_label(corrected_label)
        key = (group, idx)
        if key not in stimulus_trials:
            stimulus_trials[key] = []
        stimulus_trials[key].append(trial["beta"])

    # Build stimulus info dataframe
    stim_info_rows = []
    for (group, idx), betas_list in sorted(stimulus_trials.items()):
        stim_info_rows.append({
            "group": group,
            "stim_idx": idx,
            "n_reps": len(betas_list),
            "stim_key": f"{group}_{idx}",
        })
    stim_info = pd.DataFrame(stim_info_rows)
    print("  Stimuli by group:")
    for group, count in stim_info.groupby("group").size().items():
        rep_stats = stim_info[stim_info["group"] == group]["n_reps"]
        print(f"    {group}: {count} stimuli, reps: {rep_stats.min()}-{rep_stats.max()} (mean {rep_stats.mean():.1f})")

    # Create averaged betas matrix
    n_stimuli = len(stim_info)
    betas_averaged = np.zeros((n_vox_brain, n_stimuli), dtype=np.float32)

    for i, row in stim_info.iterrows():
        key = (row["group"], row["stim_idx"])
        betas_list = stimulus_trials[key]
        betas_averaged[:, i] = np.mean(betas_list, axis=0)

    print(f"  Averaged betas shape: {betas_averaged.shape}")

    # Create per-rep betas for split-half analysis
    betas_by_rep = {}
    for i, row in stim_info.iterrows():
        key = (row["group"], row["stim_idx"])
        betas_list = stimulus_trials[key]
        betas_by_rep[row["stim_key"]] = np.stack(betas_list, axis=1)

    # Save outputs
    print("  Saving outputs...")

    np.savez_compressed(
        data_dir / "cstim_betas_averaged.npz",
        betas=betas_averaged,
        stim_keys=stim_info["stim_key"].values,
    )

    np.savez_compressed(
        data_dir / "cstim_betas_by_rep.npz",
        **{k: v for k, v in betas_by_rep.items()},
    )

    stim_info.to_csv(data_dir / "cstim_stimulus_info.csv", index=False)

    brain_flat_indices = np.where(brain_mask.ravel())[0]
    np.savez_compressed(
        data_dir / "voxel_metadata.npz",
        brain_flat_indices=brain_flat_indices,
        visual_mask=visual_mask,
        hlvis_mask=hlvis_mask,
        volume_shape=volume_shape,
    )

    print(f"  Done! Outputs saved to {data_dir}")


def main():
    parser = argparse.ArgumentParser(description="Load cstim brain data")
    parser.add_argument("--subject", default="all",
                        help="Subject ID (e.g. sub-05) or 'all' (default: all)")
    args = parser.parse_args()

    subjects = parse_subject_arg(args.subject)
    print(f"Processing subjects: {subjects}")

    for subject in subjects:
        process_subject(subject)

    print("\nAll done!")


if __name__ == "__main__":
    main()
