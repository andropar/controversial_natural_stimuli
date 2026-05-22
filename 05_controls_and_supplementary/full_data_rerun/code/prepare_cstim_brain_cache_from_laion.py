#!/usr/bin/env python3
"""Build cstim brain-cache files from fully preprocessed LAION-fMRI derivatives.

The output contract intentionally matches the cache consumed by the existing
RSA scripts:

    results/brain_data_cache/{subject}/cstim_betas_averaged.npz
    results/brain_data_cache/{subject}/cstim_betas_by_rep.npz
    results/brain_data_cache/{subject}/cstim_stimulus_info.csv
    results/brain_data_cache/{subject}/voxel_metadata.npz

Inputs are expected under the GLMsingle-tedana BIDS derivative layout used by
the public LAION-fMRI release.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


THIS = Path(__file__).resolve()
RERUN_ROOT = THIS.parents[1]
SHARE_ROOT = THIS.parents[3]
HELPERS = SHARE_ROOT / "shared" / "code" / "paper_helpers"
sys.path.insert(0, str(HELPERS))

from utils import correct_stimulus_label, parse_stimulus_label  # noqa: E402


SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
CSTIM_SESSIONS = ["ses-32", "ses-33", "ses-34"]
CVE_THRESHOLD = 0.2

DEFAULT_LAION_ROOT = Path("/data/home_roth/datasets/LAION-fMRI")
DEFAULT_OUT_ROOT = RERUN_ROOT / "results" / "brain_data_cache"

LAION_ROIS = {
    "EVC": ["laionEVC"],
    "ventral": ["laionventral"],
    "lateral": ["laionlateral"],
    "dorsal": ["laiondorsal"],
    "general": ["laiongeneral"],
}

FLOC_ROIS = {
    "EBA": ["EBA"],
    "FBA": ["FBA"],
    "FFA": ["FFA1", "FFA2"],
    "OFA": ["OFA"],
    "PPA": ["PPA"],
    "OPA": ["OPA"],
    "MPA": ["MPA"],
    "LOTC": ["LO1", "LO2", "TO1", "TO2", "lobjects", "vobjects"],
    "floc_body": ["EBA", "FBA"],
    "floc_face": ["FFA1", "FFA2", "OFA", "pSTSfaces", "AFP1", "AFP2"],
    "floc_place": ["PPA", "OPA", "MPA", "SPL"],
    "floc_object": ["lobjects", "vobjects"],
    "floc_lotc": ["LO1", "LO2", "TO1", "TO2", "lobjects", "vobjects"],
}


def require_imports():
    missing = []
    try:
        import nibabel  # noqa: F401
    except ImportError:
        missing.append("nibabel")
    try:
        import pandas  # noqa: F401
    except ImportError:
        missing.append("pandas")
    if missing:
        raise SystemExit(
            "Missing Python packages: "
            + ", ".join(missing)
            + ". Activate the project analysis environment or install the "
              "'full' optional dependencies from pyproject.toml."
        )


def read_table(path: Path):
    import pandas as pd

    return pd.read_csv(path, sep="\t")


def load_nii(path: Path):
    import nibabel as nib

    return nib.load(str(path))


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No match for {pattern} in {root}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple matches for {pattern} in {root}: {matches}")
    return matches[0]


def trial_label_columns(columns) -> list[str]:
    preferred = [
        "stimulus",
        "stimulus_name",
        "image_name",
        "image",
        "filename",
        "file_name",
        "label",
        "trial_type",
    ]
    cols = list(columns)
    ordered = [c for c in preferred if c in cols]
    ordered.extend([c for c in cols if c not in ordered])
    return ordered


def clean_label(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "blank"}:
        return text if text.lower() == "blank" else None
    return Path(text).name


def extract_label(row) -> str | None:
    for col in trial_label_columns(row.index):
        label = clean_label(row[col])
        if not label:
            continue
        try:
            parse_stimulus_label(correct_stimulus_label(label))
            return label
        except Exception:
            continue
    return None


def parse_beta_image(path: Path) -> np.ndarray:
    img = load_nii(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim != 4:
        raise RuntimeError(f"Expected 4D beta image, got {data.shape}: {path}")
    return data


def flatten_betas(beta_4d: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    n_trials = beta_4d.shape[3]
    flat = beta_4d.reshape((-1, n_trials))
    return flat[brain_mask.ravel(), :]


def zscore_by_voxel(betas: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.nanmean(betas, axis=1, keepdims=True)
    std = np.nanstd(betas, axis=1, keepdims=True)
    out = (betas - mean) / np.maximum(std, eps)
    return np.nan_to_num(out, copy=False).astype(np.float32)


def load_roi_label_mask(
    laion_root: Path,
    subject: str,
    labels: list[str],
    target_shape: tuple[int, ...],
) -> np.ndarray:
    roi_subject_dir = laion_root / "derivatives" / "rois" / subject
    roi = np.zeros(target_shape, dtype=bool)
    found = []
    for label in labels:
        paths = sorted(
            roi_subject_dir.glob(
                f"**/{subject}_space-T1w_res-1pt8_label-{label}_mask.nii.gz"
            )
        )
        if not paths:
            continue
        if len(paths) > 1:
            raise RuntimeError(f"Multiple ROI masks for {subject} {label}: {paths}")
        path = paths[0]
        data = np.asarray(load_nii(path).dataobj)
        if data.shape != target_shape:
            raise RuntimeError(f"ROI shape mismatch for {path}: {data.shape} vs {target_shape}")
        roi |= data > 0
        found.append(label)
    if not found:
        print(f"  warning: no ROI labels found for {subject}: {', '.join(labels)}")
    return roi


def build_roi_masks(
    laion_root: Path,
    subject: str,
    target_shape: tuple[int, ...],
    brain_mask: np.ndarray,
    visual_vol: np.ndarray,
) -> dict[str, np.ndarray]:
    roi_vols = {}
    for name, labels in {**LAION_ROIS, **FLOC_ROIS}.items():
        roi_vols[name] = load_roi_label_mask(laion_root, subject, labels, target_shape)

    roi_vols["floc_all"] = np.zeros(target_shape, dtype=bool)
    for name in FLOC_ROIS:
        roi_vols["floc_all"] |= roi_vols[name]

    roi_vols["ventral_lateral_floc"] = (
        roi_vols["ventral"] | roi_vols["lateral"] | roi_vols["floc_all"]
    )

    roi_masks = {}
    for name, roi_vol in roi_vols.items():
        roi_masks[name] = (visual_vol & roi_vol)[brain_mask]
    return roi_masks


def load_visual_mask(laion_root: Path, subject: str, target_shape: tuple[int, ...]) -> np.ndarray:
    p = (
        laion_root
        / "derivatives"
        / "glmsingle-tedana"
        / subject
        / f"{subject}_task-images_space-T1w_desc-Noiseceiling4rep_statmap.nii.gz"
    )
    if not p.exists():
        raise FileNotFoundError(f"Missing 4-rep noise ceiling map: {p}")
    data = np.asarray(load_nii(p).dataobj)
    if data.shape != target_shape:
        raise RuntimeError(f"Noise ceiling shape mismatch for {p}: {data.shape} vs {target_shape}")
    return data > CVE_THRESHOLD


def available_cstim_sessions(laion_root: Path, subject: str) -> list[str]:
    subject_root = laion_root / "derivatives" / "glmsingle-tedana" / subject
    sessions = []
    for ses in CSTIM_SESSIONS:
        func = subject_root / ses / "func"
        if (
            func.exists()
            and list(func.glob("*_desc-SingletrialBetas_trials.tsv"))
            and list(func.glob("*_stat-effect_desc-SingletrialBetas_statmap.nii.gz"))
        ):
            sessions.append(ses)
    return sessions


def process_subject(laion_root: Path, out_root: Path, subject: str, overwrite: bool):
    out_dir = out_root / subject
    done = out_dir / "cstim_betas_averaged.npz"
    if done.exists() and not overwrite:
        print(f"{subject}: cache exists, skipping ({done})")
        return

    sessions = available_cstim_sessions(laion_root, subject)
    if not sessions:
        print(f"{subject}: no cstim GLMsingle-tedana sessions found, skipping")
        return

    print(f"{subject}: sessions {', '.join(sessions)}")
    all_trials = []
    volume_shape = None
    brain_mask = None

    for ses in sessions:
        func = laion_root / "derivatives" / "glmsingle-tedana" / subject / ses / "func"
        beta_path = find_one(func, "*_stat-effect_desc-SingletrialBetas_statmap.nii.gz")
        trials_path = find_one(func, "*_desc-SingletrialBetas_trials.tsv")

        beta_4d = parse_beta_image(beta_path)
        trials = read_table(trials_path)
        if len(trials) != beta_4d.shape[3]:
            raise RuntimeError(
                f"{subject} {ses}: trials rows {len(trials)} != beta volumes {beta_4d.shape[3]}"
            )

        if volume_shape is None:
            volume_shape = beta_4d.shape[:3]
            finite_any = np.isfinite(beta_4d).any(axis=3)
            nonzero_any = np.nan_to_num(beta_4d, nan=0.0).any(axis=3)
            brain_mask = finite_any & nonzero_any
        elif beta_4d.shape[:3] != volume_shape:
            raise RuntimeError(f"{subject} {ses}: volume shape changed to {beta_4d.shape[:3]}")

        betas = zscore_by_voxel(flatten_betas(beta_4d, brain_mask))
        labels = []
        for _, row in trials.iterrows():
            labels.append(extract_label(row))

        for trial_idx, label in enumerate(labels):
            if label is None or label == "blank":
                continue
            corrected = correct_stimulus_label(label)
            group, idx = parse_stimulus_label(corrected)
            all_trials.append(
                {
                    "group": group,
                    "stim_idx": int(idx),
                    "session": ses,
                    "trial_idx": int(trial_idx),
                    "beta": betas[:, trial_idx],
                }
            )

        print(f"  {ses}: {len(all_trials)} cumulative non-blank cstim/vicco trials")

    if not all_trials:
        print(f"{subject}: no parseable cstim/vicco trials, skipping")
        return

    visual_vol = load_visual_mask(laion_root, subject, volume_shape)
    visual_mask = visual_vol[brain_mask]
    roi_masks = build_roi_masks(laion_root, subject, volume_shape, brain_mask, visual_vol)
    hlvis_mask = roi_masks["ventral_lateral_floc"]

    grouped = {}
    for trial in all_trials:
        key = (trial["group"], trial["stim_idx"])
        grouped.setdefault(key, []).append(trial["beta"])

    rows = []
    for group, idx in sorted(grouped):
        rows.append(
            {
                "group": group,
                "stim_idx": idx,
                "n_reps": len(grouped[(group, idx)]),
                "stim_key": f"{group}_{idx}",
            }
        )

    import pandas as pd

    stim_info = pd.DataFrame(rows)
    betas_averaged = np.zeros((int(brain_mask.sum()), len(stim_info)), dtype=np.float32)
    betas_by_rep = {}
    for col_idx, row in stim_info.iterrows():
        reps = np.stack(grouped[(row["group"], row["stim_idx"])], axis=1)
        betas_averaged[:, col_idx] = reps.mean(axis=1)
        betas_by_rep[row["stim_key"]] = reps

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "cstim_betas_averaged.npz",
        betas=betas_averaged,
        stim_keys=stim_info["stim_key"].values,
    )
    np.savez_compressed(out_dir / "cstim_betas_by_rep.npz", **betas_by_rep)
    stim_info.to_csv(out_dir / "cstim_stimulus_info.csv", index=False)
    roi_payload = {f"roi_{name}": mask for name, mask in roi_masks.items()}
    roi_names = np.asarray(sorted(roi_masks))
    np.savez_compressed(
        out_dir / "voxel_metadata.npz",
        brain_flat_indices=np.where(brain_mask.ravel())[0],
        visual_mask=visual_mask,
        hlvis_mask=hlvis_mask,
        volume_shape=np.asarray(volume_shape),
        roi_names=roi_names,
        roi_definition=np.asarray("noiseceiling4rep_gt_0p2_intersected_with_named_rois"),
        **roi_payload,
    )

    print(
        f"{subject}: wrote {len(stim_info)} stimuli, "
        f"{int(visual_mask.sum())} visual voxels, {int(hlvis_mask.sum())} compatibility hlvis voxels"
    )
    for roi_name in roi_names:
        print(f"  roi_{roi_name}: {int(roi_masks[str(roi_name)].sum())} voxels")


def main():
    require_imports()
    parser = argparse.ArgumentParser()
    parser.add_argument("--laion-root", type=Path, default=DEFAULT_LAION_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--subject", default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    subjects = SUBJECTS if args.subject == "all" else [args.subject]
    for subject in subjects:
        process_subject(args.laion_root, args.out_root, subject, args.overwrite)


if __name__ == "__main__":
    main()
