#!/usr/bin/env python3
"""
How well does a purely low-level image-similarity RDM explain the brain RDM,
per stimulus set?

This is the question the per-stat distributions cannot answer directly: even
if two sets have different *marginal* stat distributions, what matters for
brain alignment is whether the *similarity structure* induced by those
low-level features matches brain structure.

For each subject × stimulus set we compute:
  - pixel RDM:  correlation distance on 64×64 grayscale pixel vectors
  - stats RDM:  correlation distance on the 14-d low-level-stats vector
                (from 01_compute_image_stats.py), z-scored per stat
  - brain RDM:  hlvis beta correlation distance (same voxels as 02_rsa_scores)

RSA scores = Spearman(RDM_lowlevel, RDM_brain) on the upper triangle.
For `vicco` we subsample to 100 images ×10 bootstraps to match cstim set sizes.

Output: data/low_level_rdm_brain_alignment.csv
  columns: subject, stimulus_set, rdm_kind, bootstrap_idx, n_stimuli, rsa
"""

import sys
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))

from cstims.paper.config import CSTIM_HDF5_ROOT, SUBJECTS, get_brain_input_dir  # noqa: E402
from cstims.paper.utils import (  # noqa: E402
    compute_rdm_correlation,
    compute_rsa_score,
    bootstrap_sample_indices,
)


HERE = Path(__file__).resolve().parent
STATS_CSV = HERE / "results" / "image_stats.csv"
OUT_CSV = HERE / "results" / "low_level_rdm_brain_alignment.csv"

STIM_SETS = ["all_models", "architecture", "dataset", "sota", "training_objective", "vicco"]
PIXEL_SIZE = 64  # downsample edge for pixel RDM


def load_image_paths(group: str) -> list[Path]:
    img_dir = CSTIM_HDF5_ROOT / ("shared_vicco" if group == "vicco" else group)
    return sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))


def pixel_vectors(paths: list[Path]) -> np.ndarray:
    vecs = []
    for p in paths:
        with Image.open(p) as im:
            g = im.convert("L").resize((PIXEL_SIZE, PIXEL_SIZE), Image.BILINEAR)
            vecs.append(np.asarray(g, dtype=np.float32).ravel() / 255.0)
    return np.stack(vecs, axis=0)


def stats_vectors(df_stats: pd.DataFrame, group: str) -> np.ndarray:
    """Order preserved by sorting image filename, matching load_image_paths()."""
    sub = df_stats[df_stats["stimulus_set"] == group].copy()
    sub = sub.sort_values("image").reset_index(drop=True)
    feat_cols = [c for c in sub.columns if c not in ("stimulus_set", "image")]
    X = sub[feat_cols].to_numpy(dtype=np.float64)
    # z-score per stat so they contribute on comparable scales in correlation distance
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    return X


def load_subject_brain(subject: str):
    data_dir = get_brain_input_dir(subject)
    betas_path = data_dir / "cstim_betas_averaged.npz"
    if not betas_path.exists():
        return None
    betas = np.load(betas_path, allow_pickle=True)
    voxel = np.load(data_dir / "voxel_metadata.npz", allow_pickle=True)
    info = pd.read_csv(data_dir / "cstim_stimulus_info.csv")
    hlvis = voxel["hlvis_mask"]
    betas_hlvis = betas["betas"][hlvis, :]
    stim_keys = list(betas["stim_keys"])
    key2col = {k: i for i, k in enumerate(stim_keys)}

    out = {"betas": betas_hlvis, "groups": {}}
    for g in info["group"].unique():
        mask = info["group"] == g
        keys = info.loc[mask, "stim_key"].values
        brain_idx = np.array([key2col[k] for k in keys])
        stim_idx = info.loc[mask, "stim_idx"].values
        # vicco stim_idx is 1-based, controversial 0-based (see 01_compute_crsa)
        file_idx = stim_idx - 1 if g == "vicco" else stim_idx
        out["groups"][g] = {"brain_idx": brain_idx, "file_idx": file_idx}
    return out


def main():
    df_stats = pd.read_csv(STATS_CSV)

    # Preload low-level features per set (once)
    set_features: dict[str, dict[str, np.ndarray]] = {}
    for g in STIM_SETS:
        paths = load_image_paths(g)
        print(f"[{g}] {len(paths)} images — computing pixel + stats RDMs")
        pix = pixel_vectors(paths)
        stt = stats_vectors(df_stats, g)
        assert len(stt) == len(pix), f"{g}: stats {len(stt)} vs images {len(pix)}"
        set_features[g] = {"pixel": pix, "stats": stt}

    rows = []
    for subject in SUBJECTS:
        sd = load_subject_brain(subject)
        if sd is None:
            print(f"  {subject}: no brain data, skip")
            continue
        print(f"\n== {subject} ==")
        betas = sd["betas"]

        for g in tqdm(STIM_SETS, desc=subject):
            if g not in sd["groups"]:
                continue
            g_info = sd["groups"][g]
            brain_idx = g_info["brain_idx"]
            file_idx = g_info["file_idx"]

            pix_full = set_features[g]["pixel"]
            stt_full = set_features[g]["stats"]

            if g == "vicco":
                n_sample = min(100, len(brain_idx))
                boots = bootstrap_sample_indices(len(brain_idx), n_sample,
                                                 n_bootstrap=10, seed=0)
                for bi, sub_idx in enumerate(boots):
                    brain_rdm = compute_rdm_correlation(
                        betas[:, brain_idx[sub_idx]].T
                    )
                    pix_rdm = compute_rdm_correlation(pix_full[file_idx[sub_idx]])
                    stt_rdm = compute_rdm_correlation(stt_full[file_idx[sub_idx]])
                    for kind, rdm in [("pixel", pix_rdm), ("stats", stt_rdm)]:
                        rows.append(dict(
                            subject=subject, stimulus_set=g, rdm_kind=kind,
                            bootstrap_idx=bi, n_stimuli=n_sample,
                            rsa=compute_rsa_score(rdm, brain_rdm, method="spearman"),
                        ))
            else:
                brain_rdm = compute_rdm_correlation(betas[:, brain_idx].T)
                pix_rdm = compute_rdm_correlation(pix_full[file_idx])
                stt_rdm = compute_rdm_correlation(stt_full[file_idx])
                for kind, rdm in [("pixel", pix_rdm), ("stats", stt_rdm)]:
                    rows.append(dict(
                        subject=subject, stimulus_set=g, rdm_kind=kind,
                        bootstrap_idx=0, n_stimuli=len(brain_idx),
                        rsa=compute_rsa_score(rdm, brain_rdm, method="spearman"),
                    ))

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
