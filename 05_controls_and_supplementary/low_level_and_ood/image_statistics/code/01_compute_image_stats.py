#!/usr/bin/env python3
"""
Compute basic low-level image statistics for every stimulus in every stimulus
set, for comparison between controversial sets and the vicco baseline.

Output: data/image_stats.csv with one row per image.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))

from config import CSTIM_HDF5_ROOT  # noqa: E402


STIMULUS_SETS = [
    "all_models",
    "architecture",
    "dataset",
    "sota",
    "training_objective",
    "vicco",
]

OUT_CSV = Path(__file__).resolve().parent / "results" / "image_stats.csv"


def load_image_paths(group: str) -> list[Path]:
    """Stimulus folder for a group.

    Applies the architecture<->dataset folder swap to match every other script
    in the pipeline (02_rsa_scores/01_compute_crsa, 02_compute_wrsa_transfer,
    04_compute_rsa_large_benchmark). The `architecture/` and `dataset/` folders
    on disk hold the wrong-group images; the swap maps a requested group to
    the physical folder whose images were actually shown to the brain for that
    experimental condition.
    """
    folder_group = group
    if group == "architecture":
        folder_group = "dataset"
    elif group == "dataset":
        folder_group = "architecture"

    if folder_group == "vicco":
        img_dir = CSTIM_HDF5_ROOT / "shared_vicco"
    else:
        img_dir = CSTIM_HDF5_ROOT / folder_group
    files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not files:
        raise FileNotFoundError(f"No images found in {img_dir}")
    return files


def _radial_profile(power: np.ndarray) -> np.ndarray:
    h, w = power.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y, x = np.indices(power.shape)
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    tbin = np.bincount(r.ravel(), power.ravel())
    nr = np.bincount(r.ravel())
    nr[nr == 0] = 1
    return tbin / nr  # mean power per radius


def compute_stats(img: Image.Image) -> dict:
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    gray = rgb @ np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)

    # --- Luminance & contrast ----------------------------------------------
    lum_mean = float(gray.mean())
    lum_rms = float(np.sqrt(((gray - gray.mean()) ** 2).mean()))

    # --- Color stats -------------------------------------------------------
    # Colorfulness (Hasler & Suesstrunk 2003) — single scalar for chromatic vividness.
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    colorfulness = float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2)
        + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )
    # LAB chroma (perceptual distance from gray)
    try:
        from skimage.color import rgb2lab
        lab = rgb2lab(rgb)  # L in [0,100], a,b in ~[-128,128]
        lab_chroma_mean = float(np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2).mean())
    except Exception:
        lab_chroma_mean = np.nan
    # Hue entropy weighted by saturation*value (ignore near-gray pixels)
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    w = (S * V).ravel()
    h_bins = 36
    hist, _ = np.histogram(H.ravel(), bins=h_bins, range=(0, 1), weights=w)
    if hist.sum() > 0:
        hp = hist / hist.sum()
        hue_entropy = float(-(hp[hp > 0] * np.log2(hp[hp > 0])).sum())
    else:
        # Fully desaturated image — no hue information.
        hue_entropy = 0.0

    # --- Spatial frequency -------------------------------------------------
    g = gray - gray.mean()
    # windowing reduces FFT edge artifacts
    hann = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1])).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(g * hann))
    power = (F.real ** 2 + F.imag ** 2)
    prof = _radial_profile(power)
    # discard DC + very last bin, keep positive frequencies
    prof = prof[1:max(2, len(prof) - 1)]
    freqs = np.arange(1, len(prof) + 1, dtype=np.float64)
    mask = prof > 0
    if mask.sum() >= 2:
        slope, _ = np.polyfit(np.log(freqs[mask]), np.log(prof[mask]), 1)
    else:
        slope = np.nan
    sf_slope = float(slope)
    mid = len(prof) // 2
    low_e = float(prof[:mid].sum())
    high_e = float(prof[mid:].sum())
    # log10 ratio: energy is multiplicative and the raw ratio has a heavy right tail
    if low_e > 0 and high_e > 0:
        sf_high_low_ratio = float(np.log10(high_e / low_e))
    else:
        sf_high_low_ratio = np.nan

    # --- Edge density (Sobel) ----------------------------------------------
    sx = ndimage.sobel(gray, axis=0)  # vertical-gradient (rows change)
    sy = ndimage.sobel(gray, axis=1)  # horizontal-gradient (cols change)
    edge_mag = np.sqrt(sx ** 2 + sy ** 2)
    edge_mag_mean = float(edge_mag.mean())

    # Orientation anisotropy: log2 ratio of horizontal- vs vertical-gradient energy
    eh = float((sy ** 2).mean())  # horizontal-oriented structure shows in d/dx
    ev = float((sx ** 2).mean())
    orient_anisotropy = float(np.log2(eh / ev)) if (eh > 0 and ev > 0) else np.nan

    # Horizontal center-of-mass of edge energy, normalised to [-0.5, 0.5]
    col_energy = edge_mag.sum(axis=0)
    if col_energy.sum() > 0:
        cols = np.arange(edge_mag.shape[1])
        com = (col_energy * cols).sum() / col_energy.sum()
        edge_com_x = float(com / (edge_mag.shape[1] - 1) - 0.5)
    else:
        edge_com_x = np.nan

    # Left-right symmetry: Pearson between left half and horizontally-flipped right half
    w_img = gray.shape[1]
    half = w_img // 2
    left = gray[:, :half]
    right = np.fliplr(gray[:, w_img - half:])
    a, b = left.ravel(), right.ravel()
    a_c, b_c = a - a.mean(), b - b.mean()
    denom = np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum())
    symmetry_lr = float((a_c * b_c).sum() / denom) if denom > 0 else np.nan

    # --- Complexity --------------------------------------------------------
    hist, _ = np.histogram((gray * 255).astype(np.uint8), bins=256, range=(0, 256))
    p = hist[hist > 0] / hist.sum()
    entropy = float(-(p * np.log2(p)).sum())

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=75)
    jpeg_bytes = buf.tell()
    raw_bytes = rgb.shape[0] * rgb.shape[1] * 3
    jpeg_ratio = jpeg_bytes / raw_bytes

    return dict(
        lum_mean=lum_mean,
        lum_rms=lum_rms,
        colorfulness=colorfulness,
        lab_chroma_mean=lab_chroma_mean,
        hue_entropy=hue_entropy,
        sf_slope=sf_slope,
        sf_high_low_ratio=sf_high_low_ratio,
        edge_mag_mean=edge_mag_mean,
        orient_anisotropy=orient_anisotropy,
        edge_com_x=edge_com_x,
        symmetry_lr=symmetry_lr,
        entropy=entropy,
        jpeg_ratio=jpeg_ratio,
    )


def main():
    rows = []
    for group in STIMULUS_SETS:
        paths = load_image_paths(group)
        print(f"[{group}] {len(paths)} images")
        for p in tqdm(paths, desc=group):
            with Image.open(p) as im:
                stats = compute_stats(im)
            stats.update(stimulus_set=group, image=p.name)
            rows.append(stats)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
