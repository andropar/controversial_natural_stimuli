#!/usr/bin/env python3
"""Create 10x10 stimulus grids for each model set (for appendix)."""

import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
_SHARE_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))  # project root
sys.path.insert(0, str(_SHARE_ROOT / "shared" / "code" / "paper_helpers"))
import config

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSTIM_DIR = config.CSTIM_HDF5_ROOT
FIG_DIR = Path(__file__).resolve().parent
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SETS = ["sota", "architecture", "dataset", "training_objective"]

MODEL_SET_LABELS = {
    "sota": "SOTA",
    "architecture": "Architecture",
    "dataset": "Dataset",
    "training_objective": "Training Objective",
}


def load_image(path, size=80):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img)


def create_grid(model_set):
    """Create a 10x10 grid of 100 controversial stimuli for one model set."""
    img_dir = CSTIM_DIR / model_set
    files = sorted(img_dir.glob("image_*.png"))[:100]
    print(f"  {model_set}: {len(files)} images")

    thumb = 80
    cols, rows = 10, 10
    gap = 2
    images = [load_image(f, size=thumb) for f in files]

    grid = np.ones((rows * thumb + (rows - 1) * gap,
                    cols * thumb + (cols - 1) * gap, 3), dtype=np.uint8) * 240
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        y = r * (thumb + gap)
        x = c * (thumb + gap)
        grid[y:y+thumb, x:x+thumb] = img

    return grid


def main():
    fig, axes = plt.subplots(2, 2, figsize=(16, 16.5))
    axes = axes.flatten()

    for ax, model_set in zip(axes, MODEL_SETS):
        grid = create_grid(model_set)
        ax.imshow(grid)
        ax.set_title(f"{MODEL_SET_LABELS[model_set]} set (N=100)",
                     fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout(pad=1.5)
    for ext in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"stimulus_grids_by_model_set.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved stimulus_grids_by_model_set")


if __name__ == "__main__":
    main()
