"""Plotting helpers used by the selection-evaluation scripts."""

import math

import matplotlib.pyplot as plt
import numpy as np


def plot_image_grid(images, save_path, subtitles=None):
    num_images = len(images)
    grid_cols = math.ceil(math.sqrt(num_images))
    grid_rows = math.ceil(num_images / grid_cols)

    fig, axes = plt.subplots(
        grid_rows, grid_cols, figsize=(grid_cols * 2.5, grid_rows * 2.5)
    )
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, img in enumerate(images):
        axes[idx].imshow(img)
        if subtitles is not None:
            axes[idx].set_title(subtitles[idx])
        axes[idx].axis("off")

    for ax in axes[num_images:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
