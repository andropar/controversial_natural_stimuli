#!/usr/bin/env python3
"""
Visualize top unexplained image pairs from the residual brain RDM analysis.

Loads the per-subject residuals (from 01_subject_consistency.py) and shows:
  - Top N pairs by mean |residual| across subjects: both images side by side
  - Per-subject residual values shown for each pair
  - Coloring by sign consistency (how much subjects agree)

Requires: results/subject_residuals.npz, results/consistent_top_pairs.csv
Run 01_subject_consistency.py first.

Outputs:
    figures/top_pairs_positive.pdf/png  - pairs where brain > model consensus
    figures/top_pairs_negative.pdf/png  - pairs where brain < model consensus
    figures/top_pairs_combined.pdf/png  - top N overall, sorted by mean |residual|

Usage:
    python 02_visualize_top_pairs.py [--n_pairs 12]
"""

import sys
import argparse
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

_PAPER = Path(__file__).resolve().parents[1]
SHARE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))
from cstims.paper import config

IMAGE_DIR = (
    SHARE_ROOT
    / "00_stimulus_selection"
    / "decision_checks"
    / "selection_evaluation"
    / "results"
    / "all_models"
    / "images"
)
DATA_DIR = Path(__file__).resolve().parent / "results"
FIG_DIR = Path(__file__).resolve().parent / "figures"
SUBJECTS = ["sub-01", "sub-03", "sub-05"]

try:
    from cstims.paper.style_improved import apply_style, DPI, W_SINGLE, W_DOUBLE
    apply_style()
except ImportError:
    DPI = 150
    W_SINGLE = 6
    W_DOUBLE = 12


def load_image(idx):
    """Load image by index from IMAGE_DIR."""
    path = IMAGE_DIR / f"image_{idx:04d}.png"
    if not path.exists():
        # Try jpg
        path = IMAGE_DIR / f"image_{idx:04d}.jpg"
    return Image.open(path).convert("RGB")


def deduplicate_pairs(pairs_df, max_appearances=1):
    """Greedily select pairs so each image appears at most max_appearances times."""
    from collections import defaultdict
    counts = defaultdict(int)
    selected = []
    for _, row in pairs_df.iterrows():
        i, j = int(row["img_i"]), int(row["img_j"])
        if counts[i] < max_appearances and counts[j] < max_appearances:
            selected.append(row)
            counts[i] += 1
            counts[j] += 1
    return pd.DataFrame(selected)


def plot_pairs(pairs_df, title, out_stem, n_pairs=12, max_appearances=1):
    """
    Plot top N image pairs in a grid.
    Each row = one pair: [image_i | image_j | residual bar]
    """
    rows = min(n_pairs, len(pairs_df))
    pairs_df = deduplicate_pairs(pairs_df, max_appearances=max_appearances)
    rows = min(n_pairs, len(pairs_df))
    if rows == 0:
        print(f"  No pairs to plot for {out_stem}")
        return

    fig = plt.figure(figsize=(W_DOUBLE, rows * 1.8))
    fig.suptitle(title, fontsize=10, y=1.01)

    outer = gridspec.GridSpec(rows, 3, figure=fig,
                              width_ratios=[2, 2, 3],
                              hspace=0.15, wspace=0.05)

    for row_idx, (_, pair) in enumerate(pairs_df.iloc[:rows].iterrows()):
        i, j = int(pair["img_i"]), int(pair["img_j"])
        img_i = load_image(i)
        img_j = load_image(j)

        # Image i
        ax_i = fig.add_subplot(outer[row_idx, 0])
        ax_i.imshow(img_i)
        ax_i.axis("off")
        ax_i.set_title(f"#{i}", fontsize=7, pad=1)

        # Image j
        ax_j = fig.add_subplot(outer[row_idx, 1])
        ax_j.imshow(img_j)
        ax_j.axis("off")
        ax_j.set_title(f"#{j}", fontsize=7, pad=1)

        # Per-subject residuals + summary
        ax_bar = fig.add_subplot(outer[row_idx, 2])
        subj_vals = [pair[f"residual_{s}"] for s in SUBJECTS]
        mean_val = pair["mean_abs_residual"] * np.sign(pair["mean_signed_residual"])
        sign_cons = pair["sign_consistency"]

        colors = ["#D64541" if v > 0 else "#2980B9" for v in subj_vals]
        ax_bar.barh(SUBJECTS, subj_vals, color=colors, alpha=0.75, height=0.5)
        ax_bar.axvline(0, color="black", linewidth=0.8)
        ax_bar.set_xlim(-2.5, 2.5)
        ax_bar.tick_params(axis="y", labelsize=6)
        ax_bar.tick_params(axis="x", labelsize=6)
        ax_bar.set_title(
            f"|res|={pair['mean_abs_residual']:.2f}, sign_cons={sign_cons:.2f}",
            fontsize=7, pad=1
        )
        if row_idx == 0:
            ax_bar.set_xlabel("Residual (z)", fontsize=7)

    fig.savefig(FIG_DIR / f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{out_stem}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: figures/{out_stem}.pdf/png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pairs", type=int, default=12)
    parser.add_argument("--max_appearances", type=int, default=1,
                        help="Max times each image can appear across shown pairs (default: 1)")
    args = parser.parse_args()

    pairs = pd.read_csv(DATA_DIR / "consistent_top_pairs.csv")

    # All top pairs
    ma = args.max_appearances
    print(f"Plotting top {args.n_pairs} pairs (max {ma} appearance(s) per image)...")
    plot_pairs(
        pairs.sort_values("mean_abs_residual", ascending=False),
        f"Top {args.n_pairs} unexplained pairs (brain ≠ model consensus)\n"
        f"Red bar = brain less similar, Blue = brain more similar",
        "top_pairs_combined",
        n_pairs=args.n_pairs,
        max_appearances=ma,
    )

    pos = pairs[pairs["mean_signed_residual"] > 0].sort_values("mean_abs_residual", ascending=False)
    print(f"Plotting top positive-residual pairs (brain > models)...")
    plot_pairs(
        pos,
        f"Top pairs: brain treats as LESS SIMILAR than models predict\n"
        f"(models overestimate similarity for these pairs)",
        "top_pairs_positive",
        n_pairs=args.n_pairs,
        max_appearances=ma,
    )

    neg = pairs[pairs["mean_signed_residual"] < 0].sort_values("mean_abs_residual", ascending=False)
    print(f"Plotting top negative-residual pairs (brain < models)...")
    plot_pairs(
        neg,
        f"Top pairs: brain treats as MORE SIMILAR than models predict\n"
        f"(models underestimate similarity for these pairs)",
        "top_pairs_negative",
        n_pairs=args.n_pairs,
        max_appearances=ma,
    )

    # Print summary
    print("\n=== Summary ===")
    print(f"Total pairs: {len(pairs)}")
    print(f"Pairs with sign_consistency=1.0 (all subjects agree): "
          f"{(pairs['sign_consistency']==1.0).sum()}")
    print(f"Pairs with sign_consistency>=0.67 (2/3 subjects agree): "
          f"{(pairs['sign_consistency']>=0.67).sum()}")

    top10 = pairs.nlargest(10, "mean_abs_residual")
    print(f"\nTop 10 by mean |residual|:")
    print(top10[["img_i", "img_j", "mean_abs_residual", "sign_consistency",
                 "mean_signed_residual"]].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
