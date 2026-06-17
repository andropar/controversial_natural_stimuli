#!/usr/bin/env python3
"""
Improved OOD-vs-alignment figure.

Fixes vs. original:
- Drops the four n=5/6 subgroup panels (Spearman ρ on n=5 is uninterpretable).
- Keeps only the n=20 'all_models' panel — the headline correlation — but
  enlarged to single-panel size with proper labels and a 95% bootstrap CI on
  the Spearman ρ to show that the correlation is essentially indistinguishable
  from zero.
- Two panels (a, b) for prediction space and feature space respectively.
- Okabe-Ito palette via style_improved.
"""
from __future__ import annotations

import sys
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from cstims.paper import config
from cstims.paper.style_improved import (
    apply_style, FONT, DPI, W_DOUBLE,
    OKABE_ITO, add_panel_label,
)

apply_style()

# Reuse data-prep logic
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_ood_vs_alignment import (
    load_wrsa, compute_delta_alignment, compute_delta_ood, GROUPS,
)

FIGURES = Path(__file__).resolve().parent
OOD_DATA = config.OOD_DATA_DIR / "pca_loglik.csv"


def bootstrap_spearman_ci(x, y, n_boot=2000, ci=95):
    n = len(x)
    if n < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(x[idx])) < 3 or len(np.unique(y[idx])) < 3:
            continue
        r, _ = stats.spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            rs.append(r)
    rs = np.array(rs)
    lo, hi = np.percentile(rs, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(lo), float(hi)


def panel(ax, sub, ood_col, color, panel_label, space_label):
    sub = sub.dropna(subset=[ood_col, "delta_alignment"])
    # Use positive magnitudes for readability:
    # higher x = cstim is more OOD than Vicco; higher y = larger alignment drop.
    x = -sub[ood_col].values
    y = -sub["delta_alignment"].values
    models = sub["model"].values

    ax.axhline(0, color="#999", lw=0.6, ls="--", zorder=0)
    ax.axvline(0, color="#999", lw=0.6, ls="--", zorder=0)
    ax.scatter(x, y, s=40, color=color, alpha=0.85, zorder=3, edgecolor="white",
                linewidth=0.6)

    if len(x) >= 3:
        m, b, *_ = stats.linregress(x, y)
        xr = np.linspace(x.min(), x.max(), 100)
        ax.plot(xr, m * xr + b, color=color, lw=1.4, alpha=0.55, zorder=2)
        r, p = stats.spearmanr(x, y)
        lo, hi = bootstrap_spearman_ci(x, y)
        ax.text(0.97, 0.96,
                 f"Spearman ρ = {r:+.2f}\n"
                 f"95% CI [{lo:+.2f}, {hi:+.2f}]\n"
                 f"n = {len(x)} models",
                 transform=ax.transAxes, ha="right", va="top",
                 fontsize=FONT["small"], color="#222",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                            edgecolor=color, linewidth=0.7, alpha=0.9))

    # Label only the extremal points; labeling all 20 models obscures the trend.
    label_idx = set(np.argsort(x)[:2])
    label_idx.update(np.argsort(x)[-2:])
    label_idx.update(np.argsort(y)[:1])
    label_idx.update(np.argsort(y)[-2:])
    for i, (xi, yi, name) in enumerate(zip(x, y, models)):
        if i not in label_idx:
            continue
        ax.annotate(
            config.MODEL_DISPLAY_NAMES.get(name, name),
            (xi, yi), fontsize=FONT["small"] - 3, ha="left", va="bottom",
            xytext=(3, 3), textcoords="offset points", color="#444",
            alpha=0.85,
        )

    ax.set_xlabel("PPCA OOD increase\n(baseline − cstim loglik z)")
    ax.set_ylabel("mixed-RSA drop (baseline − cstim)")
    ax.set_title(f"{space_label.capitalize()} space", pad=4)
    add_panel_label(ax, panel_label)


def main():
    ood_df = pd.read_csv(OOD_DATA)
    wrsa_df = load_wrsa()
    delta_aln = compute_delta_alignment(wrsa_df)
    delta_ood = compute_delta_ood(ood_df)
    merged = delta_aln.merge(delta_ood, on=["subject", "group", "model"])
    combined = (merged.groupby(["group", "model"])
                 [["delta_alignment", "delta_ood_feature", "delta_ood_pred"]]
                 .mean().reset_index())

    sub = combined[combined["group"] == "all_models"]

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE * 0.65, 4.2))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.21, wspace=0.32)

    panel(axes[0], sub, "delta_ood_pred",   OKABE_ITO["blue"],
           "a", "prediction")
    panel(axes[1], sub, "delta_ood_feature", OKABE_ITO["vermillion"],
           "b", "feature")

    fig.suptitle("PPCA OOD increase does not explain the brain-alignment drop "
                  "(All Models, n=20)", fontsize=FONT["title"], y=1.005)

    for ext in ("pdf", "png"):
        # Single combined figure replaces both pred/feature originals.
        out = FIGURES / f"ood_vs_alignment_improved.{ext}"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
