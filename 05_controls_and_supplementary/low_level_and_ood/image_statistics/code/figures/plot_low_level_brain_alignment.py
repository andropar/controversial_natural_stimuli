#!/usr/bin/env python3
"""
Bar chart: low-level image RDM ↔ brain RDM alignment, per stimulus set.

If low-level features (pixels or basic image stats) explain the brain
alignment drop observed for controversial stimuli, this plot should show
*lower* RSA for controversial sets vs vicco. If instead the pixel/stats
alignment is similar across sets, the drop is model-specific — low-level
structure cannot be the culprit.

One bar per (stimulus_set × rdm_kind), averaged over subjects. For vicco,
values are first averaged across bootstrap samples within-subject, then
across subjects. Individual subjects are shown as dots.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER))
from style_improved import apply_style, FONT  # noqa: E402

apply_style()

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "results" / "low_level_rdm_brain_alignment.csv"
OUT_PDF = HERE / "low_level_brain_alignment.pdf"
OUT_PNG = HERE / "low_level_brain_alignment.png"

SET_ORDER = ["vicco", "all_models", "architecture", "dataset", "sota", "training_objective"]
SET_LABELS = {
    "vicco": "baseline\n(vicco)",
    "all_models": "all",
    "architecture": "arch",
    "dataset": "data",
    "sota": "sota",
    "training_objective": "obj",
}
KINDS = ["pixel", "stats"]
KIND_LABELS = {"pixel": "Pixel RDM (64×64 grayscale)",
               "stats": "Stats RDM (14-d low-level)"}
KIND_COLORS = {"pixel": "#2c7fb8", "stats": "#d95f0e"}


def main():
    df = pd.read_csv(CSV)
    # collapse bootstraps (vicco) to per-subject mean
    per_subj = (
        df.groupby(["subject", "stimulus_set", "rdm_kind"], as_index=False)["rsa"]
        .mean()
    )

    sets = [s for s in SET_ORDER if s in per_subj["stimulus_set"].unique()]
    xs = np.arange(len(sets))
    width = 0.38

    fig, ax = plt.subplots(figsize=(6.0, 2.6))

    # vicco reference lines (per kind, mean across subjects) ----------------
    vicco_means = {}
    if "vicco" in sets:
        for k in KINDS:
            v = per_subj[(per_subj["stimulus_set"] == "vicco")
                         & (per_subj["rdm_kind"] == k)]["rsa"].mean()
            vicco_means[k] = v
            ax.axhline(v, color=KIND_COLORS[k], lw=0.6, ls="--", alpha=0.55, zorder=0)

    # bars + per-subject dots ------------------------------------------------
    for i, kind in enumerate(KINDS):
        offs = (i - 0.5) * width
        means, sems, per_vals = [], [], []
        for s in sets:
            vals = per_subj[(per_subj["stimulus_set"] == s)
                            & (per_subj["rdm_kind"] == kind)]["rsa"].values
            means.append(vals.mean() if len(vals) else np.nan)
            sems.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
            per_vals.append(vals)
        ax.bar(xs + offs, means, width=width, color=KIND_COLORS[kind],
               yerr=sems, capsize=2, ecolor="black", linewidth=0,
               label=KIND_LABELS[kind], alpha=0.85)
        # dots
        rng = np.random.default_rng(i)
        for xi, vals in zip(xs, per_vals):
            jitter = rng.uniform(-width * 0.25, width * 0.25, size=len(vals))
            ax.scatter(xi + offs + jitter, vals, s=6, color="black",
                       alpha=0.55, linewidths=0, zorder=3)

    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([SET_LABELS[s] for s in sets], fontsize=FONT["tick"])
    ax.set_ylabel("RSA (Spearman)\nlow-level RDM ↔ brain RDM",
                  fontsize=FONT["axis_label"])
    ax.set_title("Low-level image similarity does not track the brain-alignment drop"
                 if True else "", fontsize=FONT["title"])
    ax.legend(fontsize=FONT["legend"], frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=300)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")

    # quick-read printout
    print("\n=== mean RSA across subjects (per set × kind) ===")
    print(per_subj.groupby(["stimulus_set", "rdm_kind"])["rsa"].mean().unstack().round(3))


if __name__ == "__main__":
    main()
