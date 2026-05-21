"""
06_per_set_ood_scatter.py

Per-model-set scatter of per-model Δalignment vs Δ(per-model OOD), separately
for feature-space and prediction-space OOD. Makes the sign-reversal across
model sets explicit in supplement.

The Δ values come from the existing per-model loglikelihoods (each model
evaluated in its own feature/prediction space), so this is the per-model
OOD axis flagged in the dissociation caption — not a global axis.

Reads:
  experiments/cstim_paper/06_ood/data/ood_vs_alignment.csv
Writes:
  experiments/cstim_paper/06_ood/figures/ood_per_model.{pdf,png}
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

STAGE = Path(__file__).resolve().parents[1]
DATA = STAGE / "data" / "ood_vs_alignment.csv"
OUT = STAGE / "figures" / "supplementary"
PNG_OUT = OUT / "png"
OUT.mkdir(parents=True, exist_ok=True)
PNG_OUT.mkdir(parents=True, exist_ok=True)

GROUPS = ["all_models", "sota", "training_objective", "architecture", "dataset"]
LABELS = {
    "all_models": "All models",
    "sota": "SOTA",
    "training_objective": "Training Obj.",
    "architecture": "Architecture",
    "dataset": "Dataset",
}
COLORS = {
    "all_models": "#666666",
    "sota": "#1f77b4",
    "training_objective": "#2ca02c",
    "architecture": "#d62728",
    "dataset": "#9467bd",
}


def main():
    df = pd.read_csv(DATA)
    print(df.head())
    print("groups:", sorted(df["group"].unique()))

    fig, axes = plt.subplots(2, len(GROUPS), figsize=(2.6 * len(GROUPS), 4.6),
                             sharey=True)
    for col, g in enumerate(GROUPS):
        sub = df[df["group"] == g].dropna()
        if sub.empty:
            for r in (0, 1):
                axes[r, col].axis("off")
            continue

        for row, ood_col, ood_label in [
            (0, "delta_ood_pred", r"$\Delta$ OOD (prediction)"),
            (1, "delta_ood_feature", r"$\Delta$ OOD (feature)"),
        ]:
            ax = axes[row, col]
            x = sub[ood_col].values
            y = sub["delta_alignment"].values
            color = COLORS.get(g, "#444")
            ax.scatter(x, y, s=28, color=color, edgecolor="white", lw=0.6, alpha=0.85)
            if len(x) >= 3:
                rho, p = spearmanr(x, y)
            else:
                rho, p = np.nan, np.nan
            sign = r"$\uparrow$" if rho > 0 else r"$\downarrow$"
            ax.text(0.04, 0.95, fr"$\rho={rho:+.2f}$",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=8.5,
                    bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.9, pad=2))
            ax.axhline(0, color="0.7", lw=0.6, ls="--")
            ax.axvline(0, color="0.7", lw=0.6, ls="--")
            if row == 0:
                ax.set_title(LABELS[g], fontsize=10)
            if col == 0:
                ax.set_ylabel(r"$\Delta$ alignment" + ("\n(prediction)" if row == 0 else "\n(feature)"))
            ax.set_xlabel(ood_label, fontsize=8)
            ax.tick_params(labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.suptitle(
        r"Per-model $\Delta$alignment vs $\Delta$OOD by model set (per-model OOD axis: each model in its own feature/prediction space)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "ood_per_model.pdf", bbox_inches="tight")
    fig.savefig(PNG_OUT / "ood_per_model.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT / 'ood_per_model.pdf'}")

    # Print the per-set rho summary
    rows = []
    for g in GROUPS:
        sub = df[df["group"] == g].dropna()
        if len(sub) < 3:
            continue
        for col in ("delta_ood_pred", "delta_ood_feature"):
            rho, p = spearmanr(sub[col], sub["delta_alignment"])
            rows.append({"group": g, "ood_space": col, "n": len(sub),
                         "rho": rho, "p": p})
    summary = pd.DataFrame(rows)
    summary.to_csv(STAGE / "data" / "ood_per_model_rho.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
