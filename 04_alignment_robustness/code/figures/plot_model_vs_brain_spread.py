"""
plot_model_vs_brain_spread.py

Plots model RDM spread vs brain noise ceiling across stimulus groups,
testing whether model-predicted discriminability predicts brain NC.

Reads:
  data/rdm_noise_ceilings.csv    (from 02_noise_ceilings.py)
  data/model_rdm_spreads.csv     (from 09_model_rdm_spreads.py)

Outputs:
  figures/model_vs_brain_spread.pdf / .png
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import stats

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER))
import config

FIG_DIR = Path(__file__).parent
DATA_DIR = config.STATS_DATA_DIR

GROUP_ORDER = ["all_models", "architecture", "training_objective", "sota", "dataset", "vicco"]
GROUP_COLORS = {
    "all_models":         "#e41a1c",
    "architecture":       "#377eb8",
    "training_objective": "#4daf4a",
    "sota":               "#984ea3",
    "dataset":            "#ff7f00",
    "vicco":              "#888888",
}
GROUP_LABELS = {
    "all_models":         "all_models",
    "architecture":       "architecture",
    "training_objective": "train. obj.",
    "sota":               "sota",
    "dataset":            "dataset",
    "vicco":              "vicco\n(baseline)",
}


def load_data():
    brain = pd.read_csv(DATA_DIR / "rdm_noise_ceilings.csv")
    model = pd.read_csv(DATA_DIR / "model_rdm_spreads.csv")

    # Brain NC per subject per group (vicco: mean over bootstraps first, then per subject)
    vicco_per_subj = (brain[brain["stimulus_type"] == "vicco"]
                      .groupby("subject")["noise_ceiling_spearman"].mean())
    cstim_per_subj = (brain[brain["stimulus_type"] == "controversial"]
                      .groupby(["group", "subject"])["noise_ceiling_spearman"].mean())

    brain_nc_subj = {}
    for g in [g for g in GROUP_ORDER if g != "vicco"]:
        if g in cstim_per_subj.index.get_level_values("group"):
            brain_nc_subj[g] = cstim_per_subj[g].to_dict()
    brain_nc_subj["vicco"] = vicco_per_subj.to_dict()

    brain_nc_mean = {g: np.mean(list(v.values())) for g, v in brain_nc_subj.items()}

    # Model RDM spread: mean across bootstraps per group
    model_spread = model.groupby("group")["model_rdm_spread"].mean().to_dict()

    return brain_nc_mean, brain_nc_subj, model_spread


def main():
    brain_nc, brain_nc_subj, model_spread = load_data()
    groups = [g for g in GROUP_ORDER if g in brain_nc]

    vicco_nc_val = brain_nc["vicco"]
    nc_norm = {g: brain_nc[g] / vicco_nc_val for g in groups}
    ms_norm = {g: model_spread[g] / model_spread["vicco"] for g in groups}

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))

    # -------------------------------------------------------------------------
    # Left: dot plot, both metrics normalized to vicco
    # -------------------------------------------------------------------------
    x_nc = np.arange(len(groups))
    x_ms = x_nc + 0.22

    for i, g in enumerate(groups):
        color = GROUP_COLORS[g]
        subj_ncs = list(brain_nc_subj[g].values())
        # Individual subjects
        ax_left.scatter(np.full(len(subj_ncs), x_nc[i]),
                        [v / vicco_nc_val for v in subj_ncs],
                        color=color, s=20, alpha=0.4, zorder=2)
        # Group means + model spread
        ax_left.scatter(x_nc[i], nc_norm[g], color=color, s=90, zorder=4, marker="o")
        ax_left.scatter(x_ms[i], ms_norm[g], color=color, s=90, zorder=4, marker="s")
        # Connect means
        ax_left.plot([x_nc[i], x_ms[i]], [nc_norm[g], ms_norm[g]],
                     color=color, lw=0.8, alpha=0.5)

    ax_left.axhline(1.0, color="#888888", lw=1.2, ls="--", alpha=0.6)
    ax_left.set_xticks(x_nc + 0.11)
    ax_left.set_xticklabels([GROUP_LABELS[g] for g in groups], fontsize=9)
    ax_left.set_ylabel("Value relative to vicco", fontsize=10)
    ax_left.set_title("Model RDM spread  vs  Brain NC\n(normalized to vicco)", fontsize=11)
    ax_left.grid(True, axis="y", alpha=0.3)

    legend_elements = [
        Line2D([0], [0], marker="o", color="k", lw=0, markersize=7, label="Brain NC (mean ± subjects)"),
        Line2D([0], [0], marker="s", color="k", lw=0, markersize=7, label="Model RDM spread"),
        Line2D([0], [0], color="#888888", lw=1.2, ls="--", label="vicco baseline"),
    ]
    for g in groups:
        legend_elements.append(
            Line2D([0], [0], marker="o", color=GROUP_COLORS[g], lw=0,
                   markersize=7, label=GROUP_LABELS[g])
        )
    ax_left.legend(handles=legend_elements, fontsize=8, framealpha=0.7, loc="upper right")

    # -------------------------------------------------------------------------
    # Right: scatter model spread vs brain NC, per-subject dots behind means
    # -------------------------------------------------------------------------
    for g in groups:
        color = GROUP_COLORS[g]
        subj_ncs = list(brain_nc_subj[g].values())
        ax_right.scatter(np.full(len(subj_ncs), model_spread[g]), subj_ncs,
                         color=color, s=20, alpha=0.4, zorder=2)
        ax_right.scatter(model_spread[g], brain_nc[g], color=color, s=90, zorder=4)
        ax_right.annotate(GROUP_LABELS[g], (model_spread[g], brain_nc[g]),
                          fontsize=8, xytext=(5, 3), textcoords="offset points")

    x_arr = np.array([model_spread[g] for g in groups])
    y_arr = np.array([brain_nc[g] for g in groups])
    slope, intercept, r, p, _ = stats.linregress(x_arr, y_arr)
    xline = np.linspace(x_arr.min() - 0.002, x_arr.max() + 0.002, 100)
    ax_right.plot(xline, slope * xline + intercept, "k--", lw=1.2, alpha=0.5,
                  label=f"r = {r:.2f},  p = {p:.2f}\n(group means, n={len(groups)})")

    ax_right.set_xlabel("Model RDM spread (mean across models)", fontsize=10)
    ax_right.set_ylabel("Brain NC (Spearman-Brown)", fontsize=10)
    ax_right.set_title("Does model-predicted spread predict brain NC?", fontsize=11)
    ax_right.legend(fontsize=9, framealpha=0.7)
    ax_right.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = FIG_DIR / f"model_vs_brain_spread.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
