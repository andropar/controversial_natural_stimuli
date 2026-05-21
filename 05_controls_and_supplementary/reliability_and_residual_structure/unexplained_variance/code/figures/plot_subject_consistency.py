#!/usr/bin/env python3
"""
Bar plot of pairwise subject consistency: residual RDM vs full brain RDM.

Reads data/subject_consistency.csv (output of 01_subject_consistency.py).

Outputs:
    figures/subject_consistency_bar.pdf/png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER / "figures"))

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parent

try:
    from style_improved import apply_style, DPI, W_SINGLE
    apply_style()
except ImportError:
    DPI = 150
    W_SINGLE = 6

df = pd.read_csv(DATA_DIR / "subject_consistency.csv")

# Two grouped bars: mean ± sem across subject pairs
means = [df["brain_rho"].mean(), df["residual_rho"].mean()]
sems  = [df["brain_rho"].sem(),  df["residual_rho"].sem()]
labels = ["Full brain RDM", "Residual RDM"]
colors = ["#2980B9", "#D64541"]

fig, ax = plt.subplots(figsize=(W_SINGLE * 0.5, 3.5))

bars = ax.bar([0, 1], means, yerr=sems, capsize=4,
              color=colors, alpha=0.85, width=0.5, error_kw={"linewidth": 1.2})

# Overlay individual pair dots
for i, col in enumerate(["brain_rho", "residual_rho"]):
    jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(df))
    ax.scatter(np.full(len(df), i) + jitter, df[col],
               color="black", s=18, zorder=3, alpha=0.7)

ax.set_xticks([0, 1])
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Spearman ρ (inter-subject)")
ax.set_title("Subject consistency\n(all_models controversial stimuli)", fontsize=9)
ax.set_ylim(0, 0.5)
ax.axhline(0, color="black", linewidth=0.6)

# *** above each bar (all p < 1e-30)
for i, (m, s) in enumerate(zip(means, sems)):
    ax.text(i, m + s + 0.015, "***", ha="center", va="bottom", fontsize=9)

fig.tight_layout()
fig.savefig(FIG_DIR / "subject_consistency_bar.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "subject_consistency_bar.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("Saved: figures/subject_consistency_bar.pdf/png")
