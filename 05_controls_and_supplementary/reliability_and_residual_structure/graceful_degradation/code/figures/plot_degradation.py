#!/usr/bin/env python3
"""
Graceful-degradation figure.

Four panels (2x2):
  Row 1: fixed RSA (cRSA)
  Row 2: mixed RSA (wRSA-transfer)

Col 1: per-subject argmax stability vs measured effective NC.
  For each subject × condition, the k=4 winner on that subsample is the canonical.
  At each measured NC (from per-subsample split-half against complement), compute
  fraction of rep-subsamples whose argmax matches canonical. Points are per
  (subject, boot, k); overlaid binned means ± SEM per condition.

Col 2: ensemble concentration (pooled across subjects/boots/subsets) on the
  top-winning model, vs measured NC. A high curve = single model dominates
  across the ensemble; a low curve = different subjects pick different winners.

NC axes are shared across panels for apples-to-apples comparison across RSA types.

Reads:  ../data/degradation_results.csv
Writes: degradation_curves.{pdf,png}
"""

from __future__ import annotations

import sys
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))

from cstims.paper.style_improved import apply_style, FONT, DPI, W_DOUBLE  # noqa: E402

apply_style()

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "results" / "degradation_results.csv"
OUT_PDF = HERE / "degradation_curves.pdf"
OUT_PNG = HERE / "degradation_curves.png"

COND_COLOR = {"controversial": "#D64541", "baseline": "#2980B9"}
COND_LABEL = {"controversial": "Controversial (all-models)", "baseline": "Baseline (vicco)"}

NC_BINS = np.arange(0.05, 0.50, 0.05)  # 0.05 ... 0.45 edges -> centers 0.075, 0.125, ..., 0.425


def short_model(name: str) -> str:
    from cstims.constants import MODEL_DISPLAY_NAMES
    return MODEL_DISPLAY_NAMES.get(name, name.replace("_", " "))[:14]


def canonical_winner(df: pd.DataFrame) -> pd.DataFrame:
    k_max = df["k_reps"].max()
    full = df[df["k_reps"] == k_max]
    keys = ["subject", "condition", "rsa_type", "boot_idx"]
    return (full.groupby(keys)["argmax_model"].first()
                .reset_index().rename(columns={"argmax_model": "canonical"}))


def _bin_center_series(nc: pd.Series) -> pd.Series:
    idx = np.digitize(nc.values, NC_BINS) - 1
    idx = np.clip(idx, 0, len(NC_BINS) - 2)
    centers = 0.5 * (NC_BINS[idx] + NC_BINS[idx + 1])
    return pd.Series(centers, index=nc.index, name="nc_bin")


def per_subject_agreement(df: pd.DataFrame) -> pd.DataFrame:
    canon = canonical_winner(df)
    m = df.merge(canon, on=["subject", "condition", "rsa_type", "boot_idx"], how="left")
    m["agree"] = (m["argmax_model"] == m["canonical"]).astype(int)
    return m


def pooled_concentration_binned(df: pd.DataFrame, rsa_type: str) -> pd.DataFrame:
    sub = df[df["rsa_type"] == rsa_type].copy()
    sub["nc_bin"] = _bin_center_series(sub["eff_nc"])
    rows = []
    for (cond, nc_bin), g in sub.groupby(["condition", "nc_bin"]):
        counts = g["argmax_model"].value_counts(normalize=True)
        rows.append({
            "rsa_type": rsa_type,
            "condition": cond,
            "nc_bin": nc_bin,
            "max_winrate": float(counts.iloc[0]) if len(counts) else np.nan,
            "top_model": counts.index[0] if len(counts) else "n/a",
            "n_rows": len(g),
        })
    return pd.DataFrame(rows)


def plot_stability_panel(ax, df_rsa: pd.DataFrame, rsa_label: str):
    agg_df = per_subject_agreement(df_rsa)
    agg_df = agg_df.copy()
    agg_df["nc_bin"] = _bin_center_series(agg_df["eff_nc"])

    for cond in ["baseline", "controversial"]:
        sub = agg_df[agg_df["condition"] == cond]
        # raw per-row dots
        ax.scatter(sub["eff_nc"], sub["agree"],
                   color=COND_COLOR[cond], s=8, alpha=0.25, zorder=2)
        # bin means
        agg = (sub.groupby("nc_bin")
                   .agg(mean=("agree", "mean"),
                        sem=("agree", lambda v: v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0),
                        n=("agree", "size"))
                   .reset_index()
                   .query("n >= 3"))
        ax.errorbar(agg["nc_bin"], agg["mean"], yerr=agg["sem"],
                    color=COND_COLOR[cond], marker="o", markersize=5,
                    linewidth=1.8, capsize=3, label=COND_LABEL[cond], zorder=4)

    ax.axhline(1.0, color="#888888", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xlabel("Measured effective NC (brain RDM split-half)",
                  fontsize=FONT["axis_label"])
    ax.set_ylabel("P(argmax = canonical 4-rep winner)", fontsize=FONT["axis_label"])
    ax.set_title(f"{rsa_label}: argmax stability", fontsize=FONT["title"])
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.0, 0.5)
    ax.legend(frameon=True, framealpha=0.9, fontsize=FONT["legend"], loc="lower right")


def plot_concentration_panel(ax, df: pd.DataFrame, rsa_type: str, rsa_label: str):
    conc = pooled_concentration_binned(df, rsa_type)
    for cond in ["baseline", "controversial"]:
        sub = conc[conc["condition"] == cond].sort_values("nc_bin")
        sub = sub[sub["n_rows"] >= 5]  # suppress thin bins
        if sub.empty:
            continue
        ax.plot(sub["nc_bin"], sub["max_winrate"],
                color=COND_COLOR[cond], marker="s", markersize=5,
                linewidth=1.8, label=COND_LABEL[cond])
        # label the top-model at each point
        for _, row in sub.iterrows():
            ax.annotate(
                short_model(row["top_model"]),
                (row["nc_bin"], row["max_winrate"]),
                textcoords="offset points", xytext=(4, 5),
                fontsize=FONT["small"], color=COND_COLOR[cond], alpha=0.9,
            )
    ax.axhline(1 / 20, color="#888888", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.text(0.01, 1 / 20 + 0.005, "chance (1/20)",
            fontsize=FONT["small"], color="#888888")
    ax.set_xlabel("Measured effective NC (brain RDM split-half)",
                  fontsize=FONT["axis_label"])
    ax.set_ylabel("Top-model win rate (ensemble)", fontsize=FONT["axis_label"])
    ax.set_title(f"{rsa_label}: ensemble concentration", fontsize=FONT["title"])
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.0, 0.5)
    ax.legend(frameon=True, framealpha=0.9, fontsize=FONT["legend"], loc="lower right")


def plot(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE, 8.4))

    # Row 0: fRSA
    plot_stability_panel(axes[0, 0], df[df["rsa_type"] == "fRSA"], "Fixed RSA")
    plot_concentration_panel(axes[0, 1], df, "fRSA", "Fixed RSA")

    # Row 1: mRSA
    plot_stability_panel(axes[1, 0], df[df["rsa_type"] == "mRSA"], "Mixed RSA")
    plot_concentration_panel(axes[1, 1], df, "mRSA", "Mixed RSA")

    for letter, ax in zip("abcd", axes.ravel()):
        ax.text(-0.14, 1.06, letter, transform=ax.transAxes,
                fontsize=FONT["panel_label"], fontweight="bold", va="top")

    plt.tight_layout(w_pad=3.0, h_pad=3.0)
    fig.savefig(OUT_PDF, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
    print(f"Saved {OUT_PDF}\nSaved {OUT_PNG}")
    plt.close(fig)


def main():
    df = pd.read_csv(DATA)

    # Sanity: per-subject × condition × rsa_type k=4 winner
    print("\nk=4 winners per (subject, condition, rsa_type):")
    k_max = df["k_reps"].max()
    w = (df[df["k_reps"] == k_max]
         .groupby(["subject", "condition", "rsa_type", "boot_idx"])[
             "argmax_model"].first().reset_index())
    for (s, c, r), g in w.groupby(["subject", "condition", "rsa_type"]):
        top = g["argmax_model"].value_counts(normalize=True)
        print(f"  {s:6s}  {c:13s}  {r:5s}  top={top.index[0]:40s} "
              f"({top.iloc[0]*100:.0f}% of boots)")

    print("\nMean margin (winner - 2nd place) by condition × rsa_type:")
    for rsa_type in ["fRSA", "mRSA"]:
        for cond in ["controversial", "baseline"]:
            g = df[(df["rsa_type"] == rsa_type) & (df["condition"] == cond) &
                   (df["k_reps"] == 4)]
            if len(g) > 0:
                print(f"  {rsa_type}  {cond:13s}  mean margin = {g['margin'].mean():.4f}  "
                      f"median = {g['margin'].median():.4f}")

    plot(df)


if __name__ == "__main__":
    main()
