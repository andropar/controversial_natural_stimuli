#!/usr/bin/env python3
"""
Plot in silico evaluation using unique per-subject encodings.

Three-panel figure:
  (a) Accuracy curves in raw feature space
  (b) Accuracy curves in encoding (predicted brain) space — mean across subjects
  (c) AUC comparison: raw vs encoding, selected vs random

Data: {model_set}_unique/discriminability.csv
  columns: noise_multiplier, error_selected, error_random_mean, error_random_std,
           n_models, n_selected, track
  tracks: raw, sub-01, sub-03, sub-05, sub-06, sub-07

Usage:
    python plot_insilico_evaluation_unique.py
"""

import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
SHARE_ROOT = _PAPER.parents[1]
HELPERS = SHARE_ROOT / "shared" / "code" / "paper_helpers"
sys.path.insert(0, str(HELPERS))
sys.path.insert(0, str(HELPERS / "figures"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

import config
from style import apply_style, FONT, DPI, W_DOUBLE

apply_style()

FIGURES_DIR = Path(__file__).resolve().parent

MODEL_SETS = ["sota", "architecture", "training_objective", "dataset", "all_models"]

MODEL_SET_LABELS = {
    "sota": "SOTA",
    "architecture": "Arch.",
    "training_objective": "Train.\nObj.",
    "dataset": "Dataset",
    "all_models": "All\nModels",
}

MODEL_SET_COLORS = {
    "sota": "#2ECC71",
    "architecture": "#E74C3C",
    "training_objective": "#3498DB",
    "dataset": "#F39C12",
    "all_models": "#9B59B6",
}

ENCODING_TRACKS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]


DATA_SUFFIX = "_unique_boot"


def load_discriminability(model_set: str) -> pd.DataFrame:
    path = config.EVAL_DATA_DIR / f"{model_set}{DATA_SUFFIX}" / "discriminability.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Normalise column names (Raven format uses noise_mult/noise_ceiling/subset_type/error_prob)
    if "noise_mult" in df.columns and "noise_multiplier" not in df.columns:
        df = df.rename(columns={"noise_mult": "noise_multiplier"})
    return df


def load_auc_significance(model_set: str) -> pd.DataFrame:
    path = config.EVAL_DATA_DIR / f"{model_set}{DATA_SUFFIX}" / "auc_significance.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def get_sel_rand(df, track_names, x_col="noise_ceiling"):
    """Extract accuracy arrays for selected and random subsets from given tracks.
    x_col: 'noise_ceiling' (for plotting) or 'noise_multiplier' (for AUC).

    Returns:
        (sel_acc, rand_acc, rand_std, x) where rand_std is the across-subset
        std of accuracy (averaged across tracks if multiple). None if unavailable.
    """
    if isinstance(track_names, str):
        track_names = [track_names]
    sel_list, rand_list, rand_std_list, x = [], [], [], None
    for t in track_names:
        sub = df[df["track"] == t].copy()
        if sub.empty:
            continue
        sel_sub  = sub[sub["subset_type"] == "selected"].sort_values(x_col)
        rand_sub = sub[sub["subset_type"] == "random"].sort_values(x_col)
        if sel_sub.empty or rand_sub.empty:
            continue
        x = sel_sub[x_col].values
        sel_list.append((1 - sel_sub["error_prob"].values))
        rand_list.append((1 - rand_sub["error_prob"].values))
        if "error_prob_std" in rand_sub.columns:
            rand_std_list.append(rand_sub["error_prob_std"].values)
    if not sel_list:
        return None, None, None, None
    rand_std = np.mean(rand_std_list, axis=0) if rand_std_list else None
    return np.mean(sel_list, axis=0), np.mean(rand_list, axis=0), rand_std, x


def compute_auc(noise_mult_vals, accuracy_vals) -> float:
    """AUC via trapezoidal integration over log10(noise_mult) — uniform spacing."""
    x = np.log10(np.array(noise_mult_vals))
    y = np.array(accuracy_vals)
    rng = x.max() - x.min()
    if rng == 0:
        return float(np.mean(y))
    return float(trapezoid(y, x) / rng)


def plot_accuracy_curves(ax, track_key, title, show_legend=True):
    """
    track_key: 'raw' or 'encoding' (averages sub-01..sub-07)
    """
    for ms in MODEL_SETS:
        df = load_discriminability(ms)
        if df.empty:
            continue

        tracks = ["raw"] if track_key == "raw" else ENCODING_TRACKS
        sel, rand, rand_std, x = get_sel_rand(df, tracks)
        if sel is None:
            continue

        col = MODEL_SET_COLORS[ms]
        label = MODEL_SET_LABELS[ms].replace("\n", " ")
        ax.plot(x, sel,  color=col, lw=1.8, ls="-",  alpha=0.9, label=label)
        ax.plot(x, rand, color=col, lw=1.2, ls="--", alpha=0.45)
        # Band = ±1 std across random subsets (stimulus-sampling variance).
        if rand_std is not None:
            ax.fill_between(
                x, rand - rand_std, rand + rand_std,
                color=col, alpha=0.12, linewidth=0,
            )

    ax.plot([], [], "k-",  lw=1.5, label="Selected")
    ax.plot([], [], "k--", lw=1.0, alpha=0.45, label="Random")

    ax.axvline(0.46, color="gray", lw=0.8, ls=":", alpha=0.5)
    ax.set_xlabel("Noise ceiling")
    ax.set_xlim(0, 1.0)
    ax.invert_xaxis()
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    if show_legend:
        ax.legend(loc="lower left", ncol=2, frameon=True, framealpha=0.9,
                  columnspacing=0.8, handletextpad=0.4, fontsize=FONT["small"])


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, 4.8))

    # --- Panel A: Raw feature space ---
    plot_accuracy_curves(axes[0], "raw", "Raw feature space", show_legend=True)
    axes[0].set_ylabel("Model recovery accuracy")
    axes[0].text(-0.12, 1.08, "a", transform=axes[0].transAxes,
                 fontsize=FONT["panel_label"], fontweight="bold", va="top")

    # --- Panel B: Encoding (predicted brain) space ---
    plot_accuracy_curves(axes[1], "encoding", "Encoding (predicted brain) space",
                         show_legend=False)
    axes[1].text(-0.12, 1.08, "b", transform=axes[1].transAxes,
                 fontsize=FONT["panel_label"], fontweight="bold", va="top")

    # --- Panel C: AUC bar chart ---
    ax = axes[2]
    ms_labels = []
    raw_rand, raw_sel, enc_rand, enc_sel = [], [], [], []
    raw_sel_err, raw_rand_err, enc_sel_err, enc_rand_err = [], [], [], []

    for ms in MODEL_SETS:
        df = load_discriminability(ms)
        auc_df = load_auc_significance(ms)
        if df.empty or auc_df.empty:
            continue

        raw_row = auc_df[auc_df["track"] == "raw"]
        enc_rows = auc_df[auc_df["track"].isin(ENCODING_TRACKS)]
        if raw_row.empty or enc_rows.empty:
            continue
        raw_row = raw_row.iloc[0]

        ms_labels.append(MODEL_SET_LABELS[ms])
        # AUCs reported in CSV are on error-prob scale; convert to accuracy.
        raw_sel.append(1 - raw_row["selected_auc"])
        raw_rand.append(1 - raw_row["random_auc_mean"])
        raw_sel_err.append(raw_row["selected_auc_mc_std"])
        raw_rand_err.append(raw_row["random_auc_subset_std"])

        # Encoding: average across subject tracks (mean of per-subject AUCs).
        enc_sel.append(1 - enc_rows["selected_auc"].mean())
        enc_rand.append(1 - enc_rows["random_auc_mean"].mean())
        # Propagate errs: quadrature-mean of per-subject errs / sqrt(n).
        n_subj = len(enc_rows)
        enc_sel_err.append(
            float(np.sqrt((enc_rows["selected_auc_mc_std"] ** 2).mean() / n_subj))
        )
        enc_rand_err.append(
            float(np.sqrt((enc_rows["random_auc_subset_std"] ** 2).mean() / n_subj))
        )

    x = np.arange(len(ms_labels))
    width = 0.19
    colors = [MODEL_SET_COLORS[ms] for ms in MODEL_SETS[:len(ms_labels)]]

    err_kw = dict(ecolor="0.3", capsize=2, elinewidth=0.8)
    ax.bar(x - 1.5*width, raw_rand, width, yerr=raw_rand_err, color="#DDDDDD", alpha=0.85,
           edgecolor="white", lw=0.5, hatch="//", label="Random (raw)", error_kw=err_kw)
    ax.bar(x - 0.5*width, raw_sel,  width, yerr=raw_sel_err, color=colors, alpha=0.5,
           edgecolor="white", lw=0.5, hatch="//", label="Selected (raw)", error_kw=err_kw)
    ax.bar(x + 0.5*width, enc_rand, width, yerr=enc_rand_err, color="#AAAAAA", alpha=0.85,
           edgecolor="white", lw=0.5, label="Random (enc.)", error_kw=err_kw)
    ax.bar(x + 1.5*width, enc_sel,  width, yerr=enc_sel_err, color=colors, alpha=0.85,
           edgecolor="white", lw=0.5, label="Selected (enc.)", error_kw=err_kw)

    for i in range(len(ms_labels)):
        ms = MODEL_SETS[i]
        # Raw: percentage improvement above the selected bar
        raw_top = max(raw_rand[i] + raw_rand_err[i], raw_sel[i] + raw_sel_err[i])
        pct = (raw_sel[i] - raw_rand[i]) / max(raw_rand[i], 1e-6) * 100
        ax.text(x[i] - width, raw_top + 0.03, f"+{pct:.0f}%",
                fontsize=FONT["small"], ha="center", va="bottom",
                color=MODEL_SET_COLORS[ms], alpha=0.7)

        # Encoding
        enc_top = max(enc_rand[i] + enc_rand_err[i], enc_sel[i] + enc_sel_err[i])
        pct = (enc_sel[i] - enc_rand[i]) / max(enc_rand[i], 1e-6) * 100
        ax.text(x[i] + width, enc_top + 0.03, f"+{pct:.0f}%",
                fontsize=FONT["small"], ha="center", va="bottom",
                color=MODEL_SET_COLORS[ms])

    ax.legend(loc="lower left", frameon=True, framealpha=0.9,
              fontsize=FONT["small"], ncol=2, columnspacing=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ms_labels, fontsize=FONT["tick"])
    ax.set_ylabel("Accuracy AUC (higher = better)")
    ax.set_ylim(0.35, 1.08)
    ax.set_title("AUC comparison")
    ax.text(-0.12, 1.08, "c", transform=ax.transAxes,
            fontsize=FONT["panel_label"], fontweight="bold", va="top")

    plt.tight_layout(w_pad=2.5)

    for ext in ["pdf", "png"]:
        out = FIGURES_DIR / f"insilico_evaluation_unique.{ext}"
        fig.savefig(out, dpi=DPI)
        print(f"Saved {out}")

    plt.close()


if __name__ == "__main__":
    main()
