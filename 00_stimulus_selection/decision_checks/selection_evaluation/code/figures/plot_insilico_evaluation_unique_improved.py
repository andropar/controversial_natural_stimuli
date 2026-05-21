#!/usr/bin/env python3
"""
Improved in-silico evaluation figure.

Fixes vs. original:
- Okabe-Ito model-set palette (consistent across all paper figures)
- Panel c: grouped bars with explicit Selected/Random uncertainty and
  percentage improvement labels.
- x-axis labelled as the injected Gaussian noise standard deviation
  multiplier. Absolute sigma is model-specific, so the shared x-axis is
  sigma / sigma_target.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))
sys.path.insert(0, str(_PAPER / "figures"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from style_improved import (
    apply_style, FONT, DPI, W_DOUBLE,
    MODEL_SET_ORDER, MODEL_SET_DISPLAY_SHORT, model_set_color,
    add_panel_label,
)

apply_style()

FIGURES_DIR = Path(__file__).resolve().parent
INSILICO_CURVE_DIR = FIGURES_DIR / "insilico_curve"
MANUSCRIPT_FIGURES_DIR = (
    _PAPER.parents[1] / "writing" / "cstims_paper" / "figures" / "01_insilico"
)
DATA_SUFFIX = "_unique_boot"
ENCODING_TRACKS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
DUPLICATE_CORR_THRESHOLD = 0.9999
ERROR_CLIP_TOL = 1e-8


@lru_cache(maxsize=None)
def duplicate_model_adjustment(model_set: str) -> tuple[int, int]:
    """Return (n_models, n_forced_duplicate_misses) for stale duplicate rosters.

    The current all_models bootstrap output was generated with two feature-
    identical CLIP variants. Torch argmax resolves the tie to the earlier model,
    so the later duplicate is counted as one guaranteed error at every noise
    draw. Remove that bookkeeping miss before plotting the aggregate curve.
    """
    if model_set != "all_models":
        return 0, 0

    path = config.EVAL_DATA_DIR / f"{model_set}{DATA_SUFFIX}" / "correlation_matrices.csv"
    if not path.exists():
        return 0, 0

    corr_df = pd.read_csv(path)
    clean = corr_df[
        (corr_df["track"] == "raw")
        & (corr_df["matrix_type"] == "selected_clean")
    ]
    if clean.empty:
        return 0, 0

    model_names = list(pd.unique(clean["model_i"]))
    offdiag = clean[clean["model_i"] != clean["model_j"]].copy()
    offdiag = offdiag[offdiag["correlation"] >= DUPLICATE_CORR_THRESHOLD]
    if offdiag.empty:
        return len(model_names), 0

    duplicate_pairs = {
        tuple(sorted((row.model_i, row.model_j)))
        for row in offdiag.itertuples(index=False)
    }
    duplicate_models = set().union(*duplicate_pairs)
    components: list[set[str]] = []
    for pair in duplicate_pairs:
        overlapping = [comp for comp in components if comp.intersection(pair)]
        if not overlapping:
            components.append(set(pair))
            continue
        merged = set(pair).union(*overlapping)
        components = [comp for comp in components if comp not in overlapping]
        components.append(merged)

    n_forced_misses = sum(len(comp) - 1 for comp in components)
    if not duplicate_models:
        n_forced_misses = 0
    return len(model_names), n_forced_misses


def correct_duplicate_error_columns(df: pd.DataFrame, model_set: str) -> pd.DataFrame:
    n_models, n_duplicate_misses = duplicate_model_adjustment(model_set)
    if n_duplicate_misses == 0:
        return df

    corrected_n = n_models - n_duplicate_misses
    if corrected_n <= 0:
        return df

    out = df.copy()
    scale = n_models / corrected_n
    offset = n_duplicate_misses / corrected_n

    def _transform_error(values: pd.Series) -> pd.Series:
        corrected = (values.astype(float) * scale - offset).clip(
            lower=0.0, upper=1.0
        )
        corrected = corrected.mask(corrected < ERROR_CLIP_TOL, 0.0)
        return corrected.mask((1.0 - corrected).abs() < ERROR_CLIP_TOL, 1.0)

    for col in [
        "error_prob",
        "error_prob_mc_ci_lo",
        "error_prob_mc_ci_hi",
        "auc",
    ]:
        if col in out.columns:
            out[col] = _transform_error(out[col])

    for col in ["error_prob_std", "error_prob_mc_std"]:
        if col in out.columns:
            out[col] = out[col].astype(float) * scale

    return out


def correct_duplicate_auc_columns(df: pd.DataFrame, model_set: str) -> pd.DataFrame:
    n_models, n_duplicate_misses = duplicate_model_adjustment(model_set)
    if n_duplicate_misses == 0:
        return df

    corrected_n = n_models - n_duplicate_misses
    if corrected_n <= 0:
        return df

    out = df.copy()
    scale = n_models / corrected_n
    offset = n_duplicate_misses / corrected_n

    auc_error_cols = [
        "selected_auc",
        "selected_auc_mc_ci_lo",
        "selected_auc_mc_ci_hi",
        "random_auc_mean",
        "random_auc_subset_ci_lo",
        "random_auc_subset_ci_hi",
    ]
    auc_std_cols = [
        "selected_auc_mc_std",
        "random_auc_subset_std",
        "random_auc_mc_std",
    ]

    for col in auc_error_cols:
        if col in out.columns:
            corrected = (out[col].astype(float) * scale - offset).clip(
                lower=0.0, upper=1.0
            )
            corrected = corrected.mask(corrected < ERROR_CLIP_TOL, 0.0)
            out[col] = corrected.mask((1.0 - corrected).abs() < ERROR_CLIP_TOL, 1.0)

    for col in auc_std_cols:
        if col in out.columns:
            out[col] = out[col].astype(float) * scale

    return out


def load_discriminability(model_set: str) -> pd.DataFrame:
    path = config.EVAL_DATA_DIR / f"{model_set}{DATA_SUFFIX}" / "discriminability.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "noise_mult" in df.columns and "noise_multiplier" not in df.columns:
        df = df.rename(columns={"noise_mult": "noise_multiplier"})
    return correct_duplicate_error_columns(df, model_set)


def load_auc(model_set: str) -> pd.DataFrame:
    path = config.EVAL_DATA_DIR / f"{model_set}{DATA_SUFFIX}" / "auc_significance.csv"
    if not path.exists():
        return pd.DataFrame()
    return correct_duplicate_auc_columns(pd.read_csv(path), model_set)


def get_sel_rand(df, track_names, x_col="noise_multiplier"):
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
        sel_list.append(1 - sel_sub["error_prob"].values)
        rand_list.append(1 - rand_sub["error_prob"].values)
        if "error_prob_std" in rand_sub.columns:
            rand_std_list.append(rand_sub["error_prob_std"].values)
    if not sel_list:
        return None, None, None, None
    rand_std = np.mean(rand_std_list, axis=0) if rand_std_list else None
    return np.mean(sel_list, axis=0), np.mean(rand_list, axis=0), rand_std, x


def plot_curves(ax, track_key, title, show_legend=False, show_noise_axis=False):
    for ms in MODEL_SET_ORDER:
        df = load_discriminability(ms)
        if df.empty:
            continue
        tracks = ["raw"] if track_key == "raw" else ENCODING_TRACKS
        sel, rand, rand_std, x = get_sel_rand(df, tracks)
        if sel is None:
            continue
        col = model_set_color(ms)
        label = MODEL_SET_DISPLAY_SHORT[ms]
        ax.plot(x, sel,  color=col, lw=1.8, ls="-",  alpha=0.95, label=label)
        ax.plot(x, rand, color=col, lw=1.0, ls="--", alpha=0.45)
        if rand_std is not None:
            ax.fill_between(
                x, rand - rand_std, rand + rand_std,
                color=col, alpha=0.10, linewidth=0,
            )

    # Style legend entries
    ax.plot([], [], "k-",  lw=1.5, label="Selected")
    ax.plot([], [], "k--", lw=1.0, alpha=0.45, label="Random")

    # Calibration target: prominent dashed line so readers can't miss it.
    ax.axvline(1.0, color="#444444", lw=1.4, ls=(0, (4, 2)), alpha=0.85, zorder=2)
    ax.text(0.92, 0.97, "selection\ntarget",
            fontsize=FONT["small"] - 1, color="#222222",
            ha="right", va="top", fontweight="bold",
            transform=ax.get_xaxis_transform())

    ax.set_xlabel(r"Gaussian noise std ($\sigma / \sigma_{\mathrm{target}}$)")
    ax.set_xscale("log")
    ax.set_xlim(0.09, 110.0)
    ax.set_xticks([0.1, 0.3, 1, 3, 10, 30, 100])
    ax.set_xticklabels(["0.1", "0.3", "1", "3", "10", "30", "100"])
    ax.set_ylim(-0.02, 1.02)
    if show_noise_axis:
        # Noise annotation under x-axis to disambiguate left vs right.
        ax.annotate(
            "", xy=(0.98, -0.30), xytext=(0.02, -0.30),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color="#888888", lw=0.7,
                            shrinkA=0, shrinkB=0),
        )
        ax.text(0.02, -0.34, "less noise",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=FONT["small"] - 1, color="#666666", style="italic")
        ax.text(0.98, -0.34, "more noise",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=FONT["small"] - 1, color="#666666", style="italic")
    ax.set_title(title, pad=4)
    if show_legend:
        ax.legend(loc="lower left", ncol=2, frameon=True, framealpha=0.92,
                  edgecolor="none", columnspacing=0.7,
                  handletextpad=0.4, fontsize=FONT["small"])


def plot_auc_bars(ax):
    """Grouped bars: Selected vs Random accuracy AUC for raw and encoding."""
    ms_labels = []
    raw_rand, raw_sel, enc_rand, enc_sel = [], [], [], []
    raw_rand_err, raw_sel_err, enc_rand_err, enc_sel_err = [], [], [], []
    colors = []

    for ms in MODEL_SET_ORDER:
        df = load_discriminability(ms)
        auc_df = load_auc(ms)
        if df.empty or auc_df.empty:
            continue

        raw_row = auc_df[auc_df["track"] == "raw"]
        enc_rows = auc_df[auc_df["track"].isin(ENCODING_TRACKS)]
        if raw_row.empty or enc_rows.empty:
            continue
        raw_row = raw_row.iloc[0]

        ms_labels.append(MODEL_SET_DISPLAY_SHORT[ms])
        colors.append(model_set_color(ms))
        raw_sel.append(1 - raw_row["selected_auc"])
        raw_rand.append(1 - raw_row["random_auc_mean"])
        raw_sel_err.append(raw_row["selected_auc_mc_std"])
        raw_rand_err.append(raw_row["random_auc_subset_std"])

        n_subj = len(enc_rows)
        enc_sel.append(1 - enc_rows["selected_auc"].mean())
        enc_rand.append(1 - enc_rows["random_auc_mean"].mean())
        enc_sel_err.append(
            float(np.sqrt((enc_rows["selected_auc_mc_std"] ** 2).mean() / n_subj))
        )
        enc_rand_err.append(
            float(np.sqrt((enc_rows["random_auc_subset_std"] ** 2).mean() / n_subj))
        )

    x = np.arange(len(ms_labels), dtype=float)
    width = 0.19
    err_kw = dict(ecolor="0.3", capsize=2, elinewidth=0.8)

    ax.bar(
        x - 1.5 * width,
        raw_rand,
        width,
        yerr=raw_rand_err,
        color="#DDDDDD",
        alpha=0.85,
        edgecolor="#666666",
        linewidth=0.5,
        hatch="//",
        label="Random (raw)",
        error_kw=err_kw,
        zorder=3,
    )
    ax.bar(
        x - 0.5 * width,
        raw_sel,
        width,
        yerr=raw_sel_err,
        color=colors,
        alpha=0.55,
        edgecolor="white",
        linewidth=0.5,
        hatch="//",
        label="Selected (raw)",
        error_kw=err_kw,
        zorder=3,
    )
    ax.bar(
        x + 0.5 * width,
        enc_rand,
        width,
        yerr=enc_rand_err,
        color="#AAAAAA",
        alpha=0.85,
        edgecolor="#666666",
        linewidth=0.5,
        label="Random (enc.)",
        error_kw=err_kw,
        zorder=3,
    )
    ax.bar(
        x + 1.5 * width,
        enc_sel,
        width,
        yerr=enc_sel_err,
        color=colors,
        alpha=0.90,
        edgecolor="white",
        linewidth=0.5,
        label="Selected (enc.)",
        error_kw=err_kw,
        zorder=3,
    )

    for i, color in enumerate(colors):
        raw_top = max(raw_rand[i] + raw_rand_err[i], raw_sel[i] + raw_sel_err[i])
        raw_pct = (raw_sel[i] - raw_rand[i]) / max(raw_rand[i], 1e-6) * 100
        ax.text(
            x[i] - width,
            raw_top + 0.025,
            f"+{raw_pct:.0f}%",
            fontsize=FONT["small"],
            ha="center",
            va="bottom",
            color=color,
            alpha=0.75,
        )

        enc_top = max(enc_rand[i] + enc_rand_err[i], enc_sel[i] + enc_sel_err[i])
        enc_pct = (enc_sel[i] - enc_rand[i]) / max(enc_rand[i], 1e-6) * 100
        ax.text(
            x[i] + width,
            enc_top + 0.025,
            f"+{enc_pct:.0f}%",
            fontsize=FONT["small"],
            ha="center",
            va="bottom",
            color=color,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(ms_labels, rotation=25, ha="right")
    ax.set_ylabel("Accuracy AUC (higher = better)")
    ax.set_ylim(0.35, 1.08)
    ax.set_title("AUC comparison", pad=4)
    ax.grid(axis="y", alpha=0.35, zorder=0)
    ax.legend(loc="lower left", ncol=2, frameon=True, framealpha=0.88,
              edgecolor="none", columnspacing=0.6, handlelength=1.2,
              handletextpad=0.3, borderpad=0.25, labelspacing=0.25,
              fontsize=max(6, FONT["small"] - 3))


def main():
    fig = plt.figure(figsize=(W_DOUBLE, 4.9))
    # Raw-feature panel is near-saturated for all model sets; give it the
    # smallest column. Encoding panel and AUC summary get the room.
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.5], wspace=0.42,
                          left=0.06, right=0.99, top=0.92, bottom=0.20)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[0, 2])

    plot_curves(ax_a, "raw",      "Raw feature space",        show_legend=False,
                 show_noise_axis=True)
    ax_a.set_ylabel("Model recovery accuracy")
    plot_curves(ax_b, "encoding", "Encoding (predicted brain) space",
                 show_legend=True, show_noise_axis=False)
    plt.setp(ax_b.get_yticklabels(), visible=False)

    plot_auc_bars(ax_c)

    add_panel_label(ax_a, "a")
    add_panel_label(ax_b, "b")
    add_panel_label(ax_c, "c", x=-0.18)

    for out_dir in [FIGURES_DIR, INSILICO_CURVE_DIR, MANUSCRIPT_FIGURES_DIR]:
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ["pdf", "png"]:
            out = out_dir / f"insilico_evaluation_unique_improved.{ext}"
            fig.savefig(out, dpi=DPI)
            print(f"Saved {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
