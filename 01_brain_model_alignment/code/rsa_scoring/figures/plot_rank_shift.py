#!/usr/bin/env python3
"""
Baseline → cstim rank-shift delta plot.

Layout mirrors brain_alignment.py:
  - Left panel: all_models (wide)
  - Right: 2×2 grid for sota / training_objective / architecture / dataset

Each panel:
  - X-axis: models sorted by mRSA Δrank descending (biggest improvers left)
  - Y-axis: Δrank = mean_baseline_rank − mean_cstim_rank
             positive = model improved on cstim (rank number decreased)
  - mRSA: solid filled bars   (matches brain_alignment filled dots)
  - fRSA: outlined hatched bars (matches open dots)
  - Error bars: ±1 std across subjects
  - Reference line at Δ=0
  - Stats annotation: ρ_base / ρ_cstim / ρ_b↔c

Outputs:
    figures/rank_shift.pdf/png

Usage:
    python plot_rank_shift.py
"""

import sys
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))
from itertools import combinations

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))
from cstims.paper import config

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats

from cstims.paper.style_improved import apply_style, FONT, DPI, W_DOUBLE

apply_style()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBJECTS   = config.SUBJECTS
MODEL_SETS = config.MODEL_SETS
DN         = config.MODEL_DISPLAY_NAMES

RSA_DATA_DIR = config.RSA_DATA_DIR
FIGURES_DIR  = Path(__file__).resolve().parent

# Match brain_alignment color scheme: red = cstim-side, blue = baseline-side
# Here we repurpose: mRSA = darker, fRSA = lighter
COLOR_MRSA = "#2471A3"   # solid blue
COLOR_FRSA = "#85C1E9"   # lighter blue, hatched

SHORT_DISPLAY_NAMES = {
    "torchvision_vgg16_imagenet1k_v1":                  "VGG-16",
    "torchvision_resnet50_imagenet1k_v1":               "ResNet-50",
    "torchvision_convnext_base_imagenet1k_v1":          "ConvNeXt-B",
    "torchvision_vit_l_16_imagenet1k_v1":               "ViT-L/16",
    "cornet_s":                                         "CORnet-S",
    "vissl_resnet50_supervised":                        "Supervised",
    "vissl_resnet50_barlowtwins":                       "BarlowTwins",
    "vissl_resnet50_mocov2":                            "MoCoV2",
    "vicreg_resnet50":                                  "VICReg",
    "robustness_imagenet_l2_eps3":                      "Robust-L2",
    "slip_vit_l_slip":                                  "SLIP",
    "slip_vit_l_simclr":                                "SimCLR-ViT",
    "timm_vit_large_patch14_clip_224_laion2b":          "CLIP-L2B",
    "dinov2_vitl14":                                    "DINOv2",
    "openclip_vit_so400m_14_siglip_webli":              "SigLIP",
    "openclip_vit_l_14_quickgelu_metaclip_400m":        "MC-400M",
    "openclip_vit_l_14_quickgelu_metaclip_fullcc":      "MC-Full",
    "timm_vit_large_patch14_clip_224_dfn2b":            "DFN-2B",
    "timm_vit_large_patch14_clip_quickgelu_224_openai": "CLIP-OAI",
    "openclip_vit_l_14_laion400m_e31":                  "CLIP-L400",
}

TITLE_MAP = {
    "all_models":        "All Models",
    "sota":              "State of the Art",
    "training_objective":"Training Objective",
    "architecture":      "Architecture",
    "dataset":           "Dataset",
}

PANEL_LABELS = list("abcde")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scores(method: str) -> pd.DataFrame:
    filename = f"{method}_scores.csv"
    dfs = []
    for subject in SUBJECTS:
        path = RSA_DATA_DIR / subject / filename
        if path.exists():
            dfs.append(pd.read_csv(path))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ---------------------------------------------------------------------------
# Rank helpers
# ---------------------------------------------------------------------------

def per_subject_ranks(df: pd.DataFrame, model_set: str, score_col: str,
                      stimulus_type: str) -> pd.DataFrame:
    """Returns DataFrame[subject, model, rank]."""
    models = MODEL_SETS[model_set]
    sub_df = df[
        (df["model_set"] == model_set) &
        (df["stimulus_type"] == stimulus_type) &
        (df["bootstrap_idx"] == 0) &
        (df["model"].isin(models))
    ]
    rows = []
    for subject in SUBJECTS:
        s_df = sub_df[sub_df["subject"] == subject]
        if s_df.empty:
            continue
        ranked = s_df.sort_values(score_col, ascending=False).reset_index(drop=True)
        for rank, row in enumerate(ranked.itertuples(), 1):
            rows.append({"subject": subject, "model": row.model, "rank": rank})
    return pd.DataFrame(rows)


def cross_subject_rho(ranks_df: pd.DataFrame) -> float:
    return float(np.mean(pairwise_rhos(ranks_df))) if len(pairwise_rhos(ranks_df)) else np.nan


def pairwise_rhos(ranks_df: pd.DataFrame) -> np.ndarray:
    """Per-subject-pair Spearman correlations (used for statistical testing)."""
    subjects = sorted(ranks_df["subject"].unique())
    corrs = []
    for s1, s2 in combinations(subjects, 2):
        r1 = ranks_df[ranks_df["subject"] == s1].set_index("model")["rank"]
        r2 = ranks_df[ranks_df["subject"] == s2].set_index("model")["rank"]
        common = r1.index.intersection(r2.index)
        if len(common) < 3:
            continue
        rho, _ = stats.spearmanr(r1[common], r2[common])
        corrs.append(rho)
    return np.array(corrs)


def test_rho_difference(base_df: pd.DataFrame, cstim_df: pd.DataFrame):
    """Wilcoxon signed-rank test on Fisher-z transformed pairwise ρ values.

    Returns (statistic, p_value). Uses the same subject pairs from both conditions.
    """
    subjects = sorted(set(base_df["subject"].unique()) & set(cstim_df["subject"].unique()))
    z_base, z_cstim = [], []
    for s1, s2 in combinations(subjects, 2):
        for df, bucket in [(base_df, z_base), (cstim_df, z_cstim)]:
            r1 = df[df["subject"] == s1].set_index("model")["rank"]
            r2 = df[df["subject"] == s2].set_index("model")["rank"]
            common = r1.index.intersection(r2.index)
            if len(common) < 3:
                bucket.append(np.nan)
                continue
            rho, _ = stats.spearmanr(r1[common], r2[common])
            # Fisher z-transform (clip to avoid ±inf)
            bucket.append(np.arctanh(np.clip(rho, -0.9999, 0.9999)))
    z_base  = np.array(z_base)
    z_cstim = np.array(z_cstim)
    mask = ~(np.isnan(z_base) | np.isnan(z_cstim))
    if mask.sum() < 2:
        return np.nan, np.nan
    stat, p = stats.wilcoxon(z_cstim[mask], z_base[mask], alternative="two-sided")
    return float(stat), float(p)


def base_vs_cstim_rho(base_df: pd.DataFrame, cstim_df: pd.DataFrame) -> float:
    b = base_df.groupby("model")["rank"].mean()
    c = cstim_df.groupby("model")["rank"].mean()
    common = b.index.intersection(c.index)
    if len(common) < 3:
        return np.nan
    rho, _ = stats.spearmanr(b[common], c[common])
    return float(rho)


def compute_deltas(df: pd.DataFrame, model_set: str, score_col: str):
    """Returns (mean_delta, std_delta, base_df, cstim_df) keyed by model."""
    base_df  = per_subject_ranks(df, model_set, score_col, "vicco")
    cstim_df = per_subject_ranks(df, model_set, score_col, "controversial")
    if base_df.empty or cstim_df.empty:
        return {}, {}, base_df, cstim_df

    models = list(MODEL_SETS[model_set])
    delta_by = {m: [] for m in models}
    for subject in SUBJECTS:
        b = base_df[base_df["subject"] == subject].set_index("model")["rank"]
        c = cstim_df[cstim_df["subject"] == subject].set_index("model")["rank"]
        for model in b.index.intersection(c.index):
            delta_by[model].append(float(b[model]) - float(c[model]))

    models = [m for m in models if delta_by[m]]
    mean_d = {m: np.mean(delta_by[m]) for m in models}
    std_d  = {m: np.std(delta_by[m], ddof=0) for m in models}
    return mean_d, std_d, base_df, cstim_df


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------

def draw_panel(ax, df_mrsa: pd.DataFrame, df_frsa: pd.DataFrame,
               model_set: str,
               title: str = "",
               panel_label: str = None,
               show_ylabel: bool = True,
               show_legend: bool = False,
               use_short_names: bool = False):

    mean_m, std_m, base_m, cstim_m = compute_deltas(df_mrsa, model_set, "wrsa_transfer")
    mean_f, std_f, base_f, cstim_f = compute_deltas(df_frsa, model_set, "crsa")

    models = sorted(
        set(mean_m) | set(mean_f),
        key=lambda m: mean_m.get(m, mean_f.get(m, 0)),
        reverse=True,   # biggest improvers on left
    )
    if not models:
        ax.set_visible(False)
        return

    n = len(models)
    x = np.arange(n)
    bar_w = 0.35
    offset = bar_w / 2 + 0.02

    # --- bars ---
    for i, model in enumerate(models):
        # mRSA: solid, left of centre
        if model in mean_m:
            ax.bar(x[i] - offset, mean_m[model], width=bar_w,
                   yerr=std_m.get(model, 0),
                   color=COLOR_MRSA, alpha=0.85,
                   error_kw=dict(linewidth=0.7, capsize=2, ecolor="#444"),
                   zorder=3)
        # fRSA: hatched, right of centre
        if model in mean_f:
            ax.bar(x[i] + offset, mean_f[model], width=bar_w,
                   yerr=std_f.get(model, 0),
                   color=COLOR_FRSA, alpha=0.75, hatch="////",
                   edgecolor=COLOR_FRSA,
                   error_kw=dict(linewidth=0.7, capsize=2, ecolor="#444"),
                   zorder=3)

    # zero reference
    ax.axhline(0, color="#333", linewidth=0.8, zorder=4)

    # --- axes ---
    ax.set_xticks(x)
    name_fn = SHORT_DISPLAY_NAMES.get if use_short_names else DN.get
    ax.set_xticklabels(
        [SHORT_DISPLAY_NAMES.get(m, DN.get(m, m)) if use_short_names
         else DN.get(m, m) for m in models],
        rotation=45, ha="right", fontsize=FONT["tick"],
    )
    ax.set_xlim(-0.7, n - 0.3)

    if show_ylabel:
        ax.set_ylabel("Δ rank  (+ = improved on cstim)", fontsize=FONT["axis_label"])
    ax.tick_params(axis="y", labelsize=FONT["tick"])
    ax.spines["bottom"].set_visible(True)
    ax.grid(axis="y", alpha=0.22, linewidth=0.4, zorder=0)

    # --- title ---
    if title:
        ax.set_title(title, fontweight="bold", pad=4, fontsize=FONT["title"])

    # --- panel label ---
    if panel_label:
        ax.text(-0.06, 1.10, panel_label, transform=ax.transAxes,
                fontsize=FONT["panel_label"], fontweight="bold", va="top")

    # --- stats annotation ---
    parts = []
    if not base_m.empty and not cstim_m.empty:
        rb  = cross_subject_rho(base_m)
        rc  = cross_subject_rho(cstim_m)
        rbc = base_vs_cstim_rho(base_m, cstim_m)
        parts.append(f"mixed RSA  ρ_base={rb:.2f}  ρ_cstim={rc:.2f}  ρ_b↔c={rbc:.2f}")
    if not base_f.empty and not cstim_f.empty:
        rb  = cross_subject_rho(base_f)
        rc  = cross_subject_rho(cstim_f)
        rbc = base_vs_cstim_rho(base_f, cstim_f)
        parts.append(f"fixed RSA  ρ_base={rb:.2f}  ρ_cstim={rc:.2f}  ρ_b↔c={rbc:.2f}")
    if parts:
        ax.text(0.02, 0.02, "\n".join(parts),
                transform=ax.transAxes,
                ha="left", va="bottom",
                fontsize=FONT["small"] - 1,
                color="#555",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8),
                zorder=10)

    # --- legend ---
    if show_legend:
        handles = [
            mpatches.Patch(facecolor=COLOR_MRSA, alpha=0.85, label="mixed RSA"),
            mpatches.Patch(facecolor=COLOR_FRSA, alpha=0.75, hatch="////",
                           edgecolor=COLOR_FRSA, label="fixed RSA"),
        ]
        ax.legend(handles=handles, loc="upper right",
                  fontsize=FONT["legend"], frameon=False,
                  ncol=2, handlelength=1.2)


# ---------------------------------------------------------------------------
# Rho summary figure
# ---------------------------------------------------------------------------

RHO_COLORS = {
    "base":        "#E67E22",   # orange  — baseline consistency
    "cstim":       "#27AE60",   # green   — cstim consistency
    "b2c_raw":     "#8E44AD",   # purple  — base↔cstim (raw)
    "b2c_corrected": "#C39BD3", # light purple — base↔cstim (SB-corrected)
}
RHO_LABELS = {
    "base":          r"$\rho_\mathrm{base}$",
    "cstim":         r"$\rho_\mathrm{cstim}$",
    "b2c_raw":       r"$\rho_{b \leftrightarrow c}$",
    "b2c_corrected": r"$\rho_{b \leftrightarrow c}^{\mathrm{corr}}$",
}

N_SUBJECTS = 5  # sub-01, sub-03, sub-05, sub-06, sub-07


def spearman_brown(r_pair: float, n: int) -> float:
    """Spearman-Brown reliability of the mean of n replicates given pairwise reliability r_pair."""
    if np.isnan(r_pair) or r_pair <= -1 / (n - 1):
        return np.nan
    return n * r_pair / (1 + (n - 1) * r_pair)

_MODEL_SET_ORDER = ["all_models", "sota", "training_objective", "architecture", "dataset"]
_MS_SHORT = {
    "all_models":         "All",
    "sota":               "SOTA",
    "training_objective": "Train.",
    "architecture":       "Arch.",
    "dataset":            "Dataset",
}


def compute_rho_summary(df: pd.DataFrame, score_col: str):
    """Return (summary_df, tests_dict).

    summary_df columns: [model_set, rho_base, rho_cstim, rho_b2c_raw, rho_b2c_corrected]
      - rho_base / rho_cstim: mean pairwise cross-subject rank correlation
      - rho_b2c_raw: Spearman r between mean-baseline and mean-cstim ranks
      - rho_b2c_corrected: rho_b2c_raw / sqrt(r_sb_base * r_sb_cstim),
          where r_sb = Spearman-Brown reliability of the mean-of-N_SUBJECTS,
          capped at 1.0
    tests_dict: {model_set: (stat, p_value)} for the ρ_base vs ρ_cstim Wilcoxon test.
    """
    rows = []
    tests = {}
    for ms in _MODEL_SET_ORDER:
        base_df  = per_subject_ranks(df, ms, score_col, "vicco")
        cstim_df = per_subject_ranks(df, ms, score_col, "controversial")
        if base_df.empty or cstim_df.empty:
            continue
        stat, p = test_rho_difference(base_df, cstim_df)
        tests[ms] = (stat, p)
        rho_base  = cross_subject_rho(base_df)
        rho_cstim = cross_subject_rho(cstim_df)
        rho_b2c   = base_vs_cstim_rho(base_df, cstim_df)
        # Spearman-Brown correction: rho_b2c is computed on mean-of-N_SUBJECTS
        # ranks, so use the SB-corrected reliability of those means in denominator
        r_sb_base  = spearman_brown(rho_base,  N_SUBJECTS)
        r_sb_cstim = spearman_brown(rho_cstim, N_SUBJECTS)
        denom = np.sqrt(r_sb_base * r_sb_cstim)
        rho_b2c_corrected = float(np.clip(rho_b2c / denom, -1.0, 1.0)) if denom > 0 else np.nan
        rows.append({
            "model_set":        ms,
            "rho_base":         rho_base,
            "rho_cstim":        rho_cstim,
            "rho_b2c_raw":      rho_b2c,
            "rho_b2c_corrected": rho_b2c_corrected,
        })
    return pd.DataFrame(rows), tests


def draw_rho_figure(df_mrsa: pd.DataFrame, df_frsa: pd.DataFrame):
    """Second figure: grouped bar chart of ρ_base / ρ_cstim per model set,
    with significance brackets for ρ_cstim vs ρ_base (Wilcoxon, Fisher-z)."""
    rho_m, tests_m = compute_rho_summary(df_mrsa, "wrsa_transfer")
    rho_f, tests_f = compute_rho_summary(df_frsa, "crsa")

    fig, axes = plt.subplots(
        1, 2,
        figsize=(W_DOUBLE * 0.70, 3.8),
        sharey=True,
    )
    fig.subplots_adjust(left=0.09, right=0.97, top=0.85, bottom=0.18, wspace=0.08)

    for ax, rho_df, tests, method_label, panel_label in zip(
        axes,
        [rho_m, rho_f],
        [tests_m, tests_f],
        ["mixed RSA", "fixed RSA"],
        ["a", "b"],
    ):
        if rho_df.empty:
            ax.set_visible(False)
            continue

        n_sets = len(rho_df)
        x = np.arange(n_sets)
        bar_w = 0.28
        offsets = np.array([-0.5, 0.5]) * (bar_w + 0.04)
        x_base  = x + offsets[0]
        x_cstim = x + offsets[1]

        bar_tops = {}
        for off, key in zip(offsets, ["base", "cstim"]):
            vals = rho_df[f"rho_{key}"].values
            ax.bar(
                x + off, vals, width=bar_w,
                color=RHO_COLORS[key], alpha=0.85,
                label=RHO_LABELS[key], zorder=3,
            )
            bar_tops[key] = vals
            for xi, v in zip(x + off, vals):
                if not np.isnan(v):
                    ax.text(xi, v + 0.015, f"{v:.2f}",
                            ha="center", va="bottom",
                            fontsize=FONT["small"] - 1, color="#333")

        ax.axhline(0, color="#444", linewidth=0.7, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_MS_SHORT.get(ms, ms) for ms in rho_df["model_set"]],
            fontsize=FONT["tick"],
        )
        ax.set_ylim(-0.05, 1.25)
        ax.set_title(method_label, fontweight="bold", pad=4, fontsize=FONT["title"])
        ax.grid(axis="y", alpha=0.22, linewidth=0.4, zorder=0)
        ax.tick_params(axis="y", labelsize=FONT["tick"])

        ax.text(-0.08, 1.10, panel_label, transform=ax.transAxes,
                fontsize=FONT["panel_label"], fontweight="bold", va="top")

        if panel_label == "a":
            ax.set_ylabel("cross-subject Spearman ρ", fontsize=FONT["axis_label"])
            ax.legend(
                loc="upper right", fontsize=FONT["legend"],
                frameon=False, ncol=2, handlelength=1.0,
                columnspacing=0.8,
            )

    return fig


def draw_b2c_figure(df_mrsa: pd.DataFrame, df_frsa: pd.DataFrame):
    """Third figure: baseline↔cstim rank correlation (raw and SB-corrected)
    per model set, separately for mRSA and fRSA.

    ρ_b↔c answers: are model rankings consistent across stimulus sets?
    The SB-corrected version divides by sqrt(r_sb_base * r_sb_cstim), where
    r_sb is the Spearman-Brown reliability of the mean-of-N_SUBJECTS ranking,
    estimated from mean pairwise cross-subject ρ. This removes attenuation
    due to noise in both rankings and gives the latent correlation.
    Corrected values are capped at 1.0.
    """
    rho_m, _ = compute_rho_summary(df_mrsa, "wrsa_transfer")
    rho_f, _ = compute_rho_summary(df_frsa, "crsa")

    fig, axes = plt.subplots(
        1, 2,
        figsize=(W_DOUBLE * 0.70, 3.8),
        sharey=True,
    )
    fig.subplots_adjust(left=0.10, right=0.97, top=0.85, bottom=0.18, wspace=0.08)

    for ax, rho_df, method_label, panel_label in zip(
        axes,
        [rho_m, rho_f],
        ["mixed RSA", "fixed RSA"],
        ["a", "b"],
    ):
        if rho_df.empty:
            ax.set_visible(False)
            continue

        n_sets = len(rho_df)
        x = np.arange(n_sets)
        bar_w = 0.28
        offsets = np.array([-0.5, 0.5]) * (bar_w + 0.04)

        for off, key in zip(offsets, ["b2c_raw", "b2c_corrected"]):
            vals = rho_df[f"rho_{key}"].values
            ax.bar(
                x + off, vals, width=bar_w,
                color=RHO_COLORS[key], alpha=0.85,
                label=RHO_LABELS[key], zorder=3,
            )
            for xi, v in zip(x + off, vals):
                if not np.isnan(v):
                    ax.text(xi, v + 0.015, f"{v:.2f}",
                            ha="center", va="bottom",
                            fontsize=FONT["small"] - 1, color="#333")

        ax.axhline(0, color="#444", linewidth=0.7, zorder=4)
        ax.axhline(1, color="#999", linewidth=0.5, linestyle=":", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_MS_SHORT.get(ms, ms) for ms in rho_df["model_set"]],
            fontsize=FONT["tick"],
        )
        ax.set_ylim(-0.05, 1.35)
        ax.set_title(method_label, fontweight="bold", pad=4, fontsize=FONT["title"])
        ax.grid(axis="y", alpha=0.22, linewidth=0.4, zorder=0)
        ax.tick_params(axis="y", labelsize=FONT["tick"])

        ax.text(-0.08, 1.10, panel_label, transform=ax.transAxes,
                fontsize=FONT["panel_label"], fontweight="bold", va="top")

        if panel_label == "a":
            ax.set_ylabel(r"$\rho_{b \leftrightarrow c}$  (Spearman)", fontsize=FONT["axis_label"])
            ax.legend(
                loc="lower right", fontsize=FONT["legend"],
                frameon=False, ncol=1, handlelength=1.0,
            )
            ax.text(
                0.02, 0.98,
                f"SB correction: n={N_SUBJECTS} subjects",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=FONT["small"] - 1, color="#666",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
            )

        # print values to terminal
        print(f"\n{method_label} — ρ_b↔c (raw / SB-corrected):")
        for _, row in rho_df.iterrows():
            print(f"  {row['model_set']:22s}  raw={row['rho_b2c_raw']:.3f}"
                  f"  corrected={row['rho_b2c_corrected']:.3f}")

    return fig


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df_mrsa = load_scores("wrsa_transfer")
    df_frsa = load_scores("crsa")

    if df_mrsa.empty or df_frsa.empty:
        print("No data found — check RSA_DATA_DIR")
        return

    fig = plt.figure(figsize=(W_DOUBLE, 7.0))
    outer = fig.add_gridspec(
        1, 2,
        width_ratios=[1.15, 0.85],
        wspace=0.12,
        left=0.06, right=0.97, top=0.90, bottom=0.20,
    )

    # Left: all_models
    ax_all = fig.add_subplot(outer[0])
    draw_panel(ax_all, df_mrsa, df_frsa, "all_models",
               title=TITLE_MAP["all_models"],
               panel_label="a",
               show_ylabel=True, show_legend=True,
               use_short_names=True)

    # Right: 2×2 grid
    gs_right = outer[1].subgridspec(2, 2, wspace=0.30, hspace=0.65)
    small_sets = ["sota", "training_objective", "architecture", "dataset"]

    for idx, model_set in enumerate(small_sets):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs_right[row, col])
        draw_panel(ax, df_mrsa, df_frsa, model_set,
                   title=TITLE_MAP[model_set],
                   panel_label=PANEL_LABELS[idx + 1],
                   show_ylabel=(col == 0),
                   use_short_names=False)

    for ext in ["pdf", "png"]:
        out = FIGURES_DIR / f"rank_shift.{ext}"
        fig.savefig(out, dpi=DPI)
        print(f"Saved {out}")
    plt.close(fig)

    # --- Second figure: ρ_base / ρ_cstim cross-subject consistency ---
    fig2 = draw_rho_figure(df_mrsa, df_frsa)
    for ext in ["pdf", "png"]:
        out = FIGURES_DIR / f"rank_rho_summary.{ext}"
        fig2.savefig(out, dpi=DPI)
        print(f"Saved {out}")
    plt.close(fig2)

    # --- Third figure: ρ_b↔c raw vs SB-corrected ---
    fig3 = draw_b2c_figure(df_mrsa, df_frsa)
    for ext in ["pdf", "png"]:
        out = FIGURES_DIR / f"rank_b2c_transfer.{ext}"
        fig3.savefig(out, dpi=DPI)
        print(f"Saved {out}")
    plt.close(fig3)


if __name__ == "__main__":
    main()
