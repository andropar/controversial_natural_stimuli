#!/usr/bin/env python3
"""
Plot brain alignment: per-subject dot + connecting-line visualization.

Two rows: all_models (top), 4 controlled sets (bottom).
Per model: dots at left (controversial, red) and right (baseline, blue).
Each subject = one dot; lines connect same subject's cstim-baseline pair.
mRSA = filled dots + solid lines, fRSA = open dots + dashed lines.
Mean shown as a larger horizontal tick.
Noise ceilings as tinted ribbons.

Outputs:
    figures/brain_alignment.pdf/png

Usage:
    python plot_brain_alignment.py
"""

import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))  # project root
sys.path.insert(0, str(_PAPER / "figures"))  # for style.py
import config

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from matplotlib.lines import Line2D

MODEL_SETS = config.MODEL_SETS
MODEL_DISPLAY_NAMES = config.MODEL_DISPLAY_NAMES
SUBJECTS = config.SUBJECTS
STATS_DATA_DIR = config.STATS_DATA_DIR
RSA_DATA_DIR = config.RSA_DATA_DIR
FIGURES_DIR = Path(__file__).resolve().parent
get_subject_data_dir = config.get_subject_data_dir

from style import apply_style, FONT, DPI, W_DOUBLE

apply_style()

# ---------------------------------------------------------------------------
# Colors: red = controversial, blue = baseline
# ---------------------------------------------------------------------------
COLOR_CSTIM = "#D64541"
COLOR_CSTIM_CRSA = "#B03A38"
COLOR_BASE = "#2980B9"
COLOR_BASE_CRSA = "#1F6699"

TITLE_MAP = {
    "training_objective": "Training Objective",
    "sota": "State of the Art",
    "architecture": "Architecture",
    "dataset": "Dataset",
    "all_models": "All Models",
}

SHORT_DISPLAY_NAMES = {
    "torchvision_vgg16_imagenet1k_v1": "VGG-16",
    "torchvision_resnet50_imagenet1k_v1": "ResNet-50",
    "torchvision_convnext_base_imagenet1k_v1": "ConvNeXt-B",
    "torchvision_vit_l_16_imagenet1k_v1": "ViT-L/16",
    "cornet_s": "CORnet-S",
    "vissl_resnet50_supervised": "Supervised",
    "vissl_resnet50_barlowtwins": "BarlowTwins",
    "vissl_resnet50_mocov2": "MoCoV2",
    "vicreg_resnet50": "VICReg",
    "robustness_imagenet_l2_eps3": "Robust-L2",
    "slip_vit_l_slip": "SLIP",
    "slip_vit_l_simclr": "SimCLR-ViT",
    "timm_vit_large_patch14_clip_224_laion2b": "CLIP-L2B",
    "dinov2_vitl14": "DINOv2",
    "openclip_vit_so400m_14_siglip_webli": "SigLIP",
    "openclip_vit_l_14_quickgelu_metaclip_400m": "MC-400M",
    "openclip_vit_l_14_quickgelu_metaclip_fullcc": "MC-Full",
    "timm_vit_large_patch14_clip_224_dfn2b": "DFN-2B",
    "timm_vit_large_patch14_clip_quickgelu_224_openai": "CLIP-OAI",
    "openclip_vit_l_14_laion400m_e31": "CLIP-L400",
}

PANEL_LABELS = list("abcdefghij")
VERSA_EXCLUDED = set()  # Previously excluded vicreg_resnet50

# Map display method labels to CSV method keys
_METHOD_CSV_KEY = {"mRSA": "wrsa_transfer", "fRSA": "crsa"}


def _sig_stars(p):
    if p is None or np.isnan(p):
        return ""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return ""


def _lookup_perm(perm_df, model_set, method_label, metric="median_pairwise_diff"):
    """Look up observed spread ratio and BH-FDR q-value from permutation test results.

    Default metric switched from CV (mean-confounded) to median pairwise diff
    (shift-invariant, robust to outliers). Uses BH-FDR q-values if present in CSV.
    """
    if perm_df is None:
        return None, None
    csv_method = _METHOD_CSV_KEY.get(method_label, method_label)
    match = perm_df[
        (perm_df["model_set"] == model_set)
        & (perm_df["method"] == csv_method)
        & (perm_df["metric"] == metric)
    ]
    if match.empty:
        return None, None
    # Prefer BH-FDR corrected q-value if the column exists, else fall back to raw p
    sig_col = "q_bh" if "q_bh" in perm_df.columns else "p_perm"
    return float(match["observed_ratio"].iloc[0]), float(match[sig_col].iloc[0])


# ===========================================================================
# Data loading
# ===========================================================================

def load_all_scores() -> pd.DataFrame:
    dfs = []
    score_files = {
        "crsa_scores.csv": ("crsa", "fRSA"),
        "wrsa_transfer_scores.csv": ("wrsa_transfer", "mRSA"),
    }
    for subject in SUBJECTS:
        data_dir = RSA_DATA_DIR / subject
        for filename, (col_name, method_name) in score_files.items():
            path = data_dir / filename
            if path.exists():
                df = pd.read_csv(path)
                df = df.rename(columns={col_name: "score"})
                df["method"] = method_name
                if "subject" not in df.columns:
                    df["subject"] = subject
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_noise_ceilings() -> pd.DataFrame:
    return pd.read_csv(STATS_DATA_DIR / "rdm_noise_ceilings.csv")


def load_between_subject_nc() -> pd.DataFrame:
    path = _PAPER / "03_statistics" / "data" / "between_subject_noise_ceilings.csv"
    return pd.read_csv(path)


# ===========================================================================
# Drawing
# ===========================================================================

def draw_box(ax, x, mean, sem, width=0.30, filled=True, color="#D64541"):
    """Colin-style box: midline +/- SEM."""
    lo = mean - sem
    hi = mean + sem
    h = hi - lo
    if h < 1e-4:
        h = 0.003
        lo = mean - h / 2

    ec = color
    fc = color if filled else "white"
    lw = 0.6 if filled else 0.8

    rect = mpatches.FancyBboxPatch(
        (x - width / 2, lo), width, h,
        boxstyle="square,pad=0",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        alpha=0.75 if filled else 0.85,
        zorder=3 if filled else 4,
    )
    ax.add_patch(rect)

    ml_color = "white" if filled else ec
    ax.hlines(mean, x - width / 2, x + width / 2,
              colors=ml_color, linewidth=0.5, zorder=5)


# ===========================================================================
# Data extraction — per-subject scores
# ===========================================================================

def get_per_subject_scores(df, model_set):
    """
    Returns:
        data: {model: {method: {stim_type: {subject: score}}}}
        subjects: sorted list of subjects
    """
    models = MODEL_SETS[model_set]
    subset = df[df["model_set"] == model_set]
    subjects = sorted(subset["subject"].unique())

    data = {}
    for model in models:
        data[model] = {}
        for method in ["mRSA", "fRSA"]:
            if method == "mRSA" and model in VERSA_EXCLUDED:
                continue
            data[model][method] = {}
            m_sub = subset[(subset["model"] == model) & (subset["method"] == method)]

            for stim_type in ["controversial", "vicco"]:
                scores = {}
                for subj in subjects:
                    v = m_sub[(m_sub["subject"] == subj) &
                              (m_sub["stimulus_type"] == stim_type)]["score"]
                    if len(v) > 0:
                        scores[subj] = v.mean()
                if scores:
                    data[model][method][stim_type] = scores

    # Filter to models with at least some data
    data = {m: d for m, d in data.items() if d}
    return data, subjects


# ===========================================================================
# Panel plotting helpers
# ===========================================================================

def compute_model_order(df, model_set):
    """Sort models by mRSA controversial mean (descending). Used as fixed order across rows."""
    data, _ = get_per_subject_scores(df, model_set)
    def key(m):
        if "mRSA" in data[m] and "controversial" in data[m]["mRSA"]:
            return np.mean(list(data[m]["mRSA"]["controversial"].values()))
        return -999
    return sorted(data.keys(), key=key, reverse=True)


def _get_subject_nc(subj, model_set, stim_type, method, nc_df, bs_nc_df):
    """Per-subject noise ceiling scalar for normalisation."""
    if method == "mRSA":
        rows = nc_df[(nc_df["group"] == model_set) &
                     (nc_df["stimulus_type"] == stim_type) &
                     (nc_df["subject"] == subj)]
        if len(rows) == 0:
            rows = nc_df[(nc_df["group"] == "vicco") & (nc_df["subject"] == subj)]
        nc = rows["noise_ceiling_spearman"].mean()
        return np.sqrt(max(nc, 1e-6))
    else:  # fRSA
        rows = bs_nc_df[(bs_nc_df["group"] == model_set) &
                        (bs_nc_df["stimulus_type"] == stim_type) &
                        (bs_nc_df["subject"] == subj)]
        if len(rows) == 0:
            rows = bs_nc_df[(bs_nc_df["group"] == "vicco") & (bs_nc_df["subject"] == subj)]
        return max(rows["nc_mid"].mean(), 1e-6)


def plot_panel(ax, df, model_set, nc_df,
               title="", show_ylabel=True, show_legend=False,
               use_short_names=False, method=None,
               bs_nc_df=None, ylim=None,
               fixed_order=None, show_xticklabels=True,
               normalized=False, perm_df=None):
    """
    Per-subject dots with connecting lines.
    Left = controversial (red), right = baseline (blue).
    method: "mRSA" or "fRSA" — one method per panel.
    bs_nc_df: between-subject NC DataFrame (used for fRSA ribbons).
    ylim: shared (ymin, ymax) across panels; computed automatically if None.
    """
    data, subjects = get_per_subject_scores(df, model_set)
    if not data:
        ax.set_visible(False)
        return

    is_mrsa = (method == "mRSA")

    # Use caller-provided order (fixed across rows) or sort locally
    if fixed_order is not None:
        order = [m for m in fixed_order if m in data]
    else:
        def sort_key(m):
            if method in data[m] and "controversial" in data[m][method]:
                return np.mean(list(data[m][method]["controversial"].values()))
            return -999
        order = sorted(data.keys(), key=sort_key, reverse=True)
    n = len(order)

    # --- Noise ceiling: ribbon (raw) or single line at 1 (normalized) ---
    if normalized:
        ax.axhline(1.0, color="#444444", linewidth=0.6, alpha=0.5, zorder=0, linestyle="-")
    elif method == "fRSA" and bs_nc_df is not None:
        # Between-subject Kriegeskorte NC: ribbon spans mean(nc_lower)–mean(nc_upper)
        cstim_nc = bs_nc_df[(bs_nc_df["group"] == model_set) &
                             (bs_nc_df["stimulus_type"] == "controversial")]
        if len(cstim_nc) == 0:
            cstim_nc = bs_nc_df[bs_nc_df["group"] == "vicco"]
        nc_cstim_lo, nc_cstim_hi = cstim_nc["nc_lower"].mean(), cstim_nc["nc_upper"].mean()

        vicco_nc = bs_nc_df[bs_nc_df["group"] == "vicco"]
        nc_vicco_lo, nc_vicco_hi = vicco_nc["nc_lower"].mean(), vicco_nc["nc_upper"].mean()

        ax.axhspan(nc_cstim_lo, nc_cstim_hi, color=COLOR_CSTIM, alpha=0.08, zorder=0, linewidth=0)
        ax.axhline((nc_cstim_lo + nc_cstim_hi) / 2, color=COLOR_CSTIM, linewidth=0.4, alpha=0.3, zorder=0)
        ax.axhspan(nc_vicco_lo, nc_vicco_hi, color=COLOR_BASE, alpha=0.08, zorder=0, linewidth=0)
        ax.axhline((nc_vicco_lo + nc_vicco_hi) / 2, color=COLOR_BASE, linewidth=0.4, alpha=0.3, zorder=0)
    else:
        # Split-half Spearman-Brown NC: ceiling = sqrt(ρ_xx)
        nc_cstim = nc_df[(nc_df["group"] == model_set) & (nc_df["stimulus_type"] == "controversial")]
        if len(nc_cstim) > 0:
            nc_cstim_vals = np.sqrt(nc_cstim.set_index("subject")["noise_ceiling_spearman"])
        else:
            nc_cstim_vals = np.sqrt(nc_df[nc_df["group"] == "vicco"]
                                    .groupby("subject")["noise_ceiling_spearman"].mean())
        nc_cstim_mean = nc_cstim_vals.mean()
        nc_cstim_sem = nc_cstim_vals.std(ddof=1) / np.sqrt(len(nc_cstim_vals)) if len(nc_cstim_vals) > 1 else 0

        nc_vicco_vals = np.sqrt(nc_df[nc_df["group"] == "vicco"]
                                .groupby("subject")["noise_ceiling_spearman"].mean())
        nc_vicco_mean = nc_vicco_vals.mean()
        nc_vicco_sem = nc_vicco_vals.std(ddof=1) / np.sqrt(len(nc_vicco_vals)) if len(nc_vicco_vals) > 1 else 0

        ax.axhspan(nc_cstim_mean - nc_cstim_sem, nc_cstim_mean + nc_cstim_sem,
                   color=COLOR_CSTIM, alpha=0.08, zorder=0, linewidth=0)
        ax.axhline(nc_cstim_mean, color=COLOR_CSTIM, linewidth=0.4, alpha=0.3, zorder=0)
        ax.axhspan(nc_vicco_mean - nc_vicco_sem, nc_vicco_mean + nc_vicco_sem,
                   color=COLOR_BASE, alpha=0.08, zorder=0, linewidth=0)
        ax.axhline(nc_vicco_mean, color=COLOR_BASE, linewidth=0.4, alpha=0.3, zorder=0)

    # --- Plot per model ---
    x = np.arange(n)
    offset = 0.20
    box_w = 0.30
    dot_size = 8

    for i, m in enumerate(order):
        md = data[m]
        if method not in md:
            continue

        cstim_scores = md[method].get("controversial", {})
        base_scores = md[method].get("vicco", {})

        # Optionally normalise per subject to their NC
        if normalized:
            cstim_scores = {s: v / _get_subject_nc(s, model_set, "controversial", method, nc_df, bs_nc_df)
                            for s, v in cstim_scores.items()}
            base_scores = {s: v / _get_subject_nc(s, "vicco", "vicco", method, nc_df, bs_nc_df)
                           for s, v in base_scores.items()}

        # --- Boxes (mean ± SEM) ---
        if cstim_scores:
            c_vals = np.array(list(cstim_scores.values()))
            draw_box(ax, x[i] - offset, c_vals.mean(),
                     c_vals.std(ddof=1) / np.sqrt(len(c_vals)) if len(c_vals) > 1 else 0,
                     width=box_w, filled=is_mrsa, color=COLOR_CSTIM)

        if base_scores:
            b_vals = np.array(list(base_scores.values()))
            draw_box(ax, x[i] + offset, b_vals.mean(),
                     b_vals.std(ddof=1) / np.sqrt(len(b_vals)) if len(b_vals) > 1 else 0,
                     width=box_w, filled=is_mrsa, color=COLOR_BASE)

        # --- Per-subject dots + connecting lines ---
        for subj in subjects:
            c_val = cstim_scores.get(subj)
            b_val = base_scores.get(subj)

            if c_val is not None and b_val is not None:
                ls = "-" if is_mrsa else "--"
                ax.plot([x[i] - offset, x[i] + offset], [c_val, b_val],
                        color="#999999", linewidth=0.3, linestyle=ls, alpha=0.4, zorder=2)

            mk = "o" if is_mrsa else "D"
            ms_scale = 1.0 if is_mrsa else 1.2
            if c_val is not None:
                ax.scatter(x[i] - offset, c_val, s=dot_size * ms_scale,
                           facecolors=COLOR_CSTIM if is_mrsa else "none",
                           edgecolors=COLOR_CSTIM,
                           linewidths=0.4 if is_mrsa else 0.6,
                           zorder=6, marker=mk, alpha=0.85 if is_mrsa else 1.0)
            if b_val is not None:
                ax.scatter(x[i] + offset, b_val, s=dot_size * ms_scale,
                           facecolors=COLOR_BASE if is_mrsa else "none",
                           edgecolors=COLOR_BASE,
                           linewidths=0.4 if is_mrsa else 0.6,
                           zorder=6, marker=mk, alpha=0.85 if is_mrsa else 1.0)

    # --- X-axis labels ---
    ax.set_xticks(x)
    if use_short_names:
        labels = [SHORT_DISPLAY_NAMES.get(m, MODEL_DISPLAY_NAMES.get(m, m)) for m in order]
    else:
        labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in order]
    if show_xticklabels:
        ax.set_xticklabels(labels, rotation=45, ha="right")
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

    if show_ylabel:
        ax.set_ylabel("NC-normalised score" if normalized else "Score ($r_s$)")
    else:
        ax.yaxis.set_visible(False)
        ax.spines["left"].set_visible(False)

    if title:
        ax.set_title(title, fontweight="bold", pad=4)

    # Y-limits
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        all_vals = []
        for m in order:
            md = data[m]
            if method in md:
                for st in ["controversial", "vicco"]:
                    if st in md[method]:
                        all_vals.extend(md[method][st].values())
        ax.set_ylim(0, max(all_vals) * 1.08 if all_vals else 1.0)

    ax.set_xlim(-0.6, n - 0.4)

    # --- Spread ratio inset (controversial / baseline median pairwise diff) ---
    # Median pairwise diff is shift-invariant and robust to outliers.
    if perm_df is not None and method is not None:
        ratio, _ = _lookup_perm(perm_df, model_set, method, "median_pairwise_diff")
        if ratio is not None:
            ax.text(
                0.97, 0.03,
                f"spread ratio = {ratio:.2f}",
                transform=ax.transAxes, fontsize=FONT["annotation"],
                ha="right", va="bottom", color="#444444",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="none", alpha=0.85),
                zorder=10,
            )

    # --- Legend ---
    if show_legend:
        mk = "o" if is_mrsa else "D"
        lw = 0.4 if is_mrsa else 0.6
        nc_label = "Noise ceiling" if normalized else "Noise ceiling (contr.)"
        handles = [
            Line2D([0], [0], marker=mk, color="none",
                   markerfacecolor=COLOR_CSTIM if is_mrsa else "none",
                   markeredgecolor=COLOR_CSTIM, markersize=4,
                   markeredgewidth=lw, label="Controversial stimuli"),
            Line2D([0], [0], marker=mk, color="none",
                   markerfacecolor=COLOR_BASE if is_mrsa else "none",
                   markeredgecolor=COLOR_BASE, markersize=4,
                   markeredgewidth=lw, label="Baseline stimuli"),
        ]
        if normalized:
            handles.append(Line2D([0], [0], color="#444444", linewidth=0.6,
                                  alpha=0.5, label="Noise ceiling"))
        else:
            handles += [
                mpatches.Patch(facecolor=COLOR_CSTIM, alpha=0.20,
                               edgecolor="none", label="Noise ceiling (contr.)"),
                mpatches.Patch(facecolor=COLOR_BASE, alpha=0.20,
                               edgecolor="none", label="Noise ceiling (baseline)"),
            ]
        ax.legend(handles=handles, loc="upper right", frameon=True,
                  framealpha=0.95, edgecolor="none", ncol=2,
                  columnspacing=0.6, handletextpad=0.3, handlelength=1.2)


# ===========================================================================
# Shared y-limit computation
# ===========================================================================

def compute_row_ylim(df, all_sets, method, nc_df, bs_nc_df, normalized=False):
    """Compute a shared y-limit across all panels in one method row."""
    if normalized:
        # Gather NC-normalised values; ceiling is 1 by definition
        all_vals = [1.0]
        for ms in all_sets:
            data, subjects = get_per_subject_scores(df, ms)
            for m, md in data.items():
                if method not in md:
                    continue
                for subj in subjects:
                    c = md[method].get("controversial", {}).get(subj)
                    b = md[method].get("vicco", {}).get(subj)
                    if c is not None:
                        nc = _get_subject_nc(subj, ms, "controversial", method, nc_df, bs_nc_df)
                        all_vals.append(c / nc)
                    if b is not None:
                        nc = _get_subject_nc(subj, "vicco", "vicco", method, nc_df, bs_nc_df)
                        all_vals.append(b / nc)
        return (0, max(all_vals) * 1.08)

    all_vals = []
    for ms in all_sets:
        data, _ = get_per_subject_scores(df, ms)
        for md in data.values():
            if method in md:
                for st in ["controversial", "vicco"]:
                    if st in md[method]:
                        all_vals.extend(md[method][st].values())

    # Add NC upper bounds so the ribbon is never clipped
    if method == "fRSA" and bs_nc_df is not None:
        all_vals.extend(bs_nc_df["nc_upper"].values)
    else:
        all_vals.extend(np.sqrt(nc_df["noise_ceiling_spearman"].clip(0).values))

    return (0, max(all_vals) * 1.08) if all_vals else (0, 1.0)


# ===========================================================================
# Figure assembly helpers
# ===========================================================================

def _build_figure(df, nc_df, bs_nc_df, normalized=False, perm_df=None):
    """
    Core 2-row figure builder.
    normalized=False → brain_alignment (raw scores, ribbons)
    normalized=True  → brain_alignment_nc (NC-normalised, single line at 1)
    """
    ctrl_sets = ["sota", "training_objective", "architecture", "dataset"]
    all_sets = ["all_models"] + ctrl_sets
    width_ratios = [4, 1, 1, 1, 1]
    row_methods = [("mRSA", "Mixed RSA"), ("fRSA", "Fixed RSA")]

    # Fixed model order per set (based on mRSA controversial mean)
    orders = {ms: compute_model_order(df, ms) for ms in all_sets}

    # Shared y-limits per row
    ylims = {
        meth: compute_row_ylim(df, all_sets, meth, nc_df, bs_nc_df, normalized)
        for meth, _ in row_methods
    }

    fig = plt.figure(figsize=(W_DOUBLE, 7.5))
    gs = fig.add_gridspec(
        2, 5,
        width_ratios=width_ratios,
        wspace=0.06,
        hspace=0.12,
        left=0.06, right=0.98,
        top=0.93, bottom=0.14,
    )

    top, bot = 0.93, 0.14
    row_centers = [top - 0.25 * (top - bot), bot + 0.25 * (top - bot)]

    for row, (method, row_label) in enumerate(row_methods):
        is_bottom = (row == len(row_methods) - 1)

        fig.text(0.005, row_centers[row], row_label,
                 va="center", ha="left", fontsize=8, fontweight="bold",
                 rotation=90, transform=fig.transFigure)

        ylim = ylims[method]

        ax_all = fig.add_subplot(gs[row, 0])
        plot_panel(ax_all, df, "all_models", nc_df,
                   title="All Models" if row == 0 else "",
                   show_ylabel=True, show_legend=(row == 0),
                   use_short_names=True, method=method,
                   bs_nc_df=bs_nc_df, ylim=ylim,
                   fixed_order=orders["all_models"],
                   show_xticklabels=is_bottom,
                   normalized=normalized, perm_df=perm_df)

        for col, ms in enumerate(ctrl_sets, start=1):
            ax = fig.add_subplot(gs[row, col])
            plot_panel(ax, df, ms, nc_df,
                       title=TITLE_MAP.get(ms, ms) if row == 0 else "",
                       show_ylabel=False, method=method,
                       bs_nc_df=bs_nc_df, ylim=ylim,
                       fixed_order=orders[ms],
                       show_xticklabels=is_bottom,
                       normalized=normalized, perm_df=perm_df)

    return fig


def plot_figure(df, nc_df, bs_nc_df, perm_df=None):
    return _build_figure(df, nc_df, bs_nc_df, normalized=False, perm_df=perm_df)


def plot_figure_nc(df, nc_df, bs_nc_df, perm_df=None):
    return _build_figure(df, nc_df, bs_nc_df, normalized=True, perm_df=perm_df)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading scores...")
    df = load_all_scores()
    nc_df = load_noise_ceilings()
    bs_nc_df = load_between_subject_nc()
    perm_path = STATS_DATA_DIR / "permutation_test_results.csv"
    perm_df = pd.read_csv(perm_path) if perm_path.exists() else None
    if perm_df is None:
        print(f"  [WARN] permutation_test_results.csv not found at {perm_path}; CV inset will be omitted.")
    print(f"  {df['subject'].nunique()} subjects, {len(df)} score records")

    print("Plotting brain_alignment...")
    fig = plot_figure(df, nc_df, bs_nc_df, perm_df=perm_df)
    fig.savefig(FIGURES_DIR / "brain_alignment.pdf")
    fig.savefig(FIGURES_DIR / "brain_alignment.png", dpi=DPI)
    print("  Saved: brain_alignment.pdf/png")
    plt.close(fig)

    print("Plotting brain_alignment_nc (NC-normalised)...")
    fig_nc = plot_figure_nc(df, nc_df, bs_nc_df, perm_df=perm_df)
    fig_nc.savefig(FIGURES_DIR / "brain_alignment_nc.pdf")
    fig_nc.savefig(FIGURES_DIR / "brain_alignment_nc.png", dpi=DPI)
    print("  Saved: brain_alignment_nc.pdf/png")
    plt.close(fig_nc)

    print("Done!")


if __name__ == "__main__":
    main()
