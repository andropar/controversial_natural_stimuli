#!/usr/bin/env python3
"""
Improved brain alignment figure.

Fixes vs. original:
- Okabe-Ito palette (vermillion / blue) — colour-blind safe.
- Noise-ceiling bands rendered as neutral grey, so the controversial/baseline
  hue is reserved for data and doesn't fight the eye.
- Spread ratio promoted from corner annotation to a bold subtitle below
  each panel title; this is the headline number of the figure.
- Panel labels (a..e) for the five model sets.
- Mixed and Fixed RSA share the same colour, distinguished by marker
  fill (mixed = filled, fixed = open) and line style — so a single legend
  works.
- Fixed RSA row: y-limit chosen per row so the noise-ceiling band is visible
  without wasting empty space.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))
sys.path.insert(0, str(_PAPER / "figures"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import config
from style_improved import (
    apply_style, FONT, DPI, W_DOUBLE,
    COLOR_CSTIM, COLOR_BASELINE,
    add_panel_label,
)

apply_style()

MODEL_SETS = config.MODEL_SETS
MODEL_DISPLAY = config.MODEL_DISPLAY_NAMES
SUBJECTS = config.SUBJECTS
STATS_DATA_DIR = config.STATS_DATA_DIR
RSA_DATA_DIR = config.RSA_DATA_DIR
FIGURES_DIR = Path(__file__).resolve().parent

SHORT_NAMES = {
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

TITLE = {
    "training_objective": "Training Objective",
    "sota":               "State of the Art",
    "architecture":       "Architecture",
    "dataset":            "Dataset",
    "all_models":         "All Models",
}

# Order of panels left → right
PANEL_ORDER = ["all_models", "sota", "training_objective", "architecture", "dataset"]
NC_GRAY = "#9A9A9A"


def load_scores():
    dfs = []
    score_files = {
        "crsa_scores.csv":          ("crsa", "fRSA"),
        "wrsa_transfer_scores.csv": ("wrsa_transfer", "mRSA"),
    }
    for sub in SUBJECTS:
        d = RSA_DATA_DIR / sub
        for fname, (col, method) in score_files.items():
            p = d / fname
            if p.exists():
                df = pd.read_csv(p).rename(columns={col: "score"})
                df["method"] = method
                if "subject" not in df.columns:
                    df["subject"] = sub
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_nc():
    return pd.read_csv(STATS_DATA_DIR / "rdm_noise_ceilings.csv")


def load_bs_nc():
    p = _PAPER / "03_statistics" / "results" / "between_subject_noise_ceilings.csv"
    return pd.read_csv(p)


def load_perm():
    p = STATS_DATA_DIR / "permutation_test_results.csv"
    return pd.read_csv(p) if p.exists() else None


def get_per_subject_scores(df, model_set):
    models = MODEL_SETS[model_set]
    sub = df[df["model_set"] == model_set]
    subjects = sorted(sub["subject"].unique())
    out = {}
    for m in models:
        out[m] = {}
        for method in ("mRSA", "fRSA"):
            block = sub[(sub["model"] == m) & (sub["method"] == method)]
            for st in ("controversial", "vicco"):
                d = (block[block["stimulus_type"] == st]
                     .groupby("subject")["score"].mean().to_dict())
                if d:
                    out[m].setdefault(method, {})[st] = d
    return {m: d for m, d in out.items() if d}, subjects


def lookup_spread(perm_df, model_set, method):
    if perm_df is None:
        return None
    csv_method = {"mRSA": "wrsa_transfer", "fRSA": "crsa"}[method]
    rows = perm_df[
        (perm_df["model_set"] == model_set)
        & (perm_df["method"] == csv_method)
        & (perm_df["metric"] == "median_pairwise_diff")
    ]
    if rows.empty:
        return None
    return float(rows["observed_ratio"].iloc[0])


def draw_box(ax, x, mean, sem, width=0.30, filled=True, color="#D55E00"):
    lo, hi = mean - sem, mean + sem
    if hi - lo < 1e-4:
        lo, hi = mean - 0.0015, mean + 0.0015
    fc = color if filled else "white"
    ec = color
    rect = mpatches.Rectangle((x - width/2, lo), width, hi - lo,
                              facecolor=fc, edgecolor=ec, linewidth=0.6,
                              alpha=0.75 if filled else 0.95, zorder=3)
    ax.add_patch(rect)
    ml_color = "white" if filled else ec
    ax.hlines(mean, x - width/2, x + width/2, colors=ml_color,
              linewidth=0.7, zorder=5)


def plot_method_panel(ax, df, nc_df, bs_nc_df, perm_df, model_set, method,
                      show_xticks, panel_label=None, show_legend=False,
                      is_largest_amp=False):
    data, subjects = get_per_subject_scores(df, model_set)
    if not data:
        ax.set_visible(False); return

    is_mrsa = (method == "mRSA")

    # Sort models by mean mixed-RSA score on the controversial stimuli.
    # Keep that order for both rows so fixed RSA is directly comparable to
    # the mixed-RSA ordering used in the paper.
    def key(m):
        s = data[m].get("mRSA", {}).get("controversial", {})
        return np.mean(list(s.values())) if s else -999
    order = [m for m in data if method in data[m]]
    order.sort(key=key, reverse=True)

    # ---- Noise ceiling band: neutral gray ----
    if method == "mRSA":
        nc_ctrl = (nc_df[(nc_df["group"] == model_set)
                          & (nc_df["stimulus_type"] == "controversial")]
                   .groupby("subject")["noise_ceiling_spearman"].mean())
        nc_base = (nc_df[nc_df["group"] == "vicco"]
                   .groupby("subject")["noise_ceiling_spearman"].mean())
        nc_ctrl_v = np.sqrt(nc_ctrl.values) if len(nc_ctrl) else np.array([np.nan])
        nc_base_v = np.sqrt(nc_base.values)
    else:
        nc_ctrl_rows = bs_nc_df[(bs_nc_df["group"] == model_set)
                                 & (bs_nc_df["stimulus_type"] == "controversial")]
        nc_base_rows = bs_nc_df[bs_nc_df["group"] == "vicco"]
        nc_ctrl_v = nc_ctrl_rows["nc_mid"].values if len(nc_ctrl_rows) else \
                    nc_base_rows["nc_mid"].values
        nc_base_v = nc_base_rows["nc_mid"].values

    def _band(vals):
        m = np.nanmean(vals); s = np.nanstd(vals, ddof=1) / np.sqrt(max(len(vals), 1))
        return m - s, m + s, m

    nc_drawn = []
    nc_ctrl_mean = float(np.nanmean(nc_ctrl_v)) if len(nc_ctrl_v) else np.nan
    nc_base_mean = float(np.nanmean(nc_base_v)) if len(nc_base_v) else np.nan
    is_mrsa_panel = (method == "mRSA")

    if is_mrsa_panel:
        # mRSA: data and NC overlap, draw bands inside panel.
        if not np.isnan(nc_ctrl_mean):
            lo, hi, m = _band(nc_ctrl_v)
            ax.axhspan(lo, hi, color=COLOR_CSTIM, alpha=0.10, zorder=0, linewidth=0)
            ax.axhline(m, color=COLOR_CSTIM, linewidth=0.9, alpha=0.7, zorder=0,
                       linestyle="-")
        if not np.isnan(nc_base_mean):
            lo, hi, m = _band(nc_base_v)
            ax.axhspan(lo, hi, color=COLOR_BASELINE, alpha=0.10, zorder=0, linewidth=0)
            ax.axhline(m, color=COLOR_BASELINE, linewidth=0.9, alpha=0.7, zorder=0,
                       linestyle="--")
    else:
        # fRSA: NC sits well above the data range. Indicate it with upward
        # arrows + numeric annotations at the top of the panel rather than
        # crushing the data by extending the y-axis to include NC.
        nc_drawn = [("base", nc_base_mean, COLOR_BASELINE),
                    ("cstim", nc_ctrl_mean, COLOR_CSTIM)]

    # ---- Per-model boxes + per-subject dots ----
    n = len(order)
    x = np.arange(n)
    offset = 0.20
    for i, m in enumerate(order):
        c = data[m][method].get("controversial", {})
        b = data[m][method].get("vicco", {})
        if c:
            v = np.array(list(c.values()))
            sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
            draw_box(ax, x[i] - offset, v.mean(), sem,
                     width=0.30, filled=is_mrsa, color=COLOR_CSTIM)
        if b:
            v = np.array(list(b.values()))
            sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
            draw_box(ax, x[i] + offset, v.mean(), sem,
                     width=0.30, filled=is_mrsa, color=COLOR_BASELINE)

        for s in subjects:
            cv, bv = c.get(s), b.get(s)
            if cv is not None and bv is not None:
                ax.plot([x[i] - offset, x[i] + offset], [cv, bv],
                        color="#888888", linewidth=0.4, alpha=0.4, zorder=2)
            mk = "o" if is_mrsa else "D"
            if cv is not None:
                ax.scatter(x[i] - offset, cv, s=8,
                           facecolors=COLOR_CSTIM if is_mrsa else "none",
                           edgecolors=COLOR_CSTIM,
                           linewidths=0.5, marker=mk, zorder=6,
                           alpha=0.9)
            if bv is not None:
                ax.scatter(x[i] + offset, bv, s=8,
                           facecolors=COLOR_BASELINE if is_mrsa else "none",
                           edgecolors=COLOR_BASELINE,
                           linewidths=0.5, marker=mk, zorder=6,
                           alpha=0.9)

    # ---- Axis cosmetics ----
    ax.set_xticks(x)
    if show_xticks:
        ax.set_xticklabels([SHORT_NAMES.get(m, MODEL_DISPLAY.get(m, m))
                            for m in order],
                           rotation=45, ha="right")
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

    # ---- Spread-ratio: prominent annotation in the panel header strip ----
    spread = lookup_spread(perm_df, model_set, method)
    if spread is not None:
        col = "#9A4500" if spread >= 1.0 else "#666666"
        # Largest-amplification panel: prepend a star and bump font slightly
        # without adding extra horizontal text (panels can be narrow).
        if is_largest_amp and method == "mRSA":
            label = f"★ spread ratio: {spread:.2f}×"
            font_size = FONT["annotation"] + 1
        else:
            label = f"spread ratio: {spread:.2f}×"
            font_size = FONT["annotation"]
        ax.text(0.5, 1.005, label,
                transform=ax.transAxes, fontsize=font_size,
                fontweight="bold", color=col, ha="center", va="bottom",
                zorder=10)

    # ---- Inline NC indicator (fRSA only) ----
    # The fRSA noise ceiling sits well above the data range. Rather than
    # crushing the data into the bottom 30% of the panel by extending y-axis,
    # we show NC values as a compact inset box at the top-right of each
    # panel: "NC base / cstim". This keeps the comparison local to each
    # model set instead of shipping it off-panel.
    if nc_drawn:
        # nc_drawn is a list of ("base"/"cstim", value, color)
        nc_map = {kind: (val, col) for kind, val, col in nc_drawn}
        nc_b_val, _ = nc_map.get("base", (np.nan, COLOR_BASELINE))
        nc_c_val, _ = nc_map.get("cstim", (np.nan, COLOR_CSTIM))
        # Two-line, color-coded NC values inside an inset box.
        from matplotlib.patches import FancyBboxPatch  # local import keeps top tidy
        # Bounding box drawn in axes-fraction via a transform-aware patch.
        ax.text(0.985, 0.965,
                "NC (off-panel)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=FONT["small"] - 1, color="#555", fontweight="bold")
        ax.text(0.985, 0.875,
                f"base  {nc_b_val:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=FONT["small"] - 1, color=COLOR_BASELINE)
        ax.text(0.985, 0.795,
                f"cstim {nc_c_val:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=FONT["small"] - 1, color=COLOR_CSTIM)

    if panel_label is not None:
        add_panel_label(ax, panel_label, x=-0.04, y=1.05)

    if show_legend:
        handles = [
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=COLOR_CSTIM, markeredgecolor=COLOR_CSTIM,
                   markersize=4, label="Controversial"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=COLOR_BASELINE, markeredgecolor=COLOR_BASELINE,
                   markersize=4, label="Baseline"),
            Line2D([0], [0], color=COLOR_CSTIM, linewidth=0.9,
                   linestyle="-",  label="Noise ceiling (cstim)"),
            Line2D([0], [0], color=COLOR_BASELINE, linewidth=0.9,
                   linestyle="--", label="Noise ceiling (base)"),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=True,
                  framealpha=0.92, edgecolor="none", fontsize=FONT["small"],
                  ncol=2, columnspacing=0.8, handletextpad=0.4, handlelength=1.4)


def main():
    df = load_scores()
    nc_df = load_nc()
    bs_nc_df = load_bs_nc()
    perm_df = load_perm()

    # Width ratios proportional to model count
    ratios = []
    for ms in PANEL_ORDER:
        n_models = len([m for m in MODEL_SETS[ms]])
        ratios.append(max(n_models, 4))

    fig = plt.figure(figsize=(W_DOUBLE, 9.0))
    gs = fig.add_gridspec(
        2, len(PANEL_ORDER),
        width_ratios=ratios,
        wspace=0.06, hspace=0.18,
        left=0.05, right=0.99, top=0.94, bottom=0.16,
    )

    # Compute y-limits anchored on the DATA (not on NC).
    # For mRSA, model scores get reasonably close to NC, so NC fits in panel.
    # For fRSA, scores are much lower than NC; rather than crush the data into
    # the bottom 30% of the panel, we trim to data range and indicate the NC
    # off-panel via an upward arrow + numerical annotation per panel.
    def row_ylim(method):
        data_max = df[df["method"] == method]["score"].max()
        if method == "mRSA":
            # Include NC so the band remains visible (data and NC overlap).
            nc_max = float(np.sqrt(nc_df["noise_ceiling_spearman"].clip(0).max()))
            upper = max(data_max, nc_max) * 1.10
        else:
            # Anchor purely on data; NC will be shown as a numeric annotation.
            upper = data_max * 1.10
        return 0, upper

    ylims = {"mRSA": row_ylim("mRSA"), "fRSA": row_ylim("fRSA")}

    # Identify the largest-amplification panel (by mRSA spread ratio) so its
    # title can flag the takeaway directly.
    mrsa_ratios = {}
    for ms in PANEL_ORDER:
        r = lookup_spread(perm_df, ms, "mRSA")
        if r is not None:
            mrsa_ratios[ms] = r
    largest_ms = max(mrsa_ratios, key=mrsa_ratios.get) if mrsa_ratios else None

    panel_letters = list("abcde")
    for r, method in enumerate(["mRSA", "fRSA"]):
        for c, ms in enumerate(PANEL_ORDER):
            ax = fig.add_subplot(gs[r, c])
            ax.set_ylim(*ylims[method])
            plot_method_panel(
                ax, df, nc_df, bs_nc_df, perm_df, ms, method,
                show_xticks=(r == 1),
                panel_label=panel_letters[c] if r == 0 else None,
                show_legend=(r == 0 and c == 0),
                is_largest_amp=(ms == largest_ms),
            )
            if c == 0:
                ax.set_ylabel(("Mixed RSA " if method == "mRSA"
                               else "Fixed RSA ") + r"($r_s$)")
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)
            if r == 0:
                # Push title up to leave a header strip for the spread ratio
                ax.set_title(TITLE[ms], y=1.08)

    out_pdf = FIGURES_DIR / "brain_alignment_improved.pdf"
    out_png = FIGURES_DIR / "brain_alignment_improved.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=DPI)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
