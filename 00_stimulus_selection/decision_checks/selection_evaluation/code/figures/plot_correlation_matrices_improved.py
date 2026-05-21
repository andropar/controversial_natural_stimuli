#!/usr/bin/env python3
"""Improved cross-model RDM correlation-matrix figures (selected vs random).

Reads per-model_set correlation-matrix CSVs from the current paper eval
pipeline output,
produces clean two-panel heatmaps comparing model-RDM agreement on:
    - Random natural baseline (averaged across 10000-image random subsets)
    - Selected controversial stimuli (n=100)

Vs. the original eval_pipeline plot, this version:
    - Drops in-cell numeric annotations (data-ink ratio) — keeps colorbar
    - Uses a diverging colormap (RdBu_r) centred at 0 so negative correlations
      are visually distinct from "uncorrelated"
    - Shared horizontal colorbar; readable model-name labels; consistent
      Okabe-Ito-styled typography from style_improved
    - Reports median off-diagonal correlation per panel as a subtitle (the
      headline number)
    - Optional "Selected + noise" third panel for tracks where it's available

Usage:
    python plot_correlation_matrices_improved.py            # all sets x (raw + sub-01)
    python plot_correlation_matrices_improved.py --include_noised
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))
sys.path.insert(0, str(_PAPER / "figures"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import config
from style_improved import (
    apply_style, FONT, DPI, W_DOUBLE, W_SINGLE, OKABE_ITO,
    COLOR_CSTIM, COLOR_BASELINE,
)

apply_style()

# Colors for the 4 conditions in the summary plot
COND_INFO = {
    "random_clean":   {"label": "Rand\nclean",   "color": COLOR_BASELINE, "filled": True},
    "selected_clean": {"label": "Sel\nclean",    "color": COLOR_CSTIM,    "filled": True},
    "random_noised":  {"label": "Rand\nnoised",  "color": COLOR_BASELINE, "filled": False},
    "selected_noised":{"label": "Sel\nnoised",   "color": COLOR_CSTIM,    "filled": False},
}
COND_ORDER = list(COND_INFO.keys())
SELECTED_MATRIX_TYPES = {"selected_clean", "selected_noised"}

PROJECT_ROOT = config.PROJECT_ROOT
MODEL_SETS = config.MODEL_SETS
SELECTION_OUTPUT_ROOT = config.SELECTION_OUTPUT_ROOT
FIGURES_DIR = Path(__file__).resolve().parent

# Short labels matching the brain-alignment figure
SHORT_NAMES = {
    "torchvision_alexnet_imagenet1k_v1":               "AlexNet",
    "torchvision_vgg16_imagenet1k_v1":                 "VGG-16",
    "torchvision_resnet50_imagenet1k_v1":              "ResNet-50",
    "torchvision_convnext_base_imagenet1k_v1":         "ConvNeXt-B",
    "torchvision_vit_l_16_imagenet1k_v1":              "ViT-L/16",
    "cornet_s":                                        "CORnet-S",
    "vissl_resnet50_supervised":                       "Supervised",
    "vissl_resnet50_barlowtwins":                      "BarlowTwins",
    "vissl_resnet50_mocov2":                           "MoCoV2",
    "vicreg_resnet50":                                 "VICReg",
    "robustness_imagenet_l2_eps3":                     "Robust-L2",
    "slip_vit_l_slip":                                 "SLIP",
    "slip_vit_l_simclr":                               "SimCLR-ViT",
    "timm_vit_large_patch14_clip_224_laion2b":         "CLIP-L2B",
    "dinov2_vitl14":                                   "DINOv2",
    "openclip_vit_so400m_14_siglip_webli":             "SigLIP",
    "openclip_vit_l_14_quickgelu_metaclip_400m":       "MC-400M",
    "openclip_vit_l_14_quickgelu_metaclip_fullcc":     "MC-Full",
    "timm_vit_large_patch14_clip_224_dfn2b":           "DFN-2B",
    "timm_vit_large_patch14_clip_quickgelu_224_openai": "CLIP-OAI",
    "openclip_vit_l_14_laion400m_e31":                 "CLIP-L400",
}

TITLE = {
    "training_objective": "Training Objective",
    "sota":               "State of the Art",
    "architecture":       "Architecture",
    "dataset":            "Dataset",
    "all_models":         "All Models",
}

# Selection-run eval output paths per model_set. These directories also hold
# the additional vicco/pool CSVs computed by the paper analysis scripts.
EVAL_DIRS = {
    "all_models":         SELECTION_OUTPUT_ROOT / "all_models" /
                          "method-raw_plus_all_encodings" / "20251222_175721" / "eval_pipeline",
    "sota":               SELECTION_OUTPUT_ROOT / "sota" /
                          "method-raw_plus_all_encodings" / "20251222_175721" / "eval_pipeline",
    "training_objective": SELECTION_OUTPUT_ROOT / "training_objective" /
                          "method-raw_plus_all_encodings" / "20251222_175721" / "eval_pipeline",
    "architecture":       SELECTION_OUTPUT_ROOT / "architecture" /
                          "method-raw_plus_all_encodings" / "20251222_175721" / "eval_pipeline",
    "dataset":            SELECTION_OUTPUT_ROOT / "dataset" /
                          "method-raw_plus_all_encodings" / "20251222_175721" / "eval_pipeline",
}

# Current paper eval pipeline output. This is the canonical in-silico eval data:
# unique per-subject encodings and the metric/correlation configuration used by
# 00_selection_evaluation/analysis/run_discriminability_unique.sh.
CURRENT_EVAL_DATA_SUFFIX = "_unique_boot"
CURRENT_EVAL_DIRS = {
    ms: config.EVAL_DATA_DIR / f"{ms}{CURRENT_EVAL_DATA_SUFFIX}"
    for ms in ["all_models", "sota", "training_objective", "architecture", "dataset"]
}


def load_matrix(df: pd.DataFrame, track: str, matrix_type: str, model_order: list) -> np.ndarray:
    """Build a (M, M) correlation matrix from the long-format CSV rows."""
    sub = df[(df["track"] == track) & (df["matrix_type"] == matrix_type)]
    M = np.full((len(model_order), len(model_order)), np.nan, dtype=np.float64)
    m2i = {m: i for i, m in enumerate(model_order)}
    for _, row in sub.iterrows():
        if row["model_i"] in m2i and row["model_j"] in m2i:
            M[m2i[row["model_i"]], m2i[row["model_j"]]] = row["correlation"]
    return M


def median_offdiag(M: np.ndarray) -> float:
    """Median of the strict off-diagonal entries (ignoring NaNs)."""
    n = M.shape[0]
    mask = ~np.eye(n, dtype=bool)
    vals = M[mask]
    return float(np.nanmedian(vals))


def median_diag(M: np.ndarray) -> float:
    """Median of the diagonal entries (ignoring NaNs).

    For noised matrices this quantifies *within-model* signal preservation under
    the calibrated brain noise; for clean matrices it is 1 by construction.
    """
    return float(np.nanmedian(np.diag(M)))


def _resolve_csv(model_set: str, eval_dir: Path, baseline: str = "vicco") -> tuple[Path, str]:
    """Pick the appropriate CSV for the requested random baseline.

    baseline = "eval_pipeline" → current paper eval data in
                                  00_selection_evaluation/data/*_unique_boot
                                  (random = 10k LAION pool averaged across
                                  random 100-stim subsets)
    baseline = "vicco"         → correlation_matrices_with_random_noised.csv
                                  (curated 292-image vicco set, x20 subsamples)
    baseline = "pool"          → correlation_matrices_with_random_noised_pool.csv
                                  (10k LAION pool subset rsynced from raven)

    Falls back to the selection-run legacy correlation_matrices.csv only when
    the current paper eval data is missing.
    """
    if baseline == "eval_pipeline":
        current = CURRENT_EVAL_DIRS[model_set] / "correlation_matrices.csv"
        if current.exists():
            return current, "current_eval_pipeline"
        legacy = eval_dir / "correlation_matrices.csv"
        return legacy, "legacy_eval_pipeline"
    if baseline == "pool":
        aug = eval_dir / "correlation_matrices_with_random_noised_pool.csv"
        if aug.exists():
            return aug, "pool"
    aug = eval_dir / "correlation_matrices_with_random_noised.csv"
    if aug.exists():
        return aug, "vicco"
    legacy = eval_dir / "correlation_matrices.csv"
    return legacy, "legacy"


def _with_current_selected_rows(df: pd.DataFrame, model_set: str) -> pd.DataFrame:
    """Use canonical current-paper selected matrices for all baselines.

    Selected matrices do not depend on the random-baseline pool. Keeping them
    anchored to the current eval pipeline prevents old vicco/pool CSVs from
    silently reintroducing legacy metric/encoding choices in the selected
    panels.
    """
    current_path = CURRENT_EVAL_DIRS[model_set] / "correlation_matrices.csv"
    if not current_path.exists():
        return df

    current = pd.read_csv(current_path)
    current = current[current["matrix_type"].isin(SELECTED_MATRIX_TYPES)]
    if current.empty:
        return df

    is_selected = df["matrix_type"].isin(SELECTED_MATRIX_TYPES)
    return pd.concat([df[~is_selected], current], ignore_index=True)


# Per-baseline output subfolder under FIGURES_DIR
BASELINE_SUBDIR = {
    "eval_pipeline": "correlation_matrices_eval_pipeline",
    "vicco":          "correlation_matrices_vicco",
    "pool":           "correlation_matrices_pool",
}
SUMMARY_SUBDIR = "correlation_matrices_summary"
INSILICO_SUBDIR = "insilico_curve"


def _ensure_subdir(out_dir: Path, sub: str) -> Path:
    p = out_dir / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


# Mode -> ordered list of (display title, matrix_type) panels
PANEL_PRESETS = {
    "clean":   [("Random natural baseline", "random_clean"),
                ("Selected controversial",  "selected_clean")],
    "noised":  [("Random + brain noise",    "random_noised"),
                ("Selected + brain noise",  "selected_noised")],
    "all4":    [("Random natural baseline", "random_clean"),
                ("Selected controversial",  "selected_clean"),
                ("Random + brain noise",    "random_noised"),
                ("Selected + brain noise",  "selected_noised")],
    # Backwards-compatible 3-panel layout (uses legacy CSV's selected_noised)
    "legacy3": [("Random natural baseline", "random_clean"),
                ("Selected controversial",  "selected_clean"),
                ("Selected + brain noise",  "selected_noised")],
}


def plot_one(model_set: str, track: str, mode: str, out_dir: Path,
             out_suffix: str | None = None, baseline: str = "vicco"):
    eval_dir = EVAL_DIRS[model_set]
    csv_path, csv_kind = _resolve_csv(model_set, eval_dir, baseline=baseline)
    if not csv_path.exists():
        print(f"  MISSING {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    df = _with_current_selected_rows(df, model_set)

    panels = PANEL_PRESETS[mode]
    available_types = set(df["matrix_type"].unique())
    missing = [m for _, m in panels if m not in available_types]
    if missing:
        print(f"  [{model_set}/{track}/{mode}] CSV ({csv_kind}) missing: {missing}; skipping")
        return None

    set_models = list(MODEL_SETS[model_set])
    csv_models = set(df["model_i"].unique()) & set(df["model_j"].unique())
    model_order = [m for m in set_models if m in csv_models]
    if not model_order:
        print(f"  no overlap of MODEL_SETS[{model_set}] with CSV models for {csv_path}")
        return None
    labels = [SHORT_NAMES.get(m, m) for m in model_order]

    n_panels = len(panels)
    n_models = len(model_order)
    panel_w = max(3.0, 0.32 * n_models)
    fig_w = min(W_DOUBLE, panel_w * n_panels + 1.2)
    fig_h = panel_w + 1.4
    fig, axes = plt.subplots(
        1, n_panels, figsize=(fig_w, fig_h),
        sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.10},
    )
    if n_panels == 1:
        axes = [axes]

    # Colormap: RdYlBu_r gives a vibrant yellow at the mid range (~0.4-0.6),
    # which is the working range for these correlations and makes the block
    # structure visible. Centered at 0 so any negative correlations still
    # render as blue. vmax compressed to 0.9 (just above typical data max,
    # excluding the always-1 diagonal) so the [0, 0.7] positive range gets
    # the bulk of the colormap dynamic range.
    vmin, vcenter, vmax = -0.4, 0.0, 0.9
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cmap = "RdYlBu_r"

    medians = {}        # off-diag medians per matrix_type
    medians_diag = {}   # diagonal medians per matrix_type
    im = None
    for ax, (display, mat_type) in zip(axes, panels):
        M = load_matrix(df, track, mat_type, model_order)
        med = median_offdiag(M)
        med_d = median_diag(M)
        medians[mat_type] = med
        medians_diag[mat_type] = med_d
        im = ax.imshow(M, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")

        ax.set_xticks(range(n_models))
        ax.set_xticklabels(labels, rotation=45, ha="right",
                           fontsize=FONT["small"])
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(labels, fontsize=FONT["small"])
        ax.tick_params(axis="both", which="both", length=0)

        # Two-line subtitle: diagonal (within-model) up top, off-diagonal below.
        # Diagonal of clean matrices is 1.0 by construction; for noised matrices
        # it quantifies how much the calibrated noise degrades within-model
        # self-correlation.
        ax.set_title(
            f"{display}\n"
            f"median diag $r$ = {med_d:.2f}    median off-diag $r$ = {med:.2f}",
            fontsize=FONT["title"], pad=6,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Suptitle: contrast Random vs Selected within the SAME mode (clean or noised)
    track_tag = "cRSA" if track == "raw" else f"mRSA ({track})"
    delta_str = ""
    if mode == "clean":
        if "random_clean" in medians and "selected_clean" in medians:
            d = medians["random_clean"] - medians["selected_clean"]
            delta_str = f"    $\\Delta$ median off-diag $r$ = {d:+.2f}"
    elif mode == "noised":
        if "random_noised" in medians and "selected_noised" in medians:
            d = medians["random_noised"] - medians["selected_noised"]
            delta_str = f"    $\\Delta$ median off-diag $r$ = {d:+.2f}"
    elif mode == "all4":
        if all(k in medians for k in ("random_clean", "selected_clean")):
            d_clean = medians["random_clean"] - medians["selected_clean"]
            d_noised = medians["random_noised"] - medians["selected_noised"]
            delta_str = (f"    clean $\\Delta$ = {d_clean:+.2f}, "
                         f"noised $\\Delta$ = {d_noised:+.2f}")

    fig.suptitle(
        f"{TITLE.get(model_set, model_set)} — {track_tag}{delta_str}",
        fontsize=FONT["title"] + 1, y=0.995,
    )

    cbar_ax = fig.add_axes([0.18, 0.04, 0.66, 0.025])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Correlation between model RDMs", fontsize=FONT["small"])
    cbar.ax.tick_params(labelsize=FONT["small"])
    cbar.outline.set_visible(False)

    fig.tight_layout(rect=[0, 0.10, 1, 0.96])

    suffix = out_suffix if out_suffix is not None else f"_{mode}"
    base = f"correlation_matrices_{model_set}_{track}{suffix}_improved"
    sub = _ensure_subdir(out_dir, BASELINE_SUBDIR.get(baseline, "correlation_matrices_misc"))
    out_pdf = sub / f"{base}.pdf"
    out_png = sub / f"{base}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    print(f"  saved {sub.name}/{out_pdf.name}")
    return out_pdf


def plot_summary(track: str, out_dir: Path,
                 model_sets: list[str],
                 baseline: str = "vicco") -> Path:
    """Discriminability-margin summary across all model sets for one track.

    Per (model_set, condition):
        - filled marker = median diagonal r       (within-model self-correlation)
        - open marker   = median off-diagonal r   (between-model agreement)
        - vertical bar between them = discriminability margin

    Layout: 1 row x N model_sets, shared y-axis. Random = blue, Selected =
    vermillion. Clean = filled, Noised = open.
    """
    n_panels = len(model_sets)
    fig_w = max(W_DOUBLE, 2.4 * n_panels + 1.6)
    fig_h = 4.6
    fig, axes = plt.subplots(
        1, n_panels, figsize=(fig_w, fig_h), sharey=True,
        gridspec_kw={"wspace": 0.25},
    )
    if n_panels == 1:
        axes = [axes]

    x_positions = np.arange(len(COND_ORDER))

    for ax, ms in zip(axes, model_sets):
        eval_dir = EVAL_DIRS[ms]
        csv_path, _ = _resolve_csv(ms, eval_dir, baseline=baseline)
        if not csv_path.exists():
            ax.set_visible(False); continue
        df = pd.read_csv(csv_path)
        df = _with_current_selected_rows(df, ms)

        set_models = list(MODEL_SETS[ms])
        csv_models = set(df["model_i"].unique()) & set(df["model_j"].unique())
        model_order = [m for m in set_models if m in csv_models]
        if not model_order:
            ax.set_visible(False); continue

        for x, cond in zip(x_positions, COND_ORDER):
            if cond not in df["matrix_type"].unique():
                continue
            M = load_matrix(df, track, cond, model_order)
            d = median_diag(M)
            o = median_offdiag(M)
            info = COND_INFO[cond]
            color = info["color"]
            mfc = color if info["filled"] else "white"
            # Vertical "margin bar" between the two stats
            ax.vlines(x, o, d, color=color, linewidth=2.0, alpha=0.55, zorder=2)
            # Diagonal marker (self) — filled diamond
            ax.scatter(x, d, s=42, marker="D", facecolors=mfc, edgecolors=color,
                       linewidths=1.0, zorder=4)
            # Off-diagonal marker (between) — circle
            ax.scatter(x, o, s=42, marker="o", facecolors=mfc, edgecolors=color,
                       linewidths=1.0, zorder=4)
            # Annotate margin numerically just above the diagonal marker
            ax.annotate(f"{d - o:+.2f}", xy=(x, d), xytext=(0, 4),
                        textcoords="offset points", ha="center",
                        fontsize=FONT["annotation"], color=color, fontweight="bold")

        ax.set_xticks(x_positions)
        ax.set_xticklabels([COND_INFO[c]["label"] for c in COND_ORDER],
                           fontsize=FONT["small"])
        ax.set_xlim(-0.6, len(COND_ORDER) - 0.4)
        ax.set_title(TITLE.get(ms, ms), fontsize=FONT["title"], pad=20)
        ax.axhline(0, color="#999", linewidth=0.4, zorder=0)
        ax.grid(axis="y", color="#eee", linewidth=0.4, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel(r"median $r$ between model RDMs", fontsize=FONT["small"])
    axes[0].set_ylim(-0.2, 1.20)
    # Suppress y-tick labels above 1.05 so the headroom for annotations doesn't
    # add visual clutter.
    for ax in axes:
        ax.set_yticks(np.arange(-0.2, 1.01, 0.2))

    # Single shared legend below the panels (markers + clean/noised encoding)
    legend_handles = [
        plt.Line2D([0], [0], marker="D", color="none",
                   markerfacecolor="#666", markeredgecolor="#666",
                   markersize=6, label="diag (within-model)"),
        plt.Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="#666", markeredgecolor="#666",
                   markersize=6, label="off-diag (between)"),
        plt.Line2D([0], [0], marker="s", color="none",
                   markerfacecolor=COLOR_BASELINE, markeredgecolor=COLOR_BASELINE,
                   markersize=6, label="Random"),
        plt.Line2D([0], [0], marker="s", color="none",
                   markerfacecolor=COLOR_CSTIM, markeredgecolor=COLOR_CSTIM,
                   markersize=6, label="Selected"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=4, fontsize=FONT["small"], frameon=False,
               bbox_to_anchor=(0.5, 0.02))

    track_tag = "cRSA (raw features)" if track == "raw" else f"mRSA ({track})"
    fig.suptitle(
        f"Discriminability margin (median diag − off-diag) · {track_tag}\n"
        f"filled = clean, open = noised; bar between markers = margin",
        fontsize=FONT["title"], y=0.99,
    )
    # Reserve generous top + bottom margins for suptitle and shared legend
    fig.subplots_adjust(left=0.06, right=0.985, top=0.82, bottom=0.18,
                        wspace=0.25)

    pool_suffix = "_pool" if baseline == "pool" else ""
    base = f"correlation_matrices_summary_{track}{pool_suffix}_improved"
    out_dir = _ensure_subdir(out_dir, SUMMARY_SUBDIR)
    out_pdf = out_dir / f"{base}.pdf"
    out_png = out_dir / f"{base}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_pdf.name}")
    return out_pdf


def _multiplier_to_noise_ceiling(k: float, nc_base: float = 0.46) -> float:
    """Match `multiplier_to_noise_ceiling` in 02_compute_discriminability.py.

    nc_base = noise ceiling at multiplier 1 (the eval pipeline's calibration
    target — 0.46).
    """
    if k <= 0:
        return 1.0
    if nc_base <= 0 or nc_base >= 1:
        return float(nc_base)
    term = k * k * (1.0 / (nc_base * nc_base) - 1.0)
    return float(1.0 / np.sqrt(1.0 + term))


def plot_margin_curve(track: str, out_dir: Path,
                      model_sets: list[str],
                      csv_name: str = "margin_curve.csv",
                      out_suffix: str = "") -> Path | None:
    """Margin (median diag - median off-diag) vs noise level.

    Uses the same x-axis as the existing in-silico evaluation plot:
        x = 1 - noise_ceiling
    on a linear scale, with a "selection target" line at the calibrated
    noise level (NC = 0.46 → x = 0.54). One panel per model_set; two lines
    per panel (Random vs Selected).

    Args:
        csv_name: Which margin-curve CSV to read in each eval_pipeline dir.
            "margin_curve.csv" (vicco baseline) or "margin_curve_pool.csv"
            (LAION 10k natural-pool baseline, after rsync from raven).
        out_suffix: Inserted into the output filename to distinguish baselines.
    """
    rows = []
    for ms in model_sets:
        fp = EVAL_DIRS[ms] / csv_name
        if not fp.exists():
            print(f"  [{ms}] no {csv_name} — skipping")
            continue
        df = pd.read_csv(fp)
        df["model_set"] = ms
        rows.append(df)
    if not rows:
        return None
    df_all = pd.concat(rows, ignore_index=True)
    df_all = df_all[df_all["track"] == track]
    if df_all.empty:
        print(f"  no rows for track {track}"); return None

    # Convert noise multipliers to noise levels (1 - NC) so the x-axis matches
    # the existing in-silico evaluation figure.
    df_all["noise_level"] = df_all["noise_mult"].apply(
        lambda k: 1.0 - _multiplier_to_noise_ceiling(k)
    )

    n_panels = len(model_sets)
    fig_w = max(W_DOUBLE, 2.4 * n_panels + 1.4)
    fig_h = 4.6
    fig, axes = plt.subplots(
        1, n_panels, figsize=(fig_w, fig_h), sharey=True,
        gridspec_kw={"wspace": 0.18},
    )
    if n_panels == 1:
        axes = [axes]

    cond_color = {"random": COLOR_BASELINE, "selected": COLOR_CSTIM}
    cond_label = {"random": "Random", "selected": "Selected"}

    # Calibrated noise position on the (1 - NC) axis: 1 - 0.46 = 0.54
    calibrated_x = 1.0 - 0.46

    for ax, ms in zip(axes, model_sets):
        sub = df_all[df_all["model_set"] == ms]
        if sub.empty:
            ax.set_visible(False); continue

        for cond in ("random", "selected"):
            cs = sub[sub["condition"] == cond].sort_values("noise_level")
            if cs.empty:
                continue
            color = cond_color[cond]
            ax.plot(cs["noise_level"], cs["margin"],
                    color=color, linewidth=1.6, marker="o", markersize=4,
                    label=cond_label[cond], zorder=3)

        ax.axvline(calibrated_x, color="#666", linewidth=0.6,
                   linestyle="--", alpha=0.7, zorder=1)
        ax.axhline(0, color="#bbb", linewidth=0.4, zorder=0)
        ax.set_xlim(-0.02, 1.02)
        ax.set_title(TITLE.get(ms, ms), fontsize=FONT["title"], pad=8)
        ax.set_xlabel("Noise level (1 − noise ceiling)", fontsize=FONT["small"])
        ax.grid(axis="y", color="#eee", linewidth=0.4, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Discriminability margin\n(median diag − off-diag)",
                       fontsize=FONT["small"])

    # Annotate the selection-target line on the leftmost panel
    ymax = axes[0].get_ylim()[1]
    axes[0].text(calibrated_x, ymax * 0.97, "selection\ntarget",
                 ha="center", va="top", color="#444",
                 fontsize=FONT["annotation"])

    legend_handles = [
        plt.Line2D([0], [0], color=COLOR_BASELINE, marker="o",
                   markersize=5, linewidth=1.6, label="Random"),
        plt.Line2D([0], [0], color=COLOR_CSTIM, marker="o",
                   markersize=5, linewidth=1.6, label="Selected"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=2, fontsize=FONT["small"], frameon=False,
               bbox_to_anchor=(0.5, 0.02))

    track_tag = "cRSA (raw features)" if track == "raw" else f"mRSA ({track})"
    fig.suptitle(
        f"Discriminability margin vs noise level · {track_tag}",
        fontsize=FONT["title"] + 1, y=0.99,
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.18,
                        wspace=0.18)

    base = f"correlation_matrices_margin_curve_{track}{out_suffix}_improved"
    out_dir = _ensure_subdir(out_dir, SUMMARY_SUBDIR)
    out_pdf = out_dir / f"{base}.pdf"
    out_png = out_dir / f"{base}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir.name}/{out_pdf.name}")
    return out_pdf


def main():
    ap = argparse.ArgumentParser(description="Plot improved cross-model RDM correlation matrices")
    ap.add_argument(
        "--mode", default="both",
        choices=["clean", "noised", "all4", "legacy3", "both"],
        help="Which matrix panels to render. 'both' produces clean and noised as separate files. "
             "'all4' is one 4-panel figure. 'legacy3' uses the old 3-panel layout.",
    )
    ap.add_argument("--example_subject", default="sub-01",
                    help="Which subject's encoding-projected (mRSA) track to plot (default sub-01)")
    ap.add_argument("--no_summary", dest="summary", action="store_false",
                    help="Skip the per-track discriminability-margin summary figure")
    ap.set_defaults(summary=True)
    args = ap.parse_args()

    if args.mode == "both":
        modes = ["clean", "noised"]
    else:
        modes = [args.mode]

    model_sets_order = ["all_models", "sota", "training_objective", "architecture", "dataset"]

    out_dir = FIGURES_DIR
    print(f"Output dir: {out_dir}")

    # Baselines available
    pool_present = any(
        (EVAL_DIRS[ms] / "correlation_matrices_with_random_noised_pool.csv").exists()
        for ms in model_sets_order
    )
    eval_pipeline_present = any(
        (CURRENT_EVAL_DIRS[ms] / "correlation_matrices.csv").exists()
        or (EVAL_DIRS[ms] / "correlation_matrices.csv").exists()
        for ms in model_sets_order
    )
    baselines = []
    if eval_pipeline_present:
        baselines.append("eval_pipeline")
    baselines.append("vicco")
    if pool_present:
        baselines.append("pool")
    print(f"Baselines: {baselines}")

    for baseline in baselines:
        for ms in model_sets_order:
            # Current eval-pipeline CSV has no random_noised panel; legacy fallback
            # does not either. Keep the 3-panel layout for this baseline.
            if baseline == "eval_pipeline":
                local_modes = ["legacy3"]
            else:
                local_modes = modes
            for track in ["raw", args.example_subject]:
                for m in local_modes:
                    print(f"-> [{baseline}] {ms}/{track}/{m}")
                    plot_one(ms, track, m, out_dir, baseline=baseline)

    if args.summary:
        # Summary lollipops only for the recomputed-CSV baselines (need
        # random_noised, which the original pipeline CSV does not store).
        for baseline in [b for b in baselines if b != "eval_pipeline"]:
            for track in ["raw", args.example_subject]:
                print(f"-> [{baseline}] SUMMARY {track}")
                plot_summary(track, out_dir, model_sets_order, baseline=baseline)
        # Margin curve (vicco baseline always)
        for track in ["raw", args.example_subject]:
            print(f"-> MARGIN CURVE (vicco baseline) {track}")
            plot_margin_curve(track, out_dir, model_sets_order,
                              csv_name="margin_curve.csv", out_suffix="")
        # Natural-pool margin curve (only when rsynced + recomputed)
        if any((EVAL_DIRS[ms] / "margin_curve_pool.csv").exists()
               for ms in model_sets_order):
            for track in ["raw", args.example_subject]:
                print(f"-> MARGIN CURVE (natural-pool baseline) {track}")
                plot_margin_curve(track, out_dir, model_sets_order,
                                  csv_name="margin_curve_pool.csv",
                                  out_suffix="_pool")
    print("Done.")


if __name__ == "__main__":
    main()
