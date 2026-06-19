#!/usr/bin/env python3
"""Full-data rerun brain-alignment plot in the paper figure layout.

This mirrors the layout of
01_brain_model_alignment/figures/rsa_scores/brain_alignment_improved_with_shared.pdf
for the full-data rerun. The current full-data rerun result tables contain
controversial cstim and vicco rows; DeepVision shared rows are drawn
automatically if they are added later.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


THIS = Path(__file__).resolve()
RERUN_ROOT = THIS.parents[1]
SHARE_ROOT = THIS.parents[3]
HELPERS = SHARE_ROOT / "src"
sys.path.insert(0, str(HELPERS))

from cstims import constants, paths
from cstims.paper.style_improved import (  # noqa: E402
    DPI,
    FONT,
    OKABE_ITO,
    W_DOUBLE,
    COLOR_BASELINE,
    COLOR_CSTIM,
    add_panel_label,
    apply_style,
)


apply_style()

RESULTS_DIR = RERUN_ROOT / "results"
FIGURES_DIR = RERUN_ROOT / "figures"

COLOR_SHARED = OKABE_ITO["bluish_green"]

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
    "sota": "State of the Art",
    "architecture": "Architecture",
    "dataset": "Dataset",
    "all_models": "All Models",
}

PANEL_ORDER = ["all_models", "sota", "training_objective", "architecture", "dataset"]
STIM_TYPES = ("controversial", "vicco", "deepvision_shared")
STIM_DISPLAY = {
    "controversial": "Controversial",
    "vicco": "Baseline (vicco)",
    "deepvision_shared": "DeepVision shared",
}
STIM_COLOR = {
    "controversial": COLOR_CSTIM,
    "vicco": COLOR_BASELINE,
    "deepvision_shared": COLOR_SHARED,
}
STIM_OFFSET = {
    "controversial": -0.27,
    "vicco": 0.0,
    "deepvision_shared": 0.27,
}
BOX_WIDTH = 0.20
DEFAULT_RDM_NC = RESULTS_DIR / "rdm_noise_ceilings_by_roi.csv"
DEFAULT_BETWEEN_NC = RESULTS_DIR / "between_subject_noise_ceilings_by_roi.csv"


def load_scores(rois: list[str] | None = None) -> pd.DataFrame:
    frames = []
    specs = [
        ("paper_layer_mrsa_by_roi.csv", "mrsa", "mRSA"),
        ("paper_layer_crsa_by_roi.csv", "crsa", "fRSA"),
    ]
    usecols = [
        "subject",
        "roi",
        "model_set",
        "model",
        "display_name",
        "stimulus_type",
        "bootstrap_idx",
        "n_stimuli",
    ]
    for fname, score_col, method in specs:
        path = RESULTS_DIR / fname
        cols = usecols + [score_col]
        df = pd.read_csv(path, usecols=cols)
        if rois is not None:
            df = df[df["roi"].isin(rois)].copy()
        df = df.rename(columns={score_col: "score"})
        df["method"] = method
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_noise_ceilings(rdm_path: Path, between_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not rdm_path.exists():
        raise FileNotFoundError(f"Missing RDM noise ceilings: {rdm_path}")
    if not between_path.exists():
        raise FileNotFoundError(f"Missing between-subject noise ceilings: {between_path}")

    rdm = pd.read_csv(rdm_path)
    cstim_rdm = rdm[rdm["stimulus_type"].eq("controversial")][
        ["roi", "subject", "group", "noise_ceiling_spearman"]
    ].rename(columns={"group": "model_set", "noise_ceiling_spearman": "rdm_nc"})
    vicco_rdm = (
        rdm[rdm["stimulus_type"].eq("vicco")]
        .groupby(["roi", "subject"], as_index=False)
        .agg(rdm_nc=("noise_ceiling_spearman", "mean"))
    )
    vicco_rdm["model_set"] = "vicco"
    rdm_lookup = pd.concat([cstim_rdm, vicco_rdm], ignore_index=True)
    rdm_lookup["mrsa_denom"] = np.where(
        rdm_lookup["rdm_nc"].gt(0),
        np.sqrt(rdm_lookup["rdm_nc"]),
        np.nan,
    )

    between = pd.read_csv(between_path)
    cstim_between = between[between["stimulus_type"].eq("controversial")][
        ["roi", "subject", "group", "nc_mid"]
    ].rename(columns={"group": "model_set", "nc_mid": "frsa_denom"})
    vicco_between = between[between["stimulus_type"].eq("vicco")][
        ["roi", "subject", "nc_mid"]
    ].rename(columns={"nc_mid": "frsa_denom"})
    vicco_between["model_set"] = "vicco"
    between_lookup = pd.concat([cstim_between, vicco_between], ignore_index=True)
    between_lookup.loc[~between_lookup["frsa_denom"].gt(0), "frsa_denom"] = np.nan
    return rdm_lookup, between_lookup


def normalize_scores(df: pd.DataFrame, rdm_nc: pd.DataFrame, between_nc: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["score_raw"] = df["score"]
    is_vicco = df["stimulus_type"].eq("vicco")
    df["nc_model_set"] = np.where(is_vicco, "vicco", df["model_set"])

    mrsa = df[df["method"].eq("mRSA")].merge(
        rdm_nc[["roi", "subject", "model_set", "mrsa_denom", "rdm_nc"]],
        left_on=["roi", "subject", "nc_model_set"],
        right_on=["roi", "subject", "model_set"],
        how="left",
        suffixes=("", "_nc"),
    )
    mrsa["noise_ceiling"] = mrsa["rdm_nc"]
    mrsa["score"] = mrsa["score_raw"] / mrsa["mrsa_denom"]

    frsa = df[df["method"].eq("fRSA")].merge(
        between_nc[["roi", "subject", "model_set", "frsa_denom"]],
        left_on=["roi", "subject", "nc_model_set"],
        right_on=["roi", "subject", "model_set"],
        how="left",
        suffixes=("", "_nc"),
    )
    frsa["noise_ceiling"] = frsa["frsa_denom"]
    frsa["score"] = frsa["score_raw"] / frsa["frsa_denom"]

    out = pd.concat([mrsa, frsa], ignore_index=True, sort=False)
    drop_cols = [c for c in ["model_set_nc", "nc_model_set", "mrsa_denom", "frsa_denom", "rdm_nc"] if c in out]
    return out.drop(columns=drop_cols)


def subject_scores(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            ["roi", "method", "model_set", "model", "display_name", "stimulus_type", "subject"],
            as_index=False,
        )["score"]
        .mean()
    )


def get_panel_data(df: pd.DataFrame, model_set: str):
    models = constants.MODEL_SETS[model_set]
    sub = df[df["model_set"].eq(model_set)]
    subjects = sorted(sub["subject"].unique())
    out = {}
    for model in models:
        out[model] = {}
        for method in ("mRSA", "fRSA"):
            block = sub[(sub["model"].eq(model)) & (sub["method"].eq(method))]
            for stim_type in STIM_TYPES:
                vals = (
                    block[block["stimulus_type"].eq(stim_type)]
                    .set_index("subject")["score"]
                    .to_dict()
                )
                if vals:
                    out[model].setdefault(method, {})[stim_type] = vals
    return {m: d for m, d in out.items() if d}, subjects


def spread_ratio(df: pd.DataFrame, model_set: str, method: str) -> float | None:
    means = (
        df[(df["model_set"].eq(model_set)) & (df["method"].eq(method))]
        .groupby(["model", "stimulus_type"], as_index=False)["score"]
        .mean()
    )
    ratios = {}
    for stim_type in ("controversial", "vicco"):
        vals = means[means["stimulus_type"].eq(stim_type)]["score"].to_numpy()
        if len(vals) < 2:
            return None
        ratios[stim_type] = np.median([abs(a - b) for a, b in combinations(vals, 2)])
    if ratios["vicco"] <= 0:
        return None
    return float(ratios["controversial"] / ratios["vicco"])


def draw_box(ax, x, mean, sem, width=BOX_WIDTH, filled=True, color=COLOR_CSTIM):
    lo, hi = mean - sem, mean + sem
    if hi - lo < 1e-4:
        lo, hi = mean - 0.0015, mean + 0.0015
    rect = mpatches.Rectangle(
        (x - width / 2, lo),
        width,
        hi - lo,
        facecolor=color if filled else "white",
        edgecolor=color,
        linewidth=0.6,
        alpha=0.75 if filled else 0.95,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.hlines(
        mean,
        x - width / 2,
        x + width / 2,
        colors="white" if filled else color,
        linewidth=0.7,
        zorder=5,
    )


def plot_method_panel(
    ax,
    df: pd.DataFrame,
    model_set: str,
    method: str,
    show_xticks: bool,
    normalized: bool,
    panel_label: str | None = None,
    show_legend: bool = False,
):
    data, subjects = get_panel_data(df, model_set)
    is_mrsa = method == "mRSA"

    def order_key(model):
        vals = data[model].get("mRSA", {}).get("controversial", {})
        return np.mean(list(vals.values())) if vals else -999

    order = [model for model in data if method in data[model]]
    order.sort(key=order_key, reverse=True)

    x = np.arange(len(order))
    present_stim_types = set()
    if normalized:
        ax.axhline(1.0, color="#666666", linewidth=0.6, linestyle="--", alpha=0.45, zorder=0)
    for i, model in enumerate(order):
        per_method = data[model].get(method, {})
        per_subject_vals = {}
        for stim_type in STIM_TYPES:
            vals_dict = per_method.get(stim_type, {})
            if not vals_dict:
                continue
            present_stim_types.add(stim_type)
            vals = np.asarray(list(vals_dict.values()), dtype=float)
            sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            draw_box(
                ax,
                x[i] + STIM_OFFSET[stim_type],
                float(vals.mean()),
                float(sem),
                filled=is_mrsa,
                color=STIM_COLOR[stim_type],
            )
            per_subject_vals[stim_type] = vals_dict

        for subject in subjects:
            xs, ys = [], []
            for stim_type in STIM_TYPES:
                vals_dict = per_subject_vals.get(stim_type, {})
                if subject in vals_dict:
                    xs.append(x[i] + STIM_OFFSET[stim_type])
                    ys.append(vals_dict[subject])
            if len(xs) >= 2:
                ax.plot(xs, ys, color="#888888", linewidth=0.4, alpha=0.4, zorder=2)
            for stim_type in STIM_TYPES:
                vals_dict = per_subject_vals.get(stim_type, {})
                if subject in vals_dict:
                    ax.scatter(
                        x[i] + STIM_OFFSET[stim_type],
                        vals_dict[subject],
                        s=8,
                        facecolors=STIM_COLOR[stim_type] if is_mrsa else "none",
                        edgecolors=STIM_COLOR[stim_type],
                        linewidths=0.5,
                        marker="o" if is_mrsa else "D",
                        zorder=6,
                        alpha=0.9,
                    )

    ratio = spread_ratio(df, model_set, method)
    if ratio is not None:
        ratio_font = 8 if len(order) <= 6 else FONT["small"]
        ax.text(
            0.5,
            1.005,
            f"spread ratio: {ratio:.2f}x",
            transform=ax.transAxes,
            fontsize=ratio_font,
            fontweight="bold",
            color="#9A4500" if ratio >= 1 else "#666666",
            ha="center",
            va="bottom",
            zorder=10,
        )

    ax.set_xticks(x)
    if show_xticks:
        ax.set_xticklabels(
            [SHORT_NAMES.get(m, constants.MODEL_DISPLAY_NAMES.get(m, m)) for m in order],
            rotation=45,
            ha="right",
        )
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

    if panel_label:
        add_panel_label(ax, panel_label, x=-0.04, y=1.05)

    if show_legend:
        handles = []
        if normalized:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="#666666",
                    linestyle="--",
                    linewidth=0.8,
                    label="Noise ceiling",
                )
            )
        for stim_type in STIM_TYPES:
            if stim_type not in present_stim_types:
                continue
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=STIM_COLOR[stim_type],
                    markeredgecolor=STIM_COLOR[stim_type],
                    markersize=4,
                    label=STIM_DISPLAY[stim_type],
                )
            )
        ax.legend(
            handles=handles,
            loc="upper right",
            frameon=True,
            framealpha=0.92,
            edgecolor="none",
            fontsize=FONT["small"],
            ncol=1,
            handletextpad=0.4,
            handlelength=1.4,
        )


def plot_roi(
    df: pd.DataFrame,
    roi: str,
    out_dir: Path,
    stem: str,
    normalized: bool,
) -> None:
    roi_df = subject_scores(df[df["roi"].eq(roi)])
    if roi_df.empty:
        print(f"{roi}: no rows, skipping")
        return

    ratios = [max(len(constants.MODEL_SETS[model_set]), 4) for model_set in PANEL_ORDER]
    fig = plt.figure(figsize=(W_DOUBLE, 9.0))
    gs = fig.add_gridspec(
        2,
        len(PANEL_ORDER),
        width_ratios=ratios,
        wspace=0.06,
        hspace=0.18,
        left=0.05,
        right=0.99,
        top=0.94,
        bottom=0.16,
    )

    ylims = {}
    for method in ("mRSA", "fRSA"):
        vals = roi_df[roi_df["method"].eq(method)]["score"].replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            ylims[method] = (0.0, 1.05 if normalized else 0.1)
            continue
        top = max(float(vals.max()) * 1.10, 1.05 if normalized else 0.0)
        bottom = min(0.0, float(vals.min()) * 1.10)
        ylims[method] = (bottom, top)

    panel_letters = list("abcde")
    for row, method in enumerate(["mRSA", "fRSA"]):
        for col, model_set in enumerate(PANEL_ORDER):
            ax = fig.add_subplot(gs[row, col])
            ax.set_ylim(*ylims[method])
            plot_method_panel(
                ax,
                roi_df,
                model_set,
                method,
                show_xticks=(row == 1),
                normalized=normalized,
                panel_label=panel_letters[col] if row == 0 else None,
                show_legend=(row == 0 and col == 0),
            )
            if col == 0:
                if normalized:
                    label = "NC-normalized Mixed RSA" if method == "mRSA" else "NC-normalized Fixed RSA"
                else:
                    label = ("Mixed RSA " if method == "mRSA" else "Fixed RSA ") + r"($r_s$)"
                ax.set_ylabel(label)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)
            if row == 0:
                ax.set_title(TITLE[model_set], y=1.08)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / f"{stem}.pdf"
    out_png = out_dir / f"{stem}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


def parse_rois(value: str, available: list[str]) -> list[str]:
    if value == "all":
        return available
    rois = [r.strip() for r in value.split(",") if r.strip()]
    missing = sorted(set(rois) - set(available))
    if missing:
        raise ValueError(f"Requested ROIs are absent from score tables: {missing}")
    return rois


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi", default="ventral_lateral_floc", help="ROI name, comma-list, or all")
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--rdm-noise-ceilings", type=Path, default=DEFAULT_RDM_NC)
    parser.add_argument("--between-noise-ceilings", type=Path, default=DEFAULT_BETWEEN_NC)
    parser.add_argument("--raw", action="store_true", help="Plot raw RSA scores instead of NC-normalized scores")
    args = parser.parse_args()

    all_df = load_scores()
    available = sorted(all_df["roi"].unique())
    rois = parse_rois(args.roi, available)
    df = all_df[all_df["roi"].isin(rois)].copy()
    normalized = not args.raw
    if normalized:
        rdm_nc, between_nc = load_noise_ceilings(args.rdm_noise_ceilings, args.between_noise_ceilings)
        df = normalize_scores(df, rdm_nc, between_nc)

    for roi in rois:
        if roi == "ventral_lateral_floc":
            plot_roi(df, roi, args.out_dir, "brain_alignment_improved_with_shared", normalized)
        if len(rois) > 1 or roi != "ventral_lateral_floc":
            roi_dir = args.out_dir / "brain_alignment_by_roi"
            plot_roi(
                df,
                roi,
                roi_dir,
                f"brain_alignment_improved_with_shared_{roi}",
                normalized,
            )


if __name__ == "__main__":
    main()
