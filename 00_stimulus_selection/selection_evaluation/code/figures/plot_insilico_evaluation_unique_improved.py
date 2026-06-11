#!/usr/bin/env python3
"""
Improved in-silico evaluation figure.

Fixes vs. original:
- Okabe-Ito model-set palette (consistent across all paper figures)
- Panel b: fixed- and mixed-RSA summaries are vertically stacked in the same
  figure row, with explicit Selected/Random uncertainty and percentage
  improvement labels.
- x-axis is plotted as relative SNR, the inverse of the injected Gaussian
  noise standard deviation multiplier, so moving right means higher-quality
  data. The empirical calibration point is at SNR = 1.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_PAPER = _SCRIPT.parents[2]


def _find_share_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (
            (candidate / "pyproject.toml").exists()
            and (candidate / "00_stimulus_selection").exists()
            and (candidate / "shared" / "code" / "paper_helpers").exists()
        ):
            return candidate
    return _PAPER.parents[1]


SHARE_ROOT = _find_share_root(_SCRIPT)
HELPERS_DIR = SHARE_ROOT / "shared" / "code" / "paper_helpers"
sys.path.insert(0, str(HELPERS_DIR))
sys.path.insert(0, str(HELPERS_DIR / "figures"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from style_improved import (
    apply_style, FONT, DPI, W_DOUBLE,
    MODEL_SET_ORDER, MODEL_SET_DISPLAY_SHORT,
    add_panel_label,
)

apply_style()

FIGURES_DIR = _PAPER / "figures"
INSILICO_CURVE_DIR = FIGURES_DIR / "insilico_curve"
MANUSCRIPT_FIGURES_DIR = SHARE_ROOT / "06_manuscript" / "figures" / "01_insilico"
DATA_SUFFIX = "_unique_boot"
ENCODING_TRACKS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
DUPLICATE_CORR_THRESHOLD = 0.9999
ERROR_CLIP_TOL = 1e-8
EMPIRICAL_SNR = 1.0
SNR_TICKS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
SNR_TICK_LABELS = ["0.01", "0.03", "0.1", "0.3", "1", "3", "10"]
OUTPUT_TARGETS = [
    (INSILICO_CURVE_DIR, INSILICO_CURVE_DIR / "png"),
    (MANUSCRIPT_FIGURES_DIR, MANUSCRIPT_FIGURES_DIR),
]

# Match the composite Figure 1 bottom-row model-set palette.
MODEL_SET_COLORS = {
    "all_models": "#222222",
    "sota": "#6A3D9A",
    "training_objective": "#8C564B",
    "architecture": "#009E73",
    "dataset": "#E7298A",
}


def model_set_color(name: str) -> str:
    return MODEL_SET_COLORS.get(name, "#666666")


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
    df["snr"] = 1.0 / df["noise_multiplier"].astype(float)
    return correct_duplicate_error_columns(df, model_set)


def load_auc(model_set: str) -> pd.DataFrame:
    path = config.EVAL_DATA_DIR / f"{model_set}{DATA_SUFFIX}" / "auc_significance.csv"
    if not path.exists():
        return pd.DataFrame()
    return correct_duplicate_auc_columns(pd.read_csv(path), model_set)


def get_sel_rand(df, track_names, x_col="snr"):
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


def _style_curve_axis(
    ax,
    title: str,
    show_xlabel: bool = True,
    show_empirical_label: bool = True,
) -> None:

    # Empirical calibration point: the original noise multiplier equals 1 here.
    ax.axvline(
        EMPIRICAL_SNR,
        color="#444444",
        lw=1.4,
        ls=(0, (4, 2)),
        alpha=0.85,
        zorder=2,
    )
    if show_empirical_label:
        ax.text(
            1.08,
            0.97,
            "Empirical\nSNR",
            fontsize=FONT["small"] - 1,
            color="#222222",
            ha="left",
            va="top",
            fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )

    ax.set_xlabel("Relative signal-to-noise ratio" if show_xlabel else "")
    ax.set_xscale("log")
    ax.set_xlim(0.009, 11.0)
    ax.set_xticks(SNR_TICKS)
    ax.set_xticklabels(SNR_TICK_LABELS)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, pad=4)


def plot_curves(ax, track_key, title, show_legend=False):
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

    ax.plot([], [], "k-",  lw=1.5, label="Controversial")
    ax.plot([], [], "k--", lw=1.0, alpha=0.45, label="Random")
    _style_curve_axis(ax, title)
    if show_legend:
        ax.legend(loc="lower right", ncol=2, frameon=True, framealpha=0.92,
                  edgecolor="none", columnspacing=0.7,
                  handletextpad=0.4, fontsize=FONT["small"])


def plot_faceted_curve(
    ax,
    model_set: str,
    track_key: str,
    title: str,
    show_ylabel: bool,
    show_xlabel: bool,
    show_legend: bool = False,
    show_empirical_label: bool = True,
) -> None:
    df = load_discriminability(model_set)
    tracks = ["raw"] if track_key == "raw" else ENCODING_TRACKS
    sel, rand, rand_std, x = get_sel_rand(df, tracks)
    col = model_set_color(model_set)

    if sel is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.plot(x, rand, color="#6F6F6F", lw=1.0, ls="--", alpha=0.75, label="Random")
        ax.plot(x, sel, color=col, lw=1.35, ls="-", alpha=0.96, label="Controversial")
        if rand_std is not None:
            ax.fill_between(
                x, rand - rand_std, rand + rand_std,
                color="#6F6F6F", alpha=0.10, linewidth=0,
            )

    _style_curve_axis(
        ax,
        title,
        show_xlabel=False,
        show_empirical_label=show_empirical_label,
    )
    ax.set_xticks([0.01, 0.1, 1, 10])
    ax.set_xticklabels(["0.01", "0.1", "1", "10"])
    if not show_xlabel:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    ax.set_ylabel("")
    if not show_ylabel:
        plt.setp(ax.get_yticklabels(), visible=False)
    if show_legend:
        ax.legend(
            loc="lower right",
            frameon=True,
            framealpha=0.92,
            edgecolor="none",
            fontsize=FONT["small"],
            handlelength=1.5,
            handletextpad=0.4,
        )


def _empty_bar_data() -> dict[str, list]:
    return {
        "labels": [],
        "colors": [],
        "random": [],
        "selected": [],
        "random_err": [],
        "selected_err": [],
    }


def _append_bar_data(data, label, color, random, selected, random_err, selected_err):
    data["labels"].append(label)
    data["colors"].append(color)
    data["random"].append(float(random))
    data["selected"].append(float(selected))
    data["random_err"].append(float(random_err))
    data["selected_err"].append(float(selected_err))


def _mean_error_from_std(rows: pd.DataFrame, std_columns: list[str]) -> float:
    for col in std_columns:
        if col not in rows.columns:
            continue
        vals = pd.to_numeric(rows[col], errors="coerce").dropna().astype(float)
        if vals.empty:
            continue
        return float(np.sqrt((vals.to_numpy() ** 2).mean() / len(vals)))
    return 0.0


def _accuracy_at_empirical_snr(df: pd.DataFrame, tracks: list[str]) -> tuple[float, float, float, float] | None:
    sub = df[
        df["track"].isin(tracks)
        & np.isclose(df["noise_multiplier"].astype(float), 1.0)
    ].copy()
    if sub.empty:
        return None

    selected_rows = sub[sub["subset_type"] == "selected"]
    random_rows = sub[sub["subset_type"] == "random"]
    if selected_rows.empty or random_rows.empty:
        return None

    selected = 1.0 - selected_rows["error_prob"].astype(float).mean()
    random = 1.0 - random_rows["error_prob"].astype(float).mean()
    selected_err = _mean_error_from_std(
        selected_rows, ["error_prob_mc_std", "error_prob_std"]
    )
    random_err = _mean_error_from_std(
        random_rows, ["error_prob_std", "error_prob_mc_std"]
    )
    return random, selected, random_err, selected_err


def collect_auc_bar_data() -> dict[str, dict[str, list]]:
    """Selected vs Random accuracy AUC for raw and encoding."""
    raw = _empty_bar_data()
    encoding = _empty_bar_data()

    for ms in MODEL_SET_ORDER:
        auc_df = load_auc(ms)
        if auc_df.empty:
            continue

        raw_row = auc_df[auc_df["track"] == "raw"]
        enc_rows = auc_df[auc_df["track"].isin(ENCODING_TRACKS)]
        if raw_row.empty or enc_rows.empty:
            continue
        raw_row = raw_row.iloc[0]

        label = MODEL_SET_DISPLAY_SHORT[ms]
        color = model_set_color(ms)
        _append_bar_data(
            raw,
            label,
            color,
            random=1 - raw_row["random_auc_mean"],
            selected=1 - raw_row["selected_auc"],
            random_err=raw_row["random_auc_subset_std"],
            selected_err=raw_row["selected_auc_mc_std"],
        )

        n_subj = len(enc_rows)
        _append_bar_data(
            encoding,
            label,
            color,
            random=1 - enc_rows["random_auc_mean"].mean(),
            selected=1 - enc_rows["selected_auc"].mean(),
            random_err=float(
                np.sqrt((enc_rows["random_auc_subset_std"] ** 2).mean() / n_subj)
            ),
            selected_err=float(
                np.sqrt((enc_rows["selected_auc_mc_std"] ** 2).mean() / n_subj)
            ),
        )

    return {"raw": raw, "encoding": encoding}


def collect_empirical_snr_bar_data() -> dict[str, dict[str, list]]:
    """Selected vs Random model-recovery accuracy at empirical SNR."""
    raw = _empty_bar_data()
    encoding = _empty_bar_data()

    for ms in MODEL_SET_ORDER:
        df = load_discriminability(ms)
        if df.empty:
            continue

        raw_stats = _accuracy_at_empirical_snr(df, ["raw"])
        enc_stats = _accuracy_at_empirical_snr(df, ENCODING_TRACKS)
        if raw_stats is None or enc_stats is None:
            continue

        label = MODEL_SET_DISPLAY_SHORT[ms]
        color = model_set_color(ms)
        _append_bar_data(raw, label, color, *raw_stats)
        _append_bar_data(encoding, label, color, *enc_stats)

    return {"raw": raw, "encoding": encoding}


def _bar_ylim(data: dict[str, list]) -> tuple[float, float]:
    random = np.asarray(data["random"], dtype=float)
    selected = np.asarray(data["selected"], dtype=float)
    random_err = np.asarray(data["random_err"], dtype=float)
    selected_err = np.asarray(data["selected_err"], dtype=float)
    vals = np.concatenate([random - random_err, selected - selected_err])
    tops = np.concatenate([random + random_err, selected + selected_err])
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(tops))
    span = max(hi - lo, 0.08)
    bottom = 0.0
    top = min(1.12, max(1.05, hi + 0.55 * span))
    if top - bottom < 0.18:
        center = (top + bottom) / 2
        bottom = max(0.0, center - 0.09)
        top = min(1.12, center + 0.09)
    return bottom, top


def plot_summary_bars(
    ax,
    data: dict[str, list],
    title: str,
    ylabel: str,
    show_xticklabels: bool,
    show_legend: bool,
):
    """Paired Controversial/Random bars for one panel-C sub-axis."""
    if not data["labels"]:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    x = np.arange(len(data["labels"]), dtype=float)
    width = 0.34
    err_kw = dict(ecolor="0.3", capsize=2, elinewidth=0.8)

    ax.bar(
        x - width / 2,
        data["random"],
        width,
        yerr=data["random_err"],
        color="#DDDDDD",
        alpha=0.85,
        edgecolor="#666666",
        linewidth=0.5,
        hatch="//",
        label="Random",
        error_kw=err_kw,
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        data["selected"],
        width,
        yerr=data["selected_err"],
        color=data["colors"],
        alpha=0.90,
        edgecolor="white",
        linewidth=0.5,
        label="Controversial",
        error_kw=err_kw,
        zorder=3,
    )

    bottom, top = _bar_ylim(data)
    y_span = top - bottom
    for i, color in enumerate(data["colors"]):
        bar_top = max(
            data["random"][i] + data["random_err"][i],
            data["selected"][i] + data["selected_err"][i],
        )
        pct = (
            (data["selected"][i] - data["random"][i])
            / max(data["random"][i], 1e-6)
            * 100
        )
        ax.text(
            x[i],
            bar_top + 0.05 * y_span,
            f"{pct:+.0f}%",
            fontsize=FONT["small"],
            ha="center",
            va="bottom",
            color=color,
            alpha=0.95,
        )

    ax.set_xticks(x)
    ax.set_xlim(-0.5, len(data["labels"]) - 0.5)
    if show_xticklabels:
        ax.set_xticklabels(data["labels"], rotation=25, ha="right")
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom, top)
    if title:
        ax.set_title(title, pad=3)
    ax.grid(axis="y", alpha=0.35, zorder=0)
    if show_legend:
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.0, -0.02),
            ncol=2,
            frameon=True,
            framealpha=0.88,
            edgecolor="none",
            columnspacing=0.6,
            handlelength=1.2,
            handletextpad=0.3,
            borderpad=0.25,
            labelspacing=0.25,
            fontsize=FONT["small"],
        )


def plot_panel_c(ax_raw, ax_encoding, metric: str):
    if metric == "auc":
        data = collect_auc_bar_data()
        ylabel = "Accuracy\nAUC"
        raw_title = ""
        encoding_title = ""
    elif metric == "empirical_snr_accuracy":
        data = collect_empirical_snr_bar_data()
        ylabel = "Accuracy"
        raw_title = ""
        encoding_title = ""
    else:
        raise ValueError(f"Unknown panel-C metric: {metric}")

    plot_summary_bars(
        ax_raw,
        data["raw"],
        raw_title,
        ylabel,
        show_xticklabels=False,
        show_legend=True,
    )
    plot_summary_bars(
        ax_encoding,
        data["encoding"],
        encoding_title,
        ylabel,
        show_xticklabels=True,
        show_legend=False,
    )


def draw_auc_schematic(
    fig,
    x_center: float,
    y_center: float,
    width: float = 0.052,
    height: float = 0.068,
) -> None:
    """Tiny non-data schematic explaining that panel c summarizes curve area."""
    ax = fig.add_axes([x_center - width / 2, y_center - height / 2, width, height])
    t = np.linspace(0.04, 0.96, 80)
    y = 0.16 + 0.72 / (1 + np.exp(-9 * (t - 0.45)))
    ax.fill_between(t, 0.12, y, color="#D7E6F5", alpha=0.90, linewidth=0)
    ax.plot(t, y, color="#3B77A8", lw=1.2)
    ax.plot([0.04, 0.96], [0.12, 0.12], color="#666666", lw=0.5)
    ax.plot([0.04, 0.04], [0.12, 0.92], color="#666666", lw=0.5)
    ax.text(0.92, 0.28, "AUC", ha="right", va="center", fontsize=FONT["small"], color="#234E70")
    ax.set_axis_off()


def save_outputs(fig, stem: str):
    for pdf_dir, png_dir in OUTPUT_TARGETS:
        pdf_dir.mkdir(parents=True, exist_ok=True)
        png_dir.mkdir(parents=True, exist_ok=True)
        pdf_out = pdf_dir / f"{stem}.pdf"
        png_out = png_dir / f"{stem}.png"
        fig.savefig(pdf_out)
        fig.savefig(png_out, dpi=DPI)
        print(f"Saved {pdf_out}")
        print(f"Saved {png_out}")


def make_figure(panel_c_metric: str, output_stem: str):
    fig = plt.figure(figsize=(W_DOUBLE, 5.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.7, 1.25], wspace=0.34,
                          left=0.065, right=0.99, top=0.86, bottom=0.20)
    curve_gs = gs[0, 0].subgridspec(2, 5, hspace=0.28, wspace=0.34)
    curve_axes = np.empty((2, len(MODEL_SET_ORDER)), dtype=object)
    for row in range(2):
        for col, model_set in enumerate(MODEL_SET_ORDER):
            sharey = curve_axes[0, 0] if row or col else None
            ax = fig.add_subplot(curve_gs[row, col], sharey=sharey)
            curve_axes[row, col] = ax

    c_gs = gs[0, 1].subgridspec(2, 1, hspace=0.42)
    ax_c_raw = fig.add_subplot(c_gs[0, 0])
    ax_c_encoding = fig.add_subplot(c_gs[1, 0])

    for col, model_set in enumerate(MODEL_SET_ORDER):
        plot_faceted_curve(
            curve_axes[0, col],
            model_set,
            "raw",
            MODEL_SET_DISPLAY_SHORT[model_set],
            show_ylabel=(col == 0),
            show_xlabel=False,
            show_legend=(col == len(MODEL_SET_ORDER) - 1),
            show_empirical_label=(col == len(MODEL_SET_ORDER) - 1),
        )
        plot_faceted_curve(
            curve_axes[1, col],
            model_set,
            "encoding",
            "",
            show_ylabel=(col == 0),
            show_xlabel=True,
            show_legend=False,
            show_empirical_label=False,
        )

    curve_axes[0, 0].text(
        -0.78, 0.50, "Fixed RSA",
        ha="right", va="center", rotation=90,
        fontsize=FONT["title"] - 1,
        transform=curve_axes[0, 0].transAxes,
    )
    curve_axes[1, 0].text(
        -0.78, 0.50, "Mixed RSA",
        ha="right", va="center", rotation=90,
        fontsize=FONT["title"] - 1,
        transform=curve_axes[1, 0].transAxes,
    )
    fig.text(
        0.030,
        0.56,
        "Model recovery accuracy",
        rotation=90,
        ha="center",
        va="center",
        fontsize=FONT["small"] - 1,
    )
    fig.text(
        0.365,
        0.105,
        "Relative signal-to-noise ratio",
        ha="center",
        va="center",
        fontsize=FONT["small"],
    )

    plot_panel_c(ax_c_raw, ax_c_encoding, panel_c_metric)

    curve_left = curve_axes[0, 0].get_position().x0
    curve_right = curve_axes[0, -1].get_position().x1
    c_left = ax_c_raw.get_position().x0
    c_right = ax_c_raw.get_position().x1
    title_y = 0.955
    fig.text(
        (curve_left + curve_right) / 2,
        title_y,
        "Recovery across noise levels",
        ha="center",
        va="top",
        fontsize=FONT["title"] + 1,
        fontweight="bold",
    )
    fig.text(
        (c_left + c_right) / 2,
        title_y,
        "Area under recovery curve",
        ha="center",
        va="top",
        fontsize=FONT["title"] + 1,
        fontweight="bold",
    )
    draw_auc_schematic(fig, x_center=(curve_right + c_left) / 2, y_center=0.56, width=0.042)

    add_panel_label(curve_axes[0, 0], "a", x=-0.38, y=1.17)
    add_panel_label(ax_c_raw, "b", x=-0.16, y=1.12)

    save_outputs(fig, output_stem)
    plt.close(fig)


def main():
    make_figure("auc", "insilico_evaluation_unique_improved")


if __name__ == "__main__":
    main()
