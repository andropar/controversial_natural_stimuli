#!/usr/bin/env python3
"""
Plot 02c — wRSA on deterministic vicco subsets that span (and exceed)
cstim mean low-level distance.

Three outputs:
  low_level_deterministic_curve.{pdf,png}     scatter of (mean_low, wRSA) per
                                              subset, with cstim per-set markers
  low_level_deterministic_per_model.{pdf,png} per-model bars showing
                                              cstim_wrsa vs match_<set>_wrsa
                                              vs top100_wrsa
  low_level_dissociation.{pdf,png}            publication summary: low-level,
                                              PPCA OOD, and reliability controls
"""

import sys
import warnings
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
_SHARE_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_SHARE_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from cstims import constants, paths
from cstims.paper.style_improved import apply_style, FONT, DPI, W_DOUBLE

apply_style()

OOD = _PAPER / "results"
RSA_DATA_DIR = _SHARE_ROOT / "01_brain_model_alignment" / "results" / "rsa_scores"
STATS_DATA_DIR = _SHARE_ROOT / "02_alignment_reliability" / "results"
DET = OOD / "wrsa_low_level_subsets.csv"
COMP = OOD / "wrsa_low_level_subsets_comparison.csv"
FIGS = Path(__file__).resolve().parents[2] / "figures"
PNG_DIR = FIGS / "png"
SUPPLEMENTARY_DIR = FIGS / "supplementary"
SUPPLEMENTARY_PNG_DIR = SUPPLEMENTARY_DIR / "png"

CSTIM_SETS = ["all_models", "architecture", "training_objective", "sota", "dataset"]
SET_COLOR = {
    "all_models":         "#d6604d",
    "architecture":       "#fb6a4a",
    "training_objective": "#fdae6b",
    "sota":               "#bf812d",
    "dataset":            "#c994c7",
}
SUBSET_ORDER = ["bottom100", "middle100",
                "match_architecture", "match_training_objective",
                "match_sota", "match_dataset", "match_all_models", "top100"]

GROUP_LABEL = {
    "all_models":         "All\nmodels",
    "architecture":       "Arch.",
    "training_objective": "Train.\nobj.",
    "sota":               "SOTA",
    "dataset":            "Dataset",
}

OKABE = {
    "orange": "#e69f00",
    "sky": "#56b4e9",
    "green": "#009e73",
    "blue": "#0072b2",
    "vermillion": "#d55e00",
    "purple": "#cc79a7",
    "black": "#000000",
}

OOD_DELTA_SPACES = [
    ("delta_ood_pred", "Predicted responses", OKABE["blue"]),
    ("delta_ood_feature", "Raw features", OKABE["vermillion"]),
]


def _sem(x):
    x = pd.Series(x).dropna()
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def _load_wrsa():
    dfs = []
    for subject in constants.SUBJECTS:
        p = RSA_DATA_DIR / subject / "wrsa_transfer_scores.csv"
        if p.exists():
            dfs.append(pd.read_csv(p))
    if not dfs:
        raise FileNotFoundError("No wrsa_transfer_scores.csv files found")
    return pd.concat(dfs, ignore_index=True)


def _load_noise_ceilings():
    """Return cstim NC per subject/set and mean vicco NC per subject.

    `noise_ceiling_spearman` is split-half RDM reliability. The upper bound
    for model-brain correlations is sqrt(reliability), used separately below.
    """
    nc = pd.read_csv(STATS_DATA_DIR / "rdm_noise_ceilings.csv")
    cstim_nc = nc[nc["stimulus_type"] == "controversial"][
        ["subject", "group", "noise_ceiling_spearman"]
    ].rename(columns={"group": "model_set",
                      "noise_ceiling_spearman": "cstim_nc"})
    vicco_nc = (nc[nc["stimulus_type"] == "vicco"]
                .groupby("subject")["noise_ceiling_spearman"]
                .mean()
                .reset_index()
                .rename(columns={"noise_ceiling_spearman": "vicco_nc"}))
    return cstim_nc, vicco_nc


def _control_absolute_values():
    """Subject-level absolute values used in the publication control figure."""
    det = pd.read_csv(DET)
    wrsa = _load_wrsa()
    cstim_nc, vicco_nc = _load_noise_ceilings()

    low_rows = []
    norm_rows = []
    rel_rows = []

    for model_set in CSTIM_SETS:
        models = constants.MODEL_SETS[model_set]

        # Raw wRSA cstim subject means for this set/roster.
        cstim = (wrsa[(wrsa["stimulus_type"] == "controversial")
                      & (wrsa["model_set"] == model_set)]
                 .groupby("subject")["wrsa_transfer"]
                 .mean()
                 .rename("cstim_wrsa"))

        # Baseline (full vicco), same model roster, subject means.
        baseline = (wrsa[(wrsa["stimulus_type"] == "vicco")
                         & (wrsa["model_set"] == model_set)]
                    .groupby(["subject", "model"])["wrsa_transfer"]
                    .mean()
                    .groupby("subject")
                    .mean()
                    .rename("baseline_wrsa"))

        # Deterministic vicco controls, same model roster, subject means.
        # Use distribution-shape match (mean + spread + shape) rather than
        # mean-only match. dist_match_<set> is the stronger control.
        dist_matched = (det[(det["subset"] == f"dist_match_{model_set}")
                            & (det["model"].isin(models))]
                        .groupby("subject")["wrsa"]
                        .mean()
                        .rename("dist_matched_wrsa"))
        high_low = (det[(det["subset"] == "top100")
                        & (det["model"].isin(models))]
                    .groupby("subject")["wrsa"]
                    .mean()
                    .rename("high_low_wrsa"))
        low = (pd.concat([cstim, baseline, dist_matched, high_low], axis=1)
               .dropna()
               .reset_index())
        for _, row in low.iterrows():
            low_rows.append({
                "model_set": model_set,
                "subject": row["subject"],
                "condition": "Cstim",
                "wrsa": row["cstim_wrsa"],
            })
            low_rows.append({
                "model_set": model_set,
                "subject": row["subject"],
                "condition": "Distribution-matched baseline",
                "wrsa": row["dist_matched_wrsa"],
            })
            low_rows.append({
                "model_set": model_set,
                "subject": row["subject"],
                "condition": "High-distance baseline",
                "wrsa": row["high_low_wrsa"],
            })
            low_rows.append({
                "model_set": model_set,
                "subject": row["subject"],
                "condition": "Baseline",
                "wrsa": row["baseline_wrsa"],
            })

        # Absolute RDM reliabilities for cstim and the same subject's vicco data.
        rel = (cstim_nc[cstim_nc["model_set"] == model_set]
               .merge(vicco_nc, on="subject"))
        for _, row in rel.iterrows():
            rel_rows.append({
                "model_set": model_set,
                "subject": row["subject"],
                "condition": "Cstim",
                "rdm_reliability": row["cstim_nc"],
            })
            rel_rows.append({
                "model_set": model_set,
                "subject": row["subject"],
                "condition": "Baseline",
                "rdm_reliability": row["vicco_nc"],
            })

        # Correlation-ceiling-normalized wRSA, computed directly from raw wRSA
        # and current split-half RDM reliabilities.
        cstim_norm = (wrsa[(wrsa["stimulus_type"] == "controversial")
                           & (wrsa["model_set"] == model_set)]
                      .merge(cstim_nc[cstim_nc["model_set"] == model_set],
                             on=["subject", "model_set"]))
        cstim_norm = cstim_norm[cstim_norm["cstim_nc"] > 0].copy()
        cstim_norm["wrsa_norm"] = (
            cstim_norm["wrsa_transfer"] / np.sqrt(cstim_norm["cstim_nc"])
        )
        cstim_norm = (cstim_norm.groupby("subject")["wrsa_norm"]
                      .mean()
                      .rename("cstim_norm"))

        vicco_norm = (wrsa[(wrsa["stimulus_type"] == "vicco")
                           & (wrsa["model_set"] == model_set)]
                      .merge(vicco_nc, on="subject"))
        vicco_norm = vicco_norm[vicco_norm["vicco_nc"] > 0].copy()
        vicco_norm["wrsa_norm"] = (
            vicco_norm["wrsa_transfer"] / np.sqrt(vicco_norm["vicco_nc"])
        )
        vicco_norm = (vicco_norm.groupby(["subject", "model"])["wrsa_norm"]
                      .mean()
                      .groupby("subject")
                      .mean()
                      .rename("vicco_norm"))

        norm = pd.concat([cstim_norm, vicco_norm], axis=1).dropna().reset_index()
        for _, row in norm.iterrows():
            norm_rows.append({
                "model_set": model_set,
                "subject": row["subject"],
                "condition": "Cstim",
                "norm_wrsa": row["cstim_norm"],
            })
            norm_rows.append({
                "model_set": model_set,
                "subject": row["subject"],
                "condition": "Baseline",
                "norm_wrsa": row["vicco_norm"],
            })

    return (pd.DataFrame(low_rows),
            pd.DataFrame(rel_rows),
            pd.DataFrame(norm_rows))


def _ood_alignment_delta_points():
    """Model-level OOD shifts and cstim-vicco wRSA drops."""
    ood = pd.read_csv(OOD / "pca_loglik.csv")
    wrsa = _load_wrsa()

    cstim_wrsa = (
        wrsa[wrsa["stimulus_type"] == "controversial"]
        .groupby(["subject", "model_set", "model"])["wrsa_transfer"]
        .mean()
        .reset_index()
        .rename(columns={"model_set": "group",
                         "wrsa_transfer": "wrsa_cstim"})
    )
    vicco_wrsa = (
        wrsa[wrsa["stimulus_type"] == "vicco"]
        .groupby(["subject", "model_set", "model"])["wrsa_transfer"]
        .mean()
        .reset_index()
        .rename(columns={"model_set": "group",
                         "wrsa_transfer": "wrsa_vicco"})
    )
    delta_wrsa = cstim_wrsa.merge(
        vicco_wrsa, on=["subject", "group", "model"]
    )
    delta_wrsa["delta_alignment"] = (
        delta_wrsa["wrsa_cstim"] - delta_wrsa["wrsa_vicco"]
    )

    cstim_ood = (
        ood[ood["stimulus_group"].isin(CSTIM_SETS)]
        .groupby(["subject", "stimulus_group", "model"])[
            ["loglik_feature_z", "loglik_pred_z"]
        ]
        .mean()
        .reset_index()
        .rename(columns={
            "stimulus_group": "group",
            "loglik_feature_z": "ood_feature_cstim",
            "loglik_pred_z": "ood_pred_cstim",
        })
    )
    vicco_ood = (
        ood[ood["stimulus_group"] == "vicco"]
        .groupby(["subject", "model"])[["loglik_feature_z", "loglik_pred_z"]]
        .mean()
        .reset_index()
        .rename(columns={
            "loglik_feature_z": "ood_feature_vicco",
            "loglik_pred_z": "ood_pred_vicco",
        })
    )
    delta_ood = cstim_ood.merge(vicco_ood, on=["subject", "model"])
    delta_ood["delta_ood_feature"] = (
        delta_ood["ood_feature_cstim"] - delta_ood["ood_feature_vicco"]
    )
    delta_ood["delta_ood_pred"] = (
        delta_ood["ood_pred_cstim"] - delta_ood["ood_pred_vicco"]
    )

    merged = delta_wrsa.merge(
        delta_ood[[
            "subject", "group", "model", "delta_ood_feature", "delta_ood_pred",
        ]],
        on=["subject", "group", "model"],
    )
    return (
        merged
        .groupby(["group", "model"])[
            ["delta_alignment", "delta_ood_feature", "delta_ood_pred"]
        ]
        .mean()
        .reset_index()
    )


def _panel_label_at(fig, x, y, label):
    """Place a panel label in figure coordinates."""
    fig.text(x, y, label,
             fontsize=FONT["panel_label"], fontweight="bold",
             ha="left", va="bottom")


def _condition_offsets(n):
    if n == 2:
        return np.array([-0.13, 0.13])
    if n == 3:
        return np.array([-0.22, 0.00, 0.22])
    return np.linspace(-0.24, 0.24, n)


def _plot_absolute_points(ax, df, value, title, ylabel, condition_order,
                          colors, markers, ylim=None, connect=True,
                          legend_loc="upper right", legend_outside=False,
                          secondary_alpha=None, suppress_pts_note=False):
    secondary_alpha = secondary_alpha or {}
    xs = np.arange(len(CSTIM_SETS))
    offsets = _condition_offsets(len(condition_order))
    for i, model_set in enumerate(CSTIM_SETS):
        sub = df[df["model_set"] == model_set]
        if connect:
            wide = sub.pivot_table(index="subject", columns="condition",
                                   values=value, aggfunc="mean")
            for _, row in wide.iterrows():
                vals = [row.get(c, np.nan) for c in condition_order]
                if np.all(np.isfinite(vals)):
                    ax.plot(xs[i] + offsets, vals, color="0.78",
                            lw=0.45, alpha=0.7, zorder=1)
        for j, condition in enumerate(condition_order):
            vals = sub[sub["condition"] == condition][value].dropna().values
            if len(vals) == 0:
                continue
            jitter = np.linspace(-0.035, 0.035, len(vals)) if len(vals) > 1 else [0]
            scatter_alpha = 0.42 * secondary_alpha.get(condition, 1.0)
            mean_alpha = secondary_alpha.get(condition, 1.0)
            ax.scatter(xs[i] + offsets[j] + jitter, vals, s=17,
                       color=colors[condition], marker=markers[condition],
                       alpha=scatter_alpha, edgecolor="none", zorder=2)
            mean = vals.mean()
            sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            ax.errorbar(xs[i] + offsets[j], mean, yerr=sem, fmt=markers[condition],
                        ms=6.5, color=colors[condition],
                        markeredgecolor="black", markeredgewidth=0.5,
                        elinewidth=0.9, capsize=3, capthick=0.9,
                        alpha=mean_alpha, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([GROUP_LABEL[g] for g in CSTIM_SETS])
    ax.set_title(title, fontsize=FONT["title"], pad=5)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="0.88", lw=0.5, zorder=0)
    handles = [
        plt.Line2D([0], [0], marker=markers[c], color="none",
                   markerfacecolor=colors[c], markeredgecolor="black",
                   markeredgewidth=0.5, markersize=6, label=c)
        for c in condition_order
    ]
    if legend_outside:
        ax.legend(handles=handles, frameon=False,
                  loc="upper center", bbox_to_anchor=(0.5, -0.10),
                  fontsize=FONT["legend"], handlelength=1.0,
                  ncol=len(condition_order), columnspacing=1.4,
                  handletextpad=0.4)
    else:
        ncol = 2 if len(condition_order) >= 4 else 1
        ax.legend(handles=handles, frameon=False, loc=legend_loc,
                  fontsize=FONT["legend"], handlelength=1.0, ncol=ncol)
    if not suppress_pts_note:
        ax.text(0.02, 0.96, "points = subjects; mean ± SEM",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=FONT["small"], color="0.25",
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec="none", alpha=0.85))


def _plot_low_level_panel(ax, low_df):
    conditions = [
        "Cstim",
        "Distribution-matched baseline",
        "High-distance baseline",
        "Baseline",
    ]
    # Make the cstim/baseline contrast visually dominant; secondary controls
    # use lighter weight (smaller alpha on the markers below).
    colors = {
        "Cstim": OKABE["vermillion"],
        "Distribution-matched baseline": OKABE["blue"],
        "High-distance baseline": OKABE["green"],
        "Baseline": OKABE["black"],
    }
    markers = {
        "Cstim": "o",
        "Distribution-matched baseline": "o",
        "High-distance baseline": "^",
        "Baseline": "s",
    }
    _plot_absolute_points(
        ax, low_df, "wrsa",
        "Low-level matching does not rescue alignment",
        "Mean mixed RSA",
        conditions, colors, markers,
        ylim=(0.0, 0.68),
        connect=True,
        legend_loc="upper center",
        legend_outside=True,
        secondary_alpha={"High-distance baseline": 0.55},
    )


def _plot_ood_delta_panel(fig, spec, delta_df, shift_df):
    """Simplified panel b: PPCA OOD shift exists (left) but does not consistently
    predict the per-model alignment drop (right).

    Layout: 2 rows × 2 cols. Col 0: per-stimulus PPCA log-lik z (predicted
    responses on top, raw features on bottom). Col 1: per-set Spearman rho
    summaries for alignment drop vs. PPCA OOD shift. The detailed per-set
    scatter matrix is available in the supplement.
    """
    plot_sets = ["all_models", "architecture", "training_objective", "sota", "dataset"]
    set_titles = {
        "all_models": "All",
        "architecture": "Arch.",
        "training_objective": "Train. obj.",
        "sota": "SOTA",
        "dataset": "Dataset",
    }
    set_colors = {
        "all_models":         OKABE["purple"],
        "sota":               OKABE["green"],
        "training_objective": OKABE["blue"],
        "architecture":       OKABE["vermillion"],
        "dataset":            OKABE["orange"],
    }

    gs = spec.subgridspec(
        2, 2,
        width_ratios=[0.62, 0.95],
        wspace=0.32,
        hspace=1.10,
    )
    shift_axes = {}
    pooled_axes = {}
    for row in range(2):
        shift_axes[row] = fig.add_subplot(gs[row, 0])
        pooled_axes[row] = fig.add_subplot(gs[row, 1])

    space_to_value = {
        "delta_ood_pred":    "loglik_pred_z",
        "delta_ood_feature": "loglik_feature_z",
    }
    space_titles = {
        "delta_ood_pred":    "Predicted responses",
        "delta_ood_feature": "Raw features",
    }

    # ---- Left col: per-stimulus PPCA shift strips ----
    for row, (x_col, _, color) in enumerate(OOD_DELTA_SPACES):
        ax_shift = shift_axes[row]
        _plot_shift_strip(
            ax_shift, shift_df,
            group_col="stimulus_group",
            value_col=space_to_value[x_col],
            ylabel=f"{space_titles[x_col]}\nlog-lik z",
            title="OOD shift" if row == 0 else None,
            cstim_color=color,
            baseline_color=OKABE["black"],
            show_xticklabels=(row == 1),
            d_invert=True,
            n_caption="points = stimuli",
        )
        ax_shift.axhline(0, color="0.55", lw=0.6, ls="--", zorder=0)

    # ---- Right col: compact per-set correlation summary ----
    for row, (x_col, _, _) in enumerate(OOD_DELTA_SPACES):
        ax = pooled_axes[row]
        ypos = np.arange(len(plot_sets))[::-1]

        ax.axvline(0, color="0.55", lw=0.7, ls="--", zorder=0)
        ax.axvspan(-0.10, 0.10, color="0.93", zorder=0)

        for yy, ms in zip(ypos, plot_sets):
            sub = delta_df[delta_df["group"] == ms].dropna(
                subset=[x_col, "delta_alignment"])
            x = -sub[x_col].values
            y = -sub["delta_alignment"].values
            rho = np.nan
            if len(x) >= 3 and len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", stats.ConstantInputWarning)
                    rho, _ = stats.spearmanr(x, y)
            if np.isfinite(rho):
                ax.hlines(yy, 0, rho, color="0.70", lw=0.8,
                          alpha=0.45, zorder=1)
                ax.scatter(rho, yy, s=38, color=set_colors[ms],
                           edgecolor="white", linewidth=0.5, zorder=3)
                dx = 0.045 if rho >= 0 else -0.045
                ha = "left" if rho >= 0 else "right"
                ax.text(rho + dx, yy, f"{rho:+.2f}",
                        ha=ha, va="center",
                        fontsize=FONT["small"] - 1, color="0.25")

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-0.6, len(plot_sets) - 0.4)
        ax.set_yticks(ypos)
        ax.set_yticklabels([set_titles[ms] for ms in plot_sets])
        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.grid(axis="x", color="0.90", lw=0.4, zorder=0)
        ax.tick_params(labelsize=FONT["small"])
        if row == 0:
            ax.set_title("Per-set OOD-alignment association",
                         fontsize=FONT["title"], pad=4)
        ax.text(0.02, 0.94, space_titles[x_col],
                transform=ax.transAxes, ha="left", va="top",
                fontsize=FONT["small"] - 1, fontweight="bold", color="0.25")
        if row == 1:
            ax.set_xlabel("Spearman ρ (OOD increase vs. alignment drop)",
                          fontsize=FONT["small"])
        else:
            ax.set_xticklabels([])

    bbox = spec.get_position(fig)
    fig.text(
        bbox.x0, bbox.y1 + 0.050,
        "PPCA OOD shifts do not consistently track alignment drops",
        ha="left", va="bottom", fontsize=FONT["title"],
    )
    fig.text(
        bbox.x0 - 0.035, bbox.y1 + 0.050, "b",
        fontsize=FONT["panel_label"], fontweight="bold",
        ha="left", va="bottom",
    )
    return pooled_axes, shift_axes


def fig_curve():
    df = pd.read_csv(DET)
    comp = pd.read_csv(COMP)

    # Per-subset subject- and model-mean mixed RSA + mean_low
    per_subset = (df.groupby("subset")
                  [["mean_low", "wrsa"]]
                  .agg(["mean", "std", "count"])
                  .reset_index())
    per_subset.columns = ["subset", "mean_low", "mean_low_std", "mean_low_count",
                          "wrsa_mean", "wrsa_std", "wrsa_count"]
    per_subset["sem"] = per_subset["wrsa_std"] / np.sqrt(per_subset["wrsa_count"])
    per_subset = per_subset.set_index("subset").loc[SUBSET_ORDER].reset_index()

    fig, ax = plt.subplots(figsize=(9.5, 5.4), constrained_layout=True)

    # Vicco subset points (with errorbars on mixed RSA)
    ax.errorbar(per_subset["mean_low"], per_subset["wrsa_mean"],
                yerr=per_subset["sem"], fmt="o", color="#2166ac", lw=1.0,
                capsize=3, markersize=8, label="vicco subset (subj×model mean ± SEM)")
    for _, row in per_subset.iterrows():
        ax.annotate(row["subset"], (row["mean_low"], row["wrsa_mean"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=7, color="#1a4575")

    # cstim points: one per set at its (cstim_mean_low, cstim_wrsa)
    cstim_pts = comp[comp["model_set"].isin(CSTIM_SETS)][
        ["model_set", "cstim_mean_low", "cstim_wrsa_mean"]
    ].drop_duplicates()
    for _, row in cstim_pts.iterrows():
        ax.scatter(row["cstim_mean_low"], row["cstim_wrsa_mean"],
                   s=180, marker="*", color=SET_COLOR[row["model_set"]],
                   edgecolor="black", lw=0.6, zorder=5,
                   label=f"cstim {row['model_set']}")
        ax.annotate(row["model_set"],
                    (row["cstim_mean_low"], row["cstim_wrsa_mean"]),
                    xytext=(6, -10), textcoords="offset points",
                    fontsize=7, color=SET_COLOR[row["model_set"]])

    ax.set_xlabel("Mean Mahalanobis low-level distance from training", fontsize=10)
    ax.set_ylabel("mean mixed RSA (subject × model)", fontsize=10)
    ax.set_title(
        "Mixed RSA on vicco subsets spanning the low-level distance range,\n"
        "vs. cstim sets at their respective mean low-level distances",
        fontsize=10,
    )
    ax.legend(fontsize=7, loc="lower left", framealpha=0.85, ncol=2)
    ax.tick_params(labelsize=8)
    return fig


def _per_image_low_level():
    """Per-image low-level Mahalanobis distance per stimulus group."""
    p = OOD / "low_level_robustness_per_image_distances.csv"
    df = pd.read_csv(p)
    keep = ["vicco"] + CSTIM_SETS
    return df[df["stim_set"].isin(keep)][["stim_set", "image_idx", "mahal_distance"]]


def _per_stimulus_loglik():
    """Per-stimulus PPCA log-likelihood z (averaged across model and subject)."""
    df = pd.read_csv(OOD / "pca_loglik.csv")
    keep = ["vicco"] + CSTIM_SETS
    return (df[df["stimulus_group"].isin(keep)]
            .groupby(["stimulus_group", "stimulus_idx"])[
                ["loglik_pred_z", "loglik_feature_z"]
            ]
            .mean()
            .reset_index())


SHIFT_GROUP_ORDER = ["vicco"] + CSTIM_SETS
SHIFT_GROUP_LABEL = {"vicco": "Baseline", **GROUP_LABEL}


def _cohens_d(a, b):
    """Cohen's d, treating a as test (cstim) and b as reference (baseline)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s1 = a.std(ddof=1)
    s2 = b.std(ddof=1)
    pooled = np.sqrt(((len(a) - 1) * s1 ** 2 + (len(b) - 1) * s2 ** 2)
                     / (len(a) + len(b) - 2))
    if pooled == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled


def _sig_stars(p):
    if not np.isfinite(p):
        return ""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def _plot_shift_strip(ax, df, group_col, value_col, ylabel, title=None,
                      cstim_color=None, baseline_color=None, ylim=None,
                      show_xticklabels=True, mean_marker_color=None,
                      annotate_summary=True, d_invert=False,
                      d_text_loc="bottom_left", n_caption=None):
    """Strip plot of per-stimulus values by group with mean ± SD overlay.

    annotate_summary=True writes one text block summarising the cstim-vs-
    baseline shift across cstim sets: range of |Cohen's d| and the maximum
    Mann–Whitney U two-sided p-value across the 5 sets. The MWU is
    distribution-free, so it does not assume normality of the per-stimulus
    distributions.
    """
    if cstim_color is None:
        cstim_color = OKABE["vermillion"]
    if baseline_color is None:
        baseline_color = OKABE["black"]

    rng = np.random.default_rng(0)
    xs = np.arange(len(SHIFT_GROUP_ORDER))
    raw = {}
    means = {}
    sds = {}
    for i, group in enumerate(SHIFT_GROUP_ORDER):
        vals = df[df[group_col] == group][value_col].dropna().values
        if len(vals) == 0:
            continue
        raw[group] = vals
        color = baseline_color if group == "vicco" else cstim_color
        jitter = rng.normal(0, 0.07, len(vals))
        ax.scatter(i + jitter, vals, s=3.0, color=color,
                   alpha=0.18, edgecolor="none", zorder=1)
        m = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        means[group] = m
        sds[group] = sd
        edge_color = mean_marker_color if mean_marker_color is not None else "black"
        ax.errorbar([i], [m], yerr=[sd], fmt="o",
                    color=color, markersize=5.5,
                    markeredgecolor=edge_color, markeredgewidth=0.6,
                    elinewidth=1.0, capsize=2.5, capthick=0.9, zorder=4)

    ax.set_xticks(xs)
    if show_xticklabels:
        ax.set_xticklabels([SHIFT_GROUP_LABEL[g] for g in SHIFT_GROUP_ORDER],
                           rotation=35, ha="right",
                           fontsize=FONT["small"])
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(ylabel, fontsize=FONT["small"])
    if title is not None:
        ax.set_title(title, fontsize=FONT["title"], pad=4)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        # Robust auto-clip so outliers don't compress the means/SDs into
        # an unreadable strip. Use the 1st–99th percentile of all values
        # and add extra headroom for asterisks at the top.
        all_vals = np.concatenate([raw[g] for g in raw]) if raw else np.array([0.0, 1.0])
        q_lo, q_hi = np.percentile(all_vals, [1, 99])
        # Make sure all means ± SDs are inside the window.
        if means:
            mean_min = min(m - s for m, s in zip(means.values(), sds.values()))
            mean_max = max(m + s for m, s in zip(means.values(), sds.values()))
            q_lo = min(q_lo, mean_min)
            q_hi = max(q_hi, mean_max)
        y_range = max(q_hi - q_lo, 1e-6)
        ax.set_ylim(q_lo - 0.08 * y_range, q_hi + 0.30 * y_range)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=FONT["small"])
    ax.grid(axis="y", color="0.90", lw=0.45, zorder=0)

    if annotate_summary and "vicco" in raw:
        # Descriptive Cohen's d only; no inferential significance stars
        # (consistent with the paper's descriptive-only reporting stance).
        d_vals = []
        for group in SHIFT_GROUP_ORDER:
            if group == "vicco" or group not in raw:
                continue
            d = _cohens_d(raw[group], raw["vicco"])
            if not np.isfinite(d):
                continue
            if d_invert:
                d = -d
            d_vals.append(d)
        if d_vals:
            d_lo, d_hi = min(d_vals), max(d_vals)
            label = f"d ∈ [{d_lo:+.2f}, {d_hi:+.2f}]"
            if d_text_loc == "top_left":
                tx, ty, va = 0.04, 0.96, "top"
            elif d_text_loc == "top_right":
                tx, ty, va = 0.96, 0.96, "top"
                ha = "right"
            else:
                tx, ty, va = 0.04, 0.04, "bottom"
            ha = "right" if d_text_loc == "top_right" else "left"
            ax.text(tx, ty, label,
                    transform=ax.transAxes, ha=ha, va=va,
                    fontsize=FONT["small"] - 1, color=cstim_color,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="none", alpha=0.85))

    if n_caption:
        # Place "points = stimuli" note opposite to the d-text so they don't
        # overlap. Default puts d at bottom-left, so n-caption goes top-right.
        if d_text_loc == "top_left":
            nx, ny, nha, nva = 0.96, 0.04, "right", "bottom"
        else:
            nx, ny, nha, nva = 0.96, 0.96, "right", "top"
        ax.text(nx, ny, n_caption,
                transform=ax.transAxes, ha=nha, va=nva,
                fontsize=FONT["small"] - 1, color="0.30",
                bbox=dict(boxstyle="round,pad=0.16", fc="white",
                          ec="none", alpha=0.78))


def fig_dissociation():
    """Publication control figure for low-level, PPCA OOD, and reliability.

    The figure avoids the failed OOD-matching design. Instead it asks:
      (a) do matched/higher-low-level Vicco controls reproduce cstim mixed RSA?
      (b) does cstim-vicco PPCA OOD shift predict the alignment drop?
      (c) are cstim RDMs uniformly less reliable than Vicco RDMs?
      (d) does the cstim drop remain after correlation-ceiling normalization?
    """
    low_df, rel_df, norm_df = _control_absolute_values()
    ood_delta_df = _ood_alignment_delta_points()
    per_img_low_df = _per_image_low_level()
    per_stim_loglik_df = _per_stimulus_loglik()

    fig = plt.figure(figsize=(W_DOUBLE, 8.7))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.95, bottom=0.075,
                        wspace=0.30, hspace=0.95)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.95, 1.55, 0.95])

    a_gs = gs[0, :].subgridspec(1, 2, width_ratios=[0.26, 1.0], wspace=0.22)
    ax_a_shift = fig.add_subplot(a_gs[0, 0])
    ax_a = fig.add_subplot(a_gs[0, 1])
    ax_c = fig.add_subplot(gs[2, 0])
    ax_d = fig.add_subplot(gs[2, 1])

    _plot_shift_strip(
        ax_a_shift, per_img_low_df,
        group_col="stim_set", value_col="mahal_distance",
        ylabel="Low-level distance\n(Mahalanobis)",
        title="Low-level shift",
        d_text_loc="top_left",
        n_caption="points = stimuli",
    )
    _plot_low_level_panel(ax_a, low_df)
    _plot_ood_delta_panel(fig, gs[1, :], ood_delta_df, per_stim_loglik_df)
    pair_conditions = ["Baseline", "Cstim"]
    pair_colors = {"Baseline": OKABE["blue"], "Cstim": OKABE["vermillion"]}
    pair_markers = {"Baseline": "s", "Cstim": "o"}
    _plot_absolute_points(
        ax_c, rel_df, "rdm_reliability",
        "Reliability differs between cstim and baseline",
        "Noise ceiling\n(split-half r_SB)",
        pair_conditions,
        pair_colors,
        pair_markers,
        ylim=(0.0, 0.78),
        connect=True,
        legend_loc="lower right",
    )
    _plot_absolute_points(
        ax_d, norm_df, "norm_wrsa",
        "Drop survives noise-ceiling normalization",
        "mixed RSA / √NC",
        pair_conditions,
        pair_colors,
        pair_markers,
        ylim=(0.0, 0.95),
        connect=True,
        legend_loc="lower right",
    )

    # Anchor labels to gridspec row positions in figure coords so a narrow
    # subpanel (like ax_a_shift) doesn't shift the label.
    row_a_pos = gs[0, :].get_position(fig)
    row_c_pos = gs[2, 0].get_position(fig)
    row_d_pos = gs[2, 1].get_position(fig)
    _panel_label_at(fig, row_a_pos.x0 - 0.020, row_a_pos.y1 + 0.005, "a")
    _panel_label_at(fig, row_c_pos.x0 - 0.020, row_c_pos.y1 + 0.005, "c")
    _panel_label_at(fig, row_d_pos.x0 - 0.020, row_d_pos.y1 + 0.005, "d")

    return fig


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    SUPPLEMENTARY_DIR.mkdir(parents=True, exist_ok=True)
    SUPPLEMENTARY_PNG_DIR.mkdir(parents=True, exist_ok=True)
    if name == "low_level_dissociation":
        pdf_path = FIGS / f"{name}.pdf"
        png_path = PNG_DIR / f"{name}.png"
    else:
        pdf_path = SUPPLEMENTARY_DIR / f"{name}.pdf"
        png_path = SUPPLEMENTARY_PNG_DIR / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved → {pdf_path}")
    print(f"Saved → {png_path}")
    plt.close(fig)


def main():
    save(fig_curve(),          "low_level_deterministic_curve")
    save(fig_dissociation(),   "low_level_dissociation")


if __name__ == "__main__":
    main()
