#!/usr/bin/env python3
"""Quantify brain placement among model pair distances for All-Models pairs."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore

_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER / "figures"))
sys.path.insert(0, str(_PAPER.parents[1]))

import config  # noqa: E402
from style_improved import OKABE_ITO, apply_style  # noqa: E402
from utils import compute_rdm_correlation  # noqa: E402


OUT = _PAPER / "15_pair_level_brain_placement" / "results"
FIG = _PAPER / "15_pair_level_brain_placement" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
TITLE_FS = 12
AXIS_FS = 11
TICK_FS = 10
LEGEND_FS = 9
ANNOTATION_FS = 9


def bootstrap_stat(values: np.ndarray, func, *, n_boot: int = 10000, null: float = 0.5) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_vs_null": np.nan}
    rng = np.random.default_rng(20260503)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = np.asarray([func(values[i]) for i in idx], dtype=float)
    estimate = float(func(values))
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
    p_lower = float(np.mean(boot <= null))
    p_upper = float(np.mean(boot >= null))
    return {
        "estimate": estimate,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_vs_null": min(1.0, 2.0 * min(p_lower, p_upper)),
    }


def rdm_vec_to_zmat(vec: np.ndarray, n: int) -> np.ndarray:
    vals = zscore(vec, nan_policy="omit")
    mat = np.zeros((n, n), dtype=float)
    tri = np.triu_indices(n, k=1)
    mat[tri] = vals
    mat = mat + mat.T
    return mat


def load_model_zmats() -> dict[str, np.ndarray]:
    with open(config.SELECTION_PAYLOAD, "rb") as f:
        payload = pickle.load(f)
    features = payload.get("selected_features_raw", payload.get("best_raw_combined_features_raw"))
    out = {}
    for model, feats in features.items():
        if hasattr(feats, "detach"):
            feats = feats.detach().cpu().numpy()
        elif hasattr(feats, "numpy"):
            feats = feats.numpy()
        feats = np.asarray(feats)
        vec = pdist(feats, metric="cosine")
        out[str(model)] = rdm_vec_to_zmat(vec, feats.shape[0])
    return out


def load_brain_zmats() -> dict[str, np.ndarray]:
    out = {}
    for subject in config.SUBJECTS:
        data_dir = config.get_subject_data_dir(subject)
        betas = np.load(data_dir / "cstim_betas_averaged.npz", allow_pickle=True)
        voxel = np.load(data_dir / "voxel_metadata.npz", allow_pickle=True)
        stim_info = pd.read_csv(data_dir / "cstim_stimulus_info.csv")
        hlvis = voxel["hlvis_mask"]
        beta_h = betas["betas"][hlvis, :]
        key_to_idx = {k: i for i, k in enumerate(betas["stim_keys"])}
        gdf = stim_info[stim_info["group"] == "all_models"].copy()
        brain_idx = np.array([key_to_idx[k] for k in gdf["stim_key"].values])
        file_idx = gdf["stim_idx"].values.astype(int)
        rdm = compute_rdm_correlation(beta_h[:, brain_idx].T)
        tri = np.triu_indices(rdm.shape[0], k=1)
        zmat_order = rdm_vec_to_zmat(rdm[tri], rdm.shape[0])
        zmat = np.full((100, 100), np.nan)
        for local_i, sel_i in enumerate(file_idx):
            for local_j, sel_j in enumerate(file_idx):
                zmat[int(sel_i), int(sel_j)] = zmat_order[local_i, local_j]
        out[subject] = zmat
    return out


def main() -> None:
    model_z = load_model_zmats()
    brain_z = load_brain_zmats()
    models = sorted(model_z)
    rows = []
    for i in range(100):
        for j in range(i + 1, 100):
            model_vals = np.array([model_z[m][i, j] for m in models], dtype=float)
            model_mean = float(np.nanmean(model_vals))
            model_sd = float(np.nanstd(model_vals, ddof=1))
            model_spread = float(np.nanmax(model_vals) - np.nanmin(model_vals))
            for subject, bmat in brain_z.items():
                bz = float(bmat[i, j])
                percentile = float(np.mean(model_vals <= bz))
                z_position = float((bz - model_mean) / model_sd) if model_sd > 0 else np.nan
                rows.append(
                    {
                        "img_i": i,
                        "img_j": j,
                        "subject": subject,
                        "brain_z": bz,
                        "brain_percentile_among_models": percentile,
                        "brain_z_position_vs_model_distribution": z_position,
                        "model_mean_z": model_mean,
                        "model_sd_z": model_sd,
                        "model_spread_z": model_spread,
                        "model_min_z": float(np.nanmin(model_vals)),
                        "model_max_z": float(np.nanmax(model_vals)),
                        "n_models": len(models),
                    }
                )
    df = pd.DataFrame(rows)
    pair_summary = (
        df.groupby(["img_i", "img_j"], as_index=False)
        .agg(
            mean_brain_z=("brain_z", "mean"),
            sd_brain_z=("brain_z", "std"),
            mean_brain_percentile=("brain_percentile_among_models", "mean"),
            mean_brain_z_position=("brain_z_position_vs_model_distribution", "mean"),
            model_spread_z=("model_spread_z", "first"),
            n_subjects=("subject", "nunique"),
        )
    )
    spread_threshold = pair_summary["model_spread_z"].quantile(0.75)
    pair_summary["eligible_high_disagreement"] = (
        (pair_summary["model_spread_z"] >= spread_threshold)
        & (pair_summary["sd_brain_z"] < 1.0)
        & (pair_summary["mean_brain_z"].abs() > 0.75)
    )
    df = df.merge(pair_summary[["img_i", "img_j", "eligible_high_disagreement"]], on=["img_i", "img_j"])
    df.to_csv(OUT / "pair_level_brain_placement.csv", index=False)
    pair_summary.to_csv(OUT / "pair_level_brain_placement_summary.csv", index=False)

    eligible = pair_summary[pair_summary["eligible_high_disagreement"]]
    pct_values = eligible["mean_brain_percentile"].to_numpy(dtype=float)
    mean_boot = bootstrap_stat(pct_values, np.mean)
    median_boot = bootstrap_stat(pct_values, np.median)
    summary = pd.DataFrame(
        [
            {
                "n_pairs_total": len(pair_summary),
                "n_pairs_eligible": len(eligible),
                "spread_threshold": spread_threshold,
                "eligible_mean_brain_percentile": mean_boot["estimate"],
                "eligible_mean_brain_percentile_ci_low": mean_boot["ci_low"],
                "eligible_mean_brain_percentile_ci_high": mean_boot["ci_high"],
                "eligible_mean_brain_percentile_boot_p_vs_0_5": mean_boot["p_vs_null"],
                "eligible_median_brain_percentile": median_boot["estimate"],
                "eligible_median_brain_percentile_ci_low": median_boot["ci_low"],
                "eligible_median_brain_percentile_ci_high": median_boot["ci_high"],
                "eligible_median_brain_percentile_boot_p_vs_0_5": median_boot["p_vs_null"],
                "eligible_mean_z_position": eligible["mean_brain_z_position"].mean(),
            }
        ]
    )
    summary.to_csv(OUT / "pair_level_brain_placement_aggregate.csv", index=False)

    import matplotlib.pyplot as plt

    apply_style()
    plot_df = eligible.copy()
    mean_pct = float(plot_df["mean_brain_percentile"].mean())
    median_pct = float(plot_df["mean_brain_percentile"].median())
    mean_ci = (
        float(summary["eligible_mean_brain_percentile_ci_low"].iloc[0]),
        float(summary["eligible_mean_brain_percentile_ci_high"].iloc[0]),
    )
    median_ci = (
        float(summary["eligible_median_brain_percentile_ci_low"].iloc[0]),
        float(summary["eligible_median_brain_percentile_ci_high"].iloc[0]),
    )
    p_mean = float(summary["eligible_mean_brain_percentile_boot_p_vs_0_5"].iloc[0])

    fig, ax = plt.subplots(figsize=(6.7, 3.65))
    bins = np.linspace(0, 1, 21)
    ax.hist(
        plot_df["mean_brain_percentile"],
        bins=bins,
        color=OKABE_ITO["blue"],
        edgecolor="white",
        linewidth=0.6,
        alpha=0.86,
    )
    ax.axvline(0.5, color="0.20", ls="--", lw=0.9, label="model median")
    ax.axvline(mean_pct, color=OKABE_ITO["vermillion"], lw=1.2, label="mean")
    ax.axvline(median_pct, color=OKABE_ITO["orange"], lw=1.2, ls="-.", label="median")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Brain percentile among model pair distances", fontsize=AXIS_FS)
    ax.set_ylabel("Eligible image pairs", fontsize=AXIS_FS)
    ax.set_title("Brain placement", fontsize=TITLE_FS, loc="left", pad=4)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.45, alpha=0.75)
    ax.text(
        0.98,
        0.92,
        "n = {n} pairs\n"
        "mean = {mean:.2f} [{mlo:.2f}, {mhi:.2f}]\n"
        "median = {med:.2f} [{dlo:.2f}, {dhi:.2f}]\n"
        "p(mean vs .5) = {p:.3f}".format(
            n=len(plot_df),
            mean=mean_pct,
            mlo=mean_ci[0],
            mhi=mean_ci[1],
            med=median_pct,
            dlo=median_ci[0],
            dhi=median_ci[1],
            p=p_mean,
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=ANNOTATION_FS,
        color="0.20",
    )
    ax.legend(frameon=False, loc="upper left", fontsize=LEGEND_FS, handlelength=1.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "brain_percentile_distribution.pdf")
    fig.savefig(FIG / "brain_percentile_distribution.png", dpi=200)
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
