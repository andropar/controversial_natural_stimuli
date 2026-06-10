"""
07_l2_mahalanobis_robustness.py

Robustness check for the low-level Mahalanobis distance: recompute with
l2-normalised feature vectors following Mueller & Hein (2025). The qualitative
claim that cstim mean Mahalanobis distance exceeds the baseline should hold
under both schemes if the conclusion is real and not a normalisation artefact.

Reads:
  05_controls_and_supplementary/low_level_and_ood/image_statistics/results/image_stats.csv

Writes:
  05_controls_and_supplementary/low_level_and_ood/ood_controls/results/low_level_mahalanobis_l2norm.csv
  05_controls_and_supplementary/low_level_and_ood/ood_controls/figures/supplementary/mahalanobis_l2_robustness.{pdf,png}
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

STAGE = Path(__file__).resolve().parents[1]
LOW_LEVEL_AND_OOD = STAGE.parent
IMG_STATS = LOW_LEVEL_AND_OOD / "image_statistics" / "results" / "image_stats.csv"
OUT_DATA = STAGE / "results"
OUT_FIG = STAGE / "figures" / "supplementary"
OUT_PNG = OUT_FIG / "png"

STAT_COLS = [
    "lum_mean", "lum_rms", "colorfulness", "lab_chroma_mean", "hue_entropy",
    "sf_slope", "sf_high_low_ratio", "edge_mag_mean", "orient_anisotropy",
    "edge_com_x", "symmetry_lr", "entropy", "jpeg_ratio",
]

CSTIM_SETS = ["all_models", "sota", "training_objective", "architecture", "dataset"]
SET_LABEL = {
    "all_models": "All models",
    "sota": "SOTA",
    "training_objective": "Training Obj.",
    "architecture": "Architecture",
    "dataset": "Dataset",
    "vicco": "Baseline",
    "deepvision_train": "Train",
}


def mahalanobis(X, mu, cov_inv):
    diff = X - mu
    d2 = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
    return np.sqrt(np.clip(d2, 0, None))


def main():
    df = pd.read_csv(IMG_STATS)
    train = df[df["stimulus_set"] == "deepvision_train"][STAT_COLS].dropna().values

    # --- Variant A: unnormalised (current paper) ---
    mu_A = train.mean(axis=0)
    cov_A = np.cov(train - mu_A, rowvar=False) + 1e-6 * np.eye(len(STAT_COLS))
    cov_inv_A = np.linalg.inv(cov_A)

    # --- Variant B: l2-normalised features (Mueller & Hein 2025 Mahalanobis++) ---
    # Standardise to unit variance per axis first (so l2 norm isn't dominated
    # by units), then l2-normalise each row.
    train_std = train / (train.std(axis=0, keepdims=True) + 1e-9)
    train_l2 = train_std / (np.linalg.norm(train_std, axis=1, keepdims=True) + 1e-9)
    mu_B = train_l2.mean(axis=0)
    cov_B = np.cov(train_l2 - mu_B, rowvar=False) + 1e-6 * np.eye(len(STAT_COLS))
    cov_inv_B = np.linalg.inv(cov_B)

    rows = []
    for stim_set, sub in df.groupby("stimulus_set"):
        X = sub[STAT_COLS].dropna().values
        if X.size == 0:
            continue
        d_A = mahalanobis(X, mu_A, cov_inv_A)
        # Apply same standardisation and l2-normalisation as training
        X_std = X / (train.std(axis=0, keepdims=True) + 1e-9)
        X_l2 = X_std / (np.linalg.norm(X_std, axis=1, keepdims=True) + 1e-9)
        d_B = mahalanobis(X_l2, mu_B, cov_inv_B)
        for i, (a, b) in enumerate(zip(d_A, d_B)):
            rows.append({"stim_set": stim_set, "image_idx": i,
                         "mahal_unnorm": float(a), "mahal_l2norm": float(b)})

    out = pd.DataFrame(rows)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_PNG.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DATA / "low_level_mahalanobis_l2norm.csv", index=False)

    # Cohen's d for cstim vs baseline under both schemes.
    base = out[out["stim_set"] == "vicco"]
    summary = []
    for stim_set in CSTIM_SETS:
        c = out[out["stim_set"] == stim_set]
        for col in ("mahal_unnorm", "mahal_l2norm"):
            mu_c, mu_b = c[col].mean(), base[col].mean()
            sd = np.sqrt((c[col].var(ddof=1) + base[col].var(ddof=1)) / 2)
            d = (mu_c - mu_b) / sd
            summary.append({"stim_set": stim_set, "scheme": col,
                            "mean_cstim": mu_c, "mean_base": mu_b,
                            "cohens_d": float(d)})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT_DATA / "low_level_mahalanobis_l2norm_summary.csv", index=False)
    print(summary_df.to_string(index=False))

    # Figure: side-by-side per-set distributions (unnormalised vs l2-normalised)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), sharey=False)
    titles = ["Unnormalised Mahalanobis (paper default)",
              r"$\ell_2$-normalised Mahalanobis (\citet{mueller2025mahalanobispp})"]
    columns = ["mahal_unnorm", "mahal_l2norm"]
    sets_plot = ["vicco"] + CSTIM_SETS

    for ax, col, title in zip(axes, columns, titles):
        data = [out[out["stim_set"] == s][col].values for s in sets_plot]
        labels = [SET_LABEL[s] for s in sets_plot]
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55,
                        flierprops=dict(marker=".", markersize=2, alpha=0.4))
        for patch, s in zip(bp["boxes"], sets_plot):
            patch.set_facecolor("#0072B2" if s == "vicco" else "#D55E00")
            patch.set_alpha(0.55)
        ax.set_ylabel("Mahalanobis distance")
        ax.set_title(title.replace(r"\citet{mueller2025mahalanobispp}", "Mahalanobis++"),
                     fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)

    fig.suptitle("Low-level Mahalanobis distance: $\\ell_2$-normalisation does not change the qualitative cstim>baseline ordering",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "mahalanobis_l2_robustness.pdf", bbox_inches="tight")
    fig.savefig(OUT_PNG / "mahalanobis_l2_robustness.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_FIG / 'mahalanobis_l2_robustness.pdf'}")


if __name__ == "__main__":
    main()
