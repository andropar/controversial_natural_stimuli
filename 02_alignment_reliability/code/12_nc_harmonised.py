"""
12_nc_harmonised.py

Side-by-side comparison of within-subject (split-half) and between-subject
noise ceilings for each model set and stimulus type, to demonstrate that the
brain-alignment results are insensitive to the choice of NC and that the
metric--ceiling pairing in earlier drafts was presentational.

Reads:
  02_alignment_reliability/results/rdm_noise_ceilings.csv
  02_alignment_reliability/results/between_subject_noise_ceilings.csv

Writes:
  02_alignment_reliability/figures/supplementary/nc_harmonised.{pdf,png}
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

STAGE = Path(__file__).resolve().parents[1]
SHARE_ROOT = STAGE.parent
sys.path.insert(0, str(SHARE_ROOT / "shared" / "code" / "paper_helpers"))
sys.path.insert(0, str(SHARE_ROOT / "shared" / "code" / "paper_helpers" / "figures"))
DATA = STAGE / "results"
OUT = STAGE / "figures" / "supplementary"
PNG_OUT = OUT / "png"
OUT.mkdir(parents=True, exist_ok=True)
PNG_OUT.mkdir(parents=True, exist_ok=True)

from style_improved import OKABE_ITO, apply_style, shade  # noqa: E402

MODEL_SETS = ["all_models", "sota", "training_objective", "architecture", "dataset"]
SET_LABEL = {
    "all_models": "All models",
    "sota": "SOTA",
    "training_objective": "Training Obj.",
    "architecture": "Architecture",
    "dataset": "Dataset",
}
COLOR_CSTIM = OKABE_ITO["vermillion"]
COLOR_BASE = OKABE_ITO["blue"]


def main():
    apply_style()
    ws = pd.read_csv(DATA / "rdm_noise_ceilings.csv")
    bs = pd.read_csv(DATA / "between_subject_noise_ceilings.csv")

    # Within-subject: take sqrt(noise_ceiling_spearman), then collapse
    # bootstraps within subject before plotting subject-level points.
    ws["nc_within"] = np.sqrt(ws["noise_ceiling_spearman"].clip(lower=0))
    ws_subject = (
        ws.groupby(["subject", "group", "stimulus_type"], as_index=False)
        .agg(nc_within=("nc_within", "mean"))
    )
    bs_subject = bs.rename(columns={"nc_mid": "nc_between"})[
        ["subject", "group", "stimulus_type", "nc_between"]
    ].copy()
    ws_summary = (ws_subject.groupby(["group", "stimulus_type"])
                    .agg(nc_within_mean=("nc_within", "mean"),
                         nc_within_sem=("nc_within", "sem"))
                    .reset_index())
    bs_summary = (bs_subject.groupby(["group", "stimulus_type"])
                    .agg(nc_between_mean=("nc_between", "mean"),
                         nc_between_sem=("nc_between", "sem"))
                    .reset_index())
    summary = ws_summary.merge(bs_summary, on=["group", "stimulus_type"], how="outer")

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 3.6))

    width = 0.17
    x = np.arange(len(MODEL_SETS))
    offsets = {("controversial", "within"): -1.5,
               ("controversial", "between"): -0.5,
               ("vicco", "within"): 0.5,
               ("vicco", "between"): 1.5}
    edge = {
        ("controversial", "within"): COLOR_CSTIM,
        ("controversial", "between"): COLOR_CSTIM,
        ("vicco", "within"): COLOR_BASE,
        ("vicco", "between"): COLOR_BASE,
    }
    marker = {
        ("controversial", "within"): "o",
        ("controversial", "between"): "s",
        ("vicco", "within"): "o",
        ("vicco", "between"): "s",
    }

    def subject_values(stim: str, nc_type: str, model_set: str) -> np.ndarray:
        source = ws_subject if nc_type == "within" else bs_subject
        value_col = "nc_within" if nc_type == "within" else "nc_between"
        if stim == "vicco":
            sub = source[(source["group"] == "vicco") & (source["stimulus_type"] == "vicco")]
        else:
            sub = source[(source["group"] == model_set) & (source["stimulus_type"] == stim)]
        return sub[value_col].to_numpy(dtype=float)

    rng = np.random.default_rng(0)
    for (stim, nc_type), off in offsets.items():
        color = edge[(stim, nc_type)]
        is_within = nc_type == "within"
        for xi, ms in enumerate(MODEL_SETS):
            vals = subject_values(stim, nc_type, ms)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            xpos = xi + off * width
            mean = float(vals.mean())
            sem = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan
            if np.isfinite(sem):
                ax.plot(
                    [xpos, xpos],
                    [mean - 1.96 * sem, mean + 1.96 * sem],
                    color=color,
                    linewidth=1.0,
                    solid_capstyle="round",
                    zorder=3,
                )
            jitter = rng.uniform(-0.025, 0.025, size=len(vals))
            ax.scatter(
                np.full(len(vals), xpos) + jitter,
                vals,
                s=9,
                marker=marker[(stim, nc_type)],
                facecolor=shade(color, 0.65) if is_within else "white",
                edgecolor=shade(color, -0.20),
                linewidth=0.35,
                alpha=0.75,
                zorder=4,
            )
            ax.scatter(
                xpos,
                mean,
                s=30,
                marker=marker[(stim, nc_type)],
                facecolor=color if is_within else "white",
                edgecolor=color,
                linewidth=1.0,
                zorder=5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([SET_LABEL[m] for m in MODEL_SETS], fontsize=9)
    ax.set_ylabel("Noise ceiling", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_ylim(0, max(0.85, summary[["nc_within_mean", "nc_between_mean"]].max().max() * 1.15))
    ax.set_title("Within- vs. between-subject RDM noise ceilings", fontsize=10)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.45, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_CSTIM,
               markeredgecolor=COLOR_CSTIM, markersize=5, label="Controversial, within-subj."),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white",
               markeredgecolor=COLOR_CSTIM, markersize=5, label="Controversial, between-subj."),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_BASE,
               markeredgecolor=COLOR_BASE, markersize=5, label="Baseline, within-subj."),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white",
               markeredgecolor=COLOR_BASE, markersize=5, label="Baseline, between-subj."),
    ]
    ax.legend(
        handles=legend_elems,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        fontsize=7.5,
        frameon=False,
        ncol=4,
        handletextpad=0.4,
        columnspacing=1.1,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "nc_harmonised.pdf", bbox_inches="tight")
    fig.savefig(PNG_OUT / "nc_harmonised.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT / 'nc_harmonised.pdf'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
