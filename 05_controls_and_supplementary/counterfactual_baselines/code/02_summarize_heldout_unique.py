#!/usr/bin/env python3
"""Build paper-facing summaries for held-out unique-image controls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER / "figures"))
sys.path.insert(0, str(_PAPER.parents[1]))

import config  # noqa: E402
from style_improved import OKABE_ITO, apply_style, shade  # noqa: E402


DATA = _PAPER / "05_heldout_unique_baseline" / "data"
FIGURES = _PAPER / "05_heldout_unique_baseline" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

MODEL_SETS = ["all_models", "sota", "training_objective", "architecture", "dataset"]
BASELINE_ORDER = [
    "same_session_unselected",
    "heldout_unique",
    "heldout_unique_matched_low_level",
    "heldout_unique_matched_embedding_pc",
    "heldout_unique_matched_ppca_ood",
    "heldout_unique_matched_combined",
    "heldout_unique_high_ppca_ood",
]
BASELINE_LABEL = {
    "same_session_unselected": "Same-session baseline",
    "heldout_unique": "Held-out unique",
    "heldout_unique_matched_low_level": "Held-out, low-level matched",
    "heldout_unique_matched_embedding_pc": "Held-out, embedding-PC matched",
    "heldout_unique_matched_ppca_ood": "Held-out, PPCA-OOD matched",
    "heldout_unique_matched_combined": "Held-out, combined matched",
    "heldout_unique_high_ppca_ood": "Held-out, high PPCA-OOD",
}
TITLE_FS = 12
AXIS_FS = 11
TICK_FS = 10


def sem(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) < 2:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def load_endpoint_summary() -> pd.DataFrame:
    path = DATA / "heldout_unique_endpoint_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"{path} is empty")
    return df


def completeness(df: pd.DataFrame) -> dict:
    return {
        "n_subjects": int(df["subject"].nunique()),
        "n_model_sets": int(df["model_set"].nunique()),
        "n_baseline_types": int(df["baseline_type"].nunique()),
        "min_splits": int(df["n_splits"].min()) if "n_splits" in df else 0,
        "complete_subjects": set(config.SUBJECTS).issubset(set(df["subject"])),
        "complete_model_sets": set(MODEL_SETS).issubset(set(df["model_set"])),
        "complete_baselines": set(BASELINE_ORDER).issubset(set(df["baseline_type"])),
        "complete_splits": bool("n_splits" in df and df["n_splits"].min() >= 10),
    }


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = [("pooled", "all_sets", df)]
    scopes.extend((("model_set", ms, df[df["model_set"] == ms]) for ms in MODEL_SETS))
    for scope, model_set, sub in scopes:
        if sub.empty:
            continue
        for baseline_type in BASELINE_ORDER:
            grp = sub[sub["baseline_type"] == baseline_type]
            if grp.empty:
                continue
            delta = pd.to_numeric(grp["delta"], errors="coerce").dropna()
            se = sem(delta)
            rows.append(
                {
                    "scope": scope,
                    "model_set": model_set,
                    "baseline_type": baseline_type,
                    "baseline_label": BASELINE_LABEL.get(baseline_type, baseline_type),
                    "n_rows": len(grp),
                    "n_subjects": int(grp["subject"].nunique()),
                    "n_model_sets": int(grp["model_set"].nunique()),
                    "n_splits_min": int(grp["n_splits"].min()) if "n_splits" in grp else np.nan,
                    "mean_delta": float(delta.mean()),
                    "sem_delta": se,
                    "ci95_low": float(delta.mean() - 1.96 * se) if np.isfinite(se) else np.nan,
                    "ci95_high": float(delta.mean() + 1.96 * se) if np.isfinite(se) else np.nan,
                    "mean_score_cstim": float(grp["score_cstim"].mean()),
                    "mean_score_baseline": float(grp["score_baseline"].mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_summary(df: pd.DataFrame, agg: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    pooled = agg[agg["scope"] == "pooled"].set_index("baseline_type").reindex(BASELINE_ORDER).dropna(
        subset=["mean_delta"], how="all"
    )
    if pooled.empty:
        return

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    y = np.arange(len(pooled))
    x = pooled["mean_delta"].to_numpy()
    ci_low = pooled["ci95_low"].to_numpy()
    ci_high = pooled["ci95_high"].to_numpy()
    colors = []
    for baseline_type in pooled.index:
        if baseline_type == "same_session_unselected":
            colors.append("#5F5F5F")
        elif baseline_type == "heldout_unique_high_ppca_ood":
            colors.append(OKABE_ITO["vermillion"])
        else:
            colors.append(OKABE_ITO["blue"])

    ax.axvline(0, color="black", linewidth=0.9, zorder=1)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.45, alpha=0.9)

    point_df = df[df["baseline_type"].isin(pooled.index)].copy()
    rng = np.random.default_rng(0)
    for i, baseline_type in enumerate(pooled.index):
        vals = point_df[point_df["baseline_type"] == baseline_type]["delta"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(
            vals,
            np.full(len(vals), i) + jitter,
            s=16,
            facecolor="white",
            edgecolor=shade(colors[i], -0.20),
            linewidth=0.45,
            alpha=0.75,
            zorder=3,
        )
        ax.plot([ci_low[i], ci_high[i]], [i, i], color=colors[i], linewidth=2.0, solid_capstyle="round", zorder=4)
        ax.scatter(x[i], i, s=42, color=colors[i], edgecolor="black", linewidth=0.55, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels([BASELINE_LABEL.get(b, b) for b in pooled.index], fontsize=TICK_FS)
    ax.invert_yaxis()
    ax.set_xlabel("Mixed-RSA delta (controversial - baseline)", fontsize=AXIS_FS)
    ax.set_title("Baseline controls", fontsize=TITLE_FS, loc="left", pad=4)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    x_min = float(np.nanmin(ci_low))
    x_max = max(0.005, float(np.nanmax(ci_high)))
    pad = 0.02
    ax.set_xlim(x_min - pad, x_max + pad)
    fig.subplots_adjust(left=0.42, right=0.98, bottom=0.18, top=0.90)
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES / f"heldout_unique_baseline_summary.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    df = load_endpoint_summary()
    status = completeness(df)
    status_df = pd.DataFrame([status])
    status_df.to_csv(DATA / "heldout_unique_completion_status.csv", index=False)
    if args.require_complete and not all(
        status[k] for k in ["complete_subjects", "complete_model_sets", "complete_baselines", "complete_splits"]
    ):
        raise RuntimeError(f"held-out unique analysis is incomplete: {status}")

    agg = aggregate(df)
    agg.to_csv(DATA / "heldout_unique_aggregate_summary.csv", index=False)
    plot_summary(df, agg)
    print(f"wrote held-out unique summaries in {DATA}")
    print(status)


if __name__ == "__main__":
    main()
