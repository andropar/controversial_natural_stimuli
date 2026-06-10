#!/usr/bin/env python3
"""Matched-counterfactual ladder for the controversial-stimulus RSA effect.

This is the primary explanation-level endpoint. It reuses the canonical
subject-level mixed-RSA endpoint table and summarizes how much of the
controversial-minus-baseline delta remains under progressively stronger
counterfactual baselines.

The available held-out controls are not a perfectly nested ladder: low-level,
PPCA-OOD, embedding-PC, and combined matching were generated as separate
counterfactuals. The script therefore reports the exact baseline type and the
covariate family matched by that baseline rather than pretending the controls
are stricter than they are.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


STAGE = Path(__file__).resolve().parents[2]
SHARE_ROOT = STAGE.parents[1]
DATA = STAGE / "results"
FIGURES = STAGE / "figures"
SOURCE = SHARE_ROOT / "03_alignment_inference" / "results" / "primary_endpoint_summary.csv"

DATA.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

BASELINE_META = {
    "same_session_unselected": {
        "order": 0,
        "ladder_step": "M0",
        "baseline_label": "Same-session baseline",
        "matched_family": "none",
        "counterfactual_pool": "same_session_vicco",
    },
    "heldout_unique": {
        "order": 1,
        "ladder_step": "M0b",
        "baseline_label": "Held-out unique baseline",
        "matched_family": "none",
        "counterfactual_pool": "heldout_unique",
    },
    "heldout_unique_matched_low_level": {
        "order": 2,
        "ladder_step": "M1",
        "baseline_label": "Low-level matched",
        "matched_family": "low_level",
        "counterfactual_pool": "heldout_unique",
    },
    "heldout_unique_matched_ppca_ood": {
        "order": 3,
        "ladder_step": "M2",
        "baseline_label": "PPCA-OOD matched",
        "matched_family": "feature_ppca_ood",
        "counterfactual_pool": "heldout_unique",
    },
    "heldout_unique_matched_embedding_pc": {
        "order": 4,
        "ladder_step": "M3",
        "baseline_label": "Embedding-PC matched",
        "matched_family": "model_feature_pcs",
        "counterfactual_pool": "heldout_unique",
    },
    "heldout_unique_matched_combined": {
        "order": 5,
        "ladder_step": "M3+",
        "baseline_label": "Low-level + OOD + embedding matched",
        "matched_family": "low_level_embedding_pc_feature_ppca_ood",
        "counterfactual_pool": "heldout_unique",
    },
    "heldout_unique_high_ppca_ood": {
        "order": 6,
        "ladder_step": "stress",
        "baseline_label": "High-PPCA-OOD held-out baseline",
        "matched_family": "feature_ppca_ood_high",
        "counterfactual_pool": "heldout_unique",
    },
}

BASELINE_ORDER = list(BASELINE_META)


def _sem(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) < 2:
        return float("nan")
    return float(vals.std(ddof=1) / np.sqrt(len(vals)))


def _subject_boot_ci(df: pd.DataFrame, value_col: str, n_boot: int = 5000) -> tuple[float, float]:
    vals = pd.to_numeric(df[value_col], errors="coerce")
    sub = df.loc[vals.notna(), ["subject", value_col]].copy()
    if sub.empty:
        return float("nan"), float("nan")
    subject_means = (
        sub.groupby("subject", sort=True)[value_col]
        .mean()
        .dropna()
        .to_numpy(dtype=float)
    )
    if len(subject_means) == 0:
        return float("nan"), float("nan")
    if len(subject_means) == 1:
        v = float(subject_means[0])
        return v, v
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(subject_means), size=(n_boot, len(subject_means)))
    boots = subject_means[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def load_ladder() -> pd.DataFrame:
    df = pd.read_csv(SOURCE)
    df = df[
        (df["metric"] == "mixed_RSA")
        & (df["baseline_type"].isin(BASELINE_ORDER))
        & (df["primary_alignment_endpoint"].fillna(False).astype(bool))
    ].copy()
    if df.empty:
        raise RuntimeError(f"No mixed-RSA primary endpoint rows found in {SOURCE}")

    meta = pd.DataFrame.from_dict(BASELINE_META, orient="index").reset_index()
    meta = meta.rename(columns={"index": "baseline_type"})
    df = df.merge(meta, on="baseline_type", how="left")
    df = df.sort_values(["order", "model_set", "subject"]).reset_index(drop=True)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("pooled", "all_sets", df)]
    for model_set, sub in df.groupby("model_set", sort=False):
        scopes.append(("model_set", model_set, sub))

    for scope, model_set, sub in scopes:
        same = sub[sub["baseline_type"] == "same_session_unselected"]
        same_mean = float(same["delta"].mean()) if not same.empty else np.nan
        same_abs = abs(same_mean) if np.isfinite(same_mean) else np.nan
        for baseline_type in BASELINE_ORDER:
            grp = sub[sub["baseline_type"] == baseline_type]
            if grp.empty:
                continue
            meta = BASELINE_META[baseline_type]
            mean_delta = float(grp["delta"].mean())
            lo, hi = _subject_boot_ci(grp, "delta")
            mean_delta_nc = (
                float(grp["delta_NCnorm"].mean())
                if "delta_NCnorm" in grp and grp["delta_NCnorm"].notna().any()
                else np.nan
            )
            lo_nc, hi_nc = (
                _subject_boot_ci(grp, "delta_NCnorm")
                if "delta_NCnorm" in grp and grp["delta_NCnorm"].notna().any()
                else (np.nan, np.nan)
            )
            rows.append(
                {
                    "scope": scope,
                    "model_set": model_set,
                    "baseline_type": baseline_type,
                    **meta,
                    "n_rows": int(len(grp)),
                    "n_subjects": int(grp["subject"].nunique()),
                    "n_model_sets": int(grp["model_set"].nunique()),
                    "mean_score_cstim": float(grp["score_cstim"].mean()),
                    "mean_score_baseline": float(grp["score_baseline"].mean()),
                    "mean_delta": mean_delta,
                    "sem_delta": _sem(grp["delta"]),
                    "ci95_delta_lo_subject_boot": lo,
                    "ci95_delta_hi_subject_boot": hi,
                    "mean_delta_NCnorm": mean_delta_nc,
                    "ci95_delta_NCnorm_lo_subject_boot": lo_nc,
                    "ci95_delta_NCnorm_hi_subject_boot": hi_nc,
                    "effect_remaining_vs_same_session_abs": (
                        abs(mean_delta) / same_abs if same_abs and np.isfinite(same_abs) else np.nan
                    ),
                    "n_negative_delta": int((grp["delta"] < 0).sum()),
                    "n_positive_delta": int((grp["delta"] > 0).sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["scope", "model_set", "order"]).reset_index(drop=True)


def plot_pooled(summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    pooled = summary[summary["scope"] == "pooled"].sort_values("order")
    if pooled.empty:
        return

    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    y = np.arange(len(pooled))
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    x = pooled["mean_delta"].to_numpy(dtype=float)
    lo = pooled["ci95_delta_lo_subject_boot"].to_numpy(dtype=float)
    hi = pooled["ci95_delta_hi_subject_boot"].to_numpy(dtype=float)
    colors = ["#666666", "#0072B2", "#009E73", "#56B4E9", "#CC79A7", "#D55E00", "#E69F00"]
    for i, (_, row) in enumerate(pooled.iterrows()):
        ax.plot([lo[i], hi[i]], [i, i], color=colors[i], linewidth=2.0, solid_capstyle="round")
        ax.scatter(x[i], i, s=44, color=colors[i], edgecolor="black", linewidth=0.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(pooled["baseline_label"].tolist(), fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mixed-RSA delta (controversial - counterfactual baseline)")
    ax.set_title("Matched-counterfactual ladder", loc="left", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.43, right=0.98, bottom=0.18, top=0.90)
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES / f"matched_counterfactual_ladder.{ext}", dpi=300)
    plt.close(fig)


def write_decision_table(summary: pd.DataFrame) -> pd.DataFrame:
    pooled = summary[summary["scope"] == "pooled"].copy()
    keep = pooled[
        [
            "ladder_step",
            "baseline_type",
            "baseline_label",
            "matched_family",
            "mean_delta",
            "ci95_delta_lo_subject_boot",
            "ci95_delta_hi_subject_boot",
            "effect_remaining_vs_same_session_abs",
            "n_rows",
            "n_subjects",
        ]
    ].copy()
    keep["interpretation"] = np.where(
        keep["ci95_delta_hi_subject_boot"] < 0,
        "effect persists relative to this counterfactual",
        "effect not clearly below this counterfactual",
    )
    keep.to_csv(DATA / "matched_counterfactual_decision_table.csv", index=False)
    return keep


def main() -> None:
    df = load_ladder()
    df.to_csv(DATA / "matched_counterfactual_ladder_by_cell.csv", index=False)
    summary = summarize(df)
    summary.to_csv(DATA / "matched_counterfactual_ladder_summary.csv", index=False)
    decision = write_decision_table(summary)
    plot_pooled(summary)

    print(f"wrote {DATA / 'matched_counterfactual_ladder_by_cell.csv'}")
    print(f"wrote {DATA / 'matched_counterfactual_ladder_summary.csv'}")
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
