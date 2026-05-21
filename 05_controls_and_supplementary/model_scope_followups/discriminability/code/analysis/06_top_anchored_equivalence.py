"""Höfling-style effective-equivalence count.

For each (model_set, stimulus_type, metric):
  1. Per-subject mean score per model (averaged over baseline-stimulus bootstraps).
  2. Bootstrap subject indices with replacement (K=10,000) → per-bootstrap mean
     score per model.
  3. Identify top-ranked model on the original (un-resampled) means.
  4. Compute its 2.5/97.5 percentile CI from the bootstrap distribution.
  5. Count how many models have an empirical mean inside that CI
     (= effectively equivalent to the top).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RSA_ROOT = ROOT.parent / "02_rsa_scores" / "data"
OUT = ROOT / "data" / "top_anchored_equivalence.csv"

SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
N_BOOT = 10_000
RNG = np.random.default_rng(0)


def load_scores(metric: str) -> pd.DataFrame:
    fname, col = {
        "crsa": ("crsa_scores.csv", "crsa"),
        "wrsa": ("wrsa_transfer_scores.csv", "wrsa_transfer"),
    }[metric]
    frames = []
    for subj in SUBJECTS:
        df = pd.read_csv(RSA_ROOT / subj / fname)
        df["score"] = df[col]
        df["metric"] = metric
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def per_subject_means(df: pd.DataFrame) -> pd.DataFrame:
    """Average over baseline bootstraps -> one score per (subject, model, set, type)."""
    return (
        df.groupby(["subject", "model_set", "stimulus_type", "model", "display_name"])
        ["score"].mean().reset_index()
    )


def equivalence_count(scores: pd.DataFrame) -> dict:
    """`scores` is a per-subject-x-model frame for one (model_set, stimulus_type)."""
    pivot = scores.pivot(index="subject", columns="model", values="score")
    pivot = pivot.loc[SUBJECTS]                # (5 subjects, M models)
    M = pivot.values                           # shape (5, n_models)
    models = list(pivot.columns)

    point_means = M.mean(axis=0)               # (n_models,)
    top_idx = int(np.argmax(point_means))
    top_model = models[top_idx]
    top_score = point_means[top_idx]

    boot_idx = RNG.integers(0, len(SUBJECTS), size=(N_BOOT, len(SUBJECTS)))
    boot_means = M[boot_idx].mean(axis=1)      # (N_BOOT, n_models)
    top_boot = boot_means[:, top_idx]
    lo, hi = np.percentile(top_boot, [2.5, 97.5])

    in_ci = (point_means >= lo) & (point_means <= hi)
    n_eq = int(in_ci.sum())                    # includes the top itself

    eq_models = [models[i] for i in np.where(in_ci)[0] if i != top_idx]

    return dict(
        top_model=top_model,
        top_score=float(top_score),
        ci_lo=float(lo),
        ci_hi=float(hi),
        ci_width=float(hi - lo),
        n_models=len(models),
        n_eq=n_eq,
        n_eq_excl_top=n_eq - 1,
        eq_models=";".join(eq_models),
    )


def main() -> None:
    rows = []
    for metric in ("crsa", "wrsa"):
        df = load_scores(metric)
        per_subj = per_subject_means(df)
        for model_set in sorted(per_subj["model_set"].unique()):
            for stim in ("controversial", "vicco"):
                sub = per_subj[
                    (per_subj["model_set"] == model_set)
                    & (per_subj["stimulus_type"] == stim)
                ]
                if sub.empty:
                    continue
                rec = equivalence_count(sub)
                rec.update(metric=metric, model_set=model_set, stimulus_type=stim)
                rows.append(rec)

    out = pd.DataFrame(rows).sort_values(["metric", "model_set", "stimulus_type"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}")

    # Pretty-print summary
    for metric, df_m in out.groupby("metric"):
        print(f"\n=== {metric.upper()} ===")
        wide = df_m.pivot_table(
            index="model_set",
            columns="stimulus_type",
            values=["n_eq_excl_top", "ci_width"],
        )
        print(wide.round(3))
        # delta
        delta = (
            df_m[df_m.stimulus_type == "controversial"].set_index("model_set")["n_eq_excl_top"]
            - df_m[df_m.stimulus_type == "vicco"].set_index("model_set")["n_eq_excl_top"]
        )
        print("delta n_eq_excl_top (cstim - baseline):")
        print(delta)


if __name__ == "__main__":
    main()
