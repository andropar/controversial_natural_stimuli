#!/usr/bin/env python3
"""
Cross-set transfer of the model-RDM ensemble.

The residual analysis shows that a within-set cross-validated ensemble remains
below the brain ceiling, especially for diagnostic images. This script asks a
different question: do ensemble weights learned on one stimulus regime transfer
to another?

For each subject and RSA type:
  - compute brain and model RDM vectors for baseline (vicco) and all_models
    diagnostic images;
  - fit ridge weights on one set's image-pair samples;
  - evaluate the learned weights on the other set's image pairs;
  - compare the transfer score to a within-set CV ensemble and to the test-set
    correlation ceiling.

The feature scaling is deliberately set-level: each model RDM column and brain
RDM target is rank-transformed and z-scored within the train/test set before
fitting or scoring. That makes this a test of whether model-combination weights
generalize across stimulus regimes, not a test of absolute distance calibration.

Outputs:
    results/ensemble_transfer.csv
    figures/ensemble_transfer.{pdf,png}
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV

_HERE = Path(__file__).resolve()
STAGE = _HERE.parents[2]
SHARE_ROOT = STAGE.parents[2]
sys.path.insert(0, str(SHARE_ROOT / "src"))

import matplotlib.pyplot as plt

from cstims.paper import config
from cstims.paper.style_improved import apply_style, DPI, FONT, W_DOUBLE

apply_style()

RR_PATH = _HERE.parent / "01_compute_residual_rsa.py"
spec = importlib.util.spec_from_file_location("residual_rsa_compute", RR_PATH)
rr = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(rr)

OUT_DATA = STAGE / "results"
OUT_FIG = STAGE / "figures"

ALPHAS = np.logspace(-2, 6, 30)
GROUPS = ["vicco", "all_models"]
GROUP_LABELS = {"vicco": "Baseline", "all_models": "All-model diagnostic"}
COLOR_BASELINE = "#2980B9"
COLOR_CSTIM = "#D64541"
COLOR_TRANSFER = "#444444"


def _rank_z(vec: np.ndarray) -> np.ndarray:
    ranks = stats.rankdata(vec).astype(np.float64)
    std = ranks.std()
    if std == 0:
        return ranks - ranks.mean()
    return (ranks - ranks.mean()) / std


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    r, _ = stats.spearmanr(a[mask], b[mask])
    return float(r)


def _sem(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def _vectorize_group(subject: str, sdata: dict, group: str, rsa_type: str) -> dict:
    keys = sdata["group_keys"][group]
    stim_idx = sdata["group_stim_idx"][group]
    brain_rdms = rr._brain_rdm_matrices(keys, sdata["betas_by_rep"])
    model_rdms = rr._model_rdm_matrices(group, stim_idx, subject, rsa_type, None)

    brain_full = rr._triu_vec(brain_rdms["full"])
    brain_even = rr._triu_vec(brain_rdms["even"])
    brain_odd = rr._triu_vec(brain_rdms["odd"])
    model_vecs = [rr._triu_vec(model_rdms[m]) for m in rr.ALL_MODELS]

    r_halves = _spearman(brain_even, brain_odd)
    reliability_ceiling = rr._sb_correct(r_halves)
    correlation_ceiling = np.sqrt(reliability_ceiling) if reliability_ceiling > 0 else np.nan

    return {
        "subject": subject,
        "group": group,
        "rsa_type": rsa_type,
        "n_stimuli": len(keys),
        "brain": brain_full,
        "model_vecs": model_vecs,
        "reliability_ceiling": reliability_ceiling,
        "correlation_ceiling": correlation_ceiling,
    }


def _design(cell: dict) -> tuple[np.ndarray, np.ndarray]:
    y = _rank_z(cell["brain"])
    X = np.column_stack([_rank_z(v) for v in cell["model_vecs"]])
    return X, y


def fit_transfer(train: dict, test: dict) -> dict:
    X_train, y_train = _design(train)
    X_test, y_test = _design(test)

    ridge = RidgeCV(alphas=ALPHAS, scoring="neg_mean_squared_error", fit_intercept=False)
    ridge.fit(X_train, y_train)

    pred_train = ridge.predict(X_train)
    pred_test = ridge.predict(X_test)
    return {
        "r_train_in_sample": _spearman(y_train, pred_train),
        "r_test_transfer": _spearman(y_test, pred_test),
        "alpha": float(ridge.alpha_),
        "coef_l2": float(np.linalg.norm(ridge.coef_)),
    }


def within_cv(
    cell: dict,
    cv_repeats: int = rr.N_CV_REPEATS,
    cv_seed: int = rr.CV_RANDOM_STATE,
) -> dict:
    pred, counts = rr._ridge_oof_ranked(
        cell["brain"], cell["model_vecs"], n_stim=cell["n_stimuli"],
        alphas=ALPHAS, n_folds=rr.N_CV_FOLDS,
        n_repeats=cv_repeats, random_state=cv_seed, return_counts=True,
    )
    n_total = int(len(cell["brain"]))
    n_pred = int((counts > 0).sum())
    return {
        "r": _spearman(stats.rankdata(cell["brain"]), pred),
        "n_pairs_total": n_total,
        "n_pairs_oof_predicted": n_pred,
        "oof_pair_coverage": n_pred / n_total if n_total else np.nan,
    }


def compute_rows(
    subjects: list[str],
    rsa_types: list[str],
    cv_repeats: int = rr.N_CV_REPEATS,
    cv_seed: int = rr.CV_RANDOM_STATE,
) -> pd.DataFrame:
    all_rows = []
    print(f"Loading subject data: {subjects}", flush=True)
    subject_data = {subject: rr._load_subject_reps(subject) for subject in subjects}

    for rsa_type in rsa_types:
        print(f"\n== {rsa_type} RSA ==", flush=True)
        for subject in subjects:
            print(f"  {subject}", flush=True)
            cells = {
                group: _vectorize_group(subject, subject_data[subject], group, rsa_type)
                for group in GROUPS
            }
            within = {
                group: within_cv(cells[group], cv_repeats=cv_repeats, cv_seed=cv_seed)
                for group in GROUPS
            }

            for train_group in GROUPS:
                for test_group in GROUPS:
                    res = fit_transfer(cells[train_group], cells[test_group])
                    row = {
                        "subject": subject,
                        "rsa_type": rsa_type,
                        "train_group": train_group,
                        "test_group": test_group,
                        "train_label": GROUP_LABELS[train_group],
                        "test_label": GROUP_LABELS[test_group],
                        "transfer_kind": (
                            "within_set_in_sample"
                            if train_group == test_group
                            else "cross_set_transfer"
                        ),
                        "n_train_stimuli": cells[train_group]["n_stimuli"],
                        "n_test_stimuli": cells[test_group]["n_stimuli"],
                        "r_within_cv_test": within[test_group]["r"],
                        "within_cv_n_pairs_total_test": within[test_group]["n_pairs_total"],
                        "within_cv_n_pairs_oof_predicted_test": within[test_group]["n_pairs_oof_predicted"],
                        "within_cv_oof_pair_coverage_test": within[test_group]["oof_pair_coverage"],
                        "cv_repeats_image_blocked": int(cv_repeats),
                        "cv_random_state": int(cv_seed),
                        "correlation_ceiling_test": cells[test_group]["correlation_ceiling"],
                        **res,
                    }
                    row["transfer_gap_to_ceiling"] = (
                        row["correlation_ceiling_test"] - row["r_test_transfer"]
                    )
                    row["within_cv_gap_to_ceiling"] = (
                        row["correlation_ceiling_test"] - row["r_within_cv_test"]
                    )
                    row["transfer_penalty_vs_within_cv"] = (
                        row["r_within_cv_test"] - row["r_test_transfer"]
                    )
                    row["transfer_fraction_ceiling"] = (
                        row["r_test_transfer"] / row["correlation_ceiling_test"]
                        if row["correlation_ceiling_test"] > 0 else np.nan
                    )
                    row["within_cv_fraction_ceiling"] = (
                        row["r_within_cv_test"] / row["correlation_ceiling_test"]
                        if row["correlation_ceiling_test"] > 0 else np.nan
                    )
                    all_rows.append(row)
                    print(
                        f"    {train_group:10s} -> {test_group:10s} "
                        f"r={row['r_test_transfer']:.3f} "
                        f"within={row['r_within_cv_test']:.3f} "
                        f"ceil={row['correlation_ceiling_test']:.3f}",
                        flush=True,
                    )
    return pd.DataFrame(all_rows)


def _bar_transfer(ax, df: pd.DataFrame, rsa_type: str) -> None:
    rows = df[df.rsa_type == rsa_type]
    categories = [
        ("vicco", "vicco", "Baseline\nwithin CV", COLOR_BASELINE),
        ("all_models", "vicco", "Diagnostic ->\nbaseline", COLOR_TRANSFER),
        ("all_models", "all_models", "Diagnostic\nwithin CV", COLOR_CSTIM),
        ("vicco", "all_models", "Baseline ->\ndiagnostic", COLOR_TRANSFER),
    ]
    xs = np.arange(len(categories))
    means, sems, colors = [], [], []
    ceiling_means = []
    for train_group, test_group, _, color in categories:
        sub = rows[(rows.train_group == train_group) & (rows.test_group == test_group)]
        metric = "r_within_cv_test" if train_group == test_group else "r_test_transfer"
        means.append(sub[metric].mean())
        sems.append(_sem(sub[metric]))
        colors.append(color)
        ceiling_means.append(sub["correlation_ceiling_test"].mean())

    ax.bar(
        xs, means, yerr=sems, color=colors, alpha=0.62, edgecolor=colors,
        linewidth=0.8, capsize=2, error_kw=dict(ecolor="0.25", elinewidth=0.7),
    )
    for x, ceiling, color in zip(xs, ceiling_means, colors):
        ax.hlines(ceiling, x - 0.32, x + 0.32, colors=color, linewidth=1.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([c[2] for c in categories])
    ax.set_ylabel("Spearman r")
    ax.set_title(f"{rsa_type.capitalize()} ensemble transfer", fontsize=FONT["title"])
    ax.set_ylim(0, 0.85)
    ax.grid(axis="y", alpha=0.20, linewidth=0.5)


def _paired_penalty(ax, df: pd.DataFrame, rsa_type: str) -> None:
    rows = df[df.rsa_type == rsa_type]
    contrasts = [
        ("all_models", "vicco", "Diagnostic ->\nbaseline"),
        ("vicco", "all_models", "Baseline ->\ndiagnostic"),
    ]
    for x, (train_group, test_group, _) in enumerate(contrasts):
        sub = rows[(rows.train_group == train_group) & (rows.test_group == test_group)]
        vals = sub.set_index("subject")["transfer_penalty_vs_within_cv"].sort_index()
        jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) else []
        ax.scatter(np.full(len(vals), x) + jitter, vals, s=24, color="#333333",
                   alpha=0.75, linewidth=0, zorder=3)
        ax.bar(x, vals.mean(), yerr=_sem(vals), color="#999999", alpha=0.35,
               edgecolor="#666666", capsize=2, linewidth=0.8, zorder=2,
               error_kw=dict(ecolor="0.25", elinewidth=0.7))
        ax.text(x, vals.mean() + _sem(vals) + 0.025, f"{vals.mean():+.2f}",
                ha="center", va="bottom", fontsize=FONT["small"])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(np.arange(len(contrasts)))
    ax.set_xticklabels([c[2] for c in contrasts])
    ax.set_ylabel("Within-CV minus transfer r")
    ax.set_title(f"{rsa_type.capitalize()} transfer penalty", fontsize=FONT["title"])
    ax.set_ylim(-0.08, 0.22)
    ax.grid(axis="y", alpha=0.20, linewidth=0.5)


def make_figure(df: pd.DataFrame, rsa_types: list[str]) -> plt.Figure:
    if len(rsa_types) == 1:
        fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.8))
        axes = np.asarray([axes])
    else:
        fig, axes = plt.subplots(len(rsa_types), 2, figsize=(W_DOUBLE, 3.7 * len(rsa_types)))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.16,
                        hspace=0.48, wspace=0.28)
    for row_idx, rsa_type in enumerate(rsa_types):
        _bar_transfer(axes[row_idx, 0], df, rsa_type)
        _paired_penalty(axes[row_idx, 1], df, rsa_type)
    fig.suptitle(
        "Model-ensemble weights do not remove the diagnostic gap across stimulus regimes",
        fontsize=FONT["title"] + 1,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="all")
    parser.add_argument("--rsa-type", choices=["mixed", "fixed", "all"], default="mixed")
    parser.add_argument("--output", default=None)
    parser.add_argument("--cv-repeats", type=int, default=rr.N_CV_REPEATS)
    parser.add_argument("--cv-seed", type=int, default=rr.CV_RANDOM_STATE)
    args = parser.parse_args()

    subjects = config.SUBJECTS if args.subject == "all" else rr.parse_subject_arg(args.subject)
    rsa_types = ["fixed", "mixed"] if args.rsa_type == "all" else [args.rsa_type]

    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    df = compute_rows(
        subjects,
        rsa_types,
        cv_repeats=args.cv_repeats,
        cv_seed=args.cv_seed,
    )
    out_csv = Path(args.output) if args.output else OUT_DATA / "ensemble_transfer.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    fig = make_figure(df, rsa_types)
    suffix = "all" if len(rsa_types) > 1 else rsa_types[0]
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"ensemble_transfer_{suffix}.{ext}"
        fig.savefig(out, dpi=DPI if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)

    summary = (
        df.groupby(["rsa_type", "train_group", "test_group"])[[
            "r_test_transfer", "r_within_cv_test", "correlation_ceiling_test",
            "transfer_penalty_vs_within_cv", "transfer_gap_to_ceiling",
        ]]
        .agg(["mean", "sem"])
        .round(3)
    )
    print("\nTransfer summary:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
