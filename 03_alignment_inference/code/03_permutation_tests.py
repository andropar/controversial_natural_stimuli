"""
03_permutation_tests.py

Permutation tests for spread amplification.

For each model_set × method × dispersion metric, test whether the observed
controversial/baseline dispersion ratio exceeds chance. Under the null, any
of the (1 + n_vicco_bootstraps) score sets per subject could have been
labeled "controversial".

With n subjects and B vicco bootstraps per subject there are (B+1)^n total
assignments. If this is tractable (<= EXHAUSTIVE_MAX) we enumerate them
exhaustively; otherwise we draw N_MC_PERMUTATIONS Monte Carlo samples.

Inputs:
    01_brain_model_alignment/results/rsa_scores/{sub}/wrsa_transfer_scores.csv
    01_brain_model_alignment/results/rsa_scores/{sub}/crsa_scores.csv

Outputs:
    03_alignment_inference/results/permutation_test_results.csv

Usage:
    python 03_permutation_tests.py
"""

import itertools
import numpy as np
import pandas as pd
from pathlib import Path
import sys
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))


# Setup imports from cstim_paper root
_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))  # project root for cstims

from cstims import constants, paths


METHODS = {
    "wrsa_transfer": ("wrsa_transfer_scores.csv", "wrsa_transfer"),
    "crsa": ("crsa_scores.csv", "crsa"),
}

METRICS = ["range", "std", "iqr", "cv", "mean_pairwise_diff", "median_pairwise_diff"]

# Permutation strategy
EXHAUSTIVE_MAX = 10_000_000   # switch to MC sampling above this many total assignments
N_MC_PERMUTATIONS = 100_000   # number of random draws when sampling
MC_SEED = 0


def compute_spread_metrics(scores: np.ndarray) -> dict:
    """Compute spread metrics for a vector of model scores.

    Identical to 01_spread_statistics.py.
    """
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)
    n = len(scores)
    pairwise_diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairwise_diffs.append(abs(scores[i] - scores[j]))
    mean_pairwise_diff = np.mean(pairwise_diffs) if pairwise_diffs else 0.0
    median_pairwise_diff = np.median(pairwise_diffs) if pairwise_diffs else 0.0
    return {
        "range": np.max(scores) - np.min(scores),
        "std": std,
        "iqr": np.subtract(*np.percentile(scores, [75, 25])),
        "cv": std / mean if mean > 0 else np.nan,
        "mean_pairwise_diff": mean_pairwise_diff,
        "median_pairwise_diff": median_pairwise_diff,
    }


def load_score_sets(subject: str, model_set: str, filename: str,
                    score_col: str) -> list[np.ndarray] | None:
    """Load score vectors for a subject × model_set: [controversial, vicco_0..B-1].

    Uses all vicco bootstraps present in the CSV. Returns None if data is unavailable.
    """
    path = paths.rsa_data_dir() / subject / filename
    if not path.exists():
        return None
    df = pd.read_csv(path)
    ms_df = df[df["model_set"] == model_set]

    # Controversial (bootstrap_idx=0)
    cstim = ms_df[ms_df["stimulus_type"] == "controversial"]
    if cstim.empty:
        return None
    cstim_scores = cstim.sort_values("model")[score_col].values

    # All vicco bootstraps present for this subject × model_set
    vicco_df = ms_df[ms_df["stimulus_type"] == "vicco"]
    if vicco_df.empty:
        return None
    score_sets = [cstim_scores]
    for bidx in sorted(vicco_df["bootstrap_idx"].unique()):
        v = vicco_df[vicco_df["bootstrap_idx"] == bidx]
        score_sets.append(v.sort_values("model")[score_col].values)

    return score_sets


def compute_mean_ratio_vec(metric_tensor: np.ndarray,
                           assignments: np.ndarray,
                           metric_idx: int) -> np.ndarray:
    """Vectorised cross-subject mean dispersion ratio for many assignments.

    Args:
        metric_tensor: shape (n_subjects, n_sets, n_metrics), precomputed values
        assignments: shape (n_perms, n_subjects); each row picks one "controversial"
                     set index per subject
        metric_idx: index into METRICS

    Returns:
        ratios: shape (n_perms,), mean across subjects of (cstim / mean_baseline).
                NaN for subjects whose mean baseline is <= 0 or whose cstim is NaN
                are dropped before averaging across subjects.
    """
    n_subj, n_sets, _ = metric_tensor.shape
    vals = metric_tensor[:, :, metric_idx]                       # (n_subj, n_sets)
    total_per_subj = np.nansum(vals, axis=1)                     # (n_subj,)

    # assignments[:, s] is which set is "controversial" for subject s
    cstim_vals = vals[np.arange(n_subj)[None, :],
                      assignments]                               # (n_perms, n_subj)
    # Mean of the other sets = (total - cstim) / (n_sets - 1)
    mean_baseline = (total_per_subj[None, :] - cstim_vals) / (n_sets - 1)  # (n_perms, n_subj)

    with np.errstate(divide="ignore", invalid="ignore"):
        subj_ratios = np.where(
            (mean_baseline > 0) & np.isfinite(cstim_vals),
            cstim_vals / mean_baseline,
            np.nan,
        )
    # Mean across subjects, ignoring NaNs
    return np.nanmean(subj_ratios, axis=1)


def run_permutation_test(subject_score_sets: list[list[np.ndarray]],
                         ) -> list[dict]:
    """Run permutation test across all metrics.

    Uses exhaustive enumeration if total assignments <= EXHAUSTIVE_MAX, otherwise
    Monte Carlo sampling with N_MC_PERMUTATIONS draws.

    Args:
        subject_score_sets: list (per subject) of score vectors (1 cstim + B vicco)

    Returns:
        List of dicts with metric, observed_ratio, p_perm, n_permutations, mode
    """
    n_subjects = len(subject_score_sets)
    n_sets = len(subject_score_sets[0])  # auto-detect: 1 cstim + B vicco bootstraps

    # Precompute spread metrics → tensor shape (n_subjects, n_sets, n_metrics)
    metric_tensor = np.empty((n_subjects, n_sets, len(METRICS)), dtype=np.float64)
    for s, subj_sets in enumerate(subject_score_sets):
        for i, score_vec in enumerate(subj_sets):
            m = compute_spread_metrics(score_vec)
            for mi, k in enumerate(METRICS):
                metric_tensor[s, i, mi] = m[k]

    total_perms = n_sets ** n_subjects
    if total_perms <= EXHAUSTIVE_MAX:
        # Exhaustive enumeration
        assignments = np.array(
            list(itertools.product(range(n_sets), repeat=n_subjects)),
            dtype=np.int64,
        )
        mode = "exhaustive"
        n_perms_reported = total_perms
    else:
        # Monte Carlo sampling
        rng = np.random.default_rng(MC_SEED)
        assignments = rng.integers(0, n_sets, size=(N_MC_PERMUTATIONS, n_subjects),
                                   dtype=np.int64)
        # Force the observed assignment (all zeros) to be included for exactness
        assignments[0] = 0
        mode = "monte_carlo"
        n_perms_reported = N_MC_PERMUTATIONS

    results = []
    observed_assignment = np.zeros((1, n_subjects), dtype=np.int64)

    for mi, metric in enumerate(METRICS):
        observed = float(compute_mean_ratio_vec(metric_tensor,
                                                observed_assignment, mi)[0])
        all_ratios = compute_mean_ratio_vec(metric_tensor, assignments, mi)
        valid = np.isfinite(all_ratios)
        n_total = int(valid.sum())
        if n_total == 0:
            p_perm = np.nan
        else:
            # For MC, standard practice adds 1 to numerator & denominator
            # (guarantees p > 0). For exhaustive, observed is itself enumerated
            # so we report the raw proportion.
            if mode == "monte_carlo":
                n_ge = int(np.sum(all_ratios[valid] >= observed))
                p_perm = (n_ge + 1) / (n_total + 1)
            else:
                n_ge = int(np.sum(all_ratios[valid] >= observed))
                p_perm = n_ge / n_total

        results.append({
            "metric": metric,
            "observed_ratio": observed,
            "p_perm": p_perm,
            "n_permutations": n_perms_reported,
            "mode": mode,
        })

    return results


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted q-values in input order."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    # Raw BH adjustment
    adjusted = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity (reverse-cummin)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    q = np.empty_like(p)
    q[order] = adjusted
    return q


def main():
    all_results = []

    for model_set in constants.MODEL_SETS:
        for method, (filename, score_col) in METHODS.items():
            print(f"  {method:15s} | {model_set:25s}", end="", flush=True)

            # Collect available subjects
            subject_score_sets = []
            available_subjects = []
            for subject in constants.SUBJECTS:
                sets = load_score_sets(subject, model_set, filename, score_col)
                if sets is not None:
                    subject_score_sets.append(sets)
                    available_subjects.append(subject)

            if len(subject_score_sets) < 2:
                print(f" — skipped (only {len(subject_score_sets)} subjects)")
                continue

            n_subj = len(subject_score_sets)
            n_sets_per_subj = len(subject_score_sets[0])
            total_perms = n_sets_per_subj ** n_subj
            mode_label = ("exhaustive" if total_perms <= EXHAUSTIVE_MAX
                          else f"MC×{N_MC_PERMUTATIONS}")
            print(f" | {n_subj} subjects, {n_sets_per_subj} sets/subj, "
                  f"{total_perms} total perms ({mode_label})", end="", flush=True)

            metric_results = run_permutation_test(subject_score_sets)

            for r in metric_results:
                r["model_set"] = model_set
                r["method"] = method
                r["n_subjects"] = n_subj
                all_results.append(r)

            # Print summary for this combination
            sig_metrics = [r["metric"] for r in metric_results if r["p_perm"] < 0.05]
            print(f" | sig: {', '.join(sig_metrics) if sig_metrics else 'none'}")

    # ------------------------------------------------------------------
    # BH-FDR correction across the 10 spread-amplification tests
    # (5 model sets × 2 RSA methods) — per metric separately.
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(all_results)
    results_df["q_bh"] = np.nan
    for metric in METRICS:
        mask = results_df["metric"] == metric
        p = results_df.loc[mask, "p_perm"].values
        results_df.loc[mask, "q_bh"] = bh_fdr(p)

    cols = ["model_set", "method", "metric", "observed_ratio", "p_perm", "q_bh",
            "n_permutations", "mode", "n_subjects"]
    results_df = results_df[cols]
    output_path = paths.stats_data_dir() / "permutation_test_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Print summary table using BH-FDR q-values
    print(f"\n{'='*95}")
    print("PERMUTATION TEST RESULTS  (stars use BH-FDR q-values across 10 tests per metric)")
    print(f"{'='*95}")
    for _, row in results_df.iterrows():
        q = row["q_bh"]
        sig = "***" if q < 0.001 else "**" if q < 0.01 \
            else "*" if q < 0.05 else "ns"
        print(f"  {row['method']:15s} | {row['model_set']:25s} | "
              f"{row['metric']:20s} | ratio={row['observed_ratio']:.3f} | "
              f"p={row['p_perm']:.4f} q={q:.4f} {sig}")


if __name__ == "__main__":
    main()
