# -*- coding: utf-8 -*-
"""
01_compare_simulation_to_fmri.py

Test whether simulation-predicted model discriminability matches fMRI-observed
pairwise score differences.

Pairwise difference correlation (Option B):
  - Simulation: pairwise RDM distance (1 - corr) from correlation_matrices.csv
    (selected_clean matrix, raw and per-subject encoding tracks)
  - fMRI: |wrsa_transfer_i - wrsa_transfer_j| for each model pair
  - Run for both controversial stimuli (the selected set) and vicco (generic baseline)
  - Correlate across model pairs; aggregate with subject-averaged permutation test

Inputs:
  Simulation: 00_stimulus_selection/decision_checks/selection_evaluation/results/{model_set}/
              correlation_matrices.csv
  fMRI: 01_brain_model_alignment/results/rsa_scores/{sub}/wrsa_transfer_scores.csv

Outputs:
  05_controls_and_supplementary/simulation_validation/results/option_b_pairwise.csv
  05_controls_and_supplementary/simulation_validation/results/prediction_summary.csv

Usage:
  python 01_compare_simulation_to_fmri.py
"""

import sys
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))

_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

from cstims.paper import config

# =============================================================================
# Paths
# =============================================================================

FMRI_DATA_DIR = config.BRAIN_DATA_DIR
OUTPUT_DIR = config.SIM_DATA_DIR

MODEL_SETS_LIST = ["architecture", "training_objective", "sota", "dataset", "all_models"]
SUBJECTS = config.SUBJECTS


def get_sim_eval_dir(model_set: str) -> Path:
    # Use the _unique_boot variant: simulation side must use per-subject
    # unique encodings to match the fMRI side (wrsa_transfer_scores.csv),
    # which also uses per-subject unique encodings. The default
    # config.get_eval_pipeline_dir() returns the shared-encoding variant
    # and would produce a cross-encoder comparison.
    return config.EVAL_DATA_DIR / f"{model_set}_unique_boot"


def load_sim_correlation_matrix(model_set: str) -> pd.DataFrame:
    path = get_sim_eval_dir(model_set) / "correlation_matrices.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)


def load_fmri_scores(subject: str) -> pd.DataFrame:
    # Must read from RSA_DATA_DIR, not from the brain-data cache, which only
    # contains raw betas/stim info and not RSA scores.
    path = config.RSA_DATA_DIR / subject / "wrsa_transfer_scores.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# =============================================================================
# Core pairwise computation
# =============================================================================

def _compute_pairwise_for_track(
    corr_df: pd.DataFrame,
    fmri_scores: pd.Series,
    track: str,
) -> dict:
    """Compute pairwise sim distances and fMRI diffs for a given track.

    Returns dict with keys: sim_distances, fmri_diffs, pair_labels,
    model_names, versa_scores. Or None if insufficient data.
    """
    sim_mask = (
        (corr_df["track"] == track) &
        (corr_df["matrix_type"] == "selected_clean")
    )
    sim_pairs = corr_df[sim_mask]
    if sim_pairs.empty:
        return None

    common_models = sorted(set(sim_pairs["model_i"].unique()) & set(fmri_scores.index))
    if len(common_models) < 3:
        return None

    sim_distances, fmri_diffs, pair_labels = [], [], []
    for mi, mj in combinations(common_models, 2):
        sim_row = sim_pairs[
            (sim_pairs["model_i"] == mi) & (sim_pairs["model_j"] == mj)
        ]
        if sim_row.empty:
            sim_row = sim_pairs[
                (sim_pairs["model_i"] == mj) & (sim_pairs["model_j"] == mi)
            ]
        if sim_row.empty:
            continue
        sim_distances.append(1 - sim_row["correlation"].values[0])
        fmri_diffs.append(abs(fmri_scores[mi] - fmri_scores[mj]))
        pair_labels.append(f"{mi}||{mj}")

    if len(sim_distances) < 3:
        return None

    return {
        "sim_distances": np.array(sim_distances),
        "fmri_diffs": np.array(fmri_diffs),
        "pair_labels": pair_labels,
        "model_names": common_models,
        "versa_scores": np.array([fmri_scores[m] for m in common_models]),
    }


# =============================================================================
# Permutation test (subject-averaged)
# =============================================================================

def mantel_permutation_test_multisubject(
    per_subject_data: list,
    n_perm: int = 10000,
    seed: int = 42,
) -> tuple:
    """Mantel permutation test averaged across subjects.

    For each permutation: shuffle model labels per subject independently,
    compute pairwise Spearman rho per subject, average across subjects.
    This correctly accounts for pairwise dependence AND subject-level variance.

    Parameters
    ----------
    per_subject_data : list of dicts, each with keys:
        sim_distances, versa_scores, model_names
    n_perm : int
    seed : int

    Returns
    -------
    observed_mean : float
        Mean per-subject rho.
    perm_p : float
        Fraction of permuted means >= observed.
    null_means : array
    observed_rhos : list of per-subject rho values
    """
    rng = np.random.default_rng(seed)

    observed_rhos = []
    for sdata in per_subject_data:
        n = len(sdata["model_names"])
        pairs = list(combinations(range(n), 2))
        diffs = np.array([abs(sdata["versa_scores"][i] - sdata["versa_scores"][j])
                          for i, j in pairs])
        r, _ = stats.spearmanr(sdata["sim_distances"], diffs)
        observed_rhos.append(r)
    observed_mean = np.mean(observed_rhos)

    null_means = np.empty(n_perm)
    for p in range(n_perm):
        perm_rhos = []
        for sdata in per_subject_data:
            n = len(sdata["model_names"])
            pairs = list(combinations(range(n), 2))
            perm_scores = rng.permutation(sdata["versa_scores"])
            perm_diffs = np.array([abs(perm_scores[i] - perm_scores[j])
                                   for i, j in pairs])
            r, _ = stats.spearmanr(sdata["sim_distances"], perm_diffs)
            perm_rhos.append(r)
        null_means[p] = np.mean(perm_rhos)

    perm_p = (np.sum(null_means >= observed_mean) + 1) / (n_perm + 1)
    return observed_mean, perm_p, null_means, observed_rhos


# =============================================================================
# Main analysis
# =============================================================================

def run_analysis() -> tuple:
    """
    For each model_set × subject × track_type × stimulus_type:
      - Compute pairwise sim distances vs fMRI |diff|
      - Record per-subject Spearman rho

    Then aggregate: mean rho across subjects with subject-averaged permutation test.

    Returns
    -------
    pairwise_df : pd.DataFrame
        Per-pair data for plotting.
    summary_df : pd.DataFrame
        Aggregated stats per (model_set, track_type, stimulus_type).
    """
    pairwise_rows = []
    # Key: (model_set, track_type, stimulus_type) -> list of per-subject dicts
    subject_collections = {}

    for model_set in MODEL_SETS_LIST:
        print(f"\n--- {model_set} ---")

        try:
            corr_df = load_sim_correlation_matrix(model_set)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

        for subject in SUBJECTS:
            fmri_df = load_fmri_scores(subject)
            if fmri_df.empty:
                continue

            for stim_type in ["controversial", "vicco"]:
                if stim_type == "controversial":
                    fmri_scores = (
                        fmri_df[
                            (fmri_df["model_set"] == model_set) &
                            (fmri_df["stimulus_type"] == "controversial")
                        ]
                        .set_index("model")["wrsa_transfer"]
                    )
                else:
                    # vicco: average bootstrap samples per model
                    fmri_scores = (
                        fmri_df[
                            (fmri_df["model_set"] == model_set) &
                            (fmri_df["stimulus_type"] == "vicco")
                        ]
                        .groupby("model")["wrsa_transfer"]
                        .mean()
                    )

                if fmri_scores.empty:
                    continue

                for track_type, track_name in [("raw", "raw"), ("encoding", subject)]:
                    result = _compute_pairwise_for_track(corr_df, fmri_scores, track_name)
                    if result is None:
                        continue

                    r_s, _ = stats.spearmanr(result["sim_distances"], result["fmri_diffs"])

                    for sd, fd, pl in zip(
                        result["sim_distances"], result["fmri_diffs"], result["pair_labels"]
                    ):
                        pairwise_rows.append({
                            "model_set": model_set,
                            "subject": subject,
                            "track_type": track_type,
                            "stimulus_type": stim_type,
                            "pair": pl,
                            "sim_distance": sd,
                            "fmri_score_diff": fd,
                            "subject_rho": r_s,
                        })

                    key = (model_set, track_type, stim_type)
                    if key not in subject_collections:
                        subject_collections[key] = []
                    subject_collections[key].append({
                        "subject": subject,
                        "sim_distances": result["sim_distances"],
                        "versa_scores": result["versa_scores"],
                        "model_names": result["model_names"],
                    })

    # Aggregated permutation tests
    print("\n--- Aggregated permutation tests ---")
    summary_rows = []
    for (model_set, track_type, stim_type), slist in subject_collections.items():
        if len(slist) < 2:
            continue

        mean_rho, perm_p, _, per_subj_rhos = mantel_permutation_test_multisubject(
            [{"sim_distances": s["sim_distances"],
              "versa_scores": s["versa_scores"],
              "model_names": s["model_names"]} for s in slist]
        )

        n_pairs = len(slist[0]["sim_distances"])
        sig = "***" if perm_p < 0.001 else ("**" if perm_p < 0.01 else ("*" if perm_p < 0.05 else ""))
        print(f"  {model_set:20s} [{track_type:8s}, {stim_type:12s}]: "
              f"n_subj={len(slist)}, n_pairs={n_pairs:3d}, "
              f"mean_rho={mean_rho:.3f}, perm_p={perm_p:.4f} {sig}")

        summary_rows.append({
            "model_set": model_set,
            "track_type": track_type,
            "stimulus_type": stim_type,
            "mean_rho": mean_rho,
            "rho_sd": float(np.std(per_subj_rhos)),
            "perm_p": perm_p,
            "n_subjects": len(slist),
            "n_pairs": n_pairs,
            "per_subject_rhos": ";".join(f"{r:.4f}" for r in per_subj_rhos),
        })

    return pd.DataFrame(pairwise_rows), pd.DataFrame(summary_rows)


def compute_paired_comparisons(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Paired tests across subjects for key contrasts:
      - encoding vs raw (controversial stimuli)
      - controversial vs vicco (encoding track)

    Uses paired t-test (n=5 subjects). Reports t, df, p, mean difference.
    """
    rows = []

    def _paired_t(a: np.ndarray, b: np.ndarray, label: str):
        diffs = a - b
        t, p = stats.ttest_rel(a, b)
        rows.append({
            "comparison": label,
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "mean_diff": float(np.mean(diffs)),
            "sd_diff": float(np.std(diffs, ddof=1)),
            "t": float(t),
            "df": len(a) - 1,
            "p": float(p),
        })

    def _get_rhos(model_set, track_type, stimulus_type):
        row = summary_df[
            (summary_df["model_set"] == model_set) &
            (summary_df["track_type"] == track_type) &
            (summary_df["stimulus_type"] == stimulus_type)
        ]
        if row.empty:
            return None
        return np.array([float(x) for x in row.iloc[0]["per_subject_rhos"].split(";")])

    print("\n--- Paired comparisons ---")
    for model_set in MODEL_SETS_LIST:
        enc_cont = _get_rhos(model_set, "encoding", "controversial")
        raw_cont = _get_rhos(model_set, "raw",      "controversial")
        enc_vicc = _get_rhos(model_set, "encoding", "vicco")
        raw_vicc = _get_rhos(model_set, "raw",      "vicco")

        if enc_cont is not None and raw_cont is not None:
            _paired_t(enc_cont, raw_cont, f"{model_set}: encoding vs raw (controversial)")
        if enc_cont is not None and enc_vicc is not None:
            _paired_t(enc_cont, enc_vicc, f"{model_set}: controversial vs vicco (encoding)")
        if raw_cont is not None and raw_vicc is not None:
            _paired_t(raw_cont, raw_vicc, f"{model_set}: controversial vs vicco (raw)")

    comp_df = pd.DataFrame(rows)
    for _, r in comp_df.iterrows():
        sig = "***" if r["p"] < 0.001 else ("**" if r["p"] < 0.01 else ("*" if r["p"] < 0.05 else ""))
        print(f"  {r['comparison']}")
        print(f"    mean_diff={r['mean_diff']:+.3f}, t({r['df']:.0f})={r['t']:.2f}, p={r['p']:.4f} {sig}")
    return comp_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairwise_df, summary_df = run_analysis()

    if not pairwise_df.empty:
        pairwise_df.to_csv(OUTPUT_DIR / "option_b_pairwise.csv", index=False)
        print(f"\nSaved pairwise data → {OUTPUT_DIR / 'option_b_pairwise.csv'}")

    if not summary_df.empty:
        summary_df.to_csv(OUTPUT_DIR / "prediction_summary.csv", index=False)
        print(f"Saved summary → {OUTPUT_DIR / 'prediction_summary.csv'}")

    if not summary_df.empty:
        comp_df = compute_paired_comparisons(summary_df)
        comp_df.to_csv(OUTPUT_DIR / "paired_comparisons.csv", index=False)
        print(f"Saved paired comparisons → {OUTPUT_DIR / 'paired_comparisons.csv'}")


if __name__ == "__main__":
    main()
