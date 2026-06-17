#!/usr/bin/env python3
"""Combine encoding model results from a run directory.

This script reads metadata.json files from each model subdirectory and optionally
recomputes metrics for the hlvis ROI subset.

Usage:
    python combine_results.py /path/to/run_dir
    python combine_results.py  # uses most recent run

    # Also compute hlvis metrics (requires loading models and features)
    python combine_results.py --compute-hlvis
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent


def _find_share_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists() and (path / "src" / "cstims").exists():
            return path
    return start.parents[1]


PROJECT_ROOT = _find_share_root(SCRIPT_DIR)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


def find_latest_run(results_root: Path) -> Path:
    """Find the most recent encoding run directory."""
    runs = sorted(results_root.glob("*"), key=lambda p: p.stat().st_mtime)
    # Filter to directories that look like runs (have subdirs with encoding_model.npz)
    valid_runs = [r for r in runs if r.is_dir() and any(r.glob("*/encoding_model.npz"))]
    if not valid_runs:
        raise FileNotFoundError(f"No encoding runs found in {results_root}")
    return valid_runs[-1]


def load_alpha_stability(model_dir: Path) -> Optional[Dict]:
    """Load fold alphas and compute stability metrics.

    Args:
        model_dir: Path to model directory containing fold_alphas.npy

    Returns:
        Dict with alpha stability metrics, or None if file doesn't exist
    """
    fold_alphas_path = model_dir / "fold_alphas.npy"
    if not fold_alphas_path.exists():
        return None

    try:
        fold_alphas = np.load(fold_alphas_path)  # (n_folds, n_voxels)
    except Exception:
        return None

    if fold_alphas.ndim != 2 or fold_alphas.shape[0] < 2:
        return None

    # CV per voxel
    alpha_cv = np.std(fold_alphas, axis=0) / (np.mean(fold_alphas, axis=0) + 1e-10)

    # Fold-fold correlations on log alpha
    from scipy.stats import spearmanr
    fold_corrs = []
    for i in range(fold_alphas.shape[0]):
        for j in range(i + 1, fold_alphas.shape[0]):
            r, _ = spearmanr(
                np.log10(fold_alphas[i] + 1e-6),
                np.log10(fold_alphas[j] + 1e-6)
            )
            if not np.isnan(r):
                fold_corrs.append(r)

    return {
        "alpha_cv_median": float(np.median(alpha_cv)),
        "alpha_cv_mean": float(np.mean(alpha_cv)),
        "alpha_fold_corr_mean": float(np.mean(fold_corrs)) if fold_corrs else np.nan,
        "alpha_fold_corr_min": float(np.min(fold_corrs)) if fold_corrs else np.nan,
        "n_alpha_folds": fold_alphas.shape[0],
    }


def load_results_from_metadata(run_dir: Path) -> pd.DataFrame:
    """Load results by reading metadata.json from each model subdirectory."""
    records = []

    for subdir in sorted(run_dir.iterdir()):
        if not subdir.is_dir():
            continue

        meta_file = subdir / "metadata.json"
        if not meta_file.exists():
            continue

        try:
            with open(meta_file) as f:
                meta = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  Warning: Could not parse {meta_file}: {e}")
            continue

        # Build flat record
        record = {
            "model": meta.get("model"),
            "source": meta.get("source"),
            "layer": meta.get("layer"),
            "subject": meta.get("subject"),
            "voxel_set": meta.get("voxel_set"),
            "eval_method": meta.get("eval_method"),
            "elapsed_sec": meta.get("elapsed_sec"),
            "model_dir": str(subdir),
        }

        # Add preprocessing info
        preprocessing = meta.get("preprocessing", {})
        record["response_zscore"] = preprocessing.get("response_zscore", meta.get("scale_features"))
        record["srp"] = preprocessing.get("srp", False)
        record["scale_features"] = preprocessing.get("scale_features", meta.get("scale_features"))

        # Add metrics (these are for the full visual ROI)
        metrics = meta.get("metrics", {})
        # Try new field first, fall back to old
        record["veRSA_visual"] = metrics.get("veRSA_visual") or metrics.get("veRSA_pearson_r")
        record["voxel_r_median_visual"] = metrics.get("voxel_r_median")
        record["voxel_r_mean_visual"] = metrics.get("voxel_r_mean")
        record["image_r_median_visual"] = metrics.get("image_r_median")
        record["loo_cv_score_mean"] = metrics.get("loo_cv_score_mean")
        record["loo_cv_score_median"] = metrics.get("loo_cv_score_median")

        # Add hlvis metrics if present
        record["veRSA_hlvis"] = metrics.get("veRSA_hlvis")
        record["voxel_r_median_hlvis"] = metrics.get("voxel_r_median_hlvis")
        record["n_voxels_hlvis"] = metrics.get("n_voxels_hlvis")

        # Add fold mean veRSA if present
        record["veRSA_fold_mean"] = metrics.get("veRSA_fold_mean")

        # Legacy column names (for backwards compatibility)
        record["veRSA"] = record["veRSA_visual"]
        record["voxel_r_median"] = record["voxel_r_median_visual"]

        # Add per-dataset metrics from metrics dict
        per_dataset = metrics.get("per_dataset", {})
        for dataset_name, dataset_metrics in per_dataset.items():
            if isinstance(dataset_metrics, dict):
                record[f"veRSA_{dataset_name}"] = dataset_metrics.get("veRSA")
                record[f"voxel_r_{dataset_name}"] = dataset_metrics.get("voxel_r_median")

        # Add cross-dataset evaluation summary
        cross_eval = meta.get("cross_dataset_evaluation", {})
        if cross_eval:
            record["veRSA_overall"] = cross_eval.get("veRSA_overall")
            record["veRSA_mean_across_datasets"] = cross_eval.get("veRSA_mean")
            record["veRSA_std_across_datasets"] = cross_eval.get("veRSA_std")

            # Per-dataset veRSA from cross_eval
            for dataset_name, versa_val in cross_eval.get("veRSA_per_dataset", {}).items():
                record[f"veRSA_{dataset_name}"] = versa_val

        # Add alpha info
        alphas = meta.get("alphas", {})
        record["alpha_median"] = alphas.get("chosen_median")
        record["alpha_min"] = alphas.get("chosen_min")
        record["alpha_max"] = alphas.get("chosen_max")

        # Add alpha stability from metadata
        alpha_stab = meta.get("alpha_stability")
        if alpha_stab:
            record["alpha_cv_median"] = alpha_stab.get("cv_median")
            record["alpha_cv_mean"] = alpha_stab.get("cv_mean")
            record["alpha_fold_corr_mean"] = alpha_stab.get("fold_correlation_mean")
            record["alpha_fold_corr_min"] = alpha_stab.get("fold_correlation_min")
            record["n_alpha_folds"] = alpha_stab.get("n_folds")
        else:
            # Try loading from fold_alphas.npy if metadata doesn't have it
            alpha_stab_from_file = load_alpha_stability(subdir)
            if alpha_stab_from_file:
                record.update(alpha_stab_from_file)
            else:
                # Fallback to cross_eval for backwards compat
                record["alpha_cv_median"] = cross_eval.get("alpha_cv_median") if cross_eval else None

        records.append(record)

    if not records:
        raise ValueError(f"No metadata.json files found in {run_dir}")

    return pd.DataFrame(records)


def compute_hlvis_metrics(
    df: pd.DataFrame,
    cache_root: Path,
    deepvision_root: Path,
) -> pd.DataFrame:
    """Compute metrics for hlvis ROI subset.

    Since encoding models are fitted on the full visual ROI, we can evaluate
    them on the hlvis subset by filtering predictions and responses.
    """
    from cstims.datasets.deepvision import DeepVisionBenchmark
    from cstims.encoding import compute_versa, compute_voxel_r, compute_image_r
    from cstims.encoding.model import LinearEncodingModel

    print("\nComputing hlvis metrics...")

    # Group by subject to avoid reloading benchmark multiple times
    subjects = df["subject"].unique()
    subject_benchmarks: Dict[str, DeepVisionBenchmark] = {}

    hlvis_metrics = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing hlvis metrics"):
        subject = row["subject"]
        model_dir = Path(row["model_dir"])
        encoding_model_path = model_dir / "encoding_model.npz"
        features_path = model_dir / "features.npz"

        if not encoding_model_path.exists():
            print(f"  Skipping {row['model']} - no encoding_model.npz")
            hlvis_metrics.append({
                "veRSA_hlvis": np.nan,
                "voxel_r_median_hlvis": np.nan,
                "voxel_r_mean_hlvis": np.nan,
                "image_r_median_hlvis": np.nan,
                "n_voxels_hlvis": np.nan,
            })
            continue

        if not features_path.exists():
            print(f"  Skipping {row['model']} - no features.npz")
            hlvis_metrics.append({
                "veRSA_hlvis": np.nan,
                "voxel_r_median_hlvis": np.nan,
                "voxel_r_mean_hlvis": np.nan,
                "image_r_median_hlvis": np.nan,
                "n_voxels_hlvis": np.nan,
            })
            continue

        # Load benchmark for this subject (cached)
        if subject not in subject_benchmarks:
            subject_benchmarks[subject] = DeepVisionBenchmark(
                cache_root=cache_root,
                deepvision_fmri_root=deepvision_root,
                subject=subject,
                voxel_set="visual",  # Load full visual, we'll filter to hlvis
                build_rdms=False,
            )

        benchmark = subject_benchmarks[subject]

        # Load encoding model
        enc_model = LinearEncodingModel.load(encoding_model_path)

        # Load features
        with np.load(features_path) as z:
            features = z["features"]

        # Get responses and hlvis mask
        responses = benchmark.response_data.to_numpy().T  # (n_images, n_voxels)
        hlvis_mask = enc_model.roi_masks.get("hlvis")

        if hlvis_mask is None:
            print(f"  Skipping {row['model']} - no hlvis mask in model")
            hlvis_metrics.append({
                "veRSA_hlvis": np.nan,
                "voxel_r_median_hlvis": np.nan,
                "voxel_r_mean_hlvis": np.nan,
                "image_r_median_hlvis": np.nan,
                "n_voxels_hlvis": np.nan,
            })
            continue

        # Make predictions for hlvis voxels
        Y_pred_full = enc_model.predict(features)
        Y_pred_hlvis = Y_pred_full[:, hlvis_mask]
        Y_true_hlvis = responses[:, hlvis_mask]

        # Z-score for metric computation (like in fit_encoding_hydra.py)
        def zscore_cols(Y, eps=1e-6):
            mean = Y.mean(axis=0)
            std = Y.std(axis=0)
            std = np.maximum(std, eps)
            return (Y - mean) / std

        Y_true_z = zscore_cols(Y_true_hlvis)
        Y_pred_z = zscore_cols(Y_pred_hlvis)

        # Use odd/even split for evaluation
        test_idx = np.arange(1, len(features), 2)  # odd indices
        Y_test = Y_true_z[test_idx]
        Y_test_pred = Y_pred_z[test_idx]

        # Compute metrics
        veRSA = compute_versa(Y_test, Y_test_pred)
        voxel_r = compute_voxel_r(Y_test, Y_test_pred)
        image_r = compute_image_r(Y_test, Y_test_pred)

        hlvis_metrics.append({
            "veRSA_hlvis": float(veRSA),
            "voxel_r_median_hlvis": float(np.median(voxel_r)),
            "voxel_r_mean_hlvis": float(np.mean(voxel_r)),
            "image_r_median_hlvis": float(np.median(image_r)),
            "n_voxels_hlvis": int(hlvis_mask.sum()),
        })

    # Add hlvis metrics to dataframe
    hlvis_df = pd.DataFrame(hlvis_metrics)
    for col in hlvis_df.columns:
        df[col] = hlvis_df[col].values

    return df


def main():
    parser = argparse.ArgumentParser(description="Combine encoding results")
    parser.add_argument("run_dir", nargs="?", type=Path, help="Run directory")
    parser.add_argument(
        "--compute-hlvis", action="store_true",
        help="Compute metrics for hlvis ROI (requires loading models)"
    )
    args = parser.parse_args()

    # Determine run directory
    if args.run_dir:
        run_dir = args.run_dir
    else:
        # Find most recent run
        results_root = SCRIPT_DIR / "results"
        if not results_root.exists():
            # Try hydra outputs
            results_root = SCRIPT_DIR.parent.parent / "outputs" / "runs"
        if not results_root.exists():
            print(f"Results directory not found")
            sys.exit(1)
        run_dir = find_latest_run(results_root)

    print(f"Combining results from: {run_dir}")
    print("=" * 60)

    # Load from metadata.json files
    df = load_results_from_metadata(run_dir)
    print(f"Loaded {len(df)} model results")

    # Optionally compute hlvis metrics
    if args.compute_hlvis:
        from cstims.paths import deepvision_fmri_root

        cache_root = Path(
            os.environ.get(
                "CSTIMS_DEEPVISION_CACHE_ROOT",
                PROJECT_ROOT / "01_brain_model_alignment/cache_or_heavy/deepvision_benchmark_cache",
            )
        )
        deepvision_root = deepvision_fmri_root()

        df = compute_hlvis_metrics(df, cache_root, deepvision_root)

    # Sort by model name and subject
    df = df.sort_values(["model", "subject"])

    # Save combined results
    out_csv = run_dir / "combined_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved to: {out_csv}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary by Model")
    print("=" * 60)

    # Determine which veRSA column to use for summary
    versa_col = "veRSA_visual" if "veRSA_visual" in df.columns else "veRSA"

    # Group by model and compute mean across subjects
    if "subject" in df.columns and df["subject"].nunique() > 1:
        agg_dict = {
            versa_col: ["mean", "std"],
            "voxel_r_median": ["mean", "std"],
            "loo_cv_score_mean": "mean",
        }
        if "veRSA_hlvis" in df.columns:
            agg_dict["veRSA_hlvis"] = ["mean", "std"]

        summary = df.groupby("model").agg(agg_dict).round(4)
        summary.columns = ["_".join(col).strip("_") for col in summary.columns]
        summary = summary.sort_values(f"{versa_col}_mean", ascending=False)
        print(summary.to_string())
    else:
        cols = ["model", versa_col, "voxel_r_median", "loo_cv_score_mean"]
        if "veRSA_hlvis" in df.columns:
            cols.append("veRSA_hlvis")
        summary = df[cols].sort_values(versa_col, ascending=False)
        print(summary.to_string(index=False))

    # Print ROI comparison if hlvis metrics exist
    if "veRSA_hlvis" in df.columns and not df["veRSA_hlvis"].isna().all():
        print("\n" + "=" * 60)
        print("ROI Comparison: visual vs hlvis")
        print("=" * 60)

        # Get n_voxels from metadata
        n_voxels_visual = df["voxel_r_median_visual"].notna().sum()  # Proxy
        if "n_voxels_hlvis" in df.columns:
            n_voxels_hlvis = df["n_voxels_hlvis"].iloc[0]
        else:
            n_voxels_hlvis = "?"

        print(f"\n  Visual ROI: {df['veRSA_visual'].mean():.4f} ± {df['veRSA_visual'].std():.4f}")
        print(f"  hlvis ROI:  {df['veRSA_hlvis'].mean():.4f} ± {df['veRSA_hlvis'].std():.4f}")
        print(f"\n  n_voxels (hlvis): {n_voxels_hlvis}")

        # Rank correlation
        from scipy.stats import spearmanr
        valid = df[["veRSA_visual", "veRSA_hlvis"]].dropna()
        if len(valid) > 2:
            r, p = spearmanr(valid["veRSA_visual"], valid["veRSA_hlvis"])
            print(f"\n  Rank correlation (Spearman): r={r:.3f}, p={p:.3g}")

    # Print per-dataset breakdown if available
    dataset_cols = [c for c in df.columns if c.startswith("veRSA_")
                    and c not in ["veRSA_visual", "veRSA_hlvis", "veRSA_overall",
                                  "veRSA_mean_across_datasets", "veRSA_std_across_datasets"]]
    if dataset_cols:
        print("\n" + "=" * 60)
        print("Per-Dataset veRSA (mean across models)")
        print("=" * 60)
        for col in sorted(dataset_cols):
            dataset = col.replace("veRSA_", "")
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f"  {dataset:>15}: {mean_val:.4f} +/- {std_val:.4f}")


if __name__ == "__main__":
    main()
