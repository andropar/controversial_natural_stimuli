#!/usr/bin/env python3
"""Compute image diversity metrics for stimulus selection results.

Outputs:
- diversity.csv: Summary diversity metrics (mean pairwise similarity, feature spread)
- pairwise_similarities.csv: Full pairwise similarity matrix (optional)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import entropy

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.utils import (
    add_standard_args,
    setup_from_args,
    get_output_dir,
    load_features_for_track,
    get_all_tracks_for_evaluation,
)


def compute_pairwise_cosine_similarity(features: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix.

    Args:
        features: Tensor of shape (n_samples, n_features)

    Returns:
        Tensor of shape (n_samples, n_samples) with pairwise cosine similarities
    """
    # Normalize features
    features_norm = features / (features.norm(dim=1, keepdim=True) + 1e-8)
    # Compute similarity matrix
    similarity = features_norm @ features_norm.T
    return similarity


def compute_feature_spread(features: torch.Tensor) -> dict:
    """Compute feature spread metrics.

    Args:
        features: Tensor of shape (n_samples, n_features)

    Returns:
        Dict with spread metrics
    """
    # Variance across samples for each feature, then mean
    var_per_feature = features.var(dim=0)
    mean_var = float(var_per_feature.mean())

    # Total variance (trace of covariance matrix)
    centered = features - features.mean(dim=0)
    total_var = float((centered ** 2).sum() / (features.shape[0] - 1))

    # Entropy of feature distribution (discretized)
    # Use histogram-based entropy estimation
    features_np = features.cpu().numpy().flatten()
    hist, _ = np.histogram(features_np, bins=50, density=True)
    hist = hist + 1e-10  # Avoid log(0)
    feature_entropy = float(entropy(hist))

    return {
        "mean_feature_variance": mean_var,
        "total_variance": total_var,
        "feature_entropy": feature_entropy,
    }


def compute_diversity_metrics(
    selected_features: dict[str, torch.Tensor],
    random_features_np: dict[str, np.ndarray] | None = None,
    n_random_samples: int = 50,
    device: torch.device | None = None,
) -> tuple[dict, np.ndarray]:
    """Compute diversity metrics for selected stimuli.

    Args:
        selected_features: Dict mapping model name to selected features tensor
        random_features_np: Optional dict mapping model name to random features (numpy, on CPU)
        n_random_samples: Number of random samples to compare against
        device: Device for computation

    Returns:
        Tuple of (metrics_dict, pairwise_similarity_matrix)
    """
    # Use first model's features for diversity analysis
    model_name = list(selected_features.keys())[0]
    features = selected_features[model_name]
    n_selected = features.shape[0]

    if device is None:
        device = features.device

    # Compute pairwise similarity
    sim_matrix = compute_pairwise_cosine_similarity(features)

    # Extract upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
    pairwise_sims = sim_matrix[mask]

    # Compute statistics
    metrics = {
        "n_selected": n_selected,
        "n_pairs": int(pairwise_sims.numel()),
        "mean_pairwise_sim": float(pairwise_sims.mean()),
        "std_pairwise_sim": float(pairwise_sims.std()),
        "min_pairwise_sim": float(pairwise_sims.min()),
        "max_pairwise_sim": float(pairwise_sims.max()),
        "median_pairwise_sim": float(pairwise_sims.median()),
    }

    # Compute feature spread
    spread = compute_feature_spread(features)
    metrics.update(spread)

    # Compare to random baseline if available
    # Only load small subsets to GPU to avoid OOM
    if random_features_np is not None:
        random_feat_np = random_features_np[model_name]
        n_random_total = random_feat_np.shape[0]

        # Sample n_selected random stimuli multiple times and compute mean similarity
        random_sims = []
        rng = np.random.default_rng(42)
        for _ in range(n_random_samples):
            idx = rng.choice(n_random_total, size=n_selected, replace=False)
            # Only move the small subset to GPU
            random_subset = torch.from_numpy(random_feat_np[idx]).to(device=device, dtype=torch.float32)
            random_sim = compute_pairwise_cosine_similarity(random_subset)
            random_mask = torch.triu(torch.ones_like(random_sim), diagonal=1).bool()
            random_sims.append(float(random_sim[random_mask].mean()))

        metrics["random_mean_pairwise_sim"] = float(np.mean(random_sims))
        metrics["random_std_pairwise_sim"] = float(np.std(random_sims))
        metrics["diversity_vs_random"] = metrics["mean_pairwise_sim"] - metrics["random_mean_pairwise_sim"]

    return metrics, sim_matrix.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Compute diversity metrics for stimulus selection"
    )
    add_standard_args(parser)
    parser.add_argument(
        "--save-pairwise",
        action="store_true",
        help="Save full pairwise similarity matrix to CSV",
    )
    parser.add_argument(
        "--which-selection",
        type=str,
        choices=["final", "greedy", "best_raw_combined"],
        default="final",
        help=(
            "Which selection variant to evaluate: "
            "'final' (after refinement), 'greedy' (before refinement), "
            "or 'best_raw_combined' (best raw combined score). "
            "Default: final"
        ),
    )
    args = parser.parse_args()

    # Setup
    payload, output_dir, device = setup_from_args(args)

    # Load features using the same infrastructure as discriminability
    print(f"Loading features for selection variant: {args.which_selection}")

    # Get tracks - use raw/identity track for diversity analysis
    tracks = get_all_tracks_for_evaluation(payload)
    raw_track = None
    for track in tracks:
        if track.get("type", "identity") == "identity":
            raw_track = track
            break

    if raw_track is None:
        print("No raw/identity track found")
        return

    print(f"Using track '{raw_track['name']}' for diversity analysis")

    # Load selected and random features
    # Note: random features stay as numpy on CPU to avoid OOM
    selected_features, random_features_np = load_features_for_track(
        payload=payload,
        track=raw_track,
        device=device,
        n_random=10000,  # Use same pool size as discriminability
        selection_variant=args.which_selection,
    )

    if not selected_features:
        print("Could not load features")
        return

    n_selected = list(selected_features.values())[0].shape[0]
    print(f"Loaded features for {len(selected_features)} models")
    print(f"Selected: {n_selected} stimuli")
    print(f"Random pool: {list(random_features_np.values())[0].shape[0]} stimuli")

    # Compute diversity metrics with random comparison
    # Random features stay on CPU, only small subsets moved to GPU as needed
    print("Computing diversity metrics...")
    metrics, sim_matrix = compute_diversity_metrics(
        selected_features=selected_features,
        random_features_np=random_features_np,
        n_random_samples=50,  # Match discriminability
        device=device,
    )

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = output_dir / "diversity.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved diversity metrics to {metrics_path}")

    # Print summary
    print("\n=== Diversity Summary ===")
    print(f"  n_selected: {metrics['n_selected']}")
    print(f"  Mean pairwise similarity: {metrics['mean_pairwise_sim']:.4f}")
    print(f"  Std pairwise similarity: {metrics['std_pairwise_sim']:.4f}")
    if "random_mean_pairwise_sim" in metrics:
        print(f"  Random mean similarity: {metrics['random_mean_pairwise_sim']:.4f}")
        print(f"  Diversity vs random: {metrics['diversity_vs_random']:+.4f}")
    print(f"  Feature entropy: {metrics['feature_entropy']:.4f}")

    # Optionally save full pairwise matrix
    if args.save_pairwise:
        n = sim_matrix.shape[0]
        selected_indices = payload.get("selected_global_indices", list(range(n)))
        rows = []
        for i in range(n):
            for j in range(i + 1, n):
                rows.append({
                    "idx_i": selected_indices[i] if i < len(selected_indices) else i,
                    "idx_j": selected_indices[j] if j < len(selected_indices) else j,
                    "similarity": sim_matrix[i, j],
                })
        pairwise_df = pd.DataFrame(rows)
        pairwise_path = output_dir / "pairwise_similarities.csv"
        pairwise_df.to_csv(pairwise_path, index=False)
        print(f"Saved pairwise similarities to {pairwise_path}")

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
