"""Deterministic sampling and split helpers."""

from __future__ import annotations

import numpy as np


def bootstrap_sample_indices(
    n_total: int, n_sample: int, n_bootstrap: int = 10, seed: int = 0
) -> list[np.ndarray]:
    """Generate sorted samples without replacement using deterministic seeds."""
    samples = []
    for i in range(n_bootstrap):
        rng = np.random.default_rng(seed + i)
        idx = rng.choice(n_total, size=n_sample, replace=False)
        samples.append(np.sort(idx))
    return samples


def stimulus_cv_splits(n_stim: int, n_splits: int = 10, random_state: int = 42):
    """Generate train/test pair indices using stimulus-level cross-validation.

    Pairs sharing a stimulus with the held-out set are excluded entirely,
    avoiding information leakage through shared stimuli.
    """
    rng = np.random.default_rng(random_state)
    stim_indices = rng.permutation(n_stim)

    pair_stim = []
    for i in range(n_stim):
        for j in range(i + 1, n_stim):
            pair_stim.append((i, j))
    pair_stim = np.array(pair_stim)

    splits = []
    fold_size = n_stim // n_splits
    for k in range(n_splits):
        start = k * fold_size
        end = start + fold_size if k < n_splits - 1 else n_stim
        test_stim = set(stim_indices[start:end].tolist())

        train_mask = np.array(
            [s[0] not in test_stim and s[1] not in test_stim for s in pair_stim]
        )
        test_mask = np.array(
            [s[0] in test_stim and s[1] in test_stim for s in pair_stim]
        )
        splits.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    return splits


__all__ = ["bootstrap_sample_indices", "stimulus_cv_splits"]
