"""
09_model_rdm_spreads.py

Compute the spread (std of RDM vector) of model RDMs for each stimulus group,
averaged across all models. Used to test whether model-predicted discriminability
of a stimulus set predicts the brain noise ceiling for that set.

Outputs:
  data/model_rdm_spreads.csv   — one row per group × bootstrap (vicco bootstrapped)

Usage:
  python 09_model_rdm_spreads.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))

import config
from utils import compute_rdm_correlation, rdm_to_vector

GROUPS_CSTIM = ["all_models", "architecture", "training_objective", "sota", "dataset"]
MODELS = config.MODEL_SETS["all_models"]
N_VICCO_SAMPLE = 100
N_VICCO_BOOTSTRAPS = 20


def model_rdm_spread_for_group(group: str, bootstrap_idx: int = 0, seed: int = 0) -> dict:
    """
    Mean RDM spread (std of upper-triangular RDM) across all models for one group.
    For vicco, draws a random sample of N_VICCO_SAMPLE stimuli.
    """
    spreads = []
    for model in MODELS:
        cache_path = config.CSTIM_FEATURE_CACHE / f"{model}.npz"
        if not cache_path.exists():
            continue
        data = np.load(cache_path)
        key = "vicco" if group == "vicco" else group
        if key not in data:
            continue
        feats = data[key]

        if group == "vicco":
            rng = np.random.default_rng(seed + bootstrap_idx)
            idx = rng.choice(len(feats), size=N_VICCO_SAMPLE, replace=False)
            feats = feats[idx]

        rdm_vec = rdm_to_vector(compute_rdm_correlation(feats))
        spreads.append(float(np.std(rdm_vec)))

    return {"model_rdm_spread": float(np.mean(spreads)), "n_models": len(spreads)}


def main():
    rows = []

    for group in GROUPS_CSTIM:
        result = model_rdm_spread_for_group(group)
        print(f"  {group:25s}: spread={result['model_rdm_spread']:.4f}  (n_models={result['n_models']})")
        rows.append({"group": group, "stimulus_type": "controversial",
                     "bootstrap_idx": 0, **result})

    print(f"  {'vicco':25s}: ", end="", flush=True)
    vicco_spreads = []
    for b in range(N_VICCO_BOOTSTRAPS):
        result = model_rdm_spread_for_group("vicco", bootstrap_idx=b)
        vicco_spreads.append(result["model_rdm_spread"])
        rows.append({"group": "vicco", "stimulus_type": "vicco",
                     "bootstrap_idx": b, **result})
    print(f"mean={np.mean(vicco_spreads):.4f}")

    df = pd.DataFrame(rows)
    out_path = config.STATS_DATA_DIR / "model_rdm_spreads.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
