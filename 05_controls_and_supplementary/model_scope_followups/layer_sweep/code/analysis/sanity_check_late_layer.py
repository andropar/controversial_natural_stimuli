#!/usr/bin/env python3
"""Verify that late-layer features reproduce existing fRSA scores.

Loads features from the existing late-layer cache (02_rsa_scores pipeline)
and from our new multi-layer cache (last layer of MODEL_LAYERS), then computes
fRSA against brain RDM for each (subject, model, set) and asserts a small
absolute difference.

Failure here indicates a feature-extraction or aggregation mismatch.
"""

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT
import numpy as np
import pandas as pd

from config import (
    MODEL_DISPLAY_NAMES, PAPER_ROOT, get_brain_input_dir, MODEL_SETS,
)
from utils import compute_rdm_correlation, compute_rsa_score
from layers_config import MODEL_LAYERS, MAIN_LAYER, LATE_LAYER

CACHE_NEW = LAYER_SWEEP_ROOT / "cache_or_heavy" / "features"
EXISTING_CSV = PAPER_ROOT / "02_rsa_scores" / "data"
# Existing crsa_scores.csv was computed on CPU; we extract on GPU. The drift
# is up to ~5e-4 RSA points for transformer architectures (DINOv2). For
# convolutional models the agreement is ~1e-5. A 1e-3 floor is more than
# enough to catch any real bug but allows CPU/GPU numerical noise.
TOL = 1e-3

# Known existing-pipeline bugs that we deliberately diverge from. For
# robustness_imagenet_l2_eps3, get_graph_node_names() flips the model into
# train() mode and the FX trace fails before the model can be reset to
# eval() — so the existing crsa_scores.csv values for this model were
# computed with BatchNorm running on BATCH statistics, not the saved
# running stats. Our multilayer extractor explicitly calls .eval() after
# the FX attempt (added to fix the CORnet hook-during-train issue), so
# our values are correct and the existing csv values are not.
EXPECTED_DIVERGENT_MODELS = {"robustness_imagenet_l2_eps3"}


def load_subject(subject: str):
    d = get_brain_input_dir(subject)
    b = np.load(d / "cstim_betas_averaged.npz", allow_pickle=True)
    v = np.load(d / "voxel_metadata.npz", allow_pickle=True)
    si = pd.read_csv(d / "cstim_stimulus_info.csv")
    hlvis = v["hlvis_mask"]
    betas = b["betas"][hlvis, :]
    stim_keys = b["stim_keys"]
    k2i = {k: i for i, k in enumerate(stim_keys)}
    g_brain, g_file = {}, {}
    for g, sub in si.groupby("group"):
        g_brain[g] = np.array([k2i[k] for k in sub["stim_key"].values])
        idx = sub["stim_idx"].values
        g_file[g] = idx - 1 if g == "vicco" else idx
    return betas, g_brain, g_file


def main():
    failures = []
    for subject in ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]:
        try:
            betas, g_brain, g_file = load_subject(subject)
        except FileNotFoundError:
            print(f"[skip] {subject} no brain data")
            continue

        existing = pd.read_csv(EXISTING_CSV / subject / "crsa_scores.csv")
        # The existing csv has many duplicate (model, model_set, controversial) rows
        # because of multiple model_set runs that include the same model — pick one.
        existing = existing[existing["stimulus_type"] == "controversial"]

        for model in MODEL_LAYERS:
            late = MAIN_LAYER[model]
            for mset in ["all_models", "architecture", "dataset", "sota", "training_objective"]:
                if model not in MODEL_SETS[mset]:
                    continue
                if mset not in g_brain:
                    continue
                cache_p = CACHE_NEW / model / f"{mset}.npz"
                d = np.load(cache_p, allow_pickle=True)
                feats = d[late]
                feats_subj = feats[g_file[mset]]
                model_rdm = compute_rdm_correlation(feats_subj)
                brain_rdm = compute_rdm_correlation(betas[:, g_brain[mset]].T)
                new_rsa = compute_rsa_score(model_rdm, brain_rdm, "spearman")

                exists = existing[(existing["model"] == model) &
                                  (existing["model_set"] == mset)]
                if exists.empty:
                    continue
                old_rsa = exists.iloc[0]["crsa"]
                diff = abs(new_rsa - old_rsa)
                ok = diff <= TOL
                if model in EXPECTED_DIVERGENT_MODELS:
                    tag = "DIVERGENT (expected)" if not ok else "OK"
                else:
                    tag = "OK" if ok else "FAIL"
                print(f"  {subject} {model:<55} {mset:<22} new={new_rsa:.5f} old={old_rsa:.5f} diff={diff:.2e} [{tag}]")
                if not ok and model not in EXPECTED_DIVERGENT_MODELS:
                    failures.append((subject, model, mset, new_rsa, old_rsa, diff))

    if failures:
        print(f"\n{len(failures)} unexpected failures (tol={TOL}):")
        for f in failures:
            print(f"  {f}")
        raise SystemExit(1)
    print(f"\nAll late-layer fRSA values match existing crsa_scores.csv within tolerance,")
    print(f"except for the explicitly known-divergent models: {sorted(EXPECTED_DIVERGENT_MODELS)}")
    print("(see comment in sanity_check_late_layer.py for explanation).")


if __name__ == "__main__":
    main()
