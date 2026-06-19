#!/usr/bin/env python3
"""Rank single-pair counterfactual candidates.

This is a lightweight, read-only selector for the first counterfactual image
alignment probe.  It joins existing pair-level brain placement summaries with
cached selection-time model features and scores pairs by model-brain mismatch.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ANALYSIS_DIR / "results"

HELPER_DIR = ROOT / "src"
sys.path.insert(0, str(HELPER_DIR))
from cstims import constants, paths


PAIR_SUMMARY = (
    ROOT
    / "05_controls_and_supplementary"
    / "stimulus_and_pair_diagnostics"
    / "pair_level_brain_placement"
    / "results"
    / "pair_level_brain_placement_summary.csv"
)
PAIR_SUBJECT = (
    ROOT
    / "05_controls_and_supplementary"
    / "stimulus_and_pair_diagnostics"
    / "pair_level_brain_placement"
    / "results"
    / "pair_level_brain_placement.csv"
)
IMAGE_DIR = (
    ROOT
    / "00_stimulus_selection"
    / "decision_checks"
    / "selection_evaluation"
    / "results"
    / "all_models"
    / "images"
)

EASY_MODELS = [
    "torchvision_resnet50_imagenet1k_v1",
    "torchvision_alexnet_imagenet1k_v1",
    "torchvision_vgg16_imagenet1k_v1",
    "torchvision_convnext_base_imagenet1k_v1",
    "cornet_s",
    "vissl_resnet50_supervised",
]


def load_selection_features() -> dict[str, np.ndarray]:
    with open(paths.selected_stimuli_payload(), "rb") as f:
        payload = pickle.load(f)
    features = payload["selected_features_raw"]
    return {name: np.asarray(value) for name, value in features.items()}


def z_distance_matrix(features: np.ndarray) -> np.ndarray:
    dmat = squareform(pdist(features, metric="cosine"))
    triu = np.triu_indices(dmat.shape[0], k=1)
    zvals = zscore(dmat[triu])
    zmat = np.zeros_like(dmat, dtype=np.float64)
    zmat[triu] = zvals
    return zmat + zmat.T


def score_model_pairs(
    features_by_model: dict[str, np.ndarray],
    pair_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for model_name, features in features_by_model.items():
        zmat = z_distance_matrix(features)
        frame = pair_summary.copy()
        frame["model"] = model_name
        frame["model_z"] = [
            zmat[int(i), int(j)] for i, j in zip(frame["img_i"], frame["img_j"])
        ]
        frame["mismatch_mean"] = (frame["model_z"] - frame["mean_brain_z"]).abs()
        frame["direction"] = np.where(
            frame["model_z"] > frame["mean_brain_z"],
            "model_more_dissimilar",
            "model_more_similar",
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def add_image_paths(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["image_i_path"] = out["img_i"].map(
        lambda idx: str(IMAGE_DIR / f"image_{int(idx):04d}.png")
    )
    out["image_j_path"] = out["img_j"].map(
        lambda idx: str(IMAGE_DIR / f"image_{int(idx):04d}.png")
    )
    return out


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pair_summary = pd.read_csv(PAIR_SUMMARY)
    pair_subject = pd.read_csv(PAIR_SUBJECT)

    strong_consistent = pair_summary[
        (pair_summary["n_subjects"] == len(constants.SUBJECTS))
        & (pair_summary["sd_brain_z"] < 0.75)
        & (pair_summary["mean_brain_z"].abs() > 0.75)
    ].copy()

    features_by_model = load_selection_features()
    scored = score_model_pairs(features_by_model, strong_consistent)
    scored = add_image_paths(scored)
    scored = scored.sort_values(["mismatch_mean", "model_spread_z"], ascending=False)
    scored.to_csv(RESULTS_DIR / "candidate_pair_model_mismatches.csv", index=False)

    easy = scored[scored["model"].isin(EASY_MODELS)].copy()
    easy.to_csv(RESULTS_DIR / "candidate_pair_model_mismatches_easy_models.csv", index=False)

    subject_rows = []
    pair_keys = set(zip(strong_consistent["img_i"].astype(int), strong_consistent["img_j"].astype(int)))
    pair_subject = pair_subject[
        [
            (int(i), int(j)) in pair_keys
            for i, j in zip(pair_subject["img_i"], pair_subject["img_j"])
        ]
    ].copy()
    for model_name in EASY_MODELS:
        zmat = z_distance_matrix(features_by_model[model_name])
        frame = pair_subject.copy()
        frame["model"] = model_name
        frame["model_z"] = [
            zmat[int(i), int(j)] for i, j in zip(frame["img_i"], frame["img_j"])
        ]
        frame["mismatch_subject"] = (frame["model_z"] - frame["brain_z"]).abs()
        frame["direction"] = np.where(
            frame["model_z"] > frame["brain_z"],
            "model_more_dissimilar",
            "model_more_similar",
        )
        subject_rows.append(frame)
    subject_scored = pd.concat(subject_rows, ignore_index=True)
    subject_scored = add_image_paths(subject_scored)
    subject_scored = subject_scored.sort_values(
        ["mismatch_subject", "model_spread_z"], ascending=False
    )
    subject_scored.to_csv(
        RESULTS_DIR / "candidate_pair_subject_mismatches_easy_models.csv",
        index=False,
    )

    chosen = subject_scored[
        (subject_scored["model"] == "torchvision_resnet50_imagenet1k_v1")
        & (subject_scored["img_i"] == 22)
        & (subject_scored["img_j"] == 46)
        & (subject_scored["subject"] == "sub-06")
    ].copy()
    if chosen.empty:
        raise RuntimeError("Expected ResNet50 pair 22-46 sub-06 candidate not found.")
    chosen = add_image_paths(chosen)
    chosen.to_csv(RESULTS_DIR / "selected_probe_pair.csv", index=False)

    print("Wrote:")
    for name in [
        "candidate_pair_model_mismatches.csv",
        "candidate_pair_model_mismatches_easy_models.csv",
        "candidate_pair_subject_mismatches_easy_models.csv",
        "selected_probe_pair.csv",
    ]:
        print(f"  {RESULTS_DIR / name}")
    print("\nSelected first probe:")
    print(
        chosen[
            [
                "model",
                "img_i",
                "img_j",
                "subject",
                "brain_z",
                "model_z",
                "mismatch_subject",
                "model_spread_z",
                "eligible_high_disagreement",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
