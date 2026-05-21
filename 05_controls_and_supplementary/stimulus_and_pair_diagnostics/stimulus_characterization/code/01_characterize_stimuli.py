#!/usr/bin/env python3
"""Characterize CSTIMS stimuli with available image statistics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / "14_stimulus_characterization" / "data"
OUT.mkdir(parents=True, exist_ok=True)
STATS = PAPER / "08_image_statistics" / "data" / "image_stats.csv"
FEATURE_COLS = [
    "lum_mean", "lum_rms", "colorfulness", "lab_chroma_mean", "hue_entropy",
    "sf_slope", "sf_high_low_ratio", "edge_mag_mean", "orient_anisotropy",
    "edge_com_x", "symmetry_lr", "entropy", "jpeg_ratio",
]


def main() -> None:
    df = pd.read_csv(STATS)
    summary = (
        df.groupby("stimulus_set")[FEATURE_COLS]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    summary.columns = ["_".join([str(x) for x in c if x]) for c in summary.columns.to_flat_index()]
    summary.to_csv(OUT / "stimulus_low_level_characterization.csv", index=False)

    # Cluster in the available low-level-statistics space. These are not semantic
    # labels, but they give a reproducible characterization of visual regimes.
    x = StandardScaler().fit_transform(df[FEATURE_COLS].fillna(df[FEATURE_COLS].median()))
    n_clusters = 8
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=20).fit_predict(x)
    df_out = df[["stimulus_set", "image"]].copy()
    df_out["low_level_cluster"] = labels
    df_out.to_csv(OUT / "stimulus_low_level_clusters.csv", index=False)
    cluster_summary = (
        df_out.groupby(["stimulus_set", "low_level_cluster"])
        .size()
        .rename("n_images")
        .reset_index()
    )
    totals = df_out.groupby("stimulus_set").size().rename("set_n").reset_index()
    cluster_summary = cluster_summary.merge(totals, on="stimulus_set")
    cluster_summary["fraction"] = cluster_summary["n_images"] / cluster_summary["set_n"]
    cluster_summary.to_csv(OUT / "content_cluster_summary.csv", index=False)

    note = OUT / "characterization_feasibility_note.md"
    note.write_text(
        "# Stimulus Characterization Feasibility\n\n"
        "Available local annotations support low-level image statistics and "
        "clusters in that statistics space. No OCR/text detections, object counts, "
        "person/animal counts, or scene labels were present in the paper cache, so "
        "those fields are not asserted here. They can be added as a separate "
        "annotation table keyed by `stimulus_set` and `image`.\n"
    )
    print(f"saved {OUT}")
    print(summary.head().to_string(index=False))


if __name__ == "__main__":
    main()
