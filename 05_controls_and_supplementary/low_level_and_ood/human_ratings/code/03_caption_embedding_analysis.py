#!/data/home_roth/_stachelschwein/miniforge3/envs/simfact/bin/python
"""Simple caption-text embedding analysis for the CSTIM annotation table.

Embeddings are sentence-transformer embeddings over the caption text.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", message="Using `TRANSFORMERS_CACHE` is deprecated.*", category=FutureWarning)

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer


STAGE = Path(__file__).resolve().parents[1]
ANNOTATIONS = STAGE / "outputs" / "annotations" / "full_minimax_m3_all_stimuli.csv"
OUT = STAGE / "outputs" / "caption_embedding_analysis"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CONDITION_ORDER = ["vicco", "all_models", "architecture", "dataset", "sota", "training_objective"]
CONDITION_LABELS = {
    "vicco": "Baseline",
    "all_models": "All-model",
    "architecture": "Arch.",
    "dataset": "Data",
    "sota": "SOTA",
    "training_objective": "Train.",
}


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for condition a minus baseline b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2) / (len(a) + len(b) - 2))
    if not np.isfinite(pooled) or pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def read_annotations() -> pd.DataFrame:
    if not ANNOTATIONS.exists():
        raise FileNotFoundError(f"Missing annotation table: {ANNOTATIONS}")
    df = pd.read_csv(ANNOTATIONS)
    df = df[df["condition"].isin(CONDITION_ORDER)].copy()
    if "validation_status" in df:
        df = df[df["validation_status"].eq("ok")].copy()
    df = df.drop_duplicates(["condition", "stim_idx"]).reset_index(drop=True)
    df["caption_text"] = df["short_caption"].fillna("").astype(str).str.strip()
    missing = df["caption_text"].eq("")
    if missing.any():
        df.loc[missing, "caption_text"] = df.loc[missing, "semantic_domain"].fillna("").astype(str)
    return df


def make_embeddings(texts: pd.Series) -> tuple[np.ndarray, pd.DataFrame]:
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    emb = model.encode(
        texts.fillna("").astype(str).tolist(),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    emb = np.asarray(emb, dtype=np.float32)
    meta = pd.DataFrame(
        {
            "embedding_method": ["sentence_transformer"],
            "embedding_model": [EMBEDDING_MODEL],
            "n_documents": [len(texts)],
            "n_dimensions": [emb.shape[1]],
            "normalized_embeddings": [True],
        }
    )
    return emb, meta


def greedy_matches(cstim_emb: np.ndarray, baseline_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dist = cdist(cstim_emb, baseline_emb, metric="euclidean")
    selected = []
    used: set[int] = set()
    for i in np.argsort(dist.min(axis=1)):
        for j in np.argsort(dist[i]):
            if int(j) not in used:
                selected.append((int(i), int(j)))
                used.add(int(j))
                break
    pairs = np.asarray(selected, dtype=int)
    pair_dist = dist[pairs[:, 0], pairs[:, 1]]
    return pairs, pair_dist


def summarize(df: pd.DataFrame, emb: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_mask = df["condition"].eq("vicco").to_numpy()
    baseline_emb = emb[baseline_mask]
    baseline_centroid = baseline_emb.mean(axis=0, keepdims=True)
    baseline_distance = cdist(baseline_emb, baseline_centroid, metric="euclidean").ravel()

    summary_rows = []
    match_rows = []
    baseline_rows = df[baseline_mask].reset_index(drop=True)

    for condition in CONDITION_ORDER[1:]:
        cond_mask = df["condition"].eq(condition).to_numpy()
        cond_emb = emb[cond_mask]
        cond_rows = df[cond_mask].reset_index(drop=True)
        cond_centroid = cond_emb.mean(axis=0, keepdims=True)
        cond_distance = cdist(cond_emb, baseline_centroid, metric="euclidean").ravel()

        pairs, pair_dist = greedy_matches(cond_emb, baseline_emb)
        matched_baseline_emb = baseline_emb[pairs[:, 1]]
        matched_centroid = matched_baseline_emb.mean(axis=0, keepdims=True)
        pre_centroid = float(cdist(cond_centroid, baseline_centroid, metric="euclidean")[0, 0])
        post_centroid = float(cdist(cond_centroid, matched_centroid, metric="euclidean")[0, 0])
        closed = 1 - (post_centroid / pre_centroid) if pre_centroid > 0 else np.nan

        summary_rows.append(
            {
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "n_condition": int(cond_mask.sum()),
                "n_baseline": int(baseline_mask.sum()),
                "baseline_mean_distance_to_baseline_centroid": float(baseline_distance.mean()),
                "condition_mean_distance_to_baseline_centroid": float(cond_distance.mean()),
                "mean_distance_difference": float(cond_distance.mean() - baseline_distance.mean()),
                "cohens_d_distance_to_baseline_centroid": cohen_d(cond_distance, baseline_distance),
                "pre_match_centroid_distance": pre_centroid,
                "post_match_centroid_distance": post_centroid,
                "fraction_centroid_distance_closed_by_matching": float(closed),
                "mean_nearest_match_distance": float(pair_dist.mean()),
                "median_nearest_match_distance": float(np.median(pair_dist)),
            }
        )

        for c_idx, b_idx in pairs:
            c_row = cond_rows.iloc[c_idx]
            b_row = baseline_rows.iloc[b_idx]
            match_rows.append(
                {
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "cstim_stimulus_id": c_row["stimulus_id"],
                    "cstim_stim_idx": int(c_row["stim_idx"]),
                    "cstim_caption": c_row["caption_text"],
                    "matched_baseline_stimulus_id": b_row["stimulus_id"],
                    "matched_baseline_stim_idx": int(b_row["stim_idx"]),
                    "matched_baseline_caption": b_row["caption_text"],
                    "embedding_distance": float(cdist(cond_emb[[c_idx]], baseline_emb[[b_idx]], metric="euclidean")[0, 0]),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(match_rows)


def write_coordinates(df: pd.DataFrame, emb: np.ndarray) -> None:
    meta = df[["stimulus_id", "condition", "stim_idx", "caption_text"]].reset_index(drop=True)
    emb_cols = pd.DataFrame(
        emb,
        columns=[f"caption_emb_{i:03d}" for i in range(emb.shape[1])],
    )
    coords = pd.concat([meta, emb_cols], axis=1)
    coords.to_csv(OUT / "caption_embedding_coordinates.csv", index=False)


def make_plot(summary: pd.DataFrame) -> None:
    x = np.arange(len(summary))
    labels = summary["condition_label"].to_list()

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), constrained_layout=True)

    ax = axes[0]
    ax.axhline(0, color="black", linewidth=0.8)
    ax.bar(x, summary["cohens_d_distance_to_baseline_centroid"], color="#009E73", width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Cohen's d")
    ax.set_title("Caption-semantic shift", loc="left", fontsize=9, fontweight="bold")
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    width = 0.34
    ax.bar(
        x - width / 2,
        summary["pre_match_centroid_distance"],
        width=width,
        color="#666666",
        label="all baseline",
    )
    ax.bar(
        x + width / 2,
        summary["post_match_centroid_distance"],
        width=width,
        color="#D55E00",
        label="caption-matched baseline",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("centroid distance")
    ax.set_title("Simple caption matching", loc="left", fontsize=9, fontweight="bold")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"caption_embedding_overview.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)


def write_report(summary: pd.DataFrame, meta: pd.DataFrame) -> None:
    rows = [
        "# Caption-Text Embedding Semantic Audit",
        "",
        f"Input: `{ANNOTATIONS}`",
        "",
        f"Embeddings: `{EMBEDDING_MODEL}` over `short_caption`, L2-normalized.",
        "This is a simple descriptive semantic-space diagnostic, not a causal control.",
        "",
        "## Embedding Metadata",
        "",
    ]
    rows.extend(f"- {col}: {meta[col].iloc[0]}" for col in meta.columns)
    rows.extend(["", "## CSTIM-vs-Baseline Summary", ""])

    cols = [
        "condition_label",
        "cohens_d_distance_to_baseline_centroid",
        "pre_match_centroid_distance",
        "post_match_centroid_distance",
        "fraction_centroid_distance_closed_by_matching",
        "mean_nearest_match_distance",
    ]
    table = summary[cols].copy()
    for col in cols[1:]:
        table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in table.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in cols) + " |")

    (OUT / "REPORT.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = read_annotations()
    emb, meta = make_embeddings(df["caption_text"])
    summary, matches = summarize(df, emb)

    meta.to_csv(OUT / "caption_embedding_metadata.csv", index=False)
    summary.to_csv(OUT / "caption_embedding_shift_summary.csv", index=False)
    matches.to_csv(OUT / "caption_embedding_matched_baselines.csv", index=False)
    write_coordinates(df, emb)
    make_plot(summary)
    write_report(summary, meta)

    print(f"read {len(df)} captions")
    print(f"embedding dimensions: {emb.shape[1]}")
    print(f"wrote caption-embedding outputs to {OUT}")


if __name__ == "__main__":
    main()
