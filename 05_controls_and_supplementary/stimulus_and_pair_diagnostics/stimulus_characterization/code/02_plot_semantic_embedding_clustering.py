#!/usr/bin/env python3
"""CLIP-embedding clustering across baseline, shared-train, and CSTIM pools.

This is a semantic-ish stimulus-composition diagnostic. It uses the same
high-level CLIP-L2B feature layer available in the existing paper caches:
controversial/baseline features from the CSTIM cache, shared DeepVision
features from the full DeepVision feature cache.
"""

from __future__ import annotations

import sys
import argparse
import pickle
import zipfile
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageOps

PAPER = Path(__file__).resolve().parents[1]
PROJECT = PAPER.parents[1]
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PAPER / "figures"))

import config  # noqa: E402
from style_improved import (  # noqa: E402
    DPI,
    FONT,
    MODEL_SET_COLORS,
    OKABE_ITO,
    W_DOUBLE,
    add_panel_label,
    apply_style,
)


MODEL = "timm_vit_large_patch14_clip_224_laion2b"
SHARED_CACHE_CANDIDATES = [
    Path(
        "/SSD/jroth/deepvision_fmri/feature_cache/full/"
        "timm_vit_large_patch14_clip_224_laion2b_blocks_18_attn_qkv_deepjuice_cls.pt"
    ),
    PROJECT
    / "scripts"
    / "claude"
    / "shared"
    / "cache"
    / "features"
    / "timm_vit_large_patch14_clip_224_laion2b_blocks.18.attn.qkv_deepjuice_cls.pt",
    PROJECT
    / "experiments"
    / "archive"
    / "cstim_fmri_analysis"
    / "shared"
    / "cache"
    / "features"
    / "timm_vit_large_patch14_clip_224_laion2b_blocks.18.attn.qkv_deepjuice_cls.pt",
]
CSTIM_CACHE = config.FEATURE_CACHE_DIR / "cstim" / f"{MODEL}.npz"

OUT = PAPER / "14_stimulus_characterization" / "data"
FIG = PAPER / "14_stimulus_characterization" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

MODEL_SETS = ["all_models", "sota", "training_objective", "architecture", "dataset"]
MODEL_SET_MARKERS = {
    "all_models": "o",
    "sota": "s",
    "training_objective": "^",
    "architecture": "D",
    "dataset": "P",
}
POOL_ORDER = ["controversial", "baseline", "shared train"]
POOL_LABELS = {
    "controversial": "CSTIMS",
    "baseline": "baseline",
    "shared train": "shared",
}
POOL_COLORS = {
    "controversial": OKABE_ITO["black"],
    "baseline": "#555555",
    "shared train": OKABE_ITO["blue"],
}
POOL_MARKERS = {
    "controversial": "o",
    "baseline": "s",
    "shared train": "^",
}
PANELS = [
    ("baseline", "same_session_vicco", "Baseline", "#555555", "s"),
    ("controversial", "all_models", "All Models", MODEL_SET_COLORS["all_models"], MODEL_SET_MARKERS["all_models"]),
    ("controversial", "sota", "SOTA", MODEL_SET_COLORS["sota"], MODEL_SET_MARKERS["sota"]),
    ("controversial", "training_objective", "Train. Obj.", MODEL_SET_COLORS["training_objective"], MODEL_SET_MARKERS["training_objective"]),
    ("controversial", "architecture", "Architecture", MODEL_SET_COLORS["architecture"], MODEL_SET_MARKERS["architecture"]),
    ("controversial", "dataset", "Dataset", MODEL_SET_COLORS["dataset"], MODEL_SET_MARKERS["dataset"]),
]
THUMBNAILS_PER_PANEL = 4
INSET_CANDIDATES = [
    (0.10, 0.88), (0.25, 0.88), (0.40, 0.88), (0.60, 0.88), (0.75, 0.88), (0.90, 0.88),
    (0.10, 0.12), (0.25, 0.12), (0.40, 0.12), (0.60, 0.12), (0.75, 0.12), (0.90, 0.12),
    (0.08, 0.30), (0.08, 0.50), (0.08, 0.70), (0.92, 0.30), (0.92, 0.50), (0.92, 0.70),
]
SELECTION_EVAL_IMAGE_ROOT = PROJECT / "experiments" / "cstim_paper" / "00_selection_evaluation" / "data"
WEBAPP_STIMULUS_ROOT = PROJECT / "writing" / "cstims_webapp" / "assets" / "stimuli"


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def cstim_image_dir(source_set: str) -> Path:
    return first_existing(
        [
            config.CSTIM_HDF5_ROOT / source_set,
            SELECTION_EVAL_IMAGE_ROOT / source_set / "images",
            WEBAPP_STIMULUS_ROOT / source_set,
        ]
    )


def baseline_image_dir() -> Path:
    return first_existing(
        [
            config.CSTIM_HDF5_ROOT / "shared_vicco",
            WEBAPP_STIMULUS_ROOT / "baseline",
        ]
    )


def shared_cache_path() -> Path:
    return first_existing(SHARED_CACHE_CANDIDATES)


def load_shared_cache(path: Path) -> dict:
    try:
        import torch

        return torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError:
        with zipfile.ZipFile(path) as zf:
            data_name = next(name for name in zf.namelist() if name.endswith("data.pkl"))
            return pickle.loads(zf.read(data_name))


def sorted_image_names(folder: Path, suffixes: tuple[str, ...] = (".jpg", ".png")) -> list[str]:
    names = [p.name for p in folder.iterdir() if p.suffix.lower() in suffixes]
    return sorted(names)


def add_block(
    rows: list[dict],
    features: list[np.ndarray],
    x: np.ndarray,
    *,
    pool: str,
    source_set: str,
    image_names: list[str],
    subject: str = "",
    max_rows: int | None = None,
    rng: np.random.Generator,
) -> None:
    if max_rows is not None and len(image_names) > max_rows:
        idx = np.sort(rng.choice(len(image_names), size=max_rows, replace=False))
        x = x[idx]
        image_names = [image_names[i] for i in idx]
    for i, image in enumerate(image_names):
        rows.append(
            {
                "pool": pool,
                "source_set": source_set,
                "subject": subject,
                "image": image,
                "row_within_source": i,
            }
        )
    features.append(np.asarray(x, dtype=np.float32))


def load_cstim_and_baseline(rng: np.random.Generator) -> tuple[list[dict], list[np.ndarray]]:
    rows: list[dict] = []
    blocks: list[np.ndarray] = []
    z = np.load(CSTIM_CACHE)
    for model_set in MODEL_SETS:
        names = sorted_image_names(cstim_image_dir(model_set))
        add_block(
            rows,
            blocks,
            z[model_set],
            pool="controversial",
            source_set=model_set,
            image_names=names,
            rng=rng,
        )
    add_block(
        rows,
        blocks,
        z["vicco"],
        pool="baseline",
        source_set="same_session_vicco",
        image_names=sorted_image_names(baseline_image_dir()),
        rng=rng,
    )
    return rows, blocks


def load_shared(rng: np.random.Generator) -> tuple[list[dict], list[np.ndarray]]:
    data = load_shared_cache(shared_cache_path())
    rows: list[dict] = []
    blocks: list[np.ndarray] = []
    add_block(
        rows,
        blocks,
        np.asarray(data["features"], dtype=np.float32),
        pool="shared train",
        source_set="deepvision_shared",
        image_names=list(data["image_names"]),
        rng=rng,
    )
    return rows, blocks


def build_embedding() -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(20260503)
    all_rows: list[dict] = []
    blocks: list[np.ndarray] = []
    for loader in (load_cstim_and_baseline, load_shared):
        rows, feats = loader(rng)
        all_rows.extend(rows)
        blocks.extend(feats)

    meta = pd.DataFrame(all_rows)
    x = np.vstack(blocks)
    if len(meta) != x.shape[0]:
        raise RuntimeError(f"metadata/features length mismatch: {len(meta)} vs {x.shape[0]}")

    # Standardize before PCA so high-variance feature coordinates do not dominate
    # the downstream t-SNE purely by scale.
    xz = StandardScaler().fit_transform(x)
    n_pcs = min(50, xz.shape[0] - 1, xz.shape[1])
    xp = PCA(n_components=n_pcs, random_state=42).fit_transform(xz)
    clusters = KMeans(n_clusters=10, random_state=42, n_init=25).fit_predict(xp)
    tsne = TSNE(
        n_components=2,
        perplexity=35,
        init="pca",
        learning_rate="auto",
        max_iter=1200,
        random_state=42,
    ).fit_transform(xp)
    meta["cluster"] = clusters
    meta["tsne_1"] = tsne[:, 0]
    meta["tsne_2"] = tsne[:, 1]

    # Sort clusters by controversial-stimulus fraction for easier reading.
    comp = (
        meta.groupby(["cluster", "pool"])
        .size()
        .rename("n")
        .reset_index()
        .pivot(index="cluster", columns="pool", values="n")
        .fillna(0)
    )
    comp_fraction = comp.div(comp.sum(axis=1), axis=0)
    ordered_clusters = comp_fraction.sort_values("controversial", ascending=False).index.tolist()
    cluster_map = {old: new + 1 for new, old in enumerate(ordered_clusters)}
    meta["cluster_ordered"] = meta["cluster"].map(cluster_map)

    meta.to_csv(OUT / "semantic_embedding_pool_sample.csv", index=False)
    summary = (
        meta.groupby(["cluster_ordered", "pool"])
        .size()
        .rename("n_images")
        .reset_index()
    )
    totals = summary.groupby("cluster_ordered")["n_images"].sum().rename("cluster_n")
    summary = summary.merge(totals, on="cluster_ordered")
    summary["fraction"] = summary["n_images"] / summary["cluster_n"]
    pool_totals = meta.groupby("pool").size().rename("pool_n").reset_index()
    summary = summary.merge(pool_totals, on="pool")
    summary.to_csv(OUT / "semantic_embedding_cluster_summary.csv", index=False)

    cstim_summary = (
        meta[meta["pool"] == "controversial"]
        .groupby(["cluster_ordered", "source_set"])
        .size()
        .rename("n_images")
        .reset_index()
    )
    cstim_totals = cstim_summary.groupby("cluster_ordered")["n_images"].sum().rename("cluster_cstim_n")
    cstim_summary = cstim_summary.merge(cstim_totals, on="cluster_ordered")
    cstim_summary["fraction"] = cstim_summary["n_images"] / cstim_summary["cluster_cstim_n"]
    cstim_summary.to_csv(OUT / "semantic_embedding_cluster_model_set_summary.csv", index=False)
    return meta


def image_path(row: pd.Series) -> Path:
    if row["pool"] == "baseline":
        return baseline_image_dir() / row["image"]
    if row["pool"] == "controversial":
        return cstim_image_dir(str(row["source_set"])) / row["image"]
    if row["pool"] == "shared train":
        return PROJECT / "data" / "cache" / "image_sets" / "deepvision_shared" / row["image"]
    raise ValueError(f"No local image path configured for pool={row['pool']!r}")


def load_thumbnail(path: Path, size: int = 96) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = ImageOps.fit(img, (size, size), method=Image.Resampling.LANCZOS)
        return np.asarray(img)


def representative_rows(sub: pd.DataFrame, n: int = THUMBNAILS_PER_PANEL) -> pd.DataFrame:
    from sklearn.cluster import KMeans

    coords = sub[["tsne_1", "tsne_2"]].to_numpy(dtype=float)
    n = min(n, len(sub))
    labels = KMeans(n_clusters=n, random_state=20260503, n_init=25).fit_predict(coords)
    centers = np.vstack([coords[labels == k].mean(axis=0) for k in range(n)])
    idx = []
    for k, center in enumerate(centers):
        members = np.flatnonzero(labels == k)
        nearest = members[np.argmin(np.linalg.norm(coords[members] - center, axis=1))]
        idx.append(nearest)
    reps = sub.iloc[idx].copy()
    reps["thumbnail_cluster"] = np.arange(1, len(reps) + 1)
    return reps.sort_values(["tsne_1", "tsne_2"]).reset_index(drop=True)


def to_axes_fraction(points: np.ndarray, xlim: tuple[float, float], ylim: tuple[float, float]) -> np.ndarray:
    xrange = xlim[1] - xlim[0]
    yrange = ylim[1] - ylim[0]
    out = np.empty_like(points, dtype=float)
    out[:, 0] = (points[:, 0] - xlim[0]) / xrange
    out[:, 1] = (points[:, 1] - ylim[0]) / yrange
    return out


def to_data_coords(points: np.ndarray, xlim: tuple[float, float], ylim: tuple[float, float]) -> np.ndarray:
    xrange = xlim[1] - xlim[0]
    yrange = ylim[1] - ylim[0]
    out = np.empty_like(points, dtype=float)
    out[:, 0] = xlim[0] + points[:, 0] * xrange
    out[:, 1] = ylim[0] + points[:, 1] * yrange
    return out


def point_segment_distances(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0:
        return np.linalg.norm(points - a, axis=1)
    t = np.clip(((points - a) @ ab) / denom, 0.0, 1.0)
    projection = a + t[:, None] * ab
    return np.linalg.norm(points - projection, axis=1)


def segments_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def ccw(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])

    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def choose_inset_positions(
    reps: pd.DataFrame,
    sub: pd.DataFrame,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> list[tuple[float, float]]:
    points = to_axes_fraction(sub[["tsne_1", "tsne_2"]].to_numpy(dtype=float), xlim, ylim)
    rep_points = to_axes_fraction(reps[["tsne_1", "tsne_2"]].to_numpy(dtype=float), xlim, ylim)
    candidates = np.asarray(INSET_CANDIDATES, dtype=float)
    per_rep: list[list[tuple[float, int]]] = []
    center = np.array([0.5, 0.5])

    for rep in rep_points:
        scored = []
        for j, cand in enumerate(candidates):
            near_thumb = np.sum(
                (np.abs(points[:, 0] - cand[0]) < 0.075)
                & (np.abs(points[:, 1] - cand[1]) < 0.090)
            )
            line_density = np.sum(point_segment_distances(points, rep, cand) < 0.018)
            line_length = np.linalg.norm(cand - rep)
            inward_penalty = max(0.0, -float(np.dot(rep - center, cand - rep)))
            edge_bonus = min(cand[0], 1 - cand[0], cand[1], 1 - cand[1])
            score = (
                4.0 * near_thumb
                + 5.0 * line_density
                + 3.0 * line_length
                + 110.0 * inward_penalty
                + 1.0 * edge_bonus
            )
            scored.append((score, j))
        per_rep.append(sorted(scored)[:12])

    best_score = np.inf
    best_indices: tuple[int, ...] | None = None
    for assignment in product(*[[j for _, j in scored] for scored in per_rep]):
        if len(set(assignment)) != len(assignment):
            continue
        base_score = sum(
            next(score for score, j in scored if j == assignment[i])
            for i, scored in enumerate(per_rep)
        )
        pair_score = 0.0
        for i in range(len(assignment)):
            for j in range(i + 1, len(assignment)):
                a0, a1 = rep_points[i], candidates[assignment[i]]
                b0, b1 = rep_points[j], candidates[assignment[j]]
                if segments_cross(a0, a1, b0, b1):
                    pair_score += 500.0
                min_sep = np.linalg.norm(candidates[assignment[i]] - candidates[assignment[j]])
                if min_sep < 0.23:
                    pair_score += 100.0 * (0.23 - min_sep)
                connector_gap = np.min(
                    [
                        point_segment_distances(np.asarray([a0]), b0, b1)[0],
                        point_segment_distances(np.asarray([a1]), b0, b1)[0],
                        point_segment_distances(np.asarray([b0]), a0, a1)[0],
                        point_segment_distances(np.asarray([b1]), a0, a1)[0],
                    ]
                )
                if connector_gap < 0.020:
                    pair_score += 60.0 * (0.020 - connector_gap)
        total = base_score + pair_score
        if total < best_score:
            best_score = total
            best_indices = assignment

    if best_indices is None:
        best_indices = tuple(scored[0][1] for scored in per_rep)
    return [tuple(candidates[j]) for j in best_indices]


def add_thumbnail_insets(
    ax,
    reps: pd.DataFrame,
    sub: pd.DataFrame,
    color: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> list[dict]:
    rows = []
    positions = choose_inset_positions(reps, sub, xlim, ylim)
    for (_, row), position in zip(reps.iterrows(), positions):
        path = image_path(row)
        if not path.exists():
            continue
        thumb = load_thumbnail(path)
        image = OffsetImage(thumb, zoom=0.26)
        line_end = to_data_coords(np.asarray([position], dtype=float), xlim, ylim)[0]
        ax.plot(
            [row["tsne_1"], line_end[0]],
            [row["tsne_2"], line_end[1]],
            color=color,
            linewidth=0.55,
            alpha=0.65,
            solid_capstyle="round",
            zorder=2,
        )
        artist = AnnotationBbox(
            image,
            (row["tsne_1"], row["tsne_2"]),
            xybox=position,
            xycoords="data",
            boxcoords=ax.transAxes,
            frameon=True,
            bboxprops=dict(edgecolor=color, linewidth=0.9, facecolor="white", boxstyle="square,pad=0.02"),
            annotation_clip=False,
            zorder=6,
        )
        ax.add_artist(artist)
        rows.append(
            {
                "panel_pool": row["pool"],
                "panel_source_set": row["source_set"],
                "image": row["image"],
                "image_path": str(path),
                "tsne_1": row["tsne_1"],
                "tsne_2": row["tsne_2"],
                "inset_x_frac": position[0],
                "inset_y_frac": position[1],
            }
        )
    return rows


def plot_background(ax, meta: pd.DataFrame) -> None:
    shared = meta[meta["pool"] == "shared train"]
    ax.scatter(
        shared["tsne_1"],
        shared["tsne_2"],
        s=5,
        c="#B9DCEF",
        marker="^",
        alpha=0.34,
        linewidths=0,
        zorder=1,
    )


def plot(meta: pd.DataFrame) -> None:
    apply_style()
    plot_meta = meta[meta["pool"].isin({"shared train", "baseline", "controversial"})].copy()
    fig, axes = plt.subplots(2, 3, figsize=(W_DOUBLE, 6.3), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    xlim = (plot_meta["tsne_1"].min() - 4, plot_meta["tsne_1"].max() + 4)
    ylim = (plot_meta["tsne_2"].min() - 4, plot_meta["tsne_2"].max() + 4)
    inset_rows: list[dict] = []

    for ax, (pool, source_set, title, color, marker), panel in zip(axes_flat, PANELS, list("abcdef")):
        plot_background(ax, plot_meta)
        sub = plot_meta[plot_meta["pool"] == pool]
        if source_set is not None:
            sub = sub[sub["source_set"] == source_set]
        ax.scatter(
            sub["tsne_1"],
            sub["tsne_2"],
            s=18 if pool == "controversial" else 13,
            c=color,
            marker=marker,
            alpha=0.88,
            linewidths=0,
            label=title,
            zorder=3,
        )
        reps = representative_rows(sub)
        inset_rows.extend(add_thumbnail_insets(ax, reps, sub, color, xlim, ylim))
        ax.set_title(f"{title} (N={len(sub)})", fontsize=FONT["title"], pad=5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("t-SNE 1" if panel in {"d", "e", "f"} else "", fontsize=FONT["axis_label"])
        ax.set_ylabel("t-SNE 2" if panel in {"a", "d"} else "", fontsize=FONT["axis_label"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        add_panel_label(ax, panel, x=-0.03, y=1.03)

    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.085, top=0.92, wspace=0.12, hspace=0.28)
    pd.DataFrame(inset_rows).to_csv(OUT / "semantic_embedding_thumbnail_representatives.csv", index=False)
    for ext in ("pdf", "png"):
        out = FIG / f"semantic_embedding_pool_clustering.{ext}"
        fig.savefig(out, dpi=DPI)
        print(f"Saved {out}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Plot from data/semantic_embedding_pool_sample.csv instead of rebuilding features and t-SNE.",
    )
    args = parser.parse_args()

    cache_path = OUT / "semantic_embedding_pool_sample.csv"
    if args.from_cache:
        meta = pd.read_csv(cache_path)
    else:
        meta = build_embedding()
    plot(meta)
    counts = meta.groupby("pool").size().reindex(POOL_ORDER)
    print(counts.to_string())


if __name__ == "__main__":
    main()
