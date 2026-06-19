#!/usr/bin/env python3
"""Create stimulus overview grids for the results summary.

Figure 1: Side-by-side grid of controversial (all_models) vs Vicco baseline stimuli.
Figure 2: High-spread image pairs for Section 6 — pairs where models disagree most,
          with compact indicators of which models find them similar vs dissimilar.
"""

import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
_SHARE_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))  # project root
sys.path.insert(0, str(_SHARE_ROOT / "src"))
from cstims import constants, paths
from cstims.cache import load_cstim_brain_cache

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

CSTIM_DIR = paths.cstim_hdf5_root()
FIG_DIR = Path(__file__).resolve().parent
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_image(path, size=128):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img)


def create_stimulus_overview():
    """Side-by-side: 10x10 controversial vs 10x10 baseline (all 100 each)."""
    n_show = 100
    cols, rows = 10, 10
    thumb = 80  # smaller thumbnails to fit 100 per side

    # Load controversial (all_models)
    cstim_dir = CSTIM_DIR / "all_models"
    cstim_files = sorted(cstim_dir.glob("image_*.png"))[:n_show]
    cstims = [load_image(f, size=thumb) for f in cstim_files]

    # Load Vicco baseline
    vicco_dir = CSTIM_DIR / "shared_vicco"
    vicco_files = sorted(vicco_dir.glob("*.jpg"))[:n_show]
    viccos = [load_image(f, size=thumb) for f in vicco_files]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8.5))

    for ax, images, title in [
        (axes[0], cstims, f"Controversial stimuli (all-models set, N={len(cstims)})"),
        (axes[1], viccos, f"Baseline stimuli (N={len(viccos)})"),
    ]:
        n = len(images)
        actual_rows = min(rows, (n + cols - 1) // cols)
        gap = 2
        grid = np.ones((actual_rows * thumb + (actual_rows - 1) * gap,
                        cols * thumb + (cols - 1) * gap, 3), dtype=np.uint8) * 240
        for idx, img in enumerate(images):
            r, c = divmod(idx, cols)
            if r >= actual_rows:
                break
            y = r * (thumb + gap)
            x = c * (thumb + gap)
            grid[y:y+thumb, x:x+thumb] = img

        ax.imshow(grid)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout(pad=1.0)
    for ext in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"stimulus_overview.{ext}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved stimulus_overview")


def _load_model_zscores():
    """Load features and compute per-model z-scored distance matrices."""
    import pickle
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import zscore as sp_zscore

    features_path = paths.selected_stimuli_payload()

    with open(features_path, "rb") as f:
        sel_data = pickle.load(f)

    if "best_raw_combined_features_raw" in sel_data:
        features_dict = sel_data["best_raw_combined_features_raw"]
    elif "selected_features_raw" in sel_data:
        features_dict = sel_data["selected_features_raw"]
    else:
        features_dict = sel_data
    model_names = list(features_dict.keys())

    z_dists = {}
    for model_name in model_names:
        feats = features_dict[model_name]
        if hasattr(feats, 'numpy'):
            feats = feats.numpy()
        dmat = squareform(pdist(feats, metric="cosine"))
        triu_idx = np.triu_indices(dmat.shape[0], k=1)
        triu_vals = dmat[triu_idx]
        z_vals = sp_zscore(triu_vals)
        z_mat = np.zeros_like(dmat)
        z_mat[triu_idx] = z_vals
        z_mat = z_mat + z_mat.T
        z_dists[model_name] = z_mat

    return model_names, z_dists


def _get_abbrev(model_name):
    """Short model abbreviation for display."""
    MODEL_ABBREVS = {
        "slip_vit_l_slip": "SLIP",
        "slip_vit_l_simclr": "SimCLR-ViT",
        "timm_vit_large_patch14_clip_224_laion2b": "CLIP-LAION2B",
        "dinov2_vitl14": "DINOv2",
        "openclip_vit_so400m_14_siglip_webli": "SigLIP",
        "torchvision_convnext_base_imagenet1k_v1": "ConvNeXt-B",
        "vissl_resnet50_supervised": "Supervised",
        "vissl_resnet50_barlowtwins": "BarlowTwins",
        "vissl_resnet50_mocov2": "MoCo-v2",
        "vicreg_resnet50": "VICReg",
        "robustness_imagenet_l2_eps3": "Robust-L2",
        "torchvision_vgg16_imagenet1k_v1": "VGG-16",
        "torchvision_resnet50_imagenet1k_v1": "ResNet-50",
        "torchvision_vit_l_16_imagenet1k_v1": "ViT-L/16",
        "cornet_s": "CORnet-S",
        "openclip_vit_l_14_quickgelu_metaclip_400m": "MetaCLIP-400M",
        "openclip_vit_l_14_quickgelu_metaclip_fullcc": "MetaCLIP-Full",
        "timm_vit_large_patch14_clip_224_dfn2b": "CLIP-DFN2B",
        "timm_vit_large_patch14_clip_quickgelu_224_openai": "CLIP-OpenAI",
        "openclip_vit_l_14_laion400m_e31": "CLIP-LAION400M",
        "torchvision_alexnet_imagenet1k_v1": "AlexNet",
    }
    return MODEL_ABBREVS.get(model_name, model_name.split("_")[-1])


def _load_brain_zscores():
    """Load brain RDMs for all_models stimuli and return z-scored distance matrices per subject.

    Returns dict: subject -> z-scored distance matrix (n_stim x n_stim).
    """
    from scipy.stats import zscore as sp_zscore

    from utils import compute_rdm_correlation

    brain_z_by_subject = {}
    for subject in constants.SUBJECTS:
        cache = load_cstim_brain_cache(subject, missing_ok=True)
        if cache is None:
            continue
        brain_patterns = cache.patterns("all_models", sort_by_stim_idx=True)
        brain_rdm = compute_rdm_correlation(brain_patterns)

        # Z-score the upper triangle, same as model z-scores
        triu_idx = np.triu_indices(brain_rdm.shape[0], k=1)
        triu_vals = brain_rdm[triu_idx]
        z_vals = sp_zscore(triu_vals)
        z_mat = np.zeros_like(brain_rdm)
        z_mat[triu_idx] = z_vals
        z_mat = z_mat + z_mat.T

        # Build lookup: selection_index (0-99) -> brain_order_index
        # file_idx[brain_order_k] = selection_index for stimulus k
        # We need: sel_to_brain[selection_idx] = brain_order_idx
        sel_to_brain = {int(fi): k for k, fi in enumerate(file_idx)}

        brain_z_by_subject[subject] = {"z_mat": z_mat, "sel_to_brain": sel_to_brain}

    return brain_z_by_subject


def create_high_spread_pairs():
    """Image-dominant 4x4 grid of high-spread pairs with heatmap + model names below.

    Brain markers (triangles) show per-subject brain z-scored distance on the same scale.
    """
    from matplotlib.colors import TwoSlopeNorm

    data_dir = paths.project_root() / "experiments" / "archive" / "cstim_image_analysis" / "model_pair_disagreement"
    spread_df = pd.read_csv(data_dir / "results" / "all_models" / "per_pair_spread.csv")
    spread_df = spread_df.sort_values("spread", ascending=False)

    model_names, z_dists = _load_model_zscores()
    brain_z_by_subject = _load_brain_zscores()

    n_pairs = 16
    n_cols, n_rows = 4, 4
    img_dir = CSTIM_DIR / "all_models"

    # Select top pairs ensuring no image appears twice
    seen_images = set()
    selected_rows = []
    for _, pair_row in spread_df.iterrows():
        i, j = int(pair_row["img_i"]), int(pair_row["img_j"])
        if i not in seen_images and j not in seen_images:
            selected_rows.append(pair_row)
            seen_images.add(i)
            seen_images.add(j)
        if len(selected_rows) == n_pairs:
            break
    top_pairs = pd.DataFrame(selected_rows)

    cmap = plt.cm.RdBu_r
    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
    BLUE = "#2166ac"
    RED = "#b2182b"

    n_extreme = 3
    img_sz = 120
    gap = 4  # pixels between images in a pair
    pair_w = 2 * img_sz + gap
    strip_h = 10  # pixels for heatmap strip
    text_h = 34   # pixels for model name text
    cell_h = img_sz + strip_h + text_h
    pad_x, pad_y = 20, 14  # padding between cells

    total_w = n_cols * pair_w + (n_cols - 1) * pad_x
    total_h = n_rows * cell_h + (n_rows - 1) * pad_y

    # Build the figure as a pixel canvas rendered via imshow + text overlays
    fig_w = 14
    fig_h = fig_w * (total_h + 60) / total_w  # +60 for title + colorbar
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255

    label_data = []  # collect (x, y, sim_names, dis_names) for text overlay
    brain_marker_data = []  # collect (strip_x0, strip_w, strip_y, brain_z_vals) for markers

    for idx, (_, pair_row) in enumerate(top_pairs.iterrows()):
        r, c = divmod(idx, n_cols)
        i, j = int(pair_row["img_i"]), int(pair_row["img_j"])

        x0 = c * (pair_w + pad_x)
        y0 = r * (cell_h + pad_y)

        # Draw image pair
        img_i = load_image(img_dir / f"image_{i:04d}.png", size=img_sz)
        img_j = load_image(img_dir / f"image_{j:04d}.png", size=img_sz)
        canvas[y0:y0 + img_sz, x0:x0 + img_sz] = img_i
        canvas[y0:y0 + img_sz, x0 + img_sz + gap:x0 + pair_w] = img_j

        # Compute sorted z-scores and render heatmap strip
        model_zvals = [(m, z_dists[m][i, j]) for m in model_names]
        model_zvals.sort(key=lambda x: x[1])
        vals = np.array([z for _, z in model_zvals])
        n_m = len(vals)

        strip_y = y0 + img_sz + 2
        # Render each model as a colored block in the strip
        block_w = pair_w / n_m
        for mi, v in enumerate(vals):
            rgba = cmap(norm(v))
            color_rgb = (np.array(rgba[:3]) * 255).astype(np.uint8)
            bx0 = int(x0 + mi * block_w)
            bx1 = int(x0 + (mi + 1) * block_w)
            canvas[strip_y:strip_y + strip_h, bx0:bx1] = color_rgb

        # Collect brain z-scores for this pair (per subject)
        brain_z_vals = {}
        val_min, val_max = vals[0], vals[-1]
        for subj, bdata in brain_z_by_subject.items():
            sel_to_brain = bdata["sel_to_brain"]
            if i in sel_to_brain and j in sel_to_brain:
                bi, bj = sel_to_brain[i], sel_to_brain[j]
                brain_z_vals[subj] = bdata["z_mat"][bi, bj]
        brain_marker_data.append((x0, pair_w, strip_y, strip_h, val_min, val_max, brain_z_vals))

        # Collect model names for text overlay
        sim_names = [_get_abbrev(m) for m, _ in model_zvals[:n_extreme]]
        dis_names = [_get_abbrev(m) for m, _ in model_zvals[-n_extreme:]]
        # Store pixel positions for text (will convert to axes coords)
        text_y = strip_y + strip_h + 2
        label_data.append((x0, x0 + pair_w, text_y, sim_names, dis_names))

    ax.imshow(canvas)
    ax.axis("off")

    # Add brain markers on each heatmap strip (centered vertically on the strip)
    subject_markers = {"sub-01": "v", "sub-03": "^", "sub-05": "s", "sub-06": "D", "sub-07": "o"}
    for strip_x0, strip_w, strip_y_px, strip_h_px, val_min, val_max, brain_z_vals in brain_marker_data:
        val_range = val_max - val_min
        if val_range < 1e-6:
            continue
        for subj, bz in brain_z_vals.items():
            frac = (bz - val_min) / val_range
            frac = np.clip(frac, 0, 1)
            marker_x = strip_x0 + frac * strip_w
            marker_y = strip_y_px + strip_h_px / 2  # centered on strip
            marker = subject_markers.get(subj, "v")
            ax.plot(marker_x, marker_y, marker=marker, color="white",
                    markersize=5.5, markeredgewidth=1.0, markeredgecolor="black",
                    clip_on=True, zorder=10)

    # Add model name text: line 1 = similar (blue, left), line 2 = dissimilar (red, right)
    for x_left, x_right, ty, sim_names, dis_names in label_data:
        sim_str = "← " + ", ".join(sim_names)
        dis_str = ", ".join(dis_names) + " →"
        ax.text(x_left + 2, ty + 2, sim_str,
                fontsize=7, color=BLUE, fontweight="bold",
                ha="left", va="top", clip_on=True)
        ax.text(x_right - 2, ty + 16, dis_str,
                fontsize=7, color=RED, fontweight="bold",
                ha="right", va="top", clip_on=True)

    # Brain marker legend (small, bottom-left)
    legend_handles = []
    for subj, marker in subject_markers.items():
        h = ax.plot([], [], marker=marker, color="white", markersize=5.5,
                    markeredgewidth=1.0, markeredgecolor="black", linestyle="none",
                    label=subj)[0]
        legend_handles.append(h)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower left", fontsize=7,
                  title="Brain", title_fontsize=7, framealpha=0.8,
                  handletextpad=0.3, borderpad=0.3,
                  bbox_to_anchor=(0.0, -0.03))

    # Title
    ax.set_title("High-spread image pairs: models disagree most about these similarities",
                 fontsize=12, fontweight="bold", pad=10)

    # Colorbar
    cbar_ax = fig.add_axes([0.30, 0.02, 0.40, 0.012])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("z-scored model RDM distance for the pair "
                   "(colour = position on a similar→dissimilar sort, "
                   "not group membership)",
                   fontsize=10, labelpad=3)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)
    for ext in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"high_spread_pairs.{ext}", dpi=250, bbox_inches="tight")
    plt.close()
    print("Saved high_spread_pairs")


def compute_all_pair_brain_stats(brain_z_by_subject, spread_df):
    """For every pair in spread_df, collect per-subject z-scores and compute mean/SD.

    Returns spread_df with extra columns: mean_z, sd_z, n_subjects, plus one
    column per subject (brain_z_{subject}).
    """
    rows = []
    for _, row in spread_df.iterrows():
        i, j = int(row["img_i"]), int(row["img_j"])
        subj_vals = {}
        for subj, bdata in brain_z_by_subject.items():
            s2b = bdata["sel_to_brain"]
            if i in s2b and j in s2b:
                subj_vals[subj] = bdata["z_mat"][s2b[i], s2b[j]]
        vals = list(subj_vals.values())
        entry = {"img_i": i, "img_j": j, "spread": row["spread"]}
        entry["n_subjects"] = len(vals)
        entry["mean_z"] = float(np.mean(vals)) if vals else np.nan
        entry["sd_z"]   = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        for subj, v in subj_vals.items():
            entry[f"brain_z_{subj}"] = v
        rows.append(entry)
    return pd.DataFrame(rows)


def create_brain_consistency_plot(sd_threshold=1.0):
    """Scatter of (mean brain z-score, SD across subjects) for ALL image pairs.

    Shows how consistent subjects are across the full pair space.
    Pairs that pass the consistency filter (all 5 subjects, SD < threshold) are
    highlighted; the rest are shown in grey.
    """
    from cstims.paper.style_improved import apply_style, FONT, DPI, W_DOUBLE
    apply_style()

    data_dir = paths.project_root() / "experiments" / "archive" / "cstim_image_analysis" / "model_pair_disagreement"
    spread_df = pd.read_csv(data_dir / "results" / "all_models" / "per_pair_spread.csv")
    spread_df = spread_df.sort_values("spread", ascending=False)

    brain_z_by_subject = _load_brain_zscores()
    stats_df = compute_all_pair_brain_stats(brain_z_by_subject, spread_df)

    n_subjects_available = len(brain_z_by_subject)
    consistent = (stats_df["n_subjects"] == n_subjects_available) & (stats_df["sd_z"] < sd_threshold)
    print(f"Pairs with all {n_subjects_available} subjects: {(stats_df['n_subjects'] == n_subjects_available).sum()}")
    print(f"Consistent pairs (SD < {sd_threshold}): {consistent.sum()}")

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE * 0.75, 3.8),
                             gridspec_kw={"width_ratios": [1.6, 1]})
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.15, wspace=0.35)

    # ---- Panel A: scatter mean_z vs sd_z ----
    ax = axes[0]
    # grey: incomplete or inconsistent
    mask_grey = ~consistent
    ax.scatter(stats_df.loc[mask_grey, "mean_z"], stats_df.loc[mask_grey, "sd_z"],
               s=2, c="#aaaaaa", alpha=0.25, linewidths=0, rasterized=True,
               label=f"other ({mask_grey.sum()})")
    # highlighted: consistent high-spread
    ax.scatter(stats_df.loc[consistent, "mean_z"], stats_df.loc[consistent, "sd_z"],
               s=4, c="#2471A3", alpha=0.6, linewidths=0, rasterized=True,
               label=f"consistent (SD<{sd_threshold}, {consistent.sum()})")
    ax.axhline(sd_threshold, color="#E74C3C", linewidth=1.0, linestyle="--",
               label=f"SD = {sd_threshold}")
    ax.set_xlabel("Mean brain z-score across subjects", fontsize=FONT["axis_label"])
    ax.set_ylabel("SD across subjects", fontsize=FONT["axis_label"])
    ax.set_title("All image pairs", fontweight="bold", fontsize=FONT["title"])
    ax.legend(fontsize=FONT["small"], frameon=False, markerscale=2)
    ax.tick_params(labelsize=FONT["tick"])
    ax.text(-0.10, 1.08, "a", transform=ax.transAxes,
            fontsize=FONT["panel_label"], fontweight="bold", va="top")

    # ---- Panel B: histogram of SD values (all pairs with all subjects) ----
    ax2 = axes[1]
    complete = stats_df[stats_df["n_subjects"] == n_subjects_available]["sd_z"].dropna()
    ax2.hist(complete, bins=40, color="#2471A3", alpha=0.75, edgecolor="none")
    ax2.axvline(sd_threshold, color="#E74C3C", linewidth=1.2, linestyle="--",
                label=f"SD = {sd_threshold}")
    ax2.set_xlabel("SD across subjects", fontsize=FONT["axis_label"])
    ax2.set_ylabel("Number of pairs", fontsize=FONT["axis_label"])
    ax2.set_title("Distribution of SD", fontweight="bold", fontsize=FONT["title"])
    ax2.legend(fontsize=FONT["small"], frameon=False)
    ax2.tick_params(labelsize=FONT["tick"])
    ax2.text(-0.15, 1.08, "b", transform=ax2.transAxes,
            fontsize=FONT["panel_label"], fontweight="bold", va="top")

    for ext in ["pdf", "png"]:
        out = FIG_DIR / f"brain_consistency.{ext}"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


def create_high_spread_pairs_consistent(sd_threshold=1.0, min_abs_mean_z=0.75):
    """Like create_high_spread_pairs but restricted to pairs where all 5 subjects
    agree (SD of brain z-scores < sd_threshold) AND have a strong brain signal
    (|mean_z| > min_abs_mean_z, i.e. clearly similar or clearly dissimilar)."""
    from matplotlib.colors import TwoSlopeNorm

    data_dir = paths.project_root() / "experiments" / "archive" / "cstim_image_analysis" / "model_pair_disagreement"
    spread_df = pd.read_csv(data_dir / "results" / "all_models" / "per_pair_spread.csv")
    spread_df = spread_df.sort_values("spread", ascending=False)

    model_names, z_dists = _load_model_zscores()
    brain_z_by_subject = _load_brain_zscores()
    n_subjects_available = len(brain_z_by_subject)

    stats_df = compute_all_pair_brain_stats(brain_z_by_subject, spread_df)
    consistent_pairs = stats_df[
        (stats_df["n_subjects"] == n_subjects_available) &
        (stats_df["sd_z"] < sd_threshold) &
        (stats_df["mean_z"].abs() > min_abs_mean_z)
    ].copy()
    # still sorted by spread (highest first)
    consistent_pairs = consistent_pairs.sort_values("spread", ascending=False)
    print(f"Consistent pairs available: {len(consistent_pairs)} "
          f"(SD<{sd_threshold}, |mean_z|>{min_abs_mean_z})")

    # Reviewer feedback: prefer 6-8 highly legible pairs over 16 cramped ones.
    n_pairs = 8
    n_cols, n_rows = 4, 2
    img_dir = CSTIM_DIR / "all_models"

    seen_images = set()
    selected_rows = []
    for _, pair_row in consistent_pairs.iterrows():
        i, j = int(pair_row["img_i"]), int(pair_row["img_j"])
        if i not in seen_images and j not in seen_images:
            selected_rows.append(pair_row)
            seen_images.add(i)
            seen_images.add(j)
        if len(selected_rows) == n_pairs:
            break

    if len(selected_rows) < n_pairs:
        print(f"Warning: only {len(selected_rows)} consistent unique pairs found")
    top_pairs = pd.DataFrame(selected_rows)

    cmap = plt.cm.RdBu_r
    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
    BLUE = "#2166ac"
    RED = "#b2182b"

    n_extreme = 3
    img_sz = 200            # larger images for legibility
    gap = 6
    pair_w = 2 * img_sz + gap
    strip_h = 14            # taller distance strip
    text_h = 42
    cell_h = img_sz + strip_h + text_h
    pad_x, pad_y = 28, 18

    total_w = n_cols * pair_w + (n_cols - 1) * pad_x
    total_h = n_rows * cell_h + (n_rows - 1) * pad_y

    fig_w = 16
    fig_h = fig_w * (total_h + 80) / total_w
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255
    label_data = []
    brain_marker_data = []

    for idx, (_, pair_row) in enumerate(top_pairs.iterrows()):
        r, c = divmod(idx, n_cols)
        i, j = int(pair_row["img_i"]), int(pair_row["img_j"])

        x0 = c * (pair_w + pad_x)
        y0 = r * (cell_h + pad_y)

        img_i = load_image(img_dir / f"image_{i:04d}.png", size=img_sz)
        img_j = load_image(img_dir / f"image_{j:04d}.png", size=img_sz)
        canvas[y0:y0 + img_sz, x0:x0 + img_sz] = img_i
        canvas[y0:y0 + img_sz, x0 + img_sz + gap:x0 + pair_w] = img_j

        model_zvals = [(m, z_dists[m][i, j]) for m in model_names]
        model_zvals.sort(key=lambda x: x[1])
        vals = np.array([z for _, z in model_zvals])
        n_m = len(vals)

        strip_y = y0 + img_sz + 2
        block_w = pair_w / n_m
        for mi, v in enumerate(vals):
            rgba = cmap(norm(v))
            color_rgb = (np.array(rgba[:3]) * 255).astype(np.uint8)
            bx0 = int(x0 + mi * block_w)
            bx1 = int(x0 + (mi + 1) * block_w)
            canvas[strip_y:strip_y + strip_h, bx0:bx1] = color_rgb

        brain_z_vals = {}
        val_min, val_max = vals[0], vals[-1]
        for subj, bdata in brain_z_by_subject.items():
            s2b = bdata["sel_to_brain"]
            if i in s2b and j in s2b:
                brain_z_vals[subj] = bdata["z_mat"][s2b[i], s2b[j]]
        brain_marker_data.append((x0, pair_w, strip_y, strip_h, val_min, val_max, brain_z_vals))

        sim_names = [_get_abbrev(m) for m, _ in model_zvals[:n_extreme]]
        dis_names = [_get_abbrev(m) for m, _ in model_zvals[-n_extreme:]]
        text_y = strip_y + strip_h + 2
        label_data.append((x0, x0 + pair_w, text_y, sim_names, dis_names))

    ax.imshow(canvas)
    ax.axis("off")

    subject_markers = {"sub-01": "v", "sub-03": "^", "sub-05": "s", "sub-06": "D", "sub-07": "o"}
    for strip_x0, strip_w, strip_y_px, strip_h_px, val_min, val_max, brain_z_vals in brain_marker_data:
        val_range = val_max - val_min
        if val_range < 1e-6:
            continue
        for subj, bz in brain_z_vals.items():
            frac = np.clip((bz - val_min) / val_range, 0, 1)
            marker_x = strip_x0 + frac * strip_w
            marker_y = strip_y_px + strip_h_px / 2
            marker = subject_markers.get(subj, "v")
            ax.plot(marker_x, marker_y, marker=marker, color="white",
                    markersize=5.5, markeredgewidth=1.0, markeredgecolor="black",
                    clip_on=True, zorder=10)

    for x_left, x_right, ty, sim_names, dis_names in label_data:
        ax.text(x_left + 2, ty + 2, "← " + ", ".join(sim_names),
                fontsize=9, color=BLUE, fontweight="bold", ha="left", va="top", clip_on=True)
        ax.text(x_right - 2, ty + 22, ", ".join(dis_names) + " →",
                fontsize=9, color=RED, fontweight="bold", ha="right", va="top", clip_on=True)

    legend_handles = []
    for subj, marker in subject_markers.items():
        h = ax.plot([], [], marker=marker, color="white", markersize=5.5,
                    markeredgewidth=1.0, markeredgecolor="black", linestyle="none",
                    label=subj)[0]
        legend_handles.append(h)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower left", fontsize=7,
                  title="Brain", title_fontsize=7, framealpha=0.8,
                  handletextpad=0.3, borderpad=0.3,
                  bbox_to_anchor=(0.0, -0.03))

    ax.set_title(
        f"High-spread pairs (consistent, SD < {sd_threshold}, |mean brain z| > {min_abs_mean_z})",
        fontsize=12, fontweight="bold", pad=10,
    )

    cbar_ax = fig.add_axes([0.30, 0.02, 0.40, 0.012])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("z-scored model RDM distance for the pair "
                   "(colour = position on a similar→dissimilar sort, "
                   "not group membership)",
                   fontsize=10, labelpad=3)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)
    for ext in ["pdf", "png"]:
        out = FIG_DIR / f"high_spread_pairs_consistent.{ext}"
        fig.savefig(out, dpi=250, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--overview", action="store_true", help="Create stimulus overview grid")
    parser.add_argument("--high-spread", action="store_true", help="Create high-spread pairs figure")
    parser.add_argument("--consistency", action="store_true", help="Create brain consistency plot")
    parser.add_argument("--consistent-pairs", action="store_true", help="Create consistent high-spread pairs figure")
    parser.add_argument("--sd-threshold", type=float, default=1.0, help="SD threshold for consistency filter")
    parser.add_argument("--min-abs-mean-z", type=float, default=0.75, help="Minimum |mean brain z-score| to exclude near-zero pairs")
    parser.add_argument("--all", action="store_true", help="Create all figures")
    args = parser.parse_args()
    if args.all or not any([args.overview, args.high_spread, args.consistency, args.consistent_pairs]):
        create_stimulus_overview()
        create_high_spread_pairs()
        create_brain_consistency_plot(sd_threshold=args.sd_threshold)
        create_high_spread_pairs_consistent(sd_threshold=args.sd_threshold, min_abs_mean_z=args.min_abs_mean_z)
    else:
        if args.overview:
            create_stimulus_overview()
        if args.high_spread:
            create_high_spread_pairs()
        if args.consistency:
            create_brain_consistency_plot(sd_threshold=args.sd_threshold)
        if args.consistent_pairs:
            create_high_spread_pairs_consistent(sd_threshold=args.sd_threshold, min_abs_mean_z=args.min_abs_mean_z)
