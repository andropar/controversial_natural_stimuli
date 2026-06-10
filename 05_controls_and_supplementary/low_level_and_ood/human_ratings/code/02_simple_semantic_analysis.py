#!/data/home_roth/miniforge3/bin/python
"""Simple semantic/perceptual audit from the VLM annotation table.

This is intentionally descriptive. It does not fit models, embed captions, or
reuse the counterfactual-baseline code. It summarizes how each CSTIM definition
differs from the Vicco baseline on the structured caption-audit fields and on
simple caption-token prevalence.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGE = Path(__file__).resolve().parents[1]
ANNOTATIONS = STAGE / "outputs" / "annotations" / "full_minimax_m3_all_stimuli.csv"
OUT = STAGE / "outputs" / "semantic_analysis"

CONDITION_ORDER = ["vicco", "all_models", "architecture", "dataset", "sota", "training_objective"]
CONDITION_LABELS = {
    "vicco": "Baseline",
    "all_models": "All-model",
    "architecture": "Arch.",
    "dataset": "Data",
    "sota": "SOTA",
    "training_objective": "Train.",
}

NUMERIC_COLS = [
    "recognizability",
    "ambiguity",
    "natural_photo_typicality",
    "visual_clutter",
    "object_centricity",
    "scene_centricity",
    "caption_confidence",
    "estimated_salient_object_count",
    "estimated_distinct_object_categories",
]
NUMERIC_LABELS = {
    "recognizability": "recognizability",
    "ambiguity": "ambiguity",
    "natural_photo_typicality": "natural-photo typicality",
    "visual_clutter": "visual clutter",
    "object_centricity": "object-centricity",
    "scene_centricity": "scene-centricity",
    "caption_confidence": "caption confidence",
    "estimated_salient_object_count": "salient object count",
    "estimated_distinct_object_categories": "distinct object categories",
}

BOOLEAN_COLS = [
    "contains_person",
    "contains_face",
    "contains_animal",
    "contains_text",
    "contains_vehicle",
    "contains_food",
    "contains_indoor_scene",
    "contains_outdoor_scene",
    "contains_artificial_or_rendered_content",
    "contains_occlusion_or_truncation",
    "contains_unusual_viewpoint",
    "contains_multiple_main_objects",
]
BOOLEAN_LABELS = {
    "contains_person": "person",
    "contains_face": "face",
    "contains_animal": "animal",
    "contains_text": "text",
    "contains_vehicle": "vehicle",
    "contains_food": "food",
    "contains_indoor_scene": "indoor scene",
    "contains_outdoor_scene": "outdoor scene",
    "contains_artificial_or_rendered_content": "artificial/rendered",
    "contains_occlusion_or_truncation": "occlusion/truncation",
    "contains_unusual_viewpoint": "unusual viewpoint",
    "contains_multiple_main_objects": "multiple main objects",
}

CATEGORY_COLS = ["dominant_content_type", "image_style", "semantic_domain"]
TEXT_COLS = ["semantic_domain", "short_caption", "main_objects", "possible_interpretations"]

STOPWORDS = {
    "and", "the", "with", "from", "into", "onto", "this", "that", "these", "those",
    "image", "photo", "picture", "showing", "shows", "scene", "view", "visible",
    "background", "foreground", "possibly", "possible", "likely", "unclear",
    "object", "objects", "main", "several", "multiple", "person", "people",
    "standing", "sitting", "holding", "wearing", "near", "inside", "outside",
    "front", "behind", "large", "small", "white", "black", "blue", "red",
    "green", "brown", "gray", "grey", "yellow", "orange", "pink",
}


def cohen_d(a: pd.Series, b: pd.Series) -> float:
    """Cohen's d for condition a minus baseline b."""
    x = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)
    pooled = np.sqrt(((len(x) - 1) * sx**2 + (len(y) - 1) * sy**2) / (len(x) + len(y) - 2))
    if not np.isfinite(pooled) or pooled == 0:
        return float("nan")
    return float((x.mean() - y.mean()) / pooled)


def read_annotations() -> pd.DataFrame:
    if not ANNOTATIONS.exists():
        raise FileNotFoundError(f"Missing annotation table: {ANNOTATIONS}")
    df = pd.read_csv(ANNOTATIONS)
    df = df[df["condition"].isin(CONDITION_ORDER)].copy()
    if "validation_status" in df:
        df = df[df["validation_status"].eq("ok")].copy()
    df = df.drop_duplicates(["condition", "stim_idx"]).reset_index(drop=True)
    return df


def numeric_shifts(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df[df["condition"].eq("vicco")]
    rows = []
    for condition in CONDITION_ORDER[1:]:
        sub = df[df["condition"].eq(condition)]
        for col in NUMERIC_COLS:
            a = pd.to_numeric(sub[col], errors="coerce")
            b = pd.to_numeric(baseline[col], errors="coerce")
            rows.append(
                {
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "field": col,
                    "field_label": NUMERIC_LABELS[col],
                    "n_condition": int(a.notna().sum()),
                    "n_baseline": int(b.notna().sum()),
                    "condition_mean": float(a.mean()),
                    "baseline_mean": float(b.mean()),
                    "mean_difference": float(a.mean() - b.mean()),
                    "cohens_d": cohen_d(a, b),
                }
            )
    return pd.DataFrame(rows)


def to_bool_series(x: pd.Series) -> pd.Series:
    if x.dtype == bool:
        return x.astype(float)
    return x.astype(str).str.lower().map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})


def binary_shifts(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df[df["condition"].eq("vicco")]
    rows = []
    for condition in CONDITION_ORDER[1:]:
        sub = df[df["condition"].eq(condition)]
        for col in BOOLEAN_COLS:
            a = to_bool_series(sub[col])
            b = to_bool_series(baseline[col])
            rows.append(
                {
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "field": col,
                    "field_label": BOOLEAN_LABELS[col],
                    "n_condition": int(a.notna().sum()),
                    "n_baseline": int(b.notna().sum()),
                    "condition_prevalence": float(a.mean()),
                    "baseline_prevalence": float(b.mean()),
                    "prevalence_difference": float(a.mean() - b.mean()),
                    "cohens_d_binary": cohen_d(a, b),
                }
            )
    return pd.DataFrame(rows)


def category_shifts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline = df[df["condition"].eq("vicco")]
    for col in CATEGORY_COLS:
        baseline_counts = baseline[col].fillna("missing").astype(str).str.strip().str.lower().value_counts()
        baseline_n = int(baseline_counts.sum())
        for condition in CONDITION_ORDER[1:]:
            sub_counts = (
                df[df["condition"].eq(condition)][col]
                .fillna("missing")
                .astype(str)
                .str.strip()
                .str.lower()
                .value_counts()
            )
            sub_n = int(sub_counts.sum())
            for category in sorted(set(baseline_counts.index) | set(sub_counts.index)):
                condition_prev = float(sub_counts.get(category, 0) / sub_n) if sub_n else np.nan
                baseline_prev = float(baseline_counts.get(category, 0) / baseline_n) if baseline_n else np.nan
                rows.append(
                    {
                        "condition": condition,
                        "condition_label": CONDITION_LABELS[condition],
                        "field": col,
                        "category": category,
                        "condition_n": int(sub_counts.get(category, 0)),
                        "baseline_n": int(baseline_counts.get(category, 0)),
                        "condition_prevalence": condition_prev,
                        "baseline_prevalence": baseline_prev,
                        "prevalence_difference": condition_prev - baseline_prev,
                    }
                )
    return pd.DataFrame(rows)


def caption_text(row: pd.Series) -> str:
    return " ".join(str(row.get(col, "")) for col in TEXT_COLS)


def tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z][a-z0-9_'-]{2,}", text.lower())
    return {tok.strip("_'-") for tok in tokens if tok not in STOPWORDS and not tok.isdigit()}


def caption_token_shifts(df: pd.DataFrame) -> pd.DataFrame:
    per_condition: dict[str, Counter[str]] = {}
    condition_ns: dict[str, int] = {}
    for condition in CONDITION_ORDER:
        sub = df[df["condition"].eq(condition)]
        counts: Counter[str] = Counter()
        for _, row in sub.iterrows():
            counts.update(tokenize(caption_text(row)))
        per_condition[condition] = counts
        condition_ns[condition] = len(sub)

    baseline_counts = per_condition["vicco"]
    baseline_n = condition_ns["vicco"]
    rows = []
    for condition in CONDITION_ORDER[1:]:
        counts = per_condition[condition]
        n = condition_ns[condition]
        vocab = set(counts) | set(baseline_counts)
        for token in vocab:
            condition_prev = counts[token] / n if n else np.nan
            baseline_prev = baseline_counts[token] / baseline_n if baseline_n else np.nan
            rows.append(
                {
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "token": token,
                    "condition_n": int(counts[token]),
                    "baseline_n": int(baseline_counts[token]),
                    "condition_prevalence": float(condition_prev),
                    "baseline_prevalence": float(baseline_prev),
                    "prevalence_difference": float(condition_prev - baseline_prev),
                }
            )
    out = pd.DataFrame(rows)
    out = out[(out["condition_n"].ge(3)) | (out["baseline_n"].ge(6))].copy()
    return out.sort_values(["condition", "prevalence_difference"], ascending=[True, False])


def plot_heatmap(ax: plt.Axes, matrix: pd.DataFrame, title: str, cbar_label: str, limit: float) -> None:
    shown = matrix.to_numpy(dtype=float)
    im = ax.imshow(shown, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold", pad=4)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.tick_params(axis="both", labelsize=6)
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.030, pad=0.015)
    cbar.set_label(cbar_label, fontsize=6)
    cbar.ax.tick_params(labelsize=5.5)


def make_figure(num: pd.DataFrame, binary: pd.DataFrame) -> None:
    num_matrix = (
        num.pivot(index="field_label", columns="condition_label", values="cohens_d")
        .reindex([NUMERIC_LABELS[col] for col in NUMERIC_COLS])
        .reindex([CONDITION_LABELS[c] for c in CONDITION_ORDER[1:]], axis=1)
    )
    bin_matrix = (
        binary.pivot(index="field_label", columns="condition_label", values="prevalence_difference")
        .reindex([BOOLEAN_LABELS[col] for col in BOOLEAN_COLS])
        .reindex([CONDITION_LABELS[c] for c in CONDITION_ORDER[1:]], axis=1)
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), constrained_layout=True)
    plot_heatmap(axes[0], num_matrix, "Numeric annotation shifts", "Cohen's d", limit=1.25)
    plot_heatmap(axes[1], bin_matrix, "Boolean annotation shifts", "prevalence diff.", limit=0.35)
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"semantic_shift_overview.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str], float_digits: int = 3) -> str:
    shown = df[columns].copy()
    for col in shown.columns:
        if pd.api.types.is_numeric_dtype(shown[col]):
            shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")
        else:
            shown[col] = shown[col].fillna("").astype(str)
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in shown.astype(str).to_numpy()]
    return "\n".join([header, rule, *body])


def write_report(num: pd.DataFrame, binary: pd.DataFrame, categories: pd.DataFrame, tokens: pd.DataFrame) -> None:
    lines = [
        "# Simple Semantic Annotation Audit",
        "",
        f"Input: `{ANNOTATIONS.relative_to(STAGE.parents[2])}`",
        "",
        "This is a descriptive VLM-caption/annotation audit. It is not a human behavioral result and not a causal control.",
        "",
        "## Largest numeric shifts by absolute Cohen's d",
        "",
    ]
    top_num = num.reindex(num["cohens_d"].abs().sort_values(ascending=False).index).head(15)
    lines.append(
        markdown_table(
            top_num,
            ["condition_label", "field_label", "baseline_mean", "condition_mean", "mean_difference", "cohens_d"],
        )
    )
    lines.extend(["", "## Largest boolean shifts by absolute prevalence difference", ""])
    top_bin = binary.reindex(binary["prevalence_difference"].abs().sort_values(ascending=False).index).head(15)
    lines.append(
        markdown_table(
            top_bin,
            ["condition_label", "field_label", "baseline_prevalence", "condition_prevalence", "prevalence_difference"],
        )
    )
    lines.extend(["", "## Largest dominant-content/style/domain prevalence shifts", ""])
    top_cat = categories.reindex(categories["prevalence_difference"].abs().sort_values(ascending=False).index).head(20)
    lines.append(
        markdown_table(
            top_cat,
            ["condition_label", "field", "category", "baseline_prevalence", "condition_prevalence", "prevalence_difference"],
        )
    )
    lines.extend(["", "## Top caption-token increases by condition", ""])
    for condition in CONDITION_ORDER[1:]:
        sub = tokens[tokens["condition"].eq(condition)].head(12)
        words = ", ".join(f"{r.token} ({r.prevalence_difference:+.2f})" for r in sub.itertuples())
        lines.append(f"- {CONDITION_LABELS[condition]}: {words}")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = read_annotations()

    num = numeric_shifts(df)
    binary = binary_shifts(df)
    categories = category_shifts(df)
    tokens = caption_token_shifts(df)

    num.to_csv(OUT / "semantic_numeric_shifts.csv", index=False)
    binary.to_csv(OUT / "semantic_binary_shifts.csv", index=False)
    categories.to_csv(OUT / "semantic_category_shifts.csv", index=False)
    tokens.to_csv(OUT / "semantic_caption_token_shifts.csv", index=False)

    make_figure(num, binary)
    write_report(num, binary, categories, tokens)

    print(f"read {len(df)} annotations")
    print(f"wrote semantic audit outputs to {OUT}")


if __name__ == "__main__":
    main()
