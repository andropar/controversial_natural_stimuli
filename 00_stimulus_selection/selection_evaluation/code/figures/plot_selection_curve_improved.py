#!/usr/bin/env python3
"""
Per-model-set selection score progressions.

This restores the compact score-trajectory view used in the final stimulus
selection report: greedy selection is plotted in blue through iteration 97,
accepted refinement replacements continue in orange, and a green star marks the
checkpoint used for the selected stimulus set.

Outputs are written to figures/insilico_curve:
    selection_curve_improved.pdf/png
    selection_curve_<model_set>.pdf/png
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable

_PAPER = Path(__file__).resolve().parents[2]
SHARE_ROOT = _PAPER.parents[1]
HELPERS = SHARE_ROOT / "src"
sys.path.insert(0, str(HELPERS))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from cstims import constants, paths
from cstims.paper.style_improved import (
    apply_style,
    FONT,
    DPI,
    W_SINGLE,
    W_1_5COL,
    OKABE_ITO,
    MODEL_SET_ORDER,
    MODEL_SET_DISPLAY,
    add_panel_label,
)

apply_style()

FIGURES_DIR = _PAPER / "figures" / "insilico_curve"
PNG_DIR = FIGURES_DIR / "png"

SCORE_CANDIDATES = ("score_combined_raw", "score_combined", "score")
COMBINED_EXCLUDE = {"score_combined", "score_combined_raw"}

GREEDY_COLOR = OKABE_ITO["blue"]
REFINEMENT_COLOR = OKABE_ITO["orange"]
STAR_COLOR = OKABE_ITO["bluish_green"]
TRACK_COLORS = {
    "score_sub-01": OKABE_ITO["sky_blue"],
    "score_sub-03": OKABE_ITO["bluish_green"],
    "score_sub-05": OKABE_ITO["reddish_purple"],
    "score_sub-06": "#E78AC3",
    "score_sub-07": "#B8A000",
    "score_raw": "#777777",
}

# Historical final-report checkpoint choices. SOTA and Dataset used the
# completed greedy checkpoint; the other sets used the best completed
# refinement checkpoint.
GREEDY_CHECKPOINT_MODEL_SETS = {"sota", "dataset"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _as_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _score_col(rows: Iterable[dict[str, str]]) -> str:
    rows = list(rows)
    if not rows:
        raise ValueError("No rows available to infer score column")
    keys = rows[0].keys()
    for col in SCORE_CANDIDATES:
        if col in keys:
            return col
    raise ValueError(f"No score column found; tried {SCORE_CANDIDATES}")


def _track_cols(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    cols = [
        c for c in rows[0].keys()
        if c.startswith("score_") and c not in COMBINED_EXCLUDE
    ]
    return sorted(cols, key=lambda c: (c == "score_raw", c))


def _track_label(col: str) -> str:
    return col.replace("score_", "")


def load_score_data(model_set: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    base = paths.selection_evaluation_results_dir() / model_set
    return _read_csv(base / "greedy_scores.csv"), _read_csv(base / "refinement.csv")


def completed_refinement_rows(
    refinement_rows: list[dict[str, str]],
    last_greedy_iter: int,
) -> list[dict[str, str]]:
    rows = [r for r in refinement_rows if _as_bool(r.get("replaced", False))]
    rows = sorted(rows, key=lambda r: (_as_int(r, "pass_num"), _as_int(r, "position")))
    for i, row in enumerate(rows, start=last_greedy_iter + 1):
        row["plot_iteration"] = str(i)
    return rows


def chosen_checkpoint(
    model_set: str,
    greedy_rows: list[dict[str, str]],
    refinement_rows: list[dict[str, str]],
    score_col: str,
) -> tuple[float, float]:
    """Return x/y for the green star.

    Only completed 100-image checkpoints are eligible: the final greedy row and
    accepted refinement replacements. This avoids marking early greedy peaks
    that occurred before the stimulus set reached size 100. The two historical
    greedy-checkpoint sets are kept at the greedy/refinement boundary to match
    the selected stimuli used in the report.
    """
    last_greedy = greedy_rows[-1]
    last_iter = _as_int(last_greedy, "iteration")
    last_score = _as_float(last_greedy, score_col)
    if model_set in GREEDY_CHECKPOINT_MODEL_SETS:
        return last_iter + 0.5, last_score

    candidates = [(float(last_iter), last_score)]
    for row in refinement_rows:
        candidates.append((
            _as_float(row, "plot_iteration"),
            _as_float(row, score_col),
        ))
    return max(candidates, key=lambda item: item[1])


def plot_model_set(
    ax,
    model_set: str,
    show_legend: bool = True,
    compact: bool = False,
) -> bool:
    greedy_rows, refinement_all = load_score_data(model_set)
    if not greedy_rows:
        ax.set_visible(False)
        return False

    score_col = _score_col(greedy_rows)
    track_cols = _track_cols(greedy_rows)
    last_greedy_iter = _as_int(greedy_rows[-1], "iteration")
    refinement_rows = completed_refinement_rows(refinement_all, last_greedy_iter)

    x_greedy = [_as_float(r, "iteration") for r in greedy_rows]

    for col in track_cols:
        color = TRACK_COLORS.get(col, "#999999")
        ax.plot(
            x_greedy,
            [_as_float(r, col) for r in greedy_rows],
            linestyle="--",
            color=color,
            linewidth=0.8 if compact else 1.0,
            alpha=0.45 if compact else 0.55,
            label=_track_label(col),
        )

    ax.plot(
        x_greedy,
        [_as_float(r, score_col) for r in greedy_rows],
        marker="o",
        linestyle="-",
        color=GREEDY_COLOR,
        linewidth=1.7 if compact else 2.0,
        markersize=2.2 if compact else 3.0,
        label="combined",
        zorder=3,
    )

    if refinement_rows:
        boundary = last_greedy_iter + 0.5
        ax.axvline(
            x=boundary,
            color="#777777",
            linestyle="--",
            linewidth=0.8 if compact else 1.0,
            alpha=0.65,
            zorder=1,
        )
        x_ref = [_as_float(r, "plot_iteration") for r in refinement_rows]
        for col in track_cols:
            if col not in refinement_rows[0]:
                continue
            color = TRACK_COLORS.get(col, "#999999")
            ax.plot(
                x_ref,
                [_as_float(r, col) for r in refinement_rows],
                linestyle="--",
                color=color,
                linewidth=0.8 if compact else 1.0,
                alpha=0.45 if compact else 0.55,
            )
        ax.plot(
            x_ref,
            [_as_float(r, score_col) for r in refinement_rows],
            marker="o",
            linestyle="-",
            color=REFINEMENT_COLOR,
            linewidth=1.7 if compact else 2.0,
            markersize=2.2 if compact else 3.0,
            zorder=3,
        )

    star_x, star_y = chosen_checkpoint(model_set, greedy_rows, refinement_rows, score_col)
    ax.axvline(
        x=star_x,
        color=STAR_COLOR,
        linestyle=":",
        linewidth=1.2 if compact else 1.5,
        alpha=0.9,
        zorder=2,
    )
    ax.scatter(
        [star_x],
        [star_y],
        color=STAR_COLOR,
        edgecolor="#004D36",
        linewidth=0.4,
        s=70 if compact else 95,
        marker="*",
        zorder=5,
    )

    ax.set_title(MODEL_SET_DISPLAY[model_set], pad=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=FONT["small"])

    if show_legend:
        ax.legend(
            loc="lower right",
            fontsize=FONT["small"],
            frameon=True,
            framealpha=0.92,
            edgecolor="none",
            ncol=2,
            handlelength=2.4,
            columnspacing=0.8,
        )
    return True


def save_individual_figures() -> None:
    for model_set in MODEL_SET_ORDER:
        fig, ax = plt.subplots(figsize=(W_SINGLE, 3.0))
        ok = plot_model_set(ax, model_set, show_legend=True, compact=False)
        if not ok:
            plt.close(fig)
            continue
        fig.tight_layout()
        for out in (
            FIGURES_DIR / f"selection_curve_{model_set}.pdf",
            PNG_DIR / f"selection_curve_{model_set}.png",
        ):
            fig.savefig(out, dpi=DPI)
            print(f"Saved {out}")
        plt.close(fig)


def save_panel_figure() -> None:
    fig, axes = plt.subplots(3, 2, figsize=(W_1_5COL, 8.2), sharex=False)
    axes_flat = axes.ravel()
    plotted = []
    for i, (ax, model_set) in enumerate(zip(axes_flat, MODEL_SET_ORDER)):
        ok = plot_model_set(
            ax,
            model_set,
            show_legend=False,
            compact=True,
        )
        if ok:
            add_panel_label(ax, chr(ord("a") + i), x=-0.12, y=1.03)
            ax.set_xlabel("")
            ax.set_ylabel("")
            plotted.append(ax)

    legend_ax = axes_flat[-1]
    legend_ax.axis("off")
    legend_handles = [
        Line2D([0], [0], color=GREEDY_COLOR, lw=2.0, marker="o", markersize=4,
               label="greedy combined"),
        Line2D([0], [0], color=REFINEMENT_COLOR, lw=2.0, marker="o", markersize=4,
               label="refinement combined"),
        Line2D([0], [0], color=STAR_COLOR, lw=1.5, ls=":", marker="*",
               markersize=10, markeredgecolor="#004D36", label="chosen checkpoint"),
        Line2D([0], [0], color="#777777", lw=1.0, ls="--",
               label="refinement starts"),
        Line2D([0], [0], color="#999999", lw=1.0, ls="--", alpha=0.6,
               label="individual tracks"),
    ]
    legend_ax.legend(
        handles=legend_handles,
        loc="center left",
        frameon=False,
        fontsize=FONT["legend"],
        handlelength=2.6,
    )

    fig.supxlabel("Iteration", fontsize=FONT["axis_label"], y=0.035)
    fig.supylabel("Score", fontsize=FONT["axis_label"], x=0.035)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.96, bottom=0.09,
                        wspace=0.28, hspace=0.45)
    for stem in ("selection_curve", "selection_curve_improved", "selection_curve_compact"):
        for out in (
            FIGURES_DIR / f"{stem}.pdf",
            PNG_DIR / f"{stem}.png",
        ):
            fig.savefig(out, dpi=DPI)
            print(f"Saved {out}")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    save_individual_figures()
    save_panel_figure()


if __name__ == "__main__":
    main()
