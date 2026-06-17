#!/usr/bin/env python3
"""Plot compact feature-method sweep summary figures."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
RECOVERY_ROOT = (
    REPO_ROOT
    / "00_stimulus_selection"
    / "selection_evaluation"
    / "feature_method_sweep_recovery"
    / "noisy_by_clean"
)
DEFAULT_RUN = RECOVERY_ROOT / "results" / "sota_20260611_112941"
ENCODING_TRACKS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")
EMPIRICAL_SNR = 1.0
SNR_TICKS = [0.01, 0.1, 1, 10]
SNR_TICK_LABELS = ["0.01", "0.1", "1", "10"]
NUMERIC_COLUMNS = [
    "selected_auc_effective_weighted",
    "random_auc_effective_weighted",
    "selected_auc_mean_tracks",
    "selected_auc_worst_track",
    "selected_auc_raw",
    "selected_auc_sub01",
    "selected_auc_mean_encoding_tracks",
    "auc_delta_effective_weighted_random_minus_selected",
    "pairwise_dominance_auc_effective_weighted",
    "mean_margin_auc_effective_weighted",
    "pairwise_dominance_auc_mean_tracks",
    "mean_margin_auc_mean_tracks",
]

LABELS = {
    "raw_only_mean_min": "Raw features only",
    "raw_enc_w05_max_mean": "Raw + enc, max/mean",
    "raw_enc_w05_mean_min": "Intended (Raw + enc, mean/min)",
    "sub01_only_mean_min": "Sub-01 only (current)",
    "paper_effective_identity_sub01_mean_min": "Paper-effective sub-01",
    "raw_enc_w05_max_min": "Raw + enc, max/min",
    "paper_effective_identity_sub01_mean_min_no_attenuation": "Sub-01 only (no attenuation)",
}

METHOD_ORDER = [
    "raw_only_mean_min",
    "sub01_only_mean_min",
    "paper_effective_identity_sub01_mean_min",
    "raw_enc_w05_mean_min",
    "raw_enc_w05_max_mean",
    "raw_enc_w05_max_min",
    "paper_effective_identity_sub01_mean_min_no_attenuation",
]

COLORS = {
    "raw_only_mean_min": "#4C78A8",
    "sub01_only_mean_min": "#F28E2B",
    "paper_effective_identity_sub01_mean_min": "#E15759",
    "raw_enc_w05_mean_min": "#59A14F",
    "raw_enc_w05_max_mean": "#76B7B2",
    "raw_enc_w05_max_min": "#B07AA1",
    "paper_effective_identity_sub01_mean_min_no_attenuation": "#9C755F",
}

MODEL_LABELS = {
    "slip_vit_l_slip": "SLIP",
    "slip_vit_l_simclr": "SimCLR-ViT",
    "timm_vit_large_patch14_clip_224_laion2b": "CLIP-L2B",
    "dinov2_vitl14": "DINOv2",
    "openclip_vit_so400m_14_siglip_webli": "SigLIP",
    "torchvision_convnext_base_imagenet1k_v1": "ConvNeXt-B",
}


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        first = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    except Exception:
        return False
    return first.startswith("version https://git-lfs.github.com/spec/")


def parse_summary_markdown(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```text\n(?P<table>.*?)\n```", text, flags=re.S)
    if not match:
        raise ValueError(f"Could not find fixed-width summary table in {path}")

    rows = []
    row_re = re.compile(
        r"^\s*(?P<method_id>\S+)\s+"
        r"(?P<method_label>.*?)\s+"
        r"(?P<within>mean|max)\s+"
        r"(?P<across>min|mean)\s+"
        r"(?P<track_agg_method>\S+)\s+"
        r"(?P<values>[-+0-9.NaInf\s]+?)\s*$"
    )
    for line in match.group("table").splitlines()[1:]:
        if not line.strip():
            continue
        parsed = row_re.match(line)
        if not parsed:
            raise ValueError(f"Could not parse summary row: {line}")
        values = parsed.group("values").split()
        if len(values) != len(NUMERIC_COLUMNS):
            raise ValueError(f"Expected {len(NUMERIC_COLUMNS)} values, got {len(values)}: {line}")
        row = {
            "method_id": parsed.group("method_id"),
            "method_label": parsed.group("method_label").strip(),
            "within": parsed.group("within"),
            "across": parsed.group("across"),
            "track_agg_method": parsed.group("track_agg_method"),
        }
        for key, value in zip(NUMERIC_COLUMNS, values):
            row[key] = np.nan if value == "NaN" else float(value)
        rows.append(row)

    return pd.DataFrame(rows)


def load_summary(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "comparison" / "method_summary.csv"
    if csv_path.exists() and not is_lfs_pointer(csv_path):
        return pd.read_csv(csv_path)
    return parse_summary_markdown(run_dir / "comparison" / "method_summary.md")


def load_recovery_auc(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "comparison" / "recovery_auc_by_method.csv"
    if not path.exists() or is_lfs_pointer(path):
        raise FileNotFoundError(f"Recovery AUC CSV is unavailable or not hydrated: {path}")
    return pd.read_csv(path)


def method_id_from_eval_dir(path: Path) -> str:
    return path.name.removesuffix("_noisy_by_clean_boot")


def load_eval_tables(eval_root: Path, filename: str) -> pd.DataFrame:
    frames = []
    for path in sorted(eval_root.glob(f"*/{filename}")):
        if is_lfs_pointer(path):
            raise FileNotFoundError(f"CSV is not hydrated: {path}")
        df = pd.read_csv(path)
        method_id = method_id_from_eval_dir(path.parent)
        df["method_id"] = method_id
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def replace_with_cross_eval(base: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    if cross.empty:
        return base
    cross_methods = set(cross["method_id"])
    return pd.concat(
        [base[~base["method_id"].isin(cross_methods)], cross],
        ignore_index=True,
        sort=False,
    )


def load_discriminability(run_dir: Path) -> pd.DataFrame:
    base = load_eval_tables(run_dir / "eval", "discriminability.csv")
    if base.empty:
        raise FileNotFoundError(f"No discriminability CSVs found under {run_dir / 'eval'}")
    cross = load_eval_tables(
        run_dir / "cross_eval_full_tracks" / "eval",
        "discriminability.csv",
    )
    return replace_with_cross_eval(base, cross)


def load_pairwise_auc(run_dir: Path) -> pd.DataFrame:
    base = load_eval_tables(run_dir / "eval", "pairwise_auc.csv")
    if base.empty:
        path = run_dir / "comparison" / "pairwise_auc_by_method.csv"
        if not path.exists() or is_lfs_pointer(path):
            raise FileNotFoundError(f"Pairwise AUC CSV is unavailable or not hydrated: {path}")
        base = pd.read_csv(path)
    cross = load_eval_tables(
        run_dir / "cross_eval_full_tracks" / "eval",
        "pairwise_auc.csv",
    )
    return replace_with_cross_eval(base, cross)


def load_correlation_matrices(run_dir: Path) -> pd.DataFrame:
    base = load_eval_tables(run_dir / "eval", "correlation_matrices.csv")
    if base.empty:
        raise FileNotFoundError(f"No correlation matrix CSVs found under {run_dir / 'eval'}")
    cross = load_eval_tables(
        run_dir / "cross_eval_full_tracks" / "eval",
        "correlation_matrices.csv",
    )
    return replace_with_cross_eval(base, cross)


def load_model_order(run_dir: Path, corr: pd.DataFrame) -> list[str]:
    config_path = run_dir / "run_config.json"
    if config_path.exists() and not is_lfs_pointer(config_path):
        with config_path.open(encoding="utf-8") as f:
            config = json.load(f)
        model_names = config.get("model_names")
        if isinstance(model_names, list) and model_names:
            return [str(model) for model in model_names]

    seen = []
    for column in ("model_i", "model_j"):
        for model in corr[column].dropna():
            if model not in seen:
                seen.append(model)
    return seen


def method_label(method_id: str) -> str:
    return LABELS.get(method_id, method_id)


def method_color(method_id: str) -> str:
    return COLORS.get(method_id, "#777777")


def sort_methods(df: pd.DataFrame, value_col: str = "selected_auc", ascending: bool = False) -> pd.DataFrame:
    order = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    return df.assign(_order=df["method_id"].map(order).fillna(999)).sort_values(
        ["_order", value_col], ascending=[True, ascending]
    ).drop(columns="_order")


def parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    return parsed or None


def filter_methods(df: pd.DataFrame, methods: list[str] | None) -> pd.DataFrame:
    if methods is None or df.empty or "method_id" not in df:
        return df
    return df[df["method_id"].isin(methods)].copy()


def compute_log_noise_auc(noise_mult: pd.Series, values: pd.Series) -> float:
    x = np.asarray(noise_mult, dtype=float)
    y = np.asarray(values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        return float("nan")
    order = np.argsort(x)
    x_log = np.log10(x[order] + 1e-10)
    y_sorted = y[order]
    span = x_log[-1] - x_log[0]
    auc = float(np.trapezoid(y_sorted, x_log))
    return auc / span if span > 0 else auc


def sample_sd(values: pd.Series | list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    return float(arr.std(ddof=1))


def sample_sem(values: pd.Series | list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def add_horizontal_errorbars(
    ax: plt.Axes,
    x: pd.Series,
    y: np.ndarray,
    xerr: pd.Series | None,
    *,
    color: str = "#222222",
    alpha: float = 0.75,
    linewidth: float = 0.8,
) -> None:
    if xerr is None:
        return
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    err_arr = np.asarray(xerr, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(err_arr) & (err_arr > 0)
    if not valid.any():
        return
    ax.errorbar(
        x_arr[valid],
        y_arr[valid],
        xerr=err_arr[valid],
        fmt="none",
        ecolor=color,
        elinewidth=linewidth,
        capsize=2.0,
        capthick=linewidth,
        alpha=alpha,
        zorder=4,
    )


def aggregate_auc_by_space(recovery: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method_id, grp in recovery.groupby("method_id", sort=False):
        label = method_label(method_id)
        raw = grp[grp["track"] == "raw"]
        if not raw.empty:
            r = raw.iloc[0]
            rows.append(
                {
                    "method_id": method_id,
                    "method_label": label,
                    "eval_space": "Raw",
                    "selected_auc": float(r["selected_auc"]),
                    "random_auc": float(r["random_auc_mean"]),
                    "selected_auc_sd": np.nan,
                    "selected_auc_sem": np.nan,
                    "random_auc_sd": np.nan,
                    "random_auc_sem": np.nan,
                    "n_tracks": 1,
                }
            )
        enc = grp[grp["track"].isin(ENCODING_TRACKS)]
        if not enc.empty:
            rows.append(
                {
                    "method_id": method_id,
                    "method_label": label,
                    "eval_space": "Encoded mean",
                    "selected_auc": float(enc["selected_auc"].mean()),
                    "random_auc": float(enc["random_auc_mean"].mean()),
                    "selected_auc_sd": sample_sd(enc["selected_auc"]),
                    "selected_auc_sem": sample_sem(enc["selected_auc"]),
                    "random_auc_sd": sample_sd(enc["random_auc_mean"]),
                    "random_auc_sem": sample_sem(enc["random_auc_mean"]),
                    "n_tracks": int(enc["track"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def aggregate_curves_by_space(discriminability: pd.DataFrame) -> pd.DataFrame:
    df = discriminability.copy()
    df["noise_ceiling_bin"] = df["noise_ceiling"].round(10)
    rows = []
    for method_id, grp in df.groupby("method_id", sort=False):
        for subset_type in ("selected", "random"):
            raw = grp[(grp["track"] == "raw") & (grp["subset_type"] == subset_type)]
            if not raw.empty:
                raw_curve = (
                    raw.groupby("noise_ceiling_bin", as_index=False)
                    .agg(
                        noise_mult=("noise_mult", "mean"),
                        noise_ceiling=("noise_ceiling", "mean"),
                        error_prob=("error_prob", "mean"),
                        error_prob_sd=("error_prob", sample_sd),
                        error_prob_sem=("error_prob", sample_sem),
                        n_tracks=("track", "nunique"),
                    )
                    .assign(
                        method_id=method_id,
                        method_label=method_label(method_id),
                        eval_space="Raw",
                        subset_type=subset_type,
                    )
                )
                rows.append(raw_curve)

            enc = grp[(grp["track"].isin(ENCODING_TRACKS)) & (grp["subset_type"] == subset_type)]
            if not enc.empty:
                enc_curve = (
                    enc.groupby("noise_ceiling_bin", as_index=False)
                    .agg(
                        noise_mult=("noise_mult", "mean"),
                        noise_ceiling=("noise_ceiling", "mean"),
                        error_prob=("error_prob", "mean"),
                        error_prob_sd=("error_prob", sample_sd),
                        error_prob_sem=("error_prob", sample_sem),
                        n_tracks=("track", "nunique"),
                    )
                    .assign(
                        method_id=method_id,
                        method_label=method_label(method_id),
                        eval_space="Encoded mean",
                        subset_type=subset_type,
                    )
                )
                rows.append(enc_curve)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["snr"] = 1.0 / out["noise_mult"].astype(float)
    return out


def aggregate_recovery_auc_by_space(discriminability: pd.DataFrame) -> pd.DataFrame:
    track_rows = []
    for (method_id, track, subset_type), grp in discriminability.groupby(
        ["method_id", "track", "subset_type"], sort=False
    ):
        if track == "raw":
            eval_space = "Raw"
        elif track in ENCODING_TRACKS:
            eval_space = "Encoded mean"
        else:
            continue
        track_rows.append(
            {
                "method_id": method_id,
                "method_label": method_label(method_id),
                "eval_space": eval_space,
                "track": track,
                "subset_type": subset_type,
                "auc": compute_log_noise_auc(grp["noise_mult"], 1.0 - grp["error_prob"]),
            }
        )
    track_auc = pd.DataFrame(track_rows)
    rows = []
    for keys, grp in track_auc.groupby(["method_id", "method_label", "eval_space"], sort=False):
        method_id, label, eval_space = keys
        row = {
            "method_id": method_id,
            "method_label": label,
            "eval_space": eval_space,
            "n_tracks": int(grp["track"].nunique()),
        }
        for subset_type in ("selected", "random"):
            sub = grp[grp["subset_type"] == subset_type]
            row[f"{subset_type}_auc"] = float(sub["auc"].mean()) if not sub.empty else np.nan
            row[f"{subset_type}_auc_sd"] = sample_sd(sub["auc"])
            row[f"{subset_type}_auc_sem"] = sample_sem(sub["auc"])
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_recovery_auc_from_curves(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, grp in curves.groupby(["method_id", "method_label", "eval_space"], sort=False):
        method_id, label, eval_space = keys
        row = {
            "method_id": method_id,
            "method_label": label,
            "eval_space": eval_space,
            "n_tracks": int(grp["n_tracks"].max()),
        }
        for subset_type in ("selected", "random"):
            sub = grp[grp["subset_type"] == subset_type]
            row[f"{subset_type}_auc"] = compute_log_noise_auc(
                sub["noise_mult"],
                1.0 - sub["error_prob"],
            )
            row[f"{subset_type}_auc_sd"] = np.nan
            row[f"{subset_type}_auc_sem"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_pairwise_metrics_by_space(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def append_metric_rows(method_id: str, label: str, space: str, grp: pd.DataFrame) -> None:
        rows.append(
            {
                "method_id": method_id,
                "method_label": label,
                "eval_space": space,
                "metric": "Pairwise dominance AUC",
                "selected_value": float(grp["selected_pairwise_dominance_auc"].mean()),
                "random_value": float(grp["random_pairwise_dominance_auc_mean"].mean()),
                "selected_value_sd": sample_sd(grp["selected_pairwise_dominance_auc"]),
                "selected_value_sem": sample_sem(grp["selected_pairwise_dominance_auc"]),
                "random_value_sd": sample_sd(grp["random_pairwise_dominance_auc_mean"]),
                "random_value_sem": sample_sem(grp["random_pairwise_dominance_auc_mean"]),
                "n_tracks": int(grp["track"].nunique()),
            }
        )
        rows.append(
            {
                "method_id": method_id,
                "method_label": label,
                "eval_space": space,
                "metric": "Mean margin AUC",
                "selected_value": float(grp["selected_mean_margin_auc"].mean()),
                "random_value": float(grp["random_mean_margin_auc_mean"].mean()),
                "selected_value_sd": sample_sd(grp["selected_mean_margin_auc"]),
                "selected_value_sem": sample_sem(grp["selected_mean_margin_auc"]),
                "random_value_sd": sample_sd(grp["random_mean_margin_auc_mean"]),
                "random_value_sem": sample_sem(grp["random_mean_margin_auc_mean"]),
                "n_tracks": int(grp["track"].nunique()),
            }
        )

    for method_id, grp in pairwise.groupby("method_id", sort=False):
        label = method_label(method_id)
        raw = grp[grp["track"] == "raw"]
        if not raw.empty:
            append_metric_rows(method_id, label, "Raw", raw)
        enc = grp[grp["track"].isin(ENCODING_TRACKS)]
        if not enc.empty:
            append_metric_rows(method_id, label, "Encoded mean", enc)

    return pd.DataFrame(rows)


def recovery_auc_as_metric_rows(auc_space: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in auc_space.iterrows():
        rows.append(
            {
                "method_id": row["method_id"],
                "method_label": row["method_label"],
                "eval_space": row["eval_space"],
                "metric": "Recovery accuracy AUC",
                "selected_value": float(row["selected_auc"]),
                "random_value": float(row["random_auc"]),
                "selected_value_sd": row.get("selected_auc_sd", np.nan),
                "selected_value_sem": row.get("selected_auc_sem", np.nan),
                "random_value_sd": row.get("random_auc_sd", np.nan),
                "random_value_sem": row.get("random_auc_sem", np.nan),
                "n_tracks": int(row["n_tracks"]),
            }
        )
    return pd.DataFrame(rows)


def aggregate_attenuated_correlations_by_space(corr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    corr = corr[corr["matrix_type"].isin(["selected_noised", "random_noised"])].copy()

    def track_summary(track_df: pd.DataFrame) -> tuple[float, float]:
        diag = track_df[track_df["model_i"] == track_df["model_j"]]["correlation"]
        offdiag = track_df[track_df["model_i"] != track_df["model_j"]]["correlation"]
        return float(diag.mean()), float(offdiag.mean())

    def append_corr_rows(method_id: str, label: str, space: str, grp: pd.DataFrame) -> None:
        selected = grp[grp["matrix_type"] == "selected_noised"]
        random = grp[grp["matrix_type"] == "random_noised"]
        selected_per_track = []
        for _, track_df in selected.groupby("track", sort=False):
            self_corr, other_corr = track_summary(track_df)
            selected_per_track.append((self_corr, other_corr))
        if not selected_per_track:
            return
        random_per_track = []
        for _, track_df in random.groupby("track", sort=False):
            self_corr, other_corr = track_summary(track_df)
            random_per_track.append((self_corr, other_corr))
        self_values = [x[0] for x in selected_per_track]
        other_values = [x[1] for x in selected_per_track]
        random_self_values = [x[0] for x in random_per_track]
        random_other_values = [x[1] for x in random_per_track]
        rows.append(
            {
                "method_id": method_id,
                "method_label": label,
                "eval_space": space,
                "metric": "Attenuated self-correlation",
                "selected_value": float(np.mean(self_values)),
                "random_value": float(np.mean(random_self_values)) if random_self_values else np.nan,
                "selected_value_sd": sample_sd(self_values),
                "selected_value_sem": sample_sem(self_values),
                "random_value_sd": sample_sd(random_self_values),
                "random_value_sem": sample_sem(random_self_values),
                "n_tracks": int(selected["track"].nunique()),
            }
        )
        rows.append(
            {
                "method_id": method_id,
                "method_label": label,
                "eval_space": space,
                "metric": "Attenuated other-correlation",
                "selected_value": float(np.mean(other_values)),
                "random_value": float(np.mean(random_other_values)) if random_other_values else np.nan,
                "selected_value_sd": sample_sd(other_values),
                "selected_value_sem": sample_sem(other_values),
                "random_value_sd": sample_sd(random_other_values),
                "random_value_sem": sample_sem(random_other_values),
                "n_tracks": int(selected["track"].nunique()),
            }
        )

    for method_id, grp in corr.groupby("method_id", sort=False):
        label = method_label(method_id)
        raw = grp[grp["track"] == "raw"]
        if not raw.empty:
            append_corr_rows(method_id, label, "Raw", raw)
        enc = grp[grp["track"].isin(ENCODING_TRACKS)]
        if not enc.empty:
            append_corr_rows(method_id, label, "Encoded mean", enc)

    return pd.DataFrame(rows)


def ordered_present_methods(df: pd.DataFrame) -> list[str]:
    present = list(dict.fromkeys(df["method_id"].dropna()))
    ordered = [method_id for method_id in METHOD_ORDER if method_id in present]
    ordered.extend(method_id for method_id in present if method_id not in ordered)
    return ordered


def selected_correlation_matrices_by_space(
    corr: pd.DataFrame,
    method_ids: list[str],
) -> pd.DataFrame:
    rows = []
    matrix_specs = [
        ("selected_clean", "Clean"),
        ("selected_noised", f"Attenuated (SNR={EMPIRICAL_SNR:g})"),
    ]
    space_specs = [
        ("Raw", ("raw",)),
        ("Mean encoding", ENCODING_TRACKS),
    ]
    selected = corr[corr["matrix_type"].isin([spec[0] for spec in matrix_specs])].copy()
    for method_id in method_ids:
        method_df = selected[selected["method_id"] == method_id]
        if method_df.empty:
            continue
        for eval_space, tracks in space_specs:
            for matrix_type, matrix_label in matrix_specs:
                sub = method_df[
                    (method_df["track"].isin(tracks))
                    & (method_df["matrix_type"] == matrix_type)
                ]
                if sub.empty:
                    continue
                n_tracks = int(sub["track"].nunique())
                pair_mean = (
                    sub.groupby(["model_i", "model_j"], as_index=False)
                    .agg(correlation=("correlation", "mean"))
                    .assign(
                        method_id=method_id,
                        method_label=method_label(method_id),
                        eval_space=eval_space,
                        matrix_type=matrix_type,
                        matrix_label=matrix_label,
                        n_tracks=n_tracks,
                    )
                )
                rows.append(pair_mean)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)[
        [
            "method_id",
            "method_label",
            "eval_space",
            "matrix_type",
            "matrix_label",
            "n_tracks",
            "model_i",
            "model_j",
            "correlation",
        ]
    ]


def matrix_from_long(df: pd.DataFrame, model_order: list[str]) -> np.ndarray:
    matrix = np.full((len(model_order), len(model_order)), np.nan, dtype=float)
    index = {model: idx for idx, model in enumerate(model_order)}
    for _, row in df.iterrows():
        model_i = row["model_i"]
        model_j = row["model_j"]
        if model_i in index and model_j in index:
            matrix[index[model_i], index[model_j]] = float(row["correlation"])
    return matrix


def median_offdiag(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 2:
        return float("nan")
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(np.nanmedian(matrix[mask]))


def add_value_labels(ax: plt.Axes, values: pd.Series, y: np.ndarray, *, fmt: str = "{:.3f}") -> None:
    xmin, xmax = ax.get_xlim()
    offset = (xmax - xmin) * 0.012
    for yi, value in zip(y, values):
        if np.isfinite(value):
            ax.text(value + offset, yi, fmt.format(value), va="center", ha="left", fontsize=8)


def add_percent_vs_random_labels(
    ax: plt.Axes,
    selected_values: pd.Series,
    random_values: pd.Series,
    y: np.ndarray,
    *,
    fontsize: float = 7,
    outside_axis: bool = False,
) -> None:
    xmin, xmax = ax.get_xlim()
    offset = (xmax - xmin) * 0.012
    for yi, selected, random in zip(y, selected_values, random_values):
        if not (np.isfinite(selected) and np.isfinite(random)) or abs(random) < 1e-12:
            continue
        pct = 100.0 * (float(selected) - float(random)) / abs(float(random))
        label = f"{pct:+.1f}%"
        if outside_axis:
            ax.text(
                1.01,
                yi,
                label,
                va="center",
                ha="left",
                fontsize=fontsize,
                color="#555555",
                transform=ax.get_yaxis_transform(),
                clip_on=False,
            )
            continue
        anchor = max(float(selected), float(random)) + 2.0 * offset
        if anchor > xmax - offset:
            ax.text(
                min(xmax - offset, max(float(selected), float(random)) - offset),
                yi,
                label,
                va="center",
                ha="right",
                fontsize=fontsize,
                color="#555555",
            )
        else:
            ax.text(
                anchor,
                yi,
                label,
                va="center",
                ha="left",
                fontsize=fontsize,
                color="#555555",
            )


def comparison_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color="#777777", linewidth=7.0, alpha=0.88, label="Selected"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#333333",
            markeredgewidth=1.1,
            markersize=6,
            label="Random mean",
        ),
    ]


def plot_model_correlation_matrices(
    matrix_rows: pd.DataFrame,
    model_order: list[str],
    out_dir: Path,
) -> list[Path]:
    if matrix_rows.empty:
        return []

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.bottom": False,
            "axes.spines.left": False,
        }
    )

    method_ids = ordered_present_methods(matrix_rows)
    model_set = set(matrix_rows["model_i"]) | set(matrix_rows["model_j"])
    model_order = [model for model in model_order if model in model_set]
    model_order.extend(sorted(model for model in model_set if model not in model_order))
    model_labels = [MODEL_LABELS.get(model, model) for model in model_order]
    panel_specs = [
        ("Raw", "selected_clean", "Raw\nclean"),
        ("Raw", "selected_noised", f"Raw\nattenuated\nSNR={EMPIRICAL_SNR:g}"),
        ("Mean encoding", "selected_clean", "Mean encoding\nclean"),
        ("Mean encoding", "selected_noised", f"Mean encoding\nattenuated\nSNR={EMPIRICAL_SNR:g}"),
    ]

    n_rows = len(method_ids)
    n_cols = len(panel_specs)
    fig_w = max(11.0, 2.35 * n_cols + 2.4)
    fig_h = max(3.2, 2.0 * n_rows + 1.3)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        constrained_layout=True,
        squeeze=False,
    )

    norm = mcolors.TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=1.0001)
    cmap = "RdYlBu_r"
    image = None

    for row_idx, method_id in enumerate(method_ids):
        for col_idx, (eval_space, matrix_type, panel_title) in enumerate(panel_specs):
            ax = axes[row_idx, col_idx]
            sub = matrix_rows[
                (matrix_rows["method_id"] == method_id)
                & (matrix_rows["eval_space"] == eval_space)
                & (matrix_rows["matrix_type"] == matrix_type)
            ]
            matrix = matrix_from_long(sub, model_order)
            image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
            ax.set_xticks(np.arange(len(model_order)))
            ax.set_yticks(np.arange(len(model_order)))
            if row_idx == n_rows - 1:
                ax.set_xticklabels(model_labels, rotation=45, ha="right", fontsize=7)
            else:
                ax.set_xticklabels([])
            if col_idx == 0:
                ax.set_yticklabels(model_labels, fontsize=7)
                row_label = "\n".join(textwrap.wrap(method_label(method_id), width=24))
                ax.set_ylabel(
                    row_label,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=72,
                    fontsize=8,
                    fontweight="bold",
                )
            else:
                ax.set_yticklabels([])
            if row_idx == 0:
                ax.set_title(panel_title, fontsize=9, pad=8)

            ax.set_xticks(np.arange(-0.5, len(model_order), 1.0), minor=True)
            ax.set_yticks(np.arange(-0.5, len(model_order), 1.0), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.5, alpha=0.8)
            ax.tick_params(axis="both", which="both", length=0)
            diag = float(np.nanmedian(np.diag(matrix)))
            offdiag = median_offdiag(matrix)
            ax.text(
                0.03,
                0.97,
                f"diag {diag:.2f}\noff {offdiag:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.3,
                color="#111111",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.72,
                },
            )

    fig.suptitle(
        "SOTA feature-method sweep: selected model-to-model RDM correlations",
        fontsize=12,
        fontweight="bold",
    )
    if image is not None:
        cbar = fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            fraction=0.025,
            pad=0.025,
            shrink=0.72,
        )
        cbar.set_label("Correlation between model RDMs", fontsize=8)
        cbar.ax.tick_params(labelsize=7, length=2)
        cbar.outline.set_visible(False)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "feature_method_sweep_raw_vs_encoded_model_correlation_matrices.pdf"
    png = out_dir / "feature_method_sweep_raw_vs_encoded_model_correlation_matrices.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def plot_space_auc(auc_space: pd.DataFrame, out_dir: Path) -> list[Path]:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.0), constrained_layout=True, sharex=True)
    finite_values = auc_space[["selected_auc", "random_auc"]].to_numpy(dtype=float).ravel()
    for value_col, sem_col in [
        ("selected_auc", "selected_auc_sem"),
        ("random_auc", "random_auc_sem"),
    ]:
        if sem_col in auc_space:
            values = auc_space[value_col].to_numpy(dtype=float)
            sem = auc_space[sem_col].fillna(0).to_numpy(dtype=float)
            finite_values = np.concatenate([finite_values, values - sem, values + sem])
    finite_values = finite_values[np.isfinite(finite_values)]
    xmin = 0.0
    xmax = 1.16

    for ax, space in zip(axes, ["Raw", "Encoded mean"]):
        sub = sort_methods(auc_space[auc_space["eval_space"] == space].copy())
        y = np.arange(len(sub))
        colors = [method_color(method_id) for method_id in sub["method_id"]]
        ax.barh(y, sub["selected_auc"], color=colors, alpha=0.88, label="Selected")
        add_horizontal_errorbars(
            ax,
            sub["selected_auc"],
            y,
            sub.get("selected_auc_sem"),
            color="#111111",
            alpha=0.72,
            linewidth=0.85,
        )
        ax.scatter(
            sub["random_auc"],
            y,
            marker="o",
            s=28,
            facecolor="white",
            edgecolor="#333333",
            linewidth=1.1,
            label="Random mean",
            zorder=3,
        )
        add_horizontal_errorbars(
            ax,
            sub["random_auc"],
            y,
            sub.get("random_auc_sem"),
            color="#333333",
            alpha=0.72,
            linewidth=0.8,
        )
        for yi, selected, random in zip(y, sub["selected_auc"], sub["random_auc"]):
            ax.plot([selected, random], [yi, yi], color="#555555", linewidth=0.8, alpha=0.6)
        add_percent_vs_random_labels(
            ax,
            sub["selected_auc"],
            sub["random_auc"],
            y,
            fontsize=7,
        )
        labels = [
            f"{label} ({int(n_tracks)})" if space == "Encoded mean" and int(n_tracks) != 5 else label
            for label, n_tracks in zip(sub["method_label"], sub["n_tracks"])
        ]
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title(space)
        ax.set_xlabel("Strict top-1 recovery accuracy AUC\nhigher is better")
        ax.set_xlim(xmin, xmax)
        ax.text(
            0.99,
            1.02,
            "% vs random",
            ha="right",
            va="bottom",
            fontsize=7,
            color="#555555",
            transform=ax.transAxes,
        )
        if space == "Raw":
            fig.legend(
                handles=comparison_legend_handles(),
                frameon=False,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=2,
                fontsize=8,
            )

    fig.suptitle("SOTA feature-method sweep: raw vs encoded evaluation", fontsize=12, fontweight="bold")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "feature_method_sweep_raw_vs_encoded_auc.pdf"
    png = out_dir / "feature_method_sweep_raw_vs_encoded_auc.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def metric_xlim(metric: str, values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    if "Recovery accuracy" in metric:
        return 0.0, 1.16
    if "dominance" in metric:
        return 0.0, 1.16
    if "margin" in metric:
        return 0.0, float(values.max()) + 0.035
    return 0.0, min(1.0, float(values.max()) + 0.055)


def plot_metric_panels(metrics: pd.DataFrame, out_dir: Path) -> list[Path]:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    metric_order = [
        "Recovery accuracy AUC",
        "Pairwise dominance AUC",
        "Mean margin AUC",
        "Attenuated self-correlation",
        "Attenuated other-correlation",
    ]
    spaces = ["Raw", "Encoded mean"]
    fig, axes = plt.subplots(
        len(metric_order),
        len(spaces),
        figsize=(12.6, 12.2),
        constrained_layout=True,
    )

    shared_xlim: dict[str, tuple[float, float]] = {}
    attenuated_metrics = ["Attenuated self-correlation", "Attenuated other-correlation"]
    attenuated = metrics[metrics["metric"].isin(attenuated_metrics)]
    if not attenuated.empty:
        attenuated_values = attenuated[["selected_value", "random_value"]].to_numpy(dtype=float).ravel()
        for value_col, sem_col in [
            ("selected_value", "selected_value_sem"),
            ("random_value", "random_value_sem"),
        ]:
            if sem_col in attenuated:
                values = attenuated[value_col].to_numpy(dtype=float)
                sem = attenuated[sem_col].fillna(0).to_numpy(dtype=float)
                attenuated_values = np.concatenate([attenuated_values, values - sem, values + sem])
        shared_attenuated_xlim = metric_xlim("Attenuated correlation", attenuated_values)
        shared_xlim.update({metric: shared_attenuated_xlim for metric in attenuated_metrics})

    for row_idx, metric in enumerate(metric_order):
        metric_values = metrics[metrics["metric"] == metric][["selected_value", "random_value"]].to_numpy(
            dtype=float
        ).ravel()
        metric_sub_all = metrics[metrics["metric"] == metric]
        for value_col, sem_col in [
            ("selected_value", "selected_value_sem"),
            ("random_value", "random_value_sem"),
        ]:
            if sem_col in metric_sub_all:
                values = metric_sub_all[value_col].to_numpy(dtype=float)
                sem = metric_sub_all[sem_col].fillna(0).to_numpy(dtype=float)
                metric_values = np.concatenate([metric_values, values - sem, values + sem])
        xmin, xmax = shared_xlim.get(metric, metric_xlim(metric, metric_values))
        for col_idx, space in enumerate(spaces):
            ax = axes[row_idx, col_idx]
            sub = metrics[(metrics["metric"] == metric) & (metrics["eval_space"] == space)].copy()
            sub = sort_methods(sub, value_col="selected_value")
            y = np.arange(len(sub))
            colors = [method_color(method_id) for method_id in sub["method_id"]]
            ax.barh(y, sub["selected_value"], color=colors, alpha=0.88, label="Selected")
            add_horizontal_errorbars(
                ax,
                sub["selected_value"],
                y,
                sub.get("selected_value_sem"),
                color="#111111",
                alpha=0.72,
                linewidth=0.75,
            )
            if sub["random_value"].notna().any():
                ax.scatter(
                    sub["random_value"],
                    y,
                    marker="o",
                    s=22,
                    facecolor="white",
                    edgecolor="#333333",
                    linewidth=1.0,
                    label="Random mean",
                    zorder=3,
                )
                add_horizontal_errorbars(
                    ax,
                    sub["random_value"],
                    y,
                    sub.get("random_value_sem"),
                    color="#333333",
                    alpha=0.72,
                    linewidth=0.7,
                )
                for yi, selected, random in zip(y, sub["selected_value"], sub["random_value"]):
                    if np.isfinite(random):
                        ax.plot([selected, random], [yi, yi], color="#555555", linewidth=0.7, alpha=0.55)
                if metric in {"Recovery accuracy AUC", "Pairwise dominance AUC"}:
                    add_percent_vs_random_labels(
                        ax,
                        sub["selected_value"],
                        sub["random_value"],
                        y,
                        fontsize=6.5,
                    )

            labels = [
                f"{label} ({int(n_tracks)})"
                if space == "Encoded mean" and int(n_tracks) != 5
                else label
                for label, n_tracks in zip(sub["method_label"], sub["n_tracks"])
            ]
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlim(xmin, xmax)
            ax.grid(axis="x", color="#E5E5E5", linewidth=0.6, alpha=0.8)
            if row_idx == 0:
                ax.set_title(space)
            ax.set_xlabel(metric)
            if metric in {"Recovery accuracy AUC", "Pairwise dominance AUC"}:
                ax.text(
                    0.99,
                    1.02,
                    "% vs random",
                    ha="right",
                    va="bottom",
                    fontsize=6.5,
                    color="#555555",
                    transform=ax.transAxes,
                )

    fig.suptitle(
        "SOTA feature-method sweep: raw vs encoded evaluation metrics",
        fontsize=12,
        fontweight="bold",
    )
    fig.legend(
        handles=comparison_legend_handles(),
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        fontsize=7,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "feature_method_sweep_raw_vs_encoded_metric_panels.pdf"
    png = out_dir / "feature_method_sweep_raw_vs_encoded_metric_panels.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def plot_space_curves(curves: pd.DataFrame, out_dir: Path) -> list[Path]:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True, sharey=True)

    for ax, space in zip(axes, ["Raw", "Encoded mean"]):
        sub = curves[curves["eval_space"] == space].copy()
        for method_id in METHOD_ORDER:
            m = sub[sub["method_id"] == method_id]
            if m.empty:
                continue
            color = method_color(method_id)
            for subset_type, linestyle, alpha, linewidth in [
                ("selected", "-", 0.95, 2.0),
                ("random", "--", 0.42, 1.35),
            ]:
                curve = m[m["subset_type"] == subset_type].copy()
                if curve.empty:
                    continue
                if "snr" not in curve:
                    curve["snr"] = 1.0 / curve["noise_mult"].astype(float)
                curve = curve.sort_values("snr")
                label = method_label(method_id) if subset_type == "selected" else None
                accuracy = 1.0 - curve["error_prob"]
                ax.plot(
                    curve["snr"],
                    accuracy,
                    color=color,
                    linestyle=linestyle,
                    alpha=alpha,
                    linewidth=linewidth,
                    label=label,
                )
                if space == "Encoded mean" and "error_prob_sem" in curve:
                    sem = curve["error_prob_sem"].to_numpy(dtype=float)
                    if np.isfinite(sem).any():
                        x = curve["snr"].to_numpy(dtype=float)
                        y = accuracy.to_numpy(dtype=float)
                        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(sem)
                        if valid.any():
                            band_alpha = 0.10 if subset_type == "selected" else 0.05
                            ax.fill_between(
                                x[valid],
                                np.clip(y[valid] - sem[valid], 0.0, 1.0),
                                np.clip(y[valid] + sem[valid], 0.0, 1.0),
                                color=color,
                                alpha=band_alpha,
                                linewidth=0,
                                zorder=1,
                            )
        ax.axvline(EMPIRICAL_SNR, color="#444444", linewidth=0.8, linestyle="-", alpha=0.35, zorder=0)
        ax.set_title(space)
        ax.set_xlabel("Relative SNR")
        ax.set_xscale("log")
        ax.set_xlim(0.009, 11.0)
        ax.set_xticks(SNR_TICKS)
        ax.set_xticklabels(SNR_TICK_LABELS)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.7)
    axes[-1].text(
        1.25,
        0.97,
        "Empirical\nSNR",
        fontsize=7,
        color="#222222",
        ha="left",
        va="top",
        transform=axes[-1].get_xaxis_transform(),
    )

    axes[0].set_ylabel("Model recovery accuracy")
    fig.suptitle("SOTA feature-method sweep: recovery curves", fontsize=12, fontweight="bold")

    present_methods = [method for method in METHOD_ORDER if method in set(curves["method_id"])]
    method_handles = [
        Line2D([0], [0], color=method_color(method), linewidth=2.0, label=method_label(method))
        for method in present_methods
    ]
    style_handles = [
        Line2D([0], [0], color="#222222", linestyle="-", linewidth=1.8, label="Selected"),
        Line2D([0], [0], color="#222222", linestyle="--", linewidth=1.2, alpha=0.55, label="Random"),
    ]
    axes[0].legend(
        handles=method_handles + style_handles,
        frameon=False,
        loc="lower left",
        ncol=2,
        fontsize=7,
        columnspacing=1.0,
        handlelength=2.5,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "feature_method_sweep_raw_vs_encoded_recovery_curves.pdf"
    png = out_dir / "feature_method_sweep_raw_vs_encoded_recovery_curves.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def plot_summary(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    df = df.copy()
    df["plot_label"] = df["method_id"].map(LABELS).fillna(df["method_label"])
    df["selected_recovery_auc_effective_weighted"] = 1.0 - df["selected_auc_effective_weighted"]
    df["random_recovery_auc_effective_weighted"] = 1.0 - df["random_auc_effective_weighted"]
    df = sort_methods(
        df,
        value_col="selected_recovery_auc_effective_weighted",
        ascending=False,
    ).reset_index(drop=True)

    colors = []
    for method_id in df["method_id"]:
        if method_id.startswith("raw_only"):
            colors.append("#4C78A8")
        elif method_id.startswith("raw_enc"):
            colors.append("#59A14F")
        elif method_id.startswith("sub01") or method_id.startswith("paper_effective"):
            colors.append("#F28E2B")
        else:
            colors.append("#777777")

    y = np.arange(len(df))
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.8, 4.8),
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]},
        constrained_layout=True,
    )

    ax = axes[0]
    ax.barh(y, df["selected_recovery_auc_effective_weighted"], color=colors, alpha=0.88, label="Selected")
    ax.scatter(
        df["random_recovery_auc_effective_weighted"],
        y,
        marker="o",
        s=28,
        facecolor="white",
        edgecolor="#333333",
        linewidth=1.1,
        label="Random mean",
        zorder=3,
    )
    for yi, selected, random in zip(
        y, df["selected_recovery_auc_effective_weighted"], df["random_recovery_auc_effective_weighted"]
    ):
        ax.plot([selected, random], [yi, yi], color="#555555", linewidth=0.8, alpha=0.6, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(df["plot_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Strict top-1 recovery accuracy AUC\nhigher is better")
    auc_min = min(
        float(df["selected_recovery_auc_effective_weighted"].min()),
        float(df["random_recovery_auc_effective_weighted"].min()),
    )
    auc_max = max(
        float(df["selected_recovery_auc_effective_weighted"].max()),
        float(df["random_recovery_auc_effective_weighted"].max()),
    )
    ax.set_xlim(max(0.0, auc_min - 0.035), min(1.0, auc_max + 0.025))
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    add_value_labels(ax, df["selected_recovery_auc_effective_weighted"], y)

    ax = axes[1]
    ax.barh(y, df["pairwise_dominance_auc_effective_weighted"], color=colors, alpha=0.88)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xlabel("Pairwise dominance AUC\nhigher is better")
    ax.set_xlim(0.88, 0.96)
    add_value_labels(ax, df["pairwise_dominance_auc_effective_weighted"], y)

    ax = axes[2]
    ax.barh(y, df["mean_margin_auc_effective_weighted"], color=colors, alpha=0.88)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xlabel("Mean margin AUC\nhigher is better")
    ax.set_xlim(0.20, 0.40)
    add_value_labels(ax, df["mean_margin_auc_effective_weighted"], y)

    fig.suptitle("SOTA feature-method sweep", fontsize=12, fontweight="bold")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "feature_method_sweep_summary.pdf"
    png = out_dir / "feature_method_sweep_summary.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def main() -> None:
    global METHOD_ORDER

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=RECOVERY_ROOT / "figures")
    parser.add_argument(
        "--methods",
        default=None,
        help="Optional comma-separated method_id list to plot, in display order.",
    )
    args = parser.parse_args()
    methods = parse_csv_list(args.methods)
    if methods is not None:
        METHOD_ORDER = methods

    df = filter_methods(load_summary(args.run_dir), methods)
    paths = plot_summary(df, args.out_dir)

    discriminability = filter_methods(load_discriminability(args.run_dir), methods)
    curves = aggregate_curves_by_space(discriminability)
    auc_space = aggregate_recovery_auc_by_space(discriminability)
    pairwise = filter_methods(load_pairwise_auc(args.run_dir), methods)
    corr = filter_methods(load_correlation_matrices(args.run_dir), methods)
    model_order = load_model_order(args.run_dir, corr)
    matrix_rows = selected_correlation_matrices_by_space(corr, ordered_present_methods(corr))
    metric_space = pd.concat(
        [
            recovery_auc_as_metric_rows(auc_space),
            aggregate_pairwise_metrics_by_space(pairwise),
            aggregate_attenuated_correlations_by_space(corr),
        ],
        ignore_index=True,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    auc_space.to_csv(args.out_dir / "feature_method_sweep_raw_vs_encoded_recovery_auc_summary.csv", index=False)
    curves.to_csv(args.out_dir / "feature_method_sweep_raw_vs_encoded_recovery_curves.csv", index=False)
    metric_space.to_csv(args.out_dir / "feature_method_sweep_raw_vs_encoded_metrics_summary.csv", index=False)
    matrix_rows.to_csv(
        args.out_dir / "feature_method_sweep_raw_vs_encoded_model_correlation_matrices.csv",
        index=False,
    )
    paths.extend(plot_space_auc(auc_space, args.out_dir))
    paths.extend(plot_space_curves(curves, args.out_dir))
    paths.extend(plot_metric_panels(metric_space, args.out_dir))
    paths.extend(plot_model_correlation_matrices(matrix_rows, model_order, args.out_dir))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
