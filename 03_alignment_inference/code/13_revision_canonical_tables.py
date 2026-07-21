#!/usr/bin/env python3
"""Build canonical revision tables for the CSTIMS paper.

This script is intentionally table-first: it consolidates existing RSA,
noise-ceiling, spread, rank, OOD, layer/discriminability, and model-roster
outputs into paper-facing CSV files. Expensive image feature extraction and
RSA recomputation live in their own section scripts.

Outputs in 03_alignment_inference/results:
  - primary_endpoint_summary.csv
  - rank_correlations.csv
  - noise_ceiling_variant_summary.csv
  - posthoc_only_model_summary.csv
  - leave_one_subject_out_summary.csv
  - posthoc_model_family_summary.csv
  - model_roster_full.csv
  - model_roster_broad_summary.csv
  - layer_sweep_paper_summary.csv
  - discriminability_paper_summary.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))

from cstims import constants, paths


DATA = paths.stats_data_dir()
RSA = paths.rsa_data_dir()
PROJECT = paths.project_root()
SUBJECTS = constants.SUBJECTS
MODEL_SETS = ["all_models", "sota", "training_objective", "architecture", "dataset"]
METHOD_LABEL = {"wrsa_transfer": "mixed_RSA", "crsa": "fixed_RSA"}
SCORE_COL = {"wrsa_transfer": "wrsa_transfer", "crsa": "crsa"}
SELECTION_MODELS = set(constants.MODEL_SETS["all_models"])
MODEL_LIST_LARGE = paths.project_root() / "00_stimulus_selection" / "resources" / "model_list_large.csv"

RANK_FIGURE_CODE = (
    PROJECT
    / "01_brain_model_alignment"
    / "code"
    / "rsa_scoring"
    / "figures"
)
sys.path.insert(0, str(RANK_FIGURE_CODE))
from plot_rank_shift import load_scores as load_rank_score_table  # noqa: E402


def sem(x: Iterable[float]) -> float:
    values = pd.Series(list(x), dtype="float64").dropna()
    if len(values) < 2:
        return np.nan
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def load_score_table(method: str) -> pd.DataFrame:
    frames = []
    for subject in SUBJECTS:
        path = RSA / subject / f"{method}_scores.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["subject"] = subject
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_nc_lookup() -> pd.DataFrame:
    nc = pd.read_csv(DATA / "rdm_noise_ceilings.csv")
    rows = []
    for _, r in nc.iterrows():
        group = "same_session" if r["stimulus_type"] == "vicco" else r["group"]
        rows.append(
            {
                "subject": r["subject"],
                "group": group,
                "stimulus_type": r["stimulus_type"],
                "bootstrap_idx": r["bootstrap_idx"],
                "within_rSB": r["noise_ceiling_spearman"],
                "within_sqrt_rSB": math.sqrt(max(r["noise_ceiling_spearman"], 0.0)),
            }
        )
    out = pd.DataFrame(rows)
    # For the same-session baseline, use the subject-level bootstrap mean.
    base = (
        out[out["stimulus_type"] == "vicco"]
        .groupby("subject", as_index=False)
        .agg(within_rSB=("within_rSB", "mean"), within_sqrt_rSB=("within_sqrt_rSB", "mean"))
    )
    base["group"] = "same_session"
    base["stimulus_type"] = "vicco"
    cstim = out[out["stimulus_type"] == "controversial"].copy()
    cstim = cstim[cstim["bootstrap_idx"] == 0]
    return pd.concat([cstim, base], ignore_index=True)


def build_primary_endpoint_summary() -> pd.DataFrame:
    nc_scores = pd.read_csv(DATA / "nc_normalized_scores.csv")
    spread = pd.read_csv(DATA / "spread_statistics.csv")
    rows = []

    for (subject, model_set, method), grp in nc_scores.groupby(["subject", "model_set", "method"]):
        if model_set not in MODEL_SETS:
            continue
        cstim = grp[grp["stimulus_type"] == "controversial"]
        base = grp[grp["stimulus_type"] == "vicco"]
        if cstim.empty or base.empty:
            continue

        # Average baseline bootstraps per model before averaging over models.
        base_model = (
            base.groupby(["model", "display_name"], as_index=False)
            .agg(score=("score", "mean"), nc_normalized=("nc_normalized", "mean"),
                 noise_ceiling=("noise_ceiling", "mean"))
        )
        cstim_model = cstim[["model", "display_name", "score", "nc_normalized", "noise_ceiling"]]
        common = sorted(set(cstim_model["model"]) & set(base_model["model"]))
        if len(common) < 2:
            continue
        cstim_common = cstim_model.set_index("model").loc[common]
        base_common = base_model.set_index("model").loc[common]

        sp = spread[
            (spread["subject"] == subject)
            & (spread["model_set"] == model_set)
            & (spread["method"] == method)
        ]
        spread_ratio = float(sp["median_pairwise_diff_ratio"].iloc[0]) if not sp.empty else np.nan

        rows.append(
            {
                "subject": subject,
                "model_set": model_set,
                "metric": METHOD_LABEL[method],
                "method_source": method,
                "baseline_type": "same_session_unselected",
                "n_models": len(common),
                "score_cstim": float(cstim_common["score"].mean()),
                "score_baseline": float(base_common["score"].mean()),
                "delta": float(cstim_common["score"].mean() - base_common["score"].mean()),
                "score_cstim_NCnorm": float(cstim_common["nc_normalized"].mean()),
                "score_baseline_NCnorm": float(base_common["nc_normalized"].mean()),
                "delta_NCnorm": float(
                    cstim_common["nc_normalized"].mean() - base_common["nc_normalized"].mean()
                ),
                "NC_cstim_mean": float(cstim_common["noise_ceiling"].mean()),
                "NC_baseline_mean": float(base_common["noise_ceiling"].mean()),
                "spread_ratio": spread_ratio,
                "primary_alignment_endpoint": method == "wrsa_transfer",
                "primary_spread_endpoint": method == "wrsa_transfer",
            }
        )

    out = pd.DataFrame(rows).sort_values(["metric", "model_set", "subject"])

    # If the held-out unique pipeline has produced results, append its subject
    # level endpoint summary without making this script depend on that pipeline.
    heldout_path = (
        paths.project_root()
        / "05_controls_and_supplementary"
        / "counterfactual_baselines"
        / "results"
        / "heldout_unique_endpoint_summary.csv"
    )
    if heldout_path.exists():
        heldout = pd.read_csv(heldout_path)
        complete_subjects = heldout["subject"].nunique() >= len(SUBJECTS)
        complete_sets = set(MODEL_SETS).issubset(set(heldout["model_set"].unique()))
        if complete_subjects and complete_sets:
            heldout = heldout[heldout["baseline_type"] != "same_session_unselected"].copy()
            out = pd.concat([out, heldout], ignore_index=True, sort=False)

    out.to_csv(DATA / "primary_endpoint_summary.csv", index=False)
    return out


def build_rank_correlations(methods: Iterable[str] = ("wrsa_transfer", "crsa")) -> pd.DataFrame:
    rank_null = pd.read_csv(DATA / "rank_null.csv") if (DATA / "rank_null.csv").exists() else pd.DataFrame()
    rows = []
    for method in methods:
        scores = load_rank_score_table(method)
        score_col = SCORE_COL[method]
        if scores.empty:
            continue
        for model_set in MODEL_SETS:
            ms = scores[scores["model_set"] == model_set]
            for subject in SUBJECTS:
                sub = ms[ms["subject"] == subject]
                base = sub[sub["stimulus_type"] == "vicco"].groupby("model")[score_col].mean()
                cstim = sub[sub["stimulus_type"] == "controversial"].groupby("model")[score_col].mean()
                common = sorted(set(base.index) & set(cstim.index))
                if len(common) < 3:
                    continue
                rho, p = spearmanr(base.loc[common], cstim.loc[common])
                rows.append(
                    {
                        "aggregation": "subject",
                        "subject": subject,
                        "model_set": model_set,
                        "metric": METHOD_LABEL[method],
                        "method_source": method,
                        "n_models": len(common),
                        "rho_base_to_controversial": float(rho),
                        "p_spearman": float(p),
                    }
                )

            subject_rows = [r for r in rows if r["aggregation"] == "subject"
                            and r["model_set"] == model_set and r["method_source"] == method]
            vals = [r["rho_base_to_controversial"] for r in subject_rows]
            if vals:
                row = {
                    "aggregation": "mean_across_subjects",
                    "subject": "all",
                    "model_set": model_set,
                    "metric": METHOD_LABEL[method],
                    "method_source": method,
                    "n_models": int(subject_rows[0]["n_models"]),
                    "rho_base_to_controversial": float(np.nanmean(vals)),
                    "rho_sem": sem(vals),
                    "p_spearman": np.nan,
                }
                if not rank_null.empty:
                    rn = rank_null[(rank_null["model_set"] == model_set) & (rank_null["method"] == method)]
                    if not rn.empty:
                        for col in ["null_p2.5", "null_p97.5", "p_two_tailed"]:
                            row[col] = float(rn[col].iloc[0])
                rows.append(row)

    out = pd.DataFrame(rows).sort_values(["metric", "model_set", "aggregation", "subject"])
    out.to_csv(DATA / "rank_correlations.csv", index=False)
    return out


def build_noise_ceiling_variant_summary() -> pd.DataFrame:
    rows = []
    nc_lookup = load_nc_lookup()
    between_path = DATA / "between_subject_noise_ceilings.csv"
    if between_path.exists():
        between = pd.read_csv(between_path)
        between = between.rename(columns={"group": "between_group", "stimulus_type": "between_type"})
    else:
        between = pd.DataFrame(columns=["subject", "between_group", "between_type", "nc_mid"])

    score_frames = []
    for method in ["wrsa_transfer", "crsa"]:
        scores = load_score_table(method)
        score_col = SCORE_COL[method]
        if scores.empty:
            continue
        scores = scores[scores["model_set"].isin(MODEL_SETS)].copy()
        scores["baseline_or_stimulus"] = np.where(
            scores["stimulus_type"] == "vicco", "same_session", "controversial"
        )
        # Collapse baseline bootstraps before applying normalization variants.
        grouped = (
            scores.groupby(["subject", "model_set", "model", "baseline_or_stimulus"], as_index=False)
            .agg(score=(score_col, "mean"))
        )
        grouped["metric"] = METHOD_LABEL[method]
        grouped["method_source"] = method
        score_frames.append(grouped)

    score_df = pd.concat(score_frames, ignore_index=True)
    score_df["nc_group"] = np.where(
        score_df["baseline_or_stimulus"] == "same_session", "same_session", score_df["model_set"]
    )
    score_df = score_df.merge(
        nc_lookup[["subject", "group", "within_rSB", "within_sqrt_rSB"]].rename(columns={"group": "nc_group"}),
        on=["subject", "nc_group"],
        how="left",
    )
    score_df["between_group"] = np.where(
        score_df["baseline_or_stimulus"] == "same_session", "vicco", score_df["model_set"]
    )
    score_df["between_type"] = np.where(
        score_df["baseline_or_stimulus"] == "same_session", "vicco", "controversial"
    )
    score_df = score_df.merge(
        between[["subject", "between_group", "between_type", "nc_mid"]].rename(columns={"nc_mid": "between_nc_mid"}),
        on=["subject", "between_group", "between_type"],
        how="left",
    )
    score_df["score_raw"] = score_df["score"]
    score_df["score_within_sqrt_rSB"] = score_df["score"] / score_df["within_sqrt_rSB"]
    score_df["score_within_rSB"] = score_df["score"] / score_df["within_rSB"]
    score_df["score_between_mid"] = score_df["score"] / score_df["between_nc_mid"]

    for (subject, model_set, metric), grp in score_df.groupby(["subject", "model_set", "metric"]):
        cstim = grp[grp["baseline_or_stimulus"] == "controversial"]
        base = grp[grp["baseline_or_stimulus"] == "same_session"]
        if cstim.empty or base.empty:
            continue
        base_model = base.groupby("model", as_index=True).mean(numeric_only=True)
        cstim_model = cstim.groupby("model", as_index=True).mean(numeric_only=True)
        common = sorted(set(base_model.index) & set(cstim_model.index))
        for variant, col in [
            ("none_raw_score", "score_raw"),
            ("within_subject_sqrt_rSB", "score_within_sqrt_rSB"),
            ("within_subject_rSB", "score_within_rSB"),
            ("between_subject_mid", "score_between_mid"),
        ]:
            delta = float(cstim_model.loc[common, col].mean() - base_model.loc[common, col].mean())
            rows.append(
                {
                    "subject": subject,
                    "model_set": model_set,
                    "metric": metric,
                    "normalization_variant": variant,
                    "score_cstim": float(cstim_model.loc[common, col].mean()),
                    "score_baseline": float(base_model.loc[common, col].mean()),
                    "delta": delta,
                    "direction_negative": bool(delta < 0),
                    "n_models": len(common),
                }
            )
    out = pd.DataFrame(rows).sort_values(["metric", "model_set", "subject", "normalization_variant"])
    out.to_csv(DATA / "noise_ceiling_variant_summary.csv", index=False)
    return out


def _model_family(model: str) -> str:
    m = model.lower()
    if "clip" in m or "siglip" in m or "coca" in m:
        return "image_text"
    if "dino" in m or "moco" in m or "barlow" in m or "vicreg" in m or "swav" in m or "simclr" in m:
        return "self_supervised"
    if "robust" in m or "adversarial" in m:
        return "adversarial_robust"
    if "taskonomy" in m or "midas" in m or "detection" in m or "segmentation" in m or "yolo" in m:
        return "task_supervised"
    if "cornet" in m:
        return "biologically_inspired"
    return "supervised_classification"


def _architecture(model: str) -> str:
    m = model.lower()
    if "resnet" in m or "regnet" in m or "resnest" in m or "resnext" in m:
        return "resnet/regnet"
    if "convnext" in m:
        return "convnext"
    if "vit" in m or "beit" in m or "deit" in m or "swin" in m or "xcit" in m or "eva" in m:
        return "vision_transformer"
    if "vgg" in m:
        return "vgg"
    if "efficientnet" in m or "mixnet" in m or "mobilenet" in m or "rexnet" in m:
        return "efficient/mobile_cnn"
    if "cornet" in m:
        return "cornet"
    if "taskonomy" in m or "faster_rcnn" in m or "mask_rcnn" in m or "yolo" in m:
        return "dense_prediction_or_detection"
    return "other"


def _training_data(model: str) -> str:
    m = model.lower()
    if "laion2b" in m or "laion2b" in m:
        return "LAION-2B"
    if "laion400m" in m or "laion400" in m:
        return "LAION-400M"
    if "webli" in m:
        return "WebLI"
    if "metaclip" in m:
        return "MetaCLIP"
    if "dfn" in m:
        return "DFN"
    if "in22k" in m or "imagenet21k" in m or "in21k" in m:
        return "ImageNet-21K/22K"
    if "imagenet1k" in m or "in1k" in m:
        return "ImageNet-1K"
    if "lvd142m" in m:
        return "LVD-142M"
    if "yfcc" in m or "slip" in m:
        return "YFCC-derived"
    return "not specified"


def _objective(model: str) -> str:
    fam = _model_family(model)
    if fam == "image_text":
        return "image-text contrastive/generative"
    if fam == "self_supervised":
        return "self-supervised"
    if fam == "adversarial_robust":
        return "adversarially robust classification"
    if fam == "task_supervised":
        return "dense/task supervision"
    return "supervised classification"


def _membership_string(model: str) -> str:
    sets = [ms for ms, models in constants.MODEL_SETS.items() if model in models and ms != "all_models"]
    return ";".join(sets)


def _feature_dim_from_cache(model: str) -> float:
    # Avoid loading compressed feature arrays while building metadata. Exact
    # dimensions vary by extracted layer and are recoverable from the feature
    # cache when needed; the paper-facing roster keeps this column explicit and
    # leaves unavailable dimensions as NaN rather than doing slow I/O here.
    return np.nan


def build_model_roster_full() -> pd.DataFrame:
    small = pd.read_csv(paths.model_list_csv())
    large = pd.read_csv(MODEL_LIST_LARGE) if MODEL_LIST_LARGE.exists() else pd.DataFrame()
    all_models = pd.concat([small, large], ignore_index=True).drop_duplicates("model")
    scored = set()
    large_scores = RSA / "rsa_large_benchmark_scores.csv"
    if large_scores.exists():
        scored = set(pd.read_csv(large_scores, usecols=["model"])["model"].unique())

    rows = []
    for _, r in all_models.sort_values("model").iterrows():
        model = r["model"]
        rows.append(
            {
                "model": model,
                "display_name": constants.MODEL_DISPLAY_NAMES.get(model, model),
                "architecture": _architecture(model),
                "training_data": _training_data(model),
                "objective": _objective(model),
                "family": _model_family(model),
                "source_package": r.get("source", ""),
                "layer": r.get("layer", ""),
                "pooling_or_aggregation": r.get("aggregation", ""),
                "feature_dimensionality": _feature_dim_from_cache(model),
                "used_during_selection": model in SELECTION_MODELS,
                "selection_model_set_membership": _membership_string(model),
                "available_in_broad_benchmark_scores": model in scored,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "model_roster_full.csv", index=False)

    summary_rows = []
    for family, fam_df in out.groupby("family", dropna=False):
        summary_rows.append(
            {
                "family": family,
                "n_full_roster": int(len(fam_df)),
                "n_broad_benchmark_scored": int(fam_df["available_in_broad_benchmark_scores"].sum()),
                "n_controlled_selection_models": int(fam_df["used_during_selection"].sum()),
            }
        )
    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["n_broad_benchmark_scored", "n_full_roster", "family"], ascending=[False, False, True])
    )
    summary.to_csv(DATA / "model_roster_broad_summary.csv", index=False)
    return out


def build_posthoc_only_model_summary(roster: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = RSA / "rsa_large_benchmark_scores.csv"
    if not path.exists():
        empty = pd.DataFrame()
        empty.to_csv(DATA / "posthoc_only_model_summary.csv", index=False)
        empty.to_csv(DATA / "posthoc_model_family_summary.csv", index=False)
        return empty, empty

    scores = pd.read_csv(path)
    nc = load_nc_lookup()
    family = roster.set_index("model")["family"].to_dict()
    rows = []
    for method, score_col in [("wrsa_transfer", "wrsa_transfer"), ("crsa", "crsa")]:
        for subject in sorted(scores["subject"].unique()):
            for group in MODEL_SETS:
                cstim = scores[(scores["subject"] == subject) & (scores["group"] == group)]
                base = scores[(scores["subject"] == subject) & (scores["stimulus_type"] == "vicco")]
                if cstim.empty or base.empty:
                    continue
                for cohort, model_filter in [
                    ("all_broad_models", lambda m: True),
                    ("posthoc_only", lambda m: m not in SELECTION_MODELS),
                    ("selection_models_only", lambda m: m in SELECTION_MODELS),
                ]:
                    c = cstim[cstim["model"].map(model_filter)]
                    b = base[base["model"].map(model_filter)]
                    common = sorted(set(c["model"]) & set(b["model"]))
                    if len(common) < 2:
                        continue
                    c_model = c[c["model"].isin(common)].groupby("model")[score_col].mean()
                    b_model = b[b["model"].isin(common)].groupby("model")[score_col].mean()
                    nc_c = nc[(nc["subject"] == subject) & (nc["group"] == group)]["within_sqrt_rSB"].mean()
                    nc_b = nc[(nc["subject"] == subject) & (nc["group"] == "same_session")]["within_sqrt_rSB"].mean()
                    rows.append(
                        {
                            "subject": subject,
                            "model_set": group,
                            "metric": METHOD_LABEL[method],
                            "model_cohort": cohort,
                            "n_models": len(common),
                            "score_cstim": float(c_model.mean()),
                            "score_baseline": float(b_model.mean()),
                            "delta": float(c_model.mean() - b_model.mean()),
                            "score_cstim_NCnorm": float(c_model.mean() / nc_c),
                            "score_baseline_NCnorm": float(b_model.mean() / nc_b),
                            "delta_NCnorm": float(c_model.mean() / nc_c - b_model.mean() / nc_b),
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "posthoc_only_model_summary.csv", index=False)

    fam_rows = []
    scores["family"] = scores["model"].map(family).fillna("unknown")
    for method, score_col in [("wrsa_transfer", "wrsa_transfer"), ("crsa", "crsa")]:
        for (subject, group, fam), c in scores[scores["stimulus_type"] == "controversial"].groupby(
            ["subject", "group", "family"]
        ):
            if group not in MODEL_SETS:
                continue
            b = scores[(scores["subject"] == subject) & (scores["stimulus_type"] == "vicco") & (scores["family"] == fam)]
            common = sorted(set(c["model"]) & set(b["model"]))
            if len(common) < 2:
                continue
            c_model = c[c["model"].isin(common)].groupby("model")[score_col].mean()
            b_model = b[b["model"].isin(common)].groupby("model")[score_col].mean()
            fam_rows.append(
                {
                    "subject": subject,
                    "model_set": group,
                    "metric": METHOD_LABEL[method],
                    "family": fam,
                    "n_models": len(common),
                    "delta": float(c_model.mean() - b_model.mean()),
                }
            )
    fam_out = pd.DataFrame(fam_rows)
    fam_out.to_csv(DATA / "posthoc_model_family_summary.csv", index=False)
    return out, fam_out


def build_leave_one_subject_out(primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for left_out in SUBJECTS:
        keep = primary[(primary["subject"].isin(SUBJECTS)) & (primary["subject"] != left_out)]
        for (model_set, metric, baseline), grp in keep.groupby(["model_set", "metric", "baseline_type"]):
            rows.append(
                {
                    "left_out_subject": left_out,
                    "model_set": model_set,
                    "metric": metric,
                    "baseline_type": baseline,
                    "n_subjects": int(grp["subject"].nunique()),
                    "delta_mean": float(grp["delta"].mean()),
                    "delta_NCnorm_mean": float(grp["delta_NCnorm"].mean()),
                    "spread_ratio_mean": float(grp["spread_ratio"].mean()),
                    "all_remaining_delta_NCnorm_negative": bool((grp["delta_NCnorm"] < 0).all()),
                }
            )
    out = pd.DataFrame(rows).sort_values(["metric", "model_set", "baseline_type", "left_out_subject"])
    out.to_csv(DATA / "leave_one_subject_out_summary.csv", index=False)
    return out


def build_existing_output_summaries() -> None:
    layer_files = [
        _PAPER / "11_layer_sweep" / "results" / "layer_drop_summary_subject_avg.csv",
        _PAPER / "11_layer_sweep" / "results" / "mrsa_layer_drop_summary_subject_avg.csv",
        _PAPER / "11_layer_sweep" / "results" / "held_out_rescue.csv",
        _PAPER / "11_layer_sweep" / "results" / "spread_summary.csv",
    ]
    layer_rows = []
    for path in layer_files:
        if path.exists():
            df = pd.read_csv(path)
            layer_rows.append(
                {
                    "source_file": str(path.relative_to(_PAPER)),
                    "n_rows": len(df),
                    "columns": ";".join(df.columns),
                }
            )
    pd.DataFrame(layer_rows).to_csv(DATA / "layer_sweep_paper_summary.csv", index=False)

    discrim_files = [
        _PAPER / "12_discriminability" / "results" / "pair_separation_summary.csv",
        _PAPER / "12_discriminability" / "results" / "sample_efficiency_summary.csv",
        _PAPER / "12_discriminability" / "results" / "top_anchored_equivalence.csv",
    ]
    discrim_rows = []
    for path in discrim_files:
        if path.exists():
            df = pd.read_csv(path)
            discrim_rows.append(
                {
                    "source_file": str(path.relative_to(_PAPER)),
                    "n_rows": len(df),
                    "columns": ";".join(df.columns),
                }
            )
    pd.DataFrame(discrim_rows).to_csv(DATA / "discriminability_paper_summary.csv", index=False)


def validate_outputs() -> None:
    expected = [
        "primary_endpoint_summary.csv",
        "rank_correlations.csv",
        "noise_ceiling_variant_summary.csv",
        "posthoc_only_model_summary.csv",
        "leave_one_subject_out_summary.csv",
        "model_roster_full.csv",
        "model_roster_broad_summary.csv",
    ]
    missing = [f for f in expected if not (DATA / f).exists()]
    if missing:
        raise RuntimeError(f"Missing expected outputs: {missing}")

    primary = pd.read_csv(DATA / "primary_endpoint_summary.csv")
    same = primary[primary["baseline_type"] == "same_session_unselected"]
    expected_cells = len(SUBJECTS) * len(MODEL_SETS) * 2
    if len(same) < expected_cells:
        raise RuntimeError(f"Primary same-session table has {len(same)} rows, expected >= {expected_cells}")
    if same.duplicated(["subject", "model_set", "metric", "baseline_type"]).any():
        raise RuntimeError("Duplicate primary endpoint rows detected")

    rank = pd.read_csv(DATA / "rank_correlations.csv")
    means = rank[rank["aggregation"] == "mean_across_subjects"]
    if not ((means["model_set"] == "all_models") & (means["metric"] == "mixed_RSA")).any():
        raise RuntimeError("Rank correlation table missing all_models mixed_RSA mean row")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rank-only",
        action="store_true",
        help="Rebuild and validate only rank_correlations.csv.",
    )
    parser.add_argument(
        "--rank-methods",
        nargs="+",
        choices=["wrsa_transfer", "crsa"],
        default=["wrsa_transfer", "crsa"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.rank_only:
        print("building rank_correlations.csv", flush=True)
        rank = build_rank_correlations(args.rank_methods)
        means = rank[rank["aggregation"].eq("mean_across_subjects")]
        expected = len(MODEL_SETS) * len(args.rank_methods)
        if len(means) != expected:
            raise RuntimeError(
                f"Rank table has {len(means)} mean rows, expected {expected}"
            )
        if means["n_models"].min() < 3:
            raise RuntimeError("Rank table contains a panel with fewer than three models")
        print(f"rank_correlations.csv: {len(rank)} rows")
        return

    print("building primary_endpoint_summary.csv", flush=True)
    primary = build_primary_endpoint_summary()
    print("building rank_correlations.csv", flush=True)
    rank = build_rank_correlations()
    print("building noise_ceiling_variant_summary.csv", flush=True)
    nc_var = build_noise_ceiling_variant_summary()
    print("building model_roster_full.csv", flush=True)
    roster = build_model_roster_full()
    print("building posthoc_only_model_summary.csv", flush=True)
    posthoc, family = build_posthoc_only_model_summary(roster)
    print("building leave_one_subject_out_summary.csv", flush=True)
    loso = build_leave_one_subject_out(primary)
    print("building existing-output summary manifests", flush=True)
    build_existing_output_summaries()
    print("validating outputs", flush=True)
    validate_outputs()

    print(f"primary_endpoint_summary.csv: {len(primary)} rows")
    print(f"rank_correlations.csv: {len(rank)} rows")
    print(f"noise_ceiling_variant_summary.csv: {len(nc_var)} rows")
    print(f"model_roster_full.csv: {len(roster)} rows")
    print(f"posthoc_only_model_summary.csv: {len(posthoc)} rows")
    print(f"posthoc_model_family_summary.csv: {len(family)} rows")
    print(f"leave_one_subject_out_summary.csv: {len(loso)} rows")


if __name__ == "__main__":
    main()
