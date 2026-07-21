#!/usr/bin/env python3
"""Compute two native, unprojected fixed-RSA layer variants.

Variant A evaluates native fixed RSA at the dense layer selected by the
existing mixed-RSA best-on-DeepVision-shared rule.  Variant B selects a layer
with native fixed RSA itself, using all held-out DeepVision shared images and
the pre-specified coarse layer landmarks, then transfers that selected layer to
the CSTIM sets and Vicco baseline.

All feature taps use the dense-sweep aggregation (flatten all non-batch
dimensions).  No feature standardization, response re-standardization, or
random projection is applied.  RDMs use correlation distance and are compared
with Spearman correlation of their upper triangles.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from PIL import Image
from scipy.stats import rankdata
from tqdm import tqdm

# robustness still imports the torchvision <0.13 compatibility module.  Modern
# torchvision removed it, while keeping the referenced helper in torch.hub.
if "torchvision.models.utils" not in sys.modules:
    torchvision_utils = types.ModuleType("torchvision.models.utils")
    torchvision_utils.load_state_dict_from_url = torch.hub.load_state_dict_from_url
    sys.modules["torchvision.models.utils"] = torchvision_utils

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT
from batch_tuning import parse_batch_candidates, parse_batch_size, tune_batch_size
from cstims import paths
from cstims.constants import MODEL_DISPLAY_NAMES, SUBJECTS
from layers_config import MODEL_SOURCE, get_layer_set
from multilayer_extractor import MultiLayerExtractor


SCRIPT = Path(__file__).resolve()
PYTHON = Path("/data/home_roth/miniforge3/bin/python")
PROJECT_ROOT = paths.project_root()
DATA_DIR = LAYER_SWEEP_ROOT / "results"
RUN_ROOT = DATA_DIR / "native_frsa_20260721"
SELECTION_CSV = DATA_DIR / "mrsa_dense_layer_selection_transfer.csv"

_FRSA_SPEC = importlib.util.spec_from_file_location(
    "frsa_srp_reference", SCRIPT.with_name("15_compute_frsa_best_shared_layer_transfer.py")
)
if _FRSA_SPEC is None or _FRSA_SPEC.loader is None:
    raise RuntimeError("Could not load fixed-RSA reference helpers")
_FRSA = importlib.util.module_from_spec(_FRSA_SPEC)
_FRSA_SPEC.loader.exec_module(_FRSA)

RULE_MRSA = "native_at_mrsa_best_shared"
RULE_FRSA = "best_native_frsa_on_shared_configured"


def _load_items(items: list) -> list[Image.Image]:
    images = []
    for item in items:
        if isinstance(item, Image.Image):
            images.append(item)
        else:
            with Image.open(item) as image:
                images.append(image.convert("RGB"))
    return images


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cudnn_status_alloc_failed" in text


def extract_raw_features(
    model: str,
    specs: list[tuple[str, str]],
    items: list,
    *,
    device: str,
    batch_size: int | str,
    batch_candidates: list[int],
) -> dict[str, np.ndarray]:
    """Extract native flattened activations, with no SRP or normalization."""
    extractor = MultiLayerExtractor(model, MODEL_SOURCE[model], specs, device=device)
    try:
        active_batch = batch_size
        if active_batch == "auto":
            probe_n = min(max(batch_candidates), len(items))
            probe = _load_items(items[:probe_n])
            active_batch, _records = tune_batch_size(
                extractor.extract,
                probe,
                candidates=batch_candidates,
                verbose=True,
            )
        chunks = {name: [] for name, _aggregation in specs}
        start = 0
        with tqdm(total=len(items), desc=f"extract native {model}", leave=False) as bar:
            while start < len(items):
                stop = min(start + int(active_batch), len(items))
                batch = _load_items(items[start:stop])
                try:
                    outputs = extractor.extract(batch)
                except Exception as exc:
                    if not _is_oom(exc) or int(active_batch) == 1:
                        raise
                    active_batch = max(1, int(active_batch) // 2)
                    torch.cuda.empty_cache()
                    print(f"[extract] OOM; retrying with batch_size={active_batch}", flush=True)
                    continue
                for name, _aggregation in specs:
                    array = np.asarray(outputs[name], dtype=np.float32)
                    if array.ndim != 2:
                        array = array.reshape(array.shape[0], -1)
                    chunks[name].append(array)
                start = stop
                bar.update(len(batch))
                del outputs, batch
        return {
            name: np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)
            for name, parts in chunks.items()
        }
    finally:
        extractor.free()


def probe_feature_dims(
    model: str,
    specs: list[tuple[str, str]],
    item,
    *,
    device: str,
) -> dict[str, int]:
    extractor = MultiLayerExtractor(model, MODEL_SOURCE[model], specs, device=device)
    try:
        output = extractor.extract(_load_items([item]))
        return {name: int(np.asarray(output[name]).reshape(1, -1).shape[1]) for name, _ in specs}
    finally:
        extractor.free()


def memory_chunks(
    specs: list[tuple[str, str]],
    dims: dict[str, int],
    *,
    n_images: int,
    max_feature_gb: float,
) -> list[list[tuple[str, str]]]:
    budget = max(1, int(max_feature_gb * 1024**3))
    chunks: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    current_bytes = 0
    for spec in specs:
        layer_bytes = n_images * dims[spec[0]] * np.dtype(np.float32).itemsize
        if current and current_bytes + layer_bytes > budget:
            chunks.append(current)
            current = []
            current_bytes = 0
        current.append(spec)
        current_bytes += layer_bytes
    if current:
        chunks.append(current)
    return chunks


def correlation_rdm(features: np.ndarray, device: str) -> np.ndarray:
    """Correlation-distance RDM using GPU matrix multiplication when possible."""
    x_np = np.ascontiguousarray(features, dtype=np.float32)
    try:
        with torch.inference_mode():
            x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
            x = x - x.mean(dim=1, keepdim=True)
            norms = torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min_(1e-12)
            x = x / norms
            similarity = x @ x.T
            rdm = (1.0 - similarity).clamp_(0.0, 2.0).cpu().numpy().astype(np.float32)
            rdm = (rdm + rdm.T) * np.float32(0.5)
            np.fill_diagonal(rdm, 0.0)
            del x, norms, similarity
        return rdm
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        x = x_np.astype(np.float32, copy=True)
        x -= x.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        x /= norms
        rdm = (1.0 - x @ x.T).astype(np.float32)
        rdm = (rdm + rdm.T) * np.float32(0.5)
        np.fill_diagonal(rdm, 0.0)
        return rdm


def ranked_upper(rdm: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(rdm.shape[0], k=1)
    return rankdata(rdm[idx], method="average").astype(np.float32)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    xm = x - x.mean(dtype=np.float64)
    ym = y - y.mean(dtype=np.float64)
    den = float(np.sqrt(np.dot(xm, xm) * np.dot(ym, ym)))
    if den <= 0:
        return float("nan")
    return float(np.dot(xm, ym) / den)


def load_shared_items(max_images: int | None) -> list[str]:
    csv_path = paths.deepvision_cache_root() / "image_sets/deepvision_shared.csv"
    image_dir = paths.deepvision_cache_root() / "image_sets/deepvision_shared"
    frame = pd.read_csv(csv_path)
    result = [(image_dir / name).as_posix() for name in frame["image_name"]]
    return result if max_images is None else result[:max_images]


def load_shared_brain_ranks(subjects: list[str], n_images: int, device: str) -> dict[str, np.ndarray]:
    result = {}
    cache_root = paths.deepvision_cache_root()
    for subject in subjects:
        root = cache_root / "voxel_sets/deepvision_shared_visual_cve0p20/finalinterp" / subject
        betas = np.load(root / "voxel_betas.npy").astype(np.float32)
        with np.load(root / "brain_space_arrays.npz") as payload:
            hlvis = np.asarray(payload["hlvis_mask"], dtype=bool)
        features = np.ascontiguousarray(betas[hlvis, :n_images].T, dtype=np.float32)
        result[subject] = ranked_upper(correlation_rdm(features, device))
        del betas, features
    return result


def load_mrsa_selections(subjects: list[str], models: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(SELECTION_CSV)
    frame = frame[
        frame["selection_rule"].eq("best_on_shared")
        & frame["selection_model_set"].eq("deepvision_shared")
        & frame["subject"].isin(subjects)
        & frame["model"].isin(models)
    ].copy()
    keep = [
        "subject", "model", "display_name", "selected_layer",
        "selected_layer_index", "selected_layer_frac", "selection_mrsa",
    ]
    frame = frame[keep].drop_duplicates()
    counts = frame.groupby(["subject", "model"]).size()
    if not counts.eq(1).all():
        raise RuntimeError("Mixed-RSA selection table is not unique per subject/model")
    return frame


def selection_candidate_specs(model: str) -> list[tuple[str, str]]:
    configured_names = [name for name, _ in get_layer_set("configured")[model]]
    dense_map = dict(get_layer_set("dense")[model])
    missing = [name for name in configured_names if name not in dense_map]
    if missing:
        raise RuntimeError(f"{model}: configured layers absent from dense map: {missing}")
    return [(name, dense_map[name]) for name in configured_names]


def score_native_shared_candidates(
    model: str,
    subjects: list[str],
    shared_items: list,
    brain_ranks: dict[str, np.ndarray],
    *,
    device: str,
    batch_size: int | str,
    batch_candidates: list[int],
    max_feature_gb: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = selection_candidate_specs(model)
    dims = probe_feature_dims(model, specs, shared_items[0], device=device)
    chunks = memory_chunks(specs, dims, n_images=len(shared_items), max_feature_gb=max_feature_gb)
    rows = []
    index_map = {name: idx for idx, (name, _agg) in enumerate(specs)}
    print(f"[selection] {model}: {len(specs)} layers in {len(chunks)} memory chunks", flush=True)
    for chunk in chunks:
        arrays = extract_raw_features(
            model, chunk, shared_items, device=device,
            batch_size=batch_size, batch_candidates=batch_candidates,
        )
        for layer, aggregation in chunk:
            rdm_rank = ranked_upper(correlation_rdm(arrays[layer], device))
            for subject in subjects:
                rows.append({
                    "subject": subject,
                    "model": model,
                    "display_name": MODEL_DISPLAY_NAMES.get(model, model),
                    "candidate_layer": layer,
                    "candidate_layer_index": index_map[layer],
                    "n_candidate_layers": len(specs),
                    "candidate_layer_frac": index_map[layer] / max(1, len(specs) - 1),
                    "aggregation": aggregation,
                    "original_feature_dim": dims[layer],
                    "n_selection_stimuli": len(shared_items),
                    "selection_frsa": pearson_r(rdm_rank, brain_ranks[subject]),
                    "selection_rule": RULE_FRSA,
                    "selection_model_set": "deepvision_shared_full",
                    "feature_space": "native_unprojected",
                    "distance_metric": "correlation",
                    "rsa_metric": "spearman",
                })
            del rdm_rank
        del arrays
    scores = pd.DataFrame(rows)
    selected_rows = []
    for subject in subjects:
        subset = scores[scores.subject.eq(subject)].sort_values(
            ["selection_frsa", "candidate_layer_index"],
            ascending=[False, True],
            kind="stable",
        )
        if subset.empty or not np.isfinite(subset.iloc[0].selection_frsa):
            raise RuntimeError(f"{subject}/{model}: no finite native shared fRSA candidate")
        winner = subset.iloc[0]
        selected_rows.append({
            "subject": subject,
            "model": model,
            "display_name": winner.display_name,
            "selected_layer": winner.candidate_layer,
            "selected_layer_index": int(winner.candidate_layer_index),
            "selected_layer_frac": float(winner.candidate_layer_frac),
            "selection_frsa": float(winner.selection_frsa),
            "original_feature_dim": int(winner.original_feature_dim),
            "selection_rule": RULE_FRSA,
            "selection_model_set": "deepvision_shared_full",
        })
    return scores, pd.DataFrame(selected_rows)


def _score_one_bootstrap(rdm: np.ndarray, idx: np.ndarray, brain_rank: np.ndarray) -> float:
    subset = rdm[np.ix_(idx, idx)]
    return pearson_r(ranked_upper(subset), brain_rank)


def summarize(values: list[float]) -> tuple[int, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return 0, float("nan"), float("nan")
    sem = array.std(ddof=1) / np.sqrt(len(array)) if len(array) > 1 else float("nan")
    return int(len(array)), float(array.mean()), float(sem)


def score_selected_layers(
    model: str,
    selections: pd.DataFrame,
    subjects: list[str],
    eval_items: list,
    cstim_slices: dict,
    ranks_by_subject: dict,
    *,
    rule: str,
    device: str,
    batch_size: int | str,
    batch_candidates: list[int],
    max_feature_gb: float,
    n_score_jobs: int,
) -> pd.DataFrame:
    dense_specs = dict(get_layer_set("dense")[model])
    layers = selections.loc[selections.model.eq(model), "selected_layer"].drop_duplicates().tolist()
    specs = [(layer, dense_specs[layer]) for layer in layers]
    dims = probe_feature_dims(model, specs, eval_items[0], device=device)
    chunks = memory_chunks(specs, dims, n_images=len(eval_items), max_feature_gb=max_feature_gb)
    rows = []
    selection_map = selections[selections.model.eq(model)].set_index("subject")
    for chunk in chunks:
        arrays = extract_raw_features(
            model, chunk, eval_items, device=device,
            batch_size=batch_size, batch_candidates=batch_candidates,
        )
        for layer, aggregation in chunk:
            layer_subjects = [
                subject for subject in subjects
                if subject in selection_map.index
                and str(selection_map.loc[subject, "selected_layer"]) == layer
            ]
            for subject in layer_subjects:
                selection = selection_map.loc[subject]
                ranks = ranks_by_subject[subject]
                for group, brain_rank in ranks["cstim_ranks"].items():
                    if group not in cstim_slices:
                        continue
                    file_idx = ranks["group_stim_idx"].get(group)
                    if file_idx is None or len(file_idx) == 0:
                        continue
                    feats = arrays[layer][cstim_slices[group]][file_idx]
                    value = pearson_r(ranked_upper(correlation_rdm(feats, device)), brain_rank)
                    row = {
                        "subject": subject,
                        "model": model,
                        "display_name": MODEL_DISPLAY_NAMES.get(model, model),
                        "selection_rule": rule,
                        "selection_model_set": str(selection["selection_model_set"]),
                        "selected_layer": layer,
                        "selected_layer_index": int(selection["selected_layer_index"]),
                        "selected_layer_frac": float(selection["selected_layer_frac"]),
                        "eval_target": "cstim",
                        "eval_model_set": group,
                        "n_bootstraps": 1,
                        "n_stimuli": int(len(file_idx)),
                        "frsa_mean": value,
                        "frsa_sem": np.nan,
                        "feature_space": "native_unprojected",
                        "feature_aggregation": aggregation,
                        "original_feature_dim": dims[layer],
                        "distance_metric": "correlation",
                        "rsa_metric": "spearman",
                    }
                    if "selection_mrsa" in selection.index:
                        row["selection_mrsa"] = float(selection["selection_mrsa"])
                    if "selection_frsa" in selection.index:
                        row["selection_frsa"] = float(selection["selection_frsa"])
                    rows.append(row)

                if "vicco" in cstim_slices and ranks["vicco_bootstrap"]:
                    file_idx = ranks["group_stim_idx"].get("vicco")
                    feats = arrays[layer][cstim_slices["vicco"]][file_idx]
                    rdm = correlation_rdm(feats, device)
                    if n_score_jobs > 1:
                        values = Parallel(n_jobs=n_score_jobs, prefer="threads")(
                            delayed(_score_one_bootstrap)(rdm, idx, ranks["vicco_ranks"][boot_idx])
                            for boot_idx, idx in enumerate(ranks["vicco_bootstrap"])
                        )
                    else:
                        values = [
                            _score_one_bootstrap(rdm, idx, ranks["vicco_ranks"][boot_idx])
                            for boot_idx, idx in enumerate(ranks["vicco_bootstrap"])
                        ]
                    n_boot, mean, sem = summarize(values)
                    row = {
                        "subject": subject,
                        "model": model,
                        "display_name": MODEL_DISPLAY_NAMES.get(model, model),
                        "selection_rule": rule,
                        "selection_model_set": str(selection["selection_model_set"]),
                        "selected_layer": layer,
                        "selected_layer_index": int(selection["selected_layer_index"]),
                        "selected_layer_frac": float(selection["selected_layer_frac"]),
                        "eval_target": "vicco",
                        "eval_model_set": "vicco",
                        "n_bootstraps": n_boot,
                        "n_stimuli": int(ranks["n_vicco_sample"]),
                        "frsa_mean": mean,
                        "frsa_sem": sem,
                        "feature_space": "native_unprojected",
                        "feature_aggregation": aggregation,
                        "original_feature_dim": dims[layer],
                        "distance_metric": "correlation",
                        "rsa_metric": "spearman",
                    }
                    if "selection_mrsa" in selection.index:
                        row["selection_mrsa"] = float(selection["selection_mrsa"])
                    if "selection_frsa" in selection.index:
                        row["selection_frsa"] = float(selection["selection_frsa"])
                    rows.append(row)
        del arrays
    return pd.DataFrame(rows)


def part_path(out_dir: Path, category: str, model: str) -> Path:
    return out_dir / "parts" / category / f"{model}.csv"


def compute_model(model: str, args, subjects: list[str]) -> dict:
    dense = get_layer_set("dense")
    if model not in dense:
        raise ValueError(f"Unknown layer-sweep model: {model}")
    eval_items, cstim_slices = _FRSA.load_eval_items()
    if args.groups:
        requested = set(args.groups.split(",")) | {"vicco"}
        # Preserve the combined list and slices, but only retain requested rank groups below.
    else:
        requested = None
    ranks = {subject: _FRSA.load_subject_rank_bundle(subject, args.n_vicco_boot) for subject in subjects}
    if requested is not None:
        for bundle in ranks.values():
            bundle["cstim_ranks"] = {
                key: value for key, value in bundle["cstim_ranks"].items() if key in requested
            }

    result = {"model": model, "rules": []}
    if args.rule in ("both", RULE_MRSA):
        out = part_path(args.out_dir, RULE_MRSA, model)
        if not out.exists() or args.overwrite:
            selections = load_mrsa_selections(subjects, [model]).copy()
            selections["selection_rule"] = RULE_MRSA
            selections["selection_model_set"] = "deepvision_shared_mrsa"
            frame = score_selected_layers(
                model, selections, subjects, eval_items, cstim_slices, ranks,
                rule=RULE_MRSA, device=args.device, batch_size=args.batch_size,
                batch_candidates=args.batch_candidates,
                max_feature_gb=args.max_feature_gb, n_score_jobs=args.n_score_jobs,
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(out, index=False)
        result["rules"].append({"rule": RULE_MRSA, "output": str(out)})

    if args.rule in ("both", RULE_FRSA):
        out = part_path(args.out_dir, RULE_FRSA, model)
        selection_out = part_path(args.out_dir, "native_shared_candidate_scores", model)
        winner_out = part_path(args.out_dir, "native_shared_selections", model)
        if not out.exists() or args.overwrite:
            shared_items = load_shared_items(args.max_shared_images)
            brain_ranks = load_shared_brain_ranks(subjects, len(shared_items), args.device)
            scores, selections = score_native_shared_candidates(
                model, subjects, shared_items, brain_ranks,
                device=args.device, batch_size=args.batch_size,
                batch_candidates=args.batch_candidates, max_feature_gb=args.max_feature_gb,
            )
            frame = score_selected_layers(
                model, selections, subjects, eval_items, cstim_slices, ranks,
                rule=RULE_FRSA, device=args.device, batch_size=args.batch_size,
                batch_candidates=args.batch_candidates,
                max_feature_gb=args.max_feature_gb, n_score_jobs=args.n_score_jobs,
            )
            for path, payload in ((selection_out, scores), (winner_out, selections), (out, frame)):
                path.parent.mkdir(parents=True, exist_ok=True)
                payload.to_csv(path, index=False)
        result["rules"].append({
            "rule": RULE_FRSA, "output": str(out),
            "candidate_scores": str(selection_out), "selections": str(winner_out),
        })
    return result


def validate_variant(frame: pd.DataFrame, models: list[str], subjects: list[str], n_boot: int) -> dict:
    key = ["subject", "model", "eval_target", "eval_model_set"]
    duplicates = int(frame.duplicated(key).sum())
    counts = frame.groupby(["subject", "model"]).size().reindex(
        pd.MultiIndex.from_product([subjects, models], names=["subject", "model"])
    )
    expected_per_cell = 6
    nonfinite = int((~np.isfinite(frame["frsa_mean"])).sum())
    bad_boot = int((frame.loc[frame.eval_target.eq("vicco"), "n_bootstraps"] != n_boot).sum())
    report = {
        "rows": int(len(frame)),
        "expected_rows": len(subjects) * len(models) * expected_per_cell,
        "subjects": frame.groupby("subject")["model"].nunique().reindex(subjects).to_dict(),
        "models": int(frame.model.nunique()),
        "duplicate_keys": duplicates,
        "incomplete_cells": int((counts != expected_per_cell).sum()),
        "nonfinite_scores": nonfinite,
        "bad_vicco_bootstrap_counts": bad_boot,
        "frsa_range": [float(frame.frsa_mean.min()), float(frame.frsa_mean.max())],
    }
    if (
        len(frame) != report["expected_rows"] or duplicates or nonfinite
        or report["incomplete_cells"] or bad_boot
    ):
        raise RuntimeError("Native fRSA validation failed:\n" + json.dumps(report, indent=2))
    return report


def validate_selection_tables(
    candidates: pd.DataFrame,
    winners: pd.DataFrame,
    models: list[str],
    subjects: list[str],
) -> dict:
    candidate_key = ["subject", "model", "candidate_layer"]
    winner_key = ["subject", "model"]
    candidate_counts = candidates.groupby(winner_key).size()
    expected_index = pd.MultiIndex.from_product(
        [subjects, models], names=winner_key
    )
    declared_count_mismatches = int(
        (
            candidates["n_candidate_layers"].astype(int)
            != candidates.groupby(winner_key)["candidate_layer"].transform("size")
        ).sum()
    )

    group_max = candidates.groupby(winner_key)["selection_frsa"].transform("max")
    expected_winners = (
        candidates[candidates["selection_frsa"].eq(group_max)]
        .sort_values(winner_key + ["candidate_layer_index"], kind="stable")
        .drop_duplicates(winner_key, keep="first")
    )
    joined = winners.merge(
        expected_winners[
            winner_key
            + [
                "candidate_layer",
                "candidate_layer_index",
                "selection_frsa",
                "original_feature_dim",
            ]
        ],
        on=winner_key,
        how="left",
        validate="one_to_one",
        suffixes=("_winner", "_expected"),
    )
    winner_mismatches = int(
        (
            joined["selected_layer"].ne(joined["candidate_layer"])
            | joined["selected_layer_index"].ne(joined["candidate_layer_index"])
            | joined["selection_frsa_winner"].ne(joined["selection_frsa_expected"])
            | joined["original_feature_dim_winner"].ne(joined["original_feature_dim_expected"])
        ).sum()
    )
    report = {
        "candidate_rows": int(len(candidates)),
        "winner_rows": int(len(winners)),
        "expected_winner_rows": len(subjects) * len(models),
        "candidate_models": int(candidates["model"].nunique()),
        "winner_models": int(winners["model"].nunique()),
        "candidate_duplicate_keys": int(candidates.duplicated(candidate_key).sum()),
        "winner_duplicate_keys": int(winners.duplicated(winner_key).sum()),
        "incomplete_candidate_cells": int(candidate_counts.reindex(expected_index).isna().sum()),
        "declared_candidate_count_mismatches": declared_count_mismatches,
        "candidate_counts_per_cell": sorted(map(int, candidate_counts.unique())),
        "selection_stimulus_counts": sorted(
            map(int, candidates["n_selection_stimuli"].unique())
        ),
        "nonfinite_candidate_scores": int(
            (~np.isfinite(candidates["selection_frsa"])).sum()
        ),
        "nonpositive_candidate_dimensions": int(
            (candidates["original_feature_dim"] <= 0).sum()
        ),
        "winner_mismatches_from_max_with_earliest_tie_break": winner_mismatches,
    }
    if (
        report["winner_rows"] != report["expected_winner_rows"]
        or report["candidate_models"] != len(models)
        or report["winner_models"] != len(models)
        or report["candidate_duplicate_keys"]
        or report["winner_duplicate_keys"]
        or report["incomplete_candidate_cells"]
        or report["declared_candidate_count_mismatches"]
        or report["selection_stimulus_counts"] != [1492]
        or report["nonfinite_candidate_scores"]
        or report["nonpositive_candidate_dimensions"]
        or report["winner_mismatches_from_max_with_earliest_tie_break"]
    ):
        raise RuntimeError(
            "Native fixed-RSA selection validation failed:\n"
            + json.dumps(report, indent=2)
        )
    return report


def merge_parts(
    out_dir: Path,
    models: list[str],
    subjects: list[str],
    n_boot: int,
    rule: str = "both",
) -> dict:
    outputs = {}
    rules = (RULE_MRSA, RULE_FRSA) if rule == "both" else (rule,)
    for selected_rule in rules:
        files = [part_path(out_dir, selected_rule, model) for model in models]
        missing = [str(path) for path in files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing {selected_rule} parts: {missing}")
        frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
        frame = frame.sort_values(
            ["subject", "model", "eval_target", "eval_model_set"], kind="stable"
        )
        out = out_dir / f"frsa_{selected_rule}.csv"
        frame.to_csv(out, index=False)
        report = validate_variant(frame, models, subjects, n_boot)
        outputs[selected_rule] = {"output": str(out), "validation": report}
    if RULE_FRSA in rules:
        candidate_files = [
            part_path(out_dir, "native_shared_candidate_scores", model)
            for model in models
        ]
        winner_files = [
            part_path(out_dir, "native_shared_selections", model)
            for model in models
        ]
        missing = [str(path) for path in (*candidate_files, *winner_files) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing native fixed-RSA selection parts: {missing}")
        candidates = pd.concat([pd.read_csv(path) for path in candidate_files], ignore_index=True)
        winners = pd.concat([pd.read_csv(path) for path in winner_files], ignore_index=True)
        selection_validation = validate_selection_tables(
            candidates, winners, models, subjects
        )
        candidates.to_csv(out_dir / "native_frsa_shared_configured_layer_scores.csv", index=False)
        winners.to_csv(out_dir / "native_frsa_shared_configured_selections.csv", index=False)
        outputs["selection_validation"] = selection_validation
    (out_dir / "validation.json").write_text(json.dumps(outputs, indent=2) + "\n")
    return outputs


def model_weight(model: str) -> float:
    name = model.lower()
    if "so400m" in name:
        return 5.0
    if "vit_l" in name or "vit_large" in name or "convnext_base" in name:
        return 3.0
    if "vgg" in name or "cornet" in name:
        return 2.0
    return 1.5


def shard_models(models: list[str], n_gpus: int) -> list[list[str]]:
    lanes = [[] for _ in range(n_gpus)]
    loads = [0.0] * n_gpus
    for model in sorted(models, key=model_weight, reverse=True):
        lane = int(np.argmin(loads))
        lanes[lane].append(model)
        loads[lane] += model_weight(model)
    return lanes


def worker_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    conda_lib = "/data/home_roth/miniforge3/lib"
    old_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{conda_lib}:{old_ld}" if old_ld else conda_lib
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    return env


def launch_worker(gpu: int, models: list[str], args) -> dict:
    log_path = args.out_dir / "logs" / f"worker_gpu{gpu}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON), str(SCRIPT), "--mode", "worker",
        "--models", *models,
        "--subjects", *args.subjects,
        "--rule", args.rule,
        "--device", "cuda:0",
        "--batch-size", str(args.batch_size),
        "--batch-candidates", ",".join(map(str, args.batch_candidates)),
        "--n-vicco-boot", str(args.n_vicco_boot),
        "--n-score-jobs", str(args.n_score_jobs),
        "--max-feature-gb", str(args.max_feature_gb),
        "--out-dir", str(args.out_dir),
    ]
    if args.groups:
        cmd.extend(["--groups", args.groups])
    if args.max_shared_images is not None:
        cmd.extend(["--max-shared-images", str(args.max_shared_images)])
    if args.overwrite:
        cmd.append("--overwrite")
    start = time.perf_counter()
    with log_path.open("a", buffering=1) as log:
        log.write("\nCOMMAND " + " ".join(cmd) + "\n")
        proc = subprocess.run(
            cmd, cwd=PROJECT_ROOT, env=worker_env(gpu), stdout=log,
            stderr=subprocess.STDOUT, text=True, check=False,
        )
    elapsed = time.perf_counter() - start
    if proc.returncode:
        raise RuntimeError(f"Native fRSA worker {gpu} failed; see {log_path}")
    return {"gpu": gpu, "models": models, "elapsed_sec": elapsed, "log": str(log_path)}


def protocol_manifest() -> dict:
    return {
        "feature_standardization": "None. Native activation columns are not standardized.",
        "response_standardization": (
            "No new standardization during RSA. Cached DeepVision/CSTIM beta estimates retain "
            "their upstream session-wise voxel standardization."
        ),
        "layer_selection_variant_a": (
            "Per-subject dense layer previously selected by mixed RSA on held-out DeepVision shared images."
        ),
        "layer_selection_variant_b": (
            "Per-subject layer maximizing native fixed RSA on all held-out DeepVision shared images, "
            "among the pre-specified coarse configured layer landmarks. Earliest candidate wins an exact tie."
        ),
        "dimensionality": "Full native flattened activation dimensionality; no SRP or PCA.",
        "aggregation": (
            "Flatten all non-batch dimensions for both variants, matching the dense mixed-RSA sweep."
        ),
        "distance_metric": (
            "Correlation distance across the feature/voxel dimension for each image pair. "
            "The row centering and L2 normalization intrinsic to correlation distance is not "
            "dataset-level feature standardization."
        ),
        "rsa_metric": "Spearman correlation of upper-triangular RDM entries with average ranks.",
        "model_identifiers": "The 20 identifiers in layers_config.py, unchanged.",
        "missing_cell_handling": "No imputation or dropping; merging fails on any absent or duplicate cell.",
        "aggregation_across_resamples": (
            "Five CSTIM set scores are single full-set values. Vicco uses 1,000 deterministic "
            "size-100 bootstrap samples and reports their arithmetic mean and SEM."
        ),
        "selection_evaluation_separation": (
            "Layer selection uses DeepVision shared images; transferred scores use CSTIM/Vicco images."
        ),
        "preserved_srp_result": str(DATA_DIR / "frsa_best_shared_layer_transfer.csv"),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["worker", "full", "merge", "validate-runtime"], default="worker")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--subjects", nargs="*", default=list(SUBJECTS))
    parser.add_argument("--rule", choices=["both", RULE_MRSA, RULE_FRSA], default="both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-gpus", type=int, default=8)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--batch-candidates", default="1,2,4,8")
    parser.add_argument("--n-vicco-boot", type=int, default=1000)
    parser.add_argument("--n-score-jobs", type=int, default=4)
    parser.add_argument("--max-feature-gb", type=float, default=10.0)
    parser.add_argument("--max-shared-images", type=int, default=None)
    parser.add_argument("--groups", default=None)
    parser.add_argument("--out-dir", type=Path, default=RUN_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.batch_size = parse_batch_size(args.batch_size)
    args.batch_candidates = parse_batch_candidates(args.batch_candidates)
    if args.n_gpus < 1:
        raise ValueError("--n-gpus must be positive")
    return args


def main() -> None:
    args = parse_args()
    dense = get_layer_set("dense")
    models = args.models or list(dense)
    unknown = sorted(set(models) - set(dense))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "protocol_assumptions.json").write_text(
        json.dumps(protocol_manifest(), indent=2) + "\n"
    )
    if args.mode == "worker":
        rows = []
        for model in models:
            start = time.perf_counter()
            result = compute_model(model, args, args.subjects)
            result["elapsed_sec"] = time.perf_counter() - start
            rows.append(result)
            print(json.dumps(result), flush=True)
        (args.out_dir / f"worker_{os.getpid()}.json").write_text(json.dumps(rows, indent=2) + "\n")
    elif args.mode == "merge":
        print(json.dumps(
            merge_parts(args.out_dir, models, args.subjects, args.n_vicco_boot, args.rule),
            indent=2,
        ))
    elif args.mode == "full":
        lanes = shard_models(models, args.n_gpus)
        plan = [
            {"gpu": gpu, "model": model}
            for gpu, lane in enumerate(lanes) for model in lane
        ]
        pd.DataFrame(plan).to_csv(args.out_dir / "workplan.csv", index=False)
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_gpus) as pool:
            futures = [pool.submit(launch_worker, gpu, lane, args) for gpu, lane in enumerate(lanes)]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        (args.out_dir / "worker_results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(
            merge_parts(args.out_dir, models, args.subjects, args.n_vicco_boot, args.rule),
            indent=2,
        ))
    else:
        # A small real-data run: one model, one subject, 80 held-out shared
        # images, one CSTIM set, and two Vicco bootstrap samples.
        args.out_dir = args.out_dir / "runtime_validation"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        args.max_shared_images = min(args.max_shared_images or 80, 80)
        args.n_vicco_boot = min(args.n_vicco_boot, 2)
        args.groups = "architecture"
        args.subjects = ["sub-05"]
        validation_model = "torchvision_resnet50_imagenet1k_v1"
        start = time.perf_counter()
        result = compute_model(validation_model, args, args.subjects)
        elapsed = time.perf_counter() - start
        report = {
            "model": validation_model,
            "subject": "sub-05",
            "shared_images": args.max_shared_images,
            "cstim_groups": ["architecture"],
            "vicco_bootstraps": args.n_vicco_boot,
            "elapsed_sec": elapsed,
            "result": result,
        }
        (args.out_dir / "native_frsa_runtime_validation.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
