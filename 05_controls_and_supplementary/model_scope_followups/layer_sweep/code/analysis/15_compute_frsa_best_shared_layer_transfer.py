#!/usr/bin/env python3
"""Compute fRSA at the mRSA-selected best-on-shared dense layer.

The mRSA layer sweep selects one layer per (subject, model) on DeepVision
shared stimuli. This script evaluates fixed RSA on the cstim model sets and
Vicco baseline at those exact selected layers, using the same dense-layer SRP
feature protocol as the streamed mRSA sweep.

Output:
    results/frsa_best_shared_layer_transfer.csv
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from scipy.stats import rankdata
from tqdm import tqdm

from batch_tuning import parse_batch_candidates, parse_batch_size, tune_batch_size
from config import CSTIM_HDF5_ROOT, MODEL_DISPLAY_NAMES, PAPER_ROOT, SUBJECTS, get_brain_input_dir
from layers_config import MODEL_SOURCE, STIMULUS_SETS, get_layer_set
from multilayer_extractor import MultiLayerExtractor
from srp_utils import SRPProjectorCache, SRP_SEED
from utils import bootstrap_sample_indices, compute_rdm_correlation


SHARE_ROOT = LAYER_SWEEP_ROOT.parents[2]
DATA_DIR = LAYER_SWEEP_ROOT / "results"
CACHE_DIR = LAYER_SWEEP_ROOT / "cache_or_heavy"
LABSHARE_CSTIM_HDF5_ROOT = Path(
    "/data/labshare/_stachelschwein/SSD/jroth/final_cstims_hdf5_files"
)

SELECTION_CSV = DATA_DIR / "mrsa_dense_layer_selection_transfer.csv"
OUT_CSV = DATA_DIR / "frsa_best_shared_layer_transfer.csv"


def _stable_layer_seed(model: str, layer: str, *, base_seed: int = SRP_SEED) -> int:
    digest = hashlib.blake2b(f"{model}::{layer}".encode("utf-8"), digest_size=4).digest()
    return (int.from_bytes(digest, byteorder="little", signed=False) + int(base_seed)) % (2**31 - 1)


def _rdm_upper_vec(rdm: np.ndarray) -> np.ndarray:
    return rdm[np.triu_indices(rdm.shape[0], k=1)]


def _ranked_rdm(features: np.ndarray) -> np.ndarray:
    return rankdata(_rdm_upper_vec(compute_rdm_correlation(features)), method="average").astype(
        np.float32
    )


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    xm = x - x.mean()
    ym = y - y.mean()
    den = float(np.sqrt(np.dot(xm, xm) * np.dot(ym, ym)))
    if den <= 0:
        return float("nan")
    return float(np.dot(xm, ym) / den)


def _atomic_savez_compressed(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def _load_cstim_images(group: str):
    folder_group = group
    if group == "architecture":
        folder_group = "dataset"
    elif group == "dataset":
        folder_group = "architecture"

    if folder_group == "vicco":
        img_dir = CSTIM_HDF5_ROOT / "shared_vicco"
    else:
        img_dir = CSTIM_HDF5_ROOT / folder_group

    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not img_files and folder_group != "vicco":
        fallback_dirs = (
            SHARE_ROOT
            / "00_stimulus_selection"
            / "decision_checks"
            / "selection_evaluation"
            / "results"
            / folder_group
            / "images",
            PAPER_ROOT / "00_selection_evaluation" / "data" / folder_group / "images",
            PAPER_ROOT / "00_selection_evaluation" / "results" / folder_group / "images",
        )
        for fallback_dir in fallback_dirs:
            img_files = sorted(list(fallback_dir.glob("*.jpg")) + list(fallback_dir.glob("*.png")))
            if img_files:
                break
    if not img_files and folder_group == "vicco":
        img_dir = LABSHARE_CSTIM_HDF5_ROOT / "shared_vicco"
        img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

    images = []
    for path in img_files:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images, [p.name for p in img_files]


def load_eval_items():
    eval_items = []
    cstim_slices = {}
    counts = {}
    for group in STIMULUS_SETS:
        images, _filenames = _load_cstim_images(group)
        start = len(eval_items)
        eval_items.extend(images)
        cstim_slices[group] = slice(start, len(eval_items))
        counts[group] = len(images)
    print("[eval] " + ", ".join(f"{k}={v}" for k, v in counts.items()), flush=True)
    return eval_items, cstim_slices


def _load_cstim_subject_indices(subject: str):
    d = get_brain_input_dir(subject)
    betas = np.load(d / "cstim_betas_averaged.npz", allow_pickle=True)
    voxel = np.load(d / "voxel_metadata.npz", allow_pickle=True)
    stim_info = pd.read_csv(d / "cstim_stimulus_info.csv")
    hlvis = np.asarray(voxel["hlvis_mask"], dtype=bool)
    betas_hlvis = np.ascontiguousarray(betas["betas"][hlvis, :], dtype=np.float32)
    key_to_idx = {k: i for i, k in enumerate(betas["stim_keys"])}

    group_indices = {}
    group_stim_idx = {}
    for group in sorted(stim_info["group"].unique()):
        mask = stim_info["group"].eq(group)
        group_indices[group] = np.array(
            [key_to_idx[k] for k in stim_info.loc[mask, "stim_key"].values],
            dtype=int,
        )
        idx = stim_info.loc[mask, "stim_idx"].values.astype(int)
        group_stim_idx[group] = idx - 1 if group == "vicco" else idx
    return betas_hlvis, group_indices, group_stim_idx


def load_cstim_subject_ranks(subject: str, n_vicco_boot: int):
    cache_dir = CACHE_DIR / "brain_ranks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"cstim_{subject}_vicco{n_vicco_boot}.npz"
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=True) as z:
                groups = [str(g) for g in z["cstim_groups"].tolist()]
                ranks = z["cstim_ranks"].astype(np.float32)
                return {
                    "group_stim_idx": {},  # filled below to avoid stale index assumptions
                    "cstim_ranks": {g: ranks[i] for i, g in enumerate(groups)},
                    "vicco_bootstrap": [idx.astype(int) for idx in z["vicco_bootstrap"]],
                    "vicco_ranks": [r.astype(np.float32) for r in z["vicco_ranks"]],
                    "n_vicco_sample": int(np.asarray(z["n_vicco_sample"]).item()),
                }
        except Exception:
            pass

    betas_hlvis, group_indices, group_stim_idx = _load_cstim_subject_indices(subject)
    cstim_ranks = {}
    for group, brain_idx in group_indices.items():
        if group == "vicco":
            continue
        cstim_ranks[group] = _ranked_rdm(betas_hlvis[:, brain_idx].T)

    vicco_boot, vicco_ranks, n_vicco_sample = [], [], 0
    if "vicco" in group_indices:
        n_vicco = len(group_indices["vicco"])
        n_vicco_sample = min(100, n_vicco)
        vicco_boot = bootstrap_sample_indices(
            n_vicco, n_vicco_sample, n_bootstrap=n_vicco_boot, seed=0
        )
        vicco_brain_idx = group_indices["vicco"]
        for idx in tqdm(vicco_boot, desc=f"brain vicco ranks {subject}", leave=False):
            vicco_ranks.append(_ranked_rdm(betas_hlvis[:, vicco_brain_idx[idx]].T))

    _atomic_savez_compressed(
        cache_path,
        cstim_groups=np.array(list(cstim_ranks.keys()), dtype=object),
        cstim_ranks=np.stack(list(cstim_ranks.values()), axis=0).astype(np.float32)
        if cstim_ranks
        else np.empty((0, 0), dtype=np.float32),
        vicco_bootstrap=np.stack(vicco_boot, axis=0).astype(np.int32)
        if vicco_boot
        else np.empty((0, 0), dtype=np.int32),
        vicco_ranks=np.stack(vicco_ranks, axis=0).astype(np.float32)
        if vicco_ranks
        else np.empty((0, 0), dtype=np.float32),
        n_vicco_sample=np.array(n_vicco_sample, dtype=np.int32),
    )
    return {
        "group_stim_idx": group_stim_idx,
        "cstim_ranks": cstim_ranks,
        "vicco_bootstrap": vicco_boot,
        "vicco_ranks": vicco_ranks,
        "n_vicco_sample": n_vicco_sample,
    }


def load_subject_rank_bundle(subject: str, n_vicco_boot: int):
    ranks = load_cstim_subject_ranks(subject, n_vicco_boot)
    _betas, _group_indices, group_stim_idx = _load_cstim_subject_indices(subject)
    ranks["group_stim_idx"] = group_stim_idx
    return ranks


def extract_reduced_features(extractor, items, *, batch_size: int, model: str):
    names = [name for name, _ in extractor.layers]
    chunks = {name: [] for name in names}
    srp_caches = {
        name: SRPProjectorCache(seed=_stable_layer_seed(model, name))
        for name in names
    }
    for start in tqdm(range(0, len(items), batch_size), desc=f"extract {model}", leave=False):
        batch = items[start:start + batch_size]
        raw = extractor.extract(batch)
        for name in names:
            reduced, _meta = srp_caches[name].transform(raw[name])
            chunks[name].append(reduced)
        del raw, batch
    return {
        name: np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)
        for name, parts in chunks.items()
    }


def score_feature_rdm(features: np.ndarray, brain_rank: np.ndarray) -> float:
    return _pearson_r(_ranked_rdm(features), brain_rank)


def score_bootstrap(boot_idx, idx, features_vicco, brain_rank):
    return score_feature_rdm(features_vicco[idx], brain_rank)


def summarize(vals) -> tuple[int, float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0, float("nan"), float("nan")
    sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else float("nan")
    return int(len(vals)), float(vals.mean()), float(sem)


def load_selections(subjects: list[str], models: list[str]) -> pd.DataFrame:
    sel = pd.read_csv(SELECTION_CSV)
    sel = sel[
        sel["selection_rule"].eq("best_on_shared")
        & sel["selection_model_set"].eq("deepvision_shared")
        & sel["subject"].isin(subjects)
        & sel["model"].isin(models)
    ].copy()
    cols = [
        "subject",
        "model",
        "display_name",
        "selected_layer",
        "selected_layer_index",
        "selected_layer_frac",
        "selection_mrsa",
    ]
    return sel[cols].drop_duplicates()


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="all")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--batch-candidates", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-vicco-boot", type=int, default=1000)
    parser.add_argument("--n-score-jobs", type=int, default=8)
    parser.add_argument("--out-csv", default=str(OUT_CSV))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    subjects = SUBJECTS if args.subject == "all" else [args.subject]
    dense_layers = get_layer_set("dense")
    models = args.models if args.models else list(dense_layers.keys())
    out_csv = Path(args.out_csv)
    if args.overwrite and out_csv.exists():
        out_csv.unlink()

    selections = load_selections(subjects, models)
    if selections.empty:
        raise RuntimeError("No best-on-shared selections matched the requested subjects/models.")

    existing_done = set()
    if out_csv.exists() and not args.overwrite:
        done = pd.read_csv(out_csv, usecols=["subject", "model"])
        existing_done = set(done.drop_duplicates().itertuples(index=False, name=None))

    eval_items, cstim_slices = load_eval_items()
    rank_by_subject = {
        subject: load_subject_rank_bundle(subject, args.n_vicco_boot)
        for subject in subjects
    }

    args.batch_size = parse_batch_size(args.batch_size)
    batch_candidates = parse_batch_candidates(args.batch_candidates)

    for model in models:
        sub_sel = selections[selections["model"].eq(model)]
        if sub_sel.empty:
            continue
        pending_subjects = [
            s for s in sorted(sub_sel["subject"].unique())
            if (s, model) not in existing_done
        ]
        if not pending_subjects:
            print(f"[cached] {model}", flush=True)
            continue

        wanted_layers = sorted(sub_sel[sub_sel["subject"].isin(pending_subjects)]["selected_layer"].unique())
        spec_map = {name: agg for name, agg in dense_layers[model]}
        missing = [layer for layer in wanted_layers if layer not in spec_map]
        if missing:
            raise RuntimeError(f"{model}: selected layers not in dense layer set: {missing}")
        layer_specs = [(layer, spec_map[layer]) for layer in wanted_layers]

        print(f"\n=== {model}: {len(layer_specs)} selected layers ===", flush=True)
        extractor = MultiLayerExtractor(model, MODEL_SOURCE[model], layer_specs, device=args.device)
        try:
            batch_size = args.batch_size
            if batch_size == "auto":
                probe = eval_items[: max(parse_batch_candidates(args.batch_candidates))]
                print("  tuning batch size...", flush=True)
                batch_size, _records = tune_batch_size(
                    extractor.extract,
                    probe,
                    candidates=batch_candidates,
                    verbose=True,
                )
                print(f"  selected batch_size={batch_size}", flush=True)
            features_by_layer = extract_reduced_features(
                extractor,
                eval_items,
                batch_size=batch_size,
                model=model,
            )
        finally:
            extractor.free()

        rows = []
        for subject in pending_subjects:
            subject_sel = sub_sel[sub_sel["subject"].eq(subject)].iloc[0]
            layer = subject_sel["selected_layer"]
            features = features_by_layer[layer]
            ranks = rank_by_subject[subject]

            for group, brain_rank in ranks["cstim_ranks"].items():
                if group not in cstim_slices:
                    continue
                file_idx = ranks["group_stim_idx"].get(group)
                if file_idx is None or len(file_idx) == 0:
                    continue
                feats_group = features[cstim_slices[group]][file_idx]
                score = score_feature_rdm(feats_group, brain_rank)
                rows.append({
                    "subject": subject,
                    "model": model,
                    "display_name": MODEL_DISPLAY_NAMES.get(model, model),
                    "selection_rule": "best_on_shared",
                    "selection_model_set": "deepvision_shared",
                    "selected_layer": layer,
                    "selected_layer_index": subject_sel["selected_layer_index"],
                    "selected_layer_frac": subject_sel["selected_layer_frac"],
                    "selection_mrsa": subject_sel["selection_mrsa"],
                    "eval_target": "cstim",
                    "eval_model_set": group,
                    "n_bootstraps": 1,
                    "n_stimuli": int(len(file_idx)),
                    "frsa_mean": float(score),
                    "frsa_sem": np.nan,
                })

            if "vicco" in cstim_slices and ranks["vicco_bootstrap"]:
                file_idx = ranks["group_stim_idx"].get("vicco")
                feats_vicco = features[cstim_slices["vicco"]][file_idx]
                if args.n_score_jobs > 1:
                    scores = Parallel(n_jobs=args.n_score_jobs, prefer="threads")(
                        delayed(score_bootstrap)(
                            boot_idx,
                            idx,
                            feats_vicco,
                            ranks["vicco_ranks"][boot_idx],
                        )
                        for boot_idx, idx in enumerate(ranks["vicco_bootstrap"])
                    )
                else:
                    scores = [
                        score_bootstrap(boot_idx, idx, feats_vicco, ranks["vicco_ranks"][boot_idx])
                        for boot_idx, idx in enumerate(ranks["vicco_bootstrap"])
                    ]
                n_boot, mean, err = summarize(scores)
                rows.append({
                    "subject": subject,
                    "model": model,
                    "display_name": MODEL_DISPLAY_NAMES.get(model, model),
                    "selection_rule": "best_on_shared",
                    "selection_model_set": "deepvision_shared",
                    "selected_layer": layer,
                    "selected_layer_index": subject_sel["selected_layer_index"],
                    "selected_layer_frac": subject_sel["selected_layer_frac"],
                    "selection_mrsa": subject_sel["selection_mrsa"],
                    "eval_target": "vicco",
                    "eval_model_set": "vicco",
                    "n_bootstraps": n_boot,
                    "n_stimuli": int(ranks["n_vicco_sample"]),
                    "frsa_mean": mean,
                    "frsa_sem": err,
                })

        append_rows(out_csv, rows)
        print(f"  wrote {len(rows)} rows -> {out_csv}", flush=True)

    print(f"\nDone: {out_csv}")


if __name__ == "__main__":
    main()
