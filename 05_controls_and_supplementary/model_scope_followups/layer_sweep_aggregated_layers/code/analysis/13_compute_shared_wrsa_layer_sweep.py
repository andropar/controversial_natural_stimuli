#!/usr/bin/env python3
"""Compute mRSA-transfer on DeepVision shared stimuli for every dense layer.

This is the in-distribution counterpart to the cstim/Vicco dense layer sweep:
the encoding models are still trained on each subject's DeepVision unique
images, but evaluation is on held-out DeepVision shared images with measured
brain responses.

Outputs by default:
    results/wrsa_dense_shared_layer_sweep.csv
or, with --part-dir:
    results/wrsa_dense_shared_parts/{model}.csv
"""

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT, SHARE_ROOT
import argparse
import importlib.util
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import rankdata
from tqdm import tqdm
from joblib import Parallel, delayed

from config import MODEL_DISPLAY_NAMES, PAPER_ROOT, SUBJECTS
from utils import (
    bootstrap_sample_indices,
    compute_rdm_correlation,
    parse_subject_arg,
    predict_voxel_responses,
)
from layers_config import LAYER_SET_CHOICES, MODEL_SOURCE, get_layer_set
from batch_tuning import (
    parse_batch_candidates,
    parse_batch_size,
    records_to_array,
    tune_batch_size,
)
from multilayer_extractor import MultiLayerExtractor
from srp_utils import cached_layer_current, metadata_arrays, SRPProjectorCache

try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

_WRSA_SPEC = importlib.util.spec_from_file_location(
    "wrsa_layer_sweep",
    Path(__file__).resolve().parent / "08_compute_wrsa_layer_sweep.py",
)
_WRSA = importlib.util.module_from_spec(_WRSA_SPEC)
_WRSA_SPEC.loader.exec_module(_WRSA)
ENCODING_PROTOCOL = _WRSA.ENCODING_PROTOCOL
encoding_path = _WRSA.encoding_path
load_encoding = _WRSA.load_encoding


CACHE_ROOT = LAYER_SWEEP_ROOT / "cache_or_heavy"
FEATURE_CACHE = CACHE_ROOT / "features"
DATA_DIR = LAYER_SWEEP_ROOT / "results"
PART_DIR = DATA_DIR / "wrsa_dense_shared_parts"
DEEPVISION_CACHE = SHARE_ROOT / "01_brain_model_alignment" / "cache_or_heavy" / "brain_data"
DEEPVISION_FMRI_ROOT = Path("/data/labshare/_stachelschwein/SSD/jroth/deepvision_fmri")
STIMULUS_TYPE = "deepvision_shared"


def _rdm_upper_vec(rdm: np.ndarray) -> np.ndarray:
    n = rdm.shape[0]
    return rdm[np.triu_indices(n, k=1)]


def _rank_vector(vec: np.ndarray) -> np.ndarray:
    return rankdata(vec, method="average").astype(np.float32)


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    xm = x - x.mean()
    ym = y - y.mean()
    den = float(np.sqrt(np.dot(xm, xm) * np.dot(ym, ym)))
    if den <= 0:
        return float("nan")
    return float(np.dot(xm, ym) / den)


def _load_pil_images(paths):
    images = []
    for path in paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images


def load_shared_image_paths():
    csv_path = DEEPVISION_CACHE / "image_sets" / "deepvision_shared.csv"
    image_dir = DEEPVISION_CACHE / "image_sets" / "deepvision_shared"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing shared image cache CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    return (image_dir.as_posix() + "/" + df["image_name"]).tolist()


def shared_feature_path(model: str) -> Path:
    return FEATURE_CACHE / model / f"{STIMULUS_TYPE}.npz"


def load_or_extract_shared_features(
    model: str,
    layer_specs,
    image_paths,
    *,
    batch_size,
    batch_candidates,
    overwrite: bool,
):
    path = shared_feature_path(model)
    needed = [name for name, _ in layer_specs]
    cached_payload = {}
    if path.exists() and not overwrite:
        try:
            with np.load(path, allow_pickle=True) as cached:
                cached_payload = {k: cached[k] for k in cached.files}
            class _PayloadView:
                files = tuple(cached_payload.keys())
                def __getitem__(self, key):
                    return cached_payload[key]
            payload_view = _PayloadView()
            if all(
                name in cached_payload and cached_layer_current(payload_view, name)
                for name in needed
            ):
                return {name: cached_payload[name] for name in needed}
        except Exception:
            cached_payload = {}
    elif overwrite:
        cached_payload = {}

    class _PayloadView:
        files = tuple(cached_payload.keys())
        def __getitem__(self, key):
            return cached_payload[key]

    payload_view = _PayloadView()
    missing_specs = [
        (name, agg)
        for name, agg in layer_specs
        if name not in cached_payload or not cached_layer_current(payload_view, name)
    ]
    if not missing_specs:
        return {name: cached_payload[name] for name in needed}
    print(f"[features] {model}: extracting {len(missing_specs)} missing / "
          f"{len(needed)} dense layers for {len(image_paths)} shared images", flush=True)
    t0 = time.time()
    extractor = MultiLayerExtractor(model, MODEL_SOURCE[model], missing_specs)
    batch_tuning_records = []

    if batch_size == "auto":
        candidates = parse_batch_candidates(batch_candidates)
        probe_paths = image_paths[:min(max(candidates), len(image_paths))]
        probe_images = _load_pil_images(probe_paths)
        print("[features] tuning batch size", flush=True)
        batch_size, batch_tuning_records = tune_batch_size(
            extractor.extract,
            probe_images,
            candidates=candidates,
            verbose=True,
        )
        print(f"[features] selected batch_size={batch_size}", flush=True)

    feats = {name: [] for name, _ in missing_specs}
    meta_by_layer = {}
    srp_cache = SRPProjectorCache()
    for start in tqdm(range(0, len(image_paths), batch_size), desc=f"extract {model}"):
        batch = _load_pil_images(image_paths[start:start + batch_size])
        out = extractor.extract(batch)
        for name, _ in missing_specs:
            reduced, meta = srp_cache.transform(out[name])
            if name not in meta_by_layer:
                meta_by_layer[name] = meta
            elif int(meta_by_layer[name]["original_feature_dim"]) != int(meta["original_feature_dim"]):
                raise RuntimeError(f"Inconsistent feature dim for layer {name}")
            feats[name].append(reduced)

    feats = {name: np.concatenate(chunks, axis=0).astype(np.float32) for name, chunks in feats.items()}
    extractor.free()

    path.parent.mkdir(parents=True, exist_ok=True)
    save_payload = dict(cached_payload)
    for layer_name, arr in feats.items():
        save_payload[layer_name] = arr
        save_payload.update(metadata_arrays(layer_name, meta_by_layer[layer_name]))
    save_payload["_batch_size"] = np.array(batch_size, dtype=np.int32)
    if batch_tuning_records:
        save_payload["_batch_tuning"] = records_to_array(batch_tuning_records)
    np.savez_compressed(path, **save_payload)
    print(f"[features] {model}: wrote {path} in {time.time() - t0:.1f}s", flush=True)
    return {name: save_payload[name] for name in needed}


def load_shared_subject_data(subjects, n_bootstrap: int, bootstrap_n: int, seed: int):
    """Load hlvis betas and pre-ranked brain RDM vectors for shared stimuli."""
    out = {}
    n_stim = None
    for subject in subjects:
        root = (
            DEEPVISION_CACHE
            / "voxel_sets"
            / "deepvision_shared_visual_cve0p20"
            / "finalinterp"
            / subject
        )
        betas_path = root / "voxel_betas.npy"
        bsa_path = root / "brain_space_arrays.npz"
        if not betas_path.exists() or not bsa_path.exists():
            print(f"[skip] {subject}: missing shared voxel cache", flush=True)
            continue
        betas = np.load(betas_path).astype(np.float32)
        bsa = np.load(bsa_path)
        hlvis = np.asarray(bsa["hlvis_mask"], dtype=bool)
        betas_hlvis = np.ascontiguousarray(betas[hlvis, :], dtype=np.float32)
        if n_stim is None:
            n_stim = betas_hlvis.shape[1]
        elif betas_hlvis.shape[1] != n_stim:
            raise RuntimeError(f"{subject}: shared n_stim mismatch")
        out[subject] = {
            "betas_hlvis": betas_hlvis,
            "n_hlvis": int(hlvis.sum()),
        }

    if not out:
        raise RuntimeError("No subjects with shared voxel cache")
    sample_n = min(bootstrap_n, n_stim)
    boot = bootstrap_sample_indices(n_stim, sample_n, n_bootstrap=n_bootstrap, seed=seed)

    print(f"[shared] n_stim={n_stim}, bootstrap_n={sample_n}, n_bootstrap={len(boot)}", flush=True)
    for subject, data in out.items():
        ranks = []
        b = data["betas_hlvis"]
        for idx in tqdm(boot, desc=f"brain ranks {subject}", leave=False):
            rdm = compute_rdm_correlation(b[:, idx].T)
            ranks.append(_rank_vector(_rdm_upper_vec(rdm)))
        data["brain_ranks"] = ranks
        data["betas_hlvis"] = None
    return out, boot, sample_n


def predict(features, enc):
    pred = predict_voxel_responses(features, enc)
    roi = enc.get("roi_hlvis")
    if roi is not None:
        pred = pred[:, np.asarray(roi, dtype=bool)]
    return np.ascontiguousarray(pred, dtype=np.float32)


def score_one_boot(boot_idx, idx, pred_all, brain_rank, n_stimuli):
    pred_rdm = compute_rdm_correlation(pred_all[idx])
    pred_rank = _rank_vector(_rdm_upper_vec(pred_rdm))
    return {
        "bootstrap_idx": boot_idx,
        "n_stimuli": n_stimuli,
        "rsa": _pearson_r(pred_rank, brain_rank),
    }


def load_existing_counts(out_csv: Path):
    if not out_csv.exists():
        return {}
    usecols = ["subject", "model", "layer", "model_set", "stimulus_type", "bootstrap_idx"]
    try:
        existing = pd.read_csv(out_csv, usecols=usecols)
    except Exception:
        return {}
    counts = (
        existing.groupby(["subject", "model", "layer", "model_set", "stimulus_type"])["bootstrap_idx"]
        .nunique()
        .to_dict()
    )
    print(f"[resume] found {len(counts)} existing scored groups in {out_csv}", flush=True)
    return counts


def group_complete(existing_counts, subject, model, layer, expected_n):
    key = (subject, model, layer, STIMULUS_TYPE, STIMULUS_TYPE)
    return existing_counts.get(key, 0) >= expected_n


def append_rows(out_csv: Path, rows):
    if not rows:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        out_csv,
        mode="a",
        header=not out_csv.exists(),
        index=False,
    )


def score_model(
    model,
    layer_specs,
    features_per_layer,
    subjects_data,
    boot,
    n_stimuli,
    *,
    subjects,
    n_jobs,
    existing_counts=None,
):
    existing_counts = existing_counts or {}
    rows = []
    display = MODEL_DISPLAY_NAMES.get(model, model)
    for layer_name, _ in tqdm(layer_specs, desc=f"score {model}"):
        feats = features_per_layer[layer_name]
        if feats.ndim != 2:
            feats = feats.reshape(feats.shape[0], -1)
        feats = np.ascontiguousarray(feats, dtype=np.float32)

        for subject in subjects:
            if subject not in subjects_data:
                continue
            if group_complete(existing_counts, subject, model, layer_name, len(boot)):
                continue
            enc_p = encoding_path(subject, model, layer_name)
            if not enc_p.exists():
                continue
            enc = load_encoding(enc_p)
            if enc is None:
                continue
            if str(np.asarray(enc.get("fit_protocol", "")).item()) != ENCODING_PROTOCOL:
                continue

            pred_all = predict(feats, enc)
            brain_ranks = subjects_data[subject]["brain_ranks"]
            if n_jobs > 1:
                res = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(score_one_boot)(
                        boot_idx, idx, pred_all, brain_ranks[boot_idx], n_stimuli
                    )
                    for boot_idx, idx in enumerate(boot)
                )
            else:
                res = [
                    score_one_boot(boot_idx, idx, pred_all, brain_ranks[boot_idx], n_stimuli)
                    for boot_idx, idx in enumerate(boot)
                ]
            for r in res:
                rows.append({
                    "subject": subject,
                    "model": model,
                    "display_name": display,
                    "layer": layer_name,
                    "model_set": STIMULUS_TYPE,
                    "stimulus_type": STIMULUS_TYPE,
                    **r,
                })
    return rows


def parse_models(value, layer_specs):
    if value is None:
        return list(layer_specs.keys())
    return [m.strip() for m in value.split(",") if m.strip()]


def main():
    global ENCODING_PROTOCOL
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="all")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model names. Default: all models.")
    parser.add_argument("--layer-set", choices=LAYER_SET_CHOICES, default="dense")
    parser.add_argument("--phase", choices=["features", "score", "all"], default="all")
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--part-dir", default=None,
                        help="Write one CSV per model into this directory.")
    parser.add_argument("--n-shared-boot", type=int, default=1000)
    parser.add_argument("--bootstrap-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--batch-candidates", default=None)
    parser.add_argument("--overwrite-features", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    layer_specs_all = get_layer_set(args.layer_set)
    models = parse_models(args.models, layer_specs_all)
    subjects = parse_subject_arg(args.subject)
    args.batch_size = parse_batch_size(args.batch_size)

    image_paths = load_shared_image_paths()
    print(f"[shared] image paths: {len(image_paths)}", flush=True)

    subjects_data = None
    boot = None
    n_stimuli = None
    if args.phase in ("score", "all"):
        subjects_data, boot, n_stimuli = load_shared_subject_data(
            subjects,
            n_bootstrap=args.n_shared_boot,
            bootstrap_n=args.bootstrap_n,
            seed=args.seed,
        )

    out = Path(args.out_csv) if args.out_csv else (
        DATA_DIR / f"wrsa_{args.layer_set}_shared_layer_sweep.csv"
    )
    global_existing_counts = (
        load_existing_counts(out)
        if args.phase != "features" and not args.part_dir
        else {}
    )
    for model in models:
        if model not in layer_specs_all:
            print(f"[skip] unknown model: {model}", flush=True)
            continue
        layer_specs = layer_specs_all[model]
        feats = load_or_extract_shared_features(
            model,
            layer_specs,
            image_paths,
            batch_size=args.batch_size,
            batch_candidates=args.batch_candidates,
            overwrite=args.overwrite_features,
        )
        if args.phase == "features":
            continue

        if args.part_dir:
            part_dir = Path(args.part_dir)
            part_dir.mkdir(parents=True, exist_ok=True)
            out_part = part_dir / f"{model}.csv"
            existing_counts = load_existing_counts(out_part)
        else:
            out_part = out
            existing_counts = global_existing_counts

        rows = score_model(
            model,
            layer_specs,
            feats,
            subjects_data,
            boot,
            n_stimuli,
            subjects=subjects,
            n_jobs=args.n_jobs,
            existing_counts=existing_counts,
        )
        if args.part_dir:
            append_rows(out_part, rows)
            print(f"[score] {model}: appended {len(rows)} rows -> {out_part}", flush=True)
        else:
            append_rows(out, rows)
            for r in rows:
                key = (r["subject"], r["model"], r["layer"], r["model_set"], r["stimulus_type"])
                global_existing_counts[key] = global_existing_counts.get(key, 0) + 1
            print(f"[score] {model}: appended {len(rows)} rows -> {out}", flush=True)

    if args.part_dir:
        print(f"[done] parts -> {args.part_dir}", flush=True)
    elif args.phase != "features":
        print(f"[done] output -> {out}", flush=True)


if __name__ == "__main__":
    main()
