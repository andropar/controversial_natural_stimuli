#!/usr/bin/env python3
"""Dispatch layer-set feature extraction and encoding fits across GPUs.

This is intentionally a thin process scheduler. Each worker runs the normal
analysis scripts for one model (cstim features) or one subject/model pair
(DeepVision encoding fits), with ``CUDA_VISIBLE_DEVICES`` pinned to a single
GPU and BLAS thread counts kept low to avoid CPU oversubscription.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT

from cstims.constants import SUBJECTS
from layers_config import get_layer_set


SCRIPT_DIR = Path(__file__).resolve().parent
LOG_ROOT = LAYER_SWEEP_ROOT / "logs"


def parse_csv(value: str):
    return [v.strip() for v in value.split(",") if v.strip()]


def build_env(gpu: str | None = None, *, allow_cuda: bool = True):
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_COMPILE_DISABLE": "1",
        "TORCHDYNAMO_DISABLE": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    })
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    elif not allow_cuda:
        env["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import certifi

        env.setdefault("SSL_CERT_FILE", certifi.where())
        env.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except Exception:
        pass
    return env


def balance_model_groups(models, layer_specs, n_groups: int):
    groups = [[] for _ in range(n_groups)]
    weights = [0 for _ in range(n_groups)]
    ordered = sorted(models, key=lambda m: len(layer_specs[m]), reverse=True)
    for model in ordered:
        idx = min(range(n_groups), key=lambda i: weights[i])
        groups[idx].append(model)
        weights[idx] += len(layer_specs[model])
    return groups, weights


def run_jobs(
    jobs,
    slots,
    *,
    log_dir: Path,
    dry_run: bool = False,
    env_builder=build_env,
    slot_label: str = "gpu",
):
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    active = []
    available_slots = list(slots)
    failures = []

    if dry_run:
        for job in pending:
            print(" ".join(job["cmd"]))
        return failures

    while pending or active:
        while pending and available_slots:
            slot = available_slots.pop(0)
            job = pending.pop(0)
            log_path = log_dir / f"{job['name']}.log"
            log_f = open(log_path, "w", buffering=1)
            print(f"[start {slot_label}={slot}] {job['name']} -> {log_path}", flush=True)
            proc = subprocess.Popen(
                job["cmd"],
                cwd=LAYER_SWEEP_ROOT,
                env=env_builder(slot),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active.append({"proc": proc, "job": job, "log": log_f, "slot": slot})

        time.sleep(5)
        still_active = []
        for item in active:
            ret = item["proc"].poll()
            if ret is None:
                still_active.append(item)
                continue
            item["log"].close()
            status = "done" if ret == 0 else f"failed rc={ret}"
            print(f"[{status} {slot_label}={item['slot']}] {item['job']['name']}", flush=True)
            if ret != 0:
                failures.append(item["job"])
            available_slots.append(item["slot"])
        active = still_active

        if failures:
            for item in active:
                item["proc"].terminate()
                item["log"].close()
            break

    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["cstim-features", "encodings", "score", "stream", "all"],
                        default="all")
    parser.add_argument("--layer-set", choices=["configured", "dense"], default="dense")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--subjects", default=",".join(SUBJECTS))
    parser.add_argument("--models", default=None,
                        help="Comma-separated model names. Default: all models in layer set.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--batch-candidates", default=None)
    parser.add_argument("--extract-prefetch-workers", type=int, default=2,
                        help="Streaming mode: CPU prefetch threads per 07 worker.")
    parser.add_argument("--n-jobs-encoding", type=int, default=None,
                        help="CPU workers for encoding fits after feature caches are prepared. "
                             "Default: number of GPU slots.")
    parser.add_argument("--n-vicco-boot", type=int, default=1000)
    parser.add_argument("--n-shared-boot", type=int, default=1000)
    parser.add_argument("--bootstrap-n", type=int, default=100)
    parser.add_argument("--n-jobs-score", type=int, default=16)
    parser.add_argument("--layers-per-chunk", default="auto",
                        help="Streaming mode: layer chunk size passed to 07; use 'auto' "
                             "for model-major GPU probing.")
    parser.add_argument("--max-layers-per-chunk", type=int, default=128,
                        help="Streaming model-major: hard cap on auto layer chunk size. "
                             "Use 0 for no explicit layer-count cap.")
    parser.add_argument("--max-feature-gb-per-chunk", type=float, default=40.0,
                        help="Streaming model-major: approximate host-memory budget for "
                             "one reduced feature block per GPU worker.")
    parser.add_argument("--n-fit-jobs", type=int, default=3)
    parser.add_argument("--n-score-jobs-stream", type=int, default=3)
    parser.add_argument("--stream-dispatch", choices=["model", "subject-model"], default="model",
                        help="Streaming scheduler. 'model' runs one long-lived worker per "
                             "GPU with a balanced model queue across all subjects.")
    parser.add_argument("--fit-backend", choices=["cpu", "gpu"], default="cpu",
                        help="Streaming mode: encoding-fit backend passed to 07.")
    parser.add_argument("--gpu-fit-dtype", choices=["float32", "float64"], default="float64",
                        help="Streaming mode: dtype passed to 07 when --fit-backend gpu.")
    parser.add_argument("--stream-part-root", default=None,
                        help="Optional stream part root passed to 07 stream/merge-stream.")
    parser.add_argument("--stream-encoding-root", default=None,
                        help="Optional root where 07 stream saves fitted encoding_model.npz files.")
    parser.add_argument("--stream-progress-log", default=None,
                        help="Optional JSONL progress log. Defaults to a timestamped file under logs/.")
    parser.add_argument("--stream-out-csv", default=None,
                        help="Optional cstim/Vicco stream merge output CSV.")
    parser.add_argument("--stream-shared-out-csv", default=None,
                        help="Optional DeepVision-shared stream merge output CSV.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpus = parse_csv(args.gpus)
    encoding_slots = [str(i) for i in range(args.n_jobs_encoding or len(gpus))]
    subjects = parse_csv(args.subjects)
    layer_specs = get_layer_set(args.layer_set)
    models = parse_csv(args.models) if args.models else list(layer_specs.keys())
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = LOG_ROOT / f"{args.layer_set}_{stamp}"
    stream_progress_log = None
    if args.phase == "stream":
        stream_progress_log = (
            Path(args.stream_progress_log)
            if args.stream_progress_log
            else log_dir / f"stream_progress_{stamp}.jsonl"
        )
        print(f"[progress-log] {stream_progress_log}", flush=True)
        if args.stream_encoding_root:
            print(f"[stream-encoding-root] {args.stream_encoding_root}", flush=True)

    common = ["--layer-set", args.layer_set, "--batch-size", "auto"]
    if args.batch_candidates:
        common.extend(["--batch-candidates", args.batch_candidates])

    if args.phase in ("cstim-features", "all"):
        jobs = []
        for model in models:
            jobs.append({
                "name": f"cstim_{model}",
                "cmd": [
                    args.python,
                    str(SCRIPT_DIR / "01_extract_layer_features.py"),
                    "--models", model,
                    *common,
                ],
            })
        failures = run_jobs(jobs, gpus, log_dir=log_dir / "cstim_features", dry_run=args.dry_run)
        if failures:
            raise SystemExit(f"{len(failures)} cstim feature jobs failed")

    if args.phase in ("encodings", "all"):
        feature_jobs = []
        for subject in subjects:
            for model in models:
                feature_jobs.append({
                    "name": f"encoding_features_{subject}_{model}",
                    "cmd": [
                        args.python,
                        str(SCRIPT_DIR / "07_fit_encodings_layer_sweep.py"),
                        "--subject", subject,
                        "--models", model,
                        "--mode", "features",
                        *common,
                    ],
                })
        failures = run_jobs(
            feature_jobs,
            gpus,
            log_dir=log_dir / "encoding_features",
            dry_run=args.dry_run,
        )
        if failures:
            raise SystemExit(f"{len(failures)} encoding feature jobs failed")

        fit_jobs = []
        for subject in subjects:
            for model in models:
                fit_jobs.append({
                    "name": f"encoding_{subject}_{model}",
                    "cmd": [
                        args.python,
                        str(SCRIPT_DIR / "07_fit_encodings_layer_sweep.py"),
                        "--subject", subject,
                        "--models", model,
                        "--mode", "fit",
                        *common,
                    ],
                })
        failures = run_jobs(
            fit_jobs,
            encoding_slots,
            log_dir=log_dir / "encodings",
            dry_run=args.dry_run,
            env_builder=lambda _slot: build_env(None, allow_cuda=False),
            slot_label="cpu",
        )
        if failures:
            raise SystemExit(f"{len(failures)} encoding jobs failed")

    if args.phase in ("score", "all"):
        cmd = [
            args.python,
            str(SCRIPT_DIR / "08_compute_wrsa_layer_sweep.py"),
            "--layer-set", args.layer_set,
            "--n-vicco-boot", str(args.n_vicco_boot),
            "--n-jobs", str(args.n_jobs_score),
        ]
        if args.dry_run:
            print(" ".join(cmd))
        else:
            env = os.environ.copy()
            env.update({
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            })
            try:
                import certifi

                env.setdefault("SSL_CERT_FILE", certifi.where())
                env.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
            except Exception:
                pass
            subprocess.run(cmd, cwd=LAYER_SWEEP_ROOT, env=env, check=True)

    if args.phase == "stream":
        jobs = []
        if args.stream_dispatch == "model":
            groups, weights = balance_model_groups(models, layer_specs, len(gpus))
            for idx, group in enumerate(groups):
                if not group:
                    continue
                print(
                    f"[stream-group {idx}] layers={weights[idx]} models={','.join(group)}",
                    flush=True,
                )
                jobs.append({
                    "name": f"stream_model_group{idx}_{group[0]}",
                    "cmd": [
                        args.python,
                        str(SCRIPT_DIR / "07_fit_encodings_layer_sweep.py"),
                        "--subject", ",".join(subjects),
                        "--models", *group,
                        "--mode", "stream-model",
                        *common,
                        "--n-vicco-boot", str(args.n_vicco_boot),
                        "--n-shared-boot", str(args.n_shared_boot),
                        "--bootstrap-n", str(args.bootstrap_n),
                        "--layers-per-chunk", str(args.layers_per_chunk),
                        "--max-layers-per-chunk", str(args.max_layers_per_chunk),
                        "--max-feature-gb-per-chunk", str(args.max_feature_gb_per_chunk),
                        "--n-fit-jobs", str(args.n_fit_jobs),
                        "--n-score-jobs", str(args.n_score_jobs_stream),
                        "--fit-backend", args.fit_backend,
                        "--gpu-fit-dtype", args.gpu_fit_dtype,
                        "--extract-prefetch-workers", str(args.extract_prefetch_workers),
                    ],
                })
                if args.stream_part_root:
                    jobs[-1]["cmd"].extend(["--stream-part-root", args.stream_part_root])
                if args.stream_encoding_root:
                    jobs[-1]["cmd"].extend(["--stream-encoding-root", args.stream_encoding_root])
                if stream_progress_log:
                    jobs[-1]["cmd"].extend(["--progress-log", str(stream_progress_log)])
        else:
            layer_chunk = (
                "16"
                if str(args.layers_per_chunk).lower() == "auto"
                else str(args.layers_per_chunk)
            )
            for model in models:
                for subject in subjects:
                    jobs.append({
                        "name": f"stream_{subject}_{model}",
                        "cmd": [
                            args.python,
                            str(SCRIPT_DIR / "07_fit_encodings_layer_sweep.py"),
                            "--subject", subject,
                            "--models", model,
                            "--mode", "stream",
                            *common,
                            "--n-vicco-boot", str(args.n_vicco_boot),
                            "--n-shared-boot", str(args.n_shared_boot),
                            "--bootstrap-n", str(args.bootstrap_n),
                            "--layers-per-chunk", layer_chunk,
                            "--n-fit-jobs", str(args.n_fit_jobs),
                            "--n-score-jobs", str(args.n_score_jobs_stream),
                            "--fit-backend", args.fit_backend,
                            "--gpu-fit-dtype", args.gpu_fit_dtype,
                            "--extract-prefetch-workers", str(args.extract_prefetch_workers),
                        ],
                    })
                    if args.stream_part_root:
                        jobs[-1]["cmd"].extend(["--stream-part-root", args.stream_part_root])
                    if args.stream_encoding_root:
                        jobs[-1]["cmd"].extend(["--stream-encoding-root", args.stream_encoding_root])
                    if stream_progress_log:
                        jobs[-1]["cmd"].extend(["--progress-log", str(stream_progress_log)])
        failures = run_jobs(
            jobs,
            gpus,
            log_dir=log_dir / "stream",
            dry_run=args.dry_run,
        )
        if failures:
            raise SystemExit(f"{len(failures)} streaming jobs failed")

        merge_cmd = [
            args.python,
            str(SCRIPT_DIR / "07_fit_encodings_layer_sweep.py"),
            "--mode", "merge-stream",
            "--layer-set", args.layer_set,
        ]
        if args.stream_part_root:
            merge_cmd.extend(["--stream-part-root", args.stream_part_root])
        if args.stream_out_csv:
            merge_cmd.extend(["--out-csv", args.stream_out_csv])
        if args.stream_shared_out_csv:
            merge_cmd.extend(["--shared-out-csv", args.stream_shared_out_csv])
        if args.dry_run:
            print(" ".join(merge_cmd))
        else:
            subprocess.run(
                merge_cmd,
                cwd=LAYER_SWEEP_ROOT,
                env=build_env(None, allow_cuda=False),
                check=True,
            )


if __name__ == "__main__":
    main()
