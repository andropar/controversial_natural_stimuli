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

from config import SUBJECTS
from layers_config import LAYER_SET_CHOICES, get_layer_set


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
    parser.add_argument("--layer-set", choices=LAYER_SET_CHOICES, default="dense")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--subjects", default=",".join(SUBJECTS))
    parser.add_argument("--models", default=None,
                        help="Comma-separated model names. Default: all models in layer set.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--batch-candidates", default=None)
    parser.add_argument("--n-jobs-encoding", type=int, default=None,
                        help="CPU workers for encoding fits after feature caches are prepared. "
                             "Default: number of GPU slots.")
    parser.add_argument("--n-vicco-boot", type=int, default=1000)
    parser.add_argument("--n-shared-boot", type=int, default=1000)
    parser.add_argument("--bootstrap-n", type=int, default=100)
    parser.add_argument("--n-jobs-score", type=int, default=16)
    parser.add_argument("--layers-per-chunk", type=int, default=16)
    parser.add_argument("--n-fit-jobs", type=int, default=3)
    parser.add_argument("--n-score-jobs-stream", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpus = parse_csv(args.gpus)
    encoding_slots = [str(i) for i in range(args.n_jobs_encoding or len(gpus))]
    subjects = parse_csv(args.subjects)
    layer_specs = get_layer_set(args.layer_set)
    models = parse_csv(args.models) if args.models else list(layer_specs.keys())
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = LOG_ROOT / f"{args.layer_set}_{stamp}"

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
        for subject in subjects:
            for model in models:
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
                        "--layers-per-chunk", str(args.layers_per_chunk),
                        "--n-fit-jobs", str(args.n_fit_jobs),
                        "--n-score-jobs", str(args.n_score_jobs_stream),
                    ],
                })
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
