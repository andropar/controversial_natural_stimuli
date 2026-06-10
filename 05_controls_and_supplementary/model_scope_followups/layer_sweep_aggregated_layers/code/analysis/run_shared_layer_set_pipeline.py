#!/usr/bin/env python3
"""Dispatch dense DeepVision-shared feature extraction, scoring, and tables."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT

from layers_config import LAYER_SET_CHOICES, get_layer_set


SCRIPT_DIR = Path(__file__).resolve().parent
LOG_ROOT = LAYER_SWEEP_ROOT / "logs"
DATA_DIR = LAYER_SWEEP_ROOT / "results"


def parse_csv(value: str):
    return [v.strip() for v in value.split(",") if v.strip()]


def build_env(gpu: str | None = None):
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        import certifi

        env.setdefault("SSL_CERT_FILE", certifi.where())
        env.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except Exception:
        pass
    return env


def run_gpu_jobs(jobs, gpus, log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    active = []
    available = list(gpus)
    failures = []

    while pending or active:
        while pending and available:
            gpu = available.pop(0)
            job = pending.pop(0)
            log_path = log_dir / f"{job['name']}.log"
            log_f = open(log_path, "w", buffering=1)
            print(f"[start gpu={gpu}] {job['name']} -> {log_path}", flush=True)
            proc = subprocess.Popen(
                job["cmd"],
                cwd=LAYER_SWEEP_ROOT,
                env=build_env(gpu),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active.append({"proc": proc, "job": job, "log": log_f, "gpu": gpu})

        time.sleep(5)
        still = []
        for item in active:
            ret = item["proc"].poll()
            if ret is None:
                still.append(item)
                continue
            item["log"].close()
            status = "done" if ret == 0 else f"failed rc={ret}"
            print(f"[{status} gpu={item['gpu']}] {item['job']['name']}", flush=True)
            if ret != 0:
                failures.append(item["job"])
            available.append(item["gpu"])
        active = still

        if failures:
            for item in active:
                item["proc"].terminate()
                item["log"].close()
            raise SystemExit(f"{len(failures)} GPU jobs failed")


def run_logged(cmd, log_path: Path, *, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {' '.join(cmd)} -> {log_path}", flush=True)
    with open(log_path, "w", buffering=1) as log_f:
        subprocess.run(
            cmd,
            cwd=LAYER_SWEEP_ROOT,
            env=env or build_env(None),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["features", "score", "tables", "all"], default="all")
    parser.add_argument("--layer-set", choices=LAYER_SET_CHOICES, default="dense")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model names. Default: all models.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--batch-candidates", default=None)
    parser.add_argument("--n-shared-boot", type=int, default=1000)
    parser.add_argument("--bootstrap-n", type=int, default=100)
    parser.add_argument("--n-jobs-score", type=int, default=32)
    args = parser.parse_args()

    gpus = parse_csv(args.gpus)
    layer_specs = get_layer_set(args.layer_set)
    models = parse_csv(args.models) if args.models else list(layer_specs.keys())
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = LOG_ROOT / f"{args.layer_set}_shared_{stamp}"

    if args.phase in ("features", "all"):
        jobs = []
        for model in models:
            cmd = [
                args.python,
                str(SCRIPT_DIR / "13_compute_shared_wrsa_layer_sweep.py"),
                "--phase", "features",
                "--layer-set", args.layer_set,
                "--models", model,
                "--batch-size", args.batch_size,
            ]
            if args.batch_candidates:
                cmd.extend(["--batch-candidates", args.batch_candidates])
            jobs.append({"name": f"shared_features_{model}", "cmd": cmd})
        run_gpu_jobs(jobs, gpus, log_dir / "shared_features")

    if args.phase in ("score", "all"):
        cmd = [
            args.python,
            str(SCRIPT_DIR / "13_compute_shared_wrsa_layer_sweep.py"),
            "--phase", "score",
            "--layer-set", args.layer_set,
            "--models", ",".join(models),
            "--n-shared-boot", str(args.n_shared_boot),
            "--bootstrap-n", str(args.bootstrap_n),
            "--n-jobs", str(args.n_jobs_score),
            "--out-csv", str(DATA_DIR / f"wrsa_{args.layer_set}_shared_layer_sweep.csv"),
        ]
        run_logged(cmd, log_dir / "shared_score.log", env=build_env(None))

    if args.phase in ("tables", "all"):
        cmd = [
            args.python,
            str(SCRIPT_DIR / "14_build_mrsa_layer_selection_tables.py"),
            "--layer-set", args.layer_set,
        ]
        run_logged(cmd, log_dir / "selection_tables.log", env=build_env(None))

    print(f"[done] logs -> {log_dir}", flush=True)


if __name__ == "__main__":
    main()
