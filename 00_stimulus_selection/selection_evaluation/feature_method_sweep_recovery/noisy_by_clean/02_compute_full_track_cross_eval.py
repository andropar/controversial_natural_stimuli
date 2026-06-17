#!/usr/bin/env python3
"""Evaluate single-track sweep selections in every raw/encoded space.

The feature-method sweep selected some image sets with deliberately restricted
objective tracks, for example ``raw_only_mean_min`` and ``sub01_only_mean_min``.
Their saved payloads therefore advertise only those objective tracks, so the
standard recovery evaluator only computes those tracks.  This helper builds
temporary augmented payloads that keep the selected image set fixed but expose
the full evaluation track set: raw plus all five subject encodings.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve()
ROOT = next(p for p in SCRIPT.parents if (p / "src" / "cstims").exists())
SWEEP_ROOT = ROOT / "00_stimulus_selection" / "feature_method_sweep"
RECOVERY_RESULTS_ROOT = (
    ROOT
    / "00_stimulus_selection"
    / "selection_evaluation"
    / "feature_method_sweep_recovery"
    / "noisy_by_clean"
    / "results"
)
DEFAULT_RUN = SWEEP_ROOT / "results" / "sota_20260611_112941"
RECOVERY_SCRIPT = SCRIPT.parent / "01_compute_recovery.py"
PYTHON = Path("/data/home_roth/miniforge3/bin/python")

ENCODING_TRACKS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")
FULL_TRACKS = ("raw", *ENCODING_TRACKS)
DEFAULT_METHODS = ("raw_only_mean_min", "sub01_only_mean_min")


def full_track_definitions() -> list[dict[str, str]]:
    return [
        {"name": "raw", "type": "identity"},
        *(
            {"name": name, "type": "encoding", "encoding_name": name}
            for name in ENCODING_TRACKS
        ),
    ]


def copy_optional_file(src_dir: Path, dst_dir: Path, filename: str) -> None:
    src = src_dir / filename
    if src.exists():
        shutil.copy2(src, dst_dir / filename)


def prepare_augmented_payloads(
    run_dir: Path,
    output_root: Path,
    methods: list[str],
    *,
    keep_precomputed_noise: bool,
) -> Path:
    source_root = run_dir / "payloads"
    payload_root = output_root / "payloads"
    payload_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for method_id in methods:
        src_dir = source_root / method_id
        if not src_dir.exists():
            raise FileNotFoundError(f"Missing source payload directory: {src_dir}")
        dst_dir = payload_root / method_id
        dst_dir.mkdir(parents=True, exist_ok=True)

        payload_path = src_dir / "selected_stimuli_data.pkl"
        with payload_path.open("rb") as f:
            payload = pickle.load(f)

        original_tracks = [track.get("name") for track in payload.get("track_definitions", [])]
        original_noise_tracks = sorted((payload.get("var_noise_by_model") or {}).keys())
        payload["track_definitions"] = full_track_definitions()
        if not keep_precomputed_noise:
            payload.pop("var_noise_by_model", None)
            payload.pop("selection_objective_var_noise_by_model", None)
        payload["feature_method_cross_eval"] = {
            "source_payload": str(payload_path),
            "original_track_definitions": payload.get("feature_method_cross_eval", {}).get(
                "original_track_definitions",
                original_tracks,
            ),
            "original_noise_tracks": original_noise_tracks,
            "keep_precomputed_noise": bool(keep_precomputed_noise),
            "evaluation_track_definitions": payload["track_definitions"],
            "note": "Selected image set unchanged; only evaluation tracks were expanded.",
        }

        with (dst_dir / "selected_stimuli_data.pkl").open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        copy_optional_file(src_dir, dst_dir, "selected_image_records.csv")
        copy_optional_file(src_dir, dst_dir, "selection_trace.csv")
        copy_optional_file(src_dir, dst_dir, "method_config.json")

        manifest_rows.append(
            {
                "method_id": method_id,
                "source_payload": str(payload_path),
                "augmented_payload": str(dst_dir / "selected_stimuli_data.pkl"),
                "original_tracks": ",".join(original_tracks),
                "evaluation_tracks": ",".join(FULL_TRACKS),
            }
        )

    pd.DataFrame(manifest_rows).to_csv(payload_root / "cross_eval_manifest.csv", index=False)
    return payload_root


def run_recovery(args: argparse.Namespace, payload_root: Path, methods: list[str]) -> None:
    output_root = args.output_root / "eval"
    output_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(args.python),
        str(RECOVERY_SCRIPT),
        "--model-sets",
        ",".join(methods),
        "--selection-root",
        str(payload_root),
        "--output-root",
        str(output_root),
        "--env",
        args.env,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--n-random-subsets",
        str(args.n_random_subsets),
        "--n-noise-samples",
        str(args.n_noise_samples),
        "--n-bootstrap",
        str(args.n_bootstrap),
        "--which-selection",
        "final",
        "--tracks",
        ",".join(FULL_TRACKS),
    ]
    if args.random_feature_dir is not None:
        cmd.extend(["--random-feature-dir", str(args.random_feature_dir)])
        cmd.append("--strict-random-models")
    if args.shared_encodings:
        cmd.append("--shared-encodings")

    env = os.environ.copy()
    conda_lib = "/data/home_roth/miniforge3/lib"
    env["LD_LIBRARY_PATH"] = (
        conda_lib
        if not env.get("LD_LIBRARY_PATH")
        else conda_lib + os.pathsep + env["LD_LIBRARY_PATH"]
    )

    metadata = {
        "command": cmd,
        "cwd": str(ROOT),
        "output_root": str(output_root),
        "methods": methods,
        "tracks": FULL_TRACKS,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "run_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print("Running full-track cross-evaluation:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--env", default="raven", choices=["raven", "iris"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-random-subsets", type=int, default=50)
    parser.add_argument("--n-noise-samples", type=int, default=100)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--random-feature-dir", type=Path, default=None)
    parser.add_argument(
        "--keep-precomputed-noise",
        action="store_true",
        help=(
            "Keep payload var_noise_by_model entries. By default, augmented "
            "cross-eval payloads drop them so every method/track uses the same "
            "fresh noise calibration source."
        ),
    )
    parser.add_argument("--shared-encodings", action="store_true")
    parser.add_argument("--python", type=Path, default=PYTHON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    args.output_root = (
        args.output_root
        or (RECOVERY_RESULTS_ROOT / args.run_dir.name / "cross_eval_full_tracks")
    ).resolve()
    if args.random_feature_dir is not None:
        args.random_feature_dir = args.random_feature_dir.resolve()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    payload_root = prepare_augmented_payloads(
        args.run_dir,
        args.output_root,
        methods,
        keep_precomputed_noise=args.keep_precomputed_noise,
    )
    run_recovery(args, payload_root, methods)


if __name__ == "__main__":
    main()
