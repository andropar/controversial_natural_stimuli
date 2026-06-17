#!/usr/bin/env python3
"""Apply noisy-by-clean recovery evaluation to feature-method-sweep payloads."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve()
ROOT = next(p for p in SCRIPT.parents if (p / "src" / "cstims").exists())
COMMON_CODE = ROOT / "00_stimulus_selection" / "selection_evaluation" / "code" / "noisy_by_clean"
COMMON_SCRIPT = COMMON_CODE / "01_compute_noisy_by_clean_recovery.py"
SWEEP_ROOT = ROOT / "00_stimulus_selection" / "feature_method_sweep"
RECOVERY_ROOT = (
    ROOT
    / "00_stimulus_selection"
    / "selection_evaluation"
    / "feature_method_sweep_recovery"
    / "noisy_by_clean"
)
DEFAULT_RUN = SWEEP_ROOT / "results" / "sota_20260611_112941"
ENCODING_TRACKS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")


def has_option(args: list[str], *names: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in args for name in names)


def payload_methods(run_dir: Path) -> list[str]:
    payload_root = run_dir / "payloads"
    if not payload_root.exists():
        return []
    return sorted(path.name for path in payload_root.iterdir() if path.is_dir())


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_known_args(argv)


def load_compute_main():
    spec = importlib.util.spec_from_file_location("compute_noisy_by_clean_recovery", COMMON_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {COMMON_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(COMMON_CODE))
    spec.loader.exec_module(module)
    return module.main


def main() -> None:
    wrapper_args, common_args = parse_wrapper_args(sys.argv[1:])
    run_dir = wrapper_args.run_dir.resolve()
    output_root = (
        wrapper_args.output_root
        or RECOVERY_ROOT / "results" / run_dir.name / "eval"
    ).resolve()
    methods = (
        [item.strip() for item in wrapper_args.methods.split(",") if item.strip()]
        if wrapper_args.methods
        else payload_methods(run_dir)
    )
    if not methods and not has_option(common_args, "--model-sets"):
        raise RuntimeError(f"No payload methods found under {run_dir / 'payloads'}")

    defaults: list[str] = []
    if not has_option(common_args, "--model-sets"):
        defaults.extend(["--model-sets", ",".join(methods)])
    if not has_option(common_args, "--selection-root"):
        defaults.extend(["--selection-root", str(run_dir / "payloads")])
    if not has_option(common_args, "--output-root"):
        defaults.extend(["--output-root", str(output_root)])
    if not has_option(common_args, "--env"):
        defaults.extend(["--env", "raven"])
    if not has_option(common_args, "--tracks"):
        defaults.extend(["--tracks", "raw," + ",".join(ENCODING_TRACKS)])
    if not has_option(common_args, "--which-selection"):
        defaults.extend(["--which-selection", "final"])

    compute_main = load_compute_main()
    sys.argv = [sys.argv[0], *defaults, *common_args]
    compute_main()


if __name__ == "__main__":
    main()
