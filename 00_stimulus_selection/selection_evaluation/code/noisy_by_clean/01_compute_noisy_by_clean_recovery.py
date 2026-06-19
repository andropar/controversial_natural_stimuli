#!/usr/bin/env python3
"""Compute recovery curves with noisy-by-clean model classification.

    old: corr(clean_i, noisy_j), classify along noisy columns
    new: corr(noisy_i, clean_j), classify along clean columns

The track-level recovery workflow lives in ``cstims.evaluation.recovery``; this
script handles repository-specific payload paths and CSV outputs.
"""

from __future__ import annotations

import argparse
from functools import partial
import gc
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[4]
EVAL_ROOT = SCRIPT.parents[2]
SRC_DIR = ROOT / "src"
for path in (SRC_DIR,):
    sys.path.insert(0, str(path))

from cstims import constants, paths
from cstims.evaluation.constants import (  # noqa: E402
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_N_NOISE_SAMPLES,
    ENCODING_TRACKS,
    MODEL_SET_ORDER,
    get_default_noise_level_multipliers,
)
from cstims.evaluation.io import load_payload  # noqa: E402
from cstims.evaluation.matrix_rows import append_correlation_matrix_rows  # noqa: E402
from cstims.evaluation.memory import log_memory  # noqa: E402
from cstims.evaluation.payload import (  # noqa: E402
    VALID_ENVS,
    apply_env_paths,
    env_config_root,
    filter_payload_to_models,
    selection_root,
)
from cstims.evaluation.random_features import (  # noqa: E402
    available_random_models,
    make_random_feature_cache_loader,
)
from cstims.evaluation.recovery import compute_discriminability_for_track  # noqa: E402
from cstims.evaluation.track_loading import (  # noqa: E402
    get_all_tracks_for_evaluation,
    load_features_for_track,
)


DEFAULT_RESULTS = EVAL_ROOT / "final_stimuli_recovery" / "noisy_by_clean" / "results"
DEFAULT_RANDOM_FEATURE_DIR = ROOT / "shared" / "cache_or_heavy" / "natural_pool_subset_10k"
SELECTION_ROOT = selection_root()
ORIENTATION = "noisy_by_clean"
ENV_CONFIG_ROOT = env_config_root()


def run_model_set(
    model_set: str,
    args: argparse.Namespace,
    device: torch.device,
    encoding_root_map: dict[str, Path] | None,
) -> None:
    result_dir = args.selection_root / model_set
    payload = apply_env_paths(
        load_payload(result_dir),
        args.env,
        config_root=ENV_CONFIG_ROOT,
        output_base=SELECTION_ROOT,
    )
    original_models = list(payload["model_names"])

    if args.random_feature_dir is None:
        available_models = original_models
        missing: list[str] = []
        random_feature_source = f"candidate_pool:env-{args.env}" if args.env else "candidate_pool:payload_paths"
    else:
        available_models = available_random_models(args.random_feature_dir, original_models)
        missing = sorted(set(original_models) - set(available_models))
        if missing and args.strict_random_models:
            raise FileNotFoundError(
                f"{model_set}: random feature cache is missing {len(missing)} models: {missing}"
            )
        if missing:
            print(
                f"[{model_set}] WARNING: dropping {len(missing)} models missing from "
                f"{args.random_feature_dir}: {missing}"
            )
        if len(available_models) < 2:
            raise RuntimeError(f"{model_set}: need at least 2 models after random-cache filtering")

        payload = filter_payload_to_models(payload, available_models)
        random_feature_source = f"local_random_pool:{args.random_feature_dir}"

    config_payload = payload.get("config", {})
    metric = args.metric or config_payload.get("metric", "cosine")
    corr_type = args.corr_type or config_payload.get("corr_type", "spearman")
    tracks = [
        track
        for track in get_all_tracks_for_evaluation(payload)
        if track["name"] in args.tracks
    ]
    noise_level_multipliers = get_default_noise_level_multipliers()
    encoding_params_cache: dict[str, Any] = {}
    random_feature_loader = (
        make_random_feature_cache_loader(args.random_feature_dir)
        if args.random_feature_dir is not None
        else None
    )
    feature_loader = partial(
        load_features_for_track,
        random_feature_loader=random_feature_loader,
        log_memory_fn=log_memory,
    )

    out_dir = args.output_root / f"{model_set}_noisy_by_clean_boot"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "model": original_models,
            "included": [model in available_models for model in original_models],
            "reason": [
                "included" if model in available_models else "missing_random_pool_feature"
                for model in original_models
            ],
            "random_feature_source": random_feature_source,
        }
    ).to_csv(out_dir / "model_roster.csv", index=False)

    all_discrim_rows = []
    all_auc_rows = []
    all_noise_rows = []
    all_corr_rows = []
    all_pairwise_rows = []
    all_pairwise_auc_rows = []

    for track_idx, track in enumerate(tracks):
        track_name = track["name"]
        print(f"\n[{model_set}] track {track_idx + 1}/{len(tracks)}: {track_name}")
        try:
            (
                discrim_df,
                correlation_info,
                noise_info,
                auc_info,
                pairwise_df,
                pairwise_auc_info,
            ) = compute_discriminability_for_track(
                payload=payload,
                track=track,
                device=device,
                n_random_subsets=args.n_random_subsets,
                n_random_images=args.n_random_images,
                n_noise_samples=args.n_noise_samples,
                noise_level_multipliers=noise_level_multipliers,
                metric=metric,
                corr_type=corr_type,
                encoding_params_cache=encoding_params_cache,
                selection_variant=args.which_selection,
                encoding_root_map=encoding_root_map,
                feature_loader=feature_loader,
                log_memory_fn=log_memory,
                orientation=ORIENTATION,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
        except Exception:
            print(f"[{model_set}/{track_name}] ERROR")
            raise

        discrim_df["model_set"] = model_set
        discrim_df["recovery_orientation"] = ORIENTATION
        discrim_df["random_feature_source"] = random_feature_source
        discrim_df["n_models"] = len(available_models)
        all_discrim_rows.append(discrim_df)

        pairwise_df["model_set"] = model_set
        pairwise_df["recovery_orientation"] = ORIENTATION
        pairwise_df["random_feature_source"] = random_feature_source
        pairwise_df["n_models"] = len(available_models)
        all_pairwise_rows.append(pairwise_df)

        all_auc_rows.append(
            {
                "track": track_name,
                "model_set": model_set,
                "recovery_orientation": ORIENTATION,
                "random_feature_source": random_feature_source,
                "n_models": len(available_models),
                **{
                    key: value
                    for key, value in auc_info.items()
                    if key != "random_auc_per_subset"
                },
            }
        )
        all_pairwise_auc_rows.append(
            {
                "track": track_name,
                "model_set": model_set,
                "recovery_orientation": ORIENTATION,
                "random_feature_source": random_feature_source,
                "n_models": len(available_models),
                **pairwise_auc_info,
            }
        )
        for model_name, noise_std in noise_info.items():
            all_noise_rows.append(
                {
                    "track": track_name,
                    "model": model_name,
                    "noise_std": noise_std,
                    "recovery_orientation": ORIENTATION,
                }
            )
        append_correlation_matrix_rows(
            all_corr_rows,
            track_name=track_name,
            correlation_info=correlation_info,
            extra_fields={"recovery_orientation": ORIENTATION},
        )

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pd.concat(all_discrim_rows, ignore_index=True).to_csv(
        out_dir / "discriminability.csv", index=False
    )
    pd.DataFrame(all_auc_rows).to_csv(out_dir / "auc_significance.csv", index=False)
    pd.DataFrame(all_noise_rows).to_csv(out_dir / "noise_calibration.csv", index=False)
    pd.DataFrame(all_corr_rows).to_csv(out_dir / "correlation_matrices.csv", index=False)
    pd.concat(all_pairwise_rows, ignore_index=True).to_csv(
        out_dir / "pairwise_margin.csv", index=False
    )
    pd.DataFrame(all_pairwise_auc_rows).to_csv(out_dir / "pairwise_auc.csv", index=False)
    print(f"[{model_set}] saved {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-sets", default=",".join(MODEL_SET_ORDER))
    parser.add_argument("--tracks", default="raw," + ",".join(ENCODING_TRACKS))
    parser.add_argument("--selection-root", type=Path, default=SELECTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--env",
        choices=VALID_ENVS,
        default=None,
        help=(
            "Override payload paths with the repo env config under "
            "00_stimulus_selection/resources/configs/paths/{env}.yaml. "
            "Use --env raven for the full Raven candidate-pool rerun."
        ),
    )
    parser.add_argument(
        "--random-feature-dir",
        type=Path,
        default=None,
        help=(
            "Optional debug-only local .npz random-feature cache. Omit this for the "
            "proper candidate-pool baseline loaded from the payload/env paths. "
            f"Local smoke-test cache, if present: {DEFAULT_RANDOM_FEATURE_DIR}"
        ),
    )
    parser.add_argument(
        "--strict-random-models",
        action="store_true",
        help="Fail if --random-feature-dir is provided and is missing any selected model.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-random-subsets", type=int, default=50)
    parser.add_argument(
        "--n-random-images",
        type=int,
        default=10000,
        help="Number of random baseline images to load from candidate pool/cache before subset sampling.",
    )
    parser.add_argument("--n-noise-samples", type=int, default=DEFAULT_N_NOISE_SAMPLES)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument(
        "--which-selection",
        choices=["final", "greedy", "best_raw_combined"],
        default="final",
    )
    parser.add_argument("--unique-encodings", action="store_true", default=True)
    parser.add_argument("--shared-encodings", action="store_false", dest="unique_encodings")
    parser.add_argument("--metric", default=None)
    parser.add_argument("--corr-type", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_sets = [item.strip() for item in args.model_sets.split(",") if item.strip()]
    args.tracks = [item.strip() for item in args.tracks.split(",") if item.strip()]
    args.selection_root = args.selection_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.random_feature_dir is not None:
        args.random_feature_dir = args.random_feature_dir.resolve()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.random_feature_dir is not None and not args.random_feature_dir.exists():
        raise FileNotFoundError(f"Random feature directory not found: {args.random_feature_dir}")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if args.random_feature_dir is None:
        print("Using candidate-pool random baseline from payload/env paths")
    else:
        print(f"Using debug local random-feature cache: {args.random_feature_dir}")

    encoding_root_map = None
    if args.unique_encodings:
        encoding_root_map = {key: Path(value) for key, value in paths.unique_encoding_dirs().items()}
        print(f"Using unique encoding roots: {list(encoding_root_map)}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    for model_set in tqdm(args.model_sets, desc="Model sets"):
        run_model_set(model_set, args, device, encoding_root_map)


if __name__ == "__main__":
    main()
