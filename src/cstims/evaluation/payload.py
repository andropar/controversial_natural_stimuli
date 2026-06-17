"""Selection-payload helpers used by evaluation scripts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
import warnings

import yaml

from cstims.paths import find_share_root, model_list_csv, resources_dir


VALID_ENVS = ("iris", "raven")


def selection_root() -> Path:
    """Default selected-stimuli result root for an in-place checkout."""
    return find_share_root() / "00_stimulus_selection" / "results" / "selected_stimuli"


def env_config_root() -> Path:
    """Default root for machine-specific path override YAML files."""
    return resources_dir() / "configs" / "paths"


def warn_path_divergence(old_paths: dict, new_paths: dict, env: str) -> None:
    """Warn if payload and env config appear to point at different resources."""
    keys_to_check = ["subset_root", "model_list_csv", "encoding_root"]
    for key in keys_to_check:
        old = old_paths.get(key)
        new = new_paths.get(key)
        if not old or not new:
            continue

        old_path = Path(old)
        new_path = Path(new)
        if old_path.name != new_path.name:
            warnings.warn(
                f"Path '{key}' basename differs: payload='{old_path.name}', "
                f"env={env}='{new_path.name}'. This may cause issues."
            )
        elif old_path.parent.name != new_path.parent.name:
            print(
                f"  Note: '{key}' parent dir differs "
                f"(payload='{old_path.parent.name}', env={env}='{new_path.parent.name}')"
            )


def load_repo_env_paths(
    env: str,
    *,
    config_root: Path | None = None,
    local_model_csv: Path | None = None,
    output_base: Path | None = None,
) -> dict[str, Any]:
    """Load repo env path overrides and pin repo-local metadata paths."""
    root = config_root or env_config_root()
    config_path = root / f"{env}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Environment config not found: {config_path}")

    with config_path.open() as f:
        config = yaml.safe_load(f) or {}

    paths = dict(config.get("paths", {}))
    csv_path = local_model_csv or model_list_csv()
    if csv_path.exists():
        paths["model_list_csv"] = str(csv_path)
    paths["output_base"] = str(output_base or selection_root())
    return paths


def apply_env_paths(
    payload: dict,
    env: str | None,
    *,
    config_root: Path | None = None,
    local_model_csv: Path | None = None,
    output_base: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Return a payload whose config paths are overridden for the given env."""
    if not env:
        return payload

    env_paths = load_repo_env_paths(
        env,
        config_root=config_root,
        local_model_csv=local_model_csv,
        output_base=output_base,
    )
    payload = dict(payload)
    config = dict(payload.get("config", {}))
    old_paths = dict(config.get("paths", {}))
    warn_path_divergence(old_paths, env_paths, env)
    config["paths"] = {**old_paths, **env_paths}
    payload["config"] = config
    if verbose:
        root = config_root or env_config_root()
        print(f"Using paths from env={env}: {root / f'{env}.yaml'}")
    return payload


def _filter_model_dict(data: Any, keep_models: Sequence[str]) -> Any:
    if not isinstance(data, dict):
        return data
    return {model: data[model] for model in keep_models if model in data}


def filter_payload_to_models(payload: dict, keep_models: Sequence[str]) -> dict:
    """Filter model-indexed payload fields to a requested model order."""
    keep_models = list(keep_models)
    payload = dict(payload)
    payload["model_names"] = keep_models

    for key in [
        "selected_features_raw",
        "greedy_features_raw",
        "best_raw_combined_features_raw",
        "selected_features",
    ]:
        if key in payload:
            payload[key] = _filter_model_dict(payload[key], keep_models)

    for key in [
        "selected_features_by_view",
        "selected_features_by_encoding",
        "greedy_features_by_encoding",
        "best_raw_combined_features_by_encoding",
    ]:
        if isinstance(payload.get(key), dict):
            payload[key] = {
                track: _filter_model_dict(features, keep_models)
                for track, features in payload[key].items()
            }

    for key in ["var_noise_by_model", "selection_objective_var_noise_by_model"]:
        if isinstance(payload.get(key), dict):
            payload[key] = {
                track: _filter_model_dict(noise_by_model, keep_models)
                for track, noise_by_model in payload[key].items()
            }

    return payload
