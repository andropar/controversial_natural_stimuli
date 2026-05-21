"""
I/O functions for evaluation (loading/saving data).
"""

import pickle
from pathlib import Path


def load_payload(result_dir: Path) -> dict:
    """
    Load payload from result directory with proper error handling.

    Args:
        result_dir: Path to result directory

    Returns:
        Payload dictionary

    Raises:
        FileNotFoundError: If payload file doesn't exist
        ValueError: If payload is invalid or missing required keys
        RuntimeError: If unexpected error occurs during loading
    """
    payload_path = result_dir / "selected_stimuli_data.pkl"

    if not payload_path.exists():
        raise FileNotFoundError(
            f"Payload not found: {payload_path}. "
            f"Expected file in result directory: {result_dir}"
        )

    try:
        with open(payload_path, "rb") as f:
            payload = pickle.load(f)
    except pickle.UnpicklingError as e:
        raise ValueError(f"Failed to unpickle payload: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading payload: {e}")

    required_keys = ["config", "model_names"]
    missing = [k for k in required_keys if k not in payload]
    if missing:
        raise ValueError(f"Payload missing required keys: {missing}")

    return payload
