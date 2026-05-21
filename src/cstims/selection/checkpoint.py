"""Checkpoint functionality for resumable stimulus selection."""

from __future__ import annotations

import pickle
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set

import numpy as np


@dataclass
class SelectionCheckpoint:
    """Minimal state needed to resume stimulus selection.

    Track states are reconstructed from current_indices on resume
    using _initialize_track_states(), so we don't serialize tensors.
    """

    # Phase tracking
    phase: Literal["greedy", "refinement", "complete"]
    greedy_iteration: int  # number of greedy iterations completed
    refinement_pass: int = -1  # -1 if not in refinement
    refinement_position: int = -1  # -1 if not started

    # Core selection state (minimal)
    current_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    failed_indices: Set[int] = field(default_factory=set)

    # Scores history (for logging/analysis continuity)
    scores_combined: List[float] = field(default_factory=list)
    scores_per_track_history: Dict[str, List[float]] = field(default_factory=dict)

    # Refinement history
    refinement_history: List[Dict] = field(default_factory=list)

    # Noise calibration (expensive to recompute)
    var_noise_raw: Optional[Dict[str, float]] = None
    var_noise_by_encoding: Dict[str, Dict[str, float]] = field(default_factory=dict)


def save_checkpoint(path: Path, checkpoint: SelectionCheckpoint) -> None:
    """Save checkpoint to file with backup of previous checkpoint.

    Creates checkpoint.pkl and keeps previous version as checkpoint.pkl.bak
    """
    path = Path(path)

    # Backup existing checkpoint
    if path.exists():
        backup_path = path.with_suffix(".pkl.bak")
        shutil.copy2(path, backup_path)

    # Write new checkpoint
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(path: Path) -> SelectionCheckpoint:
    """Load checkpoint from file."""
    with open(path, "rb") as f:
        return pickle.load(f)
