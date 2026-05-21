"""
Image quality filter for stimulus selection.

This module provides filtering of candidate stimuli based on:
1. Successful download from original LAION URL
2. Image resolution (>= min_resolution on at least one side)
3. Natural image classifier confidence (P(natural) >= threshold)
"""

from __future__ import annotations

import json
import logging
import pickle
import tarfile
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from cstims.data_loader import FeatureShardSlice

_LOG = logging.getLogger(__name__)


def _center_crop_to_square(img: Image.Image) -> Image.Image:
    """Center crop an image to a square."""
    width, height = img.size
    new_size = min(width, height)
    left = (width - new_size) // 2
    top = (height - new_size) // 2
    right = left + new_size
    bottom = top + new_size
    return img.crop((left, top, right, bottom))


@dataclass
class ValidationResult:
    """Result of validating a single candidate image."""

    passed: bool
    reason: str  # "passed", "resolution_too_low", "download_failed", etc.
    shard_name: Optional[str] = None
    image_name: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    natural_prob: Optional[float] = None
    saved_path: Optional[str] = None  # Path where image was saved (if applicable)


@dataclass
class FilterRecord:
    """Record of a single image filter evaluation."""

    global_idx: int
    passed: bool
    reason: str  # "passed", "resolution_too_low", "download_failed", etc.

    # Image metadata
    shard_name: Optional[str] = None
    image_name: Optional[str] = None

    # Quality metrics (captured even for passed images when available)
    width: Optional[int] = None
    height: Optional[int] = None
    natural_prob: Optional[float] = None

    # Selection context
    score: float = 0.0  # Combined z-scored score
    scores_per_track: Optional[Dict[str, float]] = None  # Raw per-track scores (before z-scoring)
    rank: int = 0  # 1-indexed: 1 = top candidate

    # Iteration context
    phase: str = "greedy"  # or "refinement"
    iteration: int = 0
    refinement_position: Optional[int] = None

    # Saved image path (if image was saved during validation)
    saved_path: Optional[str] = None


@dataclass
class ImageFilterConfig:
    """Configuration for image quality filtering during selection."""

    enabled: bool = False
    min_resolution: int = 1000  # Min pixels on at least one side
    natural_prob_threshold: float = 0.85  # P(natural) from classifier
    download_timeout: float = 10.0  # Seconds
    max_attempts_per_iteration: int = 100  # Max candidates to try before fallback
    classifier_path: Optional[Path] = None  # Path to pickled sklearn classifier
    save_dir: Optional[Path] = None  # Directory to save validated images (center-cropped)
    parallel_batch_size: int = 10  # Number of candidates to validate in parallel


@dataclass
class ImageFilter:
    """
    Filters candidate images based on quality criteria.

    Checks:
    1. Image can be downloaded from LAION URL
    2. Downloaded image has resolution >= min_resolution on at least one side
    3. Natural image classifier gives P(natural) >= threshold
    """

    config: ImageFilterConfig
    shard_slices: Sequence[FeatureShardSlice]
    subset_root: Path  # Root directory containing tar files and features

    # Internal state
    _classifier: Any = field(default=None, repr=False)
    _classifier_loaded: bool = field(default=False, repr=False)
    _shard_offsets: List[int] = field(default_factory=list, repr=False)
    _clip_cache: Dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    _failed_indices: Set[int] = field(default_factory=set, repr=False)
    _filter_records: List[FilterRecord] = field(default_factory=list, repr=False)

    def __post_init__(self):
        self._shard_offsets = self._compute_shard_offsets()

    def _compute_shard_offsets(self) -> List[int]:
        """Compute cumulative offsets for binary search into shards."""
        offsets = [0]
        for shard in self.shard_slices:
            offsets.append(shard.end_index)
        return offsets

    def _load_classifier(self) -> None:
        """Load the natural image classifier from disk."""
        if self._classifier_loaded:
            return
        if self.config.classifier_path is None:
            _LOG.warning("No classifier path configured; skipping classifier check")
            self._classifier_loaded = True
            return
        clf_path = Path(self.config.classifier_path)
        if not clf_path.exists():
            _LOG.warning("Classifier not found at %s; skipping classifier check", clf_path)
            self._classifier_loaded = True
            return
        _LOG.info("Loading natural image classifier from %s", clf_path)
        with open(clf_path, "rb") as f:
            self._classifier = pickle.load(f)
        self._classifier_loaded = True
        _LOG.info("Classifier loaded successfully")

    def _index_to_shard_info(self, global_idx: int) -> Tuple[int, FeatureShardSlice, int]:
        """
        Map a global index to (shard_index, shard, local_index).
        """
        shard_pos = bisect_right(self._shard_offsets, global_idx) - 1
        if shard_pos < 0 or shard_pos >= len(self.shard_slices):
            raise IndexError(f"Global index {global_idx} out of range")
        shard = self.shard_slices[shard_pos]
        local_idx = global_idx - shard.start_index
        return shard_pos, shard, local_idx

    def _get_image_name(self, shard: FeatureShardSlice, local_idx: int) -> Optional[str]:
        """Get the image filename from tar for a given local index."""
        tar_path = shard.tar_path
        if not tar_path.exists():
            _LOG.warning("Tar file not found: %s", tar_path)
            return None
        try:
            with tarfile.open(tar_path, "r:*", ignore_zeros=True) as tf:
                names = tf.getnames()
            # Filter to valid images (those with matching .json metadata)
            name_set = set(names)
            images = [n for n in names if n.endswith(".jpg") and (n[:-4] + ".json") in name_set]
            # Exclude failed images
            if shard.failed_images:
                failed_set = set(shard.failed_images)
                images = [n for n in images if n not in failed_set]
            if local_idx >= len(images):
                _LOG.warning(
                    "Local index %d out of range for shard %s (has %d images)",
                    local_idx, tar_path, len(images)
                )
                return None
            return images[local_idx]
        except Exception as exc:
            _LOG.warning("Failed to read tar %s: %s", tar_path, exc)
            return None

    def _load_url_from_tar_metadata(
        self, tar_path: Path, image_name: str
    ) -> Optional[str]:
        """Load LAION URL from JSON metadata inside the tar shard."""
        json_name = f"{Path(image_name).stem}.json"
        try:
            with tarfile.open(tar_path, "r:*") as tar:
                ef_meta = tar.extractfile(json_name)
                if ef_meta is None:
                    _LOG.debug("Metadata %s not found in %s", json_name, tar_path)
                    return None
                meta_bytes = ef_meta.read()
        except Exception as exc:
            _LOG.debug("Failed to read metadata %s from %s: %s", json_name, tar_path, exc)
            return None

        try:
            meta = json.loads(meta_bytes.decode("utf-8"))
        except Exception as exc:
            _LOG.debug("Failed to parse metadata %s: %s", json_name, exc)
            return None

        url = meta.get("url") or meta.get("URL")
        if isinstance(url, str) and url:
            return url
        return None

    def _download_image(self, url: str) -> Optional[Image.Image]:
        """Download image from URL."""
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=self.config.download_timeout) as resp:
                data = resp.read()
        except (URLError, HTTPError, TimeoutError) as exc:
            _LOG.debug("HTTP error downloading %s: %s", url, exc)
            return None
        except Exception as exc:
            _LOG.debug("Unexpected error downloading %s: %s", url, exc)
            return None

        try:
            img = Image.open(BytesIO(data)).convert("RGB")
            return img
        except Exception as exc:
            _LOG.debug("Failed to decode image from %s: %s", url, exc)
            return None

    def _check_resolution(self, img: Image.Image) -> Tuple[bool, int, int]:
        """Check if image meets minimum resolution requirement.

        Returns:
            (passed, width, height)
        """
        width, height = img.size
        passed = max(width, height) >= self.config.min_resolution
        return passed, width, height

    def _load_clip_features_for_shard(self, shard_pos: int) -> Optional[np.ndarray]:
        """Load CLIP visual features for a shard, with caching."""
        if shard_pos in self._clip_cache:
            return self._clip_cache[shard_pos]

        shard = self.shard_slices[shard_pos]
        tar_path = shard.tar_path
        stem = tar_path.stem

        # Try to find CLIP features file
        feature_dir = tar_path.parent
        clip_path = feature_dir / f"{stem}.clip.visual.npz"

        if not clip_path.exists():
            # Try alternate locations
            clip_path_npy = feature_dir / f"{stem}.clip.visual.npy"
            if clip_path_npy.exists():
                clip_path = clip_path_npy
            else:
                _LOG.warning("CLIP features not found for shard %s", tar_path)
                return None

        try:
            if clip_path.suffix == ".npz":
                data = np.load(clip_path)
                # Try common keys
                if "arr_0" in data:
                    features = data["arr_0"]
                elif "features" in data:
                    features = data["features"]
                else:
                    # Use first available key
                    keys = list(data.keys())
                    if keys:
                        features = data[keys[0]]
                    else:
                        _LOG.warning("No arrays in %s", clip_path)
                        return None
            else:
                features = np.load(clip_path)

            self._clip_cache[shard_pos] = features
            return features
        except Exception as exc:
            _LOG.warning("Failed to load CLIP features from %s: %s", clip_path, exc)
            return None

    def _check_classifier(self, shard_pos: int, local_idx: int) -> Tuple[bool, Optional[float]]:
        """Check if classifier gives P(natural) >= threshold.

        Returns:
            (passed, natural_prob): Whether passed, and the probability (None if couldn't compute)
        """
        self._load_classifier()
        if self._classifier is None:
            # No classifier available, pass by default
            return True, None

        clip_features = self._load_clip_features_for_shard(shard_pos)
        if clip_features is None:
            _LOG.debug("No CLIP features for shard %d; skipping classifier check", shard_pos)
            return True, None  # Pass if we can't check

        if local_idx >= len(clip_features):
            _LOG.warning(
                "Local index %d out of range for CLIP features (has %d)",
                local_idx, len(clip_features)
            )
            return True, None  # Pass if index out of range

        feature_vector = clip_features[local_idx : local_idx + 1]
        try:
            probs = self._classifier.predict_proba(feature_vector)
            natural_prob = float(probs[0, 1])  # P(natural) is class 1
            passed = natural_prob >= self.config.natural_prob_threshold
            if not passed:
                _LOG.info(
                    "Classifier check failed: P(natural)=%.3f < %.3f",
                    natural_prob, self.config.natural_prob_threshold
                )
            return passed, natural_prob
        except Exception as exc:
            _LOG.warning("Classifier prediction failed: %s", exc)
            return True, None  # Pass on error

    def mark_candidate_failed(self, global_idx: int) -> None:
        """Mark a candidate as failed (will be excluded from future checks)."""
        self._failed_indices.add(global_idx)

    def validate_candidate(self, global_idx: int) -> ValidationResult:
        """
        Validate a candidate image.

        Returns:
            ValidationResult with all available metadata about the validation.
        """
        if global_idx in self._failed_indices:
            return ValidationResult(passed=False, reason="previously_failed")

        # Get shard info
        try:
            shard_pos, shard, local_idx = self._index_to_shard_info(global_idx)
        except IndexError as exc:
            self.mark_candidate_failed(global_idx)
            return ValidationResult(passed=False, reason=f"index_error: {exc}")

        shard_name = shard.tar_path.stem

        # Get image name
        image_name = self._get_image_name(shard, local_idx)
        if image_name is None:
            self.mark_candidate_failed(global_idx)
            return ValidationResult(
                passed=False, reason="image_name_not_found", shard_name=shard_name
            )

        tar_path = shard.tar_path

        # Step 1: Get URL and download
        url = self._load_url_from_tar_metadata(tar_path, image_name)
        if url is None:
            self.mark_candidate_failed(global_idx)
            return ValidationResult(
                passed=False,
                reason="url_not_found",
                shard_name=shard_name,
                image_name=image_name,
            )

        img = self._download_image(url)
        if img is None:
            self.mark_candidate_failed(global_idx)
            return ValidationResult(
                passed=False,
                reason="download_failed",
                shard_name=shard_name,
                image_name=image_name,
            )

        # Step 2: Check resolution
        res_passed, width, height = self._check_resolution(img)
        if not res_passed:
            self.mark_candidate_failed(global_idx)
            return ValidationResult(
                passed=False,
                reason="resolution_too_low",
                shard_name=shard_name,
                image_name=image_name,
                width=width,
                height=height,
            )

        # Step 3: Check classifier
        clf_passed, natural_prob = self._check_classifier(shard_pos, local_idx)
        if not clf_passed:
            self.mark_candidate_failed(global_idx)
            return ValidationResult(
                passed=False,
                reason="classifier_below_threshold",
                shard_name=shard_name,
                image_name=image_name,
                width=width,
                height=height,
                natural_prob=natural_prob,
            )

        # Step 4: Save image if save_dir is configured
        saved_path: Optional[str] = None
        if self.config.save_dir is not None:
            try:
                self.config.save_dir.mkdir(parents=True, exist_ok=True)
                cropped = _center_crop_to_square(img)
                save_file = self.config.save_dir / f"{global_idx}.png"
                cropped.save(save_file)
                saved_path = str(save_file)
                _LOG.debug("Saved validated image to %s", save_file)
            except Exception as exc:
                _LOG.warning("Failed to save image %d: %s", global_idx, exc)

        prob_str = f"{natural_prob:.3f}" if natural_prob is not None else "N/A"
        _LOG.info(
            "Candidate %d passed: %s (%dx%d, P(natural)=%s)",
            global_idx,
            image_name,
            width,
            height,
            prob_str,
        )
        return ValidationResult(
            passed=True,
            reason="passed",
            shard_name=shard_name,
            image_name=image_name,
            width=width,
            height=height,
            natural_prob=natural_prob,
            saved_path=saved_path,
        )

    def select_first_valid(
        self,
        candidate_indices: np.ndarray,
        candidate_scores: np.ndarray,
        candidate_scores_per_track: Optional[Dict[str, np.ndarray]] = None,
        phase: str = "greedy",
        iteration: int = 0,
        refinement_position: Optional[int] = None,
    ) -> Tuple[int, float, int]:
        """
        Select the best valid candidate from a sorted list using parallel validation.

        Validates candidates in parallel batches, then returns the highest-scoring
        one that passed (not just the first to finish).

        Args:
            candidate_indices: Array of candidate global indices, sorted by score (descending)
            candidate_scores: Array of corresponding combined z-scored scores
            candidate_scores_per_track: Optional dict of raw per-track scores (before z-scoring),
                keyed by track name, each value is array sorted same as candidate_indices
            phase: "greedy" or "refinement"
            iteration: Greedy iteration number (0-indexed)
            refinement_position: Position being refined (if phase == "refinement")

        Returns:
            (selected_idx, selected_score, attempts): The selected index, its score,
            and how many candidates were checked.
        """
        max_attempts = min(self.config.max_attempts_per_iteration, len(candidate_indices))
        batch_size = self.config.parallel_batch_size
        total_checked = 0

        # Process in batches
        for batch_start in range(0, max_attempts, batch_size):
            batch_end = min(batch_start + batch_size, max_attempts)
            batch_indices = list(range(batch_start, batch_end))

            # Validate batch in parallel
            batch_results: List[Tuple[int, int, float, ValidationResult, Optional[Dict[str, float]]]] = []

            def validate_one(pos: int) -> Tuple[int, int, float, ValidationResult, Optional[Dict[str, float]]]:
                idx = int(candidate_indices[pos])
                score = float(candidate_scores[pos])
                scores_per_track: Optional[Dict[str, float]] = None
                if candidate_scores_per_track is not None:
                    scores_per_track = {
                        name: float(scores[pos].item())
                        for name, scores in candidate_scores_per_track.items()
                    }
                result = self.validate_candidate(idx)
                return pos, idx, score, result, scores_per_track

            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {executor.submit(validate_one, pos): pos for pos in batch_indices}
                for future in as_completed(futures):
                    try:
                        batch_results.append(future.result())
                    except Exception as e:
                        _LOG.warning("Validation thread failed: %s", e)

            # Sort by position (to maintain score ranking)
            batch_results.sort(key=lambda x: x[0])

            # Record all results and find best passing candidate
            best_passing: Optional[Tuple[int, int, float]] = None  # (pos, idx, score)

            for pos, idx, score, result, scores_per_track in batch_results:
                rank = pos + 1  # 1-indexed
                total_checked += 1

                # Record this evaluation (skip "previously_failed")
                if result.reason != "previously_failed":
                    record = FilterRecord(
                        global_idx=idx,
                        passed=result.passed,
                        reason=result.reason,
                        shard_name=result.shard_name,
                        image_name=result.image_name,
                        width=result.width,
                        height=result.height,
                        natural_prob=result.natural_prob,
                        score=score,
                        scores_per_track=scores_per_track,
                        rank=rank,
                        phase=phase,
                        iteration=iteration,
                        refinement_position=refinement_position,
                        saved_path=result.saved_path,
                    )
                    self._filter_records.append(record)

                if result.passed:
                    _LOG.debug("Candidate %d (score=%.4f, rank=%d) passed", idx, score, rank)
                    if best_passing is None or pos < best_passing[0]:
                        best_passing = (pos, idx, score)
                else:
                    _LOG.debug("Candidate %d (score=%.4f) rejected: %s", idx, score, result.reason)

            # Return best passing candidate from this batch
            if best_passing is not None:
                pos, idx, score = best_passing
                _LOG.info(
                    "Selected candidate %d (score=%.4f, rank=%d) after checking %d candidate(s)",
                    idx, score, pos + 1, total_checked
                )
                return idx, score, total_checked

        # Fallback to best scoring candidate
        _LOG.warning(
            "No valid candidate found in %d attempts; falling back to best scoring (idx=%d)",
            total_checked, int(candidate_indices[0])
        )
        return int(candidate_indices[0]), float(candidate_scores[0]), total_checked

    @property
    def num_failed(self) -> int:
        """Number of candidates marked as failed."""
        return len(self._failed_indices)

    @property
    def filter_records(self) -> List[FilterRecord]:
        """All filter evaluation records."""
        return self._filter_records
