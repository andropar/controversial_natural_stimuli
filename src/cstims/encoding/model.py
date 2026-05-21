"""Linear encoding model with brain space mapping and ROI filtering.

This module provides the LinearEncodingModel class for managing encoding models
that map visual features to brain responses, with support for:
- Brain space visualization (NIfTI export)
- ROI-based filtering (visual, hlvis)
- Backwards-compatible export for use with existing selection pipelines
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import nibabel as nib
import numpy as np

_LOG = logging.getLogger(__name__)

__all__ = ["LinearEncodingModel"]


@dataclass
class LinearEncodingModel:
    """Linear encoding model with brain space mapping.

    This class encapsulates a fitted linear encoding model (y = X @ W + b) along with
    all metadata needed for brain space visualization and ROI filtering.

    Attributes:
        weights: Linear weights matrix (n_features, n_voxels)
        intercept: Bias/intercept vector (n_voxels,)
        alphas: Per-voxel regularization parameters (n_voxels,)
        feature_mean: Mean used to standardize features during training (n_features,)
        feature_scale: Scale used to standardize features during training (n_features,)
        volume_shape: Shape of the brain volume (x, y, z)
        affine: 4x4 NIfTI affine transformation matrix
        voxel_indices: Flat indices into the volume for each voxel (n_voxels,)
        roi_masks: Dict mapping ROI names to boolean masks
        subject: Subject identifier (e.g., 'sub-01')
        model_name: Name of the feature extraction model
        layer: Layer identifier for the feature extraction model
        source: Source of the model (e.g., 'timm', 'torchvision')
        cve_threshold: Cross-validated effect threshold used for voxel selection
        metrics: Dict of evaluation metrics (veRSA, voxel_r, etc.)
    """

    # Model parameters
    weights: np.ndarray  # (n_features, n_voxels)
    intercept: np.ndarray  # (n_voxels,)
    alphas: np.ndarray  # (n_voxels,)

    # Feature scaling (for reference)
    feature_mean: np.ndarray  # (n_features,)
    feature_scale: np.ndarray  # (n_features,)

    # Brain space mapping
    volume_shape: Tuple[int, int, int]
    affine: np.ndarray  # (4, 4)
    voxel_indices: np.ndarray  # (n_voxels,) flat indices

    # ROI membership
    roi_masks: Dict[str, np.ndarray]  # {"visual": bool array, "hlvis": bool array}

    # Metadata
    subject: str
    model_name: str
    layer: Union[str, int]
    source: str
    cve_threshold: float
    metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Validate shapes and types after initialization."""
        n_features, n_voxels = self.weights.shape
        assert self.intercept.shape == (n_voxels,), f"intercept shape mismatch: {self.intercept.shape}"
        assert self.alphas.shape == (n_voxels,), f"alphas shape mismatch: {self.alphas.shape}"
        assert self.feature_mean.shape == (n_features,), f"feature_mean shape mismatch"
        assert self.feature_scale.shape == (n_features,), f"feature_scale shape mismatch"
        assert self.voxel_indices.shape == (n_voxels,), f"voxel_indices shape mismatch"
        assert self.affine.shape == (4, 4), f"affine shape mismatch: {self.affine.shape}"

        for roi_name, mask in self.roi_masks.items():
            assert mask.shape == (n_voxels,), f"ROI mask '{roi_name}' shape mismatch"

    @property
    def n_features(self) -> int:
        """Number of input features."""
        return self.weights.shape[0]

    @property
    def n_voxels(self) -> int:
        """Total number of voxels in the model."""
        return self.weights.shape[1]

    def n_voxels_roi(self, roi: str = "visual") -> int:
        """Number of voxels in a specific ROI."""
        if roi not in self.roi_masks:
            raise ValueError(f"Unknown ROI '{roi}'. Available: {list(self.roi_masks.keys())}")
        return int(self.roi_masks[roi].sum())

    # =========================================================================
    # Prediction methods
    # =========================================================================

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict brain responses for all voxels.

        Args:
            features: Input features of shape (n_images, n_features)

        Returns:
            Predicted responses of shape (n_images, n_voxels)
        """
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[1] != self.n_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.n_features}, got {features.shape[1]}"
            )
        return features @ self.weights + self.intercept

    # =========================================================================
    # ROI filtering methods
    # =========================================================================

    def get_roi_mask(self, roi: str = "visual") -> np.ndarray:
        """Get boolean mask for ROI.

        Args:
            roi: ROI name ('visual' or 'hlvis')

        Returns:
            Boolean array of shape (n_voxels,)
        """
        if roi not in self.roi_masks:
            raise ValueError(f"Unknown ROI '{roi}'. Available: {list(self.roi_masks.keys())}")
        return self.roi_masks[roi]

    # =========================================================================
    # Brain space mapping methods
    # =========================================================================

    def to_volume(self, voxel_values: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
        """Map 1D voxel values to 3D brain volume.

        Args:
            voxel_values: Array of shape (n_voxels,)
            fill_value: Value for voxels not in the mask

        Returns:
            3D numpy array of shape volume_shape
        """
        if len(voxel_values) != self.n_voxels:
            raise ValueError(
                f"voxel_values length ({len(voxel_values)}) doesn't match "
                f"n_voxels ({self.n_voxels})"
            )
        volume = np.full(self.volume_shape, fill_value, dtype=np.float32)
        volume.flat[self.voxel_indices] = voxel_values
        return volume

    def to_nifti(
        self,
        voxel_values: np.ndarray,
        output_path: Optional[Union[str, Path]] = None,
        fill_value: float = 0.0,
    ) -> nib.Nifti1Image:
        """Create NIfTI image from voxel values.

        Args:
            voxel_values: Array of shape (n_voxels,)
            output_path: If provided, save to this path
            fill_value: Value for voxels not in the mask

        Returns:
            NIfTI image object
        """
        volume = self.to_volume(voxel_values, fill_value=fill_value)
        img = nib.Nifti1Image(volume, affine=self.affine)
        if output_path is not None:
            nib.save(img, str(output_path))
            _LOG.info(f"Saved NIfTI to {output_path}")
        return img

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: Union[str, Path]) -> None:
        """Save model to NPZ file.

        Args:
            path: Output path (should end with .npz)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare metadata as JSON string
        metadata = {
            "subject": self.subject,
            "model_name": self.model_name,
            "layer": str(self.layer),
            "source": self.source,
            "cve_threshold": self.cve_threshold,
            "metrics": self.metrics,
            "n_voxels": self.n_voxels,
            "n_features": self.n_features,
            "volume_shape": list(self.volume_shape),
            "roi_names": list(self.roi_masks.keys()),
        }

        # Save arrays
        save_dict = {
            "weights": self.weights.astype(np.float32),
            "intercept": self.intercept.astype(np.float32),
            "alphas": self.alphas.astype(np.float32),
            "feature_mean": self.feature_mean.astype(np.float32),
            "feature_scale": self.feature_scale.astype(np.float32),
            "affine": self.affine.astype(np.float64),
            "voxel_indices": self.voxel_indices.astype(np.int64),
            "metadata": json.dumps(metadata),
        }

        # Add ROI masks
        for roi_name, mask in self.roi_masks.items():
            save_dict[f"roi_{roi_name}"] = mask.astype(np.bool_)

        np.savez_compressed(path, **save_dict)
        _LOG.info(f"Saved LinearEncodingModel to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LinearEncodingModel":
        """Load model from NPZ file.

        Args:
            path: Path to NPZ file

        Returns:
            LinearEncodingModel instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with np.load(path, allow_pickle=True) as z:
            # Load metadata
            metadata = json.loads(str(z["metadata"]))

            # Load ROI masks
            roi_names = metadata.get("roi_names", ["visual", "hlvis"])
            roi_masks = {}
            for roi_name in roi_names:
                key = f"roi_{roi_name}"
                if key in z:
                    roi_masks[roi_name] = z[key].astype(np.bool_)

            # Handle legacy files that might not have all ROI masks
            if "visual" not in roi_masks:
                n_voxels = z["weights"].shape[1]
                roi_masks["visual"] = np.ones(n_voxels, dtype=np.bool_)

            model = cls(
                weights=z["weights"].astype(np.float32),
                intercept=z["intercept"].astype(np.float32),
                alphas=z["alphas"].astype(np.float32),
                feature_mean=z["feature_mean"].astype(np.float32),
                feature_scale=z["feature_scale"].astype(np.float32),
                volume_shape=tuple(metadata["volume_shape"]),
                affine=z["affine"].astype(np.float64),
                voxel_indices=z["voxel_indices"].astype(np.int64),
                roi_masks=roi_masks,
                subject=metadata["subject"],
                model_name=metadata["model_name"],
                layer=metadata["layer"],
                source=metadata["source"],
                cve_threshold=metadata["cve_threshold"],
                metrics=metadata.get("metrics", {}),
            )

        _LOG.info(f"Loaded LinearEncodingModel from {path}: {model.n_features} features, {model.n_voxels} voxels")
        return model

    def __repr__(self) -> str:
        return (
            f"LinearEncodingModel("
            f"model={self.model_name}, "
            f"layer={self.layer}, "
            f"subject={self.subject}, "
            f"n_features={self.n_features}, "
            f"n_voxels={self.n_voxels}, "
            f"n_voxels_hlvis={self.n_voxels_roi('hlvis')})"
        )
