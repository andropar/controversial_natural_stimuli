"""DeepVision fMRI benchmark data loader.

Loads DeepVision shared stimulus images and voxel beta responses for encoding model fitting.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from scipy.stats import zscore

from cstims.paths import deepvision_fmri_source_root as default_deepvision_fmri_root

_LOG = logging.getLogger(__name__)

__all__ = [
    "DeepVisionBenchmark",
    "correct_stimulus_label",
    "parse_stimulus_label",
]

# Default threshold for cross-validated effect mask
DEFAULT_CVE_THRESHOLD = 0.2

# Subject ID to participant code mapping
SUBJECT_TO_PARTICIPANT = {
    "sub-01": "p01",
    "sub-03": "p02",
    "sub-05": "p03",
    "sub-06": "p04",
    "sub-07": "p05",
}

# Unique images were assigned with a different participant mapping for sub-03/sub-06
SUBJECT_TO_UNIQUE_PARTICIPANT = {
    "sub-01": "p01",
    "sub-03": "p04",
    "sub-05": "p03",
    "sub-06": "p02",
    "sub-07": "p05",
}


def correct_stimulus_label(label: str) -> str:
    """Correct the swapped dataset/architecture labels in cstim trial_info."""
    if label.startswith("shared_4rep_LAION_controversial_dataset_"):
        return label.replace("_dataset_", "_architecture_")
    if label.startswith("shared_4rep_LAION_controversial_architecture_"):
        return label.replace("_architecture_", "_dataset_")
    return label


def parse_stimulus_label(label: str) -> Tuple[str, Optional[int]]:
    """Parse a DeepVision/CSTIM trial label into ``(group, index)``."""
    if label == "blank":
        return "blank", None

    label = correct_stimulus_label(label)

    if "vicco" in label:
        idx = int(label.split("_")[-1].replace(".jpg", ""))
        return "vicco", idx
    if "controversial" in label:
        label_clean = label.replace(".jpg", "")
        idx_part = label_clean.split("_i")[-1]
        idx = int(idx_part)
        after_controversial = label_clean.split("controversial_")[1]
        model_set = after_controversial.rsplit("_i", 1)[0]
        if model_set == "all":
            model_set = "all_models"
        elif model_set == "training":
            model_set = "training_objective"
        return model_set, idx
    raise ValueError(f"Unknown label format: {label}")


def _read_binary_img(img_bytes) -> Image.Image:
    """Decode HDF5-stored image bytes to PIL Image."""
    return Image.open(io.BytesIO(np.array(img_bytes).tobytes()))


def _load_session_betas(h5_path: Path, expected_n_vox: int):
    """Load betas from a single session HDF5 file (module-level for joblib)."""
    session_dir = h5_path.parent
    tsv = session_dir / "trial_info.tsv"
    if not tsv.exists():
        return None
    try:
        tinfo = pd.read_csv(tsv, sep="\t")
    except Exception:
        return None
    labels = tinfo["label"].astype(str).tolist() if "label" in tinfo.columns else []
    try:
        with h5py.File(h5_path, "r") as hf:
            if "betasmd" not in hf:
                return None
            betas = np.asarray(hf["betasmd"]).squeeze().astype(np.float32)
    except Exception:
        return None
    if betas.shape[0] != expected_n_vox or betas.shape[1] != len(labels):
        return None
    betas_z = zscore(betas, axis=1, nan_policy="omit")
    betas_z = np.nan_to_num(betas_z, copy=False)
    return labels, betas_z


class DeepVisionBenchmark:
    """Loader for DeepVision fMRI benchmark data.

    Handles extraction of shared stimuli from HDF5 and loading of GLMsingle beta weights
    with ROI masking based on cross-validated effect (CVE) and optional hlvis filtering.

    Args:
        cache_root: Directory for caching extracted images and processed betas.
            Must be provided explicitly.
        subject: Subject identifier (e.g., 'sub-01', 'sub-03', etc.)
        voxel_set: ROI identifier for voxel selection:
            - 'visual': All visually-responsive voxels (CVE > threshold)
            - 'hlvis': High-level visual only (CVE > threshold AND hlvis atlas)
        cve_threshold: Threshold for cross-validated effect mask (default: 0.2)
        input_source: GLMsingle input type ('finalinterp' or 'tedana')
        deepvision_fmri_root: Root path to DeepVision fMRI dataset
        build_rdms: Whether to compute representational dissimilarity matrices (default: False)
        clean_rdms_only: If building RDMs, exclude those with NaN values
        train_test_split: If True, split data into train/test halves

    Attributes:
        stimulus_data: DataFrame with image metadata and paths
        response_data: DataFrame of voxel betas (voxels × images)
        metadata: DataFrame of voxel metadata including ROI membership
        n_stimuli: Number of stimulus images
        image_root: Path to extracted images directory
        brain_space_info: Dict with volume_shape, affine, voxel_indices for brain mapping
    """

    def __init__(
        self,
        cache_root: str | Path,
        subject: str = "sub-01",
        voxel_set: str = "visual",
        cve_threshold: float = DEFAULT_CVE_THRESHOLD,
        input_source: str = "finalinterp",
        deepvision_fmri_root: str | Path | None = None,
        build_rdms: bool = False,
        clean_rdms_only: bool = True,
        train_test_split: bool = False,
        n_jobs: int = 8,
        # Legacy parameters (kept for compatibility)
        image_set: str = "shared",
        anatomical_roi_subset=None,
        functional_roi_subset=None,
    ):
        # Normalize legacy image_set values
        if image_set == "deepvision_shared":
            image_set = "shared"
        if image_set not in ("shared", "unique"):
            raise ValueError(f"image_set must be 'shared' or 'unique', got '{image_set}'")
        self.image_set = image_set

        _LOG.info(
            f"Loading DeepVision (subject={subject}, image_set={image_set}, "
            f"voxel_set={voxel_set}, cve_threshold={cve_threshold})..."
        )

        if subject not in SUBJECT_TO_PARTICIPANT:
            raise ValueError(
                f"Unknown subject '{subject}'. "
                f"Valid subjects: {list(SUBJECT_TO_PARTICIPANT.keys())}"
            )

        # Validate voxel_set
        supported_voxel_sets = ["visual", "hlvis"]
        if voxel_set not in supported_voxel_sets:
            raise ValueError(
                f"Unsupported voxel_set '{voxel_set}'. "
                f"Supported: {supported_voxel_sets}"
            )

        self.voxel_set = voxel_set
        self.cve_threshold = cve_threshold
        self.n_jobs = n_jobs

        # Core identifiers
        self.name = f"deepvision_{image_set}_{voxel_set}"
        self.index_name = "voxel_id"
        self.subject = subject
        self.input_source = input_source

        # Resolve paths
        cache_base = Path(cache_root).expanduser().resolve()
        if deepvision_fmri_root is None:
            deepvision_fmri_root = default_deepvision_fmri_root()
        self._deepvision_fmri_root = Path(deepvision_fmri_root).expanduser().resolve()

        # Cache directories (per subject, input_source, and cve_threshold)
        # Use "visual" as the base voxel set name for caching (since we always load all visual voxels)
        cve_str = f"cve{cve_threshold:.2f}".replace(".", "p")
        # Unique images are per-subject, so include subject in image cache path
        if image_set == "unique":
            image_set_dir = f"deepvision_unique_{subject}"
        else:
            image_set_dir = "deepvision_shared"
        self._image_dir = cache_base / "image_sets" / image_set_dir
        self._image_csv = cache_base / "image_sets" / f"{image_set_dir}.csv"
        self._voxel_dir = (
            cache_base
            / "voxel_sets"
            / f"{image_set_dir}_visual_{cve_str}"
            / input_source
            / subject
        )
        self._betas_csv = self._voxel_dir / "voxel_betas.csv"
        self._meta_csv = self._voxel_dir / "voxel_metadata.csv"
        self._brain_info_json = self._voxel_dir / "brain_space_info.json"
        self._brain_info_npz = self._voxel_dir / "brain_space_arrays.npz"

        # Fast numpy cache paths
        self._betas_npy = self._voxel_dir / "voxel_betas.npy"
        self._betas_cols_npy = self._voxel_dir / "voxel_betas_cols.npy"
        self._betas_idx_npy = self._voxel_dir / "voxel_betas_idx.npy"

        # Source data paths (subject-specific)
        if image_set == "unique":
            participant = SUBJECT_TO_UNIQUE_PARTICIPANT[subject]
        else:
            participant = SUBJECT_TO_PARTICIPANT[subject]
        self._stim_h5 = (
            self._deepvision_fmri_root / f"stimuli_participant_{participant}.hdf5"
        )
        self._stim_meta_csv = self._deepvision_fmri_root / f"metadata_{participant}.csv"
        self._deriv_root = (
            self._deepvision_fmri_root / "derivatives/functional/1sTR_1pt5mm"
        )
        self._glms_root = self._deriv_root / "glmsingle" / input_source
        self._atlas_root = self._deriv_root / "atlas"
        self._hlvis_roi_path = self._atlas_root / subject / "hlvis_p20_mask_func.nii.gz"
        self._cve_path = self._atlas_root / f"cross_validated_effect_{subject}.nii.gz"

        # Prepare caches if missing
        self._ensure_images_extracted()
        self._ensure_betas_and_metadata()

        # Load stimulus data
        self.stimulus_data = pd.read_csv(self._image_csv)
        self.n_stimuli = len(self.stimulus_data)
        self.image_root = str(self._image_dir)
        self.stimulus_data["image_path"] = (
            self._image_dir.as_posix() + "/" + self.stimulus_data.image_name
        )

        # Load response data (prefer fast numpy cache)
        if (
            self._betas_npy.exists()
            and self._betas_cols_npy.exists()
            and self._betas_idx_npy.exists()
        ):
            values = np.load(self._betas_npy)
            columns = np.load(self._betas_cols_npy, allow_pickle=True)
            index = np.load(self._betas_idx_npy, allow_pickle=True)
            self.response_data = pd.DataFrame(values, columns=columns, index=index)
            self.response_data.index.name = self.index_name
        else:
            self.response_data = pd.read_csv(self._betas_csv).set_index(self.index_name)
            # Create fast cache for next time
            np.save(self._betas_npy, self.response_data.values)
            np.save(self._betas_cols_npy, self.response_data.columns.values)
            np.save(self._betas_idx_npy, self.response_data.index.values)
            _LOG.info(f"Created fast numpy cache at {self._betas_npy}")

        # Load metadata
        self.metadata = pd.read_csv(self._meta_csv).set_index(self.index_name)

        # Load brain space info
        self.brain_space_info = self._load_brain_space_info()

        # Define available ROIs (both visual and hlvis are now in metadata)
        self.anatomical_rois: List[str] = []
        self.functional_rois: List[str] = []
        self.available_rois = ["visual", "hlvis"]
        self.all_rois = [
            roi for roi in self.metadata.columns if roi in self.available_rois
        ]

        # Filter response data and metadata if voxel_set is hlvis (intersection)
        if voxel_set == "hlvis":
            hlvis_mask = self.metadata["hlvis"] == 1
            self.response_data = self.response_data.loc[hlvis_mask]
            self.metadata = self.metadata.loc[hlvis_mask]
            # Update brain space info for filtered voxels
            self.brain_space_info = self._filter_brain_space_info(hlvis_mask)

        # Build RDMs if requested
        if build_rdms:
            self.rdm_indices = self.get_rdm_indices()
            self.rdms = self.get_rdms()

            if clean_rdms_only:
                clean_rdms = {}
                for roi in self.rdms:
                    clean_rdms[roi] = {}
                    for subj_id in self.rdms[roi]:
                        if np.sum(np.isnan(self.rdms[roi][subj_id])) == 0:
                            clean_rdms[roi][subj_id] = self.rdms[roi][subj_id]
                self.rdms = clean_rdms

            if train_test_split:
                self.response_data = {
                    "train": self.response_data.iloc[:, ::2],
                    "test": self.response_data.iloc[:, 1::2],
                }
                self.stimulus_data = {
                    "train": self.stimulus_data.iloc[::2, :],
                    "test": self.stimulus_data.iloc[1::2, :],
                }
                self.rdms = self.get_splithalf_rdms()

    def _discover_session_hdf5s(self) -> list[Path]:
        """Find all GLMsingle HDF5 files for this subject."""
        subj_dir = self._glms_root / self.subject
        if not subj_dir.is_dir():
            return []
        return sorted(subj_dir.glob("ses-*/*TYPED_*RR*.hdf5"))

    def _pick_template(self, session: str) -> nib.Nifti1Image | None:
        """Load functional template mask for the given session."""
        if self.input_source == "finalinterp":
            cand_root = self._deriv_root / self.subject / session
            pattern = "*_bold_final_mask.nii.gz"
        else:
            cand_root = self._deriv_root / self.subject / session / "tedana"
            pattern = "*/*_desc-adaptiveGoodSignal_mask.nii.gz"
        if not cand_root.is_dir():
            return None
        cands = sorted(cand_root.glob(pattern))
        if not cands:
            return None
        return nib.load(str(cands[0]))

    def _ensure_images_extracted(self) -> None:
        """Extract stimulus images from HDF5 if not already cached."""
        self._image_dir.mkdir(parents=True, exist_ok=True)
        if self._image_csv.exists() and any(self._image_dir.iterdir()):
            return
        if not self._stim_meta_csv.exists() or not self._stim_h5.exists():
            raise FileNotFoundError(
                f"DeepVision stimuli files not found at {self._stim_h5}"
            )
        meta = pd.read_csv(self._stim_meta_csv)
        subset = meta[meta["unique_or_shared"] == self.image_set].copy()
        subset.reset_index(drop=False, inplace=True)

        _LOG.info(f"Extracting {self.image_set} images ({len(subset)}) from {self._stim_h5}...")
        with h5py.File(self._stim_h5, "r") as hf:
            imgs = hf["images"]
            names = []
            datasets = []
            for _, row in subset.iterrows():
                idx = (
                    int(row["index"])
                    if "index" in subset.columns
                    else int(row["level_0"])
                    if "level_0" in subset.columns
                    else int(row.name)
                )
                name = str(row["image_name"])
                dataset = str(row["dataset"]) if "dataset" in subset.columns else "unknown"
                out_path = self._image_dir / name
                if not out_path.exists():
                    img = _read_binary_img(imgs[idx])
                    img.convert("RGB").save(out_path.as_posix())
                names.append(name)
                datasets.append(dataset)
        pd.DataFrame({"image_name": names, "dataset": datasets}).to_csv(self._image_csv, index=False)

    def _ensure_betas_and_metadata(self) -> None:
        """Load and cache voxel betas, metadata, and brain space info if not already present."""
        self._voxel_dir.mkdir(parents=True, exist_ok=True)
        if (
            self._betas_csv.exists()
            and self._meta_csv.exists()
            and self._brain_info_npz.exists()
        ):
            return

        _LOG.info("Discovering sessions and building beta cache...")
        h5s = self._discover_session_hdf5s()
        if not h5s:
            raise RuntimeError(
                f"No GLMsingle HDF5 files found for {self.subject} at {self._glms_root}"
            )

        first_session = h5s[0].parent.name
        template = self._pick_template(first_session)
        if template is None:
            raise RuntimeError("Could not locate functional template mask")

        # Get volume info for brain space mapping
        volume_shape = template.shape
        affine = template.affine

        # Create brain mask from template
        brain_mask = template.get_fdata().astype(bool)
        n_vox_brain = int(brain_mask.sum())

        def load_volume_masked(path: Path, dtype=np.float32) -> np.ndarray:
            """Load a NIfTI and extract values within brain mask."""
            if not path.exists():
                return np.zeros(n_vox_brain, dtype=dtype)
            vol = nib.load(str(path)).get_fdata()
            if vol.shape != volume_shape:
                raise RuntimeError(f"Volume shape mismatch: {path}")
            return np.asarray(vol[brain_mask], dtype=dtype)

        # Load cross-validated effect (CVE) mask
        if not self._cve_path.exists():
            raise RuntimeError(f"Cross-validated effect map missing: {self._cve_path}")
        cve_vec = load_volume_masked(self._cve_path, dtype=np.float32)
        cve_mask = cve_vec > self.cve_threshold
        _LOG.info(f"CVE > {self.cve_threshold}: {cve_mask.sum()} voxels")

        # Load hlvis ROI
        hlvis_vec = load_volume_masked(self._hlvis_roi_path, dtype=np.float32) > 0
        hlvis_in_cve = cve_mask & hlvis_vec
        _LOG.info(f"hlvis within CVE mask: {hlvis_in_cve.sum()} voxels")

        # The visual mask is CVE > threshold (base mask for all voxels)
        visual_mask = cve_mask

        # Get flat indices into the full volume for voxels in the visual mask
        # This allows mapping back to brain space later
        brain_flat_indices = np.where(brain_mask.ravel())[0]
        visual_voxel_flat_indices = brain_flat_indices[visual_mask]

        # Load target stimulus list
        stim_df = pd.read_csv(self._image_csv)
        target_images = stim_df["image_name"].astype(str).tolist()
        target_index = {name: j for j, name in enumerate(target_images)}

        # Aggregate betas across sessions (for ALL brain voxels first)
        responses_full = np.full(
            (n_vox_brain, len(target_images)), np.nan, dtype=np.float32
        )
        counts = np.zeros(len(target_images), dtype=np.int32)

        def add_to_accum(image_name: str, vec: np.ndarray):
            j = target_index.get(image_name)
            if j is None:
                return
            if np.isnan(responses_full[0, j]):
                responses_full[:, j] = vec
            else:
                responses_full[:, j] += vec
            counts[j] += 1

        def aggregate_session_result(result) -> bool:
            if result is None:
                return False
            labels, betas_z = result
            for col_idx, name in enumerate(labels):
                lname = str(name).strip()
                if lname.lower() == "blank":
                    continue
                add_to_accum(lname, betas_z[:, col_idx])
            return True

        # Load all sessions.  For n_jobs=1, aggregate each session immediately so
        # large subject caches can be built without retaining every session array.
        print(f"Loading {len(h5s)} session files with {self.n_jobs} workers...")
        loaded_sessions = 0
        if self.n_jobs == 1:
            for h5 in h5s:
                if aggregate_session_result(_load_session_betas(h5, n_vox_brain)):
                    loaded_sessions += 1
        else:
            session_results = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=10)(
                delayed(_load_session_betas)(h5, n_vox_brain) for h5 in h5s
            )
            for result in session_results:
                if aggregate_session_result(result):
                    loaded_sessions += 1
        print(f"Loaded {loaded_sessions} sessions")

        # Finalize averages
        for j in range(len(target_images)):
            if counts[j] > 1:
                responses_full[:, j] /= counts[j]

        # Filter to visual mask only
        responses = responses_full[visual_mask, :]
        n_vox = responses.shape[0]
        _LOG.info(f"Final visual voxel count: {n_vox}")

        # Create voxel IDs (0-indexed within the visual mask)
        voxel_ids = np.arange(n_vox, dtype=int)

        # Create metadata with both visual and hlvis ROI columns
        # visual is all True (since we're only keeping visual voxels)
        # hlvis indicates which of those are also in hlvis
        hlvis_in_visual = hlvis_vec[visual_mask]

        resp_df = pd.DataFrame(responses, columns=target_images)
        resp_df.insert(0, self.index_name, voxel_ids)
        meta_df = pd.DataFrame(
            {
                self.index_name: voxel_ids,
                "subj_id": [self.subject] * n_vox,
                "visual": np.ones(
                    n_vox, dtype=int
                ),  # All voxels pass visual (CVE) threshold
                "hlvis": hlvis_in_visual.astype(int),  # Subset that are also hlvis
            }
        )
        resp_df.to_csv(self._betas_csv, index=False)
        meta_df.to_csv(self._meta_csv, index=False)

        # Save brain space info
        brain_info = {
            "volume_shape": list(volume_shape),
            "subject": self.subject,
            "cve_threshold": self.cve_threshold,
            "n_voxels_visual": int(n_vox),
            "n_voxels_hlvis": int(hlvis_in_visual.sum()),
        }
        with open(self._brain_info_json, "w") as f:
            json.dump(brain_info, f, indent=2)

        np.savez_compressed(
            self._brain_info_npz,
            affine=affine,
            voxel_indices=visual_voxel_flat_indices,
            hlvis_mask=hlvis_in_visual,
        )
        _LOG.info(f"Saved brain space info to {self._brain_info_npz}")

    def _load_brain_space_info(self) -> Dict:
        """Load brain space mapping info from cache."""
        if not self._brain_info_json.exists() or not self._brain_info_npz.exists():
            raise RuntimeError(
                f"Brain space info not found. Expected {self._brain_info_npz}"
            )

        with open(self._brain_info_json, "r") as f:
            info = json.load(f)

        with np.load(self._brain_info_npz) as z:
            info["affine"] = z["affine"]
            info["voxel_indices"] = z["voxel_indices"]
            info["hlvis_mask"] = z["hlvis_mask"]

        info["volume_shape"] = tuple(info["volume_shape"])
        return info

    def _filter_brain_space_info(self, mask: np.ndarray) -> Dict:
        """Filter brain space info to a subset of voxels."""
        mask_array = mask.values if hasattr(mask, "values") else np.asarray(mask)
        filtered = {
            "volume_shape": self.brain_space_info["volume_shape"],
            "affine": self.brain_space_info["affine"],
            "voxel_indices": self.brain_space_info["voxel_indices"][mask_array],
            "hlvis_mask": self.brain_space_info["hlvis_mask"][mask_array],
            "subject": self.brain_space_info["subject"],
            "cve_threshold": self.brain_space_info["cve_threshold"],
            "n_voxels_visual": int(mask_array.sum()),
            "n_voxels_hlvis": int(
                self.brain_space_info["hlvis_mask"][mask_array].sum()
            ),
        }
        return filtered

    def get_roi_mask(self, roi: str = "visual") -> np.ndarray:
        """Get boolean mask for ROI within current voxel set.

        Args:
            roi: ROI name ('visual' or 'hlvis')

        Returns:
            Boolean array of shape (n_voxels,)
        """
        if roi not in self.available_rois:
            raise ValueError(f"Unknown ROI '{roi}'. Available: {self.available_rois}")
        return (self.metadata[roi] == 1).values

    def get_brain_space_info(self) -> Dict:
        """Get brain space mapping info for creating NIfTI outputs.

        Returns:
            Dict with:
                - volume_shape: Tuple of (x, y, z) dimensions
                - affine: 4x4 NIfTI affine matrix
                - voxel_indices: Flat indices into volume for each voxel
                - hlvis_mask: Boolean mask for hlvis voxels within current set
        """
        return self.brain_space_info

    def to_volume(
        self, voxel_values: np.ndarray, fill_value: float = 0.0
    ) -> np.ndarray:
        """Map 1D voxel values to 3D volume array.

        Args:
            voxel_values: Array of shape (n_voxels,) with values for each voxel
            fill_value: Value to use for voxels not in the mask

        Returns:
            3D numpy array of shape volume_shape
        """
        if len(voxel_values) != len(self.brain_space_info["voxel_indices"]):
            raise ValueError(
                f"voxel_values length ({len(voxel_values)}) doesn't match "
                f"number of voxels ({len(self.brain_space_info['voxel_indices'])})"
            )

        volume = np.full(
            self.brain_space_info["volume_shape"], fill_value, dtype=np.float32
        )
        volume.flat[self.brain_space_info["voxel_indices"]] = voxel_values
        return volume

    def to_nifti(
        self,
        voxel_values: np.ndarray,
        output_path: str | Path,
        fill_value: float = 0.0,
    ) -> "nib.Nifti1Image":
        """Save voxel values as NIfTI file.

        Args:
            voxel_values: Array of shape (n_voxels,) with values for each voxel
            output_path: Path to save NIfTI file
            fill_value: Value to use for voxels not in the mask

        Returns:
            The created NIfTI image
        """
        volume = self.to_volume(voxel_values, fill_value=fill_value)
        img = nib.Nifti1Image(volume, affine=self.brain_space_info["affine"])
        nib.save(img, str(output_path))
        return img

    def get_stimulus(self, image_index: int | None = None) -> Image.Image:
        """Load a stimulus image by index (random if not specified)."""
        if image_index is None:
            image_index = np.random.randint(self.n_stimuli)
        image_path = self.stimulus_data.image_name.iloc[image_index]
        return Image.open(self._image_dir / image_path)

    get_sample_stimulus = get_stimulus

    def get_rdm_indices(self, roi_subset=None, row_number: bool = False):
        """Get voxel indices grouped by ROI and subject."""
        metadata = self.metadata
        if self.index_name in metadata.columns:
            metadata = metadata.set_index(self.index_name)
        if not roi_subset:
            roi_subset = self.all_rois
        if row_number:
            metadata = metadata.reset_index()
        rdm_indices = {}
        for roi in roi_subset:
            roi_subset_df = metadata[metadata[roi] == 1]
            rdm_indices[roi] = {}
            for subj_id in roi_subset_df.subj_id.unique():
                subj_id_subset = roi_subset_df[roi_subset_df["subj_id"] == subj_id]
                rdm_indices[roi][subj_id] = subj_id_subset.index.to_numpy()
        return rdm_indices

    def get_rdms(self, roi_subset=None, include_group_average: bool = False):
        """Compute RDMs (1 - correlation) for each ROI and subject."""
        responses = self.response_data
        if self.index_name in responses.columns:
            responses = responses.set_index(self.index_name)
        if not roi_subset:
            roi_subset = self.all_rois
        if not getattr(self, "rdm_indices", None):
            self.rdm_indices = self.get_rdm_indices(roi_subset=roi_subset)
        brain_rdms = {}
        for roi in self.rdm_indices:
            brain_rdms[roi] = {}
            for subj_id in self.rdm_indices[roi]:
                target_responses = responses.loc[self.rdm_indices[roi][subj_id]]
                if target_responses.shape[0] > 10:
                    brain_rdms[roi][subj_id] = 1 - np.corrcoef(
                        target_responses.transpose()
                    )
        return brain_rdms

    def get_splithalf_rdms(self):
        """Split RDMs into train/test halves (by stimulus index, not trials).

        NOTE: This splits by stimulus index (odd/even stimuli), not by trial repetitions.
        """
        split_rdms = {}
        for roi in self.rdms:
            split_rdms[roi] = {}
            for subj_id in self.rdms[roi]:
                split_rdms[roi][subj_id] = {
                    "train": self.rdms[roi][subj_id][::2, ::2],
                    "test": self.rdms[roi][subj_id][1::2, 1::2],
                }
        return split_rdms
