from __future__ import annotations

import csv
import os
import re
import socket
from functools import lru_cache
from pathlib import Path

from cstims.constants import EXPECTED_CSTIM_IMAGE_COUNTS


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")
_NATURAL_SORT_RE = re.compile(r"(\d+)")
_CONFIG_CACHE: dict[str, str] | None = None
_CONFIG_CACHE_KEY: tuple[str, str | None, str | None, str | None] | None = None


def find_share_root(start: Path | None = None) -> Path:
    """Find the cstims_share root from an installed or in-place package."""
    start = (start or Path(__file__)).resolve()
    for path in (start, *start.parents):
        if (
            (path / "pyproject.toml").exists()
            and (path / "00_stimulus_selection").exists()
            and (path / "01_brain_model_alignment").exists()
        ):
            return path
    raise RuntimeError(f"Could not locate cstims_share root from {start}")


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            raise ValueError(f"Invalid path config line {line_no} in {path}: {raw!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            raise ValueError(f"Empty key on path config line {line_no} in {path}")
        values[key] = value
    return values


def _resolve_config_reference(root: Path, value: str) -> Path:
    ref = Path(value).expanduser()
    if ref.is_absolute():
        return ref if ref.suffix else ref.with_suffix(".env")
    if len(ref.parts) > 1:
        path = (root / ref).resolve()
        return path if path.suffix else path.with_suffix(".env")
    name = ref.name if ref.suffix else f"{ref.name}.env"
    return root / "conf" / "paths" / name


def _path_config() -> dict[str, str]:
    global _CONFIG_CACHE, _CONFIG_CACHE_KEY

    root = find_share_root()
    hostname = socket.gethostname().split(".", 1)[0]
    explicit_config = os.environ.get("CSTIMS_PATH_CONFIG")
    explicit_env = os.environ.get("CSTIMS_PATH_ENV") or os.environ.get("CSTIMS_ENV")
    cache_key = (str(root), explicit_config, explicit_env, hostname)
    if _CONFIG_CACHE is not None and _CONFIG_CACHE_KEY == cache_key:
        return dict(_CONFIG_CACHE)

    if explicit_config:
        candidates = [_resolve_config_reference(root, explicit_config)]
    elif explicit_env:
        candidates = [_resolve_config_reference(root, explicit_env)]
    else:
        candidates = [
            root / "conf" / "paths" / f"{hostname}.env",
            root / "conf" / "paths" / "default.env",
        ]
    for candidate in candidates:
        if candidate.exists():
            _CONFIG_CACHE = _parse_env_file(candidate)
            _CONFIG_CACHE["_CONFIG_FILE"] = str(candidate)
            _CONFIG_CACHE_KEY = cache_key
            return dict(_CONFIG_CACHE)
    tried = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Missing cstims path configuration. Expected a server-specific file "
        f"under conf/paths/. Tried:\n  {tried}"
    )


def _configured_path(key: str) -> Path:
    config = _path_config()
    value = config.get(key)
    if not value:
        raise KeyError(f"Missing {key} in {config.get('_CONFIG_FILE', 'path config')}")
    return Path(value).expanduser().resolve()


def _configured_path_or_default(key: str, default: Path) -> Path:
    config = _path_config()
    value = config.get(key)
    if value:
        return Path(value).expanduser().resolve()
    return default.expanduser().resolve()


def _require_dir(path: Path, label: str, markers: tuple[str, ...] = ()) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist or is not a directory: {path}")
    missing = [marker for marker in markers if not (path / marker).exists()]
    if missing:
        raise FileNotFoundError(
            f"{label} is missing expected content at {path}: {', '.join(missing)}"
        )
    return path


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")
    return path


def _image_files(path: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in IMAGE_EXTENSIONS:
        files.extend(path.glob(pattern))
    return sorted(set(files), key=_natural_path_key)


def _natural_path_key(path: Path) -> tuple[tuple[int, int | str], ...]:
    parts = _NATURAL_SORT_RE.split(path.name)
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in parts
    )


def _cstim_folder_name(stimulus_set: str, *, apply_architecture_dataset_swap: bool = False) -> str:
    if apply_architecture_dataset_swap:
        if stimulus_set == "architecture":
            stimulus_set = "dataset"
        elif stimulus_set == "dataset":
            stimulus_set = "architecture"
    return "shared_vicco" if stimulus_set == "vicco" else stimulus_set


def _cstim_public_name(folder_name: str) -> str:
    return "vicco" if folder_name == "shared_vicco" else folder_name


@lru_cache(maxsize=8)
def _cstim_hdf5_records(root_str: str) -> tuple[tuple[int, str, str], ...]:
    root = Path(root_str)
    metadata_path = _require_file(root / "metadata.csv", "CSTIM_HDF5_ROOT/metadata.csv")
    with metadata_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"group", "image_name"} - set(reader.fieldnames or [])
        if missing:
            raise KeyError(f"{metadata_path} is missing required columns: {sorted(missing)}")
        records = tuple(
            (row_idx, str(row["group"]), str(row["image_name"]))
            for row_idx, row in enumerate(reader)
        )

    h5_path = _require_file(root / "stimuli.hdf5", "CSTIM_HDF5_ROOT/stimuli.hdf5")
    import h5py

    with h5py.File(h5_path, "r") as h5:
        if "images" not in h5:
            raise KeyError(f"{h5_path} is missing required dataset 'images'")
        n_images = len(h5["images"])
    if n_images != len(records):
        raise RuntimeError(
            f"CSTIM metadata/HDF5 count mismatch: metadata has {len(records)} rows, "
            f"stimuli.hdf5 has {n_images} images."
        )
    return records


def _cstim_hdf5_group_counts(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _row_idx, group, _image_name in _cstim_hdf5_records(str(root)):
        counts[group] = counts.get(group, 0) + 1
    return counts


def _cstim_extracted_image_cache_root() -> Path:
    root = _configured_path_or_default(
        "CSTIM_EXTRACTED_IMAGE_CACHE_DIR",
        feature_cache_dir() / "cstim_hdf5_images",
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_cstim_image_name(image_name: str, fallback_idx: int) -> str:
    name = Path(image_name).name
    if not name:
        name = f"image_{fallback_idx}.jpg"
    if Path(name).suffix:
        return name
    return f"{name}.jpg"


def _ensure_cstim_hdf5_images(
    stimulus_set: str,
    *,
    apply_architecture_dataset_swap: bool = False,
) -> list[Path]:
    root = cstim_hdf5_root()
    folder = _cstim_folder_name(
        stimulus_set,
        apply_architecture_dataset_swap=apply_architecture_dataset_swap,
    )
    records = [
        (row_idx, image_name)
        for row_idx, group, image_name in _cstim_hdf5_records(str(root))
        if group == folder
    ]
    expected = EXPECTED_CSTIM_IMAGE_COUNTS.get(stimulus_set)
    if expected is not None and len(records) != expected:
        raise RuntimeError(
            f"Unexpected CSTIM metadata count for {stimulus_set!r}: "
            f"found {len(records)} in group {folder!r}, expected {expected}."
        )

    out_dir = _cstim_extracted_image_cache_root() / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    missing: list[tuple[int, Path]] = []
    for fallback_idx, (row_idx, image_name) in enumerate(records):
        out_path = out_dir / _safe_cstim_image_name(image_name, fallback_idx)
        paths.append(out_path)
        if not out_path.is_file() or out_path.stat().st_size == 0:
            missing.append((row_idx, out_path))

    if missing:
        import h5py
        import numpy as np

        with h5py.File(root / "stimuli.hdf5", "r") as h5:
            images = h5["images"]
            for row_idx, out_path in missing:
                raw = np.asarray(images[row_idx]).tobytes()
                tmp = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
                tmp.write_bytes(raw)
                os.replace(tmp, out_path)

    return paths


def project_root() -> Path:
    return _require_dir(
        _configured_path("SHARE_ROOT"),
        "SHARE_ROOT",
        markers=("pyproject.toml", "00_stimulus_selection", "01_brain_model_alignment"),
    )


def paper_root() -> Path:
    return project_root()


def external_data_root() -> Path:
    return _require_dir(
        _configured_path_or_default("EXTERNAL_DATA_ROOT", project_root() / "external_data"),
        "external_data_root",
    )


def resources_dir() -> Path:
    return _require_dir(
        _configured_path_or_default(
            "RESOURCES_DIR",
            project_root() / "00_stimulus_selection" / "resources",
        ),
        "resources_dir",
        markers=("model_list.csv",),
    )


def model_list_csv() -> Path:
    return _require_file(resources_dir() / "model_list.csv", "model_list_csv")


def model_layer_mapping() -> dict[str, str]:
    """Return model -> encoding layer folder tag from the canonical model list."""
    mapping: dict[str, str] = {}
    csv_path = model_list_csv()
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row.get("model")
            layer = row.get("layer")
            if not model or layer is None:
                raise KeyError(f"{csv_path} must contain non-empty 'model' and 'layer' columns")
            if model in mapping:
                raise RuntimeError(f"Duplicate model {model!r} in {csv_path}")
            mapping[model] = f"layer{str(layer).replace('.', '_')}"
    return mapping


def model_layer_tag(model: str) -> str:
    """Return the encoding folder layer tag for one model."""
    mapping = model_layer_mapping()
    try:
        return mapping[model]
    except KeyError as exc:
        raise KeyError(f"Model {model!r} is absent from {model_list_csv()}") from exc


def encoding_model_dir(
    subject: str,
    model: str,
    *,
    encoding_root: Path | str | None = None,
) -> Path:
    """Return the encoding artifact directory for one subject/model pair."""
    root = (
        get_encoding_root(subject)
        if encoding_root is None
        else Path(encoding_root).expanduser().resolve()
    )
    return root / f"{subject}_{model}.{model_layer_tag(model)}"


def deepvision_fmri_root() -> Path:
    return _require_dir(
        _configured_path("DEEPVISION_FMRI_ROOT"),
        "DEEPVISION_FMRI_ROOT",
        markers=("stimuli_participant_p01.hdf5", "metadata_p01.csv", "derivatives"),
    )


def deepvision_fmri_source_root() -> Path:
    return _configured_path("DEEPVISION_FMRI_ROOT")


def deepvision_cache_root() -> Path:
    return _require_dir(
        _configured_path("DEEPVISION_CACHE_ROOT"),
        "DEEPVISION_CACHE_ROOT",
        markers=("image_sets", "voxel_sets"),
    )


def cstim_hdf5_root() -> Path:
    root = _require_dir(
        _configured_path("CSTIM_HDF5_ROOT"),
        "CSTIM_HDF5_ROOT",
        markers=("metadata.csv", "stimuli.hdf5"),
    )
    for stimulus_set, expected in EXPECTED_CSTIM_IMAGE_COUNTS.items():
        folder = _cstim_folder_name(stimulus_set)
        img_dir = root / folder
        files = _image_files(img_dir)
        if len(files) != expected:
            raise RuntimeError(
                f"Unexpected CSTIM image count for {stimulus_set!r}: "
                f"found {len(files)} in {img_dir}, expected {expected}."
            )
    return root


def cstim_image_dir(
    stimulus_set: str,
    *,
    apply_architecture_dataset_swap: bool = False,
) -> Path:
    root = cstim_hdf5_root()
    folder = _cstim_folder_name(
        stimulus_set,
        apply_architecture_dataset_swap=apply_architecture_dataset_swap,
    )
    img_dir = _require_dir(root / folder, f"cstim_image_dir({stimulus_set!r})")
    expected = EXPECTED_CSTIM_IMAGE_COUNTS.get(stimulus_set)
    if expected is not None:
        files = _image_files(img_dir)
        if len(files) != expected:
            raise RuntimeError(
                f"Unexpected CSTIM image count for {stimulus_set!r}: "
                f"found {len(files)} in {img_dir}, expected {expected}."
            )
    return img_dir


def cstim_image_paths(
    stimulus_set: str,
    *,
    apply_architecture_dataset_swap: bool = False,
) -> list[Path]:
    img_dir = cstim_image_dir(
        stimulus_set,
        apply_architecture_dataset_swap=apply_architecture_dataset_swap,
    )
    files = _image_files(img_dir)
    expected = EXPECTED_CSTIM_IMAGE_COUNTS.get(stimulus_set)
    if expected is not None and len(files) != expected:
        raise RuntimeError(
            f"Unexpected CSTIM image count for {stimulus_set!r}: "
            f"found {len(files)} in {img_dir}, expected {expected}."
        )
    return files


def shared_encoding_root() -> Path:
    return _require_dir(
        _configured_path_or_default(
            "SHARED_ENCODING_ROOT",
            project_root()
            / "01_brain_model_alignment"
            / "results"
            / "encoding_models"
            / "shared_subject_encoding_models"
            / "encoding_20251222_141301",
        ),
        "shared_encoding_root",
    )


def unique_encoding_root() -> Path:
    return _require_dir(
        _configured_path_or_default(
            "UNIQUE_ENCODING_ROOT",
            project_root()
            / "01_brain_model_alignment"
            / "results"
            / "encoding_models"
            / "subject_unique_encoding_models"
            / "runs",
        ),
        "unique_encoding_root",
    )


def unique_encoding_dirs() -> dict[str, Path]:
    root = unique_encoding_root()
    dirs = {
        "sub-01": root / "20260317_170621",
        "sub-03": root / "20260319_152751",
        "sub-05": root / "20260317_170621",
        "sub-06": root / "20260319_152752",
        "sub-07": root / "20260317_170621",
    }
    for subject, path in dirs.items():
        _require_dir(path, f"unique_encoding_dirs[{subject!r}]")
    return dirs


def get_encoding_root(subject: str | None = None) -> Path:
    dirs = unique_encoding_dirs()
    if subject in dirs:
        return dirs[subject]
    return shared_encoding_root()


def selected_stimuli_root() -> Path:
    return _require_dir(
        _configured_path_or_default(
            "SELECTED_STIMULI_ROOT",
            project_root() / "00_stimulus_selection" / "results" / "selected_stimuli",
        ),
        "selected_stimuli_root",
    )


def selected_stimuli_payload(model_set: str = "all_models") -> Path:
    return _require_file(
        selected_stimuli_root() / model_set / "selected_stimuli_data.pkl",
        f"selected_stimuli_payload({model_set!r})",
    )


def selection_evaluation_results_dir() -> Path:
    return _require_dir(
        _configured_path_or_default(
            "SELECTION_EVALUATION_RESULTS_DIR",
            project_root() / "00_stimulus_selection" / "selection_evaluation" / "results",
        ),
        "selection_evaluation_results_dir",
    )


def get_eval_pipeline_dir(model_set: str) -> Path:
    return _require_dir(
        selection_evaluation_results_dir() / model_set,
        f"get_eval_pipeline_dir({model_set!r})",
    )


def brain_data_dir() -> Path:
    return _require_dir(
        _configured_path_or_default(
            "BRAIN_DATA_DIR",
            project_root()
            / "01_brain_model_alignment"
            / "cache_or_heavy"
            / "cstim_brain_response_cache"
            / "data",
        ),
        "brain_data_dir",
    )


def rsa_data_dir() -> Path:
    return _require_dir(
        _configured_path_or_default(
            "RSA_DATA_DIR",
            project_root() / "01_brain_model_alignment" / "results" / "rsa_scores",
        ),
        "rsa_data_dir",
    )


def reliability_data_dir() -> Path:
    return _require_dir(
        _configured_path_or_default(
            "RELIABILITY_DATA_DIR",
            project_root() / "02_alignment_reliability" / "results",
        ),
        "reliability_data_dir",
    )


def stats_data_dir() -> Path:
    return _require_dir(
        _configured_path_or_default(
            "STATS_DATA_DIR",
            project_root() / "03_alignment_inference" / "results",
        ),
        "stats_data_dir",
    )


def robustness_data_dir() -> Path:
    return _require_dir(
        _configured_path_or_default(
            "ROBUSTNESS_DATA_DIR",
            project_root() / "04_alignment_robustness" / "results",
        ),
        "robustness_data_dir",
    )


def simulation_data_dir() -> Path:
    return _require_dir(
        _configured_path_or_default(
            "SIMULATION_DATA_DIR",
            project_root() / "05_controls_and_supplementary" / "simulation_validation" / "results",
        ),
        "simulation_data_dir",
    )


def counterfactual_baseline_data_dir() -> Path:
    return _require_dir(
        _configured_path_or_default(
            "COUNTERFACTUAL_BASELINE_DATA_DIR",
            project_root()
            / "05_controls_and_supplementary"
            / "counterfactual_baselines"
            / "results",
        ),
        "counterfactual_baseline_data_dir",
    )


def ood_data_dir() -> Path:
    return _require_dir(
        _configured_path_or_default(
            "OOD_DATA_DIR",
            project_root()
            / "05_controls_and_supplementary"
            / "low_level_and_ood"
            / "ood_controls"
            / "results",
        ),
        "ood_data_dir",
    )


def consensus_data_dir() -> Path:
    return _require_dir(
        _configured_path_or_default(
            "CONSENSUS_DATA_DIR",
            project_root()
            / "05_controls_and_supplementary"
            / "integrated_explanation"
            / "results",
        ),
        "consensus_data_dir",
    )


def feature_cache_dir() -> Path:
    return _require_dir(_configured_path("FEATURE_CACHE_DIR"), "FEATURE_CACHE_DIR")


def cstim_feature_cache_dir() -> Path:
    return _require_dir(feature_cache_dir() / "cstim", "cstim_feature_cache_dir")


def deepvision_feature_cache_dir() -> Path:
    return _require_dir(feature_cache_dir() / "deepvision", "deepvision_feature_cache_dir")


def vicco_feature_cache_dir() -> Path:
    return _require_dir(feature_cache_dir() / "vicco", "vicco_feature_cache_dir")


def voxel_cache_dir() -> Path:
    return _require_dir(deepvision_cache_root() / "voxel_sets", "voxel_cache_dir")


def get_brain_input_dir(subject: str) -> Path:
    return _require_dir(brain_data_dir() / subject, f"get_brain_input_dir({subject!r})")


def get_subject_data_dir(subject: str) -> Path:
    return get_brain_input_dir(subject)
