import csv
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Sequence

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from cstims.data_loader import (
    build_selected_image_records,
    load_natural_features_with_metadata,
    max_images_for_ram,
)
from cstims.encoding.linear import (
    EncodingParamsByEncoding,
    encode_batch_for_all_encodings,
    load_encoding_params_by_encoding,
)
from cstims.evaluation.model_discrimination import calibrate_feature_noise
from cstims.feature_accessor import FeatureAccessor, PrefetchingFeatureAccessor
from cstims.noise_estimation import rdm_noise_by_model
from cstims.selection import (
    ImageFilter,
    ImageFilterConfig,
    SelectionCheckpoint,
    TrackAggregationConfig,
    TrackDefinition,
    extract_features_for_indices,
    load_checkpoint,
    select_stimuli_multitrack,
)
from cstims.timing import FunctionTimer

log = logging.getLogger(__name__)


def load_layer_names(model_list_csv: Path, model_names: list[str]) -> list[str]:
    """Reads all layer names from the model list CSV file."""
    with open(model_list_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        layer_names = [
            row["layer"]
            for row in reader
            if "layer" in row and row["layer"] and row["model"] in model_names
        ]
    return layer_names


def _to_python_dict(cfg_item) -> dict:
    if isinstance(cfg_item, DictConfig):
        return OmegaConf.to_container(cfg_item, resolve=True)  # type: ignore[arg-type]
    return cfg_item


def _parse_encoding_list(val) -> list[str]:
    """Helper to parse encoding names from config, handling potential whitespace issues."""
    if not val:
        return []
    if isinstance(val, str):
        return val.replace(",", " ").split()

    res = []
    for item in val:
        if isinstance(item, str):
            res.extend(item.replace(",", " ").split())
        else:
            res.append(str(item))
    return res


def collect_required_encoding_names(track_cfgs: Sequence[dict]) -> list[str]:
    names: set[str] = set()
    for track in track_cfgs:
        track = _to_python_dict(track)
        track_type = track.get("type", "identity")
        if track_type == "encoding" and track.get("encoding_name"):
            names.add(track["encoding_name"])
        elif track_type == "group_encoding":
            for enc_name in _parse_encoding_list(track.get("encoding_names")):
                names.add(enc_name)
    return sorted(names)


def build_track_definitions(
    track_cfgs: Sequence[dict],
    var_noise_raw: Optional[dict[str, float]],
    var_noise_by_encoding: dict[str, dict[str, float]],
) -> list[TrackDefinition]:
    definitions: list[TrackDefinition] = []
    for raw_cfg in track_cfgs:
        cfg_dict = _to_python_dict(raw_cfg)
        track_type = cfg_dict.get("type", "identity")
        name = cfg_dict.get("name")
        if not name:
            raise ValueError("Each track must define a unique 'name'.")

        if track_type == "identity":
            if not var_noise_raw:
                raise ValueError(f"Track '{name}' requires raw noise calibration.")
            var_noise = dict(var_noise_raw)
            definitions.append(
                TrackDefinition(
                    name=name,
                    type="identity",
                    var_noise_by_model=var_noise,
                )
            )
        elif track_type == "encoding":
            enc_name = cfg_dict.get("encoding_name")
            if not enc_name:
                raise ValueError(f"Track '{name}' must set 'encoding_name'.")
            if enc_name not in var_noise_by_encoding:
                raise ValueError(
                    f"Noise calibration missing for encoding '{enc_name}' used by track '{name}'."
                )
            var_noise = dict(var_noise_by_encoding[enc_name])
            definitions.append(
                TrackDefinition(
                    name=name,
                    type="encoding",
                    encoding_name=enc_name,
                    var_noise_by_model=var_noise,
                )
            )
        elif track_type == "group_encoding":
            encoding_names = _parse_encoding_list(cfg_dict.get("encoding_names"))
            if not encoding_names:
                raise ValueError(
                    f"Track '{name}' must provide a non-empty 'encoding_names' list."
                )
            definitions.append(
                TrackDefinition(
                    name=name,
                    type="group_encoding",
                    encoding_names=encoding_names,
                )
            )
        else:
            raise ValueError(f"Unknown track type '{track_type}' for track '{name}'.")

    return definitions


def calibrate_encoding_noise(
    encoding_name: str,
    raw_features: dict[str, np.ndarray],
    encoding_params: EncodingParamsByEncoding,
    model_names: list[str],
    metric: str,
    corr_type: str,
    noise_ceiling_target: float,
    device: torch.device,
    max_images: int,
) -> dict[str, float]:
    if encoding_name not in encoding_params:
        raise ValueError(f"Encoding parameters not loaded for '{encoding_name}'.")

    n_calib = min(1000, max_images)
    calib_indices = np.arange(n_calib)
    raw_batch = {
        name: torch.from_numpy(raw_features[name][calib_indices]).to(
            device=device, dtype=torch.float32
        )
        for name in model_names
    }
    encoded_batch = encode_batch_for_all_encodings(
        raw_batch, {encoding_name: encoding_params[encoding_name]}
    )[encoding_name]
    encoded_np = {name: encoded_batch[name].cpu().numpy() for name in model_names}
    return rdm_noise_by_model(
        encoded_np,
        model_names,
        device,
        metric=metric,
        target_nc=noise_ceiling_target,
        corr_type=corr_type,
    )


OmegaConf.register_new_resolver("join", lambda sep, items: sep.join(items))


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Check for resume mode
    resume_from_dir = cfg.get("resume_from", None)
    resume_checkpoint: Optional[SelectionCheckpoint] = None

    if resume_from_dir:
        resume_dir = Path(resume_from_dir)
        if not resume_dir.exists():
            raise FileNotFoundError(f"Resume directory does not exist: {resume_dir}")

        # Load config from original run's .hydra directory
        original_config_path = resume_dir / ".hydra" / "config.yaml"
        if not original_config_path.exists():
            raise FileNotFoundError(f"No config found at {original_config_path}")

        cfg = OmegaConf.load(original_config_path)
        log.info(f"Loaded config from {original_config_path}")

        # Load checkpoint
        checkpoint_path = resume_dir / "checkpoint.pkl"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        resume_checkpoint = load_checkpoint(checkpoint_path)
        log.info(
            f"Resuming from checkpoint: phase={resume_checkpoint.phase}, "
            f"greedy_iter={resume_checkpoint.greedy_iteration}, "
            f"refine_pass={resume_checkpoint.refinement_pass}, "
            f"refine_pos={resume_checkpoint.refinement_position}"
        )

        # Check if already complete
        if resume_checkpoint.phase == "complete":
            log.info("Selection already complete! Nothing to do.")
            return

        # Use same output directory
        output_dir = resume_dir
    else:
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    log.info(f"Output directory: {output_dir}")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Model names is now a list from config
    model_names = cfg.model_names
    log.info(f"Using models: {model_names}")

    layer_names = load_layer_names(Path(cfg.paths.model_list_csv), model_names)
    log.info(f"Using layers: {layer_names}")

    # Set random seeds
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Setup device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # Use paths from config
    max_images = (
        max_images_for_ram(
            subset_root=Path(cfg.paths.subset_root),
            model_names=model_names,
            max_ram_bytes=int(cfg.max_ram_gb * 1024**3),
            model_csv=Path(cfg.paths.model_list_csv),
        )
        if cfg.max_ram_gb
        else cfg.max_images
    )

    log.info(f"Loading raw features for {max_images} images")
    preprocessed_dirs = cfg.paths.preprocessed_dirs
    if "raw" not in preprocessed_dirs:
        raise ValueError("paths.preprocessed_dirs must define a 'raw' entry.")

    raw_features_np, raw_shard_slices = load_natural_features_with_metadata(
        subset_root=Path(cfg.paths.subset_root),
        preprocessed_dir=Path(preprocessed_dirs["raw"]),
        model_names=model_names,
        layer_names=layer_names,
        max_images=max_images,
        model_csv=Path(cfg.paths.model_list_csv),
    )

    accessor_cls = (
        PrefetchingFeatureAccessor if cfg.prefetch_features else FeatureAccessor
    )
    raw_accessor = accessor_cls(
        features_by_model=raw_features_np,
        model_names=model_names,
        pool_indices=np.arange(max_images),
        batch_size=cfg.batch_size,
        device=device,
        dtype=torch.float32,
    )

    track_cfgs = cfg.tracks
    if not track_cfgs:
        raise ValueError("cfg.tracks must define at least one scoring track.")
    track_cfgs_resolved = [_to_python_dict(track_cfg) for track_cfg in track_cfgs]

    required_encoding_names = collect_required_encoding_names(track_cfgs_resolved)
    if required_encoding_names:
        roi_subset = cfg.get("encoding_roi_subset", "hlvis")
        log.info(
            f"Loading encoding params for tracks: {required_encoding_names} (ROI subset: {roi_subset})"
        )
        encoding_params_by_encoding = load_encoding_params_by_encoding(
            encoding_root=Path(cfg.paths.encoding_root),
            model_list_csv=Path(cfg.paths.model_list_csv),
            encoding_names=required_encoding_names,
            device=device,
            roi_subset=roi_subset if roi_subset else None,
        )
    else:
        encoding_params_by_encoding = {}

    # Noise calibration - skip if resuming with checkpoint data
    if resume_checkpoint is not None and resume_checkpoint.var_noise_raw is not None:
        log.info("Using noise calibration from checkpoint")
        var_noise_raw = resume_checkpoint.var_noise_raw
        var_noise_by_encoding_model = resume_checkpoint.var_noise_by_encoding
    else:
        if cfg.noise_in_feature_space:
            log.info("Calibrating feature-space noise for raw track")
            var_noise_raw = {}
            for name in model_names:
                n_samples = min(1000, raw_features_np[name].shape[0])
                calib_features = torch.from_numpy(raw_features_np[name][:n_samples]).to(
                    device, dtype=torch.float32
                )
                sigma = calibrate_feature_noise(
                    features=calib_features,
                    target_self_correlation=cfg.noise_ceiling_target,
                    rdm_metric=cfg.metric,
                    n_samples=cfg.feature_noise_n_samples,
                    max_iterations=cfg.feature_noise_max_iter,
                    tolerance=cfg.feature_noise_tolerance,
                    device=device,
                )
                var_noise_raw[name] = float(sigma**2)
                log.info(f"  raw, model={name}: sigma={sigma:.6f}")
        else:
            log.info("Estimating RDM-space noise for raw features")
            var_noise_raw = rdm_noise_by_model(
                raw_features_np,
                model_names,
                device,
                metric=cfg.metric,
                target_nc=cfg.noise_ceiling_target,
                corr_type=cfg.corr_type,
            )

        var_noise_by_encoding_model = {}
        for track in track_cfgs_resolved:
            if track.get("type") != "encoding":
                continue
            enc_name = track.get("encoding_name")
            if not enc_name or enc_name in var_noise_by_encoding_model:
                continue
            log.info(f"Calibrating noise for encoding '{enc_name}'")
            var_noise_by_encoding_model[enc_name] = calibrate_encoding_noise(
                encoding_name=enc_name,
                raw_features=raw_features_np,
                encoding_params=encoding_params_by_encoding,
                model_names=model_names,
                metric=cfg.metric,
                corr_type=cfg.corr_type,
                noise_ceiling_target=cfg.noise_ceiling_target,
                device=device,
                max_images=max_images,
            )

    track_definitions = build_track_definitions(
        track_cfgs_resolved,
        var_noise_raw,
        var_noise_by_encoding_model,
    )

    track_agg_cfg_dict = _to_python_dict(cfg.track_aggregation)

    # Optional: derive per-track weights from a single raw_vs_encoded weight.
    #
    # If `raw_weight` is provided (0–1), we interpret it as the total weight for the
    # raw track, and distribute the remaining mass uniformly across all encoding
    # tracks. This is only supported for setups with exactly one identity track
    # (assumed to be the raw space) and one or more encoding tracks.
    raw_weight_cfg = track_agg_cfg_dict.get("raw_weight", None)
    explicit_weights_cfg = track_agg_cfg_dict.get("weights")

    agg_method: str
    weights: Optional[dict[str, float]]

    if raw_weight_cfg is not None:
        if explicit_weights_cfg is not None:
            raise ValueError(
                "track_aggregation.raw_weight and track_aggregation.weights "
                "are mutually exclusive."
            )

        raw_weight = float(raw_weight_cfg)
        if not (0.0 <= raw_weight <= 1.0):
            raise ValueError(
                f"track_aggregation.raw_weight must be in [0, 1], got {raw_weight_cfg!r}"
            )

        raw_tracks = [t.name for t in track_definitions if t.type == "identity"]
        encoding_tracks = [t.name for t in track_definitions if t.type == "encoding"]

        if len(raw_tracks) != 1:
            raise ValueError(
                "track_aggregation.raw_weight currently requires exactly one "
                "identity track (the raw space)."
            )
        if not encoding_tracks:
            raise ValueError(
                "track_aggregation.raw_weight requires at least one encoding track."
            )

        n_enc = len(encoding_tracks)
        remaining = 1.0 - raw_weight
        per_enc = remaining / n_enc if n_enc > 0 else 0.0

        weights = {raw_tracks[0]: raw_weight}
        for name in encoding_tracks:
            weights[name] = per_enc

        # Use config's agg_method if explicitly set, otherwise default to weighted_mean
        config_agg_method = track_agg_cfg_dict.get("agg_method")
        if config_agg_method:  # Explicit method set in config
            agg_method = config_agg_method
        else:  # null/unset -> auto-select weighted_mean when raw_weight is used
            agg_method = "weighted_mean"
    else:
        weights = (
            _to_python_dict(explicit_weights_cfg)
            if explicit_weights_cfg is not None
            else None
        )
        # Use explicit agg_method if set, otherwise default to "mean"
        agg_method = track_agg_cfg_dict.get("agg_method") or "mean"

    track_aggregation_config = TrackAggregationConfig(
        norm_method=track_agg_cfg_dict.get("norm_method", "zscore"),
        agg_method=agg_method,
        beta=float(track_agg_cfg_dict.get("beta", 5.0)),
        power=float(track_agg_cfg_dict.get("power", 1.0)),
        weights=weights,
    )

    # Setup image filter if enabled
    image_filter: Optional[ImageFilter] = None
    image_filter_cfg = _to_python_dict(cfg.get("image_filter", {}))
    if image_filter_cfg.get("enabled", False):
        log.info("Image filter enabled - will validate candidates during selection")
        # Save validated images to output directory
        images_save_dir = output_dir / "validated_images"
        filter_config = ImageFilterConfig(
            enabled=True,
            min_resolution=int(image_filter_cfg.get("min_resolution", 1000)),
            natural_prob_threshold=float(
                image_filter_cfg.get("natural_prob_threshold", 0.85)
            ),
            download_timeout=float(image_filter_cfg.get("download_timeout", 10.0)),
            max_attempts_per_iteration=int(
                image_filter_cfg.get("max_attempts_per_iteration", 100)
            ),
            parallel_batch_size=int(image_filter_cfg.get("parallel_batch_size", 10)),
            classifier_path=Path(image_filter_cfg["classifier_path"])
            if image_filter_cfg.get("classifier_path")
            else None,
            save_dir=images_save_dir,
        )
        image_filter = ImageFilter(
            config=filter_config,
            shard_slices=raw_shard_slices,
            subset_root=Path(cfg.paths.subset_root),
        )
        log.info(
            "Image filter config: min_resolution=%d, natural_prob_threshold=%.2f, max_attempts=%d, save_dir=%s",
            filter_config.min_resolution,
            filter_config.natural_prob_threshold,
            filter_config.max_attempts_per_iteration,
            images_save_dir,
        )

    # Parse refinement config
    refinement_cfg = _to_python_dict(cfg.get("refinement", {}))
    refine_max_passes = int(refinement_cfg.get("max_passes", 0))
    refine_min_replacements = int(refinement_cfg.get("min_replacements", 0))

    if refine_max_passes > 0:
        log.info(
            "Refinement enabled: max_passes=%d, min_replacements=%d",
            refine_max_passes,
            refine_min_replacements,
        )

    selection_result = select_stimuli_multitrack(
        raw_features=raw_accessor,
        track_definitions=track_definitions,
        encoding_params_by_encoding=encoding_params_by_encoding,
        track_aggregation=track_aggregation_config,
        target_size=cfg.target_size,
        init_size=cfg.init_size,
        metric=cfg.metric,
        corr_type=cfg.corr_type,
        n_mc_samples=cfg.n_mc_samples,
        use_analytical=cfg.use_analytical,
        aggregation_within_models=cfg.aggregation_within,
        aggregation_across_models=cfg.aggregation_across,
        device=device,
        image_filter=image_filter,
        refine_max_passes=refine_max_passes,
        refine_min_replacements=refine_min_replacements,
        checkpoint_dir=output_dir,
        resume_from=resume_checkpoint,
        var_noise_raw=var_noise_raw,
        var_noise_by_encoding=var_noise_by_encoding_model,
    )

    # Extract result fields
    selected_indices = selection_result.current_indices
    scores_combined = selection_result.scores_combined
    track_states = selection_result.track_states
    scores_per_track_history = selection_result.scores_per_track_history
    refinement_history = selection_result.refinement_history

    # Log best raw combined tracking info
    if refine_max_passes > 0:
        log.info(
            "Best raw combined score: %.4f at pass %d",
            selection_result.best_raw_combined_score,
            selection_result.best_raw_combined_pass + 1,
        )
        if selection_result.best_raw_combined_pass >= 0:
            n_different = np.sum(
                selection_result.current_indices
                != selection_result.best_raw_combined_indices
            )
            log.info(
                "Final indices differ from best_raw_combined by %d positions",
                n_different,
            )

    identity_or_encoding_states = [
        state
        for state in track_states
        if state.definition.type in ("identity", "encoding")
    ]
    group_states = [
        state for state in track_states if state.definition.type == "group_encoding"
    ]

    def tensor_dict_to_numpy(
        data: Optional[Dict[str, torch.Tensor]],
    ) -> dict[str, np.ndarray]:
        if not data:
            return {}
        return {name: tensor.detach().cpu().numpy() for name, tensor in data.items()}

    selected_features_by_view: dict[str, dict[str, np.ndarray]] = {}
    for state in identity_or_encoding_states:
        if not state.selected_features:
            continue
        selected_features_by_view[state.definition.name] = tensor_dict_to_numpy(
            state.selected_features
        )

    selected_features_single = (
        next(iter(selected_features_by_view.values()))
        if len(selected_features_by_view) == 1
        else None
    )

    raw_state = next(
        (state for state in track_states if state.definition.type == "identity"), None
    )
    selected_features_raw = (
        tensor_dict_to_numpy(raw_state.selected_features)
        if raw_state and raw_state.selected_features
        else None
    )

    selected_features_by_encoding: dict[str, dict[str, np.ndarray]] = {}
    for state in group_states:
        if not state.selected_features_by_encoding:
            continue
        for enc_name, per_model in state.selected_features_by_encoding.items():
            selected_features_by_encoding[enc_name] = tensor_dict_to_numpy(per_model)

    multi_view = len(selected_features_by_view) > 1
    encoding_multi = bool(group_states)

    # -------------------------------------------------------------------------
    # Extract features for alternative selections (greedy, best_raw_combined)
    # -------------------------------------------------------------------------
    # Collect required encoding names from track definitions
    required_encoding_names = []
    for td in track_definitions:
        if td.encoding_name:
            required_encoding_names.append(td.encoding_name)
        if td.encoding_names:
            required_encoding_names.extend(td.encoding_names)
    required_encoding_names = list(set(required_encoding_names))

    def extract_variant_features(
        indices: np.ndarray,
    ) -> tuple[
        Optional[dict[str, np.ndarray]],
        Optional[dict[str, dict[str, np.ndarray]]],
    ]:
        """Extract raw and encoded features for a set of indices."""
        extracted = extract_features_for_indices(
            indices=indices,
            raw_accessor=raw_accessor,
            device=device,
            encoding_params_by_encoding=encoding_params_by_encoding,
            encoding_names=required_encoding_names if required_encoding_names else None,
        )
        raw_np = tensor_dict_to_numpy(extracted.raw_features)
        encoded_np = {
            enc_name: tensor_dict_to_numpy(feats)
            for enc_name, feats in extracted.encoded_features.items()
        }
        return raw_np, encoded_np if encoded_np else None

    # Extract features for greedy indices (before refinement)
    greedy_features_raw: Optional[dict[str, np.ndarray]] = None
    greedy_features_by_encoding: Optional[dict[str, dict[str, np.ndarray]]] = None
    if not np.array_equal(selection_result.greedy_indices, selected_indices):
        log.info("Extracting features for greedy indices (differ from final)")
        greedy_features_raw, greedy_features_by_encoding = extract_variant_features(
            selection_result.greedy_indices
        )
    else:
        log.info("Greedy indices match final - reusing features")
        greedy_features_raw = selected_features_raw
        greedy_features_by_encoding = (
            selected_features_by_encoding if selected_features_by_encoding else None
        )

    # Extract features for best_raw_combined indices
    best_raw_combined_features_raw: Optional[dict[str, np.ndarray]] = None
    best_raw_combined_features_by_encoding: Optional[
        dict[str, dict[str, np.ndarray]]
    ] = None
    if np.array_equal(selection_result.best_raw_combined_indices, selected_indices):
        log.info("Best raw combined indices match final - reusing features")
        best_raw_combined_features_raw = selected_features_raw
        best_raw_combined_features_by_encoding = (
            selected_features_by_encoding if selected_features_by_encoding else None
        )
    elif np.array_equal(
        selection_result.best_raw_combined_indices, selection_result.greedy_indices
    ):
        log.info("Best raw combined indices match greedy - reusing features")
        best_raw_combined_features_raw = greedy_features_raw
        best_raw_combined_features_by_encoding = greedy_features_by_encoding
    else:
        log.info("Extracting features for best_raw_combined indices (differ from both)")
        best_raw_combined_features_raw, best_raw_combined_features_by_encoding = (
            extract_variant_features(selection_result.best_raw_combined_indices)
        )

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    selected_list = [int(x) for x in selected_indices.tolist()]
    selected_image_records = build_selected_image_records(
        selected_list, raw_shard_slices
    )

    # Build image records for variant selections
    greedy_list = [int(x) for x in selection_result.greedy_indices.tolist()]
    greedy_image_records = build_selected_image_records(greedy_list, raw_shard_slices)

    best_raw_combined_list = [
        int(x) for x in selection_result.best_raw_combined_indices.tolist()
    ]
    best_raw_combined_image_records = build_selected_image_records(
        best_raw_combined_list, raw_shard_slices
    )

    out_pkl = output_dir / "selected_stimuli_data.pkl"
    log.info(f"Saving results to {out_pkl}")

    var_noise_payload = {
        state.definition.name: dict(state.definition.var_noise_by_model)
        for state in track_states
        if state.definition.var_noise_by_model is not None
    }

    scores_per_view_history = (
        {
            name: scores_per_track_history.get(name, [])
            for name in selected_features_by_view.keys()
        }
        if multi_view
        else None
    )
    scores_per_rep_history = (
        {
            state.definition.name: scores_per_track_history.get(
                state.definition.name, []
            )
            for state in track_states
        }
        if encoding_multi
        else None
    )
    payload_scores = scores_combined

    # Convert refinement history to serializable format
    refinement_history_payload = None
    if refinement_history:
        refinement_history_payload = [
            {
                "pass": rec.pass_num,
                "position": rec.position,
                "old_idx": rec.old_idx,
                "new_idx": rec.new_idx,
                "score": rec.score,
                "replaced": rec.replaced,
                "scores_per_track": rec.scores_per_track,
            }
            for rec in refinement_history
        ]
        # Log summary
        total_replacements = sum(1 for rec in refinement_history if rec.replaced)
        num_passes = max(rec.pass_num for rec in refinement_history) + 1
        log.info(
            "Refinement summary: %d passes, %d total replacements",
            num_passes,
            total_replacements,
        )

    # Convert filter records to serializable format
    filter_records_payload = None
    if image_filter is not None and image_filter.filter_records:
        filter_records_payload = [
            {
                "global_idx": rec.global_idx,
                "passed": rec.passed,
                "reason": rec.reason,
                "shard_name": rec.shard_name,
                "image_name": rec.image_name,
                "width": rec.width,
                "height": rec.height,
                "natural_prob": rec.natural_prob,
                "score": rec.score,
                "scores_per_track": rec.scores_per_track,
                "rank": rec.rank,
                "phase": rec.phase,
                "iteration": rec.iteration,
                "refinement_position": rec.refinement_position,
                "saved_path": rec.saved_path,
            }
            for rec in image_filter.filter_records
        ]
        # Log summary
        num_passed = sum(1 for rec in image_filter.filter_records if rec.passed)
        num_failed = len(image_filter.filter_records) - num_passed
        log.info(
            "Filter records: %d total evaluations (%d passed, %d failed)",
            len(image_filter.filter_records),
            num_passed,
            num_failed,
        )

    payload = {
        "multi_view": multi_view,
        "encoding_multi": encoding_multi,
        "selected_global_indices": selected_indices,
        "greedy_indices": selection_result.greedy_indices,
        "best_raw_combined_indices": selection_result.best_raw_combined_indices,
        "best_raw_combined_score": selection_result.best_raw_combined_score,
        "best_raw_combined_pass": selection_result.best_raw_combined_pass,
        "model_names": model_names,
        "selected_image_records": selected_image_records,
        "greedy_image_records": greedy_image_records,
        "best_raw_combined_image_records": best_raw_combined_image_records,
        "var_noise_by_model": var_noise_payload,
        "scores": payload_scores,
        # Final selection features
        "selected_features": selected_features_single,
        "selected_features_by_view": selected_features_by_view if multi_view else None,
        "scores_per_view_history": scores_per_view_history if multi_view else None,
        "selected_features_raw": selected_features_raw,
        "selected_features_by_encoding": selected_features_by_encoding
        if selected_features_by_encoding
        else None,
        # Greedy selection features (before refinement)
        "greedy_features_raw": greedy_features_raw,
        "greedy_features_by_encoding": greedy_features_by_encoding,
        # Best raw combined features (peak combined score during refinement)
        "best_raw_combined_features_raw": best_raw_combined_features_raw,
        "best_raw_combined_features_by_encoding": best_raw_combined_features_by_encoding,
        "scores_per_rep_history": scores_per_rep_history if encoding_multi else None,
        "refinement_history": refinement_history_payload,
        "filter_records": filter_records_payload,
        "track_definitions": track_cfgs_resolved,
        "track_aggregation": track_agg_cfg_dict,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f)

    FunctionTimer.log_summary()
    log.info("Done!")


if __name__ == "__main__":
    main()
