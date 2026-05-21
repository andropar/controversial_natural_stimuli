from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm, trange

if TYPE_CHECKING:
    from .checkpoint import SelectionCheckpoint
    from .image_filter import ImageFilter

_LOG = logging.getLogger(__name__)

from ..encoding.linear import (
    EncodingParamsByEncoding,
    encode_batch_for_all_encodings,
)
from ..feature_accessor import PrefetchingFeatureAccessor
from ..rdm_cuda import get_rdm_vector
from ..timing import timed
from .primitives import compute_pairwise_distances
from .utility import (
    compute_batch_utilities,
    compute_multi_subject_encoded_utilities_optimized,
)

TrackType = Literal["identity", "encoding", "group_encoding"]


@dataclass
class TrackDefinition:
    """Configuration for a single scoring track."""

    name: str
    type: TrackType
    encoding_name: Optional[str] = None
    encoding_names: Optional[List[str]] = None
    var_noise_by_model: Optional[Dict[str, float]] = None


@dataclass
class TrackAggregationConfig:
    """Global normalization and aggregation settings across tracks."""

    norm_method: str = "zscore"
    agg_method: str = "mean"
    beta: float = 5.0
    power: float = (
        1.0  # For power_mean: p=1 is arithmetic, p=-1 is harmonic, p→-∞ is min
    )
    weights: Optional[Dict[str, float]] = None


@dataclass
class TrackRuntimeState:
    """Holds mutable selection state for each track."""

    definition: TrackDefinition
    selected_features: Optional[Dict[str, torch.Tensor]] = None
    rdm_by_model: Optional[Dict[str, torch.Tensor]] = None
    noise_stds: Optional[torch.Tensor] = None
    noise_vars: Optional[torch.Tensor] = None
    selected_features_by_encoding: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    rdm_by_encoding_model: Optional[Dict[str, Dict[str, torch.Tensor]]] = None


def _collect_required_encodings_from_defs(
    track_definitions: Sequence[TrackDefinition],
) -> List[str]:
    names = []
    for track in track_definitions:
        if track.type == "encoding" and track.encoding_name:
            names.append(track.encoding_name)
        elif track.type == "group_encoding" and track.encoding_names:
            names.extend(track.encoding_names)
    return sorted(set(names))


def _collect_required_encodings_from_states(
    track_states: Sequence[TrackRuntimeState],
) -> List[str]:
    names = []
    for state in track_states:
        if state.definition.type == "encoding" and state.definition.encoding_name:
            names.append(state.definition.encoding_name)
        elif (
            state.definition.type == "group_encoding"
            and state.definition.encoding_names
        ):
            names.extend(state.definition.encoding_names)
    return sorted(set(names))


def _load_raw_features(
    accessor: PrefetchingFeatureAccessor,
    indices: np.ndarray,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        name: torch.from_numpy(accessor.features_cpu[name][indices])
        .to(device=device, dtype=torch.float32)
        .clone()
        for name in accessor.model_names
    }


def _clone_feature_dict(features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in features.items()}


@dataclass
class ExtractedFeatures:
    """Features extracted for a set of indices."""

    raw_features: Dict[str, torch.Tensor]  # {model_name: (N, D)}
    encoded_features: Dict[
        str, Dict[str, torch.Tensor]
    ]  # {encoding_name: {model_name: (N, D)}}


def extract_features_for_indices(
    indices: np.ndarray,
    raw_accessor: PrefetchingFeatureAccessor,
    device: torch.device,
    encoding_params_by_encoding: EncodingParamsByEncoding,
    encoding_names: Optional[Sequence[str]] = None,
) -> ExtractedFeatures:
    """Extract raw and encoded features for a given set of indices.

    Args:
        indices: Global indices of stimuli to extract features for
        raw_accessor: Feature accessor for raw features
        device: Device to load features onto
        encoding_params_by_encoding: Encoding parameters
        encoding_names: Which encodings to apply (None = all available)

    Returns:
        ExtractedFeatures with raw and encoded features
    """
    raw_features = _load_raw_features(raw_accessor, indices, device)

    encoded_features: Dict[str, Dict[str, torch.Tensor]] = {}
    if encoding_names is None:
        encoding_names = list(encoding_params_by_encoding.keys())

    if encoding_names:
        subset = _prepare_encoding_subset(encoding_params_by_encoding, encoding_names)
        encoded_features = encode_batch_for_all_encodings(raw_features, subset)

    return ExtractedFeatures(
        raw_features=raw_features,
        encoded_features=encoded_features,
    )


def _prepare_encoding_subset(
    encoding_params_by_encoding: EncodingParamsByEncoding,
    encoding_names: Sequence[str],
) -> EncodingParamsByEncoding:
    if not encoding_names:
        return {}
    subset: EncodingParamsByEncoding = {}
    for name in encoding_names:
        if name not in encoding_params_by_encoding:
            raise ValueError(f"Encoding '{name}' not available in loaded params.")
        subset[name] = encoding_params_by_encoding[name]
    return subset


def _initialize_track_states(
    track_definitions: Sequence[TrackDefinition],
    model_names: Sequence[str],
    raw_accessor: PrefetchingFeatureAccessor,
    current_indices: np.ndarray,
    device: torch.device,
    metric: str,
    encoding_params_by_encoding: EncodingParamsByEncoding,
) -> List[TrackRuntimeState]:
    raw_selected = _load_raw_features(raw_accessor, current_indices, device)
    required_encodings = _collect_required_encodings_from_defs(track_definitions)
    encoded_selected: Dict[str, Dict[str, torch.Tensor]] = {}
    if required_encodings:
        subset = _prepare_encoding_subset(
            encoding_params_by_encoding, required_encodings
        )
        encoded_selected = encode_batch_for_all_encodings(raw_selected, subset)

    track_states: List[TrackRuntimeState] = []
    for definition in track_definitions:
        if definition.type == "identity":
            if not definition.var_noise_by_model:
                raise ValueError(
                    f"Track '{definition.name}' is missing noise calibration."
                )
            noise_stds = torch.tensor(
                [np.sqrt(definition.var_noise_by_model[name]) for name in model_names],
                device=device,
                dtype=torch.float32,
            )
            noise_vars = torch.tensor(
                [definition.var_noise_by_model[name] for name in model_names],
                device=device,
                dtype=torch.float32,
            )
            selected = _clone_feature_dict(raw_selected)
            rdm_by_model = {
                name: get_rdm_vector(selected[name], metric) for name in model_names
            }
            track_states.append(
                TrackRuntimeState(
                    definition=definition,
                    selected_features=selected,
                    rdm_by_model=rdm_by_model,
                    noise_stds=noise_stds,
                    noise_vars=noise_vars,
                )
            )
        elif definition.type == "encoding":
            if not definition.encoding_name:
                raise ValueError(f"Track '{definition.name}' is missing encoding_name.")
            if definition.encoding_name not in encoded_selected:
                raise ValueError(
                    f"Encoding '{definition.encoding_name}' not available for track '{definition.name}'."
                )
            if not definition.var_noise_by_model:
                raise ValueError(
                    f"Track '{definition.name}' is missing noise calibration."
                )
            encoded_feats = _clone_feature_dict(
                encoded_selected[definition.encoding_name]
            )
            noise_stds = torch.tensor(
                [np.sqrt(definition.var_noise_by_model[name]) for name in model_names],
                device=device,
                dtype=torch.float32,
            )
            noise_vars = torch.tensor(
                [definition.var_noise_by_model[name] for name in model_names],
                device=device,
                dtype=torch.float32,
            )
            rdm_by_model = {
                name: get_rdm_vector(encoded_feats[name], metric)
                for name in model_names
            }
            track_states.append(
                TrackRuntimeState(
                    definition=definition,
                    selected_features=encoded_feats,
                    rdm_by_model=rdm_by_model,
                    noise_stds=noise_stds,
                    noise_vars=noise_vars,
                )
            )
        else:  # group_encoding
            encoding_names = definition.encoding_names or []
            if not encoding_names:
                raise ValueError(
                    f"Track '{definition.name}' requires at least one encoding."
                )
            missing = [name for name in encoding_names if name not in encoded_selected]
            if missing:
                raise ValueError(
                    f"Track '{definition.name}' missing encodings: {', '.join(missing)}"
                )
            selected_by_encoding: Dict[str, Dict[str, torch.Tensor]] = {}
            rdm_by_encoding_model: Dict[str, Dict[str, torch.Tensor]] = {}
            for enc_name in encoding_names:
                per_model = _clone_feature_dict(encoded_selected[enc_name])
                selected_by_encoding[enc_name] = per_model
                rdm_by_encoding_model[enc_name] = {
                    name: get_rdm_vector(per_model[name], metric)
                    for name in model_names
                }
            track_states.append(
                TrackRuntimeState(
                    definition=definition,
                    selected_features_by_encoding=selected_by_encoding,
                    rdm_by_encoding_model=rdm_by_encoding_model,
                )
            )
    return track_states


def _normalize_scores(scores: torch.Tensor, method: str) -> torch.Tensor:
    if method == "none":
        return scores
    if method == "zscore":
        mu = scores.mean()
        std = scores.std(unbiased=False) + 1e-6
        return (scores - mu) / std
    if method == "minmax":
        min_val = scores.min()
        max_val = scores.max()
        return (scores - min_val) / (max_val - min_val + 1e-6)
    raise ValueError(f"Unknown normalization method '{method}'.")


def _aggregate_track_scores(
    scores_per_track: Dict[str, torch.Tensor],
    cfg: TrackAggregationConfig,
    track_order: Sequence[str],
) -> torch.Tensor:
    normalized = {
        name: _normalize_scores(scores_per_track[name], cfg.norm_method)
        for name in track_order
    }

    if not track_order:
        raise ValueError("No tracks provided for aggregation.")

    if cfg.agg_method == "identity":
        return normalized[track_order[0]]

    stacked = torch.stack([normalized[name] for name in track_order], dim=0)

    if cfg.agg_method == "mean":
        return stacked.mean(dim=0)
    if cfg.agg_method == "min":
        return stacked.min(dim=0).values
    if cfg.agg_method == "smooth_min":
        beta = cfg.beta if cfg.beta > 0 else 1.0
        return -torch.logsumexp(-beta * stacked, dim=0) / beta
    if cfg.agg_method == "power_mean":
        # Generalized (weighted) mean: M_p(x, w) = (sum(w_i * x_i^p))^(1/p)
        # p=1: arithmetic mean, p=-1: harmonic mean, p→-∞: approaches min
        # Supports optional weights via cfg.weights
        p = cfg.power if cfg.power != 0 else 1.0

        # Get weights (default to uniform)
        weights = []
        for name in track_order:
            if not cfg.weights:
                weights.append(1.0)
            else:
                weights.append(cfg.weights.get(name, 1.0))
        weights_tensor = torch.tensor(
            weights, dtype=stacked.dtype, device=stacked.device
        )
        weights_tensor = weights_tensor / weights_tensor.sum()  # Normalize

        # Shift scores to be positive (power mean requires positive values for negative p)
        min_val = stacked.min()
        shifted = stacked - min_val + 1.0  # Shift to [1, ...]

        # Weighted power mean: (sum(w_i * x_i^p))^(1/p)
        weighted_powers = weights_tensor[:, None] * shifted.pow(p)
        power_mean = weighted_powers.sum(dim=0).pow(1.0 / p)

        # Shift back
        return power_mean - 1.0 + min_val
    if cfg.agg_method == "weighted_mean":
        weights = []
        for name in track_order:
            if not cfg.weights:
                weights.append(1.0)
            else:
                weights.append(cfg.weights.get(name, 0.0))
        weights_tensor = torch.tensor(
            weights, dtype=stacked.dtype, device=stacked.device
        )
        if weights_tensor.sum() == 0:
            raise ValueError("All track weights are zero.")
        weights_tensor = weights_tensor / weights_tensor.sum()
        return (weights_tensor[:, None] * stacked).sum(dim=0)

    raise ValueError(f"Unknown aggregation method '{cfg.agg_method}'.")


def _pick_best_candidate(
    raw_features: PrefetchingFeatureAccessor,
    track_states: Sequence[TrackRuntimeState],
    pool_mask: np.ndarray,
    metric: str,
    corr_type: str,
    n_mc_samples: int,
    use_analytical: bool,
    aggregation_within_models: str,
    aggregation_across_models: str,
    track_aggregation: TrackAggregationConfig,
    encoding_params_by_encoding: EncodingParamsByEncoding,
    image_filter: Optional["ImageFilter"],
    phase: str = "greedy",
    iteration: int = 0,
    refinement_position: Optional[int] = None,
) -> tuple[int, float, Dict[str, float]]:
    """Score all candidates and return the best valid one (with filtering if enabled).

    Args:
        raw_features: Feature accessor for raw features.
        track_states: Current track states with RDMs.
        pool_mask: Boolean mask for available candidates.
        metric: Distance metric.
        corr_type: Correlation type.
        n_mc_samples: Number of Monte Carlo samples.
        use_analytical: Whether to use analytical scoring.
        aggregation_within_models: Aggregation method within models.
        aggregation_across_models: Aggregation method across models.
        track_aggregation: Track aggregation configuration.
        encoding_params_by_encoding: Encoding parameters.
        image_filter: Optional image filter.
        phase: "greedy" or "refinement" (for filter record tracking).
        iteration: Iteration number (for filter record tracking).
        refinement_position: Position being refined (for filter record tracking).

    Returns:
        best_idx: Index of the best candidate
        best_score: Combined score of the best candidate
        best_scores_per_track: Per-track scores for the selected candidate
    """
    if image_filter is not None:
        sorted_indices, sorted_scores, sorted_scores_per_track = (
            evaluate_candidates_tracks(
                raw_features=raw_features,
                track_states=track_states,
                pool_mask=pool_mask,
                metric=metric,
                corr_type=corr_type,
                n_mc_samples=n_mc_samples,
                use_analytical=use_analytical,
                aggregation_within_models=aggregation_within_models,
                aggregation_across_models=aggregation_across_models,
                track_aggregation=track_aggregation,
                encoding_params_by_encoding=encoding_params_by_encoding,
                return_all_sorted=True,
            )
        )
        best_idx, best_score, selected_pos = image_filter.select_first_valid(
            sorted_indices,
            sorted_scores,
            candidate_scores_per_track=sorted_scores_per_track,
            phase=phase,
            iteration=iteration,
            refinement_position=refinement_position,
        )
        # Mark failed candidates as unavailable
        for failed_idx in image_filter._failed_indices:
            if pool_mask[failed_idx]:
                pool_mask[failed_idx] = False
                _LOG.debug("Excluding failed candidate %d from pool", failed_idx)
        # Get per-track scores for the selected candidate
        # selected_pos is 1-indexed (count of attempts), convert to 0-indexed
        pos_0idx = selected_pos - 1
        best_scores_per_track: Dict[str, float] = {
            name: float(scores[pos_0idx].item())
            for name, scores in sorted_scores_per_track.items()
        }
    else:
        best_score, best_idx, best_scores_per_track = evaluate_candidates_tracks(
            raw_features=raw_features,
            track_states=track_states,
            pool_mask=pool_mask,
            metric=metric,
            corr_type=corr_type,
            n_mc_samples=n_mc_samples,
            use_analytical=use_analytical,
            aggregation_within_models=aggregation_within_models,
            aggregation_across_models=aggregation_across_models,
            track_aggregation=track_aggregation,
            encoding_params_by_encoding=encoding_params_by_encoding,
        )
    return best_idx, best_score, best_scores_per_track


@timed
def evaluate_candidates_tracks(
    raw_features: PrefetchingFeatureAccessor,
    track_states: Sequence[TrackRuntimeState],
    pool_mask: np.ndarray,
    metric: str,
    corr_type: str,
    n_mc_samples: int,
    use_analytical: bool,
    aggregation_within_models: str,
    aggregation_across_models: str,
    track_aggregation: TrackAggregationConfig,
    encoding_params_by_encoding: EncodingParamsByEncoding,
    return_all_sorted: bool = False,
) -> (
    tuple[float, int, Dict[str, float]]
    | tuple[np.ndarray, np.ndarray, Dict[str, torch.Tensor]]
):
    model_names = list(raw_features.model_names)
    required_encodings = _collect_required_encodings_from_states(track_states)
    encoding_subset = _prepare_encoding_subset(
        encoding_params_by_encoding, required_encodings
    )

    num_candidates = int(pool_mask.sum())
    if num_candidates == 0:
        raise ValueError("No candidates left in the pool.")

    scores_per_track = {
        state.definition.name: torch.empty(num_candidates, dtype=torch.float32)
        for state in track_states
    }
    candidate_indices = torch.empty(num_candidates, dtype=torch.long)

    write_pos = 0
    with raw_features:
        for batch_start, batch_end, batch_features in tqdm(
            raw_features,
            desc="Evaluating candidates (tracks)",
            total=len(raw_features),
            leave=False,
        ):
            batch_indices = raw_features.pool_indices[batch_start:batch_end]
            valid_mask = pool_mask[batch_indices]
            B_valid = int(valid_mask.sum())
            if B_valid == 0:
                continue

            candidate_indices[write_pos : write_pos + B_valid] = torch.from_numpy(
                batch_indices[valid_mask]
            )

            if valid_mask.all():
                candidate_raw = batch_features
            else:
                candidate_raw = {
                    name: batch_features[name][valid_mask] for name in model_names
                }

            encoded_batches: Dict[str, Dict[str, torch.Tensor]] = {}
            if encoding_subset:
                encoded_batches = encode_batch_for_all_encodings(
                    candidate_raw, encoding_subset
                )

            for state in track_states:
                name = state.definition.name
                if state.definition.type == "identity":
                    scores = compute_batch_utilities(
                        candidate_features=candidate_raw,
                        rdm_by_model=state.rdm_by_model,
                        selected_features=state.selected_features,
                        noise_stds=state.noise_stds,
                        noise_vars=state.noise_vars,
                        metric=metric,
                        corr_type=corr_type,
                        n_mc_samples=n_mc_samples,
                        aggregation_within=aggregation_within_models,
                        aggregation_across=aggregation_across_models,
                        use_analytical=use_analytical,
                        noise_in_feature_space=False,
                    )
                elif state.definition.type == "encoding":
                    enc_name = state.definition.encoding_name
                    if not enc_name or enc_name not in encoded_batches:
                        raise ValueError(
                            f"Encoded features for '{enc_name}' not available."
                        )
                    scores = compute_batch_utilities(
                        candidate_features=encoded_batches[enc_name],
                        rdm_by_model=state.rdm_by_model,
                        selected_features=state.selected_features,
                        noise_stds=state.noise_stds,
                        noise_vars=state.noise_vars,
                        metric=metric,
                        corr_type=corr_type,
                        n_mc_samples=n_mc_samples,
                        aggregation_within=aggregation_within_models,
                        aggregation_across=aggregation_across_models,
                        use_analytical=use_analytical,
                        noise_in_feature_space=False,
                    )
                else:
                    enc_names = state.definition.encoding_names or []
                    missing = [enc for enc in enc_names if enc not in encoded_batches]
                    if missing:
                        raise ValueError(
                            f"Missing encoded batches for track '{name}': {missing}"
                        )
                    candidate_features_by_encoding = {
                        enc: encoded_batches[enc] for enc in enc_names
                    }
                    scores = compute_multi_subject_encoded_utilities_optimized(
                        candidate_features_by_encoding=candidate_features_by_encoding,
                        rdm_by_encoding_model=state.rdm_by_encoding_model,
                        selected_features_by_encoding=state.selected_features_by_encoding,
                        metric=metric,
                        corr_type=corr_type,
                    )

                scores_per_track[name][write_pos : write_pos + B_valid] = (
                    scores.detach().cpu().to(torch.float32)
                )

            write_pos += B_valid

    if write_pos != num_candidates:
        raise RuntimeError(
            f"Bookkeeping mismatch: wrote {write_pos} scores but expected {num_candidates}."
        )

    track_order = [state.definition.name for state in track_states]
    combined_scores = _aggregate_track_scores(
        scores_per_track, track_aggregation, track_order
    )

    if return_all_sorted:
        # Return all candidates sorted by score (descending), plus per-track scores
        sorted_order = torch.argsort(combined_scores, descending=True)
        sorted_indices = candidate_indices[sorted_order].numpy()
        sorted_scores = combined_scores[sorted_order].numpy()
        # Reorder per-track scores to match sorted order
        sorted_scores_per_track = {
            name: scores[sorted_order] for name, scores in scores_per_track.items()
        }
        return sorted_indices, sorted_scores, sorted_scores_per_track

    best_pos = int(combined_scores.argmax().item())
    best_idx = int(candidate_indices[best_pos].item())
    best_score = float(combined_scores[best_pos].item())
    best_scores_per_track = {
        name: float(scores_per_track[name][best_pos].item())
        for name in scores_per_track
    }
    return best_score, best_idx, best_scores_per_track


def _append_single_representation(
    selected_features: Dict[str, torch.Tensor],
    rdm_by_model: Dict[str, torch.Tensor],
    new_features: Dict[str, torch.Tensor],
    metric: str,
) -> None:
    for name, feat in new_features.items():
        selected_features[name] = torch.cat([selected_features[name], feat], dim=0)
        new_dists = compute_pairwise_distances(
            feat, selected_features[name][:-1], metric=metric
        ).squeeze()
        rdm_by_model[name] = torch.cat([rdm_by_model[name], new_dists])


def _append_group_representation(
    selected_features_by_encoding: Dict[str, Dict[str, torch.Tensor]],
    rdm_by_encoding_model: Dict[str, Dict[str, torch.Tensor]],
    new_features_by_encoding: Dict[str, Dict[str, torch.Tensor]],
    metric: str,
) -> None:
    for enc_name, per_model_feats in new_features_by_encoding.items():
        for model_name, feat in per_model_feats.items():
            selected = selected_features_by_encoding[enc_name][model_name]
            selected_features_by_encoding[enc_name][model_name] = torch.cat(
                [selected, feat], dim=0
            )
            new_dists = compute_pairwise_distances(
                feat,
                selected_features_by_encoding[enc_name][model_name][:-1],
                metric=metric,
            ).squeeze()
            rdm_by_encoding_model[enc_name][model_name] = torch.cat(
                [rdm_by_encoding_model[enc_name][model_name], new_dists]
            )


def _update_track_states(
    track_states: Sequence[TrackRuntimeState],
    best_idx: int,
    raw_accessor: PrefetchingFeatureAccessor,
    device: torch.device,
    metric: str,
    encoding_params_by_encoding: EncodingParamsByEncoding,
) -> None:
    raw_new = _load_raw_features(
        raw_accessor, np.array([best_idx], dtype=np.int64), device
    )
    required_encodings = _collect_required_encodings_from_states(track_states)
    encoded_new: Dict[str, Dict[str, torch.Tensor]] = {}
    if required_encodings:
        subset = _prepare_encoding_subset(
            encoding_params_by_encoding, required_encodings
        )
        encoded_new = encode_batch_for_all_encodings(raw_new, subset)

    for state in track_states:
        if state.definition.type == "identity":
            _append_single_representation(
                state.selected_features,
                state.rdm_by_model,
                {name: feat for name, feat in raw_new.items()},
                metric,
            )
        elif state.definition.type == "encoding":
            enc_name = state.definition.encoding_name
            if enc_name not in encoded_new:
                raise ValueError(
                    f"Encoded features for '{enc_name}' not available when updating state."
                )
            _append_single_representation(
                state.selected_features,
                state.rdm_by_model,
                {name: feat for name, feat in encoded_new[enc_name].items()},
                metric,
            )
        else:
            enc_names = state.definition.encoding_names or []
            missing = [enc for enc in enc_names if enc not in encoded_new]
            if missing:
                raise ValueError(
                    f"Missing encodings {missing} when updating track '{state.definition.name}'."
                )
            per_encoding_features = {enc: encoded_new[enc] for enc in enc_names}
            _append_group_representation(
                state.selected_features_by_encoding,
                state.rdm_by_encoding_model,
                per_encoding_features,
                metric,
            )


@dataclass
class RefinementRecord:
    """Record of a single refinement step."""

    pass_num: int
    position: int
    old_idx: int
    new_idx: int
    score: float
    replaced: bool
    scores_per_track: Optional[Dict[str, float]] = None


@dataclass
class SelectionResult:
    """Result of stimulus selection including refinement tracking."""

    current_indices: np.ndarray  # Final indices after all refinement
    greedy_indices: np.ndarray  # Indices right after greedy phase (before refinement)
    best_raw_combined_indices: np.ndarray  # Indices with best raw combined score
    best_raw_combined_score: float  # The best raw combined score seen
    best_raw_combined_pass: int  # Which pass achieved the best (-1 = greedy)
    scores_combined: List[float]  # Combined scores from greedy phase
    track_states: List[TrackRuntimeState]  # Final track states
    scores_per_track_history: Dict[str, List[float]]  # Per-track score history
    refinement_history: List[RefinementRecord]  # Refinement records


@timed
def select_stimuli_multitrack(
    raw_features: PrefetchingFeatureAccessor,
    track_definitions: Sequence[TrackDefinition],
    encoding_params_by_encoding: EncodingParamsByEncoding,
    track_aggregation: TrackAggregationConfig,
    target_size: int,
    init_size: int,
    metric: str,
    corr_type: str,
    n_mc_samples: int,
    use_analytical: bool,
    aggregation_within_models: str,
    aggregation_across_models: str,
    device: torch.device,
    image_filter: Optional["ImageFilter"] = None,
    refine_max_passes: int = 0,
    refine_min_replacements: int = 0,
    checkpoint_dir: Optional[Path] = None,
    resume_from: Optional["SelectionCheckpoint"] = None,
    var_noise_raw: Optional[Dict[str, float]] = None,
    var_noise_by_encoding: Optional[Dict[str, Dict[str, float]]] = None,
) -> SelectionResult:
    """Select stimuli using greedy selection with optional refinement.

    Args:
        raw_features: Feature accessor for raw features.
        track_definitions: List of track definitions.
        encoding_params_by_encoding: Encoding parameters by encoding name.
        track_aggregation: Track aggregation configuration.
        target_size: Target number of stimuli to select.
        init_size: Initial number of stimuli (random).
        metric: Distance metric for RDM computation.
        corr_type: Correlation type for scoring.
        n_mc_samples: Number of Monte Carlo samples.
        use_analytical: Whether to use analytical scoring.
        aggregation_within_models: Aggregation method within models.
        aggregation_across_models: Aggregation method across models.
        device: Torch device.
        image_filter: Optional image filter for candidate validation.
        refine_max_passes: Maximum refinement passes (0 = disabled).
        refine_min_replacements: Stop refinement if replacements <= this value.
        checkpoint_dir: Directory to save checkpoints (None = no checkpointing).
        resume_from: Checkpoint to resume from (None = start fresh).
        var_noise_raw: Noise calibration for raw features (from checkpoint on resume).
        var_noise_by_encoding: Noise calibration by encoding (from checkpoint on resume).

    Returns:
        SelectionResult containing:
        - current_indices: Final indices after all refinement
        - greedy_indices: Indices right after greedy phase (before refinement)
        - best_raw_combined_indices: Indices with best raw combined score seen
        - best_raw_combined_score: The best raw combined score
        - best_raw_combined_pass: Which pass achieved the best (-1 = greedy)
        - scores_combined: Combined scores from greedy phase
        - track_states: Final track states
        - scores_per_track_history: Per-track score history from greedy phase
        - refinement_history: List of refinement records
    """
    if target_size <= init_size:
        raise ValueError("target_size must be greater than init_size.")
    if not track_definitions:
        raise ValueError("At least one track must be defined.")

    model_names = list(raw_features.model_names)
    num_samples = len(raw_features.pool_indices)
    num_to_pick = target_size - init_size

    # Helper for checkpoint saving
    def _save_ckpt(
        phase: str,
        greedy_iter: int,
        refine_pass: int = -1,
        refine_pos: int = -1,
    ) -> None:
        if checkpoint_dir is None:
            return
        from .checkpoint import SelectionCheckpoint, save_checkpoint

        failed_set = set(image_filter._failed_indices) if image_filter else set()
        refine_hist = [
            {
                "pass_num": r.pass_num,
                "position": r.position,
                "old_idx": r.old_idx,
                "new_idx": r.new_idx,
                "score": r.score,
                "replaced": r.replaced,
                "scores_per_track": r.scores_per_track,
            }
            for r in refinement_history
        ]
        ckpt = SelectionCheckpoint(
            phase=phase,  # type: ignore[arg-type]
            greedy_iteration=greedy_iter,
            refinement_pass=refine_pass,
            refinement_position=refine_pos,
            current_indices=current_indices.copy(),
            failed_indices=failed_set,
            scores_combined=list(scores_combined),
            scores_per_track_history={
                k: list(v) for k, v in scores_per_track_history.items()
            },
            refinement_history=refine_hist,
            var_noise_raw=var_noise_raw,
            var_noise_by_encoding=var_noise_by_encoding or {},
        )
        save_checkpoint(checkpoint_dir / "checkpoint.pkl", ckpt)

    # Initialize or restore state
    if resume_from is not None:
        _LOG.info(
            "Resuming from checkpoint: phase=%s, greedy_iter=%d, refine_pass=%d, refine_pos=%d",
            resume_from.phase,
            resume_from.greedy_iteration,
            resume_from.refinement_pass,
            resume_from.refinement_position,
        )
        current_indices = resume_from.current_indices.copy()
        scores_combined = list(resume_from.scores_combined)
        scores_per_track_history = {
            k: list(v) for k, v in resume_from.scores_per_track_history.items()
        }
        refinement_history: List[RefinementRecord] = [
            RefinementRecord(
                pass_num=r["pass_num"],
                position=r["position"],
                old_idx=r["old_idx"],
                new_idx=r["new_idx"],
                score=r["score"],
                replaced=r["replaced"],
                scores_per_track=r.get("scores_per_track"),
            )
            for r in resume_from.refinement_history
        ]

        # Reconstruct pool_mask
        pool_mask = np.ones(num_samples, dtype=bool)
        pool_mask[current_indices] = False
        for idx in resume_from.failed_indices:
            pool_mask[idx] = False

        # Restore image_filter failed indices
        if image_filter is not None:
            image_filter._failed_indices = set(resume_from.failed_indices)

        # Reconstruct track states (fast)
        track_states = _initialize_track_states(
            track_definitions,
            model_names,
            raw_features,
            current_indices,
            device,
            metric,
            encoding_params_by_encoding,
        )

        # Determine resume points
        start_greedy = resume_from.greedy_iteration
        start_refine_pass = (
            resume_from.refinement_pass if resume_from.refinement_pass >= 0 else 0
        )
        start_refine_pos = (
            resume_from.refinement_position + 1
            if resume_from.refinement_position >= 0
            else 0
        )
    else:
        current_indices = np.random.choice(
            np.arange(num_samples), size=init_size, replace=False
        )

        track_states = _initialize_track_states(
            track_definitions,
            model_names,
            raw_features,
            current_indices,
            device,
            metric,
            encoding_params_by_encoding,
        )

        pool_mask = np.ones(num_samples, dtype=bool)
        pool_mask[current_indices] = False

        scores_combined = []
        scores_per_track_history = {track.name: [] for track in track_definitions}
        refinement_history = []

        start_greedy = 0
        start_refine_pass = 0
        start_refine_pos = 0

    # === GREEDY PHASE ===
    # Skip completed iterations on resume
    remaining_greedy = num_to_pick - start_greedy
    if remaining_greedy > 0:
        for greedy_iter in trange(
            start_greedy,
            num_to_pick,
            desc="Selecting stimuli (greedy)",
            initial=start_greedy,
            total=num_to_pick,
        ):
            best_idx, best_score, best_scores_per_track = _pick_best_candidate(
                raw_features=raw_features,
                track_states=track_states,
                pool_mask=pool_mask,
                metric=metric,
                corr_type=corr_type,
                n_mc_samples=n_mc_samples,
                use_analytical=use_analytical,
                aggregation_within_models=aggregation_within_models,
                aggregation_across_models=aggregation_across_models,
                track_aggregation=track_aggregation,
                encoding_params_by_encoding=encoding_params_by_encoding,
                image_filter=image_filter,
                phase="greedy",
                iteration=greedy_iter,
            )

            current_indices = np.concatenate([current_indices, [best_idx]])
            pool_mask[best_idx] = False
            scores_combined.append(best_score)

            for track_name, score in best_scores_per_track.items():
                scores_per_track_history[track_name].append(score)

            _update_track_states(
                track_states=track_states,
                best_idx=best_idx,
                raw_accessor=raw_features,
                device=device,
                metric=metric,
                encoding_params_by_encoding=encoding_params_by_encoding,
            )

            # Save checkpoint after each greedy iteration
            _save_ckpt("greedy", greedy_iter + 1)

    if image_filter is not None:
        _LOG.info(
            "Greedy phase: %d candidates marked as failed during selection",
            image_filter.num_failed,
        )

    # === TRACK BEST RAW COMBINED SCORE ===
    # Save greedy indices and compute initial raw combined score
    greedy_indices = current_indices.copy()

    cfg = track_aggregation  # Alias for convenience in helper function

    def _compute_raw_combined_score(per_track_scores: Dict[str, float]) -> float:
        """Compute raw combined score as weighted average of per-track scores."""
        total_weight = sum(
            cfg.weights.get(name, 1.0) if cfg.weights else 1.0
            for name in per_track_scores
        )
        return sum(
            ((cfg.weights.get(name, 1.0) if cfg.weights else 1.0) / total_weight)
            * score
            for name, score in per_track_scores.items()
        )

    # Use the last per-track scores from greedy phase as baseline
    if scores_per_track_history and all(scores_per_track_history.values()):
        greedy_final_per_track = {
            name: scores[-1] for name, scores in scores_per_track_history.items()
        }
        best_raw_combined_score = _compute_raw_combined_score(greedy_final_per_track)
    else:
        best_raw_combined_score = float("-inf")

    best_raw_combined_indices = greedy_indices.copy()
    best_raw_combined_pass = -1  # -1 indicates greedy phase

    # === REFINEMENT PHASE ===
    # Note: refinement_history is already initialized in the resume/fresh-start block above

    if refine_max_passes > 0:
        _LOG.info(
            "Starting refinement phase (max_passes=%d, min_replacements=%d)",
            refine_max_passes,
            refine_min_replacements,
        )
        total_replacements = 0

        for pass_num in range(start_refine_pass, refine_max_passes):
            replacements_this_pass = 0

            # Determine starting position for this pass
            pos_start = start_refine_pos if pass_num == start_refine_pass else 0
            n_positions = len(current_indices)

            for pos in trange(
                pos_start,
                n_positions,
                desc=f"Refinement pass {pass_num + 1}/{refine_max_passes}",
                initial=pos_start,
                total=n_positions,
            ):
                original_idx = int(current_indices[pos])

                # Build indices without position `pos`
                indices_without = np.delete(current_indices, pos)

                # Reinitialize track states from N-1 stimuli
                track_states = _initialize_track_states(
                    track_definitions,
                    model_names,
                    raw_features,
                    indices_without,
                    device,
                    metric,
                    encoding_params_by_encoding,
                )

                # Rebuild pool: exclude indices_without and failed candidates
                pool_mask = np.ones(num_samples, dtype=bool)
                pool_mask[indices_without] = False
                if image_filter is not None:
                    for failed_idx in image_filter._failed_indices:
                        pool_mask[failed_idx] = False

                # Find best candidate (original_idx is back in the pool)
                best_idx, best_score, best_scores_per_track = _pick_best_candidate(
                    raw_features=raw_features,
                    track_states=track_states,
                    pool_mask=pool_mask,
                    metric=metric,
                    corr_type=corr_type,
                    n_mc_samples=n_mc_samples,
                    use_analytical=use_analytical,
                    aggregation_within_models=aggregation_within_models,
                    aggregation_across_models=aggregation_across_models,
                    track_aggregation=track_aggregation,
                    encoding_params_by_encoding=encoding_params_by_encoding,
                    image_filter=image_filter,
                    phase="refinement",
                    iteration=pass_num,
                    refinement_position=pos,
                )

                replaced = best_idx != original_idx
                if replaced:
                    replacements_this_pass += 1

                refinement_history.append(
                    RefinementRecord(
                        pass_num=pass_num,
                        position=pos,
                        old_idx=original_idx,
                        new_idx=best_idx,
                        score=best_score,
                        replaced=replaced,
                        scores_per_track=best_scores_per_track
                        if best_scores_per_track
                        else None,
                    )
                )

                current_indices[pos] = best_idx

                # Save checkpoint after each refinement position
                _save_ckpt("refinement", num_to_pick, pass_num, pos)

            total_replacements += replacements_this_pass
            _LOG.info(
                "Refinement pass %d: %d replacements",
                pass_num + 1,
                replacements_this_pass,
            )

            # Track best raw combined score after each pass
            # Get the last per-track scores from this pass
            pass_records = [r for r in refinement_history if r.pass_num == pass_num]
            if pass_records and pass_records[-1].scores_per_track:
                last_per_track = pass_records[-1].scores_per_track
                current_raw_combined = _compute_raw_combined_score(last_per_track)
                _LOG.info(
                    "Refinement pass %d: raw_combined=%.4f (best=%.4f at pass %d)",
                    pass_num + 1,
                    current_raw_combined,
                    best_raw_combined_score,
                    best_raw_combined_pass + 1,
                )
                if current_raw_combined > best_raw_combined_score:
                    best_raw_combined_score = current_raw_combined
                    best_raw_combined_indices = current_indices.copy()
                    best_raw_combined_pass = pass_num
                    _LOG.info(
                        "New best raw combined score: %.4f at pass %d",
                        best_raw_combined_score,
                        pass_num + 1,
                    )

            if replacements_this_pass <= refine_min_replacements:
                _LOG.info(
                    "Refinement converged (replacements=%d <= min_replacements=%d)",
                    replacements_this_pass,
                    refine_min_replacements,
                )
                break

        _LOG.info("Refinement complete: %d total replacements", total_replacements)

        # Rebuild final track states from refined indices
        track_states = _initialize_track_states(
            track_definitions,
            model_names,
            raw_features,
            current_indices,
            device,
            metric,
            encoding_params_by_encoding,
        )

        # Rebuild pool_mask for consistency
        pool_mask = np.ones(num_samples, dtype=bool)
        pool_mask[current_indices] = False
        if image_filter is not None:
            for failed_idx in image_filter._failed_indices:
                pool_mask[failed_idx] = False

    # Save final checkpoint
    _save_ckpt("complete", num_to_pick)

    return SelectionResult(
        current_indices=current_indices,
        greedy_indices=greedy_indices,
        best_raw_combined_indices=best_raw_combined_indices,
        best_raw_combined_score=best_raw_combined_score,
        best_raw_combined_pass=best_raw_combined_pass,
        scores_combined=scores_combined,
        track_states=track_states,
        scores_per_track_history=scores_per_track_history,
        refinement_history=refinement_history,
    )
