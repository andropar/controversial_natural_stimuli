"""Refit-robust selection helpers."""

from __future__ import annotations

from .cache import NUMBA_AVAILABLE as CACHE_NUMBA_AVAILABLE
from .cache import PredictionPath, RoundCache, StudentRoundCache
from .data import (
    DEFAULT_ALPHA_TARGET_BATCH_SIZE,
    DEFAULT_SCORE_TARGET_BATCH_SIZE,
    FitContext,
    StudentOps,
    TeacherNoiseState,
    TeacherTargets,
    build_fit_context,
    build_noise_states,
    build_proxy_runtime,
    build_refit_splits,
    bound_natural_feature_pool,
    choose_target_columns,
    encode_indices_modelwise_cached,
    proxy_scores_for_pool,
    resolve_base_kernel_precompute,
    topk_shortlist,
)
from .io import (
    format_seconds,
    load_existing_indices,
    load_resume_state,
    make_method,
    parse_csv_floats,
    save_filter_records,
)
from .scoring import NUMBA_AVAILABLE as SCORING_NUMBA_AVAILABLE
from .scoring import score_shortlist_refit_robust

NUMBA_AVAILABLE = CACHE_NUMBA_AVAILABLE and SCORING_NUMBA_AVAILABLE

__all__ = [
    "DEFAULT_ALPHA_TARGET_BATCH_SIZE",
    "DEFAULT_SCORE_TARGET_BATCH_SIZE",
    "FitContext",
    "NUMBA_AVAILABLE",
    "PredictionPath",
    "RoundCache",
    "StudentOps",
    "StudentRoundCache",
    "TeacherNoiseState",
    "TeacherTargets",
    "build_fit_context",
    "build_noise_states",
    "build_proxy_runtime",
    "build_refit_splits",
    "bound_natural_feature_pool",
    "choose_target_columns",
    "encode_indices_modelwise_cached",
    "format_seconds",
    "load_existing_indices",
    "load_resume_state",
    "make_method",
    "parse_csv_floats",
    "proxy_scores_for_pool",
    "resolve_base_kernel_precompute",
    "save_filter_records",
    "score_shortlist_refit_robust",
    "topk_shortlist",
]
