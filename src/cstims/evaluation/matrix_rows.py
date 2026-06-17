"""Helpers for flattening evaluation matrices into CSV rows."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


CORRELATION_MATRIX_TYPES = (
    "selected_clean",
    "selected_noised",
    "random_clean",
    "random_noised",
)


def correlation_matrix_rows(
    *,
    track_name: str,
    correlation_info: dict,
    matrix_types: Iterable[str] = CORRELATION_MATRIX_TYPES,
    extra_fields: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Flatten correlation matrices returned by recovery evaluation."""
    model_names = correlation_info["model_names"]
    extra_fields = dict(extra_fields or {})
    rows: list[dict[str, Any]] = []
    for matrix_type in matrix_types:
        if matrix_type not in correlation_info:
            continue
        matrix = correlation_info[matrix_type]
        for i, model_i in enumerate(model_names):
            for j, model_j in enumerate(model_names):
                rows.append(
                    {
                        "track": track_name,
                        "matrix_type": matrix_type,
                        "model_i": model_i,
                        "model_j": model_j,
                        "correlation": float(matrix[i][j]),
                        **extra_fields,
                    }
                )
    return rows


def append_correlation_matrix_rows(
    rows: list[dict[str, Any]],
    *,
    track_name: str,
    correlation_info: dict,
    matrix_types: Iterable[str] = CORRELATION_MATRIX_TYPES,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    """Append flattened correlation-matrix rows to an existing list."""
    rows.extend(
        correlation_matrix_rows(
            track_name=track_name,
            correlation_info=correlation_info,
            matrix_types=matrix_types,
            extra_fields=extra_fields,
        )
    )
