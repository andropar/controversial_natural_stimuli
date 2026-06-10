"""Batch-size tuning helpers for layer-sweep feature extraction.

The all-layer variants are sensitive to both GPU memory and extraction
throughput. A hand-picked batch size is usually either conservative for small
models or too aggressive for large ViTs, so the scripts can benchmark a short
probe batch per model and reuse the best successful value.
"""

from __future__ import annotations

import gc
import time
from typing import Callable, Iterable, List, Sequence, Tuple


DEFAULT_BATCH_CANDIDATES = (1, 2, 4, 8, 16, 32)


def parse_batch_size(value):
    """Parse ``--batch-size`` values.

    Returns either ``"auto"`` or a positive integer. Kept separate from
    argparse so callers can use the same logic from tests or notebooks.
    """
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("batch size must be positive")
        return value
    value = str(value).strip().lower()
    if value == "auto":
        return "auto"
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("batch size must be 'auto' or a positive integer") from exc
    if parsed <= 0:
        raise ValueError("batch size must be positive")
    return parsed


def parse_batch_candidates(value: str | Sequence[int] | None) -> List[int]:
    """Parse a comma-separated candidate list and ensure candidate 1 exists."""
    if value is None:
        candidates = list(DEFAULT_BATCH_CANDIDATES)
    elif isinstance(value, str):
        candidates = [int(v) for v in value.split(",") if v.strip()]
    else:
        candidates = [int(v) for v in value]
    candidates = sorted({c for c in candidates if c > 0})
    if 1 not in candidates:
        candidates.insert(0, 1)
    return candidates


def _is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "out of memory" in text
        or "cuda error: out of memory" in text
        or "cudnn_status_alloc_failed" in text
        or "defaultcpuallocator" in text and "can't allocate memory" in text
    )


def _cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _peak_cuda_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / 1024**2)
    except Exception:
        pass
    return None


def _reset_peak_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _make_probe_batch(items: Sequence, batch_size: int) -> List:
    if not items:
        raise ValueError("Need at least one probe item to tune batch size")
    reps = (batch_size + len(items) - 1) // len(items)
    return list((list(items) * reps)[:batch_size])


def tune_batch_size(
    extract_batch: Callable[[Sequence], object],
    probe_items: Sequence,
    candidates: Iterable[int] = DEFAULT_BATCH_CANDIDATES,
    *,
    warmup_batches: int = 1,
    timed_batches: int = 3,
    verbose: bool = True,
) -> Tuple[int, List[dict]]:
    """Benchmark candidate batch sizes and return the fastest successful one.

    ``extract_batch`` should run one real extraction for a sequence of probe
    items and return activations. Any returned object is immediately discarded.
    CUDA OOMs are caught, memory is cleared, and larger candidates are skipped
    because activation memory is monotonic in batch size for a fixed extractor.
    """
    candidates = parse_batch_candidates(candidates)
    records: List[dict] = []
    best_bs = None
    best_rate = -1.0

    for bs in candidates:
        batch = _make_probe_batch(probe_items, bs)
        _cleanup_cuda()
        try:
            for _ in range(warmup_batches):
                out = extract_batch(batch)
                del out
                _sync_cuda()

            _reset_peak_cuda()
            t0 = time.perf_counter()
            for _ in range(timed_batches):
                out = extract_batch(batch)
                del out
                _sync_cuda()
            elapsed = time.perf_counter() - t0
            rate = (bs * timed_batches) / max(elapsed, 1e-9)
            peak_mb = _peak_cuda_mb()
        except RuntimeError as exc:
            _cleanup_cuda()
            if _is_oom_error(exc):
                records.append({
                    "batch_size": bs,
                    "ok": False,
                    "samples_per_sec": 0.0,
                    "elapsed_sec": None,
                    "peak_cuda_mb": None,
                    "error": "oom",
                })
                if verbose:
                    print(f"    batch {bs:>4}: OOM", flush=True)
                break
            raise

        record = {
            "batch_size": bs,
            "ok": True,
            "samples_per_sec": rate,
            "elapsed_sec": elapsed,
            "peak_cuda_mb": peak_mb,
            "error": "",
        }
        records.append(record)
        if verbose:
            mem = "" if peak_mb is None else f", peak={peak_mb:.0f} MiB"
            print(f"    batch {bs:>4}: {rate:7.1f} img/s{mem}", flush=True)
        if rate > best_rate:
            best_bs = bs
            best_rate = rate

    if best_bs is None:
        raise RuntimeError("No batch-size candidate completed successfully")
    return int(best_bs), records


def records_to_array(records: Sequence[dict]):
    """Convert tuning records to a numpy object array for npz metadata."""
    import numpy as np

    dtype = [
        ("batch_size", "i4"),
        ("ok", "?"),
        ("samples_per_sec", "f8"),
        ("elapsed_sec", "f8"),
        ("peak_cuda_mb", "f8"),
        ("error", "U64"),
    ]
    arr = np.zeros(len(records), dtype=dtype)
    for i, rec in enumerate(records):
        arr["batch_size"][i] = int(rec["batch_size"])
        arr["ok"][i] = bool(rec["ok"])
        arr["samples_per_sec"][i] = float(rec.get("samples_per_sec") or 0.0)
        arr["elapsed_sec"][i] = (
            float(rec["elapsed_sec"]) if rec.get("elapsed_sec") is not None else float("nan")
        )
        arr["peak_cuda_mb"][i] = (
            float(rec["peak_cuda_mb"]) if rec.get("peak_cuda_mb") is not None else float("nan")
        )
        arr["error"][i] = str(rec.get("error") or "")
    return arr
