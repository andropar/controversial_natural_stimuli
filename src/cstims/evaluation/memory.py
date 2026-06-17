"""Runtime memory logging helpers for evaluation scripts."""

from __future__ import annotations

import torch


def _get_memory_usage_mb() -> dict[str, float | str]:
    result: dict[str, float | str] = {}
    try:
        import psutil

        process = psutil.Process()
        mem_info = process.memory_info()
        result["rss_mb"] = mem_info.rss / 1024 / 1024
        result["vms_mb"] = mem_info.vms / 1024 / 1024
        vm = psutil.virtual_memory()
        result["system_used_mb"] = vm.used / 1024 / 1024
        result["system_available_mb"] = vm.available / 1024 / 1024
        result["system_percent"] = vm.percent
    except ImportError:
        result["error"] = "psutil not installed"
    return result


def _get_gpu_memory_mb() -> dict[str, float]:
    result: dict[str, float] = {}
    if torch.cuda.is_available():
        result["allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        result["reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        result["max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
    return result


def log_memory(label: str) -> None:
    """Print current CPU and GPU memory usage."""
    cpu_mem = _get_memory_usage_mb()
    gpu_mem = _get_gpu_memory_mb()

    parts = [f"[MEMORY {label}]"]
    if "rss_mb" in cpu_mem:
        parts.append(
            "CPU: "
            f"{cpu_mem['rss_mb']:.0f}MB "
            f"(system: {cpu_mem['system_percent']:.0f}% used, "
            f"{cpu_mem['system_available_mb']:.0f}MB free)"
        )
    if gpu_mem:
        parts.append(
            "GPU: "
            f"{gpu_mem['allocated_mb']:.0f}MB allocated, "
            f"{gpu_mem['reserved_mb']:.0f}MB reserved"
        )

    print(" | ".join(parts))
