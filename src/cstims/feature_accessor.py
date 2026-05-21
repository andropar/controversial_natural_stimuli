from __future__ import annotations

import multiprocessing as mp
import queue
from typing import Dict, Iterator, List

import numpy as np
import torch
from torch.multiprocessing import Queue

try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass


class FeatureAccessor:
    """
    Provides on-demand GPU staging for features stored in CPU memory.
    Now supports iterator interface matching PrefetchingFeatureAccessor.
    """

    def __init__(
        self,
        features_by_model: Dict[str, np.ndarray],
        model_names: List[str],
        pool_indices: np.ndarray,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        prefetch_batches: int = 3,  # Ignored but accepted for compatibility
    ):
        self.features_cpu = features_by_model
        self.model_names = model_names
        self.pool_indices = pool_indices
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.num_samples = next(iter(features_by_model.values())).shape[0]
        self.num_batches = (len(pool_indices) + batch_size - 1) // batch_size
        self.batch_idx = 0

    def __len__(self):
        return self.num_batches

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __iter__(self) -> Iterator[tuple[int, int, Dict[str, torch.Tensor]]]:
        self.batch_idx = 0
        return self

    def __next__(self) -> tuple[int, int, Dict[str, torch.Tensor]]:
        if self.batch_idx >= self.num_batches:
            raise StopIteration

        start = self.batch_idx * self.batch_size
        end = min(start + self.batch_size, len(self.pool_indices))

        batch_data = {}
        for name in self.model_names:
            batch_np = self.features_cpu[name][start:end]
            if batch_np.dtype == np.float16 and self.dtype == torch.float32:
                batch_np = batch_np.astype(np.float32)
            batch_data[name] = torch.from_numpy(batch_np).to(self.device, self.dtype)

        self.batch_idx += 1
        return start, end, batch_data

class PrefetchingFeatureAccessor:
    """
    An iterator that uses a separate PROCESS to prefetch feature batches.
    This version uses a multiprocessing.Event for robust lifecycle control,
    preventing race conditions during shutdown.
    """

    def __init__(
        self,
        features_by_model: Dict[str, np.ndarray],
        model_names: List[str],
        pool_indices: np.ndarray,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        prefetch_batches: int = 3,
    ):
        self.features_cpu = features_by_model
        self.model_names = model_names
        self.pool_indices = pool_indices
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.num_batches = (len(pool_indices) + batch_size - 1) // batch_size
        self.prefetch_batches = prefetch_batches
        self.prefetch_stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self._queue = None
        self._worker_process = None
        self._stop_event = None

    def __len__(self):
        return self.num_batches

    def __enter__(self):
        self._stop_event = mp.Event()
        self._queue = Queue(maxsize=self.prefetch_batches)
        self._worker_process = mp.Process(
            target=self._producer_loop,
            args=(self._queue, self._stop_event),
            daemon=True,
        )
        self._worker_process.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def __iter__(self) -> Iterator[tuple[int, int, Dict[str, torch.Tensor]]]:
        if not self._worker_process or not self._worker_process.is_alive():
            raise RuntimeError("Iterator must be used within a 'with' block.")
        return self

    def __next__(self) -> tuple[int, int, Dict[str, torch.Tensor]]:
        try:
            item = self._queue.get(timeout=300)
        except queue.Empty:
            if not self._worker_process.is_alive():
                raise StopIteration("Worker process died unexpectedly.")
            else:
                raise RuntimeError(
                    "Data loader timed out. Worker process may be stuck."
                )

        if item is None:
            raise StopIteration

        start, end, batch_data_cpu = item
        if self.prefetch_stream is not None:
            torch.cuda.current_stream(device=self.device).wait_stream(
                self.prefetch_stream
            )
            with torch.cuda.stream(self.prefetch_stream):
                batch_data_gpu = {
                    name: tensor.pin_memory().to(
                        self.device, self.dtype, non_blocking=True
                    )
                    for name, tensor in batch_data_cpu.items()
                }
        else:
            batch_data_gpu = {
                name: tensor.to(self.device, self.dtype)
                for name, tensor in batch_data_cpu.items()
            }
        return start, end, batch_data_gpu

    def _producer_loop(self, queue: Queue, stop_event: mp.Event):
        """Worker process loop: produces all data, sends a sentinel, then waits."""
        try:
            for batch_idx in range(self.num_batches):
                if stop_event.is_set():
                    break
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, len(self.pool_indices))
                batch_data_cpu: Dict[str, torch.Tensor] = {}
                for name in self.model_names:
                    batch_np = self.features_cpu[name][start:end]
                    if batch_np.dtype == np.float16 and self.dtype == torch.float32:
                        batch_np = batch_np.astype(np.float32)
                    batch_data_cpu[name] = torch.from_numpy(batch_np)
                queue.put((start, end, batch_data_cpu))
        finally:
            # Signal completion and then wait for the main process to signal shutdown.
            # This keeps the process alive and its shared memory resources available.
            queue.put(None)
            stop_event.wait()

    def shutdown(self):
        if self._worker_process and self._worker_process.is_alive():
            # Signal the worker that it's safe to exit
            self._stop_event.set()

            # Drain the queue to ensure the worker is not blocked on a `put`
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

            # Wait for the worker to exit gracefully
            self._worker_process.join(timeout=5)
            if self._worker_process.is_alive():
                self._worker_process.terminate()  # Fallback
        self._worker_process = None
