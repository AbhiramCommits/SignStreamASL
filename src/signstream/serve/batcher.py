"""Async micro-batching for concurrent inference requests."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

import numpy as np

from signstream.serve.runtime import InferenceBackend

LOGGER = logging.getLogger(__name__)


class MicroBatcher:
    """Coalesce concurrent single-clip requests into bounded backend batches."""

    def __init__(
        self,
        backend: InferenceBackend,
        *,
        max_batch: int = 32,
        max_wait_ms: float = 10.0,
        on_batch: Any | None = None,
    ) -> None:
        self.backend = backend
        self.max_batch = int(max_batch)
        self.max_wait_s = float(max_wait_ms) / 1000.0
        self.on_batch = on_batch
        self._queue: asyncio.Queue[tuple[np.ndarray, asyncio.Future[np.ndarray]]] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._closed = False

    def start(self) -> None:
        """Start the background worker once."""
        if self._worker is None or self._worker.done():
            self._closed = False
            self._worker = asyncio.create_task(self._run())

    async def submit(self, features: np.ndarray) -> np.ndarray:
        """Submit one clip and await its probability row."""
        if self._closed:
            raise RuntimeError("MicroBatcher is closed")
        self.start()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[np.ndarray] = loop.create_future()
        await self._queue.put((np.asarray(features, dtype=np.float32), future))
        return await future

    async def _run(self) -> None:
        while True:
            try:
                first = await self._queue.get()
            except asyncio.CancelledError:  # pragma: no cover - shutdown path
                return
            batch = [first]
            deadline = asyncio.get_running_loop().time() + self.max_wait_s
            while len(batch) < self.max_batch:
                timeout = deadline - asyncio.get_running_loop().time()
                if timeout <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                except TimeoutError:
                    break
                batch.append(item)
            features = np.stack([item[0] for item in batch], axis=0)
            try:
                probabilities = await asyncio.to_thread(self.backend.predict, features)
                if self.on_batch is not None:
                    self.on_batch(len(batch))
                for index, (_, future) in enumerate(batch):
                    if not future.done():
                        future.set_result(probabilities[index])
            except Exception as exc:  # noqa: BLE001 - propagate to every request
                LOGGER.exception("Micro-batch inference failed")
                for _, future in batch:
                    if not future.done():
                        future.set_exception(exc)
            finally:
                for _ in batch:
                    self._queue.task_done()

    async def close(self) -> None:
        """Stop the worker and fail queued requests."""
        self._closed = True
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None
