from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

BatchGroupKey = Tuple[Optional[str], int, frozenset]  # (action_norm_key, steps, input_keys)


def resolve_policy_device_dtype(model) -> tuple[torch.device, torch.dtype]:
    """Use the action policy contract for mixed-precision inference inputs."""
    transformer = getattr(model, "transformer", None)
    reference = (
        next(transformer.parameters(), None)
        if transformer is not None
        else None
    )
    if reference is None:
        reference = next(model.parameters())
    return reference.device, reference.dtype


@dataclass
class BatchItem:
    inputs: Dict[str, torch.Tensor]
    action_norm_key: Optional[str]
    steps: int
    future: asyncio.Future
    enqueue_time: float = field(default_factory=time.monotonic)

    @property
    def group_key(self) -> BatchGroupKey:
        return (self.action_norm_key, self.steps, frozenset(self.inputs.keys()))


class InferenceBatcher:

    def __init__(self, model, max_batch_size: int = 8, max_wait_ms: float = 50.0):
        self._model = model
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms

        self._lock = threading.Lock()
        self._queues: Dict[BatchGroupKey, List[BatchItem]] = defaultdict(list)
        self._has_items = threading.Event()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._thread.start()
        logger.info("InferenceBatcher started (max_batch_size=%d, max_wait_ms=%.1f)",
                     self._max_batch_size, self._max_wait_ms)

    def shutdown(self) -> None:
        self._shutdown.set()
        self._has_items.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        with self._lock:
            for queue in self._queues.values():
                for item in queue:
                    if not item.future.done():
                        self._loop.call_soon_threadsafe(
                            item.future.set_exception,
                            RuntimeError("InferenceBatcher is shutting down"),
                        )
            self._queues.clear()
        logger.info("InferenceBatcher stopped")

    def enqueue(self, item: BatchItem) -> None:
        with self._lock:
            self._queues[item.group_key].append(item)
        self._has_items.set()

    def _dispatch_loop(self) -> None:
        while not self._shutdown.is_set():
            self._has_items.wait(timeout=0.001)
            self._has_items.clear()

            batches_to_run: List[Tuple[List[BatchItem], BatchGroupKey]] = []
            now = time.monotonic()

            with self._lock:
                for key, queue in list(self._queues.items()):
                    if not queue:
                        continue
                    oldest_age_ms = (now - queue[0].enqueue_time) * 1000
                    if len(queue) >= self._max_batch_size or oldest_age_ms >= self._max_wait_ms:
                        batch = queue[: self._max_batch_size]
                        self._queues[key] = queue[self._max_batch_size :]
                        batches_to_run.append((batch, key))
                    else:
                        self._has_items.set()

            for batch, key in batches_to_run:
                self._run_batch(batch, key)

    @torch.no_grad()
    def _run_batch(self, items: List[BatchItem], group_key: BatchGroupKey) -> None:
        action_norm_key, steps, _ = group_key
        try:
            device, dtype = resolve_policy_device_dtype(self._model)

            def to_model(t: torch.Tensor) -> torch.Tensor:
                return t.to(device=device, dtype=dtype) if t.is_floating_point() else t.to(device=device)

            tensor_keys = list(items[0].inputs.keys())
            batched: Dict[str, Any] = {}
            for k in tensor_keys:
                batched[k] = to_model(torch.cat([item.inputs[k] for item in items], dim=0))

            if action_norm_key is not None:
                batched["action_norm_key"] = action_norm_key

            self._model.eval()
            actions = self._model.generate_actions(**batched, steps=steps)
            actions_cpu = actions.float().cpu()

            for i, item in enumerate(items):
                result = actions_cpu[i].numpy()
                self._loop.call_soon_threadsafe(item.future.set_result, result)

            logger.debug("Batch dispatched: group=%s, size=%d", group_key, len(items))

        except Exception as exc:
            logger.error("Batch inference failed for group=%s: %s", group_key, exc)
            for item in items:
                if not item.future.done():
                    self._loop.call_soon_threadsafe(item.future.set_exception, exc)
