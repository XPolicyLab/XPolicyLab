"""CPU embedding retention for a fixed encoder, independent of serving topology."""

import hashlib
import json
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CacheStats:
    entries: int
    bytes: int
    hits: int
    misses: int
    evictions: int
    inflight: int
    shared_entries: int


class EmbeddingCache:
    """Thread-safe byte-bounded CPU LRU cache with per-key singleflight."""

    def __init__(self, max_bytes: int = 4 * 1024**3):
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.max_bytes = max_bytes
        self._values: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._sizes: dict[str, int] = {}
        self._inflight: dict[str, Future] = {}
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = threading.RLock()

    @staticmethod
    def _normalize(value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"embedding cache values must be tensors, got {type(value).__name__}")
        value = value.detach().to("cpu").contiguous()
        if value.is_shared():
            value = value.clone()
        return value

    @staticmethod
    def _nbytes(value: torch.Tensor) -> int:
        return value.numel() * value.element_size()

    def get(self, key: str) -> torch.Tensor | None:
        with self._lock:
            value = self._values.get(key)
            if value is None:
                self._misses += 1
                return None
            self._values.move_to_end(key)
            self._hits += 1
            return value.clone()

    def put(self, key: str, value: torch.Tensor) -> None:
        # The cache owns its storage.  Callers may later pass their tensor to a
        # multiprocessing Queue, which converts that storage to shared memory
        # in place under PyTorch's default IPC reducer.
        value = self._normalize(value).clone()
        size = self._nbytes(value)
        with self._lock:
            if key in self._values:
                self._bytes -= self._sizes.pop(key)
                del self._values[key]
            if self.max_bytes == 0 or size > self.max_bytes:
                return
            self._values[key] = value
            self._sizes[key] = size
            self._bytes += size
            while self._bytes > self.max_bytes:
                old_key, _ = self._values.popitem(last=False)
                self._bytes -= self._sizes.pop(old_key)
                self._evictions += 1

    def get_or_compute(self, key: str, compute: Callable[[], torch.Tensor]) -> torch.Tensor:
        with self._lock:
            value = self._values.get(key)
            if value is not None:
                self._values.move_to_end(key)
                self._hits += 1
                return value.clone()
            self._misses += 1
            future = self._inflight.get(key)
            if future is None:
                future = Future()
                self._inflight[key] = future
                owner = True
            else:
                owner = False

        if not owner:
            return future.result().clone()

        try:
            value = self._normalize(compute())
            self.put(key, value)
            future.set_result(value)
            return value.clone()
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._inflight.pop(key, None)

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                entries=len(self._values),
                bytes=self._bytes,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                inflight=len(self._inflight),
                shared_entries=sum(value.is_shared() for value in self._values.values()),
            )


def text_embedding_key(prompt: str, *, max_length: int, dtype: str) -> str:
    """Identify exact text and encoding options within one fixed encoder.

    Encoder identity belongs to the cache owner: separate model instances or
    deployments must not share a cache. ``native`` denotes the direct encoder's
    fixed output dtype; no tensor is cast when storing or retrieving it.
    """
    value = json.dumps((prompt, max_length, dtype), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
