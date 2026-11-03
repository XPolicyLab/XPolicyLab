"""Bounded LRU of sparse episode payloads, with one speculative shard read."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from worldscape_policy.data.robodojo_selected_frames import SelectedPayload, load_selected_frames
from worldscape_policy.training.timing_trace import TraceSpan

if TYPE_CHECKING:
    from worldscape_policy.data.mixture import NativeShardedMixtureDataset


class SelectedFrameShardCache:
    """Only the foreground owns the LRU; readers return immutable arrays."""

    def __init__(self, mixture: NativeShardedMixtureDataset, capacity: int) -> None:
        if capacity < 1:
            raise ValueError('selected frame cache capacity must be positive')
        self.mixture = mixture
        self.capacity = capacity
        self.values: OrderedDict[int, SelectedPayload] = OrderedDict()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future: tuple[int, Future[SelectedPayload]] | None = None

    def _indices(self, shard_id, indices):
        shard = self.mixture.shards[shard_id]
        indices = tuple(indices)
        if not indices or any(not shard.start <= index < shard.stop for index in indices):
            raise ValueError('selected samples must be nonempty and inside their shard')
        return indices

    def _read(self, shard_id, indices, cached):
        shard = self.mixture.shards[shard_id]
        with TraceSpan('selected_frames', shard=shard_id, indices=list(indices),
                       cached_episodes=len(cached)):
            return load_selected_frames(self.mixture.datasets[shard.dataset_index], indices, cached=cached)

    def _store(self, shard_id, payload):
        cached = self.values.setdefault(shard_id, {})
        cached.update(payload)
        self.values.move_to_end(shard_id)
        while len(self.values) > self.capacity:
            self.values.popitem(last=False)
        return cached

    def _harvest(self):
        shard_id, future = self.future
        self.future = None
        with TraceSpan('prefetch_wait', shard=shard_id, ready=future.done()):
            self._store(shard_id, future.result())

    def get(self, shard_id: int, indices: Iterable[int]) -> SelectedPayload:
        indices = self._indices(shard_id, indices)
        if self.future is not None and self.future[0] == shard_id:
            self._harvest()
        # A shallow snapshot is safe: loading adds coverage by copying arrays,
        # never by mutating frames still used by a materialization thread.
        cached = dict(self.values.get(shard_id, {}))
        return self._store(shard_id, self._read(shard_id, indices, cached))

    def prefetch(self, shard_id: int, indices: Iterable[int]) -> None:
        indices = self._indices(shard_id, indices)
        if self.future is not None:
            if not self.future[1].done():
                return
            self._harvest()
        cached = dict(self.values.get(shard_id, {}))
        self.future = (shard_id, self.executor.submit(self._read, shard_id, indices, cached))

    def close(self) -> None:
        if self.future is not None:
            self.future[1].cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.future = None
        self.values.clear()
