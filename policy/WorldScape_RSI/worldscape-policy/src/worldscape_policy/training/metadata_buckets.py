"""Ordered RoboDojo batches planned before decoding and augmentation."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any


from worldscape_policy.data.mixture import NativeShardedMixtureDataset


class _ShardCache:
    """LRU of materialized shards, plus at most one asynchronous prefetch."""

    def __init__(self, mixture: NativeShardedMixtureDataset, capacity: int):
        self.mixture = mixture
        self.capacity = capacity
        self.values: OrderedDict[int, Any] = OrderedDict()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future: tuple[int, Future] | None = None

    def _store(self, key: int, value: Any) -> Any:
        self.values[key] = value
        self.values.move_to_end(key)
        while len(self.values) > self.capacity:
            self.values.popitem(last=False)
        return value

    def get(self, key: int) -> Any:
        if key in self.values:
            self.values.move_to_end(key)
            return self.values[key]
        if self.future is not None and self.future[0] == key:
            _, future = self.future
            self.future = None
            return self._store(key, future.result())
        return self._store(key, self.mixture._load_shard(self.mixture.shards[key]))

    def prefetch(self, key: int) -> None:
        if key in self.values or (self.future is not None and self.future[0] == key):
            return
        if self.future is not None:
            previous, future = self.future
            if not future.done():
                return
            self.future = None
            self._store(previous, future.result())
        self.future = (key, self.executor.submit(self.mixture._load_shard, self.mixture.shards[key]))

    def close(self) -> None:
        if self.future is not None:
            self.future[1].cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.future = None
        self.values.clear()
