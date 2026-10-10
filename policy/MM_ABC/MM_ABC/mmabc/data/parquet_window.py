"""Bounded, windowed reads of fixed-size-list columns from parquet shards."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq

import os

from mmabc.data.pagecache import drop_page_cache

# Bound row-group cache memory per worker; total memory scales with ranks and workers.
DEFAULT_BUDGET_BYTES = int(os.environ.get("MMABC_PARQUET_CACHE_MB", "96")) * 1024 * 1024

# Bound parquet handles as well as row groups, since each handle retains footer metadata.
DEFAULT_MAX_HANDLES = int(os.environ.get("MMABC_PARQUET_HANDLES", "16"))


def fixed_list_to_numpy(column) -> np.ndarray:
    """(n,) fixed-size-list column -> (n, width) numpy, without Python objects."""
    col = column.combine_chunks()
    n = len(col)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    flat = col.flatten().to_numpy(zero_copy_only=False)
    return flat.reshape(n, -1)


@dataclass
class _Handle:
    file: pq.ParquetFile
    group_starts: np.ndarray  # (num_groups + 1,) cumulative row offsets


class ParquetWindowReader:
    """Process-local cache of shard handles and decoded row groups."""

    def __init__(
        self,
        columns: tuple[str, ...],
        budget_bytes: int = DEFAULT_BUDGET_BYTES,
        max_handles: int = DEFAULT_MAX_HANDLES,
    ) -> None:
        self.columns = columns
        self.budget = budget_bytes
        self.max_handles = max_handles
        self._handles: OrderedDict[str, _Handle] = OrderedDict()
        self._groups: OrderedDict[tuple[str, int], tuple[np.ndarray, ...]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def _handle(self, path: str) -> _Handle:
        with self._lock:
            h = self._handles.get(path)
            if h is not None:
                self._handles.move_to_end(path)
                return h

        # Open outside the lock because remote footer reads may block.
        pf = pq.ParquetFile(path, memory_map=False, pre_buffer=False)
        counts = [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)]
        starts = np.zeros(len(counts) + 1, dtype=np.int64)
        starts[1:] = np.cumsum(counts)
        h = _Handle(file=pf, group_starts=starts)

        evicted: list[_Handle] = []
        with self._lock:
            existing = self._handles.get(path)
            if existing is not None:
                # Another caller opened the same shard while we were reading its
                # footer; keep theirs and drop ours to avoid two open handles.
                self._handles.move_to_end(path)
                evicted.append(h)
                h = existing
            else:
                self._handles[path] = h
                self._handles.move_to_end(path)
                while len(self._handles) > self.max_handles:
                    _, old = self._handles.popitem(last=False)
                    evicted.append(old)
        for old in evicted:
            try:
                old.file.close()
            except Exception:
                pass
        return h

    def _group(self, path: str, gidx: int) -> tuple[np.ndarray, ...]:
        key = (path, gidx)
        with self._lock:
            hit = self._groups.pop(key, None)
            if hit is not None:
                self._groups[key] = hit
                return hit

        h = self._handle(path)
        tbl = h.file.read_row_group(gidx, columns=list(self.columns))
        arrays = tuple(
            np.ascontiguousarray(fixed_list_to_numpy(tbl.column(c)), dtype=np.float32)
            for c in self.columns
        )
        nbytes = sum(a.nbytes for a in arrays)
        # Decoded rows are in the bounded cache; release clean file-backed pages when enabled.
        drop_page_cache(path)

        with self._lock:
            self._groups[key] = arrays
            self._bytes += nbytes
            while self._bytes > self.budget and len(self._groups) > 1:
                _, old = self._groups.popitem(last=False)
                self._bytes -= sum(a.nbytes for a in old)
        return arrays

    def read(self, path: str, row_start: int, row_stop: int) -> tuple[np.ndarray, ...]:
        """Rows [row_start, row_stop) of each configured column."""
        h = self._handle(path)
        total = int(h.group_starts[-1])
        row_start = max(0, min(int(row_start), total))
        row_stop = max(row_start, min(int(row_stop), total))
        if row_stop == row_start:
            return tuple(np.zeros((0, 0), dtype=np.float32) for _ in self.columns)

        first = int(np.searchsorted(h.group_starts, row_start, side="right") - 1)
        last = int(np.searchsorted(h.group_starts, row_stop - 1, side="right") - 1)

        chunks: list[list[np.ndarray]] = [[] for _ in self.columns]
        for g in range(first, last + 1):
            g_start = int(h.group_starts[g])
            g_stop = int(h.group_starts[g + 1])
            arrays = self._group(path, g)
            lo = max(row_start, g_start) - g_start
            hi = min(row_stop, g_stop) - g_start
            for i, arr in enumerate(arrays):
                chunks[i].append(arr[lo:hi])
        return tuple(
            c[0] if len(c) == 1 else np.concatenate(c, axis=0) for c in chunks
        )

    def num_rows(self, path: str) -> int:
        return int(self._handle(path).group_starts[-1])


_READERS: dict[tuple[str, ...], ParquetWindowReader] = {}


def reader(columns: tuple[str, ...]) -> ParquetWindowReader:
    """Process-local reader. Dataloader workers each build their own."""
    r = _READERS.get(columns)
    if r is None:
        r = ParquetWindowReader(columns)
        _READERS[columns] = r
    return r
