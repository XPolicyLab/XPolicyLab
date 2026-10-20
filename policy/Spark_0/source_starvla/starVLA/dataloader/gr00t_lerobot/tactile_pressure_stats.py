# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Local Spark-0 addition (NOT upstream starVLA).
"""q99 of packed tactile |pressure| (region all channels; taxel field 3).

Action-state statistics skip every tactile column so a heatmap cannot
explode RAM. Pressure scale still needs a quantile of the force channel,
so this module reads ONLY the region / taxel parquet columns and pools
``|p|`` into one scalar q99 per stream (NaN |p| — EgoPro unplaced pads —
is dropped before either reducer; the streaming builder would otherwise
treat any NaN as a poisoned action column and return NaN q99).
``scale = log1p(q99)`` when log1p is on (same number the old egopro
constants were).
"""

from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from starVLA.dataloader.gr00t_lerobot.streaming_stats import (
    _CollectBudget,
    _StreamingStatsBuilder,
)

# Same batch size as the raw parquet streaming path.
_STREAMING_PARQUET_BATCH_ROWS = 1 << 16

_REGION_KEY = "region_pressure"
_TAXEL_KEY = "taxel_pressure_3d"

# Read-ahead for the per-file parquet scan. The pressure columns are tiny
# next to the heatmap that shares the file, so a cold episode read is almost
# pure open + seek latency: on HumanTouch_518h a serial scan measures
# ~2.2 files/s with <2% of that in the accumulators, i.e. ~9 h per pass over
# 57k files, against ~150 files/s with a thread pool. Reads overlap but the
# builders are still fed in path order, so every statistic stays
# bit-identical to the serial scan.
_READ_WORKERS_ENV = "SPARK_0_TACTILE_STATS_READ_WORKERS"
_DEFAULT_READ_WORKERS = 32
# One Tianji episode file holds 145k rows x 320 taxels, so the window has to
# be bounded by decoded bytes rather than by file count.
_PREFETCH_BUDGET_BYTES = 2 << 30


def tactile_pressure_stream_plan(tactile_spec: dict) -> dict[str, dict]:
    """``{batch_key: {kind, columns}}`` for streams that carry a force channel."""
    plan: dict[str, dict] = {}
    for batch_key, spec in (tactile_spec or {}).items():
        if not isinstance(spec, dict) or spec.get("kind") == "heatmap":
            continue
        columns = [str(c) for c in (spec.get("columns") or []) if c]
        if not columns:
            continue
        if batch_key == _REGION_KEY:
            plan[batch_key] = {"kind": "region", "columns": columns}
        elif batch_key == _TAXEL_KEY:
            plan[batch_key] = {"kind": "taxel", "columns": columns}
    return plan


def _flatten_arrow_column(column) -> np.ndarray:
    """``(N, D)`` float32 from a scalar / fixed-size / nested-list column."""
    import pyarrow as pa

    n_rows = len(column)
    if n_rows == 0:
        return np.zeros((0, 1), dtype=np.float32)
    if column.null_count:
        raise ValueError("null values in a tactile pressure column")
    column_type = column.type
    if pa.types.is_fixed_size_list(column_type):
        width = int(column_type.list_size)
        flat = column.flatten().to_numpy(zero_copy_only=False)
        return np.asarray(flat, dtype=np.float32).reshape(n_rows, width)
    if (
        pa.types.is_floating(column_type)
        or pa.types.is_integer(column_type)
        or pa.types.is_boolean(column_type)
    ):
        values = column.to_numpy(zero_copy_only=False)
        return np.asarray(values, dtype=np.float32).reshape(n_rows, 1)
    flat = column
    while (
        pa.types.is_list(flat.type)
        or pa.types.is_large_list(flat.type)
        or pa.types.is_fixed_size_list(flat.type)
    ):
        flat = flat.flatten()
    values = np.asarray(flat.to_numpy(zero_copy_only=False), dtype=np.float32)
    if values.size % n_rows != 0:
        raise ValueError(
            f"tactile column flattened to {values.size} values for {n_rows} "
            "rows; expected a rectangular layout"
        )
    return values.reshape(n_rows, -1)


def pressure_abs_from_matrix(matrix: np.ndarray, kind: str) -> np.ndarray:
    """1-d ``|p|`` from one column's ``(N, D)`` matrix."""
    x = np.asarray(matrix, dtype=np.float32)
    if x.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if kind == "taxel":
        if x.shape[-1] % 4 != 0:
            raise ValueError(
                f"taxel pressure column last dim must be a multiple of 4 "
                f"(xyz + force), got shape {tuple(x.shape)}"
            )
        force = x.reshape(x.shape[0], -1, 4)[..., 3]
        return np.abs(force).reshape(-1)
    return np.abs(x).reshape(-1)


def _iter_pressure_batches(
    path: Path, plan: dict[str, dict], batch_size: int
):
    """Yield ``{batch_key: 1-d |p| float32}`` per parquet batch."""
    import pyarrow.parquet as pq

    columns = sorted(
        {column for spec in plan.values() for column in spec["columns"]}
    )
    if not columns:
        return
    parquet_file = pq.ParquetFile(path)
    names = set(parquet_file.schema_arrow.names)
    missing = [column for column in columns if column not in names]
    if missing:
        raise KeyError(
            f"{path} is missing tactile pressure columns {missing}"
        )
    kind_by_column = {
        column: spec["kind"]
        for spec in plan.values()
        for column in spec["columns"]
    }
    keys_by_column: dict[str, list[str]] = {}
    for batch_key, spec in plan.items():
        for column in spec["columns"]:
            keys_by_column.setdefault(column, []).append(batch_key)

    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        out: dict[str, list[np.ndarray]] = {key: [] for key in plan}
        for column in columns:
            matrix = _flatten_arrow_column(batch.column(column))
            values = pressure_abs_from_matrix(matrix, kind_by_column[column])
            # Unplaced EgoPro pads are NaN. The streaming builder treats
            # acc.nan > 0 as a poisoned ACTION column and returns NaN q99;
            # drop them here so both modes see only finite |p|.
            values = values[~np.isnan(values)]
            for batch_key in keys_by_column[column]:
                out[batch_key].append(values)
        yield {
            key: (
                np.concatenate(parts)
                if len(parts) > 1
                else (parts[0] if parts else np.zeros((0,), dtype=np.float32))
            )
            for key, parts in out.items()
        }


def _read_workers() -> int:
    raw = os.environ.get(_READ_WORKERS_ENV)
    if raw is None or not str(raw).strip():
        return _DEFAULT_READ_WORKERS
    workers = int(raw)
    if workers < 1:
        raise ValueError(
            f"{_READ_WORKERS_ENV} must be >= 1 (1 disables the read-ahead), "
            f"got {workers}"
        )
    return workers


def _read_file_pressure(path: Path, plan: dict[str, dict], batch_size: int) -> list:
    """One file's batches, materialised so a worker thread can hand them over."""
    return list(_iter_pressure_batches(path, plan, batch_size))


def _prefetch_window(first_file_batches: list, workers: int) -> int:
    """In-flight file count that keeps the decoded read-ahead under budget."""
    nbytes = sum(
        int(values.nbytes) for batch in first_file_batches for values in batch.values()
    )
    if nbytes <= 0:
        return workers
    return max(1, min(workers, _PREFETCH_BUDGET_BYTES // nbytes))


def _iter_paths_pressure_batches(
    paths: list[Path], plan: dict[str, dict], batch_size: int, desc: str
):
    """Yield every file's batches in ``paths`` order, reading ahead in threads.

    The consumer sees the exact sequence the serial scan produced — only the
    parquet reads overlap — so histograms, min/max, quantiles and the float64
    moment merge tree are all unchanged.
    """
    workers = _read_workers()
    if workers <= 1 or len(paths) <= 1:
        for path in tqdm(paths, desc=desc):
            yield from _iter_pressure_batches(path, plan, batch_size)
        return

    with ThreadPoolExecutor(max_workers=workers) as pool, tqdm(
        total=len(paths), desc=desc
    ) as progress:
        batches = _read_file_pressure(paths[0], plan, batch_size)
        window = _prefetch_window(batches, workers)
        pending: deque = deque()
        next_index = 1
        while next_index < len(paths) and len(pending) < window:
            pending.append(
                pool.submit(_read_file_pressure, paths[next_index], plan, batch_size)
            )
            next_index += 1
        while True:
            yield from batches
            progress.update(1)
            if not pending:
                return
            future = pending.popleft()
            if next_index < len(paths):
                pending.append(
                    pool.submit(
                        _read_file_pressure, paths[next_index], plan, batch_size
                    )
                )
                next_index += 1
            batches = future.result()


def _scalar_stats_from_values(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    # EgoPro unplaced pads are NaN (xyz + pressure). Streaming already drops
    # NaN before the histogram; memory must do the same or ``np.quantile``
    # returns NaN and ``scale_from_pressure_q99`` raises. The 1 h EgoPro set
    # is under the 8 GiB auto threshold, so it uses this path.
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "q01": float("nan"),
            "q99": float("nan"),
            "n": 0,
        }
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "q01": float(np.quantile(values, 0.01)),
        "q99": float(np.quantile(values, 0.99)),
        "n": int(values.size),
    }


def _unwrap_builder_result(result: dict, n: int) -> dict:
    """``_StreamingStatsBuilder`` stores one-column lists; unwrap to scalars."""

    def _one(name: str) -> float:
        value = result[name]
        if isinstance(value, (list, tuple)):
            value = value[0]
        return float(value)

    return {
        "mean": _one("mean"),
        "std": _one("std"),
        "min": _one("min"),
        "max": _one("max"),
        "q01": _one("q01"),
        "q99": _one("q99"),
        "n": int(n),
    }


def calculate_tactile_pressure_statistics(
    parquet_paths: list[Path],
    plan: dict[str, dict],
    *,
    mode: str = "memory",
) -> dict:
    """Per-stream ``|p|`` statistics. Heatmaps are never in ``plan``.

    ``mode="streaming"`` is bit-identical to memory on min/max/quantiles
    (same builder as converted-pose stats). ``n`` is the pooled scalar count.
    """
    if not plan:
        return {}
    paths = sorted(list(parquet_paths))
    if not paths:
        raise FileNotFoundError(
            "No parquet files for tactile pressure statistics."
        )
    if mode == "streaming":
        return _calculate_tactile_pressure_statistics_streaming(paths, plan)
    pooled: dict[str, list[np.ndarray]] = {key: [] for key in plan}
    for batch in _iter_paths_pressure_batches(
        paths, plan, _STREAMING_PARQUET_BATCH_ROWS, "Tactile pressure statistics"
    ):
        for key, values in batch.items():
            if values.size:
                pooled[key].append(values)
    return {
        key: _scalar_stats_from_values(
            np.concatenate(parts) if parts else np.zeros((0,), dtype=np.float32)
        )
        for key, parts in pooled.items()
    }


def _calculate_tactile_pressure_statistics_streaming(
    paths: list[Path], plan: dict[str, dict]
) -> dict:
    budget = _CollectBudget()
    builders = {key: _StreamingStatsBuilder(budget=budget) for key in plan}
    counts = {key: 0 for key in plan}
    for batch in _iter_paths_pressure_batches(
        paths,
        plan,
        _STREAMING_PARQUET_BATCH_ROWS,
        "Tactile pressure statistics (pass 1/2: histograms)",
    ):
        for key, values in batch.items():
            counts[key] += int(values.size)
            builders[key].update(values.reshape(-1, 1))
    for builder in builders.values():
        builder.finalize_pass1()
    for batch in _iter_paths_pressure_batches(
        paths,
        plan,
        _STREAMING_PARQUET_BATCH_ROWS,
        "Tactile pressure statistics (pass 2/2: quantiles)",
    ):
        for key, values in batch.items():
            builders[key].collect_update(values.reshape(-1, 1))
    return {
        key: _unwrap_builder_result(builders[key].result(), counts[key])
        for key in plan
    }
