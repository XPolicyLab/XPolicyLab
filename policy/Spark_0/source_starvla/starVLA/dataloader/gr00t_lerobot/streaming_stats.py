"""Two-pass streaming statistics (~1 GB RAM regardless of dataset size).

"memory" mode pools every value in RAM and reduces with numpy; its footprint
grows linearly with the dataset (open_aoe-class datasets peak at tens of GB).
"streaming" mode makes two passes over the same blocks: pass 1 builds a
65536-bucket histogram per column over the float32 total order plus
Welford/NaN accumulators; pass 2 collects only the values inside each target
rank's bucket and selects exactly. min/max and every quantile are
bit-identical to the memory path (the selection is exact and the
interpolation mirrors numpy's float32-diff lerp); mean/std are
float64-accurate (error ~1e-15), while the memory path's float32 pairwise
summation has a systematic error that GROWS with the row count (~3e-5
relative at 240k rows, ~3.5e-4 at 1.28M — measured).

Both the raw parquet statistics (``gr00t_lerobot/datasets.py``) and the
converted-pose statistics (``XPolicyLab`` ``ee_delta_dataset.py``) drive this
module; the mode is recorded in their cache configs only when streaming, so
memory-mode caches keep their existing hashes.
"""

from __future__ import annotations

import os

import numpy as np

_STATS_MODE_ENV = "SPARK_0_STATS_MODE"
_STATS_STREAMING_THRESHOLD_ENV = "SPARK_0_STATS_STREAMING_THRESHOLD_BYTES"
# Auto-switch point: datasets whose parquet bytes exceed this stream.
_DEFAULT_STATS_STREAMING_THRESHOLD = 8 << 30  # 8 GiB

_FLOAT32_BUCKET_SHIFT = 16
_FLOAT32_NUM_BUCKETS = 1 << (32 - _FLOAT32_BUCKET_SHIFT)  # 65536 high-bit buckets
_FLOAT32_LOW_MASK = np.uint32((1 << _FLOAT32_BUCKET_SHIFT) - 1)
_FLOAT32_SIGN_BIT = np.uint32(0x80000000)
# Buckets holding more values than this are tallied by a 65536-entry count
# array (bounded memory) instead of collecting the raw values.
_STREAMING_COLLECT_CAP = 1 << 24
# Budget for collected candidate VALUES across all (column, bucket) targets:
# the per-bucket cap is tightened to remaining_budget / #targets so pass 2
# can never exceed ~1 GiB of raw candidates even for adversarially
# distributed columns (discrete/quantized data with many dense buckets).
_STREAMING_COLLECT_BUDGET = 1 << 28
# Never tighten below this: under it the 512 KiB tally is cheaper anyway.
_STREAMING_COLLECT_FLOOR = 1 << 16
# Rows are consumed in slices of about this many VALUES. Each pass-1 slice
# allocates a handful of same-shaped temporaries (uint32 order keys, the
# float64 upcast for Welford), so an unsliced 1M-row x 54-column parquet
# block peaked at ~2 GB and defeated the point of streaming. 4Mi values
# caps those temporaries at tens of MB while staying far above the size
# where the per-column Python loop starts to dominate.
_STREAMING_CHUNK_VALUES = 1 << 22


def _row_chunk_rows(block: np.ndarray) -> int:
    """Rows per slice for ``block``, from its per-row value count."""
    per_row = 1
    for size in block.shape[1:]:
        per_row *= int(size)
    return max(1, _STREAMING_CHUNK_VALUES // max(1, per_row))


def _resolve_stats_mode(data_cfg, total_data_bytes: int) -> str:
    """``memory`` | ``streaming``: env > config > auto-by-size (deterministic).

    Auto derives from the parquet sizes already in the data fingerprint, so
    the offline precompute and the training launch always pick the same mode
    (and therefore the same cache hash) for the same data. An EMPTY env value
    counts as unset, so exporting the variable without a value from a launch
    script cannot shadow ``datasets.vla_data.stats_mode``.
    """
    raw = os.environ.get(_STATS_MODE_ENV)
    if raw is not None and not str(raw).strip():
        raw = None
    if raw is None and data_cfg is not None:
        raw = data_cfg.get("stats_mode")
    mode = str(raw).strip().lower() if raw is not None else "auto"
    if mode not in ("auto", "memory", "streaming"):
        raise ValueError(
            f"stats mode {mode!r}; expected 'auto', 'memory' or 'streaming' "
            f"({_STATS_MODE_ENV} or datasets.vla_data.stats_mode)."
        )
    if mode != "auto":
        return mode
    threshold = int(
        os.environ.get(
            _STATS_STREAMING_THRESHOLD_ENV, _DEFAULT_STATS_STREAMING_THRESHOLD
        )
    )
    return "streaming" if total_data_bytes >= threshold else "memory"


def _f32_ordered_key(values: np.ndarray) -> np.ndarray:
    """float32 -> uint32 with the same total order (sign-magnitude flip).

    NaN is NOT orderable and must be filtered by the caller; -0.0 and +0.0
    map to adjacent keys (-0.0 sorts first), matching IEEE comparison order.
    """
    u = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return np.where(u & _FLOAT32_SIGN_BIT, ~u, u | _FLOAT32_SIGN_BIT)


def _f32_from_ordered_key(keys) -> np.ndarray:
    """Inverse of :func:`_f32_ordered_key`."""
    k = np.asarray(keys, dtype=np.uint32)
    u = np.where(k & _FLOAT32_SIGN_BIT, k & ~_FLOAT32_SIGN_BIT, ~k)
    return u.view(np.float32)


def _lerp_float32_pair(a32, b32, t: float) -> float:
    """Bit-exact mirror of ``numpy.lib.function_base._lerp`` on float32 inputs.

    numpy computes ``diff = float32(b - a)`` FIRST (rounded to float32) and
    only then interpolates in float64, using ``a + diff*t`` for ``t < 0.5``
    and ``b - diff*(1-t)`` otherwise. Replicating that rounding is what makes
    the streaming quantiles bit-identical to ``np.quantile``.
    """
    a32 = np.float32(a32)
    b32 = np.float32(b32)
    if a32 == b32:
        return float(np.float64(a32))
    diff = np.float32(b32 - a32)
    if t < 0.5:
        return float(np.float64(a32) + np.float64(diff) * t)
    return float(np.float64(b32) - np.float64(diff) * (1.0 - t))


class _CollectBudget:
    """Shared allowance for pass-2 candidate values, in VALUES not bytes.

    One statistics run builds several independent builders (one per pose key
    per modality) and every builder collects for two scopes (pooled and
    per-row), so a per-builder budget bounds nothing at the process level:
    with two arms it allowed ~6 GiB. Handing every builder the SAME budget
    object makes the bound global — each :meth:`reserve` takes its share out
    of what is left, and once the budget is exhausted every remaining target
    falls back to the floor (where the 512 KiB tally is cheaper than a list
    anyway), so exactness never depends on the reservation order.
    """

    def __init__(
        self,
        total: int = _STREAMING_COLLECT_BUDGET,
        cap: int = _STREAMING_COLLECT_CAP,
    ) -> None:
        self._remaining = int(total)
        self._cap = int(cap)

    @property
    def cap(self) -> int:
        return self._cap

    def reserve(self, n_targets: int) -> int:
        """Per-bucket cap for ``n_targets`` (column, bucket) targets."""
        if n_targets <= 0:
            return self._cap
        cap = max(
            _STREAMING_COLLECT_FLOOR,
            min(self._cap, self._remaining // n_targets),
        )
        self._remaining = max(0, self._remaining - cap * n_targets)
        return cap


class _StreamingColumnStats:
    """Pass-1 accumulator for one column: count, NaN count, exact min/max,
    Welford mean/M2 (float64), and the 65536-bucket order histogram."""

    __slots__ = ("n", "nan", "minimum", "maximum", "mean", "m2", "hist")

    def __init__(self) -> None:
        self.n = 0
        self.nan = 0
        self.minimum = np.float32(np.inf)
        self.maximum = np.float32(-np.inf)
        self.mean = 0.0
        self.m2 = 0.0
        self.hist = np.zeros(_FLOAT32_NUM_BUCKETS, dtype=np.uint64)

    def update(self, values: np.ndarray) -> None:
        """Per-column update; cleans NaNs first (rare path)."""
        x = np.asarray(values, dtype=np.float32)
        nan_mask = np.isnan(x)
        if nan_mask.any():
            self.nan += int(nan_mask.sum())
            x = x[~nan_mask]
        if x.size == 0:
            return
        counts = np.bincount(_f32_ordered_key(x) >> _FLOAT32_BUCKET_SHIFT)
        self.hist[: counts.size] += counts.astype(np.uint64)
        cmin = x.min()
        cmax = x.max()
        # Strict comparisons keep the FIRST value on ±0 ties, the same tie
        # behaviour as a left-to-right minimum/maximum reduction.
        if cmin < self.minimum:
            self.minimum = cmin
        if cmax > self.maximum:
            self.maximum = cmax
        xf = x.astype(np.float64)
        mean2 = float(xf.mean())
        m2_2 = float(np.square(xf - mean2).sum())
        self._merge_moments(x.size, mean2, m2_2)

    def _merge_moments(self, n2: int, mean2: float, m2_2: float) -> None:
        """Chan parallel-variance merge (associative in float64)."""
        n = self.n + n2
        delta = mean2 - self.mean
        self.mean += delta * (n2 / n)
        self.m2 += m2_2 + delta * delta * (self.n * n2 / n)
        self.n = n


class _QuantileRankTarget:
    """Pass-2 target for one (column, quantile): bucket + rank inside it."""

    __slots__ = ("lo_bucket", "lo_rank", "hi_bucket", "hi_rank", "t")

    def __init__(self, lo_bucket, lo_rank, hi_bucket, hi_rank, t) -> None:
        self.lo_bucket = lo_bucket
        self.lo_rank = lo_rank
        self.hi_bucket = hi_bucket
        self.hi_rank = hi_rank
        self.t = t


class _QuantileCandidateCollector:
    """Pass-2 storage for one column set: per target bucket either a value
    list (small buckets) or a 65536-entry tally (concentrated buckets, so a
    near-constant column can never blow up memory)."""

    def __init__(
        self,
        targets: dict[int, set[int]],
        bucket_counts: dict[int, dict[int, int]],
        cap: int,
    ) -> None:
        self._lists: dict[tuple[int, int], list[np.ndarray]] = {}
        self._tallies: dict[tuple[int, int], np.ndarray] = {}
        # Precomputed per-column targets: pass 2 calls update_column once per
        # column per episode, so scanning the pair dicts there would cost
        # O(#pairs) Python work per call (days at 万小时 episode counts).
        self._by_col: dict[int, tuple[list[int], list[int]]] = {}
        for col, buckets in targets.items():
            for bucket in buckets:
                if bucket_counts[col][bucket] <= cap:
                    self._lists[(col, bucket)] = []
                else:
                    self._tallies[(col, bucket)] = np.zeros(
                        1 << _FLOAT32_BUCKET_SHIFT, dtype=np.uint64
                    )
            list_buckets = [b for b in buckets if (col, b) in self._lists]
            tally_buckets = [b for b in buckets if (col, b) in self._tallies]
            if list_buckets or tally_buckets:
                self._by_col[col] = (list_buckets, tally_buckets)

    def update_column(self, col: int, values: np.ndarray) -> None:
        target = self._by_col.get(col)
        if target is None:
            return
        list_buckets, tally_buckets = target
        values = np.asarray(values, dtype=np.float32)
        keys = _f32_ordered_key(values)
        buckets = keys >> _FLOAT32_BUCKET_SHIFT
        for bucket in list_buckets:
            mask = buckets == bucket
            if mask.any():
                self._lists[(col, bucket)].append(values[mask])
        for bucket in tally_buckets:
            mask = buckets == bucket
            if mask.any():
                low = (keys & _FLOAT32_LOW_MASK)[mask]
                np.add.at(self._tallies[(col, bucket)], low, 1)

    def select(self, col: int, bucket: int, within_bucket_rank: int):
        """The exact ``within_bucket_rank``-th value (0-indexed) in ``bucket``."""
        if (col, bucket) in self._lists:
            parts = self._lists[(col, bucket)]
            if not parts:
                raise RuntimeError(
                    f"no candidates collected for column {col} bucket {bucket}; "
                    "pass 2 must cover the same data as pass 1"
                )
            values = np.concatenate(parts) if len(parts) > 1 else parts[0]
            values = np.sort(values)
            return np.float32(values[within_bucket_rank])
        tally = self._tallies[(col, bucket)]
        cum = np.cumsum(tally)
        low = int(np.searchsorted(cum, within_bucket_rank + 1, side="left"))
        key = (np.uint32(bucket) << np.uint32(_FLOAT32_BUCKET_SHIFT)) | np.uint32(low)
        return np.float32(_f32_from_ordered_key(key))


class _StreamingStatsBuilder:
    """Two-pass exact statistics for one key's values.

    Feed pass 1 with :meth:`update`, call :meth:`finalize_pass1`, feed pass 2
    with :meth:`collect_update`, then read :meth:`result`. Blocks must be
    float32. Quantile/min/max results are bit-identical to reducing the
    pooled float32 array with numpy; mean/std are float64-accurate.

    ``budget`` shares one process-wide pass-2 allowance across every builder
    of a run (see :class:`_CollectBudget`); without it each builder would get
    its own ~1 GiB.
    """

    def __init__(
        self,
        quantiles=(0.01, 0.99),
        collect_cap=_STREAMING_COLLECT_CAP,
        collect_budget=_STREAMING_COLLECT_BUDGET,
        budget: _CollectBudget | None = None,
    ) -> None:
        self._quantiles = tuple(float(q) for q in quantiles)
        self._collect_cap = int(collect_cap)
        self._collect_budget = int(collect_budget)
        self._budget = budget if budget is not None else _CollectBudget(
            self._collect_budget, self._collect_cap
        )
        self._pooled: list[_StreamingColumnStats] | None = None
        self._per_row: list[_StreamingColumnStats] | None = None
        self._per_row_shape: tuple[int, int] | None = None  # (H, D)
        self._targets: dict[str, dict[int, dict[float, _QuantileRankTarget | None]]] = {}
        self._collectors: dict[str, _QuantileCandidateCollector] = {}
        self._collect_cols_cache: dict[str, list[int]] = {}

    # ---------------- pass 1 ----------------
    def update(self, block: np.ndarray) -> None:
        """Accumulate ``block`` (``(N, D)`` or ``(N, H, D)``), sliced by rows.

        Slicing only changes the float64 merge tree of mean/M2 (~1e-15) and
        leaves histograms, min/max and every quantile bit-identical.
        """
        block = np.asarray(block, dtype=np.float32)
        if block.shape[0] == 0:
            self._update_block(block)
            return
        step = _row_chunk_rows(block)
        for start in range(0, block.shape[0], step):
            self._update_block(block[start : start + step])

    def _update_block(self, block: np.ndarray) -> None:
        if block.ndim == 3:
            length, horizon, dim = block.shape
            if self._per_row is None:
                self._per_row_shape = (horizon, dim)
                self._per_row = [_StreamingColumnStats() for _ in range(horizon * dim)]
            self._update_columns(self._per_row, block.reshape(length, horizon * dim))
            block = block.reshape(length * horizon, dim)
        if self._pooled is None:
            self._pooled = [_StreamingColumnStats() for _ in range(block.shape[1])]
        self._update_columns(self._pooled, block)

    @staticmethod
    def _update_columns(accs: list[_StreamingColumnStats], values2d: np.ndarray) -> None:
        x = np.ascontiguousarray(values2d, dtype=np.float32)
        if np.isnan(x).any():
            # Rare path: let each column clean its own NaNs.
            for j, acc in enumerate(accs):
                acc.update(x[:, j])
            return
        n2 = x.shape[0]
        if n2 == 0:
            return
        buckets = _f32_ordered_key(x) >> _FLOAT32_BUCKET_SHIFT
        cmin = x.min(axis=0)
        cmax = x.max(axis=0)
        xf = x.astype(np.float64)
        mean2 = xf.mean(axis=0)
        m2_2 = np.square(xf - mean2).sum(axis=0)
        for j, acc in enumerate(accs):
            counts = np.bincount(buckets[:, j])
            acc.hist[: counts.size] += counts.astype(np.uint64)
            if cmin[j] < acc.minimum:
                acc.minimum = cmin[j]
            if cmax[j] > acc.maximum:
                acc.maximum = cmax[j]
            acc._merge_moments(n2, float(mean2[j]), float(m2_2[j]))

    # ---------------- between passes ----------------
    def finalize_pass1(self) -> None:
        self._targets = {}
        self._collectors = {}
        for scope, accs in (("pooled", self._pooled), ("per_row", self._per_row)):
            if accs is None:
                continue
            scope_targets: dict[int, dict[float, _QuantileRankTarget | None]] = {}
            bucket_sets: dict[int, set[int]] = {}
            bucket_counts: dict[int, dict[int, int]] = {}
            for col, acc in enumerate(accs):
                col_targets: dict[float, _QuantileRankTarget | None] = {}
                if acc.nan > 0 or acc.n == 0 or acc.minimum == acc.maximum:
                    # NaN column (numpy propagates NaN to every field),
                    # empty (defensive), or constant: no pass-2 work needed.
                    for q in self._quantiles:
                        col_targets[q] = None
                else:
                    cum = np.cumsum(acc.hist)
                    for q in self._quantiles:
                        target = self._rank_target(acc.n, q, cum)
                        col_targets[q] = target
                        for bucket in {target.lo_bucket, target.hi_bucket}:
                            bucket_sets.setdefault(col, set()).add(bucket)
                            bucket_counts.setdefault(col, {})[bucket] = int(
                                acc.hist[bucket]
                            )
                scope_targets[col] = col_targets
            self._targets[scope] = scope_targets
            self._collectors[scope] = _QuantileCandidateCollector(
                bucket_sets, bucket_counts, self._effective_cap(bucket_sets)
            )
        # Cached once: collect_update would otherwise rebuild these lists for
        # every episode block.
        self._collect_cols_cache = {
            scope: self._collect_columns(scope) for scope in self._targets
        }

    def _effective_cap(self, bucket_sets: dict[int, set[int]]) -> int:
        """Per-bucket collect cap under the shared candidate budget.

        Without it, ~1800 (column, bucket) targets each holding just under
        the per-bucket cap could collect ~120 GB of raw candidates on
        adversarially distributed columns. Reserving from the run-wide
        budget bounds every list of every builder to ~1 GiB in total;
        buckets above the cap fall back to the 512 KiB tally, so exactness
        is unaffected.
        """
        n_targets = sum(len(buckets) for buckets in bucket_sets.values())
        return self._budget.reserve(n_targets)

    @staticmethod
    def _rank_target(n: int, q: float, cum_hist: np.ndarray) -> _QuantileRankTarget:
        pos = q * (n - 1)
        lo = int(np.floor(pos))
        hi = min(lo + 1, n - 1)

        def locate(rank: int) -> tuple[int, int]:
            bucket = int(np.searchsorted(cum_hist, rank + 1, side="left"))
            before = int(cum_hist[bucket - 1]) if bucket > 0 else 0
            return bucket, rank - before

        lo_bucket, lo_rank = locate(lo)
        hi_bucket, hi_rank = locate(hi)
        return _QuantileRankTarget(lo_bucket, lo_rank, hi_bucket, hi_rank, pos - lo)

    # ---------------- pass 2 ----------------
    def collect_update(self, block: np.ndarray) -> None:
        block = np.asarray(block, dtype=np.float32)
        if block.shape[0] == 0:
            return
        step = _row_chunk_rows(block)
        for start in range(0, block.shape[0], step):
            self._collect_block(block[start : start + step])

    def _collect_block(self, block: np.ndarray) -> None:
        per_row_collector = self._collectors.get("per_row")
        if block.ndim == 3 and per_row_collector is not None:
            length, horizon, dim = block.shape
            flat = block.reshape(length, horizon * dim)
            for col in self._collect_cols_cache.get("per_row", ()):
                per_row_collector.update_column(col, flat[:, col])
            block = block.reshape(length * horizon, dim)
        pooled_collector = self._collectors.get("pooled")
        if pooled_collector is not None:
            for col in self._collect_cols_cache.get("pooled", ()):
                pooled_collector.update_column(col, block[:, col])

    def _collect_columns(self, scope: str) -> list[int]:
        return [
            col
            for col, col_targets in self._targets.get(scope, {}).items()
            if any(target is not None for target in col_targets.values())
        ]

    # ---------------- results ----------------
    def result(self) -> dict:
        pooled = self._scope_result("pooled", self._pooled)
        stats = {
            name: [col[name] for col in pooled]
            for name in ("mean", "std", "min", "max", "q01", "q99")
        }
        if self._per_row is not None:
            horizon, dim = self._per_row_shape
            per_row = self._scope_result("per_row", self._per_row)
            for name, field in (("q01_per_row", "q01"), ("q99_per_row", "q99")):
                flat = [col[field] for col in per_row]
                stats[name] = [
                    flat[h * dim : (h + 1) * dim] for h in range(horizon)
                ]
        return stats

    def _scope_result(self, scope: str, accs) -> list[dict]:
        return [
            self._column_result(
                acc, self._targets[scope][col], self._collectors[scope], col
            )
            for col, acc in enumerate(accs)
        ]

    def _column_result(self, acc, col_targets, collector, col) -> dict:
        if acc.nan > 0 or acc.n == 0:
            return {
                name: float("nan")
                for name in ("mean", "std", "min", "max", "q01", "q99")
            }
        result = {
            "mean": acc.mean,
            "std": float(np.sqrt(acc.m2 / acc.n)),
            "min": float(acc.minimum),
            "max": float(acc.maximum),
        }
        for name, q in (("q01", 0.01), ("q99", 0.99)):
            target = col_targets[q]
            if target is None:
                # Constant column: numpy's lerp of two equal float32 values
                # is that value upcast to float64.
                result[name] = float(np.float64(acc.minimum))
            else:
                lo_value = collector.select(col, target.lo_bucket, target.lo_rank)
                if (
                    target.lo_bucket == target.hi_bucket
                    and target.lo_rank == target.hi_rank
                ):
                    result[name] = float(np.float64(lo_value))
                else:
                    hi_value = collector.select(col, target.hi_bucket, target.hi_rank)
                    result[name] = _lerp_float32_pair(lo_value, hi_value, target.t)
        return result
