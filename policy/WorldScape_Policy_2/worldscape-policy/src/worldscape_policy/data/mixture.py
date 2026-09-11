"""Deterministic native sharded mixtures."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from worldscape_policy.data.schema import EventSample


def _profile_enabled() -> bool:
    return os.environ.get("WSP_DATALOADER_PROFILE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _profile_log(event: str, payload: dict[str, Any]) -> None:
    if not _profile_enabled():
        return
    print(
        json.dumps({"dataloader_profile": event, **payload}, sort_keys=True),
        flush=True,
    )


def _profile_sample_enabled() -> bool:
    return os.environ.get("WSP_DATALOADER_PROFILE_SAMPLE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class NativeShard:
    dataset_index: int
    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


class NativeShardedMixtureDataset(IterableDataset[EventSample]):
    """Weighted shard schedule with resumable global-position semantics.

    Map-style native HDF5/LeRobot readers can be supplied directly.  The global
    schedule is generated before rank/worker filtering, so all processes agree
    on ordering and resume positions.
    """

    partitions_workers = True

    def __init__(
        self,
        datasets: Sequence[Any],
        *,
        mixture_weights: Sequence[float] | None = None,
        shard_size: int = 10_000,
        shard_sampling_rate: float = 0.1,
        num_shards_to_sample: int = 2**20,
        seed: int = 42,
        training: bool = True,
        exec_early_sampling_enabled: bool = False,
        exec_early_ratio: float = 0.25,
        exec_early_weight: float = 3.0,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        super().__init__()
        if not datasets:
            raise ValueError("datasets cannot be empty")
        if shard_size <= 0 or num_shards_to_sample <= 0:
            raise ValueError("shard_size and num_shards_to_sample must be positive")
        if not 0 <= shard_sampling_rate <= 1:
            raise ValueError("shard_sampling_rate must be in [0, 1]")
        self.datasets = tuple(datasets)
        lengths = np.asarray([len(dataset) for dataset in datasets], dtype=np.int64)
        if np.any(lengths <= 0):
            raise ValueError("mixture datasets cannot be empty")
        weights = (
            np.ones(len(datasets), dtype=np.float64)
            if mixture_weights is None
            else np.asarray(mixture_weights, dtype=np.float64)
        )
        if weights.shape != (len(datasets),) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("mixture_weights must be nonnegative and match datasets")
        self.shard_size = int(shard_size)
        self.shard_sampling_rate = float(shard_sampling_rate)
        self.num_shards_to_sample = int(num_shards_to_sample)
        self.seed = int(seed)
        self.training = bool(training)
        self.exec_early_sampling_enabled = bool(exec_early_sampling_enabled)
        self.exec_early_ratio = float(np.clip(exec_early_ratio, 0, 1))
        self.exec_early_weight = max(1.0, float(exec_early_weight))
        self.rank = (
            int(rank)
            if rank is not None
            else (dist.get_rank() if dist.is_available() and dist.is_initialized() else 0)
        )
        self.world_size = (
            int(world_size)
            if world_size is not None
            else (dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1)
        )
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        shards = []
        for dataset_index, (dataset, length) in enumerate(zip(datasets, lengths)):
            provider = getattr(dataset, "shard_ranges", None)
            ranges = (
                provider(shard_size)
                if provider is not None
                else tuple(
                    (start, min(start + shard_size, int(length)))
                    for start in range(0, int(length), shard_size)
                )
            )
            shards.extend(
                NativeShard(dataset_index, int(start), int(stop))
                for start, stop in ranges
            )
        self.shards = tuple(shards)
        shard_weights = np.asarray(
            [
                weights[shard.dataset_index]
                * shard.length
                / lengths[shard.dataset_index]
                for shard in self.shards
            ],
            dtype=np.float64,
        )
        self.shard_weights = shard_weights / shard_weights.sum()
        self.schedule = self._make_schedule()
        self.global_cursor = 0
        self.sample_cursor = 0

    def _make_schedule(self) -> np.ndarray:
        if not self.training:
            return np.arange(self.num_shards_to_sample, dtype=np.int64) % len(self.shards)
        rng = np.random.default_rng(self.seed)
        schedule = rng.choice(
            len(self.shards),
            size=self.num_shards_to_sample,
            replace=True,
            p=self.shard_weights,
        )
        rng.shuffle(schedule)
        return schedule.astype(np.int64)

    def _sample_shard(
        self,
        shard: NativeShard,
        rng: np.random.Generator,
    ) -> np.ndarray:
        count = min(
            shard.length,
            int(self.shard_size * self.shard_sampling_rate),
        )
        if count <= 0:
            return np.empty(0, dtype=np.int64)
        indices = np.arange(shard.start, shard.stop, dtype=np.int64)
        if not self.training:
            return indices[:count]
        if self.exec_early_sampling_enabled and self.exec_early_ratio > 0:
            progress_provider = getattr(
                self.datasets[shard.dataset_index],
                "sampling_progress",
                None,
            )
            if callable(progress_provider):
                progress = np.asarray(progress_provider(indices), dtype=np.float64)
                if (
                    progress.shape != indices.shape
                    or not np.isfinite(progress).all()
                    or np.any((progress < 0) | (progress > 1))
                ):
                    raise ValueError(
                        "sampling_progress must return finite values in [0, 1] "
                        "matching the requested indices"
                    )
                early_mask = progress < self.exec_early_ratio
            else:
                early = max(1, int(np.floor(self.exec_early_ratio * len(indices))))
                early_mask = np.arange(len(indices)) < early
            weights = np.ones(len(indices), dtype=np.float64)
            weights[early_mask] = self.exec_early_weight
            chosen = rng.choice(
                len(indices),
                size=count,
                replace=False,
                p=weights / weights.sum(),
            )
            return indices[chosen]
        rng.shuffle(indices)
        return indices[:count]

    def _load_shard(self, shard: NativeShard) -> Any:
        dataset = self.datasets[shard.dataset_index]
        loader = getattr(dataset, "load_shard", None)
        return None if loader is None else loader(shard.start, shard.stop)

    def __iter__(self) -> Iterator[EventSample]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        partition = self.rank * num_workers + worker_id
        partitions = self.world_size * num_workers
        start = self.global_cursor
        positions = [
            position
            for position in range(start, len(self.schedule))
            if position % partitions == partition
        ]
        rng = np.random.default_rng(self.seed)
        # Reproduce the worker-local legacy RNG stream after an exact resume.
        for schedule_position in range(start):
            if schedule_position % partitions == partition:
                prior = self.shards[int(self.schedule[schedule_position])]
                self._sample_shard(prior, rng)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future: Future[Any] | None = None
            if positions:
                first = self.shards[int(self.schedule[positions[0]])]
                future = executor.submit(self._load_shard, first)
            for local_position, schedule_position in enumerate(positions):
                shard = self.shards[int(self.schedule[schedule_position])]
                assert future is not None
                wait_start = time.perf_counter()
                cached_shard = future.result()
                wait_s = time.perf_counter() - wait_start
                future = None
                if local_position + 1 < len(positions):
                    following = self.shards[
                        int(self.schedule[positions[local_position + 1]])
                    ]
                    future = executor.submit(self._load_shard, following)
                sampled = self._sample_shard(shard, rng)
                _profile_log(
                    "mixture_shard_ready",
                    {
                        "rank": self.rank,
                        "worker_id": worker_id,
                        "schedule_position": int(schedule_position),
                        "dataset_index": int(shard.dataset_index),
                        "start": int(shard.start),
                        "stop": int(shard.stop),
                        "shard_length": int(shard.length),
                        "sampled": int(len(sampled)),
                        "wait_s": round(wait_s, 6),
                    },
                )
                offset = self.sample_cursor if schedule_position == start else 0
                dataset = self.datasets[shard.dataset_index]
                getter = getattr(dataset, "get_from_shard", None)
                sample_profile = _profile_sample_enabled()
                for sample_position, index in enumerate(
                    sampled[offset:], start=offset
                ):
                    if worker is None:
                        self.global_cursor = schedule_position
                        self.sample_cursor = sample_position + 1
                    sample_start = time.perf_counter()
                    sample = (
                        dataset[int(index)]
                        if getter is None
                        else getter(int(index), cached_shard)
                    )
                    get_sample_s = time.perf_counter() - sample_start
                    if not isinstance(sample, EventSample):
                        raise TypeError("native mixture children must yield EventSample")
                    if sample_profile:
                        _profile_log(
                            "mixture_sample",
                            {
                                "rank": self.rank,
                                "worker_id": worker_id,
                                "schedule_position": int(schedule_position),
                                "dataset_index": int(shard.dataset_index),
                                "sample_position": int(sample_position),
                                "index": int(index),
                                "get_sample_s": round(get_sample_s, 6),
                            },
                        )
                    yield sample
                del cached_shard
                if worker is None:
                    self.global_cursor = schedule_position + 1
                    self.sample_cursor = 0

    def state_dict(self) -> dict[str, int]:
        if get_worker_info() is not None:
            raise RuntimeError(
                "mixture worker copies cannot be checkpointed; use DataLoader num_workers=0"
            )
        return {
            "version": 2,
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "global_cursor": self.global_cursor,
            "sample_cursor": self.sample_cursor,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        if (
            int(state.get("version", -1)) != 2
            or int(state.get("seed", -1)) != self.seed
            or int(state.get("rank", -1)) != self.rank
            or int(state.get("world_size", -1)) != self.world_size
        ):
            raise ValueError("mixture resume state version/seed/rank/world_size mismatch")
        cursor = int(state["global_cursor"])
        if not 0 <= cursor <= len(self.schedule):
            raise ValueError("mixture global_cursor is outside schedule")
        self.global_cursor = cursor
        self.sample_cursor = int(state.get("sample_cursor", 0))
        if self.sample_cursor < 0:
            raise ValueError("mixture sample_cursor must be non-negative")

    def __len__(self) -> int:
        return sum(
            min(shard.length, int(self.shard_size * self.shard_sampling_rate))
            for shard_id in self.schedule
            for shard in (self.shards[int(shard_id)],)
        )


WeightedShardedMixtureDataset = NativeShardedMixtureDataset

__all__ = [
    "NativeShard",
    "NativeShardedMixtureDataset",
    "WeightedShardedMixtureDataset",
]
