"""Map-style LeRobot readers and global batch sampling.

Every valid physical anchor has a stable integer index. Reads decode only the
requested observation/action window. The global batch sampler uses exact
source quotas and stateless permutations."""

from __future__ import annotations

import os
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from lerobot.datasets.video_utils import VideoDecoderCache
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import functional as transforms_F

from ._utils import get_logger
from .sharded_robot_video_dataset import (
    ShardedLerobotV2VideoDataset,
    ShardedLerobotV3VideoDataset,
)

_UINT64_MASK = (1 << 64) - 1
_GOLDEN_GAMMA = 0x9E3779B97F4A7C15
logger = get_logger(__name__)


def _mix_uint64(value: int) -> int:
    value = (int(value) + _GOLDEN_GAMMA) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK


def _feistel_permute_power_of_two(value: int, bits: int, key: int) -> int:
    if bits <= 0 or bits % 2:
        raise ValueError("Feistel domain bits must be a positive even number")
    half_bits = bits // 2
    half_mask = (1 << half_bits) - 1
    left = (int(value) >> half_bits) & half_mask
    right = int(value) & half_mask
    for round_index in range(6):
        round_key = _mix_uint64(int(key) ^ ((round_index + 1) * 0xD1B54A32D192ED03))
        round_value = _mix_uint64(right ^ round_key) & half_mask
        left, right = right, left ^ round_value
    return (left << half_bits) | right


def _permute_index(index: int, size: int, key: int) -> int:
    index = int(index)
    size = int(size)
    if size <= 0:
        raise ValueError("Permutation size must be positive")
    if not 0 <= index < size:
        raise IndexError(f"Permutation index {index} is outside [0, {size})")
    if size == 1:
        return 0
    bits = (size - 1).bit_length()
    if bits % 2:
        bits += 1
    value = index
    while True:
        value = _feistel_permute_power_of_two(value, bits, key)
        if value < size:
            return value


@dataclass(frozen=True)
class MapStyleIndex:
    """One physical read plus its immutable virtual task identity."""

    physical_index: int
    stream_position: int

    def __post_init__(self) -> None:
        if self.physical_index < 0 or self.stream_position < 0:
            raise ValueError("map-style indexes must be non-negative")


class MapStyleRobotVideoDataset(Dataset):
    """One physical ME-U0 source addressed by child/episode/anchor."""

    _group_cls: type

    def __init__(
        self,
        *args: Any,
        enabled: bool = True,
        logical_domain_id: int,
        num_cameras: int,
        main_camera_index: int = 0,
        main_image_size: Optional[list[int]] = None,
        corrupt_sample_retries: int = 16,
        **kwargs: Any,
    ) -> None:
        self.enabled = bool(enabled)
        self.logical_domain_id = int(logical_domain_id)
        self.num_cameras = int(num_cameras)
        self.main_camera_index = int(main_camera_index)
        self.main_image_size = tuple(int(v) for v in (main_image_size or [224, 224]))
        self.corrupt_sample_retries = int(corrupt_sample_retries)
        if self.main_camera_index < 0:
            raise ValueError("main_camera_index must be non-negative")
        if self.logical_domain_id < 0:
            raise ValueError("logical_domain_id must be non-negative")
        if self.num_cameras <= 0:
            raise ValueError("num_cameras must be positive")
        if len(self.main_image_size) != 2 or any(v <= 0 for v in self.main_image_size):
            raise ValueError("main_image_size must contain positive [height,width]")
        if self.corrupt_sample_retries < 0:
            raise ValueError("corrupt_sample_retries must be non-negative")

        self._runtime_corrupt_episodes: set[tuple[int, int]] = set()
        self._configured_source_name = str(
            kwargs.get("dataset_name", type(self).__name__)
        )
        self.epoch = 0
        if not self.enabled:
            self.group = None
            self._episode_cumulative: list[np.ndarray] = []
            self._child_cumulative = np.asarray([], dtype=np.int64)
            self._length = 0
            return
        self.group = self._group_cls(*args, **kwargs)
        self._rebuild_physical_index()


    def _rebuild_physical_index(self) -> None:
        self._episode_cumulative = []
        child_lengths = []
        for child in self.group.children:
            episode_lengths = np.asarray(
                getattr(child, "anchor_lengths", child.episode_lengths),
                dtype=np.int64,
            )
            if np.any(episode_lengths < 0):
                raise ValueError("Anchor lengths must be non-negative")
            cumulative = np.cumsum(episode_lengths)
            self._episode_cumulative.append(cumulative)
            child_lengths.append(int(cumulative[-1]) if len(cumulative) else 0)
        self._child_cumulative = np.cumsum(np.asarray(child_lengths, dtype=np.int64))
        self._length = (
            int(self._child_cumulative[-1]) if len(self._child_cumulative) else 0
        )
        if self._length <= 0:
            raise ValueError(f"Map-style source {self.source_name!r} has no anchors")

    def __getattr__(self, name: str) -> Any:
        group = self.__dict__.get("group")
        if group is None:
            raise AttributeError(name)
        return getattr(group, name)

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _location(self, index: int) -> tuple[int, int, int]:
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        child_index = int(np.searchsorted(self._child_cumulative, index, side="right"))
        child_start = (
            0 if child_index == 0 else int(self._child_cumulative[child_index - 1])
        )
        child_offset = index - child_start
        episodes = self._episode_cumulative[child_index]
        episode_index = int(np.searchsorted(episodes, child_offset, side="right"))
        episode_start = 0 if episode_index == 0 else int(episodes[episode_index - 1])
        return child_index, episode_index, child_offset - episode_start

    @staticmethod
    def _read_anchor(child: Any, episode_index: int, anchor: int) -> Mapping[str, Any]:
        descriptor = (episode_index, anchor, anchor + 1)
        decoder_cache = VideoDecoderCache()
        try:
            payload = child._decode_chunk(descriptor, decoder_cache)
        finally:
            decoder_cache.clear()
        sample = child.get_sample(payload, 0)
        if not isinstance(sample, Mapping):
            raise TypeError(
                f"provider returned {type(sample).__name__}, expected mapping"
            )
        return sample

    def _main_image(self, sample: Mapping[str, Any]) -> torch.Tensor:
        existing = sample.get("main_image")
        if existing is not None:
            image = torch.as_tensor(existing).float()
        else:
            pixels = sample.get("pixel_values")
            if not isinstance(pixels, torch.Tensor):
                raise TypeError("map-style Lance data requires tensor pixel_values")
            if pixels.ndim == 5:
                if self.main_camera_index >= pixels.shape[0]:
                    raise IndexError(
                        f"main_camera_index={self.main_camera_index} exceeds "
                        f"camera count {pixels.shape[0]} for {self.source_name}"
                    )
                image = pixels[self.main_camera_index, 0].float()
            elif pixels.ndim == 4:
                if self.main_camera_index != 0:
                    raise IndexError(
                        "single-camera sample requires main_camera_index=0"
                    )
                image = pixels[0].float()
            else:
                raise ValueError(
                    "pixel_values must be [camera,time,channel,height,width] "
                    f"or [time,channel,height,width], got {tuple(pixels.shape)}"
                )
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"main image must be [3,H,W], got {tuple(image.shape)}")
        if tuple(image.shape[-2:]) != self.main_image_size:
            image = transforms_F.resize(
                image,
                size=list(self.main_image_size),
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
        return image

    def _read_index(self, index: int) -> dict[str, Any]:
        child_index, episode_index, anchor = self._location(index)
        child = self.group.children[child_index]
        sample = self._read_anchor(child, episode_index, anchor)
        main_image = self._main_image(sample)
        output = self.group._postprocess(sample)
        if not isinstance(output, dict):
            raise TypeError("ME-U0 group postprocess must return a dictionary")
        output.setdefault("source", self.source_name)
        output["domain_id"] = self.logical_domain_id
        output["num_cameras"] = self.num_cameras
        output["main_image"] = main_image
        metadata = dict(output.get("metadata") or {})
        metadata.setdefault("future_video_modality", "rgb")
        metadata.setdefault("main_image_modality", "rgb")
        metadata.setdefault("current_video_frame_modality", "rgb")
        output["metadata"] = metadata
        return output

    def _replacement_indices(self, requested_index: int) -> Iterator[int]:
        yield requested_index
        if self.corrupt_sample_retries == 0 or len(self) <= 1:
            return
        key = _mix_uint64(
            self.epoch * 1_000_003
            ^ requested_index * 0xD1B54A32D192ED03
            ^ 0xA24BAED4963EE407
        )
        yielded = {requested_index}
        slot = 0
        target_count = min(len(self), self.corrupt_sample_retries + 1)
        while len(yielded) < target_count:
            candidate = _permute_index(slot, len(self), key)
            slot += 1
            if candidate not in yielded:
                yielded.add(candidate)
                yield candidate

    def __getitem__(self, index: int) -> dict[str, Any]:
        requested_index = int(index)
        if requested_index < 0:
            requested_index += len(self)
        if not 0 <= requested_index < len(self):
            raise IndexError(index)
        skip_corrupt = bool(getattr(self.group, "skip_corrupt_data", False))
        failures = []
        for candidate in self._replacement_indices(requested_index):
            child_index, episode_index, _anchor = self._location(candidate)
            episode_key = (child_index, episode_index)
            if skip_corrupt and episode_key in self._runtime_corrupt_episodes:
                continue
            try:
                output = self._read_index(candidate)
            except Exception as exc:
                if not skip_corrupt:
                    raise
                failures.append(f"{type(exc).__name__}: {exc}")
                self._runtime_corrupt_episodes.add(episode_key)
                logger.warning(
                    "skip_corrupt_data: source=%s requested=%d candidate=%d failed: %s",
                    self.source_name,
                    requested_index,
                    candidate,
                    failures[-1],
                )
                continue
            if candidate != requested_index:
                output.setdefault("metadata", {})["replacement_physical_index"] = (
                    candidate
                )
            return output
        raise RuntimeError(
            f"Map-style retry budget exhausted for source={self.source_name} "
            f"index={requested_index}: {'; '.join(failures[-3:])}"
        )

    @property
    def source_name(self) -> str:
        if not self.enabled:
            return self._configured_source_name
        return str(getattr(self.group, "_dataset_name", type(self.group).__name__))

    def get_source(self, _index: int) -> str:
        return self.source_name


class MapStyleLerobotV3VideoDataset(MapStyleRobotVideoDataset):
    _group_cls = ShardedLerobotV3VideoDataset


class MapStyleLerobotV2VideoDataset(MapStyleRobotVideoDataset):
    _group_cls = ShardedLerobotV2VideoDataset


class MapStyleRobotMixtureDataset(Dataset):
    """Concatenate enabled physical map-style sources without virtual repeats."""

    def __init__(
        self,
        datasets: list[Any],
        *,
        skip_corrupt_data: bool = False,
        corrupt_sample_retries: int = 16,
    ) -> None:
        self.datasets: list[MapStyleRobotVideoDataset] = []
        lengths = []
        for index, dataset in enumerate(datasets):
            if not isinstance(dataset, MapStyleRobotVideoDataset):
                raise TypeError(
                    f"datasets[{index}] must be map-style, got {type(dataset).__name__}"
                )
            if not dataset.enabled:
                continue
            dataset.corrupt_sample_retries = int(corrupt_sample_retries)
            dataset.group.skip_corrupt_data = bool(skip_corrupt_data)
            for child in dataset.group.children:
                child.skip_corrupt = bool(skip_corrupt_data)
            self.datasets.append(dataset)
            lengths.append(len(dataset))
        if not self.datasets:
            raise ValueError("datasets must contain at least one enabled source")
        self._cumulative = np.cumsum(np.asarray(lengths, dtype=np.int64))
        self._length = int(self._cumulative[-1])

    def set_epoch(self, epoch: int) -> None:
        for dataset in self.datasets:
            dataset.set_epoch(epoch)

    def __len__(self) -> int:
        return self._length

    def _location(self, index: int) -> tuple[int, int]:
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        dataset_index = int(np.searchsorted(self._cumulative, index, side="right"))
        start = 0 if dataset_index == 0 else int(self._cumulative[dataset_index - 1])
        return dataset_index, index - start

    def __getitem__(self, index: int) -> dict[str, Any]:
        dataset_index, local_index = self._location(index)
        return self.datasets[dataset_index][local_index]

    def get_source(self, index: int) -> str:
        dataset_index, local_index = self._location(index)
        return self.datasets[dataset_index].get_source(local_index)

    @property
    def source_counts(self) -> Counter[str]:
        return Counter({dataset.source_name: len(dataset) for dataset in self.datasets})

    @property
    def source_ranges(self) -> list[tuple[int, int, str]]:
        ranges = []
        start = 0
        for dataset, stop in zip(self.datasets, self._cumulative):
            ranges.append((start, int(stop), dataset.source_name))
            start = int(stop)
        return ranges


class MapStyleGlobalBatchSampler(Sampler[list[MapStyleIndex]]):
    """Exactly weighted global batches over a stateless virtual sample space."""

    def __init__(
        self,
        base_sampler: Sampler[int],
        sample_type_fn: Optional[Any] = None,
        *,
        batch_size: int,
        drop_last: bool = True,
        seed: int = 0,
        num_processes: Optional[int] = None,
        num_samples_per_epoch: Optional[int] = None,
        source_weights: Optional[Mapping[str, float]] = None,
        replacement: bool = True,
        round_up_to_global_batch: bool = True,
    ) -> None:
        del sample_type_fn
        self.base_sampler = base_sampler
        self.data_source = getattr(base_sampler, "data_source", None)
        raw = self.data_source
        seen = set()
        while (
            raw is not None
            and not hasattr(raw, "source_ranges")
            and id(raw) not in seen
        ):
            seen.add(id(raw))
            raw = getattr(raw, "dataset", None)
        if raw is None or not hasattr(raw, "source_ranges"):
            raise TypeError("MapStyleGlobalBatchSampler needs source_ranges")
        self.map_dataset = raw
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.num_processes = int(num_processes or os.environ.get("WORLD_SIZE", "1"))
        self.num_samples_per_epoch = (
            None if num_samples_per_epoch is None else int(num_samples_per_epoch)
        )
        self.source_weights = (
            None
            if source_weights is None
            else {str(k): float(v) for k, v in dict(source_weights).items()}
        )
        self.replacement = bool(replacement)
        self.round_up_to_global_batch = bool(round_up_to_global_batch)
        self.epoch = 0
        if self.batch_size <= 0 or self.num_processes <= 0:
            raise ValueError("batch_size and num_processes must be positive")
        self._validate_source_weights()

    def _validate_source_weights(self) -> None:
        if self.source_weights is None:
            return
        names = {source for _start, _stop, source in self.map_dataset.source_ranges}
        missing = names - set(self.source_weights)
        extra = {
            source
            for source in set(self.source_weights) - names
            if self.source_weights[source] != 0
        }
        if missing or extra:
            raise ValueError(
                f"source_weights mismatch: missing={sorted(missing)}, "
                f"nonzero_disabled={sorted(extra)}"
            )
        weights = [self.source_weights[name] for name in names]
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError(
                "active source weights must be non-negative with positive sum"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        setter = getattr(self.data_source, "set_epoch", None)
        if callable(setter):
            setter(self.epoch)

    def _total_samples(self) -> int:
        total = (
            sum(stop - start for start, stop, _source in self.map_dataset.source_ranges)
            if self.num_samples_per_epoch is None
            else self.num_samples_per_epoch
        )
        global_batch = self.batch_size * self.num_processes
        remainder = total % global_batch
        if remainder:
            if self.round_up_to_global_batch:
                total += global_batch - remainder
            elif self.drop_last:
                total -= remainder
            else:
                raise ValueError(
                    f"sample budget {total} is not divisible by global batch {global_batch}"
                )
        if total <= 0:
            raise ValueError("sample budget is smaller than one global batch")
        return total

    def _sample_quotas(self) -> list[int]:
        ranges = self.map_dataset.source_ranges
        lengths = np.asarray(
            [stop - start for start, stop, _ in ranges], dtype=np.int64
        )
        total = self._total_samples()
        weights = (
            lengths.astype(np.float64)
            if self.source_weights is None
            else np.asarray(
                [self.source_weights[source] for _start, _stop, source in ranges],
                dtype=np.float64,
            )
        )
        exact = weights / weights.sum() * total
        allocated = np.floor(exact).astype(np.int64)
        remainder = int(total - allocated.sum())
        if remainder:
            order = sorted(
                range(len(ranges)),
                key=lambda index: (-(exact[index] - allocated[index]), index),
            )
            allocated[order[:remainder]] += 1
        quotas = [int(value) for value in allocated]
        if not self.replacement:
            excessive = [
                (ranges[index][2], quota, int(lengths[index]))
                for index, quota in enumerate(quotas)
                if quota > lengths[index]
            ]
            if excessive:
                raise ValueError(f"source quota exceeds physical anchors: {excessive}")
        return quotas

    @property
    def allocated_samples(self) -> dict[str, int]:
        return {
            source: quota
            for quota, (_start, _stop, source) in zip(
                self._sample_quotas(), self.map_dataset.source_ranges
            )
        }

    def _physical_index(self, virtual_index: int, quota_stops: list[int]) -> int:
        source_index = bisect_right(quota_stops, int(virtual_index))
        quota_start = 0 if source_index == 0 else quota_stops[source_index - 1]
        local_occurrence = int(virtual_index) - quota_start
        start, stop, _source = self.map_dataset.source_ranges[source_index]
        physical_size = stop - start
        cycle, position = divmod(local_occurrence, physical_size)
        source_key = _mix_uint64(
            self.seed
            ^ self.epoch * 1_000_003
            ^ source_index * 0x94D049BB133111EB
            ^ cycle * 0xD1B54A32D192ED03
        )
        return start + _permute_index(position, physical_size, source_key)

    def __iter__(self) -> Iterator[list[MapStyleIndex]]:
        total = self._total_samples()
        quota_stops = np.cumsum(
            np.asarray(self._sample_quotas(), dtype=np.int64)
        ).tolist()
        global_batch = self.batch_size * self.num_processes
        global_batches = total // global_batch
        global_key = _mix_uint64(
            self.seed ^ self.epoch * 1_000_003 ^ 0xA0761D6478BD642F
        )
        epoch_offset = self.epoch * total
        for global_batch_index in range(global_batches):
            first_slot = global_batch_index * global_batch
            tokens = []
            for offset in range(global_batch):
                virtual_index = _permute_index(first_slot + offset, total, global_key)
                tokens.append(
                    MapStyleIndex(
                        physical_index=self._physical_index(virtual_index, quota_stops),
                        stream_position=epoch_offset + virtual_index,
                    )
                )
            for process_index in range(self.num_processes):
                start = process_index * self.batch_size
                yield tokens[start : start + self.batch_size]

    def __len__(self) -> int:
        global_batch = self.batch_size * self.num_processes
        return (self._total_samples() // global_batch) * self.num_processes


__all__ = [
    "MapStyleGlobalBatchSampler",
    "MapStyleIndex",
    "MapStyleLerobotV2VideoDataset",
    "MapStyleLerobotV3VideoDataset",
    "MapStyleRobotMixtureDataset",
    "MapStyleRobotVideoDataset",
]
