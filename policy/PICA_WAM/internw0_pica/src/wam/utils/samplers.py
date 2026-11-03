from dataclasses import dataclass
from itertools import islice
from typing import Iterator, Sequence, Sized

import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    def __init__(
        self,
        dataset: Sized,
        seed: int,
        batch_size: int,
        num_processes: int,
        *,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            start = (self.epoch + self.epoch_offset) * len(self.dataset)
            indices = range(start, start + len(self.dataset))
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)


@dataclass(frozen=True)
class _EpisodeShardSegment:
    episode: int
    start: int
    stop: int


class _SegmentCursor:
    def __init__(
        self,
        segment: _EpisodeShardSegment,
        samples_per_episode: int,
        generator: torch.Generator,
    ) -> None:
        self.segment = segment
        self.samples_per_episode = int(samples_per_episode)
        self.chunk_count = (segment.stop - segment.start) // self.samples_per_episode
        self.chunk_order = torch.randperm(
            self.chunk_count,
            generator=generator,
        ).tolist()
        self.position = 0

    @property
    def exhausted(self) -> bool:
        return self.position >= self.chunk_count

    def pop_chunk(self) -> range:
        if self.exhausted:
            raise RuntimeError("Cannot pop from an exhausted episode-shard segment.")
        chunk_index = self.chunk_order[self.position]
        self.position += 1
        start = self.segment.start + chunk_index * self.samples_per_episode
        return range(start, start + self.samples_per_episode)

    def remaining_indices(self) -> Iterator[int]:
        for chunk_index in self.chunk_order[self.position :]:
            start = self.segment.start + chunk_index * self.samples_per_episode
            yield from range(start, start + self.samples_per_episode)


class EpisodeShardSampler(Sampler[int]):
    """Batch-local episode sampling with cache-shard locality.

    The flattened index stream is arranged in global batches.  Accelerate's
    no-split batch sharding assigns consecutive batches to ranks, so every
    local batch contains ``episodes_per_batch`` distinct episode/shard streams
    and ``batch_size // episodes_per_batch`` consecutive samples from each.

    Episode/shard intersections that are not divisible by the per-episode
    sample count are shuffled into a small tail.  This keeps every dataset
    index exactly once per sampler epoch while preserving the structured
    layout for the bulk of the epoch.
    """

    def __init__(
        self,
        dataset: Sized,
        seed: int,
        batch_size: int,
        num_processes: int,
        *,
        episode_ranges: Sequence[Sequence[int]],
        shard_ranges: Sequence[Sequence[int]],
        episodes_per_batch: int = 4,
    ) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.episodes_per_batch = int(episodes_per_batch)
        if self.batch_size <= 0 or self.num_processes <= 0:
            raise ValueError("batch_size and num_processes must be positive.")
        if self.episodes_per_batch <= 0:
            raise ValueError("episodes_per_batch must be positive.")
        if self.batch_size % self.episodes_per_batch != 0:
            raise ValueError(
                "batch_size must be divisible by episodes_per_batch: "
                f"{self.batch_size} vs {self.episodes_per_batch}."
            )
        self.samples_per_episode = self.batch_size // self.episodes_per_batch
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

        dataset_size = len(dataset)
        episodes = self._normalize_partition_ranges(
            episode_ranges,
            size=dataset_size,
            label="episode",
        )
        shards = self._normalize_partition_ranges(
            shard_ranges,
            size=dataset_size,
            label="cache shard",
        )
        self._segments, self._tail_indices = self._intersect_partitions(
            episodes,
            shards,
        )
        self.structured_sample_count = sum(
            ((segment.stop - segment.start) // self.samples_per_episode)
            * self.samples_per_episode
            for segment in self._segments
        )
        self.tail_sample_count = len(self._tail_indices)
        if self.structured_sample_count + self.tail_sample_count != dataset_size:
            raise RuntimeError(
                "Episode-shard sampler coverage mismatch: "
                f"structured={self.structured_sample_count} "
                f"tail={self.tail_sample_count} dataset={dataset_size}."
            )

    @staticmethod
    def _normalize_partition_ranges(
        ranges: Sequence[Sequence[int]],
        *,
        size: int,
        label: str,
    ) -> tuple[tuple[int, int], ...]:
        normalized: list[tuple[int, int]] = []
        cursor = 0
        for ordinal, raw_range in enumerate(ranges):
            if len(raw_range) < 2:
                raise ValueError(f"Invalid {label} range at {ordinal}: {raw_range!r}.")
            start, stop = int(raw_range[0]), int(raw_range[1])
            if start != cursor or stop <= start:
                raise ValueError(
                    f"{label.title()} ranges must be a contiguous partition; "
                    f"range {ordinal} is [{start}, {stop}) after cursor {cursor}."
                )
            normalized.append((start, stop))
            cursor = stop
        if cursor != size:
            raise ValueError(
                f"{label.title()} ranges cover {cursor} samples, expected {size}."
            )
        return tuple(normalized)

    def _intersect_partitions(
        self,
        episodes: tuple[tuple[int, int], ...],
        shards: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[_EpisodeShardSegment, ...], tuple[int, ...]]:
        segments: list[_EpisodeShardSegment] = []
        tail_indices: list[int] = []
        episode_index = 0
        shard_index = 0
        while episode_index < len(episodes) and shard_index < len(shards):
            episode_start, episode_stop = episodes[episode_index]
            shard_start, shard_stop = shards[shard_index]
            start = max(episode_start, shard_start)
            stop = min(episode_stop, shard_stop)
            if start < stop:
                structured_stop = start + (
                    (stop - start) // self.samples_per_episode
                ) * self.samples_per_episode
                if structured_stop > start:
                    segments.append(
                        _EpisodeShardSegment(
                            episode=episode_index,
                            start=start,
                            stop=structured_stop,
                        )
                    )
                tail_indices.extend(range(structured_stop, stop))
            if episode_stop <= shard_stop:
                episode_index += 1
            if shard_stop <= episode_stop:
                shard_index += 1
        return tuple(segments), tuple(tail_indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int) -> None:
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int) -> None:
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self) -> None:
        self.resume_batch_offset = 0

    @staticmethod
    def _take_compatible_segment(
        pending: list[_EpisodeShardSegment],
        excluded_episodes: set[int],
    ) -> _EpisodeShardSegment | None:
        for position in range(len(pending) - 1, -1, -1):
            if pending[position].episode not in excluded_episodes:
                return pending.pop(position)
        return None

    def _epoch_indices(self) -> Iterator[int]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch + self.epoch_offset)
        order = torch.randperm(len(self._segments), generator=generator).tolist()
        pending = [self._segments[index] for index in order]
        active: list[list[_SegmentCursor]] = [
            [] for _ in range(self.num_processes)
        ]

        while True:
            all_ranks_ready = True
            for rank_streams in active:
                excluded = {cursor.segment.episode for cursor in rank_streams}
                while len(rank_streams) < self.episodes_per_batch:
                    segment = self._take_compatible_segment(pending, excluded)
                    if segment is None:
                        all_ranks_ready = False
                        break
                    rank_streams.append(
                        _SegmentCursor(
                            segment,
                            self.samples_per_episode,
                            generator,
                        )
                    )
                    excluded.add(segment.episode)
                if not all_ranks_ready:
                    break
            if not all_ranks_ready:
                break

            # Emit one global wave in rank order.  Accelerate assigns batch N
            # in this wave to process N without splitting the local batch.
            for rank_streams in active:
                for cursor in rank_streams:
                    yield from cursor.pop_chunk()
                rank_streams[:] = [
                    cursor for cursor in rank_streams if not cursor.exhausted
                ]

        # Use every sample once.  The small non-divisible remainder cannot
        # satisfy an exact four-episode batch without repetition, so it is
        # shuffled together with any streams left at the final global wave.
        remaining = list(self._tail_indices)
        for rank_streams in active:
            for cursor in rank_streams:
                remaining.extend(cursor.remaining_indices())
        for segment in pending:
            remaining.extend(range(segment.start, segment.stop))
        if remaining:
            tail_order = torch.randperm(len(remaining), generator=generator).tolist()
            for position in tail_order:
                yield remaining[position]

    def __iter__(self) -> Iterator[int]:
        indices: Iterator[int] = self._epoch_indices()
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = (
                self.resume_batch_offset * self.batch_size * self.num_processes
            )
            indices = islice(indices, sample_offset, None)
        return indices

    def __len__(self) -> int:
        return len(self.dataset)
