"""Global metadata scheduling before rank slicing and worker materialization.

The planner runs identically in every rank's parent process. Worker prefetch
advances only speculative state; the loader acknowledges batches on delivery.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from worldscape_policy.data.adapters.robodojo import RoboDojoJointHDF5Dataset
from worldscape_policy.data.collate import NativeTrainingCollator
from worldscape_policy.data.mixture import NativeShardedMixtureDataset
from worldscape_policy.data.transforms import NativeEventTransform
from worldscape_policy.training.metadata_buckets import _ShardCache
from worldscape_policy.training.selected_frames import SelectedFrameShardCache
from worldscape_policy.training.timing_trace import TraceSpan, configure_trace
from worldscape_policy.types import InteractionMode


def _identity(value):
    return value


class GlobalBucketPlanner(Sampler):
    """Bounded windows of whole global batches, without reading image payloads."""

    def __init__(self, mixture: NativeShardedMixtureDataset, batch_size: int,
                 window_batches: int = 128) -> None:
        if batch_size <= 0 or window_batches <= 0:
            raise ValueError("global batch size and window_batches must be positive")
        if int(mixture.shard_size * mixture.shard_sampling_rate) <= 0:
            raise ValueError("global bucketing requires a nonempty sample stream")
        self.mixture = mixture
        self.batch_size = int(batch_size)
        self.global_size = self.batch_size * mixture.world_size
        self.window_batches = int(window_batches)
        digest = hashlib.sha256(mixture.schedule.tobytes())
        for child in mixture.datasets:
            digest.update(np.asarray(child.index, dtype=np.int64).tobytes())
            digest.update(json.dumps([str(p) for p in child.paths]).encode())
            # Temporal geometry is part of the data-state identity as well.
            digest.update(str((child.action_horizon, child.temporal_max_chunk_size,
                               child.chunk_policy)).encode())
        self.identity = dict(
            version=3, kind="global_length_buckets", seed=mixture.seed,
            world_size=mixture.world_size, batch_size=self.batch_size,
            window_batches=self.window_batches, dataset_digest=digest.hexdigest(),
            shard_size=mixture.shard_size, sampling_rate=mixture.shard_sampling_rate,
            training=mixture.training, early=mixture.exec_early_sampling_enabled,
            early_ratio=mixture.exec_early_ratio, early_weight=mixture.exec_early_weight,
        )
        self.committed = self._initial(0)
        self.inflight: dict[int, dict] = {}
        self.last_info: dict[str, Any] = {}
        self.tail_counts: dict[str, int] = {}

    @staticmethod
    def _initial(epoch):
        return dict(epoch=epoch, position=0, offset=0, window=0, pending={},
                    ready=[], delivered=0, batches=0, dropped={})

    def key(self, reference: tuple[int, int]) -> tuple[bool, int, int, int]:
        position, index = reference
        shard = self.mixture.shards[int(self.mixture.schedule[position])]
        child = self.mixture.datasets[shard.dataset_index]
        return (bool(child.memory_task(index)), *map(int, child.length_signature(index)))

    def _fill(self, state):
        """Return a new immutable window snapshot; carry incomplete buckets."""
        pending = defaultdict(list, {k: list(v) for k, v in state['pending'].items()})
        position, offset = state['position'], state['offset']
        remaining = self.window_batches * self.global_size
        while remaining and position < len(self.mixture.schedule):
            shard = self.mixture.shards[int(self.mixture.schedule[position])]
            rng = np.random.default_rng(np.random.SeedSequence(
                [self.mixture.seed, state['epoch'], position, 0]))
            indices = self.mixture._sample_shard(shard, rng)
            take = min(remaining, len(indices) - offset)
            for index in indices[offset:offset + take]:
                ref = (position, int(index))
                pending[self.key(ref)].append(ref)
            remaining -= take
            offset += take
            if offset == len(indices):
                position, offset = position + 1, 0
        ready = []
        tails = {}
        member_rng = np.random.default_rng(np.random.SeedSequence(
            [self.mixture.seed, state['epoch'], state['window'], 2]))
        block_rng = np.random.default_rng(np.random.SeedSequence(
            [self.mixture.seed, state['epoch'], state['window'], 3]))
        for key in sorted(pending):
            refs = pending[key]
            # Choose the same complete-bucket population before packing. The
            # tail remains a random subset, not systematically rare shards.
            member_rng.shuffle(refs)
            complete = len(refs) // self.global_size * self.global_size
            by_shard = defaultdict(list)
            for ref in refs[:complete]:
                by_shard[int(self.mixture.schedule[ref[0]])].append(ref)
            local_batches, remainder = [], []
            for shard_refs in by_shard.values():
                stop = len(shard_refs) // self.batch_size * self.batch_size
                local_batches.extend(shard_refs[i:i + self.batch_size]
                                     for i in range(0, stop, self.batch_size))
                remainder.extend(shard_refs[stop:])
            # Pool only per-shard remainders; never discard a rare shard or
            # add draws to fill its local batch. The total is divisible by B.
            local_batches.extend(remainder[i:i + self.batch_size]
                                 for i in range(0, len(remainder), self.batch_size))
            # Keep packing randomness separate so it cannot change which
            # samples are carried to the next window in subsequent buckets.
            block_rng.shuffle(local_batches)
            for start in range(0, len(local_batches), self.mixture.world_size):
                blocks = local_batches[start:start + self.mixture.world_size]
                # Keep a stable shard order across ranks to help their caches,
                # but do not split the shard-local blocks selected above.
                blocks.sort(key=lambda block: int(self.mixture.schedule[block[0][0]]))
                ready.append((key, [ref for block in blocks for ref in block]))
            if complete < len(refs):
                tails[key] = refs[complete:]
        rng = np.random.default_rng(np.random.SeedSequence(
            [self.mixture.seed, state['epoch'], state['window'], 1]))
        rng.shuffle(ready)
        dropped = {}
        if position == len(self.mixture.schedule):
            dropped = {str(key): len(refs) for key, refs in tails.items()}
            tails = {}
        return dict(epoch=state['epoch'], position=position, offset=offset,
                    window=state['window'] + 1, pending=tails, ready=ready,
                    delivered=0, batches=state['batches'], dropped=dropped)

    def __iter__(self) -> Iterator:
        # No decoded samples or worker-side state is part of the planner.
        self.inflight.clear()
        state = self.committed
        sequence = state['batches']
        while True:
            for offset in range(state['delivered'], len(state['ready'])):
                key, refs = state['ready'][offset]
                sequence += 1
                receipt = dict(state, delivered=offset + 1, batches=sequence)
                self.inflight[sequence] = receipt
                start = self.mixture.rank * self.batch_size
                yield (sequence, key, refs[start:start + self.batch_size])
            if state['position'] == len(self.mixture.schedule):
                self.tail_counts = state['dropped']
                if sequence == 0:
                    raise ValueError("global bucketing produced no complete global batch")
                return
            with TraceSpan('planner_fill', epoch=state['epoch'], window=state['window'],
                           after_batch=sequence, global_size=self.global_size,
                           window_batches=self.window_batches):
                state = self._fill(dict(state, batches=sequence))

    def acknowledge(self, sequence: int, key: tuple) -> None:
        if sequence != self.committed['batches'] + 1:
            raise ValueError("global batches must be delivered in order")
        self.committed = self.inflight.pop(sequence)
        self.last_info = dict(batch=sequence, memory=bool(key[0]),
                              lengths=list(key[1:]), dropped=self.committed['dropped'])

    def finish_epoch(self) -> None:
        # Called only after the consumer reaches EOF, never by prefetch.
        self.committed = self._initial(self.committed['epoch'] + 1)
        self.inflight.clear()

    def state_dict(self) -> dict[str, Any]:
        return dict(identity=self.identity, state=copy.deepcopy(self.committed))

    def load_state_dict(self, value: dict[str, Any]) -> None:
        if value.get('identity') != self.identity:
            raise ValueError("global bucket resume requires the same loader, data, world size and batch configuration")
        state = copy.deepcopy(value['state'])
        if (not 0 <= state['position'] <= len(self.mixture.schedule)
                or not 0 <= state['delivered'] <= len(state['ready'])
                or state['offset'] < 0 or state['epoch'] < 0 or state['batches'] < 0):
            raise ValueError("invalid global bucket resume cursor")
        for key, refs in list(state['pending'].items()) + list(state['ready']):
            if not refs:
                raise ValueError("invalid empty global bucket")
            for position, index in refs:
                if not 0 <= position < len(self.mixture.schedule):
                    raise ValueError("invalid global bucket reference")
                shard = self.mixture.shards[int(self.mixture.schedule[position])]
                if not shard.start <= index < shard.stop or self.key((position, index)) != key:
                    raise ValueError("invalid global bucket reference or geometry")
        if any(len(refs) != self.global_size for _, refs in state['ready']):
            raise ValueError("invalid global batch size in resume state")
        if any(len(refs) >= self.global_size for refs in state['pending'].values()):
            raise ValueError("invalid pending global bucket size")
        self.committed = state
        self.inflight.clear()


class _ShardLookaheadSampler(Sampler):
    """Attach the next shard for the same round-robin DataLoader worker.

    Lookahead is speculative only. The underlying planner still acknowledges
    exclusively delivered batches, including when restarted from a checkpoint.
    """

    def __init__(self, planner, num_workers, selected_frames=False):
        self.planner = planner
        self.distance = max(1, num_workers)
        self.selected_frames = selected_frames

    def __iter__(self):
        source = iter(self.planner)
        queue = deque()
        exhausted = False
        while True:
            while len(queue) <= self.distance and not exhausted:
                try:
                    queue.append(next(source))
                except StopIteration:
                    exhausted = True
            if not queue:
                return
            next_shard = None
            next_indices = ()
            if len(queue) > self.distance:
                next_ref = queue[self.distance][2][0]
                next_shard = int(self.planner.mixture.schedule[next_ref[0]])
                if self.selected_frames:
                    next_indices = tuple(index for position, index in queue[self.distance][2]
                                         if int(self.planner.mixture.schedule[position]) == next_shard)
            descriptor = (*queue.popleft(), next_shard)
            yield (*descriptor, next_indices) if self.selected_frames else descriptor


class _GlobalBatchDataset(Dataset):
    def __init__(self, mixture, transform, cache_shards, sample_workers=1,
                 selected_frames=False):
        self.mixture = mixture
        self.transform = transform
        self.cache_shards = cache_shards
        self.sample_workers = sample_workers
        self.selected_frames = selected_frames
        self.sample_executor = None
        self.cache = None
        self.collator = NativeTrainingCollator()

    def _sample(self, child, index, payload, key, trace=None):
        from worldscape_policy.training.data import _temporal_shape_key

        with TraceSpan('sample', parent=trace, index=index):
            with TraceSpan('decode_pack'):
                raw = child.get_from_shard(index, payload)
            with TraceSpan('transform_augment'):
                sample = self.transform(raw)
        observations, actions, states = _temporal_shape_key(sample)
        if (actions, states) != tuple(key[1:3]) or any(n != key[3] for _, n in observations):
            raise ValueError("global metadata length signature disagrees with transformed sample")
        return sample

    def __getitem__(self, descriptor):
        worker = torch.utils.data.get_worker_info()
        with TraceSpan('worker_batch', batch=descriptor[0],
                       worker=None if worker is None else worker.id,
                       lengths=list(descriptor[1]), refs=descriptor[2],
                       sample_workers=self.sample_workers) as trace:
            return self._materialize(descriptor, trace)

    def _materialize(self, descriptor, trace):
        sequence, key, refs = descriptor[:3]
        next_shard = descriptor[3] if len(descriptor) > 3 else None
        next_indices = descriptor[4] if len(descriptor) > 4 else ()
        if self.cache is None:
            cache_type = SelectedFrameShardCache if self.selected_frames else _ShardCache
            self.cache = cache_type(self.mixture, self.cache_shards)
        # Only the current batch uses these threads. Shard cache access
        # remains on the worker's foreground thread; no additional batch or
        # shard cache is retained. Per-sample augmentation owns its RNG.
        if self.sample_workers > 1 and self.sample_executor is None:
            self.sample_executor = ThreadPoolExecutor(max_workers=self.sample_workers)
        groups = [(shard_id, list(group)) for shard_id, group in groupby(
            refs, key=lambda ref: int(self.mixture.schedule[ref[0]]))]
        if self.selected_frames:
            # Sparse payloads cover this batch and previously cached selections.
            # Keep their references alive through parallel sample construction,
            # including when a later shard evicts an earlier cache entry.
            payloads = {}
            by_shard = defaultdict(list)
            for position, index in refs:
                by_shard[int(self.mixture.schedule[position])].append(index)
            for shard_id, indices in by_shard.items():
                with TraceSpan('cache_get', shard=shard_id):
                    payloads[shard_id] = self.cache.get(shard_id, indices)
            if next_shard is not None and next_indices:
                self.cache.prefetch(next_shard, next_indices)
            def materialize_selected(ref):
                shard_id = int(self.mixture.schedule[ref[0]])
                child = self.mixture.datasets[self.mixture.shards[shard_id].dataset_index]
                return self._sample(child, ref[1], payloads[shard_id], key, trace)
            if self.sample_workers > 1:
                samples = list(self.sample_executor.map(materialize_selected, refs))
            else:
                samples = list(map(materialize_selected, refs))
            with TraceSpan('collate'):
                batch = self.collator(tuple(samples))
            return sequence, key, batch
        samples = []
        for offset, (shard_id, group) in enumerate(groups):
            shard = self.mixture.shards[shard_id]
            child = self.mixture.datasets[shard.dataset_index]
            with TraceSpan('cache_get', shard=shard_id):
                payload = self.cache.get(shard_id)
            upcoming = groups[offset + 1][0] if offset + 1 < len(groups) else next_shard
            if upcoming is not None:
                self.cache.prefetch(upcoming)
            # Keep payload alive even when prefetch harvests an LRU eviction.
            # Complete this group before accessing the next shard. Ordered map
            # preserves batch order and surfaces materialization exceptions.
            def materialize(ref):
                return self._sample(child, ref[1], payload, key, trace)
            if self.sample_workers > 1:
                samples.extend(self.sample_executor.map(materialize, group))
            else:
                samples.extend(map(materialize, group))
        with TraceSpan('collate'):
            batch = self.collator(tuple(samples))
        return sequence, key, batch

    def close(self) -> None:
        if self.sample_executor is not None:
            self.sample_executor.shutdown(wait=True, cancel_futures=True)
            self.sample_executor = None
        if self.cache is not None:
            self.cache.close()
            self.cache = None


class GlobalBucketLoader:
    """Ordered delivery with prefetch-safe checkpoint state in the parent."""

    supports_exact_worker_resume = True

    def __init__(self, mixture: NativeShardedMixtureDataset,
                 transform: NativeEventTransform, batch_size: int, *, num_workers: int = 0,
                 prefetch_factor: int = 2, persistent_workers: bool = False,
                 pin_memory: bool = False, window_batches: int = 128,
                 cache_shards: int = 2, sample_workers: int = 1,
                 selected_frames: bool = False,
                 validate_distributed: bool = True) -> None:
        if not isinstance(mixture, NativeShardedMixtureDataset):
            raise ValueError("global bucketing requires a native sharded mixture")
        if (type(transform) is not NativeEventTransform
                or transform.fixed_mode != InteractionMode.INTERACTIVE
                or transform.prompt_modality_sampler is not None):
            raise ValueError("global bucketing requires fixed native interactive transform")
        if type(selected_frames) is not bool:
            raise ValueError("global bucket selected_frames must be a boolean")
        for child in mixture.datasets:
            if (not isinstance(child, RoboDojoJointHDF5Dataset)
                    or not child.temporal_packing or child.robodojo_visual_prompt != 'none'):
                raise ValueError("global bucketing requires temporal-packed RoboDojo text data")
            if selected_frames and child.cache_video_decode:
                raise ValueError("selected frames require cache_video_decode=false")
        if cache_shards <= 0:
            raise ValueError("global bucket cache_shards must be positive")
        if type(sample_workers) is not int or sample_workers < 1:
            raise ValueError("global bucket sample_workers must be a positive integer")
        self.num_workers = num_workers
        configure_trace()
        self.sampler = GlobalBucketPlanner(mixture, batch_size, window_batches)
        if validate_distributed:
            self.validate_distributed()
        self.dataset = _GlobalBatchDataset(mixture, transform, cache_shards, sample_workers,
                                          selected_frames)
        self._loader = DataLoader(
            self.dataset, batch_size=None,
            sampler=_ShardLookaheadSampler(self.sampler, num_workers, selected_frames),
            num_workers=num_workers,
            pin_memory=pin_memory, collate_fn=_identity,
            generator=torch.Generator().manual_seed(mixture.seed),
            **(dict(prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
               if num_workers else {}))
        self._iterator = None

    def validate_distributed(self) -> None:
        """Validate only after every rank has completed fallible local setup."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            mixture = self.sampler.mixture
            identities = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(
                identities, (mixture.rank, self.sampler.identity))
            if (mixture.rank != torch.distributed.get_rank()
                    or mixture.world_size != len(identities)
                    or any(rank != i or value != self.sampler.identity
                           for i, (rank, value) in enumerate(identities))):
                raise ValueError("global bucket configuration must agree across all ranks")

    def __iter__(self):
        self._iterator = iter(self._loader)
        while True:
            started = time.perf_counter()
            try:
                sequence, key, batch = next(self._iterator)
            except StopIteration:
                break
            self.sampler.acknowledge(sequence, key)
            self.sampler.last_info['data_wait_s'] = time.perf_counter() - started
            yield batch
        self.sampler.last_info['dropped'] = self.sampler.tail_counts
        if any(self.sampler.tail_counts.values()):
            print(json.dumps({'global_bucket_tail': self.sampler.tail_counts}), flush=True)
        self.sampler.finish_epoch()

    @property
    def last_batch_info(self) -> dict[str, Any]:
        return self.sampler.last_info

    def state_dict(self) -> dict[str, Any]:
        return self.sampler.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.close()
        self.sampler.load_state_dict(state)

    def close(self) -> None:
        if self._iterator is not None:
            shutdown = getattr(self._iterator, '_shutdown_workers', None)
            if shutdown is not None:
                shutdown()
            self._iterator = None
            self._loader._iterator = None
        self.dataset.close()
