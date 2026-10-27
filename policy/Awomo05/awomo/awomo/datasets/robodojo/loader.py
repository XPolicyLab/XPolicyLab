# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense window sampling with contiguous per-rank global-batch partitions."""

import random
from collections.abc import Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from torchdata.stateful_dataloader import StatefulDataLoader

from .dataset import RoboDojoHDF5Dataset


class DenseBatchIterator(Iterator[list[int]]):
    """Checkpoint the next batch independently of DataLoader prefetching."""

    def __init__(self, batches: torch.Tensor, position: int):
        self.batches = batches
        self.position = position

    def __iter__(self):
        return self

    def __next__(self):
        if self.position >= len(self.batches):
            raise StopIteration
        batch = self.batches[self.position].tolist()
        self.position += 1
        return batch

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        position = int(state["position"])
        if not 0 <= position <= len(self.batches):
            raise ValueError("Invalid dense-sampler iterator position")
        self.position = position


class RoboDojoDataLoader(StatefulDataLoader):
    """Restore worker snapshots without advancing the training RNG at restart."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resume_pending = False

    def _settings(self):
        sampler = self.batch_sampler
        return {
            "version": 1,
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor,
            "persistent_workers": self.persistent_workers,
            "sampler": {name: getattr(sampler, name) for name in (
                "length", "seed", "local_batch_size", "rank", "world_size")},
        }

    def state_dict(self):
        state = super().state_dict()
        state["_awomo_dataloader"] = self._settings()
        return state

    def load_state_dict(self, state):
        state = dict(state)
        if state.pop("_awomo_dataloader", None) != self._settings():
            raise ValueError("DataLoader worker and sampler settings must match the checkpoint")
        super().load_state_dict(state)
        self._resume_pending = True

    def __iter__(self):
        # Restarting workers generates a base seed even when their saved state
        # will replace it. That extra draw must not shift model noise sampling.
        rng = torch.get_rng_state() if self._resume_pending and self.num_workers else None
        try:
            iterator = super().__iter__()
        finally:
            if rng is not None:
                torch.set_rng_state(rng)
        self._resume_pending = False
        return iterator


class DenseBatchSampler(Sampler[list[int]]):
    """Shuffle globally, then partition contiguous batches across ranks."""

    def __init__(self, length: int, seed: int, local_batch_size: int, rank: int, world_size: int) -> None:
        self.length = length
        self.seed = seed
        self.local_batch_size = local_batch_size
        self.global_batch_size = local_batch_size * world_size
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        self.data_step = 0

    def __len__(self) -> int:
        return (self.length + self.global_batch_size - 1) // self.global_batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_data_step(self, data_step: int) -> None:
        self.data_step = int(data_step)

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "data_step": self.data_step}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.epoch = state["epoch"]
        self.data_step = state["data_step"]

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.length, generator=generator)
        padding = len(self) * self.global_batch_size - self.length
        if padding:
            indices = torch.cat(
                [
                    indices,
                    indices.repeat((padding + self.length - 1) // self.length)[:padding],
                ]
            )
        local = indices.reshape(-1, self.world_size, self.local_batch_size)[:, self.rank]
        offset = max(0, self.data_step - self.epoch * len(self))
        return DenseBatchIterator(local, offset)


def seed_worker(worker_id: int) -> None:
    rank = 0
    if dist.is_initialized():
        rank = dist.get_rank()
    base_seed = torch.initial_seed() - worker_id
    seed_seq = np.random.SeedSequence([base_seed, worker_id, rank])
    np.random.seed(seed_seq.generate_state(4))
    torch_seed_seq, random_seed_seq, _ = seed_seq.spawn(3)
    torch.manual_seed(torch_seed_seq.generate_state(1, dtype=np.uint64)[0])
    random_seed = (random_seed_seq.generate_state(2, dtype=np.uint64).astype(list) * [1 << 64, 1]).sum()
    random.seed(random_seed)


def build_robodojo_dataloader(
    dataset: RoboDojoHDF5Dataset,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    seed: int,
    dp_rank: int,
    dp_world_size: int,
    batch_size: int,
) -> tuple[RoboDojoDataLoader, DenseBatchSampler]:
    assert isinstance(dataset, RoboDojoHDF5Dataset)
    sampler = DenseBatchSampler(len(dataset), seed, batch_size, dp_rank, dp_world_size)
    kwargs = dict(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    if num_workers:
        kwargs.update(prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
    return RoboDojoDataLoader(**kwargs), sampler
