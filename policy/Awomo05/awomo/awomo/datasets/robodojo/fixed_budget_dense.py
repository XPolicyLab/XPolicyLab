# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-update window sampler with shuffling, tail padding, rank partitioning and resume."""

import json
import logging

from .dataset import RoboDojoHDF5Dataset
from .loader import DenseBatchSampler, RoboDojoDataLoader, seed_worker

logger = logging.getLogger(__name__)


class FixedBudgetDenseBatchSampler(DenseBatchSampler):
    """Pad the shuffled window sequence to the configured optimizer-step budget."""

    def __init__(
        self,
        length: int,
        seed: int,
        local_batch_size: int,
        rank: int,
        world_size: int,
        steps_per_epoch: int,
    ) -> None:
        self.steps_per_epoch = int(steps_per_epoch)
        super().__init__(length, seed, local_batch_size, rank, world_size)
        assert self.steps_per_epoch == 6755 and self.global_batch_size == 256
        assert 0 < self.length <= self.steps_per_epoch * self.global_batch_size

    def __len__(self) -> int:
        return self.steps_per_epoch


def build_fixed_budget_dataloader(
    dataset: RoboDojoHDF5Dataset,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    seed: int,
    dp_rank: int,
    dp_world_size: int,
    batch_size: int,
    steps_per_epoch: int,
) -> tuple[RoboDojoDataLoader, FixedBudgetDenseBatchSampler]:
    assert isinstance(dataset, RoboDojoHDF5Dataset) and dataset.window_stride == 1
    assert len(dataset) == 1724338 and dp_world_size == 32 and batch_size == 8
    sampler = FixedBudgetDenseBatchSampler(
        len(dataset), seed, batch_size, dp_rank, dp_world_size, steps_per_epoch
    )
    logger.info(
        "DENSE32_SAMPLING_READY %s",
        json.dumps(
            {
                "windows": len(dataset),
                "draws": len(sampler) * 256,
                "padding": len(sampler) * 256 - len(dataset),
            }
        ),
    )
    kwargs = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": num_workers,
        "collate_fn": dataset.collate_fn,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
    }
    if num_workers:
        kwargs.update(
            prefetch_factor=prefetch_factor, persistent_workers=persistent_workers
        )
    return RoboDojoDataLoader(**kwargs), sampler
