from __future__ import annotations

from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
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
        indices = self._weighted_indices(g)
        if indices is None:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)

    def _weighted_indices(self, generator: torch.Generator) -> list[int] | None:
        weights = getattr(self.dataset, "dataset_weights", None)
        lengths = getattr(self.dataset, "dataset_lengths", None)
        cum_lengths = getattr(self.dataset, "dataset_cum_lengths", None)
        if weights is None or lengths is None or cum_lengths is None:
            return None

        weights = torch.as_tensor(weights, dtype=torch.float32, device="cpu")
        if weights.numel() == 0:
            return None
        if weights.sum() <= 0:
            raise ValueError("dataset_weights must contain at least one positive value.")
        weights = weights / weights.sum()
        lengths = [int(length) for length in lengths]
        cum_lengths = [int(length) for length in cum_lengths]
        if len(lengths) != weights.numel() or len(cum_lengths) != weights.numel():
            raise ValueError("dataset_weights, dataset_lengths, and dataset_cum_lengths must have matching lengths.")

        sampled_dataset_indices = torch.multinomial(
            weights,
            num_samples=len(self.dataset),
            replacement=True,
            generator=generator,
        ).tolist()
        indices = []
        for dataset_idx in sampled_dataset_indices:
            local_len = lengths[dataset_idx]
            if local_len <= 0:
                raise ValueError(f"Cannot sample from empty dataset index {dataset_idx}.")
            local_idx = int(torch.randint(local_len, (1,), generator=generator).item())
            global_start = 0 if dataset_idx == 0 else cum_lengths[dataset_idx - 1]
            indices.append(global_start + local_idx)
        return indices
