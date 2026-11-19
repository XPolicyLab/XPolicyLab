from typing import Iterator, List, Sized

import torch
from torch.utils.data import Sampler


class WeightedResumableEpochSampler(Sampler[int]):
    """Per-frame weighted, with-replacement multi-dataset sampler.

    Drop-in replacement for ResumableEpochSampler when mixing multiple datasets
    by a size-agnostic ratio. Each of the `samples_per_epoch` draws first picks
    dataset i with probability weights[i] (normalized), then a uniform frame
    within that dataset — so the mixture ratio (e.g. 2:8) is independent of how
    many frames each dataset actually holds. The emitted value is the global
    frame index the concatenated map-style dataset's __getitem__ consumes.

    Resume / epoch-seed semantics mirror ResumableEpochSampler exactly (same
    seed = seed+epoch+epoch_offset, same epoch-0 batch-offset slice) so
    Accelerate's per-rank sharding and the trainer's batch-offset resume keep
    working unchanged. With-replacement draws mean an epoch is just a fixed
    count of samples, not a single pass over the data.
    """

    def __init__(
        self,
        per_dataset_num_frames: List[int],
        weights: List[float],
        samples_per_epoch: int,
        seed: int,
        batch_size: int,
        num_processes: int,
    ):
        if len(per_dataset_num_frames) != len(weights):
            raise ValueError(
                f"per_dataset_num_frames ({len(per_dataset_num_frames)}) and weights "
                f"({len(weights)}) must be parallel."
            )
        self.per_dataset_num_frames = [int(n) for n in per_dataset_num_frames]
        if any(n <= 0 for n in self.per_dataset_num_frames):
            raise ValueError(
                f"every dataset must contribute >=1 frame; got {self.per_dataset_num_frames}."
            )
        self.samples_per_epoch = int(samples_per_epoch)
        if self.samples_per_epoch <= 0:
            raise ValueError(f"samples_per_epoch must be > 0; got {samples_per_epoch}.")
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        # Cumulative offsets: global index of (dataset i, local frame j) = offsets[i] + j.
        offsets = [0]
        for n in self.per_dataset_num_frames:
            offsets.append(offsets[-1] + n)
        self._offsets = torch.tensor(offsets[:-1], dtype=torch.long)
        self._counts = torch.tensor(self.per_dataset_num_frames, dtype=torch.long)
        self._weights = torch.tensor([float(w) for w in weights], dtype=torch.double)
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
        # Stage 1: pick a source dataset per draw with the target probabilities.
        ds = torch.multinomial(
            self._weights, self.samples_per_epoch, replacement=True, generator=g
        )
        # Stage 2: pick a uniform frame within the chosen dataset.
        counts = self._counts[ds]
        local = (torch.rand(self.samples_per_epoch, generator=g) * counts).long()
        local = torch.minimum(local, counts - 1)  # guard the rand==1.0 boundary
        indices = (self._offsets[ds] + local).tolist()
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return self.samples_per_epoch


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
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)
