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
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)


class HistoryAwareResumableEpochSampler(ResumableEpochSampler):
    """Shuffle history-length-homogeneous batches across the whole epoch."""

    def __init__(
        self,
        dataset: Sized,
        seed: int,
        batch_size: int,
        num_processes: int,
    ):
        super().__init__(
            dataset=dataset,
            seed=seed,
            batch_size=batch_size,
            num_processes=num_processes,
        )
        get_valid_lens = getattr(
            dataset, "get_video_history_valid_len_for_all_indices", None
        )
        if get_valid_lens is None:
            raise TypeError(
                "`HistoryAwareResumableEpochSampler` requires dataset method "
                "`get_video_history_valid_len_for_all_indices()`."
            )
        valid_lens = get_valid_lens()
        if not isinstance(valid_lens, torch.Tensor):
            valid_lens = torch.as_tensor(valid_lens)
        if valid_lens.ndim != 1 or int(valid_lens.shape[0]) != len(dataset):
            raise ValueError(
                "`get_video_history_valid_len_for_all_indices()` must return "
                f"[len(dataset)], got shape {tuple(valid_lens.shape)}."
            )
        self.valid_history_lens = valid_lens.to(device="cpu", dtype=torch.long)

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        perm = torch.randperm(len(self.dataset), generator=g)
        valid_lens = self.valid_history_lens
        max_len = int(valid_lens.max().item()) if valid_lens.numel() else 0
        chunk_size = max(self.batch_size * self.num_processes, 1)

        chunks: list[list[int]] = []
        for bucket_id in range(max_len + 1):
            bucket_indices = perm[valid_lens[perm] == int(bucket_id)]
            for start in range(0, int(bucket_indices.numel()), chunk_size):
                chunk = bucket_indices[start : start + chunk_size].tolist()
                if chunk:
                    chunks.append(chunk)

        if chunks:
            chunk_order = torch.randperm(len(chunks), generator=g).tolist()
            ordered = [
                sample_idx
                for chunk_idx in chunk_order
                for sample_idx in chunks[int(chunk_idx)]
            ]
        else:
            ordered = []

        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = (
                self.resume_batch_offset * self.batch_size * self.num_processes
            )
            ordered = ordered[sample_offset:]
        return iter(ordered)
