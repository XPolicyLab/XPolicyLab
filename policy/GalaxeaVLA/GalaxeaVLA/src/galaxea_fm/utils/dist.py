import itertools
from torch.utils.data.distributed import DistributedSampler

class ResumableDistributedSampler(DistributedSampler):
    def __init__(self, *args, batch_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.start_batch_idx = 0  # per-rank dataloader batch index
        self._log_indices = False
        self._logger = None

    def set_start_batch(self, start_batch_idx: int):
        self.start_batch_idx = int(start_batch_idx)

    def __iter__(self):
        # super().__iter__() already returns the per-rank index stream (shuffled + padded)
        base_iter = super().__iter__()
        # Skip indices corresponding to already-consumed batches WITHOUT loading data
        skip = self.start_batch_idx * self.batch_size
        if skip > 0:
            base_iter = itertools.islice(base_iter, skip, None)

        return base_iter
