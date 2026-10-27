"""Mixed-task robot batches with per-task statistics and masked 14D actions."""

from functools import lru_cache

import torch
from torch.utils.data import ConcatDataset

from prismwam.datasets.real_world_dataset import RealWorldRGBDepthActionDataset


def pad_bimanual(sample: dict) -> dict:
    for name in ("action", "proprio"):
        value = sample[name]
        if value.ndim != 2 or value.shape[-1] not in (7, 14):
            raise ValueError(f"{name} must have shape [H,7] or [H,14]")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite {name}")
        dim = value.shape[-1]
        sample[f"{name}_dim_mask"] = torch.arange(14) < dim
        sample[name] = torch.nn.functional.pad(value, (0, 14 - dim))
    return sample


class RealWorldTaskDataset(RealWorldRGBDepthActionDataset):
    def __init__(self, *args, **kwargs) -> None:
        # The action-statistics signature hashes root_dir; keep the configured
        # path rather than its resolved target so that moving a source behind
        # the same symlink does not invalidate the statistics.
        kwargs.setdefault("preserve_root_alias", True)
        super().__init__(*args, **kwargs)

    @lru_cache(maxsize=None)
    def load_episode_vectors(self, episode):
        return super().load_episode_vectors(episode)

    def __getitem__(self, index):
        return pad_bimanual(super().__getitem__(index))


class RealWorldMixedDataset(ConcatDataset):
    """Concatenate retained windows; expose task groups for the chosen sampler."""

    def __init__(self, sources):
        datasets = [RealWorldTaskDataset(**dict(source)) for source in sources]
        super().__init__(datasets)
        self.task_names = tuple(task for ds in self.datasets for task in ds.task_names)
        if len(set(self.task_names)) != len(self.task_names):
            raise ValueError("Duplicate task sources")

    def prompts(self):
        return sorted({prompt for ds in self.datasets for prompt in ds.prompts()})

    def task_index_groups(self):
        groups, offset = {}, 0
        for ds in self.datasets:
            groups.update(
                {
                    task: [offset + i for i in indices]
                    for task, indices in ds.task_index_groups().items()
                }
            )
            offset += len(ds)
        return groups
