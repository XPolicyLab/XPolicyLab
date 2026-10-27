"""Add prompt IDs without changing the accepted Robot80 reader or sampler."""
import copy

import torch
from torch.utils.data import Dataset

from wam.embodiment_registry import EmbodimentRegistry


class SoftPromptDataset(Dataset):
    def __init__(self, dataset, registry_path):
        self.dataset = dataset
        self.embodiment_registry = EmbodimentRegistry(registry_path)
        self.embodiment_registry.validate_package(dataset.package)
        self.fixed_validation_panel = copy.deepcopy(dataset.fixed_validation_panel)
        for entry in self.fixed_validation_panel:
            profile = dataset.package.episode(entry["episode_id"])["profile"]
            entry["embodiment_id"] = self.embodiment_registry.id_for(profile)

    def __getattr__(self, name):
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def __len__(self):
        return len(self.dataset)

    def _attach(self, item):
        result = dict(item)
        result["embodiment_id"] = torch.tensor(
            self.embodiment_registry.id_for(item["profile_id"]), dtype=torch.long)
        return result

    def __getitem__(self, index):
        return self._attach(self.dataset[index])

    def read_window(self, *args, **kwargs):
        return self._attach(self.dataset.read_window(*args, **kwargs))

    def collate_fn(self, samples):
        for sample in samples:
            value = sample.get("embodiment_id")
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.long or value.ndim != 0
                    or int(value) not in self.embodiment_registry.assigned_ids):
                raise ValueError("Expected a valid scalar int64 embodiment_id")
            if int(value) != self.embodiment_registry.id_for(sample["profile_id"]):
                raise ValueError("Prompt ID does not match the sample profile")
        return self.dataset.collate_fn(samples)
