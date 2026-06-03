# Copyright (C) 2026 Xiaomi Corporation.
from copy import deepcopy

from lightning import LightningDataModule
from mmengine import Config, DATASETS
from torch.utils.data import DataLoader

from mibot.data.collate.custom_collate import CustomCollate
from mibot.data.datasets.json_dataset import JsonDataset


@DATASETS.register_module()
class BaseDataModule(LightningDataModule):
    """Single training dataloader for the hard-coded JSON dataset."""

    def __init__(self, params: Config) -> None:
        super().__init__()
        self.params: Config = params
        self.batch_size: int = params.train_datasets.get("batch_size", 16)
        self.collate_fn = CustomCollate()

    def train_dataloader(self) -> DataLoader:
        train_set = JsonDataset(deepcopy(self.params))
        return DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=16,
            prefetch_factor=8,
            collate_fn=self.collate_fn,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> list:
        return []
