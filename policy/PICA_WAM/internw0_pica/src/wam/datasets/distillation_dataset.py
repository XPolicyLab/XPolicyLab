"""Pretraining with one offline-supervised window per local batch."""
from __future__ import annotations

import math

import torch

from .distillation_cache import FEATURE_DIM, TeacherCache, dataset_id
from .pretrain_wam_dataset import PretrainWAMDataset


class SparseTeacherMixture:
    prefer_sequential_indices = True

    def __init__(self, full_dataset, cache, batch_size, samples_per_batch=1, seed=42):
        self.full_dataset = full_dataset
        self.cache = cache
        self.batch_size = int(batch_size)
        self.samples_per_batch = int(samples_per_batch)
        self.seed = int(seed)
        if len(cache) <= 0:
            raise ValueError("The teacher cache must contain at least one window.")
        self.teacher_stride = 104729
        while math.gcd(self.teacher_stride, len(cache)) != 1:
            self.teacher_stride += 2
        if not 0 < self.samples_per_batch <= self.batch_size:
            raise ValueError("teacher_samples_per_batch must be in [1, batch_size].")
        self.positions = {dataset_id(d.spec): i for i, d in enumerate(full_dataset.datasets)}
        if len(self.positions) != len(full_dataset.datasets):
            raise ValueError("Dataset source/name identities must be unique.")
        missing = set(cache.datasets) - self.positions.keys()
        if missing:
            raise ValueError(f"Teacher sources absent from the active mixture: {sorted(missing)[:8]}. Build a matching window plan.")

    def __getattr__(self, name):
        # Avoid recursion while unpickling spawned DataLoader workers.
        if name == "full_dataset":
            raise AttributeError(name)
        return getattr(self.full_dataset, name)

    def __len__(self):
        return len(self.full_dataset)

    def teacher_row(self, index):
        position = int(index) % self.batch_size
        if position >= self.samples_per_batch:
            return None
        return (self.seed + self.teacher_stride * (int(index) // self.batch_size) + position) % len(self.cache)

    def optimizer_stratified_replacement_index(self, index, attempt):
        batches = len(self) // self.batch_size
        if batches <= 1:
            return int(index)
        batch = (int(index) // self.batch_size + 104729 * int(attempt) + 17) % batches
        return batch * self.batch_size + int(index) % self.batch_size

    def get_items(self, indices):
        output = [None] * len(indices)
        ordinary = []
        grouped = {}
        for pos, index in enumerate(indices):
            row = self.teacher_row(index)
            if row is None:
                ordinary.append((pos, int(index)))
                continue
            name, episode, start = self.cache.identity(row)
            ds_pos = self.positions[name]
            dataset = self.full_dataset.datasets[ds_pos]
            meta = dataset.episodes_dict.get(episode)
            if meta is None or not 0 <= start < int(meta["length"]):
                raise ValueError(f"Cached window {name}/{episode}/{start} is absent from the local data index.")
            grouped.setdefault((ds_pos, episode), []).append((pos, row, start))
        for (ds_pos, episode), requests in grouped.items():
            dataset = self.full_dataset.datasets[ds_pos]
            parquet = dataset._load_episode_parquet(episode)
            text_cache = {}
            for pos, row, start in requests:
                item = dataset.get_step_item(episode, start, parquet=parquet, text_context_cache=text_cache)
                item["dataset_index"] = torch.tensor(ds_pos, dtype=torch.long)
                item["_teacher_row"] = row
                output[pos] = item
        if ordinary:
            items = self.full_dataset.get_items([idx for _, idx in ordinary])
            if len(items) != len(ordinary):
                raise RuntimeError("Mixture returned an incomplete batch.")
            for (pos, _), item in zip(ordinary, items):
                item["_teacher_row"] = None
                output[pos] = item
        return output

    def __getitem__(self, index):
        return self.get_items([index])[0]


class DistillationDataset(PretrainWAMDataset):
    def __init__(self, *args, teacher_cache, teacher_samples_per_batch=1, teacher_seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_cache = None
        if self.is_training_set:
            if (self.num_frames, self.action_size, self.global_sample_stride, self.video_size) != (33, 32, 1, (384, 256)):
                raise ValueError("4D caches require H32, stride 1 and a 384x256 canvas.")
            self.teacher_cache = TeacherCache(teacher_cache)
            self.mixed_dataset = SparseTeacherMixture(
                self.mixed_dataset, self.teacher_cache,
                self.training_batch_size_per_rank, teacher_samples_per_batch, teacher_seed,
            )
            self.prefer_sequential_indices = True

    def _pack_training_item(self, idx, item, video, **kwargs):
        output = super()._pack_training_item(idx, item, video, **kwargs)
        if self.is_training_set:
            row = item.get("_teacher_row")
            output["track_teacher_feature"] = (
                torch.zeros(1, FEATURE_DIM, dtype=torch.float16)
                if row is None else self.teacher_cache.feature(row)
            )
            output["track_teacher_valid"] = torch.tensor([row is not None], dtype=torch.bool)
        return output

    def make_validation_dataset(self, *args, **kwargs):
        result = super().make_validation_dataset(*args, **kwargs)
        result.teacher_cache = None
        return result
