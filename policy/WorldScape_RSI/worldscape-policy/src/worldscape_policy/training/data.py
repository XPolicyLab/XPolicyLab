from __future__ import annotations

import importlib
import copy
import itertools
import json
import os
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
    IterableDataset,
    Sampler,
    get_worker_info,
)

from worldscape_policy.data import (
    DATASETS,
    EventSample,
    NativeEventTransform,
    NativeTrainingCollator,
    NativeVideoAugmentation,
    PromptModalitySampler,
    TrainingBatch,
    TransformedEventSample,
)
from worldscape_policy.types import InteractionMode
from worldscape_policy.data.configured_augmentation import ConfiguredVideoAugmentation


def _profile_enabled() -> bool:
    return os.environ.get("WSP_DATALOADER_PROFILE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _profile_log(event: str, payload: dict[str, Any]) -> None:
    if not _profile_enabled():
        return
    print(
        json.dumps({"dataloader_profile": event, **payload}, sort_keys=True),
        flush=True,
    )


def _profile_sample_enabled() -> bool:
    return os.environ.get("WSP_DATALOADER_PROFILE_SAMPLE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class _MapTransformedDataset(Dataset):
    def __init__(self, dataset: Any, transform: NativeEventTransform) -> None:
        self.dataset = dataset
        self.transform = transform
        group_provider = getattr(dataset, "sample_group_key", None)
        if group_provider is not None:
            self.sample_group_key = group_provider

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        if not isinstance(sample, EventSample):
            raise TypeError("registered native datasets must yield EventSample values")
        return self.transform(sample)

    def length_signature(self, index: int) -> tuple[int, ...] | None:
        signature = getattr(self.dataset, "length_signature", None)
        if signature is None:
            return None
        return tuple(int(value) for value in signature(index))

class _IterableTransformedDataset(IterableDataset):
    def __init__(self, dataset: Any, transform: NativeEventTransform) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __iter__(self) -> Iterator:
        worker = get_worker_info()
        worker_id = None if worker is None else worker.id
        sample_profile = _profile_sample_enabled()
        iterator = iter(self.dataset)
        index = 0
        yielded = 0
        while True:
            next_start = time.perf_counter()
            try:
                sample = next(iterator)
            except StopIteration:
                break
            dataset_next_s = time.perf_counter() - next_start
            if (
                worker is not None
                and not getattr(self.dataset, "partitions_workers", False)
                and index % worker.num_workers != worker.id
            ):
                index += 1
                continue
            if not isinstance(sample, EventSample):
                raise TypeError("registered native datasets must yield EventSample values")
            transform_start = time.perf_counter()
            transformed = self.transform(sample)
            transform_s = time.perf_counter() - transform_start
            if sample_profile:
                _profile_log(
                    "iterable_sample",
                    {
                        "worker_id": worker_id,
                        "source_index": index,
                        "yielded_index": yielded,
                        "dataset_next_s": round(dataset_next_s, 6),
                        "transform_s": round(transform_s, 6),
                    },
                )
            yielded += 1
            index += 1
            yield transformed


class _LengthBucketBatchSampler(Sampler[list[int]]):
    """Deterministic, resumable batches that never mix temporal geometries."""

    def __init__(
        self,
        dataset: Any,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
        rank: int,
        world_size: int,
    ) -> None:
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.offset = 0
        self.distributed = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )
        # Signatures are cheap index metadata for native readers and are filled
        # on first scheduling use. Dataset construction never decodes samples.
        self.signatures: dict[int, tuple[int, ...]] = {}

    def _signature(self, index: int) -> tuple[int, ...]:
        cached = self.signatures.get(index)
        if cached is not None:
            return cached
        provider = getattr(self.distributed.dataset, "length_signature", None)
        signature = None if provider is None else provider(index)
        if signature is None:
            # Without a metadata contract, singleton buckets are the only way
            # to guarantee that unknown temporal geometries are never mixed
            # without decoding each sample in the sampler.
            signature = (-1, index)
        result = tuple(int(value) for value in signature)
        self.signatures[index] = result
        return result

    def _batches(self) -> list[list[int]]:
        self.distributed.set_epoch(self.epoch)
        group_provider = getattr(
            self.distributed.dataset, "sample_group_key", None
        )
        if group_provider is not None:
            grouped: dict[
                tuple[int, ...], dict[object, list[int]]
            ] = defaultdict(lambda: defaultdict(list))
            for index in self.distributed:
                value = int(index)
                grouped[self._signature(value)][group_provider(value)].append(
                    value
                )
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            batches: list[list[int]] = []
            for signature in sorted(grouped):
                groups = list(grouped[signature].values())
                if self.shuffle and len(groups) > 1:
                    order = torch.randperm(
                        len(groups), generator=generator
                    ).tolist()
                    groups = [groups[index] for index in order]
                indices = [
                    index
                    for group in groups
                    for index in group
                ]
                batches.extend(
                    indices[start : start + self.batch_size]
                    for start in range(0, len(indices), self.batch_size)
                )
            return batches
        buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for index in self.distributed:
            buckets[self._signature(int(index))].append(int(index))
        batches = [
            indices[start : start + self.batch_size]
            for signature in sorted(buckets)
            for indices in (buckets[signature],)
            for start in range(0, len(indices), self.batch_size)
        ]
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[index] for index in order]
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._batches()
        for position in range(self.offset, len(batches)):
            self.offset = position + 1
            yield batches[position]
        self.epoch += 1
        self.offset = 0

    def __len__(self) -> int:
        return len(self._batches())

    def state_dict(self) -> dict[str, int]:
        return {"version": 1, "epoch": self.epoch, "offset": self.offset}

    def load_state_dict(self, state: dict[str, int]) -> None:
        if int(state.get("version", -1)) != 1:
            raise ValueError("unsupported length-bucket sampler state")
        self.epoch = int(state["epoch"])
        self.offset = int(state["offset"])
        if self.epoch < 0 or not 0 <= self.offset <= len(self._batches()):
            raise ValueError("invalid length-bucket sampler epoch/offset")


class _LengthBucketedBatchDataset(IterableDataset):
    """Batch transformed samples only when unmasked temporal shapes match."""

    partitions_workers = True

    def __init__(self, dataset: Any, batch_size: int, *, shuffle: bool, seed: int):
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.cursor = 0
        self.buckets: dict[tuple[object, ...], list[TransformedEventSample]] = {}
        self.collator = NativeTrainingCollator()

    def __iter__(self) -> Iterator[TrainingBatch]:
        profile = _profile_enabled()
        worker = get_worker_info()
        worker_id = None if worker is None else worker.id
        map_style = hasattr(self.dataset, "__len__") and hasattr(
            self.dataset, "__getitem__"
        )
        if map_style:
            indices = torch.arange(len(self.dataset))
            if self.shuffle:
                indices = indices[
                    torch.randperm(
                        len(indices),
                        generator=torch.Generator().manual_seed(self.seed),
                    )
                ]
            source = (
                self.dataset[int(index)]
                for index in indices[self.cursor :].tolist()
            )
        else:
            source = iter(self.dataset)
            if self.cursor and _stateful_dataset(self.dataset) is None:
                source = itertools.islice(source, self.cursor, None)
        source_iter = iter(source)
        batch_index = 0
        fetched_since_yield = 0
        fetch_time_since_yield = 0.0
        while True:
            fetch_start = time.perf_counter()
            try:
                sample = next(source_iter)
            except StopIteration:
                break
            fetch_time_since_yield += time.perf_counter() - fetch_start
            fetched_since_yield += 1
            if not isinstance(sample, TransformedEventSample):
                raise TypeError(
                    "length bucketing requires TransformedEventSample values"
                )
            self.cursor += 1
            key = _temporal_shape_key(sample)
            bucket = self.buckets.setdefault(key, [])
            bucket.append(sample)
            if len(bucket) == self.batch_size:
                ready = tuple(bucket)
                del self.buckets[key]
                collate_start = time.perf_counter()
                batch = self.collator(ready)
                collate_time = time.perf_counter() - collate_start
                if profile:
                    _profile_log(
                        "length_bucket_batch",
                        {
                            "worker_id": worker_id,
                            "batch_index": batch_index,
                            "cursor": self.cursor,
                            "samples_fetched": fetched_since_yield,
                            "fetch_transform_s": round(fetch_time_since_yield, 6),
                            "collate_s": round(collate_time, 6),
                            "bucket_size": len(ready),
                            "bucket_key": repr(key),
                        },
                    )
                batch_index += 1
                fetched_since_yield = 0
                fetch_time_since_yield = 0.0
                yield batch
        pending = tuple(self.buckets.items())
        self.buckets.clear()
        if map_style or _stateful_dataset(self.dataset) is None:
            self.cursor = 0
        for _, bucket in pending:
            if bucket:
                collate_start = time.perf_counter()
                batch = self.collator(tuple(bucket))
                collate_time = time.perf_counter() - collate_start
                if profile:
                    _profile_log(
                        "length_bucket_partial",
                        {
                            "worker_id": worker_id,
                            "batch_index": batch_index,
                            "cursor": self.cursor,
                            "samples_fetched": fetched_since_yield,
                            "fetch_transform_s": round(fetch_time_since_yield, 6),
                            "collate_s": round(collate_time, 6),
                            "bucket_size": len(bucket),
                        },
                    )
                batch_index += 1
                fetched_since_yield = 0
                fetch_time_since_yield = 0.0
                yield batch

    def state_dict(self) -> dict[str, Any]:
        child = _stateful_dataset(self.dataset)
        child_state = (
            child.state_dict()
            if child is not None
            else None
        )
        return {
            "version": 1,
            "cursor": self.cursor,
            "buckets": self.buckets,
            "child": child_state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("kind") is not None:
            raise ValueError("length-bucket state mismatch; exact resume requires the same loader path")
        if int(state.get("version", -1)) != 1:
            raise ValueError("unsupported length-bucket state version")
        self.cursor = int(state["cursor"])
        self.buckets = dict(state["buckets"])
        child_state = state.get("child")
        if child_state is not None:
            child = _stateful_dataset(self.dataset)
            if child is None:
                raise ValueError("bucket state contains unsupported child state")
            child.load_state_dict(child_state)


def _temporal_shape_key(sample: TransformedEventSample) -> tuple[object, ...]:
    return (
        tuple(
            (name, int(value.shape[0]))
            for name, value in sorted(sample.observations.items())
        ),
        int(sample.actions.shape[0]),
        int(sample.robot_state.shape[0]),
    )


def _stateful_dataset(source: Any) -> Any | None:
    while source is not None:
        if hasattr(source, "state_dict") and hasattr(source, "load_state_dict"):
            return source
        child = getattr(source, "dataset", None)
        if child is source:
            break
        source = child
    return None


def _identity(value: TrainingBatch) -> TrainingBatch:
    return value


@dataclass
class _SnapshotLoaderFactory:
    """Pickleable CPU-only candidate builder, including explicit global ranks.

    Preparation may run in a bounded subprocess with no distributed group. The
    constructor must not infer that subprocess's rank or reseed the sample reader.
    """

    dataset_name: str
    dataset_kwargs: dict[str, Any]
    transform: NativeEventTransform
    loader_options: dict[str, Any]
    rank: int
    world_size: int
    dataset_plugin: str | None = None
    isolated_preparation = True

    def __call__(self, index_path, planner_seed, validate_distributed=False):
        from worldscape_policy.training.global_buckets import GlobalBucketLoader

        if self.dataset_plugin:
            importlib.import_module(self.dataset_plugin)
        options = copy.deepcopy(self.dataset_kwargs)
        options.update(precomputed_index_path=str(index_path), rank=self.rank,
                       world_size=self.world_size)
        candidate = DATASETS.create(self.dataset_name, **options)
        candidate.seed = int(planner_seed)
        candidate.schedule = candidate._make_schedule()
        return GlobalBucketLoader(
            candidate, self.transform, validate_distributed=validate_distributed,
            **self.loader_options,
        )


def build_registered_loader(
    *,
    dataset_plugin: str | None = None,
    dataset_name: str,
    mode: InteractionMode | str,
    batch_size: int,
    dataset_kwargs: dict[str, Any] | None = None,
    shuffle: bool = False,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    pin_memory: bool = False,
    seed: int = 0,
    augment_video: bool = False,
    raw_video_observation_key: str | None = None,
    augment_crop: bool = True,
    augment_hue: bool = True,
    augmentation_hue: float = 0.08,
    video_augmentation_config: dict[str, Any] | None = None,
    training: bool = True,
    wo_norm: bool = True,
    action_mode: str = "eef",
    relative_action: bool = False,
    action_dim_mask: Sequence[float | bool] | None = None,
    normalization_statistics_path: str | None = None,
    normalization_clip_range: Sequence[float] = (-5.0, 5.0),
    prompt_modalities: Sequence[str] | None = None,
    prompt_modality_probabilities: Sequence[float] | None = None,
    bucket_by_length: bool = False,
    metadata_bucket_cache_shards: int = 2,
    global_bucket_by_length: bool = False,
    global_bucket_window_batches: int = 128,
    global_bucket_sample_workers: int = 1,
    global_bucket_selected_frames: bool = False,
    hotload: Mapping[str, Any] | None = None,
) -> DataLoader:
    """Build a native loader with fixed interaction mode and mixed WAM conditions."""
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if prefetch_factor < 1:
        raise ValueError("prefetch_factor must be positive")
    if not 0.0 <= augmentation_hue <= 0.5:
        raise ValueError("augmentation_hue must be in [0, 0.5]")
    hotload_options = dict(hotload or {})
    unknown = set(hotload_options) - {"enabled", "manifest_dir", "journal_dir", "prepare_timeout_seconds"}
    if unknown:
        raise ValueError(f"unknown data_loader.hotload options: {sorted(unknown)}")
    hotload_enabled = hotload_options.get("enabled", False)
    if type(hotload_enabled) is not bool:
        raise TypeError("data_loader.hotload.enabled must be a boolean")
    if hotload_enabled:
        if not global_bucket_by_length:
            raise ValueError("RoboDojo hot loading requires global_bucket_by_length=true")
        if not training:
            raise ValueError("RoboDojo hot loading is only supported for training")
        for key in ("manifest_dir", "journal_dir"):
            if not hotload_options.get(key):
                raise ValueError(f"data_loader.hotload.{key} is required")
        for key in ("precomputed_index_path", "stats_path"):
            if not (dataset_kwargs or {}).get(key):
                raise ValueError(f"RoboDojo hot loading requires dataset_kwargs.{key}")
    worker_options = (
        {
            "prefetch_factor": prefetch_factor,
            "persistent_workers": persistent_workers,
        }
        if num_workers > 0
        else {}
    )

    video_augmentation = (
        ConfiguredVideoAugmentation(video_augmentation_config, training=training)
        if video_augmentation_config is not None
        else NativeVideoAugmentation(
            training=training,
        )
    )
    if dataset_plugin is not None:
        if not isinstance(dataset_plugin, str) or not dataset_plugin.strip():
            raise ValueError("dataset_plugin must be a non-empty module name or null")
        try:
            importlib.import_module(dataset_plugin)
        except Exception as exc:
            raise RuntimeError(
                f"failed to import native dataset plugin {dataset_plugin!r}; "
                "ensure it is importable and registers DATASETS entries"
            ) from exc
    try:
        dataset = DATASETS.create(dataset_name, **(dataset_kwargs or {}))
    except KeyError as exc:
        raise RuntimeError(
            f"{'dataset plugin ' + repr(dataset_plugin) if dataset_plugin else 'built-in datasets'} "
            f"did not register dataset {dataset_name!r}; registered datasets: "
            f"{', '.join(DATASETS.names()) or '<none>'}"
        ) from exc
    transform = NativeEventTransform(
        fixed_mode=mode,
        prompt_modality_sampler=(
            PromptModalitySampler(
                modalities=tuple(prompt_modalities),
                probabilities=(
                    None
                    if prompt_modality_probabilities is None
                    else tuple(float(value) for value in prompt_modality_probabilities)
                ),
            )
            if prompt_modalities is not None
            else None
        ),
        video_augmentation=video_augmentation if augment_video else None,
        seed=seed,
        wo_norm=wo_norm,
        action_mode=action_mode,
        relative_action=relative_action,
        action_dim_mask=action_dim_mask,
        normalization_statistics_path=normalization_statistics_path,
        normalization_clip_range=normalization_clip_range,
        raw_video_observation_key=raw_video_observation_key,
    )
    if isinstance(dataset, IterableDataset) or not (
        hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")
    ):
        transformed = _IterableTransformedDataset(dataset, transform)
        if shuffle:
            raise ValueError("shuffle is unsupported for iterable native datasets")
    else:
        transformed = _MapTransformedDataset(dataset, transform)
    if global_bucket_by_length:
        from worldscape_policy.training.global_buckets import GlobalBucketLoader

        loader_options = dict(batch_size=batch_size, num_workers=num_workers,
            prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
            pin_memory=pin_memory, window_batches=global_bucket_window_batches,
            cache_shards=metadata_bucket_cache_shards,
            sample_workers=global_bucket_sample_workers,
            selected_frames=global_bucket_selected_frames,
        )
        loader = GlobalBucketLoader(dataset, transform, **loader_options)
        if not hotload_enabled:
            return loader
        from worldscape_policy.training.hotload import DynamicGlobalBucketLoader

        base_kwargs = copy.deepcopy(dict(dataset_kwargs or {}))

        loader_factory = _SnapshotLoaderFactory(
            dataset_name, base_kwargs, transform, loader_options,
            rank=dataset.rank, world_size=dataset.world_size, dataset_plugin=dataset_plugin,
        )

        return DynamicGlobalBucketLoader(
            loader, loader_factory=loader_factory,
            snapshot_dir=str(hotload_options["manifest_dir"]),
            journal_dir=str(hotload_options["journal_dir"]),
            base_index_path=str(base_kwargs["precomputed_index_path"]),
            stats_path=str(base_kwargs["stats_path"]), seed=dataset.seed,
            prepare_timeout_seconds=hotload_options.get("prepare_timeout_seconds", 120.0),
        )
    if bucket_by_length and isinstance(transformed, IterableDataset):
        batched = _LengthBucketedBatchDataset(
            transformed,
            batch_size,
            shuffle=shuffle,
            seed=seed,
        )
        return DataLoader(
            batched,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **worker_options,
            generator=torch.Generator().manual_seed(seed),
            collate_fn=_identity,
        )
    if not isinstance(transformed, IterableDataset):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        else:
            rank, world_size = 0, 1
        return DataLoader(
            transformed,
            batch_sampler=_LengthBucketBatchSampler(
                transformed,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                rank=rank,
                world_size=world_size,
            ),
            num_workers=num_workers,
            pin_memory=pin_memory,
            **worker_options,
            generator=torch.Generator().manual_seed(seed),
            collate_fn=NativeTrainingCollator(),
        )
    return DataLoader(
        transformed,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **worker_options,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=NativeTrainingCollator(),
    )


__all__ = ["build_registered_loader"]
