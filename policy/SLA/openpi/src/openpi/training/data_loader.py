from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import functools
import logging
import multiprocessing
import os
import pathlib
import queue
import threading
import time
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.distributed as _distributed
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __getitems__(self, indices: Sequence[SupportsIndex]) -> list[T_co]:
        getitems = getattr(self._dataset, "__getitems__", None)
        samples = getitems(indices) if getitems is not None else [self._dataset[index] for index in indices]
        return [self._transform(sample) for sample in samples]

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


class _FastLocalLeRobotDataset(lerobot_dataset.LeRobotDataset):
    """Read an immutable local LeRobot dataset without unused training-time validation."""

    def __init__(self, repo_id: str, info: dict, *, delta_timestamps: dict[str, list[float]]):
        torch.utils.data.Dataset.__init__(self)
        root = pathlib.Path(repo_id).expanduser().resolve()
        self.repo_id = str(root)
        self.root = root
        self.image_transforms = None
        self.delta_timestamps = delta_timestamps
        self.episodes = None
        self.tolerance_s = 1e-4
        self.revision = lerobot_dataset.CODEBASE_VERSION
        self.video_backend = lerobot_dataset.get_safe_default_codec()
        self.delta_indices = None
        self.image_writer = None
        self.episode_buffer = None

        # Training uses external normalization stats, so per-episode stats need not be parsed here.
        meta = lerobot_dataset.LeRobotDatasetMetadata.__new__(lerobot_dataset.LeRobotDatasetMetadata)
        meta.repo_id = self.repo_id
        meta.revision = self.revision
        meta.root = root
        meta.info = info
        meta.tasks, meta.task_to_task_index = lerobot_dataset.load_tasks(root)
        meta.episodes = lerobot_dataset.load_episodes(root)
        meta.stats = {}
        meta.episodes_stats = {}
        self.meta = meta

        self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = lerobot_dataset.get_episode_data_index(self.meta.episodes, self.episodes)
        lerobot_dataset.check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
        self.delta_indices = lerobot_dataset.get_delta_indices(self.delta_timestamps, self.fps)
        self._query_datasets = {
            key: self.hf_dataset.with_format("torch", columns=[key])
            for key in self.delta_indices
            if key not in self.meta.video_keys
        }

    def _decode_video_frames(self, video_key: str, requested_timestamps: list[float], ep_idx: int) -> torch.Tensor:
        video_path = self.root / self.meta.get_video_file_path(ep_idx, video_key)
        if self.video_backend != "torchcodec":
            return lerobot_dataset.decode_video_frames(
                video_path, requested_timestamps, self.tolerance_s, self.video_backend
            )

        from torchcodec.decoders import VideoDecoder  # noqa: PLC0415 - optional backend

        decoder = VideoDecoder(video_path, device="cpu", seek_mode="approximate")
        average_fps = decoder.metadata.average_fps
        frame_indices = [round(timestamp * average_fps) for timestamp in requested_timestamps]
        clamped_indices = [min(max(index, 0), len(decoder) - 1) for index in frame_indices]
        if clamped_indices != frame_indices:
            logging.warning("Clamped video frame indices: video=%s", video_path)
        frame_batch = decoder.get_frames_at(indices=clamped_indices)
        # Preserve dataset compatibility: clamp boundary frames and validate frame-grid times,
        # not the original requests, which may contain timestamp offsets.
        query = torch.tensor([index / average_fps for index in clamped_indices], dtype=torch.float32)
        distances = (query - frame_batch.pts_seconds.to(dtype=torch.float32, device="cpu")).abs()
        if not bool((distances < self.tolerance_s).all()):
            raise AssertionError(
                f"Video timestamps violate tolerance {self.tolerance_s}: "
                f"distance={distances.tolist()} video={video_path}"
            )
        return frame_batch.data.to(dtype=torch.float32) / 255

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        return {
            video_key: self._decode_video_frames(video_key, requested_timestamps, ep_idx).squeeze(0)
            for video_key, requested_timestamps in query_timestamps.items()
        }

    def __getitems__(self, indices: Sequence[SupportsIndex]) -> list[dict]:
        if not indices:
            return []
        items = []
        item_query_indices: list[dict[str, list[int | bool]]] = []
        grouped: dict[tuple[int, str], list[tuple[int, list[float]]]] = {}
        for item_index, index in enumerate(indices):
            index_value = index.__index__()
            item = self.hf_dataset[index_value]
            ep_idx = item["episode_index"].item()
            query_indices = {}
            if self.delta_indices is not None:
                query_indices, padding = self._get_query_indices(index_value, ep_idx)
                item = {**item, **padding}
            query_timestamps = (
                self._get_query_timestamps(item["timestamp"].item(), query_indices) if self.meta.video_keys else {}
            )
            items.append(item)
            item_query_indices.append(query_indices)
            for video_key, timestamps in query_timestamps.items():
                grouped.setdefault((ep_idx, video_key), []).append((item_index, timestamps))

        for key, query_dataset in self._query_datasets.items():
            flat_indices = [index for query in item_query_indices for index in query[key]]
            values = query_dataset[flat_indices][key]
            offset = 0
            for item, query in zip(items, item_query_indices, strict=True):
                stop = offset + len(query[key])
                item[key] = values[offset:stop]
                offset = stop

        for (ep_idx, video_key), requests in grouped.items():
            timestamps = [timestamp for _, requested in requests for timestamp in requested]
            frames = self._decode_video_frames(video_key, timestamps, ep_idx)
            offset = 0
            for item_index, requested in requests:
                stop = offset + len(requested)
                items[item_index][video_key] = frames[offset:stop].squeeze(0)
                offset = stop

        for item in items:
            if self.image_transforms is not None:
                for camera_key in self.meta.camera_keys:
                    item[camera_key] = self.image_transforms(item[camera_key])
            item["task"] = self.meta.tasks[item["task_index"].item()]
        return items


class WeightedMixedDataset(Dataset):
    """
    Mixes multiple datasets with configurable source weights.
    Each dataset is sampled with probability proportional to ``len(dataset) * weight``.
    """
    def __init__(self, datasets: Sequence[Dataset], weights: Sequence[float], *, seed: int = 0):
        weight_array = np.asarray(weights)
        dataset_lengths = np.asarray([len(dataset) for dataset in datasets])
        effective_weights = dataset_lengths * weight_array
        total_weight = effective_weights.sum()

        self._datasets = datasets
        self._weights = effective_weights / total_weight
        self._length = int(dataset_lengths.sum())
        self._seed = seed
        self._epoch = 0
        self._read_chunk_size = int(os.environ.get("OPENPI_MIXED_READ_CHUNK_SIZE", "1"))
        if self._read_chunk_size < 1:
            raise ValueError("OPENPI_MIXED_READ_CHUNK_SIZE must be positive.")
        self._read_threads = 0
        self._executor: ThreadPoolExecutor | None = None

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __getitem__(self, index: SupportsIndex) -> dict:
        index_value = index.__index__()
        if index_value < 0:
            index_value %= self._length

        dataset_index, sample_index = self._sample_location(index_value)
        return self._datasets[dataset_index][sample_index]

    def _sample_location(self, index: int) -> tuple[int, int]:
        rng = np.random.default_rng(np.random.SeedSequence([self._seed, self._epoch, index]))
        dataset_index = int(rng.choice(len(self._datasets), p=self._weights))
        return dataset_index, int(rng.integers(len(self._datasets[dataset_index])))

    def __getitems__(self, indices: Sequence[SupportsIndex]) -> list[dict]:
        if not indices:
            return []

        chunks: list[tuple[int, tuple[int, ...]]] = []
        for chunk_start in range(0, len(indices), self._read_chunk_size):
            chunk = indices[chunk_start : chunk_start + self._read_chunk_size]
            component_index, start = self._sample_location(chunk[0].__index__())
            dataset = self._datasets[component_index]
            sample_indices = tuple((start + offset) % len(dataset) for offset in range(len(chunk)))
            chunks.append((component_index, sample_indices))

        def load_chunk(chunk: tuple[int, tuple[int, ...]]) -> list[dict]:
            component_index, sample_indices = chunk
            dataset = self._datasets[component_index]
            getitems = getattr(dataset, "__getitems__", None)
            return getitems(sample_indices) if getitems is not None else [dataset[index] for index in sample_indices]

        if self._read_threads <= 1 or len(chunks) == 1:
            results = [load_chunk(chunk) for chunk in chunks]
        else:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=self._read_threads, thread_name_prefix="mixed-data")
            results = list(self._executor.map(load_chunk, chunks))
        return [item for chunk_items in results for item in chunk_items]

    def set_read_threads(self, count: int) -> None:
        if count < 0:
            raise ValueError(f"Read thread count must be non-negative, got {count}.")
        if count != self._read_threads:
            self.close()
            self._read_threads = count

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def __len__(self) -> int:
        return self._length


class _PrefetchIterator:
    """Background-prefetch an iterator into a bounded queue."""

    _END = object()

    def __init__(self, iterator: Iterator, depth: int):
        if depth <= 0:
            raise ValueError(f"Prefetch depth must be positive, got {depth}.")
        self._iterator = iterator
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(iterator,), daemon=True)
        self._thread.start()

    def _put(self, item: object) -> bool:
        while not self._stop_event.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _close_iterator(self) -> None:
        for method_name in ("close", "_shutdown_workers"):
            method = getattr(self._iterator, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    logging.exception("Failed to close prefetched iterator via %s.", method_name)

    def _run(self, iterator: Iterator) -> None:
        try:
            for item in iterator:
                if not self._put(item):
                    return
        except Exception as exc:  # pragma: no cover - surfaced in __next__
            self._put(exc)
        finally:
            self._close_iterator()
            self._put(self._END)

    def __iter__(self) -> "_PrefetchIterator":
        return self

    def __next__(self):
        item = self._queue.get()
        if item is self._END:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self._stop_event.set()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)


def _action_delta_timestamps(data_config: _config.DataConfig, action_horizon: int, fps: int) -> dict[str, list[float]]:
    return {key: [t / fps for t in range(action_horizon)] for key in data_config.action_sequence_keys}


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    fast_local_loading: bool = False,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    datasets = [
        _create_single_lerobot_dataset(
            dataclasses.replace(data_config, repo_id=expanded_repo_id),
            action_horizon,
            fast_local_loading=fast_local_loading,
        )
        for expanded_repo_id in _config.expand_local_repo_ids(repo_id)
    ]
    return datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)


def _create_single_lerobot_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    *,
    fast_local_loading: bool,
) -> Dataset:
    repo_id = data_config.repo_id
    if fast_local_loading:
        root = pathlib.Path(repo_id).expanduser().resolve()
        info = lerobot_dataset.load_info(root)
        dataset = _FastLocalLeRobotDataset(
            str(root),
            info,
            delta_timestamps=_action_delta_timestamps(data_config, action_horizon, info["fps"]),
        )
    else:
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        dataset = lerobot_dataset.LeRobotDataset(
            repo_id,
            delta_timestamps=_action_delta_timestamps(data_config, action_horizon, dataset_meta.fps),
        )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset.meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(
    dataset: Dataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    tail_transforms: Sequence[_transforms.DataTransformFn] = (),
) -> Dataset:
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
            *tail_transforms,
        ],
    )


def create_transformed_single_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    skip_norm_stats: bool = False,
    tail_transforms: Sequence[_transforms.DataTransformFn] = (),
    fast_local_loading: bool = False,
    torch_dataset: Dataset | None = None,
) -> Dataset:
    dataset = torch_dataset
    if dataset is None:
        dataset = create_torch_dataset(
            data_config,
            action_horizon,
            model_config,
            fast_local_loading=fast_local_loading,
        )
    return transform_dataset(
        dataset,
        data_config,
        skip_norm_stats=skip_norm_stats,
        tail_transforms=(_transforms.PadCameraParams(), *tail_transforms) if data_config.pad_camera_params else tail_transforms,
    )


def create_transformed_mixed_torch_dataset(
    data_config: _config.MixedDataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    seed: int = 0,
    skip_norm_stats: bool = False,
    tail_transforms: Sequence[_transforms.DataTransformFn] = (),
) -> Dataset:
    torch_datasets: list[Dataset | None] = [None] * len(data_config.components)
    if data_config.fast_local_loading:
        process_count = jax.process_count()
        process_index = jax.process_index()
        assigned_indices = range(process_index, len(data_config.components), process_count)
        warmup_start = time.perf_counter()
        logging.info(
            "Fast local cache warmup: process=%s/%s assigned_components=%s total_components=%s",
            process_index,
            process_count,
            len(assigned_indices),
            len(data_config.components),
        )
        for index in assigned_indices:
            component = data_config.components[index]
            torch_datasets[index] = create_torch_dataset(
                component,
                action_horizon,
                model_config,
                fast_local_loading=True,
            )
        logging.info(
            "Fast local cache warmup complete: process=%s elapsed_s=%.1f",
            process_index,
            time.perf_counter() - warmup_start,
        )
        _distributed.sync_global_devices("fast local dataset cache warmup")

    materialize_start = time.perf_counter()
    mixed_tail_transforms = (
        (_transforms.PadCameraParams(), *tail_transforms) if data_config.pad_camera_params else tail_transforms
    )
    aligned_datasets = [
        create_transformed_single_torch_dataset(
            component,
            action_horizon,
            model_config,
            skip_norm_stats=skip_norm_stats,
            tail_transforms=mixed_tail_transforms,
            fast_local_loading=data_config.fast_local_loading,
            torch_dataset=torch_datasets[index],
        )
        for index, component in enumerate(data_config.components)
    ]
    logging.info(
        "Mixed dataset materialized: components=%s total_frames=%s fast_local_loading=%s elapsed_s=%.1f",
        len(aligned_datasets),
        sum(len(dataset) for dataset in aligned_datasets),
        data_config.fast_local_loading,
        time.perf_counter() - materialize_start,
    )
    return WeightedMixedDataset(aligned_datasets, data_config.weights, seed=seed)


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def prepend_state_dropout(data_config: _config.DataConfig, probability: float) -> _config.DataConfig:
    dropout = _transforms.StateDropout(probability)
    components = data_config.components if isinstance(data_config, _config.MixedDataConfig) else (data_config,)
    components = tuple(
        dataclasses.replace(
            component,
            model_transforms=dataclasses.replace(
                component.model_transforms,
                inputs=(dropout, *component.model_transforms.inputs),
            ),
        )
        for component in components
    )
    return (
        dataclasses.replace(data_config, components=components)
        if isinstance(data_config, _config.MixedDataConfig)
        else components[0]
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
    return_host_batch: bool = False,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
        return_host_batch: If true, each item also includes the transformed host batch before JAX sharding.
    """
    data_config = config.data.create(config.assets_dirs, config.model)

    # Robot state dropout for training
    if config.state_dropout_prob:
        if framework != "jax" or config.model.model_type != _model.ModelType.PI05:
            raise ValueError("state_dropout_prob is only supported for JAX pi0.5 training.")
        data_config = prepend_state_dropout(data_config, config.state_dropout_prob)

    if isinstance(data_config, _config.MixedDataConfig):
        logging.info(
            "data_config: MixedDataConfig(components=%s, fast_local_loading=%s, asset_id=%s)",
            len(data_config.components),
            data_config.fast_local_loading,
            data_config.asset_id,
        )
    else:
        logging.info("data_config: %s", data_config)

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
            return_host_batch=return_host_batch,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        return_host_batch=return_host_batch,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    return_host_batch: bool = False,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
        return_host_batch: If true, each item also includes the transformed host batch before JAX sharding.
    """
    if isinstance(data_config, _config.MixedDataConfig):
        dataset = create_transformed_mixed_torch_dataset(
            data_config,
            action_horizon,
            model_config,
            seed=seed,
            skip_norm_stats=skip_norm_stats,
        )
    else:
        dataset = create_transformed_single_torch_dataset(
            data_config,
            action_horizon,
            model_config,
            skip_norm_stats=skip_norm_stats,
        )

    # Use TorchDataLoader for both frameworks. PyTorch DDP and JAX multi-process
    # both shard the torch dataset with DistributedSampler.
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        process_count = jax.process_count()
        if process_count > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=process_count,
                rank=jax.process_index(),
                shuffle=shuffle,
                drop_last=True,
                seed=seed,
            )
        local_batch_size = batch_size // process_count

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
        return_host_batch=return_host_batch,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
    return_host_batch: bool = False,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
        return_host_batch=return_host_batch,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
        prefetch_depth: int = 4,
        return_host_batch: bool = False,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1 and sampler is None:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches
        self._sampler = sampler
        self._dataset = dataset
        self._epoch = 0
        self._framework = framework
        self._prefetch_depth = prefetch_depth
        self._return_host_batch = return_host_batch

        process_workers = num_workers
        if isinstance(dataset, WeightedMixedDataset):
            dataset.set_read_threads(num_workers)
            process_workers = 0
            logging.info(
                "Mixed batch loading: read_threads=%s, read_chunk_size=%s, batched_video_decode=True",
                num_workers,
                dataset._read_chunk_size,  # noqa: SLF001
            )

        mp_context = None
        if process_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=process_workers,
            multiprocessing_context=mp_context,
            persistent_workers=process_workers > 0,
            collate_fn=functools.partial(_collate_fn, drop_index=isinstance(dataset, WeightedMixedDataset)),
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def _yield_batches(self):
        num_items = 0
        try:
            while True:
                if self._sampler is not None and hasattr(self._sampler, "set_epoch"):
                    self._sampler.set_epoch(self._epoch)
                if hasattr(self._dataset, "set_epoch"):
                    self._dataset.set_epoch(self._epoch)
                self._epoch += 1
                data_iter = iter(self._data_loader)
                while True:
                    if self._num_batches is not None and num_items >= self._num_batches:
                        return
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        break  # We've exhausted the dataset. Create a new iterator and start over.
                    num_items += 1
                    # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                    if self._sharding is not None:
                        sharded_batch = jax.tree.map(
                            lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch
                        )
                        yield (sharded_batch, batch) if self._return_host_batch else sharded_batch
                    else:
                        torch_batch = jax.tree.map(torch.as_tensor, batch)
                        yield (torch_batch, batch) if self._return_host_batch else torch_batch
        finally:
            if isinstance(self._dataset, WeightedMixedDataset):
                self._dataset.close()

    def __iter__(self):
        iterator = self._yield_batches()
        if self._framework == "jax" and self._prefetch_depth > 0:
            prefetched_iterator = _PrefetchIterator(iterator, self._prefetch_depth)
            try:
                yield from prefetched_iterator
            finally:
                prefetched_iterator.close()
            return
        yield from iterator


def _collate_fn(items, *, drop_index: bool = False):
    """Collate samples, optionally dropping inconsistent mixed-source indices."""
    if drop_index:
        items = [{key: value for key, value in item.items() if key != "index"} for item in items]
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    torch.set_num_threads(1)


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
        return_host_batch: bool = False,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches
        self._return_host_batch = return_host_batch

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                sharded_batch = jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                yield (sharded_batch, batch) if self._return_host_batch else sharded_batch


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            if isinstance(batch, tuple) and len(batch) == 2:
                sharded_batch, host_batch = batch
                yield (
                    (_model.Observation.from_dict(sharded_batch), sharded_batch["actions"]),
                    (_model.Observation.from_dict(host_batch), host_batch["actions"]),
                )
            else:
                yield _model.Observation.from_dict(batch), batch["actions"]
