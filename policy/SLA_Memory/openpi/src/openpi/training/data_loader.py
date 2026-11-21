from collections.abc import Iterator, Sequence
import dataclasses
import logging
import multiprocessing
import os
import queue
import threading
import time
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

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


# Placeholder delta for an episode-anchored slot: it only has to land before the episode start,
# since `_EpisodeAnchoredLeRobotDataset` overwrites the resolved index before anything reads it.
# Slots sharing one placeholder is fine, and so is landing on one frame after the overwrite:
# lerobot matches every query timestamp independently (`decode_video_frames_torchvision`).
#
# Fetching a far-away frame is cheap because the vendored lerobot (`packages/lerobot`) decodes
# well-separated timestamps in separate passes (`_decode_video_frames_clustered` in
# `lerobot/common/datasets/lerobot_dataset.py`); upstream lerobot decodes and holds every frame in
# between.
_ANCHOR_PLACEHOLDER = -100_000


def _delta_timestamps(data_config: _config.DataConfig, action_horizon: int, fps: int) -> dict[str, list[float]]:
    deltas = {key: [t / fps for t in range(action_horizon)] for key in data_config.action_sequence_keys}
    if data_config.extra_frames:
        # Order matters: the transform reads slots [:-1] as the extra frames, [-1] as the current one.
        offsets = [
            (_ANCHOR_PLACEHOLDER if offset >= 0 else offset) / fps for offset in data_config.extra_frames
        ] + [0.0]
        for key in data_config.extra_frame_keys:
            deltas[key] = offsets
    return deltas


def _episode_anchors(data_config: _config.DataConfig) -> dict[str, list[int | None]]:
    """Per camera key, the episode-relative frame each slot must resolve to (None = leave alone)."""
    anchors = [offset if offset >= 0 else None for offset in data_config.extra_frames] + [None]
    return {key: anchors for key in data_config.extra_frame_keys}


class _EpisodeAnchoredLeRobotDataset(lerobot_dataset.LeRobotDataset):
    """Resolve some query slots to `ep_start + k` instead of an offset from the current frame.

    LeRobot deltas are relative and clamped to the episode (`_get_query_indices`), so every
    sufficiently negative delta collapses onto the episode's first frame and no static delta can
    mean "frame k of this episode" -- the offset needed is `ep_start + k - idx`, which varies per
    sample. Overwriting the already-resolved index is the smallest way to say it.
    """

    def __init__(self, *args, episode_anchors: dict[str, list[int | None]], **kwargs):
        super().__init__(*args, **kwargs)
        self._episode_anchors = episode_anchors

    def _get_query_indices(self, idx: int, ep_idx: int):
        query_indices, padding = super()._get_query_indices(idx, ep_idx)
        ep_start = self.episode_data_index["from"][ep_idx].item()
        ep_end = self.episode_data_index["to"][ep_idx].item()
        for key, anchors in self._episode_anchors.items():
            for slot, anchor in enumerate(anchors):
                if anchor is None:
                    continue
                # An episode shorter than the anchor clamps to its last frame; the pad flag then
                # says so honestly, unlike the out-of-range placeholder it replaces.
                query_indices[key][slot] = min(ep_start + anchor, ep_end - 1)
                padding[f"{key}_is_pad"][slot] = ep_start + anchor >= ep_end
        return query_indices, padding


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    kwargs = {
        "delta_timestamps": _delta_timestamps(data_config, action_horizon, dataset_meta.fps),
        "tolerance_s": 0.05,
        "video_backend": "pyav",
    }
    dataset = (
        _EpisodeAnchoredLeRobotDataset(repo_id, episode_anchors=_episode_anchors(data_config), **kwargs)
        if data_config.extra_frames
        else lerobot_dataset.LeRobotDataset(repo_id, **kwargs)
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset.meta.tasks)])

    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
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
        ],
    )


def create_transformed_single_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    skip_norm_stats: bool = False,
) -> Dataset:
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    return transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)


def create_transformed_mixed_torch_dataset(
    data_config: _config.MixedDataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    skip_norm_stats: bool = False,
) -> Dataset:
    materialize_start = time.perf_counter()
    datasets = [
        create_transformed_single_torch_dataset(component, action_horizon, model_config, skip_norm_stats=skip_norm_stats)
        for component in data_config.components
    ]
    logging.info(
        "Mixed dataset materialized: components=%s total_frames=%s elapsed_s=%.1f",
        len(datasets),
        sum(len(dataset) for dataset in datasets),
        time.perf_counter() - materialize_start,
    )
    if len(set(data_config.weights)) > 1:
        raise NotImplementedError(f"Unequal mixed source weights are not supported: {data_config.weights}")
    # Every frame exactly once per epoch: the sampler shuffles the concatenated indices in the main process, so
    # persistent workers do not need to know the epoch.
    return torch.utils.data.ConcatDataset(datasets)


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
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
    """
    data_config = config.data.create(config.assets_dirs, config.model)

    # Robot state dropout for training
    if config.state_dropout_prob:
        if config.model.model_type != _model.ModelType.PI05:
            raise ValueError("state_dropout_prob is only supported for pi0.5.")
        data_config = prepend_state_dropout(data_config, config.state_dropout_prob)

    if isinstance(data_config, _config.MixedDataConfig):
        logging.info(
            "data_config: MixedDataConfig(components=%s, asset_id=%s)",
            len(data_config.components),
            data_config.asset_id,
        )
    else:
        logging.info("data_config: %s", data_config)

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
    """
    if isinstance(data_config, _config.MixedDataConfig):
        dataset = create_transformed_mixed_torch_dataset(
            data_config,
            action_horizon,
            model_config,
            skip_norm_stats=skip_norm_stats,
        )
    else:
        dataset = create_transformed_single_torch_dataset(
            data_config,
            action_horizon,
            model_config,
            skip_norm_stats=skip_norm_stats,
        )

    # Multi-process training shards the dataset across processes with DistributedSampler.
    sampler = None
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
        sharding=sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
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
        prefetch_depth: int = 4,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            sampler: Sampler used instead of shuffling, e.g. to split the data across processes.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
            prefetch_depth: Number of batches fetched ahead in a background thread.
        """
        if jax.process_count() > 1 and sampler is None:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches
        self._sampler = sampler
        self._epoch = 0
        self._prefetch_depth = prefetch_depth

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def _yield_batches(self):
        num_items = 0
        while True:
            if self._sampler is not None and hasattr(self._sampler, "set_epoch"):
                self._sampler.set_epoch(self._epoch)
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
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)

    def __iter__(self):
        iterator = self._yield_batches()
        if self._prefetch_depth > 0:
            prefetched_iterator = _PrefetchIterator(iterator, self._prefetch_depth)
            try:
                yield from prefetched_iterator
            finally:
                prefetched_iterator.close()
            return
        yield from iterator


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
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


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
