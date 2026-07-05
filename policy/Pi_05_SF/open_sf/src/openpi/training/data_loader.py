from collections.abc import Iterator, Sequence
import queue
import logging
import multiprocessing
import os
import threading
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)
_SF_IDENTITY_FIELDS = ("episode_index", "frame_index", "index")


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
        sample = self._dataset[index]
        identity = {field: sample[field] for field in _SF_IDENTITY_FIELDS if field in sample}
        transformed = self._transform(sample)
        for field, value in identity.items():
            transformed.setdefault(field, value)
        return transformed

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
                identity = {field: sample[field] for field in _SF_IDENTITY_FIELDS if field in sample}
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                transformed_batch = jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
                for field, value in identity.items():
                    transformed_batch.setdefault(field, value)
                yield transformed_batch
            else:
                identity = {field: sample[field] for field in _SF_IDENTITY_FIELDS if field in sample}
                transformed = self._transform(sample)
                for field, value in identity.items():
                    transformed.setdefault(field, value)
                yield transformed

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
            "episode_index": np.asarray(0, dtype=np.int64),
            "frame_index": np.asarray(index.__index__(), dtype=np.int64),
            "index": np.asarray(index.__index__(), dtype=np.int64),
        }

    def __len__(self) -> int:
        return self._num_samples


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

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __getitem__(self, index: SupportsIndex) -> dict:
        index_value = index.__index__()
        if index_value < 0:
            index_value %= self._length

        # TODO: more efficient and fair sampling
        rng = np.random.default_rng(np.random.SeedSequence([self._seed, self._epoch, index_value]))
        dataset_index = int(rng.choice(len(self._datasets), p=self._weights))
        sample_index = int(rng.integers(len(self._datasets[dataset_index])))
        return self._datasets[dataset_index][sample_index]

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


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

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
) -> Dataset:
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
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
    aligned_datasets = [
        create_transformed_single_torch_dataset(
            component,
            action_horizon,
            model_config,
            skip_norm_stats=skip_norm_stats,
            tail_transforms=(_transforms.PadCameraParams(), *tail_transforms) if data_config.pad_camera_params else tail_transforms,
        )
        for component in data_config.components
    ]
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


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

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
            return_sf_identity=config.sf_cache_enable,
            sf_dataset_uid=config.sf_dataset_uid,
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
        return_sf_identity=config.sf_cache_enable,
        sf_dataset_uid=config.sf_dataset_uid,
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
    return_sf_identity: bool = False,
    sf_dataset_uid: int = 0,
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

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
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
        local_batch_size = batch_size // jax.process_count()

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
    )

    return DataLoaderImpl(
        data_config,
        data_loader,
        return_sf_identity=return_sf_identity,
        dataset_uid=sf_dataset_uid,
    )


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
    return_sf_identity: bool = False,
    sf_dataset_uid: int = 0,
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
    )

    return DataLoaderImpl(
        data_config,
        data_loader,
        return_sf_identity=return_sf_identity,
        dataset_uid=sf_dataset_uid,
    )


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
        if jax.process_count() > 1:
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
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)

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
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(
        self,
        data_config: _config.DataConfig,
        data_loader: TorchDataLoader | RLDSDataLoader,
        *,
        return_sf_identity: bool = False,
        dataset_uid: int = 0,
    ):
        self._data_config = data_config
        self._data_loader = data_loader
        self._return_sf_identity = return_sf_identity
        self._dataset_uid = dataset_uid

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            observation = _model.Observation.from_dict(batch)
            actions = batch["actions"]
            if self._return_sf_identity:
                yield observation, actions, extract_sf_identity(batch, self._dataset_uid)
            else:
                yield observation, actions


def extract_sf_identity(batch: dict, dataset_uid: int) -> dict[str, torch.Tensor]:
    if "image" in batch:
        batch_size = next(iter(batch["image"].values())).shape[0]
    elif "actions" in batch:
        batch_size = batch["actions"].shape[0]
    elif "episode_index" in batch:
        batch_size = np.asarray(batch["episode_index"]).shape[0]
    elif "frame_index" in batch:
        batch_size = np.asarray(batch["frame_index"]).shape[0]
    elif "index" in batch:
        batch_size = np.asarray(batch["index"]).shape[0]
    else:
        raise ValueError("Unable to infer batch size for SF identity extraction")

    if "frame_index" in batch:
        step = batch["frame_index"]
    elif "index" in batch:
        step = batch["index"]
    else:
        raise ValueError("SF identity extraction requires a real frame_index or index field.")
    episode = batch.get("episode_index", np.zeros(batch_size, dtype=np.int64))
    episode = np.asarray(jax.device_get(episode), dtype=np.int64)
    step = np.asarray(jax.device_get(step), dtype=np.int64)
    uid = np.full(batch_size, int(dataset_uid), dtype=np.int64)
    return {
        "dataset_uid": torch.as_tensor(uid, dtype=torch.long),
        "episode_index": torch.as_tensor(episode, dtype=torch.long),
        "step_index": torch.as_tensor(step, dtype=torch.long),
    }
