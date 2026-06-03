from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp

import importlib
import lerobot
importlib.reload(lerobot)

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch
import random

import openpi_value.models.model as _model
import openpi_value.training.config as _config
import openpi_value.transforms as _transforms
from openpi_value.training.custom_lerobot_dataset import CustomLeRobotDataset, CustomMultiLeRobotDataset
from inspect import signature


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


# * Used for norm states. NOT used for training
def create_torch_dataset_naive(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    if isinstance(repo_id, str) and os.path.exists(repo_id):

        if 'data' not in os.listdir(repo_id) and 'videos' not in os.listdir(repo_id):
            repo_id = [os.path.join(repo_id, d) for d in os.listdir(repo_id) if os.path.isdir(os.path.join(repo_id, d))]

            
    if not isinstance(repo_id, list):
        repo_id = [repo_id]


    # for rp in repo_id:
    repo_id = [
        rp for rp in repo_id if 'Put_The_Items_Into_The_Storage_Box_20250929_002_007' not in rp
    ]  # * Remove this problematic dataset


    if len(repo_id) > 1:

        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id[0])  # * Just use the first repo to get fps
        dataset = lerobot_dataset.MultiLeRobotDataset(
            repo_id,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
            },

            # * tolerance
            tolerances_s = dict.fromkeys(repo_id, 0.1)
        )

    else:
        repo_id = repo_id[0]
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        dataset = lerobot_dataset.LeRobotDataset(
            repo_id,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
            },

            # * tolerance
            tolerance_s = 0.1,
            # video_backend="pyav",  # smch: Force pyav backend to avoid torchcodec issues, need change in the future
        )

    return dataset


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    config=None,
) -> Dataset:
    """Create a dataset for training (supports multiple repo_ids)."""
    
    split=config.split
    
    # Automatically collect all kwargs accepted by CustomLeRobotDataset and override if present in config.
    from inspect import signature

    # Get argument names for CustomLeRobotDataset's __init__, skipping 'self' and required args
    dataset_kwargs = signature(CustomLeRobotDataset.__init__).parameters
    skip_args = {"self", "repo_id", "episodes", "image_transforms", "delta_timestamps", "tolerance_s", "download_videos", "video_backend"}  # handled separately/custom elsewhere
    valid_kwargs = [k for k in dataset_kwargs if k not in skip_args]

    # Build data_kwargs only if config has the attribute
    data_kwargs = {k: getattr(config, k) for k in valid_kwargs if hasattr(config, k)}
    video_backend = os.environ.get("RISE_VIDEO_BACKEND", "pyav")

    repo_ids = data_config.repo_id
    if not repo_ids:
        raise ValueError("Repo ID(s) not set in data_config. Cannot create dataset.")

    # * Support folder containing multiple lerobot datasetss
    if isinstance(repo_ids, str) and os.path.exists(repo_ids):
        if 'data' not in os.listdir(repo_ids) and 'videos' not in os.listdir(repo_ids):
            repo_ids = [os.path.join(repo_ids, d) for d in os.listdir(repo_ids) if os.path.isdir(os.path.join(repo_ids, d))]


    if not isinstance(repo_ids, list):
        repo_ids = [repo_ids]

    if repo_ids == ["fake"]:
        return FakeDataset(model_config, num_samples=1024)
    
    repo_ids = [
        rp for rp in repo_ids if 'Put_The_Items_Into_The_Storage_Box_20250929_002_007' not in rp
    ]  # * Remove this problematic dataset

    repo_to_episodes: dict[str, list[int]] = {}
    valid_repo_ids: list[str] = []
    for rid in repo_ids:
        if os.path.isabs(rid):
            info_path = os.path.join(rid, "meta", "info.json")
            if not os.path.isfile(info_path):
                logging.warning("Skipping local dataset without metadata: %s", info_path)
                continue

        episodes = get_episodes(rid, split)
        if len(episodes) == 0:
            logging.warning("Skipping dataset with no episodes for split '%s': %s", split, rid)
            continue

        repo_to_episodes[rid] = episodes
        valid_repo_ids.append(rid)

    repo_ids = valid_repo_ids
    if len(repo_ids) == 0:
        raise ValueError("No valid datasets found after filtering unavailable local dataset paths.")

    if len(repo_ids) > 1:
        all_delta_timestamps = []
        for rid in repo_ids:
            meta = lerobot_dataset.LeRobotDatasetMetadata(rid)
            delta_timestamps = {
                key: [t / meta.fps for t in range(action_horizon)]
                for key in data_config.action_sequence_keys
            }
            all_delta_timestamps.append(delta_timestamps)

        dataset = CustomMultiLeRobotDataset(
            repo_ids=repo_ids,
            episodes=repo_to_episodes,  # dict[repo_id] -> list[int]
            delta_timestamps=all_delta_timestamps,
            video_backend=video_backend,
            **data_kwargs
        )
    else:
        rid = repo_ids[0]

        meta = lerobot_dataset.LeRobotDatasetMetadata(rid)
        delta_timestamps = {
            key: [t / meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        }

        dataset = CustomLeRobotDataset(
            repo_id=rid,
            episodes=repo_to_episodes[rid],
            delta_timestamps=delta_timestamps,
            video_backend=video_backend,

            **data_kwargs
        )

    # TODO: we need to fix this. Now only use one dataset's meta for tasks.
    if data_config.prompt_from_task:
        if len(repo_ids) == 1:
            dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_ids[0])
            dataset = TransformedDataset(
                dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
            )
        else:
            dataset = TransformedDataset(
                dataset, [_transforms.PromptFromLeRobotTask(tasks=None)]
            )

    return dataset


# * split tasks could be troublesome, 
# * suppose you have a dataset with two tasks: task1 and task2, when split task,
# * it's possible to train on one task only, and val on another task only, which is not desired.

# * One simple update is to train/val on all tasks without heldout tasks.
def get_episodes(repo_id, split):

    assert split in ['all', 'train_tasks', 'val_tasks'], \
        f"Invalid split option: {split}. Choose from 'all', 'train_tasks', 'val_tasks'. heldout_tasks is deprecated."

    if os.path.isabs(repo_id):
        info_path = os.path.join(repo_id, "meta", "info.json")
        if not os.path.isfile(info_path):
            raise FileNotFoundError(f"Local dataset metadata not found: {info_path}")

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)


    episodes_meta = dataset_meta.episodes

    # Step 1: Group episodes by task (assuming episodes have a "tasks" field)
    task_to_episodes = {}
    for episode_index, episode_data in episodes_meta.items():
        tasks = episode_data['tasks']  # This should be a list of tasks for this episode
        for task in tasks:
            if task not in task_to_episodes:
                task_to_episodes[task] = []
            task_to_episodes[task].append(episode_index)

    # Step 2: Split the episodes
    total_episodes = set()
    train_episodes = set()
    val_episodes = set()
    all_tasks = list(task_to_episodes.keys())
    
    train_val_ratio = 0.9

    # train_val tasks could be the same, but different episodes
    for task in all_tasks:
        task_episodes = task_to_episodes[task]
        
        if len(task_episodes) <= 1:
            train_val_ratio = 1.
        else:
            train_val_ratio = 0.9

        split_index = int(train_val_ratio * len(task_episodes))  # 90% for train, 20% for val
        
        total_episodes.update(task_episodes)
        train_episodes.update(task_episodes[:split_index])
        val_episodes.update(task_episodes[split_index:])

    if split == "all":
        out_eps = list(total_episodes)
    
    elif split == "train_tasks":
        out_eps = list(train_episodes - val_episodes)

    elif split == "val_tasks":
        out_eps = list(val_episodes - train_episodes)

    else:
        raise ValueError(f"Invalid split option: {split}. Choose from 'train', 'val', or 'heldout'.")

    # * Order the episodes
    out_eps = sorted(out_eps)
    
    return out_eps




def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    # norm_stats = {}
    norm_stats = None
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
        config=config,
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
    config: str = None,
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
    dataset = create_torch_dataset(data_config, 
                                   action_horizon, 
                                   model_config, 
                                    config=config)
    
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

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
                # drop_last=True,
                drop_last=config.drop_last,
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
        drop_last=config.drop_last,
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
        drop_last: bool = True,
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
            drop_last=drop_last,
            generator=generator,
            
            # pin_memory=True, # * A little bit speedup, not obvious. Back to default (False).
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
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

class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
