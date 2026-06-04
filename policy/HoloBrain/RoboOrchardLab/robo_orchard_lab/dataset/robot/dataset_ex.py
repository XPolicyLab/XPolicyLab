# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from __future__ import annotations
import copy
import inspect
import threading
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from types import GeneratorType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Generic,
    Iterable,
    Iterator,
    overload,
)

import numpy as np
import torch
from datasets import IterableDataset as HFIterableDataset
from pydantic import Field
from robo_orchard_core.utils.config import ClassType, Config
from robo_orchard_core.utils.logging import LoggerManager
from torch.utils.data import (
    DataLoader as TorchDataLoader,
    Dataset as TorchDataset,
    IterableDataset as TorchIterableDataset,
)
from torch.utils.data._utils.fetch import _IterableDatasetFetcher
from torch.utils.data.dataloader import (
    _MultiProcessingDataLoaderIter,
    _SingleProcessDataLoaderIter,
)
from typing_extensions import TypeVar

from robo_orchard_lab.dataset.sampler import (
    ChunkedIndiceTable,
    IndiceTable,
    IndiceTableSampler,
    ShardStrategy,
    Sized,
)

logger = LoggerManager().get_child(__name__)

__all__ = [
    "ShardConfig",
    "BatchLoaderConfig",
    "DataLoader",
    "ShuffleConfig",
    "IterableDatasetMixin",
    "DatasetWithIndices",
    "IterableWithLenDataset",
    "DatasetItem",
    "DictIterableDataset",
]


DatasetType = TypeVar("DatasetType", bound=TorchDataset)
_TORCH_DATALOADER_INIT_SIGNATURE = inspect.signature(TorchDataLoader.__init__)
_DEFAULT_VIRTUAL_GETITEMS_BATCH_SIZE = 32
_PREFETCH_CLOSE_JOIN_TIMEOUT_SEC = 1.0


class ShardConfig(Config):
    contiguous: bool = True
    shard_strategy: ShardStrategy = None


class BatchLoaderConfig(Config):
    batch_size: int = 1
    collate_fn: Callable | None = None
    drop_last: bool = False


def _collate_self_batched_item(
    batch: list[Any], user_collate_fn: Callable | None = None
) -> Any:
    if len(batch) != 1:
        raise ValueError(
            "Self-batched datasets expect DataLoader to receive exactly "
            f"one item per batch, but got {len(batch)} items."
        )
    item = batch[0]
    if user_collate_fn is None:
        return item
    return user_collate_fn(item)


def _should_use_dataset_batch_loader(
    dataset: Any, use_dataset_side_batching: bool
) -> bool:
    if (
        isinstance(dataset, IterableDatasetMixin)
        and dataset.batch_loader_kwargs is not None
    ):
        return True

    return (
        use_dataset_side_batching
        and isinstance(dataset, (IterableWithLenDataset, DictIterableDataset))
        and dataset.batch_loader_kwargs is None
    )


def _normalize_shuffle_for_non_iterable_dataset_mixin(
    dataset: Any,
    dataloader_kwargs: dict[str, Any],
) -> dict[str, Any]:
    dataloader_shuffle = dataloader_kwargs.get("shuffle")
    if not isinstance(dataloader_shuffle, ShuffleConfig):
        return dataloader_kwargs

    if dataloader_shuffle.chunk_size is not None:
        warnings.warn(
            "`ShuffleConfig.chunk_size` is only supported for "
            "IterableDatasetMixin datasets. Falling back to the boolean "
            "`shuffle` value for this DataLoader.",
            UserWarning,
        )

    if isinstance(dataset, TorchIterableDataset):
        if dataloader_shuffle.shuffle:
            warnings.warn(
                "Non-IterableDatasetMixin iterable datasets do not support "
                "outer DataLoader shuffling. Resetting `shuffle=False`.",
                UserWarning,
            )
        dataloader_kwargs["shuffle"] = False
        return dataloader_kwargs

    dataloader_kwargs["shuffle"] = dataloader_shuffle.shuffle
    return dataloader_kwargs


def _batched_iterator_with_indices(
    dataset: TorchDataset,
    indice_iter: Iterable[int],
    batch_size: int = _DEFAULT_VIRTUAL_GETITEMS_BATCH_SIZE,
) -> Iterator[Any]:
    if not hasattr(dataset, "__getitems__"):
        for idx in indice_iter:
            yield dataset[idx]
        return

    batch_indices: list[int] = []
    for idx in indice_iter:
        batch_indices.append(int(idx))
        if len(batch_indices) >= batch_size:
            yield from dataset.__getitems__(batch_indices)  # type: ignore[attr-defined]
            batch_indices = []

    if batch_indices:
        yield from dataset.__getitems__(batch_indices)  # type: ignore[attr-defined]


def _wrap_with_prefetch_if_needed(
    iterator: Iterator[Any],
    shuffle_config: ShuffleConfig,
    generator: torch.Generator | np.random.Generator | None,
    batch_loader_kwargs: BatchLoaderConfig | None,
) -> Iterator[Any]:
    prefetch_size = shuffle_config.prefetch_size
    if (
        prefetch_size is not None
        and shuffle_config.shuffle
        and batch_loader_kwargs is None
    ):
        logger.debug(
            "Applying prefetching with prefetch size: %d", prefetch_size
        )
        return _create_prefetch_iterator(
            iterator,
            prefetch_size,
            shuffle=shuffle_config.shuffle,
            generator=generator,
        )

    logger.debug(
        "No prefetching applied, shuffle: %s",
        shuffle_config.shuffle,
    )
    return iterator


class DataLoader(TorchDataLoader):
    """A thin wrapper around PyTorch ``DataLoader``.

    For iterable datasets this loader can operate with two batching layers:

    1. The ordinary outer ``TorchDataLoader`` batching layer.
    2. A dataset-side batching layer driven by ``batch_loader_kwargs``.

    For iterable datasets that already yield batches through
    ``batch_loader_kwargs``, this loader clones the input dataset, aligns the
    dataset-side batch settings with the caller-provided dataloader batch
    arguments, and then configures the outer ``TorchDataLoader`` to forward one
    already-formed batch at a time.

    In that self-batched mode the outer loader may expose ``batch_size == 1``
    because it is only transporting one ready-made batch per iteration. The
    effective sample batch size is tracked separately and is the value used by
    ``__len__`` and the iterable dataset batch-count helpers.

    When ``use_dataset_side_batching`` is True and the input dataset is a
    supported iterable dataset without ``batch_loader_kwargs``, this loader
    will internally enable aligned ``batch_loader_kwargs`` on a cloned dataset.

    Args:
        dataset: The dataset to load.
        use_dataset_side_batching: When True and ``dataset`` is a supported
            iterable dataset without ``batch_loader_kwargs``, enable
            dataset-side batch loading on a cloned dataset.
        *args: Positional arguments forwarded to ``TorchDataLoader``.
        **kwargs: Keyword arguments forwarded to ``TorchDataLoader``. Relevant
            batch-related arguments, and ``shuffle`` when supported by the
            dataset, are also aligned into dataset-side configuration when
            self-batched loading is enabled.
    """

    @overload
    def __init__(
        self,
        dataset: Any,
        batch_size: int | None = 1,
        shuffle: bool | ShuffleConfig | None = None,
        sampler: Any | None = None,
        batch_sampler: None = None,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Callable | None = None,
        multiprocessing_context: Any = None,
        generator: torch.Generator | None = None,
        *,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
        in_order: bool = True,
        use_dataset_side_batching: bool = False,
    ) -> None: ...

    @overload
    def __init__(
        self,
        dataset: Any,
        batch_size: None = None,
        shuffle: bool | ShuffleConfig | None = None,
        sampler: None = None,
        batch_sampler: Any = None,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Callable | None = None,
        multiprocessing_context: Any = None,
        generator: torch.Generator | None = None,
        *,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
        in_order: bool = True,
        use_dataset_side_batching: bool = False,
    ) -> None: ...

    def __init__(
        self,
        dataset,
        *args,
        use_dataset_side_batching: bool = False,
        **kwargs,
    ):
        dataloader_kwargs = self._bind_dataloader_kwargs(
            dataset=dataset,
            args=args,
            kwargs=kwargs,
        )
        aligned_batch_loader_kwargs = None
        if isinstance(dataset, IterableDatasetMixin):
            (
                dataset,
                self._uses_dataset_batch_loader,
                aligned_batch_loader_kwargs,
            ) = self._clone_iterable_dataset_for_dataloader(
                dataset=dataset,
                dataloader_kwargs=dataloader_kwargs,
                use_dataset_side_batching=use_dataset_side_batching,
            )
            dataloader_kwargs["dataset"] = dataset
        else:
            self._uses_dataset_batch_loader = False
            dataloader_kwargs = (
                _normalize_shuffle_for_non_iterable_dataset_mixin(
                    dataset=dataset,
                    dataloader_kwargs=dataloader_kwargs,
                )
            )

        batch_size = dataloader_kwargs.get("batch_size", 1)
        self._effective_batch_size = 1 if batch_size is None else batch_size
        self._effective_drop_last = dataloader_kwargs.get("drop_last", False)

        if aligned_batch_loader_kwargs is not None:
            self._effective_batch_size = aligned_batch_loader_kwargs.batch_size
            self._effective_drop_last = aligned_batch_loader_kwargs.drop_last
            dataloader_kwargs = (
                self._normalize_outer_dataloader_for_self_batched_dataset(
                    dataloader_kwargs
                )
            )

        super().__init__(**dataloader_kwargs)

    def __len__(self) -> int:
        """Return the batch count using the effective batching layer.

        For iterable datasets this may differ from the outer dataloader's
        visible ``batch_size`` because dataset-side batching normalizes the
        outer loader to forward one already-built batch at a time.
        """
        if isinstance(self.dataset, IterableDatasetMixin):
            return self.dataset.get_total_batch_num(
                num_workers=self.num_workers,
                batch_size=self._effective_batch_size,
                drop_last=self._effective_drop_last,
            )

        return super().__len__()

    @staticmethod
    def _bind_dataloader_kwargs(
        dataset: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        bound = _TORCH_DATALOADER_INIT_SIGNATURE.bind_partial(
            None, dataset, *args, **kwargs
        )
        dataloader_kwargs = dict(bound.arguments)
        dataloader_kwargs.pop("self", None)
        return dataloader_kwargs

    @staticmethod
    def _clone_iterable_dataset_for_dataloader(
        dataset: IterableDatasetMixin,
        dataloader_kwargs: dict[str, Any],
        use_dataset_side_batching: bool,
    ) -> tuple[IterableDatasetMixin, bool, BatchLoaderConfig | None]:
        """Clone iterable datasets when loader-local state must diverge.

        The clone keeps caller-owned dataset objects immutable while this
        dataloader rewrites shuffle or dataset-side batching configuration for
        its own execution.
        """
        uses_dataset_batch_loader = _should_use_dataset_batch_loader(
            dataset=dataset,
            use_dataset_side_batching=use_dataset_side_batching,
        )
        should_clone_for_shuffle = (
            not uses_dataset_batch_loader and "shuffle" in dataloader_kwargs
        )
        if not uses_dataset_batch_loader and not should_clone_for_shuffle:
            return dataset, False, None

        aligned_batch_loader_kwargs = (
            DataLoader._align_batch_loader_kwargs(
                dataset=dataset,
                dataloader_kwargs=dataloader_kwargs,
            )
            if uses_dataset_batch_loader
            else None
        )

        aligned_shuffle_config = DataLoader._align_dataset_shuffle_config(
            dataset=dataset,
            dataloader_shuffle=dataloader_kwargs.get("shuffle"),
        )
        logger.debug("new shuffle cfg: %s", aligned_shuffle_config)

        if isinstance(dataset, IterableWithLenDataset):
            cloned_dataset: IterableDatasetMixin = IterableWithLenDataset(
                dataset=dataset.dataset,
                indices=dataset.indice_sampler.table,
                shuffle=aligned_shuffle_config,
                shard_kwargs=dataset.shard_kwargs,
                generator=dataset.indice_sampler.generator,
                batch_loader_kwargs=aligned_batch_loader_kwargs,
            )
        elif isinstance(dataset, DictIterableDataset):
            cloned_dataset = DictIterableDataset(
                datasets=dataset.dataset_items,
                shuffle=aligned_shuffle_config,
                shard_kwargs=dataset.shard_kwargs,
                generator=dataset._generator,
                batch_loader_kwargs=aligned_batch_loader_kwargs,
                max_dataset_concurrency=dataset._max_dataset_concurrency,
            )
        else:
            raise TypeError(
                "Iterable dataset cloning only supports "
                "IterableWithLenDataset and DictIterableDataset."
            )

        if should_clone_for_shuffle:
            dataloader_kwargs["shuffle"] = False

        return (
            cloned_dataset,
            uses_dataset_batch_loader,
            aligned_batch_loader_kwargs,
        )

    @staticmethod
    def _align_batch_loader_kwargs(
        dataset: IterableDatasetMixin,
        dataloader_kwargs: dict[str, Any],
    ) -> BatchLoaderConfig:
        """Merge dataset batch defaults with explicit dataloader arguments.

        ``batch_size``, ``collate_fn`` and ``drop_last`` from the caller win
        over the dataset defaults so the cloned dataset behaves as if those
        arguments had been supplied at dataset construction time.
        """
        dataset_batch_loader_kwargs = dataset.batch_loader_kwargs
        aligned_batch_loader_kwargs = (
            BatchLoaderConfig(**dataset_batch_loader_kwargs.to_dict())
            if dataset_batch_loader_kwargs is not None
            else BatchLoaderConfig()
        )
        for key in BatchLoaderConfig.model_fields:
            if key in dataloader_kwargs:
                setattr(
                    aligned_batch_loader_kwargs,
                    key,
                    dataloader_kwargs[key],
                )
        return aligned_batch_loader_kwargs

    @staticmethod
    def _align_dataset_shuffle_config(
        dataset: IterableDatasetMixin,
        dataloader_shuffle: bool | ShuffleConfig | None,
    ) -> ShuffleConfig:
        """Translate dataloader shuffle requests into dataset shuffle state.

        A boolean request only replaces the ``shuffle`` flag. A full
        ``ShuffleConfig`` replaces the whole configuration so the caller can
        override chunking and prefetch-related settings as well.
        """
        if isinstance(dataset, IterableWithLenDataset):
            dataset_shuffle = dataset._shuffle_config
        elif isinstance(dataset, DictIterableDataset):
            dataset_shuffle = dataset._shuffle
        else:
            raise TypeError(
                "Dataset shuffle alignment only supports "
                "IterableWithLenDataset and DictIterableDataset."
            )

        aligned_shuffle_config = ShuffleConfig(**dataset_shuffle.to_dict())
        if dataloader_shuffle is None:
            return aligned_shuffle_config
        if isinstance(dataloader_shuffle, ShuffleConfig):
            return ShuffleConfig(**dataloader_shuffle.to_dict())

        aligned_shuffle_config.shuffle = dataloader_shuffle
        return aligned_shuffle_config

    @staticmethod
    def _normalize_outer_dataloader_for_self_batched_dataset(
        dataloader_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize outer DataLoader kwargs for self-batched datasets.

        In this mode the dataset itself already yields complete batches. The
        outer ``TorchDataLoader`` should therefore only transport one dataset
        item at a time and unwrap it, instead of trying to batch samples again.

        Any user ``collate_fn`` has already been aligned into the dataset-side
        ``batch_loader_kwargs``. The outer loader only needs to unwrap the
        single item it receives from the dataset.
        """
        dataloader_kwargs["batch_size"] = 1
        dataloader_kwargs["collate_fn"] = partial(_collate_self_batched_item)
        # ``drop_last`` has already been applied by the inner dataset batch
        # generation logic via ``batch_loader_kwargs``. The outer dataloader is
        # only used to forward one already-formed batch at a time, so keeping
        # ``drop_last=True`` here would risk dropping an entire final batch at
        # the wrong layer.
        dataloader_kwargs["drop_last"] = False
        dataloader_kwargs["shuffle"] = False
        return dataloader_kwargs


class ShuffleConfig(Config):
    """Configuration for shuffling the dataset indices.

    Args:
        shuffle (bool): Whether to shuffle the dataset indices.
        chunk_size (int | None): The chunk size for the indices. If provided,
            the indices will be split into chunks of the given size, and each
            chunk will be treated as a unit for sharding. This can help reduce
            the overhead of sharding when the dataset is very large. If None,
            then no chunking will be done and the indices will be treated as
            individual samples. Defaults to None.
        prefetch_factor (int): The factor to determine the prefetch size for
            prefetching the dataset. The prefetch size will be calculated as
            `chunk_size * prefetch_factor` if `chunk_size` is provided, otherwise
            the prefetch size will be `None` and no prefetching will be applied.
            This argument is usually only valid when `chunk_size` is provided and
            `shuffle` is True. Defaults to 4.

    """  # noqa: E501

    shuffle: bool = False
    chunk_size: int | None = None
    prefetch_factor: int = 4

    @property
    def prefetch_size(self) -> int | None:
        if self.chunk_size is not None:
            return self.chunk_size * self.prefetch_factor
        return None


class IterableDatasetMixin(metaclass=ABCMeta):
    @property
    @abstractmethod
    def batch_loader_kwargs(self) -> BatchLoaderConfig | None:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def shard_kwargs(self) -> ShardConfig:
        raise NotImplementedError

    @abstractmethod
    def shard(self, num_shards: int, index: int):
        """Shard the dataset into multiple shards.

        Args:
            num_shards (int): The total number of shards to create.
            index (int): The ID of the shard to return. Must be in the
                range [0, num_shards - 1].
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def total_iterator_length(self) -> int:
        """Get the total number of data samples in the iterator."""
        raise NotImplementedError

    @property
    @abstractmethod
    def total_dataset_length(self) -> int:
        """Get the total length of the underlying dataset."""
        raise NotImplementedError

    @abstractmethod
    def get_total_batch_num(
        self, num_workers: int, batch_size: int = 1, drop_last: bool = False
    ) -> int:
        """Calculate the total number of batches for the dataset.

        Pytorch `DataLoader` with multiple workers will shard the dataset into
        `num_workers` shards, and the default method to calculate the total
        number of batches does not consider the sharding, which will cause
        inaccurate total batch number when using multiple workers. This method
        provides a way to calculate the actual batch number.

        Note:
            The parameters should be the same as the parameters used in the
            DataLoader, otherwise the calculated batch number may
            be inaccurate.

        Args:
            num_workers (int): The number of workers to use for loading
                the data.
            batch_size (int, optional): The batch size to use for loading
                the data. Defaults to 1.
            drop_last (bool, optional): Whether to drop the last incomplete
                batch. Defaults to False.

        """
        raise NotImplementedError


class DatasetWithIndices(TorchDataset, Generic[DatasetType]):
    """A dataset wrapper that allows indexing with an IndiceTable.

    Args:
        dataset (DatasetType): The underlying dataset to wrap.
        indices (IndiceTable | None): An optional IndiceTable to specify which
            indices of the dataset to use. If None, all indices will be used.

    """

    dataset: DatasetType
    indices: IndiceTable

    def __init__(
        self, dataset: DatasetType, indices: IndiceTable | None = None
    ):
        self.dataset = dataset
        if indices is None:
            if isinstance(dataset, Sized):
                indices = IndiceTable(len(dataset))
            else:
                raise ValueError(
                    "Dataset does not have a length, indices must be provided."
                )
        self.indices = indices

    def shard(
        self,
        num_shards: int,
        index: int,
        contiguous: bool = True,
        shard_strategy: ShardStrategy | None = None,
    ):
        """Shard the dataset into multiple shards.

        Args:
            num_shards (int): The total number of shards to create.
            index (int): The ID of the shard to return. Must be in the
                range [0, num_shards - 1].
            contiguous (bool, optional): Whether to create contiguous shards.
                If True, each shard will contain contiguous indices. If False,
                the indices will be distributed in a round-robin fashion.
                Defaults to True.
            shard_strategy (ShardStrategy | None, optional): The strategy to
                use for sharding the dataset. If None, the default strategy
                will be used, which is to drop the last incomplete shard if
                the total number of indices is not divisible by the number of
                shards. Defaults to None.
        """
        return DatasetWithIndices(
            dataset=self.dataset,
            indices=self.indices.shard(
                num_shards=num_shards,
                shard_id=index,
                contiguous=contiguous,
                shard_strategy=shard_strategy,
            ),
        )

    def shuffle(
        self,
        generator: torch.Generator | np.random.Generator | None = None,
    ):
        """Shuffle the dataset indices.

        Args:
            generator (torch.Generator | np.random.Generator | None): An
                optional generator to use for shuffling. If None, a new
                generator will be created with a random seed.

        """
        return DatasetWithIndices(
            dataset=self.dataset,
            indices=self.indices.shuffle(generator),
        )

    def take(
        self, key: int | slice | range | Iterator[int]
    ) -> DatasetWithIndices:
        """Return a new DatasetWithIndices with the rows specified by key."""
        return DatasetWithIndices(
            dataset=self.dataset,
            indices=self.indices.take(key),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({repr(self.dataset)}, "
            f"indices={repr(self.indices)})"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        actual_index = self.indices[index]
        return self.dataset[actual_index]

    def __getitems__(self, index: list[int]) -> list:
        if hasattr(self.dataset, "__getitems__"):
            actual_indices = [self.indices[i] for i in index]
            return self.dataset.__getitems__(actual_indices)  # type: ignore

        else:
            return [self.dataset[self.indices[i]] for i in index]

    def to_iterable_dataset(
        self,
        shuffle: bool | ShuffleConfig = False,
        shard_kwargs: ShardConfig | None = None,
        generator: torch.Generator | np.random.Generator | None = None,
        batch_loader_kwargs: BatchLoaderConfig | dict | None = None,
    ) -> IterableWithLenDataset[DatasetType]:
        return IterableWithLenDataset(
            dataset=self.dataset,
            indices=self.indices,
            shuffle=shuffle,
            shard_kwargs=shard_kwargs,
            generator=generator,
            batch_loader_kwargs=batch_loader_kwargs,
        )


class IterableWithLenDataset(
    TorchIterableDataset, IterableDatasetMixin, Generic[DatasetType]
):
    """A Iterable dataset wrapper that allows indexing with an IndiceTable.

    This class is designed to be compatible with PyTorch's DataLoader with
    multiple workers. When used with multiple workers, each worker will only
    iterate over its own shard of the data.

    Note:
        The purpose of this class is to provide a way to wrap an indexable
        dataset as an iterable dataset. This is useful when we partition
        a very large dataset into multiple subsets/chunks that can be indexed,
        but we want to load them in an iterable way to save resources.
        The input dataset should be indexable with an IndiceTable, and the
        indices should be compatible with the sharding strategy used in
        the DataLoader.

          At runtime this wrapper has two distinct iteration modes:

          1. If ``batch_loader_kwargs`` is None, it yields individual samples by
              resolving indices from ``indice_sampler``. In this mode outer
              PyTorch worker sharding is applied directly to the sampler.
          2. If ``batch_loader_kwargs`` is set, it builds an inner
              single-process dataloader over the current dataset view and lets
              that inner loader form ready-made batches. The outer loader then
              only forwards those ready-made batches.

          ``__iter__`` wraps either mode with optional prefetch buffering when
          sample-level iteration is active.

    Args:
        dataset (DatasetType): The underlying dataset to wrap.
        indices (IndiceTable | None): An optional IndiceTable to specify which
            indices of the dataset to use. If None, all indices will be used.
        shuffle (bool | ShuffleConfig, optional): Whether to shuffle the dataset
            indices. If a ShuffleConfig is provided, it will be used to configure
            the shuffling behavior. Defaults to False, which means no shuffling
            will be applied.
        shard_kwargs (ShardConfig | None, optional): Configuration for
            sharding the dataset. Sharding will be applied when using multiple
            processors in `accelerate`. Defaults to None, which means the
            default sharding strategy will be used (contiguous shards).
        generator (torch.Generator | np.random.Generator | None, optional): An
            optional generator to use for shuffling. If None, a new generator
            will be created with a random seed. Defaults to None.
        batch_loader_kwargs (BatchLoaderConfig | dict | None, optional): An
            optional configuration for using a batch loader. If provided, the
            dataset will be wrapped with a DataLoader to return batches
            of data. Defaults to None, which means no batch loader will
            be used.

    """  # noqa: E501

    dataset: DatasetType
    indice_sampler: IndiceTableSampler
    _batch_loader_kwargs: BatchLoaderConfig | None

    def __init__(
        self,
        dataset: DatasetType,
        indices: IndiceTable | ChunkedIndiceTable | None = None,
        shuffle: bool | ShuffleConfig = False,
        shard_kwargs: ShardConfig | None = None,
        generator: torch.Generator | np.random.Generator | None = None,
        batch_loader_kwargs: BatchLoaderConfig | dict | None = None,
    ):
        logger.debug(
            "Initializing IterableWithLenDataset with shuffle config: %s, "
            "shard config: %s and batch loader kwargs: %s",
            shuffle,
            shard_kwargs,
            batch_loader_kwargs,
        )
        self.dataset = dataset
        indices = self._resolve_indices(dataset, indices)
        self._shuffle_config = self._normalize_shuffle_config(shuffle)

        self.indice_sampler = self._create_indice_sampler(
            indices=indices,
            shuffle_config=self._shuffle_config,
            generator=generator,
        )

        # add to base classes but not inherit to avoid unnecessary methods.
        # prefer modifying class bases, but allow instance-level fallback
        # _add_hf_iterable_cls(self.__class__, instance=self)

        self._shard_kwargs = (
            shard_kwargs if shard_kwargs is not None else ShardConfig()
        )
        self._batch_loader_kwargs = self._normalize_batch_loader_kwargs(
            batch_loader_kwargs
        )

    @staticmethod
    def _normalize_shuffle_config(
        shuffle: bool | ShuffleConfig,
    ) -> ShuffleConfig:
        if isinstance(shuffle, bool):
            return ShuffleConfig(shuffle=shuffle)
        return shuffle

    @staticmethod
    def _resolve_indices(
        dataset: DatasetType,
        indices: IndiceTable | ChunkedIndiceTable | None,
    ) -> IndiceTable | ChunkedIndiceTable:
        if indices is not None:
            return indices
        if isinstance(dataset, Sized):
            return IndiceTable(len(dataset))
        raise ValueError(
            "Dataset does not have a length, indices must be provided."
        )

    @staticmethod
    def _create_indice_sampler(
        indices: IndiceTable | ChunkedIndiceTable,
        shuffle_config: ShuffleConfig,
        generator: torch.Generator | np.random.Generator | None,
    ) -> IndiceTableSampler:
        return IndiceTableSampler(
            indices=indices,
            shuffle=shuffle_config.shuffle,
            generator=generator,
            shuffle_chunk_size=(
                shuffle_config.chunk_size
                if not isinstance(indices, ChunkedIndiceTable)
                else None
            ),
        )

    @staticmethod
    def _normalize_batch_loader_kwargs(
        batch_loader_kwargs: BatchLoaderConfig | dict | None,
    ) -> BatchLoaderConfig | None:
        if isinstance(batch_loader_kwargs, dict):
            return BatchLoaderConfig(**batch_loader_kwargs)
        return batch_loader_kwargs

    @property
    def batch_loader_kwargs(self) -> BatchLoaderConfig | None:
        return self._batch_loader_kwargs

    @property
    def shard_kwargs(self) -> ShardConfig:
        return self._shard_kwargs

    def shuffle_indices(self):
        """Shuffle the dataset indices."""
        self.indice_sampler.shuffle_indices()

    def shard(self, num_shards: int, index: int):
        """Shard the dataset into multiple shards.

        Args:
            num_shards (int): The total number of shards to create.
            index (int): The ID of the shard to return. Must be in the
                range [0, num_shards - 1].

        Returns:
            IterableWithLenDataset[DatasetType]: A new dataset view with the
                same shuffle and batching configuration, but restricted to the
                selected shard of indices.
        """
        shard_sampler = self.indice_sampler.shard(
            num_shards=num_shards,
            shard_id=index,
            contiguous=self.shard_kwargs.contiguous,
        )
        return IterableWithLenDataset(
            dataset=self.dataset,
            indices=shard_sampler.table,
            shard_kwargs=self.shard_kwargs,
            shuffle=self._shuffle_config,
            generator=shard_sampler.generator,
            batch_loader_kwargs=self.batch_loader_kwargs,
        )

    def take(
        self, key: int | slice | range | Iterator[int]
    ) -> IterableWithLenDataset[DatasetType]:
        """Return a new IterableWithLenDataset with the rows specified by key."""  # noqa: E501
        return IterableWithLenDataset(
            dataset=self.dataset,
            indices=self.indice_sampler.table.take(key),
            shard_kwargs=self.shard_kwargs,
            shuffle=self._shuffle_config,
            generator=self.indice_sampler.generator,
            batch_loader_kwargs=self.batch_loader_kwargs,
        )

    def iter(self):
        """Iterate over the current dataset view.

        This method does not apply outer PyTorch worker sharding by itself;
        ``__iter__`` chooses the worker-local view first and then delegates
        here.

        Returns:
            Iterator[Any]: Either individual samples or ready-made batches,
                depending on whether ``batch_loader_kwargs`` is configured.

        """
        if self.batch_loader_kwargs is None:
            logger.debug("Iterating without batch loader,...")
            yield from self._iter_indices(self.indice_sampler)
            return

        logger.debug(
            "Iterating with batch loader, shuffle: %s, batch loader: %s",
            self._shuffle_config,
            self.batch_loader_kwargs,
        )
        yield from self._create_inner_batch_loader()

    def _iter_indices(self, indice_iter: Iterable[int]) -> Iterator[Any]:
        """Yield samples for the provided indices.

        When the wrapped dataset implements ``__getitems__``, this helper uses
        small index batches to amortize indexing overhead while still exposing
        a sample-by-sample iterator to callers.
        """
        yield from _batched_iterator_with_indices(
            self.dataset,
            indice_iter,
        )

    def _create_inner_batch_loader(self) -> TorchDataLoader:
        """Build the inner dataloader used for dataset-side batching.

        The inner loader always uses ``num_workers=0``. Worker/process sharding
        has already been decided by the surrounding ``IterableWithLenDataset``
        instance, so spawning another worker pool here would duplicate that
        logic and make nested batching much harder to reason about.
        """
        assert self.batch_loader_kwargs is not None
        # create a DataLoader with 0 worker to load batches of data,
        # and the sharding will be handled by the DataLoader's worker
        # initialization function.
        return torch.utils.data.DataLoader(
            dataset=IterableWithLenDataset(
                dataset=self.dataset,
                indices=self.indice_sampler.table,
                shard_kwargs=self.shard_kwargs,
                shuffle=self._shuffle_config,
                generator=self.indice_sampler.generator,
                batch_loader_kwargs=None,
            ),
            num_workers=0,
            **self.batch_loader_kwargs.to_dict(),
        )

    def _torch_iter(self):
        """Iterate over the dataset and yield data samples.

        This method is designed to be compatible with PyTorch's DataLoader with
        multiple workers.

        In plain sample mode, worker sharding happens here by slicing the
        sampler per worker. In dataset-side batching mode, the method skips
        that extra branch and delegates to ``iter()``, which rebuilds batches
        from the already worker-local dataset view.
        """
        if (
            self.batch_loader_kwargs is not None
            or not self._is_torch_multi_worker()
        ):
            yield from self.iter()
            return

        yield from self._iter_indices(self._get_multi_worker_sharded_indices())

    def _get_multi_worker_sharded_indices(self) -> IndiceTableSampler:
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is not None
        # do not call shard() here to avoid recursive sharding.
        return self.indice_sampler.shard(
            num_shards=worker_info.num_workers,
            shard_id=worker_info.id,
            contiguous=self.shard_kwargs.contiguous,
        )

    def __iter__(self):
        """Return the public iterator, optionally wrapped with prefetching.

        Prefetch buffering is only added when iteration is still sample-level.
        Once dataset-side batching is active, the inner batching layer remains
        the single source of batch construction.
        """
        yield from _wrap_with_prefetch_if_needed(
            self._torch_iter(),
            shuffle_config=self._shuffle_config,
            generator=self.indice_sampler.generator,
            batch_loader_kwargs=self.batch_loader_kwargs,
        )

    @property
    def total_iterator_length(self) -> int:
        """Get the total number of data samples in the iterator."""
        return len(self.indice_sampler)

    @property
    def total_dataset_length(self) -> int:
        """Get the total length of the underlying dataset."""
        if not isinstance(self.dataset, Sized):
            raise ValueError(
                "Underlying dataset does not have a length, cannot get "
                "total dataset length."
            )
        return len(self.dataset)

    def get_total_batch_num(
        self, num_workers: int, batch_size: int = 1, drop_last: bool = False
    ) -> int:
        """Calculate the total number of batches for the dataset.

        Pytorch `DataLoader` with multiple workers will shard the dataset into
        `num_workers` shards, and the default method to calculate the total
        number of batches does not consider the sharding, which will cause
        inaccurate total batch number when using multiple workers. This method
        provides a way to calculate the actual batch number.

        Note:
            The parameters should be the same as the parameters used in the
            DataLoader, otherwise the calculated batch number may
            be inaccurate.

        Args:
            num_workers (int): The number of workers to use for loading
                the data.
            batch_size (int, optional): The batch size to use for loading
                the data. Defaults to 1.
            drop_last (bool, optional): Whether to drop the last incomplete
                batch. Defaults to False.

        """
        return _get_total_batch_num(
            rows=self.total_iterator_length,
            num_workers=num_workers,
            batch_size=batch_size,
            drop_last=drop_last,
        )

    def _is_torch_multi_worker(self) -> bool:
        import torch.utils.data

        worker_info = torch.utils.data.get_worker_info()
        return worker_info is not None and worker_info.num_workers > 1

    @property
    def n_shards(self) -> int:
        """Get the number of shards for the current dataset.

        Currently this property returns the total number of data samples in the
        iterator.

        Note:
            In most cases, we do not need to know the number of shards, but
            this is reserved to be compatible with `prepare` method in
            accelerate, which needs to know the number of shards to prepare
            the dataset.
        """
        return self.total_iterator_length

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({repr(self.dataset)}, "
            f"indices={repr(self.indice_sampler)}, "
        )


class DatasetItem(Config, Generic[DatasetType], metaclass=ABCMeta):
    """A configuration for creating a dataset.

    User should inherit this class and implement the `_create_dataset` method
    to create a dataset from the configuration, and implement the
    `get_dataset_row_num` method to return the number of rows in the dataset
    before sharding.

    This class also include the sharding information, and the `create_dataset`
    method will apply the sharding to the created dataset. This is useful when
    we want to create a sharded dataset directly from the configuration.
    """

    class_type: ClassType[DatasetType]

    shard_id: int = Field(
        default=0, description="The ID of the shard to return.", ge=0
    )
    num_shards: int = Field(
        default=1, description="The total number of shards to create.", ge=1
    )

    def __post_init__(self):
        if self.shard_id >= self.num_shards:
            raise ValueError(
                f"shard_id must be in the range [0, num_shards - 1], but got "
                f"shard_id={self.shard_id} and num_shards={self.num_shards}."
            )

    @abstractmethod
    def get_dataset_row_num(self) -> int:
        """Get the number of rows in the dataset.

        This method should provide a lightweight way to get the
        number of rows in the dataset. This is important for efficiently
        calculating the total number of batches when using
        multiple workers in a DataLoader.
        """
        raise NotImplementedError(
            "get_dataset_row_num must be implemented by subclasses "
            "of DatasetItem."
        )

    def get_sharded_row_num(self, shard_config: ShardConfig) -> int:
        """Get the number of rows in the sharded dataset.

        This method calculates the number of rows in the dataset after sharding
        based on the sharding configuration.
        """
        total_rows = self.get_dataset_row_num()
        if self.num_shards <= 1:
            return total_rows

        if shard_config.shard_strategy is None:
            # Default sharding strategy: drop the last incomplete shard
            rows_per_shard = total_rows // self.num_shards
            residual = total_rows % self.num_shards
            return rows_per_shard + (1 if self.shard_id < residual else 0)
        elif shard_config.shard_strategy == "drop_last":
            rows_per_shard = total_rows // self.num_shards
            return rows_per_shard
        elif shard_config.shard_strategy == "pad_last":
            rows_per_shard = (
                total_rows + self.num_shards - 1
            ) // self.num_shards
            return rows_per_shard
        else:
            raise ValueError(
                f"Invalid shard strategy: {shard_config.shard_strategy}"
            )

    @abstractmethod
    def _create_dataset(self) -> DatasetType:
        """Create a dataset from the dataset item configuration."""
        raise NotImplementedError(
            "_create_dataset must be implemented by subclasses of DatasetItem."
        )

    def create_dataset(
        self, shard_config: ShardConfig
    ) -> DatasetWithIndices[DatasetType]:
        """Create a DatasetWithIndices from the dataset item configuration.

        This method applies the sharding configuration to the dataset by
        creating a DatasetWithIndices with the appropriate shard of indices.

        """
        ret = DatasetWithIndices(dataset=self._create_dataset())
        if self.is_sharded:
            return ret.shard(
                num_shards=self.num_shards,
                index=self.shard_id,
                **shard_config.to_dict(),
            )
        return ret

    @property
    def is_sharded(self) -> bool:
        return self.num_shards > 1

    def shard(self, num_shards: int, index: int) -> DatasetItem[DatasetType]:
        """Shard the dataset item by returning a new DatasetItem.

        The new DatasetItem will have the same configuration as the original
        one, but with the updated shard_id and num_shards. The new sharding
        information will be calculated by:
        - new_num_shards: self.num_shards * num_shards
        - new_shard_id: self.shard_id * num_shards + index

        Note that the sharding information is always calculated based on the
        original dataset.

        """
        if index >= num_shards:
            raise ValueError(
                f"index must be in the range [0, num_shards - 1], but got "
                f"index={index} and num_shards={num_shards}."
            )
        if index < 0:
            raise ValueError(
                f"index must be non-negative, but got index={index}."
            )
        if num_shards < 1:
            raise ValueError(
                "num_shards must be at least 1, "
                f"but got num_shards={num_shards}."
            )

        return self.replace(
            num_shards=self.num_shards * num_shards,
            shard_id=self.shard_id * num_shards + index,
        )


class DictIterableDataset(TorchIterableDataset, IterableDatasetMixin):
    """A dataset that is created from a list of DatasetItems.

    This dataset will create a DatasetWithIndices for each DatasetItem, and
    iterate over the datasets in a round-robin way. This is useful when we want
    to combine multiple datasets together and load them in an iterable way.

    Args:
        datasets (Iterable[DatasetItem]): An iterable of DatasetItems to create
            the dataset from.
        shuffle (bool | ShuffleConfig, optional): Whether to shuffle the dataset
            indices. If a ShuffleConfig is provided, it will be used to configure
            the shuffling behavior. Defaults to False, which means no shuffling
            will be applied.
        shard_kwargs (ShardConfig | None, optional): Configuration for
            sharding the dataset. Sharding will be applied when using multiple
            processors in `accelerate`. Defaults to None, which means the
            default sharding strategy will be used (contiguous shards).
        generator (torch.Generator | np.random.Generator | None, optional): An
            optional generator to use for shuffling. If None, a new generator
            will be created with a random seed. Defaults to None.
        batch_loader_kwargs (BatchLoaderConfig | dict | None, optional): An
            optional configuration for using a batch loader. If provided, the
            dataset will be wrapped with a DataLoader to return batches of
            data. Defaults to None, which means no batch loader will be used.

    """  # noqa: E501

    dataset_items: list[DatasetItem]

    def __init__(
        self,
        datasets: Iterable[DatasetItem],
        shuffle: bool | ShuffleConfig = False,
        shard_kwargs: ShardConfig | None = None,
        generator: torch.Generator | np.random.Generator | None = None,
        batch_loader_kwargs: BatchLoaderConfig | dict | None = None,
        max_dataset_concurrency: int = 4,
    ):
        # try to make this instance compatible with HF Iterable at class-level
        # or instance-level if class-level MRO change fails
        # _add_hf_iterable_cls(self.__class__, instance=self)
        self.dataset_items = list(datasets)

        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)

        if isinstance(shuffle, bool):
            shuffle = ShuffleConfig(shuffle=shuffle)

        self._shard_kwargs = (
            shard_kwargs if shard_kwargs is not None else ShardConfig()
        )
        self._generator = generator
        self._shuffle = shuffle
        if isinstance(batch_loader_kwargs, dict):
            batch_loader_kwargs = BatchLoaderConfig(**batch_loader_kwargs)
        self._batch_loader_kwargs = batch_loader_kwargs
        self._max_dataset_concurrency = max_dataset_concurrency
        self._total_dataset_length: list[int] | None = None
        self._total_indices_length: list[int] | None = None

    @property
    def batch_loader_kwargs(self) -> BatchLoaderConfig | None:
        return self._batch_loader_kwargs

    @property
    def shard_kwargs(self) -> ShardConfig:
        return self._shard_kwargs

    def shard(self, num_shards: int, index: int) -> DictIterableDataset:
        """Shard the dataset by sharding each dataset item."""
        sharded_items = [
            item.shard(num_shards=num_shards, index=index)
            for item in self.dataset_items
        ]
        return DictIterableDataset(
            datasets=sharded_items,
            shuffle=self._shuffle,
            generator=self._generator,
            batch_loader_kwargs=self.batch_loader_kwargs,
            max_dataset_concurrency=self._max_dataset_concurrency,
            shard_kwargs=self.shard_kwargs,
        )

    def __repr__(self) -> str:
        """Return a safe summary repr for notebook and console display.

        The runtime class also inherits from Hugging Face's
        ``IterableDataset`` for compatibility with downstream integrations.
        That base class expects internal attributes such as ``_info`` and
        ``_ex_iterable`` to exist when building its repr, which this custom
        iterable does not initialize. Defining a local repr keeps interactive
        display and debugging safe without changing the dataset's iteration
        behavior.

        Returns:
            str: Concise summary of the iterable dataset configuration.
        """
        dataset_items_repr = ",\n    ".join(
            repr(item) for item in self.dataset_items
        )
        if dataset_items_repr:
            dataset_items_repr = f"[\n    {dataset_items_repr}\n  ]"
        else:
            dataset_items_repr = "[]"

        return (
            f"{self.__class__.__name__}("
            f"dataset_items={len(self.dataset_items)}, "
            f"items={dataset_items_repr}, "
            f"shuffle={self._shuffle.shuffle}, "
            f"batch_loader_kwargs={self.batch_loader_kwargs!r}, "
            f"max_dataset_concurrency={self._max_dataset_concurrency})"
        )

    @property
    def total_dataset_length(self) -> int:
        if self._total_dataset_length is None:
            self._total_dataset_length = [
                item.get_dataset_row_num() for item in self.dataset_items
            ]
        return sum(self._total_dataset_length)

    @property
    def total_iterator_length(self) -> int:
        if self._total_indices_length is None:
            self._total_indices_length = [
                item.get_sharded_row_num(shard_config=self.shard_kwargs)
                for item in self.dataset_items
            ]
        return sum(self._total_indices_length)

    def get_total_batch_num(
        self, num_workers: int, batch_size: int, drop_last: bool
    ) -> int:
        total_batch_num = 0
        _ = self.total_iterator_length
        assert self._total_indices_length is not None

        if self.batch_loader_kwargs is not None:
            for indices_length in self._total_indices_length:
                total_batch_num += _get_total_batch_num(
                    rows=indices_length,
                    num_workers=num_workers,
                    batch_size=batch_size,
                    drop_last=drop_last,
                )
            return total_batch_num

        if num_workers <= 1:
            return _get_total_batch_num(
                rows=self.total_iterator_length,
                num_workers=1,
                batch_size=batch_size,
                drop_last=drop_last,
            )

        ret = 0
        for workder_id in range(num_workers):
            # get the total number of rows for the worker by
            # summing up the rows for each dataset item.
            total_worker_rows = 0
            for indices_length in self._total_indices_length:
                worker_rows = indices_length // num_workers
                if workder_id < indices_length % num_workers:
                    worker_rows += 1
                total_worker_rows += worker_rows
            ret += _get_total_batch_num(
                rows=total_worker_rows,
                num_workers=1,
                batch_size=batch_size,
                drop_last=drop_last,
            )
        return ret

    @property
    def n_shards(self) -> int:
        """Return an accelerate-compatible shard count hint.

        ``accelerate.prepare_data_loader`` only uses the native Hugging Face
        iterable-dataset sharding path when ``n_shards > num_processes``.
        Keep this value strictly larger than the current process count so
        accelerate prefers dataset-native sharding over its much slower
        ``IterableDatasetShard`` wrapper.
        """
        from accelerate.state import AcceleratorState

        state = AcceleratorState()
        return max(self.total_iterator_length, state.num_processes + 1)

    def _prepare_dataset_for_iter(
        self,
        cur_dataset_iters: list[tuple[int, Generator[Any, None, None]]],
        remaining_dataset_indices: list[int],
    ) -> np.ndarray:
        """Prepare the dataset for iteration and return the sampling weights.

        Args:
            cur_dataset_iters (list[tuple[int, Iterator]]): The current
                dataset iterators.
            remaining_dataset_indices (list[int]): The remaining dataset
                indices to be processed.

        Returns:
            np.ndarray: The sampling weights for each dataset iterator.
        """
        while (
            len(cur_dataset_iters) < self._max_dataset_concurrency
            and len(remaining_dataset_indices) > 0
        ):
            idx = remaining_dataset_indices.pop(0)
            data_item = self.dataset_items[idx]
            iter_dataset = data_item.create_dataset(
                shard_config=self.shard_kwargs
            ).to_iterable_dataset(
                shuffle=self._shuffle,
                shard_kwargs=self.shard_kwargs,
                generator=self._generator,
                batch_loader_kwargs=self.batch_loader_kwargs,
            )
            cur_dataset_iters.append((idx, iter(iter_dataset)))
        assert self._total_indices_length is not None
        weights = []
        for idx, _ in cur_dataset_iters:
            weights.append(self._total_indices_length[idx])
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.sum()
        return weights

    def __iter__(self):
        cur_dataset_iters: list[tuple[int, Generator[Any, None, None]]] = []
        dataset_indices = list(
            IndiceTableSampler(
                len(self.dataset_items),
                shuffle=self._shuffle.shuffle,
                generator=self._generator,
            )
        )
        # Access total_iterator_length to trigger the calculation of total
        # iterator length
        _ = self.total_iterator_length
        assert self._total_indices_length is not None
        weights = self._prepare_dataset_for_iter(
            cur_dataset_iters=cur_dataset_iters,
            remaining_dataset_indices=dataset_indices,
        )

        try:
            while len(cur_dataset_iters) > 0:
                # calulate the sampling weight for each dataset iterator based
                # on the indices length of the corresponding dataset.
                if self._shuffle.shuffle:
                    if isinstance(self._generator, np.random.Generator):
                        selected_idx = self._generator.choice(
                            len(cur_dataset_iters), p=weights, replace=False
                        )
                    elif isinstance(self._generator, torch.Generator):
                        selected_idx = int(
                            torch.multinomial(
                                torch.tensor(weights),
                                1,
                                generator=self._generator,
                            ).item()
                        )
                    else:
                        raise ValueError(
                            "Generator must be either a torch.Generator or a "
                            "numpy.random.Generator."
                        )
                else:
                    selected_idx = 0
                idx, iter_dataset = cur_dataset_iters[selected_idx]
                try:
                    item = next(iter_dataset)
                    yield item
                except StopIteration:
                    cur_dataset_iters.pop(selected_idx)
                    iter_dataset.close()
                    weights = self._prepare_dataset_for_iter(
                        cur_dataset_iters=cur_dataset_iters,
                        remaining_dataset_indices=dataset_indices,
                    )
        finally:
            for _, iter_dataset in cur_dataset_iters:
                iter_dataset.close()


def _get_batch_num(batch_size: int, num_samples: int, drop_last: bool) -> int:
    if drop_last:
        return num_samples // batch_size
    else:
        return (num_samples + batch_size - 1) // batch_size


def _get_total_batch_num(
    rows: int, num_workers: int, batch_size: int = 1, drop_last: bool = False
) -> int:
    """Calculate the total number of batches for the dataset.

    Pytorch `DataLoader` with multiple workers will shard the dataset into
    `num_workers` shards, and the default method to calculate the total
    number of batches does not consider the sharding, which will cause
    inaccurate total batch number when using multiple workers. This method
    provides a way to calculate the actual batch number.

    Note:
        The parameters should be the same as the parameters used in the
        DataLoader, otherwise the calculated batch number may
        be inaccurate.

    Args:
        rows (int): The total number of rows in the dataset.
        num_workers (int): The number of workers to use for loading
            the data.
        batch_size (int, optional): The batch size to use for loading
            the data. Defaults to 1.
        drop_last (bool, optional): Whether to drop the last incomplete
            batch. Defaults to False.

    """
    if num_workers <= 1:
        return _get_batch_num(
            batch_size=batch_size,
            num_samples=rows,
            drop_last=drop_last,
        )
    total_batches = 0
    for worker_id in range(num_workers):
        worker_num_samples = rows // num_workers
        if worker_id < rows % num_workers:
            worker_num_samples += 1

        total_batches += _get_batch_num(
            batch_size=batch_size,
            num_samples=worker_num_samples,
            drop_last=drop_last,
        )
    return total_batches


def _create_prefetch_iterator(
    source_iter: Iterator,
    prefetch_size: int,
    shuffle: bool,
    generator: torch.Generator | np.random.Generator | None,
) -> Iterator:
    """Create a prefetch iterator from the given iterator.

    This function creates a prefetch iterator that prefetches the next
    `prefetch_size` items from the given iterator. This can help improve
    the data loading performance by overlapping the data loading and data
    processing.

    Args:
        source_iter (Iterator): The input iterator to create a prefetch
            iterator from.
        prefetch_size (int): The number of items to prefetch.

    Returns:
        Iterator: A prefetch iterator that yields items from the input iterator
            with prefetching.

    """
    if prefetch_size <= 0:
        raise ValueError("prefetch_size must be greater than 0.")

    if prefetch_size == 1:
        yield from source_iter
        return

    if shuffle and generator is None:
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)

    def shuffle_queue(queue: list) -> list:
        if isinstance(generator, np.random.Generator):
            ret = copy.copy(queue)
            generator.shuffle(ret)
            return ret
        elif isinstance(generator, torch.Generator):
            indices = torch.randperm(len(queue), generator=generator).tolist()
            return [queue[i] for i in indices]
        else:
            raise ValueError(
                "Generator must be either a torch.Generator or a "
                "numpy.random.Generator."
            )

    # `queue` is the mutable buffer currently being filled by the producer
    # thread. Once it reaches a consumable state, the consumer swaps it out as
    # `ready_queue` and replaces `queue` with a fresh list so the producer can
    # continue filling the next window in parallel.
    queue: list[Any] = []
    # A single condition variable protects the shared state below:
    # - `queue`: current fill buffer
    # - `producer_done`: upstream iterator has exited
    # - `consumer_closed`: downstream no longer needs more data
    # - `producer_error`: exception raised by the producer side
    condition = threading.Condition()
    producer_done = False
    consumer_closed = False
    producer_error: BaseException | None = None

    def producer() -> None:
        nonlocal producer_done, producer_error
        try:
            for item in source_iter:
                with condition:
                    # Stop filling when the current buffer is already full.
                    # The consumer will swap in a fresh buffer after it takes
                    # over this full window for shuffle/consumption.
                    while len(queue) >= prefetch_size and not consumer_closed:
                        condition.wait()
                    # If the consumer closed early, exit without touching the
                    # queue again.
                    if consumer_closed:
                        return
                    queue.append(item)
                    # Wake the consumer so it can observe newly available data
                    # or a fully prepared prefetch window.
                    condition.notify_all()
        except BaseException as exc:
            with condition:
                # Record the producer-side failure and let the consumer raise
                # it from the foreground thread on the next check.
                producer_error = exc
        finally:
            with condition:
                # Always mark the producer as done so the consumer can stop
                # waiting even if the upstream iterator exited via exception.
                producer_done = True
                condition.notify_all()

    producer_thread = threading.Thread(
        target=producer,
        name="dataset-prefetch-producer",
        daemon=True,
    )
    producer_thread.start()

    try:
        while True:
            with condition:
                # For shuffled mode we wait until a full window is available so
                # the randomization has enough candidates. For the terminal
                # partial window, `producer_done=True` breaks this wait.
                while (
                    len(queue) < prefetch_size
                    and not producer_done
                    and producer_error is None
                ):
                    condition.wait()

                if producer_error is not None:
                    raise producer_error

                if len(queue) == 0 and producer_done:
                    break

                # Hand the current buffer to the consumer and immediately
                # replace it with a fresh list. Because this happens under the
                # same lock, the producer will see the new buffer atomically
                # and can start filling the next window right away.
                ready_queue = queue
                queue = []
                condition.notify_all()

            if shuffle:
                # Shuffle happens only after a full window is sealed as
                # `ready_queue`, so randomization quality is not degraded by
                # consuming under-filled buffers too early.
                ready_queue = shuffle_queue(ready_queue)

            for item in ready_queue:
                with condition:
                    # Re-check producer failure between yielded items so an
                    # upstream crash is surfaced promptly instead of waiting
                    # until the entire ready window has been drained.
                    if producer_error is not None:
                        raise producer_error
                yield item
    finally:
        with condition:
            # Notify the producer that the consumer is done, including cases
            # where the generator is closed early by the caller.
            consumer_closed = True
            condition.notify_all()
        # Best-effort cleanup only: if the producer is blocked inside
        # `source_iter`, it cannot observe `consumer_closed` yet. Keep close
        # bounded so caller teardown does not hang forever on a stalled
        # upstream iterator.
        producer_thread.join(timeout=_PREFETCH_CLOSE_JOIN_TIMEOUT_SEC)
        if producer_thread.is_alive():
            warnings.warn(
                "Prefetch producer thread did not exit within "
                f"{_PREFETCH_CLOSE_JOIN_TIMEOUT_SEC:.1f}s during close(); "
                "it will finish in the background when the upstream iterator "
                "returns.",
                UserWarning,
            )


def _close_dataloader_iterator(
    dataloader_iter: (
        GeneratorType
        | _SingleProcessDataLoaderIter
        | _MultiProcessingDataLoaderIter
    ),
    _visited: set[int] | None = None,
) -> None:
    """Close a dataloader iterator and the nested iterator layers it owns.

    This helper only tears down resources owned by the active iterator stack.
    Prepared-wrapper lifecycle state such as `accelerate`'s
    `DataLoaderStateMixin` must be ended separately by the owner that
    prepared the dataloader.
    """

    if _visited is None:
        _visited = set()

    iterator_id = id(dataloader_iter)
    if iterator_id in _visited:
        return
    _visited.add(iterator_id)

    if isinstance(dataloader_iter, GeneratorType):
        generator_locals = inspect.getgeneratorlocals(dataloader_iter)
        for nested_iter_name in ("dataloader_iter", "main_iterator"):
            nested_dataloader_iter = generator_locals.get(nested_iter_name)
            if isinstance(
                nested_dataloader_iter,
                (
                    GeneratorType,
                    _SingleProcessDataLoaderIter,
                    _MultiProcessingDataLoaderIter,
                ),
            ):
                _close_dataloader_iterator(nested_dataloader_iter, _visited)
        dataloader_iter.close()
        return

    if isinstance(dataloader_iter, _SingleProcessDataLoaderIter):
        if not isinstance(
            dataloader_iter._dataset_fetcher, _IterableDatasetFetcher
        ):
            return
        dataset_iter = dataloader_iter._dataset_fetcher.dataset_iter
        if isinstance(dataset_iter, GeneratorType) or (
            hasattr(dataset_iter, "close") and callable(dataset_iter.close)
        ):
            dataset_iter.close()
        return

    if (
        isinstance(dataloader_iter, _MultiProcessingDataLoaderIter)
        and not dataloader_iter._persistent_workers
    ):
        dataloader_iter._shutdown_workers()


if not TYPE_CHECKING:
    _IterableWithLenDataset = IterableWithLenDataset
    _DictIterableDataset = DictIterableDataset

    class IterableWithLenDataset(
        _IterableWithLenDataset[DatasetType], HFIterableDataset
    ):
        def __init__(self, *args, **kwargs):
            _IterableWithLenDataset.__init__(self, *args, **kwargs)
            self._epoch = 0

    class DictIterableDataset(_DictIterableDataset, HFIterableDataset):
        def __init__(self, *args, **kwargs):
            _DictIterableDataset.__init__(self, *args, **kwargs)
            self._epoch = 0
