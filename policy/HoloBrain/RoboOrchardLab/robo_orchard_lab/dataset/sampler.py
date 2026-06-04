# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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
import os
from abc import ABCMeta, abstractmethod
from typing import Any, Iterator, Literal, Protocol, runtime_checkable

import numpy as np
import pyarrow as pa
import torch
from datasets.arrow_dataset import InMemoryTable, MemoryMappedTable, Table
from datasets.formatting import query_table
from datasets.table import concat_tables
from torch.utils.data.sampler import Sampler
from typing_extensions import TypeAlias

__all__ = [
    "IndiceTable",
    "ChunkedIndiceTable",
    "IndiceTableSampler",
]


@runtime_checkable
class Sized(Protocol):
    """A protocol for object that have a length."""

    def __len__(self) -> int: ...


ShardStrategy: TypeAlias = Literal["drop_last", "pad_last"] | None


@runtime_checkable
class AccessibleByIndex(Protocol):
    """A protocol for object that can be accessed by index."""

    def __getitem__(self, index: int) -> int: ...


class TableMixin(metaclass=ABCMeta):
    @abstractmethod
    def take(self, key: int | slice | range | Iterator[int]): ...

    @abstractmethod
    def shuffle(
        self, generator: torch.Generator | np.random.Generator | None = None
    ):
        """Shuffle the indices and return a new table.

        After shuffling, the order of the indices will be changed, but the
        set of indices will remain the same.

        Args:
            generator (torch.Generator | np.random.Generator | None, optional):
                Generator used in shuffling. If None, a new generator will be
                created with a random seed. Default: None.
        """
        ...

    @abstractmethod
    def to_pylist(self) -> list: ...

    @abstractmethod
    def to_numpy(self) -> np.ndarray: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __iter__(self): ...

    @abstractmethod
    def __getitem__(self, index: int) -> int: ...

    @abstractmethod
    def shard(
        self,
        num_shards: int,
        shard_id: int,
        contiguous: bool = True,
        shard_strategy: ShardStrategy = None,
    ):
        """Shard the indices into num_shards shards.

        Args:
            num_shards (int): Number of shards to divide the dataset into.
            shard_id (int): Index of the current shard.
            contiguous (bool, optional): If True, then the dataset will
                be split into contiguous chunks. If False, then the dataset
                will be split into non-contiguous chunks. Default: True.
            shard_strategy (ShardStrategy, optional): Strategy to handle
                the last few indices if the dataset size is not
                divisible by num_shards. Options are "drop_last" to drop the last
                few indices or "pad_last" to pad the last few indices with the
                beginning indices. If None, then no special handling will be done and
                the last shard may have fewer indices than the others. Default: None.
        """  # noqa: E501
        ...


class IndiceTable(TableMixin):
    """Class that use pyarrow Table to store indices.

    Args:
        indices (Table | list[int] | str | torch.Tensor | np.ndarray): The
            indices to sample from. It can be a pyarrow Table with one column
            of type uint64, a list of integers, a numpy array of integers,
            a torch tensor of integers, or a string representing the path to
            a pyarrow Table file.

    """

    def __init__(
        self,
        indices: (
            Table
            | list[int]
            | str
            | torch.Tensor
            | np.ndarray
            | int
            | pa.Table
        ),
    ):
        if isinstance(indices, Table):
            self.table = indices
        elif isinstance(indices, pa.Table):
            self.table = Table(indices)
        elif isinstance(indices, int):
            self.table = self._list2memtable(
                np.arange(indices, dtype=np.int64)
            )
        elif isinstance(indices, (list, np.ndarray, torch.Tensor)):
            self.table = self._list2memtable(indices)
        elif isinstance(indices, str):
            self.table = self._table_from_file(indices)
        else:
            raise TypeError(
                f"indices must be of type Table, list[int], or str, "
                f"but got {type(indices)}"
            )
        if self.table.num_columns != 1:
            raise ValueError(
                f"indices table must have exactly one column, "
                f"but got {self.table.num_columns}"
            )
        if self.table.column(0).type not in [pa.uint64(), pa.int64()]:
            raise ValueError(
                f"indices table column must be of type uint64 or int64, "
                f"but got {self.table.column(0).type}"
            )

    def _list2memtable(
        self, indices: list[int] | np.ndarray | torch.Tensor
    ) -> InMemoryTable:
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()
        if isinstance(indices, np.ndarray):
            indice_arr = pa.array(indices)
        else:
            indice_arr = pa.array(indices, type=pa.uint64())
        return InMemoryTable.from_arrays([indice_arr], names=["indices"])

    def _table_from_file(self, filepath: str) -> MemoryMappedTable:
        return MemoryMappedTable.from_file(filepath)

    def __getitem__(self, index: Any) -> Any:
        return self.table.column(0)[index].as_py()

    def __len__(self) -> int:
        return self.table.num_rows

    def __iter__(self) -> Iterator[int]:
        for i in range(len(self)):
            yield self[i]

    def take(self, key: int | slice | range | Iterator[int]) -> IndiceTable:
        """Return a new IndiceTable with the rows specified by key."""
        return IndiceTable(query_table(self.table, key))

    def save_to_file(self, filepath: str, reload: bool = False) -> None:
        """Save the indices to a file in pyarrow format.

        After saving, the indices can be reloaded from the file by setting
        `reload` to True. If `reload` is False, the indices will remain in
        memory and can still be accessed without reading from the file.

        """
        with pa.RecordBatchStreamWriter(filepath, self.table.schema) as writer:
            for batch in self.table.to_batches():
                writer.write_batch(batch)
        if reload:
            self.table = self._table_from_file(filepath)

    def shuffle(
        self, generator: torch.Generator | np.random.Generator | None = None
    ) -> IndiceTable:
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
            indices = torch.randperm(len(self), generator=generator).numpy()
        elif isinstance(generator, torch.Generator):
            indices = torch.randperm(len(self), generator=generator).numpy()
        elif isinstance(generator, np.random.Generator):
            # get genrerator seed:
            assert isinstance(generator, np.random.Generator)
            indices = generator.permutation(len(self))
        old_indices = self.to_numpy()
        return IndiceTable(self._list2memtable(old_indices[indices]))

    def to_pylist(self) -> list[int]:
        return self.table.column(0).to_pylist()

    def to_numpy(self) -> np.ndarray:
        return self.table.column(0).to_numpy()

    @property
    def indice_source(self) -> str:
        """Return the source of the indices, either "memory" or "file"."""
        if isinstance(self.table, InMemoryTable):
            return "memory"
        elif isinstance(self.table, MemoryMappedTable):
            return "file://" + os.path.abspath(self.table.path)
        else:
            raise TypeError(
                f"Unsupported table type: {type(self.table)}. "
                f"Expected InMemoryTable or MemoryMappedTable."
            )

    def shard(
        self,
        num_shards: int,
        shard_id: int,
        contiguous: bool = True,
        shard_strategy: ShardStrategy = None,
    ) -> IndiceTable:
        if not 0 <= shard_id < num_shards:
            raise ValueError("shard_id should be in [0, num_shards-1]")

        if shard_strategy == "drop_last":
            table_for_shard = self._prepare_table_shard_even_drop_last(
                num_shards
            )
        elif shard_strategy == "pad_last":
            table_for_shard = self._prepare_table_shard_even_pad_last(
                num_shards
            )
        else:
            table_for_shard = self.table

        return IndiceTable(
            InMemoryTable(
                _shard_table(
                    table_for_shard,
                    num_shards=num_shards,
                    shard_id=shard_id,
                    contiguous=contiguous,
                )
            )
        )

    def _prepare_table_shard_even_drop_last(self, num_shards: int) -> Table:
        dataset_len = len(self)
        drop_num = dataset_len % num_shards
        if drop_num >= dataset_len:
            raise ValueError(
                "IndiceTable has fewer rows than num_shards, cannot "
                "shard by dropping last few indices."
            )
        table_for_shard = (
            Table(query_table(self.table, slice(0, dataset_len - drop_num)))
            if drop_num > 0
            else self.table
        )
        return table_for_shard

    def _prepare_table_shard_even_pad_last(self, num_shards: int) -> Table:
        dataset_len = len(self)
        if dataset_len == 0:
            raise ValueError(
                "Cannot shard an empty indice table when using "
                "pad_last strategy."
            )
        pad_num = (num_shards - dataset_len % num_shards) % num_shards
        if pad_num > 0:
            to_pad = [self[i % dataset_len] for i in range(pad_num)]
            to_pad_table = InMemoryTable.from_arrays(
                [pa.array(to_pad, type=self.table.column(0).type)],
                names=["indices"],
            )
            return concat_tables([self.table, to_pad_table])
        else:
            return self.table

    def __repr__(self) -> str:
        return (
            f"IndiceTable(num_rows={len(self)}, source={self.indice_source})"
        )


class ChunkedIndiceTable(TableMixin):
    """A chunked version of IndiceTable.

    This class wrappes an IndiceTable and provides a way to access
    the indices in chunks. This is useful when the number of indices is very
    large and we want to process them in smaller batches to benefit from
    better cache locality.

    Args:
        indice_table (IndiceTable): The underlying IndiceTable to sample from.
        chunk_size (int): The size of each chunk. The last chunk may be
            smaller than chunk_size if the total number of indices is not
            divisible by chunk_size.
    """  # noqa: E501

    def __init__(
        self,
        indice_table: IndiceTable | list[IndiceTable],
        chunk_size: int | None,
    ):
        if isinstance(indice_table, IndiceTable) and chunk_size is not None:
            MIN_CHUNK_SIZE = 2  # noqa: N806
            chunk_size = min(
                max(chunk_size, MIN_CHUNK_SIZE), len(indice_table)
            )
            # special case for empty indice_table.
            if chunk_size == 0:
                self._full_table = IndiceTable([])
                self.chunks = []
                self._chunk_size = 0
                self._cum_chunk_sizes = np.array([], dtype=np.int64)
                return
            num_chunks = (len(indice_table) + chunk_size - 1) // chunk_size
            self.chunks = [
                indice_table.take(
                    slice(
                        i * chunk_size,
                        min((i + 1) * chunk_size, len(indice_table)),
                    )
                )
                for i in range(num_chunks)
            ]
            self._full_table = indice_table
            self._chunk_size = chunk_size

        elif isinstance(indice_table, list) and chunk_size is None:
            self.chunks = indice_table
            self._full_table = IndiceTable(
                concat_tables([chunk.table for chunk in self.chunks])
            )
            self._chunk_size = None
        else:
            raise ValueError(
                "If indice_table is an IndiceTable, chunk_size must "
                "be provided. "
                "If indice_table is a list of IndiceTable, "
                "chunk_size must be None."
            )

        self._cum_chunk_sizes = np.cumsum(
            [len(chunk) for chunk in self.chunks]
        )

    def __getitem__(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")
        chunk_idx = np.searchsorted(self._cum_chunk_sizes, index, side="right")
        if chunk_idx == 0:
            local_idx = index
        else:
            local_idx = index - self._cum_chunk_sizes[chunk_idx - 1]
        return self.chunks[chunk_idx][local_idx]

    @property
    def chunk_size(self) -> int:
        if self._chunk_size is not None:
            return self._chunk_size
        else:
            if len(self.chunks) == 0:
                return 0
            elif len(self.chunks) == 1:
                return len(self.chunks[0])
            else:
                # For simplicity, we assume all chunks have the same
                # size except the last one. Two chunks are enough to
                # determine the chunk size in this case.
                return max(len(chunk) for chunk in self.chunks[0:2])

    def shuffle(
        self, generator: torch.Generator | np.random.Generator | None = None
    ) -> ChunkedIndiceTable:
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
            chunk_indices = torch.randperm(
                len(self.chunks), generator=generator
            ).numpy()
        elif isinstance(generator, torch.Generator):
            chunk_indices = torch.randperm(
                len(self.chunks), generator=generator
            ).numpy()
        elif isinstance(generator, np.random.Generator):
            # get genrerator seed:
            assert isinstance(generator, np.random.Generator)
            chunk_indices = generator.permutation(len(self.chunks))
        new_chunks = [
            chunk.shuffle(generator=generator) for chunk in self.chunks
        ]
        new_chunks = [new_chunks[i] for i in chunk_indices]
        return ChunkedIndiceTable(indice_table=new_chunks, chunk_size=None)

    def take(
        self, key: int | slice | range | Iterator[int]
    ) -> ChunkedIndiceTable:
        """Return a new ChunkedIndiceTable with the rows specified by key."""

        return ChunkedIndiceTable(
            indice_table=self._full_table.take(key),
            chunk_size=self.chunk_size,
        )

    def to_pylist(self) -> list[int]:
        return self._full_table.to_pylist()

    def to_numpy(self) -> np.ndarray:
        return self._full_table.to_numpy()

    def __len__(self) -> int:
        return len(self._full_table)

    def __iter__(self) -> Iterator[int]:
        for chunk in self.chunks:
            for index in chunk:
                yield index

    def shard(
        self,
        num_shards: int,
        shard_id: int,
        contiguous: bool = True,
        shard_strategy: ShardStrategy = None,
    ):
        if contiguous is False:
            raise TypeError(
                "Non-contiguous sharding is not supported for "
                "ChunkedIndiceTable"
            )
        return ChunkedIndiceTable(
            indice_table=self._full_table.shard(
                num_shards=num_shards,
                shard_id=shard_id,
                contiguous=contiguous,
                shard_strategy=shard_strategy,
            ),
            chunk_size=self.chunk_size,
        )


class IndiceTableSampler(Sampler[int]):
    """Sampler that samples elements from a given list of indices.

    Args:
        indices (Table | list[int] | str | torch.Tensor | np.ndarray): The
            indices to sample from. It can be a pyarrow Table with one column
            of type uint64, a list of integers, a numpy array of integers,
            a torch tensor of integers, or a string representing the path to
            a pyarrow Table file.
        shuffle_chunk_size (int, optional): The size of each chunk for shuffle.
            If provided, the indices will be split into chunks of the given
            size. If None, then no chunking will be done and the indices will
            be treated as a single chunk. Default: None.
        shuffle (bool, optional): If True, then the indices will be shuffled
            before being returned. Default: False.
        generator (torch.Generator, optional): Generator used in sampling.
            Default: None.

    """

    def __init__(
        self,
        indices: (
            Table
            | list[int]
            | str
            | torch.Tensor
            | np.ndarray
            | int
            | IndiceTable
            | ChunkedIndiceTable
        ),
        shuffle: bool = False,
        shuffle_chunk_size: int | None = None,
        generator: torch.Generator | np.random.Generator | None = None,
    ) -> None:
        self.generator = generator
        self.shuffle = shuffle
        if not isinstance(indices, TableMixin):
            self.table = IndiceTable(indices)
        else:
            self.table = indices
        if shuffle_chunk_size is not None and shuffle:
            if not isinstance(self.table, ChunkedIndiceTable):
                self.table = ChunkedIndiceTable(
                    self.table, chunk_size=shuffle_chunk_size
                )
            else:
                raise ValueError(
                    "shuffle_chunk_size should not be provided when "
                    "indices is already a ChunkedIndiceTable."
                )

    def shuffle_indices(self) -> None:
        """Shuffle the indices in place."""
        self.table = self.table.shuffle(generator=self.generator)

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            new_table = self.table.shuffle(generator=self.generator)
            for index in new_table:
                yield index
        else:
            for index in self.table:
                yield index

    def __len__(self) -> int:
        return len(self.table)

    def take(
        self, key: int | slice | range | Iterator[int]
    ) -> IndiceTableSampler:
        """Return a new IndiceTableSampler with the rows specified by key."""
        return IndiceTableSampler(
            indices=self.table.take(key),
            shuffle=self.shuffle,
            generator=self.generator,
        )

    def shard(
        self,
        num_shards: int,
        shard_id: int,
        contiguous: bool = True,
        shard_strategy: ShardStrategy = None,
    ) -> IndiceTableSampler:
        """Shard the indices into num_shards shards.

        Args:
            num_shards (int): Number of shards to divide the dataset into.
            shard_id (int): Index of the current shard.
            contiguous (bool, optional): If True, then the dataset will
                be split into contiguous chunks. If False, then the dataset
                will be split into non-contiguous chunks. Default: True.
            shard_strategy (ShardStrategy, optional): Strategy to handle
                the last few indices if the dataset size is not divisible by
                num_shards. Options are "drop_last" to drop the last
                few indices or "pad_last" to pad the last few indices with the
                beginning indices. If None, then no special handling will be
                done and the last shard may have fewer indices than the others.
                Default: None.

        """
        new_table = self.table.shard(
            num_shards=num_shards,
            shard_id=shard_id,
            contiguous=contiguous,
            shard_strategy=shard_strategy,
        )
        return IndiceTableSampler(
            indices=new_table,
            shuffle=self.shuffle,
            generator=self.generator,
        )


def _shard_table(
    table: Table, num_shards: int, shard_id: int, contiguous: bool = True
) -> Table:
    """Shard the indices into num_shards shards.

    Args:
        num_shards (int): Number of shards to divide the dataset into.
        shard_id (int): Index of the current shard.
        contiguous (bool, optional): If True, then the dataset will
            be split into contiguous chunks. If False, then the dataset
            will be split into non-contiguous chunks. Default: True.
    """
    if not 0 <= shard_id < num_shards:
        raise ValueError("shard_id should be in [0, num_shards-1]")
    dataset_len = len(table)
    if contiguous:
        div = dataset_len // num_shards
        mod = dataset_len % num_shards
        start = div * shard_id + min(shard_id, mod)
        end = start + div + (1 if shard_id < mod else 0)
        new_table = query_table(table, slice(start, end))
    else:
        new_table = query_table(
            table,
            np.arange(shard_id, dataset_len, num_shards, dtype=np.uint64),
        )
    return new_table
