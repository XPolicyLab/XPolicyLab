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

import os
import tempfile
from typing import Type, TypeVar

import numpy as np
import pyarrow as pa
from datasets import (
    Dataset as HFDataset,
    Features,
)
from datasets.arrow_dataset import (
    _align_features,
    _concatenate_map_style_datasets,
)
from datasets.arrow_writer import ArrowWriter
from datasets.features.features import _check_if_features_can_be_aligned
from sqlalchemy import select
from sqlalchemy.orm import Session
from tqdm import tqdm

from robo_orchard_lab.dataset.robot import (
    create_engine,
    create_tables,
    get_local_db_url,
)
from robo_orchard_lab.dataset.robot.dataset import RODataset
from robo_orchard_lab.dataset.robot.db_orm import (
    Episode,
    Instruction,
    Robot,
    Task,
)
from robo_orchard_lab.dataset.robot.db_orm.md5 import MD5ObjCache

__all__ = [
    "create_merged_dataset",
    "merge_datasets",
]

MD5TableType = TypeVar("MD5TableType", bound=Instruction | Robot | Task)


def _merge_md5_table(
    orm_type: Type[MD5TableType],
    src_session: Session,
    dst_session: Session,
    src_index_mapping: np.ndarray | dict[int, int],
    batch_size: int = 500,
) -> None:
    """Merge MD5-based table from src_session to dst_session.

    This method merges the MD5-based table (Instruction, Robot, Task) from the
    source session and write to destination session.
    The src_index_mapping will be updated to map from source index to
    destination index as well for future reference when merging Episode table.

    """
    dst_max_index = (
        dst_session.query(orm_type.index)
        .order_by(orm_type.index.desc())
        .first()
    )
    dst_max_index = dst_max_index[0] if dst_max_index is not None else -1
    dst_next_index = dst_max_index + 1  # next available index in dst
    check_column_names = orm_type.md5_content_fields()

    bar = tqdm(
        unit="rows",
        total=src_session.query(orm_type).count(),
        desc=f" Merging {orm_type.__tablename__} ",
    )

    with bar:

        def batch_processor(
            src_batch: list[MD5TableType], dst_session: Session
        ) -> None:
            nonlocal dst_next_index

            # find existing md5 in dst_session
            batch_md5 = [obj.md5 for obj in src_batch]
            cache = MD5ObjCache[orm_type](
                check_column_names=check_column_names
            )

            cache.extend(
                dst_session.scalars(
                    select(orm_type).where(orm_type.md5.in_(batch_md5))
                ).all()
            )
            # check each src_obj, if exists in dst_session (cache),
            # if yes, use existing index, if not, create new dst_obj and
            # add to dst_session
            for src_obj in src_batch:
                if (existing_obj := cache.find(src_obj)) is not None:
                    # already exists, use existing index
                    src_index_mapping[src_obj.index] = existing_obj.index
                    continue
                # create new dst_obj and add to dst_session
                dst_obj_dict = {}
                orm_type.column_copy(src_obj, dst_obj_dict)
                dst_obj = orm_type(**dst_obj_dict)
                dst_obj.index = dst_next_index
                src_index_mapping[src_obj.index] = dst_next_index
                dst_next_index += 1
                dst_session.add(dst_obj)

            dst_session.commit()  # commit the batch to get indexes assigned
            bar.update(len(src_batch))

        batch = []
        for src_obj in src_session.scalars(select(orm_type)):
            # cache src_obj data as batch
            batch.append(src_obj)
            if len(batch) < batch_size:
                continue
            else:
                batch_processor(batch, dst_session)
                batch.clear()

        if len(batch) > 0:
            batch_processor(batch, dst_session)
            batch.clear()


def _merge_episode_table(
    src_session: Session,
    dst_session: Session,
    robot_mapping: np.ndarray | dict[int, int],
    task_mapping: np.ndarray | dict[int, int],
    src_index_mapping: np.ndarray | dict[int, int],
    batch_size: int = 500,
):
    """Merge Episode table from src_session to dst_session."""
    # get next available episode index in dst_session
    src_max_episode_index = (
        dst_session.query(Episode.index).order_by(Episode.index.desc()).first()
    )
    src_max_episode_index = (
        src_max_episode_index[0] if src_max_episode_index is not None else -1
    )
    next_dst_episode_index = src_max_episode_index + 1

    with tqdm(
        unit="rows",
        total=src_session.query(Episode).count(),
        desc=f" Merging {Episode.__tablename__} ",
    ) as bar:
        for i, src_episode in enumerate(src_session.scalars(select(Episode))):
            dst_episode_dict = {}
            Episode.column_copy(src_episode, dst_episode_dict)
            new_episode = Episode(**dst_episode_dict)
            # remap foreign keys
            if new_episode.task_index is not None:
                new_episode.task_index = int(
                    task_mapping[new_episode.task_index]
                )
            if new_episode.robot_index is not None:
                new_episode.robot_index = int(
                    robot_mapping[new_episode.robot_index]
                )
            new_episode.index = next_dst_episode_index
            src_index_mapping[src_episode.index] = next_dst_episode_index
            next_dst_episode_index += 1
            dst_session.add(new_episode)

            if (i + 1) % batch_size == 0:
                dst_session.commit()
            bar.update(1)
        dst_session.commit()


def _merge_meta_db(
    src_session: Session, dst_session: Session, cache_dir: str | None
) -> dict[str, np.ndarray | dict[int, int]]:
    """Merge meta database from src_session to dst_session.

    Args:
        src_session (Session): The source database session.
        dst_session (Session): The destination database session.
        cache_dir: str | None: The directory to store the index mapping
            files. If None, use memory dict instead.

    Returns:
        dict[str, np.ndarray | dict[int, int]]: A mapping from table name
            to index mapping array. Each index mapping array maps from
            source index to destination index.
    """

    def prepare_index_mapping(
        cache_dir: str | None,
        orm_type: Type[Instruction | Robot | Task | Episode],
    ) -> np.ndarray | dict[int, int]:
        if cache_dir is None:
            return {}
        max_src_index = (
            src_session.query(orm_type.index)
            .order_by(orm_type.index.desc())
            .first()
        )
        if max_src_index is None:
            return {}
        max_src_index = max_src_index[0] if max_src_index is not None else -1
        max_src_index += 1
        # create mmap numpy array to store the mapping
        # from src index to dst index
        mapping_file = os.path.join(
            cache_dir, f"{orm_type.__tablename__}_index_mapping.dat"
        )
        if os.path.exists(mapping_file):
            os.remove(mapping_file)

        index_mapping = np.memmap(
            filename=mapping_file,
            dtype=np.int64,
            mode="w+",
            shape=(max_src_index,),
        )
        index_mapping[:] = -1  # initialize to -1
        return index_mapping

    src_index_mapping: dict[str, np.ndarray | dict[int, int]] = {}

    for orm_type in (Instruction, Robot, Task):
        index_mapping = prepare_index_mapping(
            cache_dir=cache_dir, orm_type=orm_type
        )
        src_index_mapping[orm_type.__tablename__] = index_mapping
        _merge_md5_table(
            orm_type=orm_type,
            src_session=src_session,
            dst_session=dst_session,
            src_index_mapping=index_mapping,
        )
    # episode_index mapping
    episode_index_mapping = prepare_index_mapping(
        cache_dir=cache_dir, orm_type=Episode
    )
    _merge_episode_table(
        src_session=src_session,
        dst_session=dst_session,
        robot_mapping=src_index_mapping[Robot.__tablename__],
        task_mapping=src_index_mapping[Task.__tablename__],
        src_index_mapping=episode_index_mapping,
    )
    src_index_mapping[Episode.__tablename__] = episode_index_mapping

    return src_index_mapping


def _remap_meta_index(
    frame_dataset: HFDataset,
    index_mapping: dict[str, np.ndarray | dict[int, int]],
    target_index_start: int,
    cached_index_path: str,
) -> HFDataset:
    """Remap meta index columns in the frame dataset.

    All meta index columns (instruction_index, task_index, robot_index,
    episode_index) will be remapped according to the provided index_mapping.
    The `index` column will be offset by `target_index_start`.

    The remapped columns will be saved to a new Arrow file at
    `cached_index_path`.

    Args:
        frame_dataset (HFDataset): The frame dataset to remap.
        index_mapping (dict[str, np.ndarray|dict[int, int]]): A mapping from
            meta table name to index mapping array.
        target_index_start (int): The starting index for the `index` column
            offset.
        cached_index_path (str): The path to save the remapped Arrow file.

    Returns:
        HFDataset: The remapped frame dataset.

    """

    mapped_arrays = []

    # remap meta index
    mapping_columns = [
        ("instruction_index", "instruction"),
        ("task_index", "task"),
        ("robot_index", "robot"),
        ("episode_index", "episode"),
    ]
    new_column: list[int | None] = []
    for column_name, src_name in tqdm(
        mapping_columns, desc=" Remapping columns "
    ):
        column_idx_mapping = index_mapping[src_name]
        origin_column = frame_dataset[column_name]
        if isinstance(column_idx_mapping, dict):
            new_column: list[int | None] = []
            for idx in origin_column:
                if idx is None:
                    new_column.append(None)
                else:
                    new_column.append(column_idx_mapping.get(int(idx), -1))

        else:
            new_column: list[int | None] = []
            for idx in origin_column:
                if idx is None:
                    new_column.append(None)
                else:
                    new_column.append(column_idx_mapping[idx])

        assert isinstance(new_column, list)
        if any(idx is not None and idx < 0 for idx in new_column):
            raise ValueError(f"Invalid index found in column {column_name}.")
        mapped_arrays.append(pa.array(new_column, pa.int64()))

    # Offset `index` by target_index_start
    origin_index_column = frame_dataset["index"]
    new_index_column = (
        np.array(origin_index_column, dtype=np.int64) + target_index_start
    )
    mapped_arrays.append(pa.array(new_index_column))
    mapping_columns.append(("index", None))  # type: ignore
    mapped_column_names = [k for k, _ in mapping_columns]

    # Write to Arrow file
    with tqdm(total=1, desc=" Writing remapped index columns ") as pbar:
        table = pa.Table.from_arrays(mapped_arrays, names=mapped_column_names)
        feature = Features(
            {k: frame_dataset.features[k] for k in mapped_column_names}
        )
        instruction_column_writer = ArrowWriter(
            path=cached_index_path, features=feature
        )
        instruction_column_writer.write_table(table)
        instruction_column_writer.finalize()
        instruction_column_writer.close()
        mapped_idx_dataset = HFDataset.from_file(cached_index_path)
        mapped_dataset = _concatenate_map_style_datasets(
            [
                frame_dataset.select_columns(
                    [
                        k
                        for k in frame_dataset.features.keys()
                        if k not in mapped_column_names
                    ]
                ),
                mapped_idx_dataset,
            ],
            axis=1,
        )
        pbar.update(1)

    return mapped_dataset


def create_merged_dataset(
    datasets: list[RODataset],
    cache_dir: str,
    db_driver: str = "duckdb",
    cache_meta_idx_mappings_in_memory: bool = False,
) -> RODataset:
    """Merge multiple RODatasets into a single RODataset.

    Args:
        datasets (list[RODataset]): The list of RODatasets to merge.
        cache_dir (str): The directory to store the merged meta database
            and temporary files.
            Note that this directory should not be deleted until the merged
            dataset is no longer needed, as the merged dataset relies on
            the meta database stored in this directory.
        db_driver (str, optional): The database driver to use for the
            merged meta database. Defaults to "duckdb".
        cache_meta_idx_mappings_in_memory (bool, optional): Whether to cache
            the source to destination index mappings in memory.
            If False, the mappings will be stored in a memory mapped temporary
            files in `cache_dir`, which use dense numpy arrays.
            If True, the mappings will be stored in memory using dicts,
            which may use more memory but may be faster for sources with
            sparse indices. Defaults to False.

    Returns:
        RODataset: The merged RODataset.
    """
    if len(datasets) == 0:
        raise ValueError("datasets cannot be empty.")
    # check and align features
    features_list = [ds.frame_dataset.features for ds in datasets]
    _check_if_features_can_be_aligned(features_list)
    features_list = _align_features(features_list)
    features = Features(
        {k: v for features in features_list for k, v in features.items()}
    )
    # initialize meta database
    db_file_name = f"meta_db.{db_driver}"
    dst_db_path = os.path.join(cache_dir, db_file_name)
    if os.path.exists(dst_db_path):
        os.remove(dst_db_path)
    meta_db_engine = create_engine(
        url=get_local_db_url(
            drivername=db_driver,
            db_path=dst_db_path,
        ),
        echo=False,
    )
    create_tables(engine=meta_db_engine)
    frame_dataset = HFDataset.from_dict(
        features=features, mapping={k: [] for k in features.keys()}
    )
    for i, dataset in enumerate(tqdm(datasets, desc="Merging datasets")):
        # create a temporary cache directory for each dataset
        with tempfile.TemporaryDirectory() as local_cache_dir:
            with (
                Session(dataset.db_engine) as src_session,
                Session(meta_db_engine) as dst_session,
            ):
                src_idx_mapping = _merge_meta_db(
                    src_session=src_session,
                    dst_session=dst_session,
                    cache_dir=(
                        local_cache_dir
                        if not cache_meta_idx_mappings_in_memory
                        else None
                    ),
                )
            target_path = os.path.join(
                cache_dir, f"mapped_meta_index_{i}.arrow"
            )
            mapped_dataset = _remap_meta_index(
                frame_dataset=dataset.frame_dataset,
                index_mapping=src_idx_mapping,
                cached_index_path=target_path,
                target_index_start=len(frame_dataset),
            )
            src_idx_mapping = None
        with tqdm(total=1, desc=" Concatenating datasets ") as pbar:
            frame_dataset = _concatenate_map_style_datasets(
                [frame_dataset, mapped_dataset],
                axis=0,
                info=None,
                split=None,
            )
            pbar.update(1)
    meta_db_engine.dispose()
    meta_db_engine = create_engine(
        url=get_local_db_url(
            drivername=db_driver,
            db_path=dst_db_path,
        ),
        echo=False,
        readonly=True,
    )
    return RODataset.from_dataset(
        frame_dataset=frame_dataset,
        meta_db_engine=meta_db_engine,
    )


def merge_datasets(
    datasets: list[RODataset],
    target_path: str,
    db_driver: str = "duckdb",
    cache_meta_idx_mappings_in_memory: bool = False,
    max_shard_size: str | int = "8000MB",
    num_shards: int | None = None,
    num_proc: int | None = None,
    storage_options: dict | None = None,
    batch_size: int | None = None,
):
    """Merge multiple RODatasets and save to disk.

    Args:
        max_shard_size (str | int , optional): The maximum size of
            each shard. Defaults to "8000MB". This can be a string
            (e.g., "8000MB") or an integer (e.g., 8000 * 1024 * 1024
            for 8000MB).
        num_shards (int | None, optional): The number of shards to create.
            Number of shards to write. By default the number of shards
            depends on `max_shard_size` and `num_proc`.
        num_proc (int | None, optional): The number of processes to use
            for saving the dataset. Defaults to None.
        storage_options (dict | None, optional): Additional Key/value pairs
            to be passed on to the file-system backend, if any. Defaults
            to None.
        batch_size (int | None, optional): The batch size to use when
            saving the dataset. If None, the default batch size from
            Hugging Face Datasets will be used. Defaults to None.
    """
    with tempfile.TemporaryDirectory() as local_cache_dir:
        merged_dataset = create_merged_dataset(
            datasets=datasets,
            cache_dir=local_cache_dir,
            db_driver=db_driver,
            cache_meta_idx_mappings_in_memory=(
                cache_meta_idx_mappings_in_memory
            ),
        )
        with tqdm(total=1, desc=" Saving merged dataset to disk ") as pbar:
            merged_dataset.save_to_disk(
                target_path,
                max_shard_size=max_shard_size,
                num_shards=num_shards,
                num_proc=num_proc,
                storage_options=storage_options,
                batch_size=batch_size,
            )
            pbar.update(1)
