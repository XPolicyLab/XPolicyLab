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

"""Packaging a RoboOrchard Dataset."""

from __future__ import annotations
import json
import os
import pickle
import warnings
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Generator, Iterable

import datasets as hg_datasets
import fsspec
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, make_transient
from typing_extensions import deprecated

from robo_orchard_lab.dataset.robot import (
    create_engine,
    create_tables,
    get_local_db_url,
)
from robo_orchard_lab.dataset.robot.columns import (
    PreservedColumnsKeys,
    PreservedIndexColumns,
    PreservedIndexColumnsKeys,
)
from robo_orchard_lab.dataset.robot.db_orm import (
    Episode,
    Instruction,
    Robot,
    RobotDescriptionFormat,
    Task,
)

__all__ = [
    "DatasetPackaging",
    "EpisodePackaging",
    "EpisodeData",
    "RobotData",
    "TaskData",
    "InstructionData",
    "EpisodeMeta",
    "DataFrame",
]

dataset_format_version = "0.1.0"


@dataclass
class EpisodeMeta:
    """Metadata for an episode packaging in a RoboOrchard dataset.

    This is the data structure used during the packaging process to
    represent the episode information to be stored in the database.
    """

    episode: EpisodeData
    robot: RobotData | None = None
    task: TaskData | None = None

    def get_transient_orm(
        self, index_state: DatasetIndexState, session: Session
    ) -> EpisodeMetaORM:
        """Get the transient ORM instance of the episode metadata."""
        episode = Episode(
            index=index_state.last_episode_idx + 1,
            **self.episode.__dict__,
        )
        robot = (
            self.robot.make_transient_orm(index_state, session=session)
            if self.robot
            else None
        )
        task = (
            self.task.make_transient_orm(index_state, session=session)
            if self.task
            else None
        )

        episode.task_index = task.index if task else None
        episode.robot_index = robot.index if robot else None
        return EpisodeMetaORM(episode=episode, robot=robot, task=task)


@dataclass
class DataFrame:
    """Data for a single frame in a RoboOrchard dataset."""

    features: dict[str, Any]
    instruction: InstructionData | None = None
    timestamp_ns_min: int | None = None
    """The minimum timestamp of the frame in nanoseconds."""
    timestamp_ns_max: int | None = None
    """The maximum timestamp of the frame in nanoseconds."""


class EpisodePackaging(metaclass=ABCMeta):
    @abstractmethod
    def generate_episode_meta(self) -> EpisodeMeta | None:
        """Generate metadata for the episode if it should be included.

        Should return None if the episode is to be skipped.

        Returns:
            EpisodeMeta | None: Metadata containing the episode, robot,
                and task. If None is returned, the episode will be skipped
                during packaging.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses to "
            "generate episode metadata."
        )

    @abstractmethod
    def generate_frames(self) -> Generator[DataFrame, None, None]:
        """Generate frame data for the episode."""
        raise NotImplementedError(
            "This method should be implemented by subclasses to "
            "generate frame data for the episode."
        )


class DatasetPackaging:
    """Class for packaging a RoboOrchard dataset.

    Args:
        features (hg_datasets.Features): The features of the dataset.
        database_driver (str): The database driver to use for the meta
            database. Default is "duckdb".
        check_timestamp (bool, optional): Whether to check the
            timestamp of the frames. Timestamps are required for time-based
            queries and operations. If True, it will raise an error
            if the timestamp is not set or if timestamp_min is greater
            than timestamp_max. Default is False.

    """

    def __init__(
        self,
        features: hg_datasets.Features,
        database_driver: str = "duckdb",
        check_timestamp: bool = False,
    ):
        self._features = self._check_and_update_features(features)
        self._database_driver = database_driver
        self._index_state: DatasetIndexState = DatasetIndexState()
        self._instruction_cache: InstructionCache = InstructionCache()
        self._check_timestamp = check_timestamp

    @property
    def features(self) -> hg_datasets.Features:
        return self._features

    def _check_and_update_features(
        self, features: hg_datasets.Features
    ) -> hg_datasets.Features:
        index_keys = PreservedIndexColumnsKeys
        for key in index_keys:
            if key in features:
                raise ValueError(
                    f"Feature '{key}' is reserved for internal use "
                    "and cannot be used in the dataset features."
                )
        # Add index fields to the features
        ret = features.copy()
        for key in index_keys:
            if key not in ret:
                ret[key] = hg_datasets.Value(dtype="int64")
        return ret

    def _extend_frame_with_index(
        self,
        # features: dict[str, Any],
        frame: DataFrame,
        episode_meta: EpisodeMetaORM,
        instruction: Instruction | None,
    ):
        """Extend the frame with index fields."""
        features = frame.features
        for key in PreservedColumnsKeys:
            if key in features:
                raise ValueError(
                    f"key '{key}' is reserved for internal use "
                    "and cannot be used in the frame features."
                )

        index_columns = PreservedIndexColumns(
            index=self._index_state.last_frame_idx + 1,
            frame_index=self._index_state.last_episode_frame_idx + 1,
            episode_index=episode_meta.episode.index,
            robot_index=episode_meta.robot.index
            if episode_meta.robot
            else None,
            task_index=episode_meta.task.index if episode_meta.task else None,
            instruction_index=instruction.index if instruction else None,
            timestamp_min=frame.timestamp_ns_min,
            timestamp_max=frame.timestamp_ns_max,
        )

        features.update(index_columns.__dict__)

    def _add_instruction(
        self, instruction_data: InstructionData, engine: Engine
    ) -> Instruction:
        """Add an instruction to the database and update the index state."""
        cached_instruction = self._instruction_cache.get(instruction_data)
        if cached_instruction is not None:
            # If the instruction is already cached, return it
            return cached_instruction

        with Session(engine, expire_on_commit=False) as session:
            instruction = instruction_data.make_transient_orm(
                self._index_state, session=session
            )
            if instruction.index > self._index_state.last_instruction_idx:
                session.add(instruction)
                session.commit()
                make_transient(instruction)
                self._instruction_cache.add(instruction)
            return instruction

    def _make_packaging_generator(
        self, episodes: Iterable[EpisodePackaging], db_path: str
    ):
        if os.path.exists(db_path):
            raise FileExistsError(
                f"The meta database path '{db_path}' already exists."
            )
        url = get_local_db_url(
            drivername=self._database_driver, db_path=db_path
        )
        engine = create_engine(url=url, echo=False)
        create_tables(engine=engine)

        def frame_generator(episode: EpisodePackaging):
            try:
                for frame in episode.generate_frames():
                    if self._check_timestamp:
                        if (
                            frame.timestamp_ns_min is None
                            or frame.timestamp_ns_max is None
                        ):
                            raise ValueError(
                                "Frame must have both timestamp_ns_min "
                                "and timestamp_ns_max set."
                            )
                        if frame.timestamp_ns_min > frame.timestamp_ns_max:
                            raise ValueError(
                                "timestamp_ns_min cannot be greater than "
                                "timestamp_ns_max."
                            )

                    yield frame
            except Exception as e:
                warnings.warn(
                    f"Failed to generate frames for {episode}. "
                    f"Skipping this episode. Error: "
                )
                import traceback

                traceback.print_exception(e)
                return

        for episode in episodes:
            try:
                episode_meta = episode.generate_episode_meta()
                if episode_meta is None:
                    continue
            except Exception as e:
                warnings.warn(
                    f"Failed to generate episode metadata for {episode}. "
                    f"Skipping this episode.  Error: "
                )
                import traceback

                traceback.print_exception(e)
                continue

            with Session(engine) as session:
                episode_meta_orm = episode_meta.get_transient_orm(
                    self._index_state, session
                )
            self._index_state.last_episode_frame_idx = -1
            episode_meta_orm.episode.dataset_begin_index = (
                self._index_state.last_frame_idx + 1
            )
            # clear _instruction_cache for each episode
            self._instruction_cache.clear()
            for frame in frame_generator(episode):
                instruction_orm = (
                    self._add_instruction(frame.instruction, engine=engine)
                    if frame.instruction
                    else None
                )
                self._extend_frame_with_index(
                    frame, episode_meta_orm, instruction_orm
                )
                # encode_example here.
                # yield self._features.encode_example(frame.features)
                yield frame.features
                # update status
                if instruction_orm is not None:
                    self._index_state.last_instruction_idx = max(
                        self._index_state.last_instruction_idx,
                        instruction_orm.index,
                    )
                self._index_state.last_episode_frame_idx += 1
                self._index_state.last_frame_idx += 1

            # Update the index state with the episode metadata
            with Session(engine, expire_on_commit=False) as session:
                if episode_meta_orm.robot:
                    if (
                        episode_meta_orm.robot.index
                        > self._index_state.last_robot_idx
                    ):
                        session.add(episode_meta_orm.robot)
                    self._index_state.last_robot_idx = max(
                        self._index_state.last_robot_idx,
                        episode_meta_orm.robot.index,
                    )
                if episode_meta_orm.task:
                    if (
                        episode_meta_orm.task.index
                        > self._index_state.last_task_idx
                    ):
                        session.add(episode_meta_orm.task)
                    self._index_state.last_task_idx = max(
                        self._index_state.last_task_idx,
                        episode_meta_orm.task.index,
                    )
                self._index_state.last_episode_idx = max(
                    self._index_state.last_episode_idx,
                    episode_meta_orm.episode.index,
                )
                # Update the episode metadata in the database
                episode_meta_orm.episode.frame_num = (
                    self._index_state.last_episode_frame_idx + 1
                )
                session.add(episode_meta_orm.episode)
                session.commit()
        engine.dispose()

    def _complete_arrow_cache_as_dataset(
        self,
        dataset_path: str,
        builder: hg_datasets.DatasetBuilder,
        split: hg_datasets.Split | None,
    ):
        dataset_dict = builder.as_dataset(split=split)
        assert isinstance(dataset_dict, hg_datasets.DatasetDict)
        assert len(dataset_dict) == 1
        for k, v in dataset_dict.items():
            dataset: hg_datasets.Dataset = v
            split_name = str(k)
            break
        ori_arrow_prefix = f"{builder.name}-{split_name}"
        fs: fsspec.AbstractFileSystem = fsspec.core.url_to_fs(dataset_path)[0]
        arrow_files: list[str] = fs.glob(
            os.path.join(dataset_path, f"{builder.name}-{split_name}*"),
            maxdepth=1,
        )  # type: ignore

        # rename file names to match the expected format
        # e.g. data-00000-of-00001.arrow
        arrow_files = [
            os.path.relpath(f, dataset_path)
            for f in sorted(
                _rename_arrow_files(
                    arrow_files, fs=fs, source_prefix=ori_arrow_prefix
                )
            )
        ]

        # add the dataset state file as it is required by datasets
        state = {
            key: dataset.__dict__[key]
            for key in [
                "_fingerprint",
                "_format_columns",
                "_format_kwargs",
                "_format_type",
                "_output_all_columns",
            ]
        }

        state["_split"] = (
            str(dataset.split) if dataset.split is not None else dataset.split
        )
        state["_data_files"] = [{"filename": f} for f in arrow_files]
        for k in state["_format_kwargs"].keys():
            try:
                json.dumps(state["_format_kwargs"][k])
            except TypeError as e:
                raise TypeError(
                    str(e) + f"\nThe format kwargs must be JSON serializable, "
                    f"but key '{k}' isn't."
                ) from None
        from datasets import config as hg_datasets_config

        # append robo_orchard_state
        state["robo_orchard_state"] = {}
        state["robo_orchard_state"]["dataset_format_version"] = (
            dataset_format_version
        )
        with fs.open(
            os.path.join(
                dataset_path, hg_datasets_config.DATASET_STATE_JSON_FILENAME
            ),
            "w",
            encoding="utf-8",
        ) as state_file:
            json.dump(state, state_file, indent=2, sort_keys=True)

        # remove lock files if exist.
        # lock files is required during building the dataset,
        # but not needed after the dataset is built.
        lockfiles_postfix = [
            ".incomplete_info.lock",
            "_builder.lock",
        ]
        dataset_parent_dir = os.path.dirname(dataset_path)
        dataset_name = os.path.basename(dataset_path)
        for postfix in lockfiles_postfix:
            lockfile_path = os.path.join(
                dataset_parent_dir, f"{dataset_name}{postfix}"
            )
            if fs.exists(lockfile_path):
                fs.rm(lockfile_path)

    def packaging(
        self,
        episodes: Iterable[EpisodePackaging],
        dataset_path: str,
        dataset_info: hg_datasets.DatasetInfo | None = None,
        writer_batch_size: int = 1024,
        max_shard_size: str | int = "8GB",
        split: hg_datasets.Split | None = None,
        force_overwrite: bool = False,
    ):
        """Package the dataset and save it to the specified path.

        Args:
            episodes (Iterable[EpisodePackaging]): An iterable of episode
                packaging instances.
            dataset_path (str): The path to save the packaged dataset.
            dataset_info (hg_datasets.DatasetInfo | None): Information about
                the dataset, such as description, citation, and homepage.
                If None, use the default dataset info.
            writer_batch_size (int): The batch size for writing the arrow file.
                This may affect the performance of packaging or reading the
                dataset later. Default is 1024.
            max_shard_size (str | int | None): The maximum size of each shard.
                If None, no sharding will be applied. This can be a string
                like '10GB' or an integer representing the size in bytes.
                default is "500MB".
            split (hg_datasets.Split | None): The split of the dataset.
                If None, use "train" as the default split.
            force_overwrite (bool): If True, overwrite the existing dataset
                at the specified path. If False, raise an error if the path
                already exists. Default is False.
        """

        dataset_path = os.path.abspath(os.path.expanduser(dataset_path))

        if os.path.exists(dataset_path):
            if not force_overwrite:
                raise FileExistsError(
                    f"The dataset path '{dataset_path}' already exists. "
                    "Please remove it or set force_overwrite=True to overwrite."  # noqa: E501
                )
            else:
                warnings.warn(
                    f"The dataset path '{dataset_path}' already exists. "
                    "It will be overwritten."
                )
                # Clean up the existing dataset path
                import shutil

                shutil.rmtree(dataset_path, ignore_errors=True)

        self._index_state: DatasetIndexState = DatasetIndexState()
        self._instruction_cache.clear()

        # We cannot use the dataset_path directly because
        # datasets will clean the folder before packaging
        # if it already exists. So we create the
        # database in a temporary path and move it later.

        db_folder = os.path.dirname(dataset_path)
        os.makedirs(db_folder, exist_ok=True)
        db_path = dataset_path + f"_meta.{self._database_driver}"
        if os.path.exists(db_path):
            warnings.warn(
                f"The temporary meta database path '{db_path}' already "
                "exists. It will be overwritten."
            )
            os.remove(db_path)

        db_new_path = os.path.join(
            dataset_path, f"meta_db.{self._database_driver}"
        )
        if os.path.exists(db_new_path) and not force_overwrite:
            raise FileExistsError(
                f"The meta database path '{db_new_path}' already exists. "
                "Please remove it or set force_overwrite=True to overwrite."
            )

        def generator():
            yield from self._make_packaging_generator(
                episodes, db_path=db_path
            )

        try:
            from datasets.packaged_modules.generator.generator import Generator

            builder = Generator(
                generator=generator,
                features=self._features,
                writer_batch_size=writer_batch_size,
                info=dataset_info,
            )
            builder.download_and_prepare(
                output_dir=dataset_path,
                max_shard_size=max_shard_size,
                file_format="arrow",
            )
            db_new_path = os.path.join(
                dataset_path, f"meta_db.{self._database_driver}"
            )
            os.rename(db_path, db_new_path)

            # Complete the dataset with the arrow cache
            self._complete_arrow_cache_as_dataset(
                dataset_path=dataset_path,
                builder=builder,
                split=split,
            )
        except Exception as e:
            raise e

        finally:
            # Clean up the temporary database file if it exists
            if os.path.exists(db_path):
                os.remove(db_path)


@dataclass
class EpisodeData:
    """Data for an episode information which is used for packaging."""

    frame_num: int | None = None
    """The total number of frames in the episode.

    No need to set this field during packaging, it will be updated
    automatically during packaging.
    """
    prev_episode_index: int | None = None
    """The index of the previous episode in the dataset."""
    dataset_begin_index: int | None = None
    """The index of the first dataset item in this episode.

    No need to set this field during packaging, it will be updated
    automatically during packaging.
    """

    truncated: bool | None = None
    """Whether the episode was truncated."""

    success: bool | None = None
    """Whether the episode was successful."""

    info: dict[str, Any] | None = None
    """Additional information about the episode."""


@dataclass
class RobotData:
    """Data for a robot information which is used for packaging."""

    name: str
    """The name of the robot."""

    content: str | None
    content_format: RobotDescriptionFormat | None

    @classmethod
    def from_orm(cls, orm_robot: Robot) -> RobotData:
        """Create a RobotData instance from an ORM Robot instance."""
        return cls(
            name=orm_robot.name,
            content=orm_robot.content,
            content_format=orm_robot.content_format,
        )

    def make_transient_orm(
        self, index_state: DatasetIndexState, session: Session | None
    ) -> Robot:
        """Create a transient ORM instance of the robot.

        If session is provided, it will check if the robot already exists
        in the database using its name and URDF content. If it exists, it
        will return the existing robot instance, otherwise it will create a
        new transient instance with the next index.

        If session is None, it will create a new transient instance with the
        next index without checking the database.

        """

        def make_new():
            ret = Robot(index=index_state.last_robot_idx + 1, **self.__dict__)
            ret.update_md5()
            return ret

        if session is not None:
            ret = Robot.query_by_content_with_md5(session, **self.__dict__)
            if ret is not None:
                # Make sure the robot is not transient
                make_transient(ret)
                return ret
            else:
                return make_new()
        else:
            return make_new()

    @property
    @deprecated("Use 'content' and 'content_format' instead.")  # type: ignore
    def urdf_content(self) -> str | None:
        """The URDF content of the robot."""
        if self.content_format == RobotDescriptionFormat.URDF:
            return self.content
        else:
            return None

    @urdf_content.setter
    @deprecated("Use 'content' and 'content_format' instead.")  # type: ignore
    def urdf_content(self, value: str | None):
        """Set the URDF content of the robot."""
        self.content = value
        self.content_format = RobotDescriptionFormat.URDF


@dataclass
class TaskData:
    """Data for a task information which is used for packaging."""

    name: str
    """The name of the task."""
    description: str | None = None
    """The description of the task."""

    def make_transient_orm(
        self, index_state: DatasetIndexState, session: Session | None
    ) -> Task:
        """Create a transient ORM instance of the task.

        If session is provided, it will check if the task already exists
        in the database using its name and description. If it exists, it
        will return the existing task instance, otherwise it will create a
        new transient instance with the next index.

        If session is None, it will create a new transient instance with the
        next index without checking the database.

        """

        def make_new():
            ret = Task(index=index_state.last_task_idx + 1, **self.__dict__)
            ret.update_md5()
            return ret

        if session is not None:
            ret = Task.query_by_content_with_md5(session, **self.__dict__)
            if ret is not None:
                make_transient(ret)
                return ret
            else:
                return make_new()
        else:
            return make_new()


@dataclass
class EpisodeMetaORM:
    """Metadata for an episode in a RoboOrchard dataset."""

    episode: Episode
    robot: Robot | None = None
    task: Task | None = None


@dataclass
class InstructionData:
    """Data for an instruction information which is used for packaging."""

    name: str | None
    """The name of the instruction."""
    json_content: dict[str, Any] | None
    """The content of the instruction, typically a dictionary with keys like
    'instruction', 'robot', and 'task'.
    """

    def make_transient_orm(
        self, index_state: DatasetIndexState, session: Session | None = None
    ) -> Instruction:
        """Create a transient ORM instance of the instruction.

        If session is provided, it will check if the instruction already exists
        in the database using its name and JSON content. If it exists, it
        will return the existing instruction instance, otherwise it will
        create a new transient instance with the next index.

        If session is None, it will create a new transient instance with the
        next index without checking the database.

        """

        def make_new():
            ret = Instruction(
                index=index_state.last_instruction_idx + 1, **self.__dict__
            )
            ret.update_md5()
            return ret

        if session is not None:
            ret = Instruction.query_by_content_with_md5(
                session, **self.__dict__
            )
            if ret is not None:
                make_transient(ret)
                return ret
            else:
                return make_new()
        else:
            return make_new()


class InstructionCache:
    def __init__(self):
        self._cache: dict[bytes, Instruction] = {}

    def _get_key(self, instruction: InstructionData) -> bytes:
        """Generate a unique key for the instruction data."""
        return pickle.dumps(instruction)

    def get(self, instruction_data: InstructionData) -> Instruction | None:
        """Get an instruction from the cache using its packing data."""
        return self._cache.get(self._get_key(instruction_data), None)

    def add(self, instruction: Instruction):
        """Add an instruction to the cache."""
        self._cache[
            self._get_key(
                InstructionData(
                    name=instruction.name,
                    json_content=instruction.json_content,
                )
            )
        ] = instruction

    def clear(self):
        self._cache.clear()


@dataclass
class DatasetIndexState:
    """State for dataset index information."""

    last_episode_idx: int = -1
    """The index of the last episode in the dataset."""
    last_robot_idx: int = -1
    """The index of the last robot in the dataset."""
    last_task_idx: int = -1
    """The index of the last task in the dataset."""
    last_instruction_idx: int = -1
    """The index of the last instruction in the dataset."""
    last_frame_idx: int = -1
    """The index of the last frame in the dataset."""
    last_episode_frame_idx: int = -1
    """The index of the last frame in the last episode in the dataset."""


def _patch_dataset_state_file(dataset_path: str):
    """Patch the dataset state file to include the robo_orchard_state."""
    from datasets import config as hg_datasets_config

    fs: fsspec.AbstractFileSystem = fsspec.core.url_to_fs(dataset_path)[0]

    state_file_path = os.path.join(
        dataset_path, hg_datasets_config.DATASET_STATE_JSON_FILENAME
    )
    if not fs.exists(state_file_path):
        raise FileNotFoundError(
            f"The dataset state file '{state_file_path}' does not exist."
        )
    # read existing state file
    with fs.open(state_file_path, "r", encoding="utf-8") as state_file:
        state = json.load(state_file)

    # append robo_orchard_state
    state["robo_orchard_state"] = {}
    state["robo_orchard_state"]["dataset_format_version"] = (
        dataset_format_version
    )

    with fs.open(state_file_path, "w", encoding="utf-8") as state_file:
        json.dump(state, state_file, indent=2, sort_keys=True)


def _rename_arrow_files(
    files: list[str], fs: fsspec.AbstractFileSystem, source_prefix: str
) -> list[str]:
    """Rename arrow files to match the expected format.

    In some cases, the arrow files generated by datasets may exceed the
    100000 shards limit, which will cause the file names to be inconsistent
    with the expected format. This function renames the arrow files to
    match the expected format.
    """
    if len(files) == 0:
        return []
    num_shards = len(files)
    digits = max(5, len(str(num_shards)))
    ret = []
    for f in files:
        dir_name = os.path.dirname(f)
        base_name = os.path.basename(f)
        if len(base_name) == len(source_prefix) + 6:
            assert base_name == f"{source_prefix}.arrow"
            shard_idx = 0
        else:
            split_info = base_name[len(source_prefix) + 1 : -6]  # noqa: E203
            split_info = split_info.split("-of-")
            assert len(split_info) == 2
            shard_idx = int(split_info[0])
            num_shards_in_file = int(split_info[1])
            assert num_shards_in_file == num_shards
        new_base_name = (
            f"data-{shard_idx:0{digits}d}-of-{num_shards:0{digits}d}.arrow"  # noqa: E501
        )
        new_f = os.path.join(dir_name, new_base_name)
        fs.move(f, new_f)
        ret.append(new_f)
    return ret
