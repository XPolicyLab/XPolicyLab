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

from typing import Generator, Generic, Iterable, TypeVar

import datasets as hg_datasets

from robo_orchard_lab.dataset.datatypes.hg_features import RODictDataFeature
from robo_orchard_lab.dataset.robot.columns import PreservedColumnsKeys
from robo_orchard_lab.dataset.robot.dataset import RODataset
from robo_orchard_lab.dataset.robot.db_orm import (
    Instruction,
    Robot,
    Task,
)
from robo_orchard_lab.dataset.robot.packaging import (
    DataFrame,
    DatasetPackaging,
    EpisodeData,
    EpisodeMeta,
    EpisodePackaging,
    InstructionData,
    RobotData,
    TaskData,
)

__all__ = ["repack_dataset"]


class DefaultRePackingEpisodeHelper(EpisodePackaging):
    """Helper class to re-package an episode from an existing dataset.

    User can inherit this class to customize the episode meta or frame
    generation by optionally overriding:
    - `generate_episode_meta` to customize the episode meta.
    - `generate_frames` to customize the frame data.

    Args:
        dataset (RODataset): The dataset to re-package from.
        episode_frames (list[int]): The list of frame indices that belong to
            the episode to be re-packaged. Note that all frames must belong
            to the same episode.
    """

    def __init__(self, dataset: RODataset, episode_frames: list[int]):
        if len(episode_frames) == 0:
            raise ValueError("episode_frames cannot be empty.")
        self.dataset = dataset
        self.frame_index_list = episode_frames
        self.frame_index_list.sort()

        index_dict = self.dataset.index_dataset[self.frame_index_list]
        episode_list = index_dict["episode_index"]
        if not all(eid == episode_list[0] for eid in episode_list):
            raise ValueError("All frames must belong to the same episode.")
        self._current_episode_index = episode_list[0]

    def generate_episode_meta(self) -> EpisodeMeta:
        frame_start = self.frame_index_list[0]
        row = self.dataset.index_dataset[frame_start]
        orm_robot: Robot = self.dataset.get_meta(Robot, row["robot_index"])
        assert orm_robot is not None
        orm_task: Task = self.dataset.get_meta(Task, row["task_index"])
        assert orm_task is not None
        robot: RobotData = RobotData.from_orm(orm_robot)
        task = TaskData(
            name=orm_task.name,
            description=orm_task.description,
        )
        return EpisodeMeta(episode=EpisodeData(), robot=robot, task=task)

    def generate_frames(self) -> Generator[DataFrame, None, None]:
        """Generate frame data for the episode."""

        preserved_columns = set(PreservedColumnsKeys)

        keep_columns = [
            key
            for key in self.dataset.features
            if key not in preserved_columns
        ]

        # cache all instructions for faster access
        index_rows = self.dataset.index_dataset[self.frame_index_list]
        all_instruction = self.dataset.get_meta(
            Instruction, index_rows["instruction_index"]
        )

        # split self.frame_index_list into batches
        batch_size = 128
        for batch_start in range(0, len(self.frame_index_list), batch_size):
            batch_end = min(
                batch_start + batch_size, len(self.frame_index_list)
            )
            batch_indices = self.frame_index_list[batch_start:batch_end]
            batch_row = self.dataset.frame_dataset.__getitems__(batch_indices)
            for i, idx in enumerate(batch_indices):
                row = batch_row[i]
                instruction_index = row["instruction_index"]
                orm_instruction = all_instruction[batch_start + i]
                if orm_instruction is None:
                    raise RuntimeError(
                        f"Instruction not found for frame index {idx} "
                        f"with instruction_index {instruction_index}"
                    )
                features = {key: row[key] for key in keep_columns}
                frame = DataFrame(
                    features=features,
                    instruction=InstructionData(
                        name=orm_instruction.name,
                        json_content=orm_instruction.json_content,
                    ),
                    timestamp_ns_min=row["timestamp_min"],
                    timestamp_ns_max=row["timestamp_max"],
                )
                yield frame


RePackingEpisodeType = TypeVar(
    "RePackingEpisodeType", bound=DefaultRePackingEpisodeHelper
)


class RePackingDatasetHelper(Generic[RePackingEpisodeType]):
    """Iterator to generate episodes for re-packaging.

    Args:
        dataset (RODataset): The dataset to re-package from.
        frame_indices (Iterable[int]): An iterable of frame indices to include
            in the re-packaged dataset. Frames from the same episode should be
            grouped together!
    """

    def __init__(
        self,
        dataset: RODataset,
        frame_indices: Iterable[int],
        packing_impl: type[
            RePackingEpisodeType
        ] = DefaultRePackingEpisodeHelper,
    ):
        self.dataset = dataset
        self.frame_indices = frame_indices
        self.packing_impl = packing_impl

    def __iter__(self):
        return self._next_helper()

    def _next_helper(
        self,
    ) -> Generator[RePackingEpisodeType, None, None]:
        current_episode_index = None
        current_episode_frames = []

        def process_batch(batch_frame_indices: list[int]):
            nonlocal current_episode_index, current_episode_frames
            batch_rows = self.dataset.index_dataset.__getitems__(
                batch_frame_indices
            )
            for i, frame_index in enumerate(batch_frame_indices):
                row = batch_rows[i]
                episode_index = row["episode_index"]
                # only for first frame, assign current_episode_index
                if current_episode_index is None:
                    current_episode_index = episode_index

                if episode_index != current_episode_index:
                    # yield the previous episode
                    yield self.packing_impl(
                        self.dataset, current_episode_frames
                    )
                    current_episode_index = episode_index
                    current_episode_frames = [frame_index]
                else:
                    current_episode_frames.append(frame_index)

        batch_size = 1024
        batch_frame_indices = []
        for frame_index in self.frame_indices:
            batch_frame_indices.append(frame_index)
            if len(batch_frame_indices) >= batch_size:
                yield from process_batch(batch_frame_indices)
                batch_frame_indices.clear()
        if batch_frame_indices:
            yield from process_batch(batch_frame_indices)
        if current_episode_frames:
            yield self.packing_impl(self.dataset, current_episode_frames)


def repack_dataset(
    source_dataset: RODataset,
    target_path: str,
    frame_indices: Iterable[int] | None = None,
    columns: str | list[str] | None = None,
    writer_batch_size: int = 1024,
    max_shard_size: str | int = "8GB",
    force_overwrite: bool = False,
    packing_impl: type[RePackingEpisodeType] = DefaultRePackingEpisodeHelper,
):
    """Re-package a RoboOrchard dataset with selected frames for each episode.

    Args:
        source_dataset (RODataset): The source dataset to re-package from.
        target_path (str): The path to save the re-packaged dataset.
        frame_indices (Iterable[int] | None): An iterable of frame indices
            to include in the re-packaged dataset. Frames from the same
            episode should be grouped together! If None, all frames will be
            included. Default is None.
        columns (str | list[str] | None): The columns to include in the
            re-packaged dataset. If None, all columns will be included. If
            a string is provided, it will be treated as a single column name.
            Default is None.
        writer_batch_size (int): The batch size for writing the arrow file.
            This may affect the performance of packaging or reading the
            dataset later. Default is 1024.
        max_shard_size (str | int | None): The maximum size of each shard.
            If None, no sharding will be applied. This can be a string
            like '10GB' or an integer representing the size in bytes.
            Default is '8GB'.
        force_overwrite (bool): Whether to overwrite the target path if it
            already exists. Default is False.
        packing_impl (type[RePackingEpisodeType]): The implementation class
            for packaging each episode. Default is
            `DefaultRePackingEpisodeHelper`.

    """
    if columns is not None:
        dataset = source_dataset.select_columns(columns, include_index=True)
    else:
        dataset = source_dataset

    features = dataset.features

    preserved_columns = set(PreservedColumnsKeys)

    features = {
        key: features[key] for key in features if key not in preserved_columns
    }
    features = hg_datasets.Features(features)
    # check if any feature is adapted for loading. If so, we need to
    # reset them to default for packaging to match current code version.
    for _, feature in features.items():
        if isinstance(feature, RODictDataFeature):
            try:
                feature.reset()
            except NotImplementedError:
                pass

    if frame_indices is None:
        frame_indices = range(len(dataset.index_dataset))

    packing = DatasetPackaging(features=features)
    packing.packaging(
        episodes=RePackingDatasetHelper(
            dataset, frame_indices, packing_impl=packing_impl
        ),
        dataset_path=target_path,
        writer_batch_size=writer_batch_size,
        max_shard_size=max_shard_size,
        force_overwrite=force_overwrite,
    )
