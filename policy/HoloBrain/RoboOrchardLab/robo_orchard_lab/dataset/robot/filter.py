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
from typing import Callable

from datasets import Dataset as HFDataset

from robo_orchard_lab.dataset.robot.dataset import RODataset
from robo_orchard_lab.dataset.robot.dataset_ex import (
    DatasetWithIndices,
    IndiceTable,
)
from robo_orchard_lab.dataset.robot.db_orm import (
    Episode,
    Instruction,
    Robot,
    Task,
)


class RODatasetFilter:
    def __init__(
        self,
        episode_filter: Callable[[Episode | None], bool] | None = None,
        task_filter: Callable[[Task | None], bool] | None = None,
        robot_filter: Callable[[Robot | None], bool] | None = None,
        instruction_filter: Callable[[Instruction], bool] | None = None,
        row_filter: Callable[[dict], bool] | None = None,
    ):
        self._episode_filter = episode_filter
        self._task_filter = task_filter
        self._robot_filter = robot_filter

        self._instruction_filter = instruction_filter
        self._row_filter = row_filter

    def get_filtered_episode_indices(self, dataset: RODataset) -> set[int]:
        """Get the episode indices that satisfy the filters.

        This method combines the episode_filter, task_filter and robot_filter
        to filter episodes.
        """

        if (
            self._episode_filter is None
            and self._task_filter is None
            and self._robot_filter is None
        ):
            # get the total episode indices without filtering
            return set(range(dataset.episode_num))

        episode_indices = set()

        # create new episode_filter to combine episode_filter, task_filter
        # and robot_filter
        def combined_episode_filter(episode: Episode) -> bool:
            episode_filter = (
                self._episode_filter
                if self._episode_filter is not None
                else lambda e: True
            )
            if self._task_filter is not None:
                task = episode.task
                if not self._task_filter(task):
                    return False
            if self._robot_filter is not None:
                robot = episode.robot
                if not self._robot_filter(robot):
                    return False
            return episode_filter(episode)

        for episode in dataset.iterate_meta(Episode, transient=False):
            if combined_episode_filter(episode):
                episode_indices.add(episode.index)
        return episode_indices

    def get_filtered_instruction_indices(
        self, dataset: RODataset
    ) -> set[int] | None:
        """Get the instruction indices that satisfy the instruction filter.

        Return None if instruction filter is not set, which means all
        instructions are included.
        """
        if self._instruction_filter is None:
            return None

        instruction_indices = set()
        for instruction in dataset.iterate_meta(Instruction, transient=False):
            if self._instruction_filter(instruction):
                instruction_indices.add(instruction.index)
        return instruction_indices

    def get_row_indices(
        self,
        dataset: RODataset,
    ) -> list[int]:
        ret = []
        index_dataset = dataset.index_dataset
        episode_indices = self.get_filtered_episode_indices(dataset)

        instruction_indices = self.get_filtered_instruction_indices(dataset)

        # create new row_filter to combine row filter with instruction filter
        def combined_row_filter(
            idx: int, dataset: RODataset, index_dataset: HFDataset
        ) -> bool:
            # get instruction index of the row
            instruction_idx = index_dataset[idx]["instruction_index"]
            # early return False if instruction index is not in the filtered
            # instruction indices if instruction filter is set
            if (
                instruction_indices is not None
                and instruction_idx not in instruction_indices
            ):
                return False

            return (
                self._row_filter(dataset[idx])
                if self._row_filter is not None
                else True
            )

        for episode_index in episode_indices:
            episode = dataset.get_meta(Episode, episode_index)
            assert episode is not None, (
                f"Episode with index {episode_index} not found"
            )
            if self._row_filter is None and self._instruction_filter is None:
                # if no row filter and instruction filter, include all rows
                # in the episode
                ret.extend(
                    range(
                        episode.dataset_begin_index,
                        episode.dataset_begin_index + episode.frame_num,
                    )
                )
            else:
                # otherwise, we need to check each row in the episode
                for local_idx in range(
                    episode.dataset_begin_index,
                    episode.dataset_begin_index + episode.frame_num,
                ):
                    if combined_row_filter(local_idx, dataset, index_dataset):
                        ret.append(local_idx)
        return ret

    def apply(self, dataset: RODataset) -> DatasetWithIndices[RODataset]:
        """Apply the filter to the dataset.

        Note that this method does not modify the original dataset, but creates
        a new dataset with the filtered data. The new dataset shares the same
        underlying data as the original dataset, so it is memory efficient.
        """
        row_indices = self.get_row_indices(dataset)
        return DatasetWithIndices[RODataset](
            dataset=dataset,
            indices=IndiceTable(row_indices),
        )
