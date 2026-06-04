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

import logging
import os
from typing import Callable, List, Optional, Union

import numpy as np
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from torch.utils.data import Dataset

from robo_orchard_lab.dataset.lmdb.lmdb_wrapper import Lmdb
from robo_orchard_lab.distributed.utils import get_dist_info
from robo_orchard_lab.utils.build import build
from robo_orchard_lab.utils.misc import as_sequence

logger = logging.getLogger(__name__)


class BaseIndexData(BaseModel):
    """Base data structure for indexing simulation or task-related information."""  # noqa: E501

    uuid: str
    num_steps: int
    task_name: str = Field(
        validation_alias=AliasChoices("task_name", "task"), default=None
    )
    user: Optional[str] = None
    embodiment: Optional[str] = None
    date: Optional[str] = None
    simulation: bool = False
    error: bool = False

    model_config = ConfigDict(extra="allow")


class BaseLmdbManipulationDataset(Dataset):
    """A dataset class for manipulation tasks stored in LMDB format.

    The dataset is structured into four fundamental components:
    `index`, `meta`, `depth`, and `image`.

    .. note::

        **index** and **meta** are organized by episode as the basic unit.

        **depth** and **image** are stored by frame as the basic unit.

    An example:

    .. code-block:: text

        - index:
            - `episode_id`: `BaseIndexData`.
        - meta:
            - `{uuid}/meta_data`: General metadata about the task.
            - `{uuid}/camera_names`: List of camera names used in the task.
            - `{uuid}/observation/joint_positions`: [num_steps * num_joint]
        - image:
            - `{uuid}/{cam_name}/{step_idx}`: image_buffer
        - depth:
            - `{uuid}/{cam_name}/{step_idx}`: depth_buffer

    Args:
        paths (Union[str, List[str]]): Path(s) to the LMDB database(s). Can be
            a single path or a list of paths.
        transforms (Optional[List[Callable]]): A function/transform to apply to
            the data samples. Can also be a sequence of transforms.
            Default: None.
        interval (Optional[int]): Interval between steps to sample.
            Default: None
        load_image (bool): Whether to load image data. Default: True.
        load_depth (bool): Whether to load depth data. Default: True.
        task_names (Optional[Union[str, List[str]]]): List of task names to
            filter by. Default: None.
        num_episode (Optional[int]): Maximum number of episodes to load.
            Default: None.
        lazy_init (bool): If True, initialization is deferred until first
            access. Default: False.
        encoding_mode (str): Encoding mode of keys from LMDB.
            Default: "utf-8".
    """

    def __init__(
        self,
        paths: Union[str, List[str]],
        transforms: Optional[List[Callable]] = None,
        interval: Optional[int] = None,
        load_image: bool = True,
        load_depth: bool = True,
        task_names: Optional[Union[str, List[str]]] = None,
        num_episode: Optional[int] = None,
        lazy_init: bool = False,
        encoding_mode: str = "utf-8",
        dataset_name: str = "",
        reset_step: int = 10000,
        lmdb_kwargs: Optional[dict] = None,
        flag: Optional[int] = None,
    ):
        if not isinstance(paths, (list, tuple)):
            paths = [paths]
        self.paths = paths
        self.transforms = [build(x) for x in as_sequence(transforms)]
        self.interval = interval
        self.load_image = load_image
        self.load_depth = load_depth
        self.task_names = task_names
        self.num_episode_ = num_episode
        self.encoding_mode = encoding_mode
        self.dataset_name = dataset_name
        self.reset_step = reset_step
        self.lmdb_kwargs = lmdb_kwargs if lmdb_kwargs is not None else {}
        self.flag = flag
        self.initialized = False
        if not lazy_init:
            self._init_lmdb()

    def _check_valid(self, index_data):
        if index_data.error:
            return False
        if (self.task_names is not None) and (
            index_data.task_name not in self.task_names
        ):
            return False
        return True

    def _init_lmdb(self):
        self.meta_lmdbs = [None for path in self.paths]
        self.img_lmdbs = [None for path in self.paths]
        self.depth_lmdbs = [None for path in self.paths]
        self.idx_lmdbs = [
            Lmdb(
                uri=os.path.join(path, "index"),
                writable=False,
                encoding_mode=self.encoding_mode,
            )
            for path in self.paths
        ]

        lmdb_indices = []
        episode_indices = []
        num_steps = []
        task_statistics = {
            "num_episode": {},
            "num_steps": {},
        }
        current_num_episode = 0
        for i, idx_lmdb in enumerate(self.idx_lmdbs):
            for episode_idx in idx_lmdb.keys():
                if episode_idx == "__len__":
                    continue
                data = BaseIndexData.model_validate(idx_lmdb.get(episode_idx))

                if (self._check_valid(data)) and (
                    self.num_episode_ is None
                    or current_num_episode < self.num_episode_
                ):
                    lmdb_indices.append(i)
                    episode_indices.append(episode_idx)
                    num_steps.append(data.num_steps)
                    current_num_episode += 1

                    if data.task_name not in task_statistics["num_episode"]:
                        task_statistics["num_episode"][data.task_name] = 0
                        task_statistics["num_steps"][data.task_name] = 0
                    task_statistics["num_episode"][data.task_name] += 1
                    task_statistics["num_steps"][data.task_name] += (
                        data.num_steps
                    )

        self.lmdb_indices = lmdb_indices
        self.episode_indices = episode_indices
        self.num_steps = np.array(num_steps)
        self.cumsum_steps = np.cumsum(num_steps)
        self.num_episode = len(num_steps)
        self.task_statistics = task_statistics
        self.initialized = True
        dist_info = get_dist_info()
        if dist_info.rank == 0:
            logger.info(
                f"{self.dataset_name} dataset length: {self.__len__()}, "
                f"number of episode: {self.num_episode}, "
                f"task_statistics: {self.task_statistics}"
            )

    def __len__(self):
        if not self.initialized:
            self._init_lmdb()
        if len(self.cumsum_steps) == 0:
            return 0
        if self.interval is None:
            return self.cumsum_steps[-1]
        else:
            return self.cumsum_steps[-1] // self.interval

    def _get_indices(self, index):
        if not self.initialized:
            self._init_lmdb()

        if self.interval is not None:
            index *= self.interval
        episode_index = np.searchsorted(self.cumsum_steps, index, side="right")
        lmdb_index = self.lmdb_indices[episode_index]
        if episode_index == 0:
            step_index = index
        else:
            step_index = index - self.cumsum_steps[episode_index - 1]
        episode_index = self.episode_indices[episode_index]
        if self.meta_lmdbs[lmdb_index] is None:
            self.meta_lmdbs[lmdb_index] = Lmdb(
                uri=os.path.join(self.paths[lmdb_index], "meta"),
                writable=False,
                encoding_mode=self.encoding_mode,
                reset_step=-1,
                **self.lmdb_kwargs,
            )
            if self.load_image:
                self.img_lmdbs[lmdb_index] = Lmdb(
                    uri=os.path.join(self.paths[lmdb_index], "image"),
                    writable=False,
                    encoding_mode=self.encoding_mode,
                    reset_step=self.reset_step,
                    **self.lmdb_kwargs,
                )
            if self.load_depth:
                self.depth_lmdbs[lmdb_index] = Lmdb(
                    uri=os.path.join(self.paths[lmdb_index], "depth"),
                    writable=False,
                    encoding_mode=self.encoding_mode,
                    reset_step=self.reset_step,
                    **self.lmdb_kwargs,
                )
        return lmdb_index, episode_index, step_index

    def __getitem__(self, index):
        """Get data dict by index.

        Obtain the hierarchical indices (lmdb_index, episode_index, step_index)
        by first calling:
            lmdb_index, episode_index, step_index = self._get_indices(index)
        """
        raise NotImplementedError

    def get_episode_range(self, ep_idx: int) -> tuple[int, int]:
        end = int(self.cumsum_steps[ep_idx])
        start = int(self.cumsum_steps[ep_idx - 1]) if ep_idx > 0 else 0
        return start, end

    def visualize(self, episode_index, output_path="./vis_data"):
        raise NotImplementedError


class BaseLmdbManipulationDataPacker(object):
    def __init__(self, input_path, output_path, commit_step=500, **kwargs):
        self.input_path = input_path
        self.output_path = output_path
        self.commit_step = commit_step
        self.lmdb_kwargs = kwargs

    def _init_lmdbs(self):
        for f in ["index", "meta", "image", "depth"]:
            uri = os.path.join(self.output_path, f)
            if not os.path.exists(uri):
                os.makedirs(uri)
            setattr(
                self,
                f"{f}_pack_file",
                Lmdb(
                    uri=uri,
                    writable=True,
                    commit_step=self.commit_step,
                    **self.lmdb_kwargs,
                ),
            )

    def close(self):
        self.index_pack_file.close()
        self.meta_pack_file.close()
        self.image_pack_file.close()
        self.depth_pack_file.close()

    def write_index(self, index: Union[int, str], index_data: dict):
        self.index_pack_file.write(
            index, BaseIndexData.model_validate(index_data).model_dump()
        )

    def __call__(self):
        self._init_lmdbs()
        self._pack()

    def _pack(self):
        raise NotImplementedError
