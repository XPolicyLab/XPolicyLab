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

import copy
import io
from dataclasses import dataclass
from typing import Generator

import datasets as hg_datasets
import h5py
import numpy as np
import torch
from robo_orchard_core.envs.task import TaskInfo
from robo_orchard_core.utils.logging import LoggerManager

from robo_orchard_lab.dataset.datatypes import (
    BatchCameraData,
    BatchCameraDataEncoded,
    BatchCameraDataEncodedFeature,
    BatchFrameTransformFeature,
    BatchFrameTransformGraph,
    BatchFrameTransformGraphFeature,
    BatchImageData,
    BatchJointsState,
    BatchJointsStateFeature,
    ImageMode,
    TypedTensorFeature,
)
from robo_orchard_lab.dataset.robot.packaging import (
    DataFrame,
    DatasetPackaging,
    EpisodeData,
    EpisodeMeta,
    EpisodePackaging,
    InstructionData,
    RobotData,
    RobotDescriptionFormat,
    TaskData,
)
from robo_orchard_lab.envs.libero import LiberoEnvCfg, get_libero_task
from robo_orchard_lab.envs.libero.env import (
    LiberoObsType,
    LiberoSuiteName,
    libero_suite_names,
)

logger = LoggerManager().get_child(__name__)


@dataclass
class LiberoEpisodeCache:
    success: bool
    cached_frames: list[dict]
    episode_meta: EpisodeMeta | None = None
    task_info: TaskInfo | None = None


@dataclass
class LiberoSuiteStatistics:
    suite_name: LiberoSuiteName
    num_tasks: int
    num_episodes_per_task: list[int]


def get_libero_suite_statistics() -> list[LiberoSuiteStatistics]:
    """Get statistics about Libero suites.

    Returns:
        list[LiberoSuiteStatistics]: The statistics of the suite.
    """
    ret = []
    for suite_name in libero_suite_names:
        suite = get_libero_task(suite_name, task_id=0).suite
        num_tasks = suite.get_num_tasks()
        num_episodes_per_task = []
        for task_id in range(0, num_tasks):
            task_info = get_libero_task(suite_name, task_id)
            with h5py.File(task_info.hdf5_path, "r") as f:
                orig_data: h5py.Group = f["data"]  # type: ignore
                num_episodes_per_task.append(len(orig_data.keys()))
        ret.append(
            LiberoSuiteStatistics(
                suite_name=suite_name,
                num_tasks=num_tasks,
                num_episodes_per_task=num_episodes_per_task,
            )
        )
    return ret


LiberoDatasetFeatures = hg_datasets.Features(
    {
        "agentview_image": BatchCameraDataEncodedFeature(dtype="float64"),
        "agentview_depth": BatchCameraDataEncodedFeature(dtype="float64"),
        "robot0_eye_in_hand_image": BatchCameraDataEncodedFeature(
            dtype="float64"
        ),
        "robot0_eye_in_hand_depth": BatchCameraDataEncodedFeature(
            dtype="float64"
        ),
        "joints": BatchJointsStateFeature(dtype="float64"),
        "env_state": TypedTensorFeature(dtype="float64"),
        "tf_world": BatchFrameTransformGraphFeature(dtype="float64"),
        "action": TypedTensorFeature(dtype="float64"),
        "action_goal_eef": BatchFrameTransformFeature(dtype="float64"),
        "reward": hg_datasets.Value("float64"),
        "terminated": hg_datasets.Value("bool"),
    }
)


class LiberoEpisodePacking(EpisodePackaging):
    """A class to package a Libero episode into dataset frames."""

    def __init__(
        self, suite_name: LiberoSuiteName, task_id: int, episode_idx: int
    ):
        self._task = get_libero_task(suite_name, task_id)

        self._suite_name = suite_name
        self._task_id = task_id
        self._episode_idx = episode_idx
        self._episode_cache: LiberoEpisodeCache | None = None

    def _encode_camera_data(
        self, cam_data: BatchCameraData
    ) -> BatchCameraDataEncoded:
        """Encode the camera data (depth/image) to compressed format.

        If pix_fmt is RGB or BGR, just encode as JPEG.
        For depth data, the source is usually in float32 format, we convert
        to uint16 and encode as PNG. The depth value is scaled by 1000 to
        preserve millimeter precision and clipped to [0, 65535] range, which
        allows representing depth up to 65.535 meters.

        """

        def encode_impl(format: str, cam_data: BatchImageData) -> list[bytes]:
            if cam_data.pix_fmt == ImageMode.F:
                # convert to uint16 as new cam_data
                data = (
                    (cam_data.sensor_data * 1000)
                    .clip(0, 65535)
                    .to(dtype=torch.uint16)
                )
                cam_data = BatchImageData(
                    sensor_data=data,
                    pix_fmt=ImageMode.I16,
                    timestamps=cam_data.timestamps,
                )
            format_kwargs = {}
            if format == "jpeg":
                format_kwargs["quality"] = 95

            pil_images = cam_data.as_pil_images()
            compressed_data = []
            for img in pil_images:
                buf = io.BytesIO()
                img.save(buf, format=format.upper(), **format_kwargs)
                compressed_data.append(buf.getvalue())
            return compressed_data

        if cam_data.pix_fmt in (ImageMode.RGB, ImageMode.BGR):
            encoded = cam_data.encode(format="jpeg", encoder=encode_impl)
        else:
            encoded = cam_data.encode(format="png", encoder=encode_impl)
        return encoded

    def _convert_ob2frame(self, obs: LiberoObsType, target_dict: dict):
        for key, value in obs.items():
            if isinstance(value, BatchJointsState):
                target_dict[key] = value
            elif isinstance(value, BatchCameraData):
                target_dict[key] = self._encode_camera_data(value)

        target_dict["tf_world"] = BatchFrameTransformGraph(
            tf_list=obs["tf_world"].values(),
        )

    @property
    def episode_cache(self) -> LiberoEpisodeCache:
        if self._episode_cache is not None:
            return self._episode_cache

        env_cfg = LiberoEnvCfg(
            suite_name=self._suite_name,  # type: ignore
            task_id=self._task_id,
            camera_depths=True,
            format_datatypes=True,
        )

        env = env_cfg()
        cache = LiberoEpisodeCache(
            success=False,
            cached_frames=[],
        )
        with h5py.File(self._task.hdf5_path, "r") as f:
            orig_data: h5py.Group = f["data"]  # type: ignore
            if self._episode_idx >= len(orig_data.keys()):
                raise IndexError(
                    f"Episode index {self._episode_idx} out of "
                    f"range for task with {len(orig_data.keys())} episodes."
                )
            i = self._episode_idx
            # Get demo data for this episode
            demo_data = orig_data[f"demo_{i}"]
            orig_actions: np.ndarray = demo_data["actions"][()]  # type: ignore
            orig_states: np.ndarray = demo_data["states"][()]  # type: ignore
            obs, _ = env.reset(seed=0, init_state=orig_states[0])
            frame = {}
            frame["env_state"] = orig_states[0]
            self._convert_ob2frame(obs, frame)
            for _, action in enumerate(orig_actions):
                frame["action"] = action
                step_ret = env.step(action)
                last_action = env.get_last_action()
                assert last_action is not None

                joints: BatchJointsState = frame["joints"]
                last_action.goal_eef.timestamps = copy.copy(joints.timestamps)
                last_action.goal_eef.child_frame_id = "goal_eef"

                frame["action_goal_eef"] = last_action.goal_eef
                frame["reward"] = step_ret.rewards
                frame["terminated"] = step_ret.terminated
                cache.cached_frames.append(frame)
                frame = {}
                obs = step_ret.observations
                self._convert_ob2frame(obs, frame)
                frame["env_state"] = env.get_sim_state()

            if step_ret.rewards > 0:
                cache.success = True

            episode_data = EpisodeData(
                truncated=False,
                success=cache.success,
            )
            robot = RobotData(
                name="libero_panda",
                content=env.get_robot_xml(),
                content_format=RobotDescriptionFormat.MJCF,
            )
            task = TaskData(
                name=f"{self._suite_name}_task_{self._task_id}",
                description=env.task_info.description,
            )
            cache.episode_meta = EpisodeMeta(
                episode=episode_data, robot=robot, task=task
            )
            cache.task_info = env.task_info
            # need to close the env to release resources
            env.close()
            self._episode_cache = cache

        return self._episode_cache

    def generate_episode_meta(self) -> EpisodeMeta | None:
        """Generate episode meta information.

        For failed episodes, this method still generates the episode meta
        information.
        """
        return self.episode_cache.episode_meta

    def generate_frames(self) -> Generator[DataFrame, None, None]:
        task_info = self.episode_cache.task_info
        assert task_info is not None
        instruction = InstructionData(
            name=f"{self._suite_name}_task_{self._task_id}",
            json_content=task_info.__dict__,
        )
        for frame_dict in self.episode_cache.cached_frames:
            ts = frame_dict["joints"].timestamps[0]
            ret = DataFrame(
                features=frame_dict,
                instruction=instruction,
                timestamp_ns_max=ts,
                timestamp_ns_min=ts,
            )
            yield ret
        # remove cache to save memory
        self._episode_cache = None


class LiberoEpisodePackingGenerator:
    def __init__(self, max_episode: int = -1):
        self._max_episode = max_episode

    def __iter__(self) -> Generator[LiberoEpisodePacking, None, None]:
        idx = 0
        for suite_stat in get_libero_suite_statistics():
            for task_id in range(suite_stat.num_tasks):
                num_episodes = suite_stat.num_episodes_per_task[task_id]
                for episode_idx in range(num_episodes):
                    if 0 <= self._max_episode <= idx:
                        return
                    yield LiberoEpisodePacking(
                        suite_name=suite_stat.suite_name,
                        task_id=task_id,
                        episode_idx=episode_idx,
                    )
                    idx += 1


def make_libero_dataset(
    dataset_path: str,
    max_shard_size: str | int = "8GB",
    split: hg_datasets.Split | None = None,
    force_overwrite: bool = False,
    max_episode=-1,
    writer_batch_size=4096,
) -> None:
    """Generate a Libero dataset at the specified path.

    Args:
        dataset_path (str): The path to save the dataset.
        max_shard_size (str | int, optional): The maximum shard size.
            Defaults to "8GB".
        split (hg_datasets.Split | None, optional): The dataset split to use.
            If None, uses the default split. Defaults to None.
        force_overwrite (bool, optional): Whether to overwrite existing
            dataset. Defaults to False.
        max_episode (int, optional): The maximum number of episodes to include.
            Defaults to -1 (all episodes).
        writer_batch_size (int, optional): The batch size for the writer.
            Defaults to 4096.
    """
    episode_generator = LiberoEpisodePackingGenerator(max_episode=max_episode)
    packing = DatasetPackaging(features=LiberoDatasetFeatures)
    packing.packaging(
        episodes=episode_generator,
        dataset_path=dataset_path,
        max_shard_size=max_shard_size,
        split=split,
        force_overwrite=force_overwrite,
        writer_batch_size=writer_batch_size,
    )
