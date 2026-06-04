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
from typing import Any, Callable, Generator

import datasets as hg_datasets
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from robo_orchard_lab.dataset.robot.packaging import (
    DataFrame,
    EpisodeData,
    EpisodeMeta,
    EpisodePackaging,
    InstructionData,
    RobotData,
    TaskData,
)
from robo_orchard_lab.utils import as_sequence

__all__ = [
    "LerobotDatasetEpisodePackaging",
    "get_hg_features",
    "DefaultTransform",
]


logger = logging.getLogger(__name__)


RESERVED_LEROBOT_KEYS = {
    "index",
    "episode_index",
    "frame_index",
    "timestamp",
    "task",
    "task_index",
}


class DefaultTransform:
    """A default transformation callable for processing frame data.

    This transform prepares raw frame data (often containing torch.Tensors)
    for storage or use in libraries like Hugging Face `datasets`.

    It performs two main operations:

    1.  Converts all torch.Tensors in the input dictionary to NumPy arrays.

    2.  Identifies images based on `camera_keys`, validates their shape,
        un-normalizes them (from [0, 1] float to [0, 255] uint8), and
        transposes them from (C, H, W) format to (H, W, C) format.
    """

    def __init__(self, camera_keys: list[str]):
        """Initializes the DefaultTransform.

        Args:
            camera_keys (list[str]): A list of dictionary keys that correspond
                to camera images (e.g., 'image_primary', 'image_wrist').
        """
        self.camera_keys = camera_keys

    def __call__(self, frame_data: dict[str, Any]) -> dict[str, Any]:
        """Applies the transformation to a single data frame.

        Args:
            frame_data (dict[str, Any]): A dictionary containing data for a
                single frame. Expected to contain torch.Tensors, with
                images in (C, H, W) format and normalized to [0, 1].

        Returns:
            dict[str, Any]: The transformed data dictionary, where all tensors
                are converted to NumPy arrays and images are formatted as
                (H, W, C) uint8 arrays (range [0, 255]).

        Raises:
            ValueError: If a value identified as an image (via `camera_keys`)
                does not have 3 dimensions (C, H, W) after being converted
                from a tensor.
        """
        new_frame_data = {}
        for key, value in frame_data.items():
            if isinstance(value, torch.Tensor):
                value = value.numpy()

            # Check if it's an image tensor
            if key in self.camera_keys:
                if value.ndim != 3:
                    raise ValueError(
                        f"Image should be with shape (C, H, W), "
                        f"but get {value.shape}"
                    )
                value = (value * 255).astype(np.uint8)
                # Convert (C, H, W) tensor to (H, W, C) for
                # hg_datasets.features.Image()
                new_frame_data[key] = value.transpose(1, 2, 0)
            else:
                # Pass other data (like action tensors) through
                new_frame_data[key] = value
        return new_frame_data


class LerobotDatasetEpisodePackaging(EpisodePackaging):
    """Implements `EpisodePackaging` for ingesting a `LeRobotDataset`.

    This class acts as an adapter, allowing the `DatasetPackaging` engine
    to read data from a source `LeRobotDataset` episode and convert it
    into the RoboOrchard format on the fly. It handles the mapping of
    metadata and the streaming generation of frame data.

    A `transform` function can be provided to handle complex
    data conversion logic, such as merging multiple LeRobot columns
    (e.g., 'joint_pos', 'joint_vel') into a single RoboOrchard
    data object (e.g., `BatchJointsState`).
    """

    def __init__(
        self,
        dataset: LeRobotDataset,
        episode_meta: dict,
        max_frames: int = 0,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        """Initializes the packager for a single LeRobot episode.

        Args:
            dataset: The loaded `LeRobotDataset` instance to read from.
            episode_meta: The metadata dictionary for the single episode
                to be packaged. This is typically one row (or dict) from
                `dataset.meta.episodes`.
            max_frames: If greater than 0, limits the number of frames
                generated for this episode. Defaults to 0 (no limit).
            transform: A callable that accepts the raw feature dictionary
                (from LeRobot, after stripping reserved keys) and returns
                a processed feature dictionary (for RoboOrchard). This is
                used to merge columns, convert data structures
                (e.g., Tensor to `BatchJointsState`), or perform other
                complex adaptations. If None, a default
                transformation (tensor to numpy, image (C,H,W) to (H,W,C))
                is applied.
        """
        self.dataset = dataset
        self.episode_meta = episode_meta
        self.max_frames = max_frames
        if transform is None:
            transform = DefaultTransform(
                camera_keys=self.dataset.meta.camera_keys
            )
        self.transform = transform

    def generate_episode_meta(self) -> EpisodeMeta:
        """Generates the static `EpisodeMeta` for this episode.

        Extracts task and robot information from the LeRobot metadata.
        Note: LeRobot format does not store URDFs, so a placeholder
        is used for `urdf_content`.

        Returns:
            EpisodeMeta: An object containing the static metadata (robot,
                task, episode info) for this episode.
        """
        # 1. Create TaskData
        # We use the task name as both name and description
        tasks = self.episode_meta.get("tasks", [])
        if tasks:
            task_data = TaskData(name=tasks[0], description=tasks[0])
        else:
            task_data = None

        # 2. Create RobotData
        # LeRobot only stores robot_type (name).
        # We must provide a placeholder for urdf_content.
        robot_data = RobotData(
            name=self.dataset.meta.robot_type or "UNKNOWN",
            content=None,
            content_format=None,
        )

        # 3. Create EpisodeData
        episode_data = EpisodeData()

        return EpisodeMeta(
            episode=episode_data, robot=robot_data, task=task_data
        )

    def generate_frames(self) -> Generator[DataFrame, None, None]:
        """Yields `DataFrame` objects for each frame in the LeRobot episode.

        This generator iterates through the global frame indices specified
        by the `episode_meta`, loads each frame from the `LeRobotDataset`
        (which handles video decoding), converts the data into the
        RoboOrchard format, and yields it.

        Yields:
            DataFrame: A `DataFrame` object containing the converted
                features and metadata for a single timestep.

        Raises:
            ValueError: If an image frame from LeRobot has an unexpected
                shape (not (C, H, W)).
        """
        start_idx = self.episode_meta["dataset_from_index"]
        end_idx = self.episode_meta["dataset_to_index"]

        if start_idx > end_idx:
            logger.warning(
                f"Failed to load episode since start_idx({start_idx}) > end_idx({end_idx})"  # noqa: E501
            )

        for idx in range(start_idx, end_idx):
            if self.max_frames > 0 and (idx - start_idx) >= self.max_frames:
                break

            # 1. Get the frame data from LeRobotDataset
            # This performs video decoding and returns a dict of tensors
            try:
                frame_data = self.dataset[idx]
            except Exception as e:
                logger.warning(
                    f"Failed to load frame {idx}, skipping. Error: {e}"
                )
                continue

            # 2. Create InstructionData from the task string
            # lerobot v3 uses "task", v2 used "task.instructions"
            task_str = frame_data.get(
                "task", frame_data.get("task.instructions", "")
            )
            instruction = InstructionData(
                name=task_str,
                json_content={"instruction": task_str},
            )

            # 3. Convert timestamp (float seconds) to nanoseconds (int)
            timestamp = as_sequence(frame_data["timestamp"].tolist())
            if not timestamp:  # missing timestamp...
                time_delta = 1.0 / self.dataset.meta.fps
                ts_ns = int((idx - start_idx) * time_delta * 1_000_000_000)
            else:
                ts_ns = int(timestamp[0] * 1_000_000_000)

            # 4. Stripping reserved keys
            new_frame_data = {}
            for key, value in frame_data.items():
                if key not in RESERVED_LEROBOT_KEYS:
                    new_frame_data[key] = value

            # 5. Apply the custom transform if provided
            processed_frame_data = self.transform(new_frame_data)

            # 6. Yield the final DataFrame object
            yield DataFrame(
                features=processed_frame_data,
                instruction=instruction,
                timestamp_ns_min=ts_ns,
                timestamp_ns_max=ts_ns,
            )


def get_hg_features(dataset: LeRobotDataset) -> hg_datasets.Features:
    """Adapts LeRobot features for RoboOrchard packaging.

    This function takes the `hf_features` from a `LeRobotDataset`,
    removes reserved keys (like 'index', 'timestamp') that will be
    regenerated by `DatasetPackaging`, and converts LeRobot's
    `VideoFrame` features into standard `hg_datasets.features.Image`
    features.

    This is necessary because the `DatasetPackaging` engine expects
    to write standard image arrays, not LeRobot's custom video objects.

    Args:
        dataset: The loaded `LeRobotDataset` instance.

    Returns:
        hg_datasets.Features: A new `Features` object compatible with
            the `DatasetPackaging` engine and standard
            Hugging Face `datasets` image loading.
    """
    features = dataset.hf_features

    for key in RESERVED_LEROBOT_KEYS:
        if key in features:
            features.pop(key)

    # Convert LeRobot's VideoFrame features to standard Image features
    # Our packager (generate_frames) will provide (H, W, C) numpy arrays
    for key in dataset.meta.camera_keys:
        features[key] = hg_datasets.features.Image()

    return features
