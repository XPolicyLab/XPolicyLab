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

from typing import Callable

import cv2
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms import Compose

from robo_orchard_lab.dataset.horizon_manipulation.row_sampler import (
    EpisodeChunkSamplerConfig,
)
from robo_orchard_lab.dataset.robot.dataset import (
    ConcatRODataset,
    ROMultiRowDataset,
)
from robo_orchard_lab.dataset.robot.db_orm import Episode

__all__ = ["AgilexRODataset"]


class ArrowDataParse:
    """The dataset class for manipulation tasks in RoboOrchard.

    Args:
        dataset_path (str): Path to the dataset.
        cam_names (list[str]): List of camera names to load data from.
        load_image (bool): Whether to load image data. Default is True.
        load_depth (bool): Whether to load depth data. Default is True.
        load_extrinsic (bool): Whether to load camera extrinsic data.
            Default is True.
        load_ee_state (bool): Whether to load end-effector state data.
            Default is False.
        transforms (list[dict] or dict, optional): List of transformations to
            apply to the data.
        depth_scale (float): Scale factor for depth data. Default is 1000.
        **kwargs: Additional arguments for the base RODataset class.
    """

    def __init__(
        self,
        cam_names: list[str],
        load_image=True,
        load_depth=True,
        load_extrinsic=True,
        load_ee_state=False,
        depth_scale=1000,
        use_detailed_instruction=False,
        hist_steps=1,
    ):
        """Initialize the ManipulationRODataset."""
        self.cam_names = cam_names
        self.load_image = load_image
        self.load_depth = load_depth
        self.load_extrinsic = load_extrinsic
        self.load_ee_state = load_ee_state
        self.depth_scale = depth_scale
        self.use_detailed_instruction = use_detailed_instruction
        self.hist_steps = hist_steps

    def get_instruction(self, data):
        """Parse instruction text from the data."""
        instruction = data.get("instruction")
        if instruction is None:
            text = data.get("task_name", "place objects to basket")
        else:
            text = instruction.json_content["description"]
        return {"text": text}

    def get_depths(self, data, default_shape):
        """Parse depth images from the data."""
        depths = []
        for cam_name in self.cam_names:
            feature_name = f"{cam_name}_depth"
            if feature_name in data and data[feature_name]:
                depth_buffer = data[feature_name].sensor_data[0]
                decoded_depth = cv2.imdecode(
                    np.frombuffer(depth_buffer, np.uint8), cv2.IMREAD_UNCHANGED
                )
            else:
                # fill missing depth
                decoded_depth = np.zeros(default_shape)
            assert decoded_depth is not None, (
                f"Failed to decode depth for {cam_name}"
            )
            depth = decoded_depth / self.depth_scale
            depths.append(depth)
        return {"depths": depths}

    def get_images(self, data):
        """Parse rgb images from the data."""
        images = []
        for cam_name in self.cam_names:
            frame_id = f"{cam_name}"
            img_buffer = data[frame_id].sensor_data[0]
            img_buffer = np.ndarray(
                shape=(1, len(img_buffer)), dtype=np.uint8, buffer=img_buffer
            )
            img = cv2.imdecode(img_buffer, cv2.IMREAD_ANYCOLOR)
            images.append(img)
        images = np.stack(images)
        return {"imgs": images}

    def get_intrinsic(self, data):
        """Parse camera intrinsic matrices from the data."""
        intrinsic = []
        for cam_name in self.cam_names:
            frame_id = f"{cam_name}"
            cam_instrinsic = np.eye(4, dtype=np.float64)
            cam_instrinsic[:3, :3] = data[frame_id].intrinsic_matrices[0]
            intrinsic.append(cam_instrinsic)
        intrinsic = np.stack(intrinsic)
        return {"intrinsic": intrinsic}

    def get_joints(self, data):
        """Parse robot joint states from the data."""
        joint_state = [item.position for item in data["joints"]]
        joint_state = np.stack(joint_state).squeeze(1).astype(np.float64)
        return {"joint_state": joint_state}

    def get_master_joints(self, data):
        """Parse master (controller) joint states from the data."""
        master_joint_state = [item.position for item in data["actions"]]
        master_joint_state = (
            np.stack(master_joint_state).squeeze(1).astype(np.float64)
        )
        return {"master_joint_state": master_joint_state}

    def get_extrinsic(self, data):
        """Parse camera extrinsic matrices from the data."""
        T_world2cam = []  # noqa: N806
        for cam_name in self.cam_names:
            frame_id = data[cam_name].frame_id
            cam_extrinsic = data[cam_name].pose
            assert cam_extrinsic.parent_frame_id == "world"
            assert (
                cam_extrinsic.child_frame_id == frame_id
                or cam_extrinsic.child_frame_id == cam_name
            )
            extrinsic = np.linalg.inv(
                data[cam_name].pose.as_Transform3D_M().get_matrix()[0].numpy()
            )
            T_world2cam.append(extrinsic)

        T_world2cam = np.stack(T_world2cam).astype(np.float64)  # noqa: N806
        return {"T_world2cam": T_world2cam}

    def __call__(self, data):
        data.update(self.get_instruction(data))
        data.update(self.get_intrinsic(data))
        data.update(self.get_joints(data))
        data.update(self.get_master_joints(data))
        if self.load_image:
            data.update(self.get_images(data))
        if self.load_depth:
            img_shape = data["imgs"][0].shape[:2]
            data.update(self.get_depths(data, default_shape=img_shape))
        if self.load_extrinsic:
            data.update(self.get_extrinsic(data))
        data["step_index"] = data["frame_index"]
        data["step_index_in_chunk"] = self.hist_steps - 1
        task = data.get("task")
        data["task_name"] = task.name if task is not None else "grasp_anything_ro"
        episode = data.get("episode")
        data["uuid"] = (
            episode.info.get("uuid")
            if episode is not None
            else f"{data['task_name']}_{data['frame_index']}"
        )
        return data


class AgilexRODataset(TorchDataset):
    """Agilex robot dataset backed by RoboOrchard dataset format.

    Exposes only the standard torch.utils.data.Dataset interface,
    hiding RO-specific internals (features, dataset_index_key, etc.).

    Args:
        paths: Arrow dataset paths to load and concatenate.
        target_columns: Column names to sample (e.g. ["joints", "actions"]).
        hist_steps: Number of historical steps in each chunk.
        pred_steps: Number of prediction steps in each chunk.
        cam_names: List of camera names to load data from.
        depth_scale: Scale factor for depth data. Default is 1000.
        load_image: Whether to load image data. Default is True.
        load_depth: Whether to load depth data. Default is True.
        load_extrinsic: Whether to load camera extrinsic data.
            Default is True.
        transforms: Optional list of callables applied after Arrow parsing.
    """

    def __init__(
        self,
        paths: list[str],
        target_columns: list[str],
        hist_steps: int,
        pred_steps: int,
        cam_names: list[str],
        depth_scale: int = 1000,
        load_image: bool = True,
        load_depth: bool = True,
        load_extrinsic: bool = True,
        transforms: list[Callable] | None = None,
    ):
        assert len(paths) > 0, "paths must not be empty"
        row_sampler = EpisodeChunkSamplerConfig(
            target_columns=target_columns,
            hist_steps=hist_steps,
            pred_steps=pred_steps,
        )
        datasets = [
            ROMultiRowDataset(dataset_path=p, row_sampler=row_sampler)
            for p in paths
        ]
        arrow_parser = ArrowDataParse(
            cam_names=cam_names,
            depth_scale=depth_scale,
            hist_steps=hist_steps,
            load_image=load_image,
            load_depth=load_depth,
            load_extrinsic=load_extrinsic,
        )
        composed = Compose([arrow_parser] + (transforms or []))
        for ds in datasets:
            ds.set_transform(composed)
        self._concat = ConcatRODataset(datasets)

    @property
    def num_episode(self) -> int:
        return sum(ds.episode_num for ds in self._concat.datasets)

    def get_episode_range(self, ep_idx: int) -> tuple[int, int]:
        global_ep = 0
        frame_offset = 0
        for ds in self._concat.datasets:
            for ep in ds.iterate_meta(Episode):
                if global_ep == ep_idx:
                    start = frame_offset + ep.dataset_begin_index
                    return start, start + ep.frame_num
                global_ep += 1
            frame_offset += len(ds)
        raise KeyError(f"Episode index {ep_idx} not found in dataset")

    def __len__(self):
        return len(self._concat)

    def __getitem__(self, idx):
        return self._concat[idx]

    def __getitems__(self, indices):
        return self._concat.__getitems__(indices)
