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

import copy
import json
import logging
import os
from typing import List

import cv2
import numpy as np
import torch

from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseIndexData,
    BaseLmdbManipulationDataset,
)

logger = logging.getLogger(__name__)


class RoboTwinLmdbDataset(BaseLmdbManipulationDataset):
    """RoboTwin LMDB Dataset.

    Index structure:

    .. code-block:: text

        {episode_idx}:
            ├── uuid: str
            ├── task_name: str
            ├── num_steps: int
            └── simulation: bool

    Meta data structure:

    .. code-block:: text

        {uuid}/meta_data: dict
        {uuid}/camera_names: list(str)
        {uuid}/extrinsic
            └── {cam_name}: np.ndarray[num_steps x 4 x 4]
        {uuid}/intrinsic
            ├── {cam_name}: np.ndarray[3 x 3]
        {uuid}/observation/robot_state/cartesian_position
        {uuid}/observation/robot_state/joint_positions

    Image storage:

    .. code-block:: text

        {uuid}/{cam_name}/{step_idx}

    Depth storage:

    .. code-block:: text

        {uuid}/{cam_name}/{step_idx}
    """

    DEFAULT_INSTRUCTIONS = {
        "block_hammer_beat": "Hit the block with a hammer.",
        "block_handover": "Left arm picks up a block and passes it to the right arm, which places it on the blue mat.",  # noqa: E501
        "blocks_stack_easy": "Stack red cubes first, then black cubes in a specific spot.",  # noqa: E501
        "blocks_stack_hard": "Stack red cubes first, then green, then blue cubes in a specific spot.",  # noqa: E501
        "bottle_adjust": "Pick up the bottle so its head faces up.",
        "container_place": "Move containers to a fixed plate.",
        "diverse_bottles_pick": "Lift two bottles with different designs to a designated location using both arms.",  # noqa: E501
        "dual_bottles_pick_easy": "Pick up a red bottle from the left and a green bottle from the right, and move them together to a designated location.",  # noqa: E501
        "dual_bottles_pick_hard": "Pick up a red bottle from the left and a green bottle from the right in any position, and move them together to a designated location.",  # noqa: E501
        "dual_shoes_place": "Pick up shoes from each side and place them in a blue area with heads facing left.",  # noqa: E501
        "empty_cup_place": "Place an empty cup on a cup mat.",
        "mug_hanging_easy": "Move a mug to the middle and hang it on a fixed rack.",  # noqa: E501
        "mug_hanging_hard": "Move a mug to the middle and hang it on a randomly placed rack.",  # noqa: E501
        "pick_apple_messy": "Pick up an apple from among other items.",
        "put_apple_cabinet": "Open a cabinet and place an apple inside.",
        "shoe_place": "Move shoes to the shoebox or the blue area with heads facing left.",  # noqa: E501
        "tool_adjust": "Pick up a tool so its head faces up.",
        "put_bottles_dustbin": "Put the bottles into the dustbin.",
        "bowls_stack": "Stack the bowls together and put them in a specific spot.",  # noqa: E501
        "classify_tactile": "Classify the block according to its shapes and put them in the corresponding positions.",  # noqa: E501
        "others": "Complete all the tasks you see",
    }

    def __init__(
        self,
        paths,
        transforms=None,
        interval=None,
        load_image=True,
        load_depth=True,
        task_names=None,
        lazy_init=False,
        num_episode=None,
        cam_names=None,
        T_base2world=None,  # noqa: N803
        T_base2ego=None,  # noqa: N803
        default_space="base",
        instructions=None,
        instruction_keys=("seen", "unseen"),
        **kwargs,
    ):
        super().__init__(
            paths=paths,
            transforms=transforms,
            interval=interval,
            load_image=load_image,
            load_depth=load_depth,
            task_names=task_names,
            lazy_init=lazy_init,
            num_episode=num_episode,
            **kwargs,
        )
        self.cam_names = cam_names
        if T_base2world is None:
            logger.warning("dataset T_base2world is not set, use default.")
            T_base2world = np.array(  # noqa: N806
                [
                    [0, -1, 0, 0],
                    [1, 0, 0, -0.65],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
        elif isinstance(T_base2world, List):
            T_base2world = np.array(T_base2world)  # noqa: N806
        self.T_base2world = T_base2world
        self.T_base2ego = T_base2ego
        assert default_space in ["base", "world", "ego"]
        self.default_space = default_space
        self.load_instructions(instructions)
        self.instruction_keys = instruction_keys

    def load_instructions(self, instructions):
        if instructions is None:
            self.instructions = self.DEFAULT_INSTRUCTIONS
        elif os.path.isfile(instructions):
            assert instructions.endswith(".json")
            self.instructions = json.load(open(instructions, "r"))
        else:
            assert isinstance(instructions, dict)
            self.instructions = instructions

    def __getitem__(self, index):
        lmdb_index, episode_index, step_index = self._get_indices(index)

        idx_data = BaseIndexData.model_validate(
            self.idx_lmdbs[lmdb_index][episode_index]
        )
        uuid = idx_data.uuid
        task_name = idx_data.task_name
        if self.cam_names is not None:
            cam_names = self.cam_names
        else:
            cam_names = self.meta_lmdbs[lmdb_index][f"{uuid}/camera_names"]

        _T_world2cam = self.meta_lmdbs[lmdb_index][f"{uuid}/extrinsic"]  # noqa: N806
        _intrinsic = self.meta_lmdbs[lmdb_index][f"{uuid}/intrinsic"]

        if self.load_image:
            images = []
        if self.load_depth:
            depths = []

        T_world2cam = []  # noqa: N806
        intrinsic = []
        for cam_name in cam_names:
            if self.load_image:
                image = self.img_lmdbs[lmdb_index][
                    f"{uuid}/{cam_name}/{step_index}"
                ]
                if isinstance(image, bytes):
                    image = np.frombuffer(image, np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
                images.append(image)
            if self.load_depth:
                depth = self.depth_lmdbs[lmdb_index][
                    f"{uuid}/{cam_name}/{step_index}"
                ]
                if isinstance(depth, bytes):
                    depth = np.frombuffer(depth, np.uint8)
                depth = (
                    cv2.imdecode(
                        depth,
                        cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED,
                    )
                    / 1000
                )
                depths.append(depth)

            _tmp = np.eye(4)
            if _T_world2cam[cam_name].ndim == 3:  # for dynamic camera
                _tmp[:3] = _T_world2cam[cam_name][step_index][:3]
            else:  # ndim == 2, for fixed camera
                _tmp[:3] = _T_world2cam[cam_name][:3]
            T_world2cam.append(_tmp)

            _tmp = np.eye(4)
            _tmp[:3, :3] = _intrinsic[cam_name][:3, :3]
            intrinsic.append(_tmp)

        if self.load_image:
            images = np.stack(images)
        if self.load_depth:
            depths = np.stack(depths)
        T_world2cam = np.stack(T_world2cam)  # noqa: N806
        intrinsic = np.stack(intrinsic)

        joint_state = self.meta_lmdbs[lmdb_index][
            f"{uuid}/observation/robot_state/joint_positions"
        ]
        ee_state = self.meta_lmdbs[lmdb_index][
            f"{uuid}/observation/robot_state/cartesian_position"
        ]
        if ee_state.ndim == 3:
            ee_state = ee_state.reshape(ee_state.shape[0], -1)

        data = dict(
            uuid=uuid,
            step_index=step_index,
            intrinsic=intrinsic,
            T_world2cam=T_world2cam,
            T_base2world=copy.deepcopy(self.T_base2world),
            joint_state=joint_state,
            ee_state=ee_state,
        )
        if self.T_base2ego is not None:
            data["T_base2ego"] = copy.deepcopy(self.T_base2ego)
        if self.load_image:
            data["imgs"] = images
        if self.load_depth:
            data["depths"] = depths

        instructions = self.meta_lmdbs[lmdb_index][f"{uuid}/instructions"]
        if instructions is None:
            meta_data = self.meta_lmdbs[lmdb_index][f"{uuid}/meta_data"]
            instructions = meta_data.get("instruction")

        if instructions is None:
            instructions = self.instructions.get(
                task_name,
                self.DEFAULT_INSTRUCTIONS["others"],
            )
        elif isinstance(instructions, dict):
            _tmp = []
            for k in self.instruction_keys:
                if isinstance(instructions[k], str):
                    _tmp.append(instructions[k])
                else:
                    _tmp.extend(instructions[k])
            instructions = _tmp

        if isinstance(instructions, str):
            text = instructions
        elif len(instructions) == 0:
            text = ""
        else:
            idx = np.random.randint(len(instructions))
            text = instructions[idx]
        data["text"] = text
        for transform in self.transforms:
            if transform is None:
                continue
            data = transform(data)
        return data

    def visualize(
        self,
        episode_index,
        output_path="./vis_data",
        fps=25,
        interval=1,
        **kwargs,
    ):
        from tqdm import tqdm

        end_idx = self.cumsum_steps[episode_index]
        if episode_index != 0:
            start_idx = self.cumsum_steps[episode_index - 1]
        else:
            start_idx = 0
        videoWriter = None  # noqa: N806
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        uuid = self.__getitem__(start_idx)["uuid"]
        file = os.path.join(output_path, f"{uuid.replace('/', '-')}.mp4")

        logger.info(f"episode start_idx: {start_idx}, end_idx: {end_idx}")
        logger.info(f"video save path: {file}")

        for i in tqdm(list(range(start_idx, end_idx, interval))):
            data = self.__getitem__(i)

            vis_imgs = self.get_vis_imgs(
                data["imgs"],
                data.get("projection_mat"),
                data.get("hist_robot_state", [None])[-1],
                **kwargs,
            )
            if "depths" in data.keys():
                vis_depths = self.depth_visualize(data["depths"])
                vis_depths = np.reshape(
                    vis_depths.transpose(1, 0, 2, 3), vis_imgs.shape
                )
                vis_imgs = np.concatenate([vis_imgs, vis_depths], axis=0)

            if videoWriter is None:
                videoWriter = cv2.VideoWriter(  # noqa: N806
                    file,
                    fourcc,
                    fps // interval,
                    vis_imgs.shape[:2][::-1],
                )
            videoWriter.write(vis_imgs)
        videoWriter.release()

    @staticmethod
    def get_vis_imgs(
        imgs,
        projection_mat=None,
        robot_state=None,
        ee_indices=(6, 13),
        channel_conversion=False,
    ):
        from scipy.spatial.transform import Rotation

        if isinstance(imgs, torch.Tensor):
            imgs = imgs.cpu().numpy()
        if isinstance(projection_mat, torch.Tensor):
            projection_mat = projection_mat.cpu().numpy()
        if isinstance(robot_state, torch.Tensor):
            robot_state = robot_state.cpu().numpy()

        vis_imgs = []
        for img_index in range(imgs.shape[0]):
            img = imgs[img_index]
            if robot_state is None or projection_mat is None:
                vis_imgs.append(img)
                continue
            for joint_index in range(robot_state.shape[0]):
                rot = Rotation.from_quat(
                    robot_state[joint_index, 4:], scalar_first=True
                ).as_matrix()
                trans = robot_state[joint_index, 1:4]

                if joint_index in ee_indices:
                    axis_length = 0.1
                else:
                    axis_length = 0.03
                points = np.float32(
                    [
                        [axis_length, 0, 0],
                        [0, axis_length, 0],
                        [0, 0, axis_length],
                        [0, 0, 0],
                    ]
                )
                points = points @ rot.T + trans

                pts_2d = points @ projection_mat[img_index, :3, :3].T
                pts_2d = pts_2d + projection_mat[img_index, :3, 3]
                depth = pts_2d[:, 2]
                pts_2d = pts_2d[:, :2] / depth[:, None]

                if depth[3] < 0.02:
                    continue

                pts_2d = pts_2d.astype(np.int32)
                for i in range(3):
                    if depth[i] < 0.02:
                        continue
                    cv2.circle(
                        img, (pts_2d[i, 0], pts_2d[i, 1]), 6, (0, 0, 255), -1
                    )
                    if i == 3:
                        continue
                    color = [0, 0, 0]
                    color[i] = 255
                    cv2.line(
                        img,
                        (pts_2d[i, 0], pts_2d[i, 1]),
                        (pts_2d[3, 0], pts_2d[3, 1]),
                        tuple(color),
                        3,
                    )
            vis_imgs.append(img)

        vis_imgs = np.concatenate(vis_imgs, axis=1)
        vis_imgs = np.uint8(vis_imgs)
        if channel_conversion:
            vis_imgs = vis_imgs[..., ::-1]
        return vis_imgs

    @staticmethod
    def depth_visualize(depth, min_depth=0.01, max_depth=1.2, mode="bwr"):
        import matplotlib.pyplot as plt

        mask = depth > 0
        cmap = plt.cm.get_cmap(mode, 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        cmap = cmap[::-1]

        depth_shape = depth.shape
        if max_depth is None:
            max_depth = depth.max()
        if min_depth is None:
            min_depth = depth.min()

        depth = (depth - min_depth) / (max_depth - min_depth)
        index = np.int32(depth * 255)
        index = np.clip(index, a_min=0, a_max=255)
        depth_color = cmap[index].reshape(*depth_shape, 3)
        depth_color = np.where(mask[..., None], depth_color, 0)
        depth_color = np.uint8(depth_color)

        return depth_color
