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

import importlib
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
from safetensors.torch import load_model
from terminaltables import AsciiTable
from tqdm import tqdm

from robo_orchard_lab.utils import as_sequence
from robo_orchard_lab.utils.huggingface import (
    auto_add_repo_type,
    download_hf_resource,
)

logger = logging.getLogger(__file__)


@dataclass
class HolobrainDataFeature:
    uuid: str
    imgs: np.ndarray  # (N_cams, H, W, 3) uint8
    depths: np.ndarray  # (N_cams, H, W) float
    projection_mat: Optional[np.ndarray] = None  # (N_cams, 4, 4)
    hist_robot_state: Optional[list] = None  # list of (N_joints, 8)

    def __post_init__(self):
        for attr in ("imgs", "depths", "projection_mat"):
            val = getattr(self, attr)
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.cpu().numpy())
        if self.hist_robot_state is not None:
            self.hist_robot_state = [
                s.cpu().numpy() if isinstance(s, torch.Tensor) else s
                for s in self.hist_robot_state
            ]

    @classmethod
    def from_dict(cls, raw: dict) -> "HolobrainDataFeature":
        return cls(
            uuid=raw["uuid"],
            imgs=raw["imgs"],
            depths=raw["depths"],
            projection_mat=raw.get("projection_mat"),
            hist_robot_state=raw.get("hist_robot_state"),
        )


def load_config(config_file):
    assert config_file.endswith(".py")
    config_dir, module_name = os.path.split(config_file)
    sys.path.insert(0, config_dir)
    module_name = module_name[:-3]
    spec = importlib.util.spec_from_file_location(module_name, config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


class GetFile:
    def __init__(self, url):
        self.url = url

    def __enter__(self):
        if self.url.startswith("http"):
            file_name = "_" + self.url.split("/")[-1]
            with requests.get(self.url, stream=True) as r:
                r.raise_for_status()
                with open(file_name, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            self.url = file_name
            return file_name
        elif self.url.startswith("hf://"):
            return download_hf_resource(auto_add_repo_type(self.url))
        elif os.path.exists(self.url):
            return self.url
        else:
            raise ValueError(f"Invalid checkpoint url: {self.url}.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def load_checkpoint(model, checkpoint=None, accelerator=None, **kwargs):
    if checkpoint is None:
        return

    logger.info(f"load checkpoint: {checkpoint}")
    with GetFile(checkpoint) as checkpoint:
        if checkpoint.endswith(".safetensors"):
            missing_keys, unexpected_keys = load_model(
                model, checkpoint, strict=False, **kwargs
            )
        else:
            state_dict = torch.load(checkpoint, weights_only=True)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False, **kwargs
            )
        if accelerator is None or accelerator.is_main_process:
            logger.info(
                f"num of missing_keys: {len(missing_keys)},"
                f"num of unexpected_keys: {len(unexpected_keys)}"
            )
            logger.info(
                f"missing_keys:\n {missing_keys}\n"
                f"unexpected_keys:\n {unexpected_keys}"
            )


class ActionMetric:
    def __init__(
        self,
        num_modes: int | list[int] = 1,
        eval_horizons: Optional[int | list[int]] = None,
        end_effector_idx: Optional[int | list[int]] = None,
    ):
        self.num_modes = as_sequence(num_modes)
        self.eval_horizons = (
            as_sequence(eval_horizons) if eval_horizons is not None else None
        )
        self.reset()
        self.end_effector_idx = (
            as_sequence(end_effector_idx)
            if end_effector_idx is not None
            else None
        )
        self.reset()

    def reset(self):
        self.results = []

    def compute(self, accelerator):
        results = accelerator.gather_for_metrics(
            self.results, use_gather_object=True
        )
        if accelerator.is_main_process:
            metrics = self.compute_metrics(results)
        else:
            metrics = None
        return metrics

    def update(self, batch, model_outputs):
        for i, output in enumerate(model_outputs):
            self.results.append(
                dict(
                    pred_actions=output["pred_actions"].cpu(),
                    gt_actions=batch["pred_robot_state"][i].cpu(),
                )
            )

    def compute_metrics(self, results):
        if isinstance(self.eval_horizons, (tuple, list)):
            metrics = dict()
            for h in self.eval_horizons:
                metrics.update(self._compute_metrics(results, h))
            horizons = [x for x in self.eval_horizons]
        else:
            metrics = self._compute_metrics(results)
            horizons = [results[0]["pred_actions"].shape[1]]

        horizons = [str(x) for x in horizons]
        table_rows = [["metric", "joint_idx", "mode"] + horizons]
        mean_table_rows = [["metric", "joint_idx", "mode"] + horizons]
        ee_table_rows = [["ee_metric", "joint_idx", "mode"] + horizons]
        num_joint = results[0]["pred_actions"].shape[2]
        joints = ["mean"] + [str(x) for x in range(num_joint)]

        if self.end_effector_idx is None:
            end_effector_idx = [f"{num_joint - 1}"]
        else:
            end_effector_idx = [f"{x}" for x in self.end_effector_idx]

        for metric in [
            "average_joint",
            "final_joint",
            "average_xyz",
            "final_xyz",
            "average_quat",
            "final_quat",
            "jerk",
            "jerk_xyz",
        ]:
            for mode in self.num_modes:
                for joint in joints:
                    values = []
                    for horizon in horizons:
                        values.append(
                            "{:.6f}".format(
                                metrics[f"{metric}@{joint}@{mode}@{horizon}"]
                            )
                        )
                    row = [metric, joint, mode] + values
                    table_rows.append(row)
                    if joint == "mean":
                        mean_table_rows.append(row)
                    if joint in end_effector_idx:
                        ee_table_rows.append(row)
            table_rows.append([])

        for rows in [table_rows, ee_table_rows, mean_table_rows]:
            table = AsciiTable(rows)
            logger.info("\n" + table.table)
        return metrics

    def _compute_metrics(self, results, horizon=None):
        average_joint_errors = []
        final_joint_errors = []
        average_xyz_errors = []
        final_xyz_errors = []
        average_quat_errors = []
        final_quat_errors = []
        jerks = []
        jerks_xyz = []

        A, XYZ, ROT = (0,), (1, 2, 3), (4, 5, 6, 7)  # noqa: N806

        for ret in results:
            pred = ret["pred_actions"]
            gt = ret["gt_actions"]
            if horizon is not None:
                pred = ret["pred_actions"][:, :horizon]
                gt = ret["gt_actions"][:horizon]
            error = torch.abs(pred - gt)
            average_error = error.mean(dim=1)
            final_error = error[:, -1]
            average_joint_errors.append(average_error[..., A])
            final_joint_errors.append(final_error[..., A])
            average_xyz_errors.append(
                torch.norm(average_error[..., XYZ], dim=-1)
            )
            final_xyz_errors.append(torch.norm(final_error[..., XYZ], dim=-1))
            average_quat_errors.append(
                torch.norm(average_error[..., ROT], dim=-1)
            )
            final_quat_errors.append(torch.norm(final_error[..., ROT], dim=-1))
            jerk = pred[..., A].diff(n=3, dim=1).abs().mean(dim=1)
            jerks.append(jerk)
            jerk_xyz = pred[..., XYZ].diff(n=3, dim=1).norm(dim=-1).mean(dim=1)
            jerks_xyz.append(jerk_xyz)

        values = dict(
            average_joint=torch.stack(average_joint_errors),
            final_joint=torch.stack(final_joint_errors),
            average_xyz=torch.stack(average_xyz_errors),
            final_xyz=torch.stack(final_xyz_errors),
            average_quat=torch.stack(average_quat_errors),
            final_quat=torch.stack(final_quat_errors),
            jerk=torch.stack(jerks),
            jerk_xyz=torch.stack(jerks_xyz),
        )

        num_joint = average_joint_errors[0].shape[1]
        metrics = dict()
        if horizon is None:
            horizon = ret["pred_actions"].shape[1]

        for num_mode in self.num_modes:
            for k, v in values.items():
                assert num_mode <= v.shape[1]
                metrics[f"{k}@mean@{num_mode}@{horizon}"] = (
                    v[:, :num_mode].mean(dim=2).min(dim=1)[0].mean()
                )
                for i in range(num_joint):
                    metrics[f"{k}@{i}@{num_mode}@{horizon}"] = (
                        v[:, :num_mode, i].min(dim=1)[0].mean()
                    )
        return metrics


class HolobrainVideoVisualizer:
    """Visualizer for robot episodes in Holobrain Data Feature format.

    Supports any dataset with ``episode_num``, ``__getitem__`` returning
    a :class:`HolobrainDataFeature`-compatible dict, and
    ``get_episode_range(ep_idx)`` returning ``(start, end)`` frame indices.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def visualize(
        self,
        episode_index,
        output_dir,
        ee_indices=(6, 13),
        fps=25,
        interval=1,
    ):
        """Visualizes a complete episode and saves it as an MP4 video file.

        Args:
            episode_index: Global index of the episode within the dataset.
            output_dir: Directory where the rendered video will be stored.
            ee_indices: Indices of joints to be highlighted as end-effectors.
            fps: Frames per second for the output video.
            interval: Step size for frame sampling (stride).
        """
        import imageio

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        start, end = self.dataset.get_episode_range(episode_index)
        logger.info(f"episode start_idx: {start}, end_idx: {end}")

        first_frame = HolobrainDataFeature.from_dict(self.dataset[start])
        uuid = first_frame.uuid
        out_name = uuid.replace("/", "-").replace(" ", "-")
        save_path = output_dir / f"{out_name}.mp4"
        logger.info(f"video save path: {save_path.absolute()}")

        frames = []
        for idx in tqdm(range(start, end, interval)):
            frame_data = HolobrainDataFeature.from_dict(self.dataset[idx])
            frame = self._render_frame(frame_data, ee_indices)
            frames.append(frame)

        imageio.mimwrite(str(save_path), frames, fps=fps)

    def _render_frame(
        self, data: HolobrainDataFeature, ee_indices
    ) -> np.ndarray:
        """Renders a single frame combining multi-view RGB and depth.

        Args:
            data: A :class:`HolobrainDataFeature` from ``dataset[idx]``.
            ee_indices: Joint indices to highlight with larger axes.

        Returns:
            np.ndarray: Combined image array (uint8) with RGB and Depth rows.
        """
        robot_state = (
            data.hist_robot_state[-1]
            if data.hist_robot_state is not None
            else None
        )

        vis_imgs = self.get_vis_imgs(
            data.imgs, data.projection_mat, robot_state, ee_indices=ee_indices
        )
        vis_depths = self.depth_visualize(data.depths)
        vis_depths = np.reshape(
            vis_depths.transpose(1, 0, 2, 3), vis_imgs.shape
        )

        return np.concatenate([vis_imgs, vis_depths], axis=0)

    @staticmethod
    def get_vis_imgs(imgs, projection_mat, robot_state, ee_indices):
        """Projects 3D robot joint frames onto 2D camera images.

        Args:
            imgs: Input images array of shape (Num_Cameras, H, W, 3).
            projection_mat: Camera projection matrices [P = K * [R|t]].
            robot_state: Joint value with joint states (pos + quat). (N, 8).
            ee_indices: Indices of joints to render with enhanced axis length.

        Returns:
            np.ndarray: Horizontally concatenated images from all camera views.
        """
        if imgs.ndim == 3:
            imgs = imgs[None]

        vis_list = []
        for cam_idx in range(imgs.shape[0]):
            img = imgs[cam_idx].copy()

            if projection_mat is not None and robot_state is not None:
                joints = HolobrainVideoVisualizer._project_joints_to_2d(
                    robot_state, projection_mat[cam_idx], ee_indices
                )
                for pts2d, j in joints:
                    HolobrainVideoVisualizer._draw_joint_overlay(
                        img, pts2d, j, robot_state, ee_indices
                    )
                img = img[:, :, ::-1]  # BGR to RGB

            vis_list.append(img)

        return np.uint8(np.concatenate(vis_list, axis=1))

    @staticmethod
    def depth_visualize(depth, min_depth=0.01, max_depth=1.2, mode="bwr"):
        """Colorizes a depth map using a matplotlib colormap.

        Args:
            depth: Raw depth map array.
            min_depth: Minimum depth value for colormap scaling.
            max_depth: Maximum depth value for colormap scaling.
            mode: Colormap name (e.g., 'plasma', 'bwr').

        Returns:
            np.ndarray: Colorized uint8 image.
        """
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

    @staticmethod
    def _project_joints_to_2d(robot_state, proj_matrix, ee_indices):
        """Projects 3D robot joint frames to 2D points for a single camera.

        Returns:
            list of (pts2d, joint_idx) tuples for joints with valid depth.
        """
        from scipy.spatial.transform import Rotation

        results = []
        for j in range(robot_state.shape[0]):
            rot = Rotation.from_quat(
                robot_state[j, 4:], scalar_first=True
            ).as_matrix()
            trans = robot_state[j, 1:4]
            axis_len = 0.1 if j in ee_indices else 0.03
            points = np.float32(
                [
                    [axis_len, 0, 0],
                    [0, axis_len, 0],
                    [0, 0, axis_len],
                    [0, 0, 0],
                ]
            )
            points = points @ rot.T + trans
            pts3 = points @ proj_matrix[:3, :3].T + proj_matrix[:3, 3]
            depth = pts3[:, 2]
            if depth[3] < 0.02:
                continue
            pts2d = (pts3[:, :2] / depth[:, None]).astype(np.int32)
            results.append((pts2d, j))
        return results

    @staticmethod
    def _draw_joint_overlay(img, pts2d, joint_idx, robot_state, ee_indices):
        """Draws axis lines, tips, and gripper value for one joint."""
        import cv2

        for ax in range(3):
            color = [0, 0, 0]
            color[ax] = 255
            cv2.line(img, tuple(pts2d[3]), tuple(pts2d[ax]), tuple(color), 3)

        for ax in range(3):
            cv2.circle(img, tuple(pts2d[ax]), 5, (0, 0, 255), -1)

        if joint_idx in ee_indices:
            gripper_value = robot_state[joint_idx, 0]
            x, y = int(pts2d[3][0]) + 5, int(pts2d[3][1]) - 5
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                cv2.putText(
                    img,
                    f"G<{joint_idx}>: {gripper_value:.2f}",
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
