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
import argparse
import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import cv2
import datasets as hg_datasets
import numpy as np
import pyarrow.parquet as pq
import pytorch_kinematics as pk
import torch
from pytorch3d.transforms import matrix_to_quaternion

from robo_orchard_lab.dataset.datatypes import (
    BatchCameraDataEncoded,
    BatchCameraDataEncodedFeature,
    BatchFrameTransform,
    BatchFrameTransformGraph,
    BatchFrameTransformGraphFeature,
    BatchJointsState,
    BatchJointsStateFeature,
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

logger = logging.getLogger(__name__)

FOLDER_NAME_TO_CANONICAL_TASK = {
    "clean_the_desktop_addition": "clean_the_desktop",
    "clean_the_desktop_part_1": "clean_the_desktop",
    "clean_the_desktop_part_2": "clean_the_desktop",
    "hold_pot": "hold_pot",
    "open_door": "open_door",
    "place_block_into_box": "place_block_into_box",
    "pour_workpiece": "pour_workpiece",
    "scoop_popcorn": "scoop_popcorn",
    "scoop_popcorn_part_2": "scoop_popcorn",
    "sorting_packages_part_1": "sorting_packages",
    "sorting_packages_part_2": "sorting_packages",
    "sorting_packages_part_3": "sorting_packages",
    "stock_and_straighten_shelf": "stock_and_straighten_shelf",
    "stock_and_straighten_shelf_part_2": "stock_and_straighten_shelf",
    "take_wrong_item_shelf": "take_wrong_item_shelf",
}

TASK_TEXT = {
    "hold_pot": "Place the pot on the stove",
    "clean_the_desktop": "Clear the desktop",
    "open_door": "Open the door",
    "place_block_into_box": "Place the block into the matching hole",
    "pour_workpiece": "Pour the workpiece into the box",
    "scoop_popcorn": "Scoop popcorn into the bucket",
    "sorting_packages": "Sort packages",
    "sorting_packages_continuous": "Sort packages",
    "stock_and_straighten_shelf": "Stock and organize the shelf",
    "take_wrong_item_shelf": "Remove misplaced items from the shelf",
}

JOINT_NAMES = [
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
    "left_gripper",
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
    "right_gripper",
    "idx11_head_joint1",
    "idx12_head_joint2",
    "idx13_head_joint3",
    "idx01_body_joint1",
    "idx02_body_joint2",
    "idx03_body_joint3",
    "idx04_body_joint4",
    "idx05_body_joint5",
]

GRIPPER_JOINT_INDICES = (7, 15)
CAMERA_NAMES = ("top_head", "hand_left", "hand_right")
DEPTH_CAMERA_MAP = {
    "top_head": "head_depth",
    "hand_left": "hand_left_depth",
    "hand_right": "hand_right_depth",
}

# Camera poses are copied from the G2_omnipicker USD asset used by GenieSim3.
ROBOT_USDA_CONFIGS = {
    "top_head": {
        "parent_link": "head_link3",
        "xformOp:orient": (
            0.5945104121873723,
            0.3868316461705512,
            -0.3787291472834019,
            -0.5945442627086477,
        ),
        "xformOp:translate": (0.10237, 0.02375, 0.10256),
    },
    "hand_left": {
        "parent_link": "gripper_l_base_link",
        "xformOp:orient": (
            0.256282394630355,
            0.6590290844890921,
            -0.6590290844890923,
            -0.25628239463035496,
        ),
        "xformOp:translate": (
            -0.08979655916171847,
            -0.0011588271391879679,
            0.06070713906090863,
        ),
    },
    "hand_right": {
        "parent_link": "gripper_r_base_link",
        "xformOp:orient": (
            -0.25628239463035507,
            -0.6590290844890921,
            -0.6590290844890923,
            -0.256282394630355,
        ),
        "xformOp:translate": (0.0898, 0.00116, 0.06070713906090863),
    },
}

DEFAULT_INTRINSIC = {
    "top_head": {
        "fx": 306.6911,
        "fy": 306.55075,
        "ppx": 319.90094,
        "ppy": 201.29141,
    },
    "hand_left": {
        "fx": 486.13733,
        "fy": 485.94153,
        "ppx": 614.31964,
        "ppy": 529.99976,
    },
    "hand_right": {
        "fx": 465.1793,
        "fy": 465.0162,
        "ppx": 630.648,
        "ppy": 527.8828,
    },
}


@dataclass
class GenieSim3PackConfig:
    """Configuration for packing GenieSim3 challenge data into RO format."""

    input_dir: str
    output_dir: str
    task_name: str
    urdf_path: str
    dataset_name: str = "GenieSim3"
    robot_name: str = "G2_omnipicker"
    force_overwrite: bool = False
    cached_meta_path: str = ""
    max_shard_size: str | int = "8GB"
    writer_batch_size: int = 500
    num_jobs: int = 1
    job_idx: int = 0
    skip_static_frames: bool = True
    static_threshold: float = 1e-6


def _get_instruction_text(task_name: str) -> str:
    canonical_name = FOLDER_NAME_TO_CANONICAL_TASK.get(task_name, task_name)
    return TASK_TEXT.get(canonical_name, canonical_name)


def _build_instruction_frame_mask(
    num_steps: int, instruction_segments: list[dict[str, Any]]
) -> np.ndarray:
    if not instruction_segments:
        return np.ones(num_steps, dtype=bool)

    frame_mask = np.zeros(num_steps, dtype=bool)
    for segment in instruction_segments:
        start = max(0, int(segment["start_frame_index"]))
        end = min(num_steps, int(segment["end_frame_index"]))
        if start < end:
            frame_mask[start:end] = True
    return frame_mask


def _parse_field_indices(field_descs: dict[str, Any], name: str) -> list[int]:
    if name not in field_descs:
        raise KeyError(f"Field '{name}' not found in field_descriptions")
    indices = field_descs[name]["indices"]
    if not indices:
        raise ValueError(f"Field '{name}' has no indices")
    return indices


def _generate_static_mask(
    positions: torch.Tensor | np.ndarray, threshold: float
) -> np.ndarray:
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()
    positions = np.asarray(positions)

    num_steps = positions.shape[0]
    if num_steps == 0:
        return np.zeros(0, dtype=bool)

    static_mask = np.ones(num_steps, dtype=bool)
    if num_steps > 1:
        diffs = np.abs(np.diff(positions, axis=0))
        static_mask[1:] = np.all(diffs < threshold, axis=1)
    return static_mask


def _select_job_slice(total: int, num_jobs: int, job_idx: int) -> slice:
    if num_jobs <= 0:
        raise ValueError("num_jobs must be > 0")
    if not 0 <= job_idx < num_jobs:
        raise ValueError(f"job_idx must be in [0, {num_jobs})")

    chunk_size = (total + num_jobs - 1) // num_jobs
    start = job_idx * chunk_size
    end = min(start + chunk_size, total)
    return slice(start, end)


def _get_static_tf_from_usda(
    usda_cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    xyz = torch.tensor(usda_cfg["xformOp:translate"], dtype=torch.float32)
    quat_wxyz = torch.tensor(usda_cfg["xformOp:orient"], dtype=torch.float32)
    return xyz, quat_wxyz


def _slice_transform(
    transform: BatchFrameTransform, index: int
) -> BatchFrameTransform:
    xyz = transform.xyz
    quat = transform.quat
    if xyz.ndim == 1:
        xyz = xyz.unsqueeze(0)
    elif xyz.shape[0] == 1:
        xyz = xyz[:1]
    else:
        xyz = xyz[index : index + 1]
    if quat.ndim == 1:
        quat = quat.unsqueeze(0)
    elif quat.shape[0] == 1:
        quat = quat[:1]
    else:
        quat = quat[index : index + 1]

    timestamps = None
    if transform.timestamps is not None:
        timestamps = [transform.timestamps[index]]
    return BatchFrameTransform(
        parent_frame_id=transform.parent_frame_id,
        child_frame_id=transform.child_frame_id,
        xyz=xyz,
        quat=quat,
        timestamps=timestamps,
    )


def _get_index_camera_data(
    data: BatchCameraDataEncoded, index: int
) -> BatchCameraDataEncoded:
    if data.timestamps is None:
        raise ValueError("Camera data timestamps must be set")
    if data.intrinsic_matrices is None:
        raise ValueError("Camera data intrinsic_matrices must be set")

    pose = None
    if data.pose is not None:
        pose = _slice_transform(data.pose, index)
    return BatchCameraDataEncoded(
        topic=data.topic,
        frame_id=data.frame_id,
        image_shape=data.image_shape,
        intrinsic_matrices=data.intrinsic_matrices[index : index + 1],
        distortion=data.distortion,
        pose=pose,
        sensor_data=[data.sensor_data[index]],
        format=data.format,
        timestamps=[data.timestamps[index]],
    )


def _get_index_transform_graph(
    data: BatchFrameTransformGraph, index: int
) -> BatchFrameTransformGraph:
    tf_list = [_slice_transform(tf, index) for tf in data.as_state().tf_list]
    return BatchFrameTransformGraph(tf_list=tf_list)


def _probe_video_size(video_path: str) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            video_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {video_path}")
    return int(streams[0]["width"]), int(streams[0]["height"])


def _wait_ffmpeg(process: subprocess.Popen, video_path: str) -> None:
    stderr = ""
    if process.stderr is not None:
        stderr = process.stderr.read().decode(errors="replace")
    retcode = process.wait()
    if retcode != 0:
        raise RuntimeError(
            f"ffmpeg exited with code {retcode} for {video_path}: {stderr}"
        )


def _read_color_frames_mjpeg(video_path: str) -> list[bytes]:
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            video_path,
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-q:v",
            "2",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError(f"Failed to read ffmpeg stdout for {video_path}")

    frames: list[bytes] = []
    buffer = bytearray()
    soi = b"\xff\xd8"
    eoi = b"\xff\xd9"
    in_frame = False
    try:
        while True:
            chunk = process.stdout.read(65536)
            if not chunk:
                break
            buffer.extend(chunk)
            while True:
                if not in_frame:
                    soi_pos = buffer.find(soi)
                    if soi_pos == -1:
                        buffer = buffer[-1:]
                        break
                    buffer = buffer[soi_pos:]
                    in_frame = True
                eoi_pos = buffer.find(eoi, 2)
                if eoi_pos == -1:
                    break
                frame_end = eoi_pos + 2
                frames.append(bytes(buffer[:frame_end]))
                buffer = buffer[frame_end:]
                in_frame = False
    finally:
        process.stdout.close()
        _wait_ffmpeg(process, video_path)
    return frames


def _read_depth_frames(video_path: str) -> list[bytes]:
    width, height = _probe_video_size(video_path)
    frame_size = width * height * 2
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            video_path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray16le",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError(f"Failed to read ffmpeg stdout for {video_path}")

    frames: list[bytes] = []
    try:
        while True:
            in_bytes = process.stdout.read(frame_size)
            if not in_bytes:
                break
            if len(in_bytes) != frame_size:
                raise RuntimeError(
                    f"Short read in {video_path}: expected {frame_size} "
                    f"bytes, got {len(in_bytes)}"
                )
            image = np.frombuffer(in_bytes, dtype="<u2").reshape(height, width)
            ok, png_buffer = cv2.imencode(".png", image)
            if not ok:
                raise RuntimeError(
                    f"cv2.imencode failed for frame {len(frames)} "
                    f"in {video_path}"
                )
            frames.append(png_buffer.tobytes())
    finally:
        process.stdout.close()
        _wait_ffmpeg(process, video_path)
    return frames


def _read_video_frames(
    video_path: str, video_info: dict[str, Any]
) -> list[bytes]:
    if video_info.get("video.is_depth_map", False):
        return _read_depth_frames(video_path)
    return _read_color_frames_mjpeg(video_path)


class GenieSim3EpisodePackaging(EpisodePackaging):
    """Pack one GenieSim3 episode into RO dataset frames."""

    def __init__(
        self,
        meta: dict[str, Any],
        urdf_content: str,
        cfg: GenieSim3PackConfig,
    ):
        self.meta = meta
        self.urdf_content = urdf_content
        self.cfg = cfg
        self.has_depth = bool(meta.get("has_depth", False))

    def generate_episode_meta(self) -> EpisodeMeta:
        return EpisodeMeta(
            episode=EpisodeData(info={"uuid": self.meta["uuid"]}),
            robot=RobotData(
                name=self.cfg.robot_name,
                content=self.urdf_content,
                content_format=RobotDescriptionFormat.URDF,
            ),
            task=TaskData(name=self.cfg.task_name),
        )

    def _build_cameras(
        self, video_base: str, ep_name: str, num_steps: int
    ) -> tuple[
        dict[str, BatchCameraDataEncoded],
        dict[str, BatchCameraDataEncoded],
    ]:
        timestamps = list(range(num_steps))
        video_tasks: list[tuple[str, str, dict[str, Any]]] = []
        for cam_name in CAMERA_NAMES:
            color_path = os.path.join(
                video_base,
                f"observation.images.{cam_name}",
                f"{ep_name}.mp4",
            )
            color_info = self.meta["features_info"][
                f"observation.images.{cam_name}"
            ]["video_info"]
            video_tasks.append((cam_name, color_path, color_info))

            if self.has_depth:
                depth_dir = DEPTH_CAMERA_MAP[cam_name]
                depth_path = os.path.join(
                    video_base,
                    f"observation.images.{depth_dir}",
                    f"{ep_name}.mp4",
                )
                depth_info = self.meta["features_info"][
                    f"observation.images.{depth_dir}"
                ]["video_info"]
                video_tasks.append(
                    (f"{cam_name}_depth", depth_path, depth_info)
                )

        with ThreadPoolExecutor(max_workers=len(video_tasks)) as pool:
            futures = {
                key: pool.submit(_read_video_frames, path, info)
                for key, path, info in video_tasks
            }
            decoded = {key: future.result() for key, future in futures.items()}

        for key, frames in decoded.items():
            if len(frames) != num_steps:
                raise ValueError(
                    f"Camera {key}: expected {num_steps} frames, "
                    f"got {len(frames)}"
                )

        color_batches = {}
        depth_batches = {}
        for cam_name in CAMERA_NAMES:
            frames = decoded[cam_name]
            first_img = cv2.imdecode(
                np.frombuffer(frames[0], np.uint8), cv2.IMREAD_COLOR
            )
            if first_img is None:
                raise RuntimeError(
                    f"Failed to decode first frame for {cam_name}"
                )
            image_shape = first_img.shape[:2]

            intrinsic = DEFAULT_INTRINSIC[cam_name]
            intrinsic_matrix = torch.tensor(
                [
                    [intrinsic["fx"], 0.0, intrinsic["ppx"]],
                    [0.0, intrinsic["fy"], intrinsic["ppy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
            intrinsics = intrinsic_matrix.unsqueeze(0).repeat(num_steps, 1, 1)
            frame_id = f"{cam_name}_camera_color_optical_frame"
            color_batches[cam_name] = BatchCameraDataEncoded(
                topic=f"/observation/cameras/{cam_name}/color_image/image_raw",
                frame_id=frame_id,
                image_shape=image_shape,
                intrinsic_matrices=intrinsics,
                sensor_data=frames,
                format="jpeg",
                timestamps=timestamps,
            )

            if self.has_depth:
                depth_key = f"{cam_name}_depth"
                depth_batches[depth_key] = BatchCameraDataEncoded(
                    topic=f"/observation/cameras/{cam_name}/depth_image/image_raw",
                    frame_id=frame_id,
                    image_shape=image_shape,
                    intrinsic_matrices=intrinsics,
                    sensor_data=decoded[depth_key],
                    format="png",
                    timestamps=timestamps,
                )

        return color_batches, depth_batches

    def _map_to_fk_input(
        self,
        urdf_joint_names: list[str],
        batch_joint_state: BatchJointsState,
    ) -> np.ndarray:
        joint_states = batch_joint_state.position.numpy()
        joint_names = batch_joint_state.names or []
        fk_input = np.zeros((joint_states.shape[0], len(urdf_joint_names)))
        for i, name in enumerate(urdf_joint_names):
            if name in joint_names:
                fk_input[:, i] = joint_states[:, joint_names.index(name)]
        return fk_input

    def _build_tf_graph(
        self, num_steps: int, batch_joints: BatchJointsState
    ) -> BatchFrameTransformGraph:
        timestamps = list(range(num_steps))
        tf_list = []
        camera_to_optical_mat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        parent_link_names = []
        for cam_name in CAMERA_NAMES:
            usda_cfg = ROBOT_USDA_CONFIGS[cam_name]
            xyz, quat_wxyz = _get_static_tf_from_usda(usda_cfg)
            camera_to_link_tf = BatchFrameTransform(
                xyz=xyz,
                quat=quat_wxyz,
                parent_frame_id=usda_cfg["parent_link"],
                child_frame_id=f"{cam_name}_camera",
            )
            # BatchFrameTransform stores child-to-parent transforms.
            camera_to_link_mat = (
                camera_to_link_tf.as_Transform3D_M().get_matrix()
            )
            optical_to_link_mat = camera_to_link_mat @ camera_to_optical_mat
            optical_to_link_xyz = optical_to_link_mat[:, :3, 3]
            optical_to_link_quat = matrix_to_quaternion(
                optical_to_link_mat[:, :3, :3]
            )
            tf_list.append(
                BatchFrameTransform(
                    xyz=optical_to_link_xyz.repeat(num_steps, 1),
                    quat=optical_to_link_quat.repeat(num_steps, 1),
                    timestamps=timestamps,
                    parent_frame_id=usda_cfg["parent_link"],
                    child_frame_id=f"{cam_name}_camera_color_optical_frame",
                )
            )
            parent_link_names.append(usda_cfg["parent_link"])

        chain = pk.build_chain_from_urdf(self.urdf_content)
        urdf_joint_names = [joint.name for joint in chain.get_joints()]
        fk_input = self._map_to_fk_input(urdf_joint_names, batch_joints)
        link_poses_dict = chain.forward_kinematics(fk_input)

        for link_name, link_tf in link_poses_dict.items():
            if link_name not in parent_link_names:
                continue
            link_to_base_mat = link_tf.get_matrix()
            tf_list.append(
                BatchFrameTransform(
                    xyz=link_to_base_mat[:, :3, 3].to(torch.float32),
                    quat=matrix_to_quaternion(
                        link_to_base_mat[:, :3, :3]
                    ).to(torch.float32),
                    timestamps=timestamps,
                    parent_frame_id="base_link",
                    child_frame_id=link_name,
                )
            )

        return BatchFrameTransformGraph(tf_list=tf_list)

    def _parse_joint_data(
        self, table: Any
    ) -> tuple[BatchJointsState, BatchJointsState]:
        num_steps = len(table)
        state = np.stack(table.column("observation.state").to_pylist())
        action = np.stack(table.column("action").to_pylist())

        state_field_descs = self.meta["state_field_descs"]
        action_field_descs = self.meta["action_field_descs"]
        arm_idx = _parse_field_indices(
            state_field_descs, "state/joint/position"
        )
        left_gripper_idx = _parse_field_indices(
            state_field_descs, "state/left_effector/position"
        )
        right_gripper_idx = _parse_field_indices(
            state_field_descs, "state/right_effector/position"
        )
        head_idx = _parse_field_indices(
            state_field_descs, "state/head/position"
        )
        waist_idx = _parse_field_indices(
            state_field_descs, "state/waist/position"
        )

        arm_action_idx = _parse_field_indices(
            action_field_descs, "action/joint/position"
        )
        left_gripper_action_idx = _parse_field_indices(
            action_field_descs, "action/left_effector/position"
        )
        right_gripper_action_idx = _parse_field_indices(
            action_field_descs, "action/right_effector/position"
        )
        head_action_idx = _parse_field_indices(
            action_field_descs, "action/head/position"
        )
        waist_action_idx = _parse_field_indices(
            action_field_descs, "action/waist/position"
        )

        joint_positions = np.concatenate(
            [
                state[:, arm_idx][:, :7],
                state[:, left_gripper_idx],
                state[:, arm_idx][:, 7:],
                state[:, right_gripper_idx],
                state[:, head_idx],
                state[:, waist_idx],
            ],
            axis=1,
        )
        action_positions = np.concatenate(
            [
                action[:, arm_action_idx][:, :7],
                action[:, left_gripper_action_idx],
                action[:, arm_action_idx][:, 7:],
                action[:, right_gripper_action_idx],
                action[:, head_action_idx],
                action[:, waist_action_idx],
            ],
            axis=1,
        )

        timestamps = list(range(num_steps))
        batch_joints = BatchJointsState(
            position=torch.tensor(joint_positions, dtype=torch.float32),
            names=JOINT_NAMES,
            timestamps=timestamps,
        )
        batch_actions = BatchJointsState(
            position=torch.tensor(action_positions, dtype=torch.float32),
            names=JOINT_NAMES,
            timestamps=timestamps,
        )
        return batch_joints, batch_actions

    def generate_frames(self) -> Generator[DataFrame, None, None]:
        logger.info("Processing GenieSim3 episode: %s", self.meta["uuid"])
        table = pq.read_table(self.meta["parquet_path"])
        num_steps = len(table)
        if num_steps == 0:
            logger.warning("Episode %s has no frames.", self.meta["uuid"])
            return

        batch_joints, batch_actions = self._parse_joint_data(table)
        color_batches, depth_batches = self._build_cameras(
            self.meta["video_base"], self.meta["ep_name"], num_steps
        )
        tf_graph = self._build_tf_graph(num_steps, batch_joints)

        joint_static_mask = _generate_static_mask(
            batch_joints.position,
            threshold=self.cfg.static_threshold,
        )
        gripper_static_mask = _generate_static_mask(
            batch_actions.position[:, GRIPPER_JOINT_INDICES],
            threshold=self.cfg.static_threshold,
        )
        valid_frame_mask = np.ones(num_steps, dtype=bool)
        if self.cfg.skip_static_frames:
            valid_frame_mask &= ~(joint_static_mask & gripper_static_mask)

        valid_frame_mask &= _build_instruction_frame_mask(
            num_steps,
            self.meta.get("instruction_segments", []),
        )
        frame_indices = np.flatnonzero(valid_frame_mask).tolist()
        if len(frame_indices) == 0:
            logger.warning(
                "Episode %s has 0 frames after filtering.",
                self.meta["uuid"],
            )
            return

        high_level_instruction = self.meta.get("high_level_instruction", "")
        instruction = InstructionData(
            name=self.meta.get("instruction_name", self.cfg.task_name),
            json_content={"instruction": high_level_instruction},
        )
        text = _get_instruction_text(self.cfg.task_name)
        for frame_index in frame_indices:
            tf_graph_i = _get_index_transform_graph(tf_graph, frame_index)
            features = {
                "raw_frame_index": int(frame_index),
                "text": text,
                "joints": batch_joints[frame_index],
                "actions": batch_actions[frame_index],
                "tf_graph": tf_graph_i,
            }
            for cam_name in CAMERA_NAMES:
                color_data = _get_index_camera_data(
                    color_batches[cam_name], frame_index
                )
                color_data.pose = tf_graph_i.get_tf(
                    "base_link",
                    f"{cam_name}_camera_color_optical_frame",
                )
                features[cam_name] = color_data

            for depth_key, depth_batch in depth_batches.items():
                features[depth_key] = _get_index_camera_data(
                    depth_batch, frame_index
                )

            yield DataFrame(
                features=features,
                instruction=instruction,
                timestamp_ns_min=int(frame_index),
                timestamp_ns_max=int(frame_index),
            )


class GenieSim3RODataPacker:
    """Pack GenieSim3 challenge episodes into sharded RO datasets."""

    def __init__(self, cfg: GenieSim3PackConfig):
        self.cfg = cfg
        self.urdf_content = Path(cfg.urdf_path).read_text()

    def get_dataset_features(self, has_depth: bool) -> hg_datasets.Features:
        features = {
            "raw_frame_index": hg_datasets.Value("int32"),
            "text": hg_datasets.Value("string"),
            "joints": BatchJointsStateFeature(),
            "actions": BatchJointsStateFeature(),
            "top_head": BatchCameraDataEncodedFeature(),
            "hand_left": BatchCameraDataEncodedFeature(),
            "hand_right": BatchCameraDataEncodedFeature(),
            "tf_graph": BatchFrameTransformGraphFeature(),
        }
        if has_depth:
            features.update(
                {
                    "top_head_depth": BatchCameraDataEncodedFeature(),
                    "hand_left_depth": BatchCameraDataEncodedFeature(),
                    "hand_right_depth": BatchCameraDataEncodedFeature(),
                }
            )
        return hg_datasets.Features(features)

    def collect_all_metas(self) -> list[dict[str, Any]]:
        input_dir = Path(self.cfg.input_dir) / self.cfg.task_name
        data_dir = input_dir / "data" / "chunk-000"
        video_base = input_dir / "videos" / "chunk-000"
        info_path = input_dir / "meta" / "info.json"
        tasks_path = input_dir / "meta" / "tasks.jsonl"

        with open(info_path, encoding="utf-8") as file:
            info = json.load(file)
        high_level_instructions = info.get("high_level_instruction", {})
        instruction_segments = info.get("instruction_segments", {})
        state_field_descs = info["features"]["observation.state"][
            "field_descriptions"
        ]
        action_field_descs = info["features"]["action"]["field_descriptions"]
        features_info = {
            key: value
            for key, value in info["features"].items()
            if isinstance(value, dict) and "video_info" in value
        }

        instruction_name = ""
        with open(tasks_path, encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                if record.get("task_index") == 0:
                    instruction_name = record["task"]
                    break

        has_depth = any(
            (video_base / f"observation.images.{depth_dir}").exists()
            for depth_dir in DEPTH_CAMERA_MAP.values()
        )
        if has_depth:
            logger.info("Depth cameras detected, will pack depth data.")

        metas = []
        for parquet_file in sorted(data_dir.glob("*.parquet")):
            ep_name = parquet_file.stem
            ep_idx = int(ep_name.replace("episode_", ""))
            if self._episode_has_missing_video(video_base, ep_name, has_depth):
                continue

            ep_idx_str = str(ep_idx)
            metas.append(
                {
                    "ep_name": ep_name,
                    "ep_idx": ep_idx,
                    "parquet_path": str(parquet_file),
                    "video_base": str(video_base),
                    "has_depth": has_depth,
                    "instruction_name": instruction_name,
                    "high_level_instruction": high_level_instructions.get(
                        ep_idx_str, {}
                    ).get("high_level_instruction", ""),
                    "instruction_segments": instruction_segments.get(
                        ep_idx_str, []
                    ),
                    "state_field_descs": state_field_descs,
                    "action_field_descs": action_field_descs,
                    "features_info": features_info,
                }
            )

        return sorted(metas, key=lambda meta: meta["ep_idx"])

    def _episode_has_missing_video(
        self, video_base: Path, ep_name: str, has_depth: bool
    ) -> bool:
        for cam_name in CAMERA_NAMES:
            color_path = (
                video_base
                / f"observation.images.{cam_name}"
                / f"{ep_name}.mp4"
            )
            if not color_path.exists():
                logger.warning("Video not found: %s", color_path)
                return True
            if has_depth:
                depth_dir = DEPTH_CAMERA_MAP[cam_name]
                depth_path = (
                    video_base
                    / f"observation.images.{depth_dir}"
                    / f"{ep_name}.mp4"
                )
                if not depth_path.exists():
                    logger.warning("Depth video not found: %s", depth_path)
                    return True
        return False

    def build_episode(self, meta: dict[str, Any]) -> GenieSim3EpisodePackaging:
        episode_meta = dict(meta)
        episode_meta["uuid"] = (
            f"{self.cfg.dataset_name}/{self.cfg.task_name}/"
            f"{episode_meta['ep_name']}"
        )
        return GenieSim3EpisodePackaging(
            meta=episode_meta,
            urdf_content=self.urdf_content,
            cfg=self.cfg,
        )

    def pack(self) -> None:
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        job_name = f"part-{self.cfg.job_idx:05d}-of-{self.cfg.num_jobs:05d}"
        output_path = output_dir / job_name

        metas = self.collect_all_metas()
        selected_slice = _select_job_slice(
            total=len(metas),
            num_jobs=self.cfg.num_jobs,
            job_idx=self.cfg.job_idx,
        )
        selected_metas = metas[selected_slice]
        if not selected_metas:
            logger.warning(
                "Job %d/%d has no episodes to pack.",
                self.cfg.job_idx,
                self.cfg.num_jobs,
            )
            return

        logger.info(
            "Packing %d/%d GenieSim3 episodes to %s.",
            len(selected_metas),
            len(metas),
            output_path,
        )
        episodes = [self.build_episode(meta) for meta in selected_metas]
        has_depth = any(bool(meta.get("has_depth", False)) for meta in metas)
        features = self.get_dataset_features(has_depth)
        DatasetPackaging(features).packaging(
            episodes=episodes,
            dataset_path=str(output_path),
            max_shard_size=self.cfg.max_shard_size,
            force_overwrite=self.cfg.force_overwrite,
            writer_batch_size=self.cfg.writer_batch_size,
        )


def make_dataset_from_geniesim3(cfg: GenieSim3PackConfig) -> None:
    """Pack GenieSim3 challenge data into RO Arrow dataset shards."""
    GenieSim3RODataPacker(cfg).pack()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pack GenieSim3 challenge data into RO Arrow format."
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="GenieSim3")
    parser.add_argument("--robot_name", type=str, default="G2_omnipicker")
    parser.add_argument("--urdf_path", type=str, required=True)
    parser.add_argument("--force_overwrite", action="store_true")
    parser.add_argument("--cached_meta_path", type=str, default="")
    parser.add_argument("--max_shard_size", type=str, default="8GB")
    parser.add_argument("--writer_batch_size", type=int, default=2000)
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--job_idx", type=int, default=0)
    parser.add_argument(
        "--skip_static_frames",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--static_threshold", type=float, default=1e-6)
    return parser


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(lineno)d %(message)s",
        level=logging.INFO,
    )
    args = _build_arg_parser().parse_args()
    make_dataset_from_geniesim3(
        GenieSim3PackConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            task_name=args.task_name,
            dataset_name=args.dataset_name,
            robot_name=args.robot_name,
            urdf_path=args.urdf_path,
            force_overwrite=args.force_overwrite,
            cached_meta_path=args.cached_meta_path,
            max_shard_size=args.max_shard_size,
            writer_batch_size=args.writer_batch_size,
            num_jobs=args.num_jobs,
            job_idx=args.job_idx,
            skip_static_frames=args.skip_static_frames,
            static_threshold=args.static_threshold,
        )
    )


if __name__ == "__main__":
    main()
