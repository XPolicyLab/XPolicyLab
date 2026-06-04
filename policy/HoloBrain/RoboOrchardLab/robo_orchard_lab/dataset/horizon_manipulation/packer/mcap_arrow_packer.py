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

import argparse
import glob
import hashlib
import json
import logging
import os
from typing import Generator

import datasets as hg_datasets
import torch
from robo_orchard_core.datatypes.tf_graph import BatchFrameTransformGraph
from robo_orchard_core.kinematics.chain import KinematicChain
from robo_orchard_core.utils.logging import LoggerManager
from robo_orchard_core.utils.math import Transform3D_M

from robo_orchard_lab.dataset.datatypes import (
    BatchCameraDataEncodedFeature,
    BatchFrameTransform,
    BatchJointsState,
    BatchJointsStateFeature,
)
from robo_orchard_lab.dataset.horizon_manipulation.packer.utils import (
    McapParseConfig,
    PackConfig,
    apply_camera_calibration_overrides,
    filter_static_frames,
    get_index_camera,
    parse_mcap,
    time_sync,
    update_camera_poses_from_tf_graph,
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

# Setup Logger
logger = LoggerManager().get_child(__name__)
logger.setLevel(logging.INFO)

# ========= Arrow Dataset Feature Definition =========
ArrowDatasetFeatures = hg_datasets.Features(
    {
        "joints": BatchJointsStateFeature(dtype="float32"),
        "actions": BatchJointsStateFeature(dtype="float32"),
        "middle": BatchCameraDataEncodedFeature(dtype="float32"),
        "middle_depth": BatchCameraDataEncodedFeature(dtype="float32"),
        "left": BatchCameraDataEncodedFeature(dtype="float32"),
        "left_depth": BatchCameraDataEncodedFeature(dtype="float32"),
        "right": BatchCameraDataEncodedFeature(dtype="float32"),
        "right_depth": BatchCameraDataEncodedFeature(dtype="float32"),
    }
)


# ========= Core Packaging Class =========
class McapEpisodePackaging(EpisodePackaging):
    """Processes single robotics episode from an MCAP file for Arrow packaging.

    This class handles the entire pipeline for a single episode: loading data
    from an MCAP file, synchronizing multimodal data streams (images, joints),
    filtering unwanted frames, and preparing the data for generation of a
    Hugging Face dataset in Arrow format.

    The synchronization strategy uses a primary camera's timestamps as the
    structural base, ensuring that each frame in the output dataset corresponds
    to a consistent point in time. Other data streams are aligned to this base
    clock by selecting the nearest temporal neighbors.
    """

    def __init__(
        self,
        episode_path: str,
        user: str,
        task_name: str,
        date: str,
        urdf_path: str,
        pack_config: PackConfig,
    ):
        """Initializes the McapEpisodePackaging instance.

        Args:
            episode_path: The path to the episode directory.
            user: The user associated with the episode.
            task_name: The name of the task performed.
            date: The recording date of the episode.
            urdf_path: A string containing the robot's URDF.
            pack_config: Configuration parameters for packaging.
        """
        self.episode_path = episode_path
        self.user = user
        self.task_name = task_name
        self.date = date
        self.pack_config = pack_config
        self.urdf_path = urdf_path

        mcap_path = glob.glob(os.path.join(self.episode_path, "*.mcap"))[0]
        self.mcap_path = mcap_path
        self.uuid = hashlib.md5(self.mcap_path.encode("utf-8")).hexdigest()

    def forward_kinematics(self, joint_states: BatchJointsState):
        """Computes the forward kinematics for the dual-arm robot.

        Args:
            joint_states (BatchJointsState): A batch of combined joint states
                for both arms, typically shaped (N, 14) for 7 DOF per arm.

        Returns:
            dict: end-effector poses (BatchFrameTransform) for both arms.
        """
        # Load URDF and Initialize Kinematic Chain
        chain = KinematicChain.from_content(
            open(self.urdf_path, "r").read().encode("utf-8"), "urdf"
        )

        position = joint_states.position

        position_urdf = torch.zeros((joint_states.batch_size, 16))
        position_urdf[:, :6] = position[:, :6]
        position_urdf[:, 8:14] = position[:, 7:13]
        position_urdf[:, 6] = position[:, 6] / 2
        position_urdf[:, 7] = -position[:, 6] / 2
        position_urdf[:, 14] = position[:, 13] / 2
        position_urdf[:, 15] = -position[:, 13] / 2

        link_poses = chain.forward_kinematics(position_urdf)
        left_ee_pose: Transform3D_M = link_poses["left_gripper_base"]
        right_ee_pose: Transform3D_M = link_poses["right_gripper_base"]

        batch_left_ee_pose = BatchFrameTransform(
            parent_frame_id="world",
            child_frame_id="left_end_effector",
            xyz=left_ee_pose.get_translation(),
            quat=left_ee_pose.get_rotation_quaternion(),
        )
        batch_right_ee_pose = BatchFrameTransform(
            parent_frame_id="world",
            child_frame_id="right_end_effector",
            xyz=right_ee_pose.get_translation(),
            quat=right_ee_pose.get_rotation_quaternion(),
        )

        return batch_left_ee_pose, batch_right_ee_pose

    def generate_episode_meta(self) -> EpisodeMeta:
        """Generates the metadata for the current episode.

        This method compiles the necessary metadata objects that describe the
        episode, the robot used, and the task performed, which will be stored
        alongside the Arrow dataset.

        Returns:
            EpisodeMeta: An object containing the structured metadata for the
                episode.
        """

        # Init Metadata
        meta_from_file = json.load(
            open(os.path.join(self.episode_path, "episode_meta.json"), "r")
        )
        self.meta_data = meta_from_file

        episode_data = EpisodeData(
            info={
                "uuid": self.uuid,
                "date": self.date,
            }
        )
        robot = RobotData(
            name="piper",
            content=open(self.urdf_path, "r").read(),
            content_format=RobotDescriptionFormat.URDF,
        )
        task = TaskData(
            name=self.meta_data["task_name"],
            description=self.meta_data["instruction"],
        )

        return EpisodeMeta(
            episode=episode_data,
            robot=robot,
            task=task,
        )

    def process_data(self):
        pack_config: PackConfig = self.pack_config
        parse_config: McapParseConfig = pack_config.PARSE_CONFIG

        # Parse MCAP
        mcap_data = parse_mcap(
            mcap_path=self.mcap_path, parse_config=parse_config
        )
        batch_tf_list, batch_joint_dict, batch_image_dict, batch_depth_dict = (
            mcap_data
        )

        # Time Synchronization
        base_time = batch_image_dict[pack_config.SYNC_CAMERA].timestamps
        time_sync(
            data=[batch_joint_dict, batch_image_dict, batch_depth_dict],
            base_time=base_time,
        )

        # Filter static frames
        slave_left_pos = batch_joint_dict[
            parse_config.SLAVE_LEFT_JOINT
        ].position
        slave_right_pos = batch_joint_dict[
            parse_config.SLAVE_RIGHT_JOINT
        ].position
        joint_positions = torch.cat([slave_left_pos, slave_right_pos], dim=-1)
        filter_static_frames(
            data=[batch_joint_dict, batch_image_dict, batch_depth_dict],
            joint_positions=joint_positions,
            base_time=base_time,
            static_threshold=pack_config.STATIC_THRESHOLD,
            head_time_to_filter=pack_config.HEAD_TIME_TO_FILTER,
            tail_time_to_filter=pack_config.TAIL_TIME_TO_FILTER,
        )

        # Apply resize and topic-based extrinsic overrides before the TF
        # graph is materialized for downstream pose queries.
        batch_tf_list = apply_camera_calibration_overrides(
            batch_tf_list=batch_tf_list,
            image_data=[batch_image_dict, batch_depth_dict],
            image_scale_ratio=pack_config.IMAGE_SCALE,
            extrinsic_overrides=pack_config.EXTRINSIC_OVERRIDES,
        )

        self.tf_graph = BatchFrameTransformGraph(batch_tf_list)
        self.batch_joint_dict = batch_joint_dict
        self.batch_image_dict = batch_image_dict
        self.batch_depth_dict = batch_depth_dict

        self.base_time = batch_image_dict[pack_config.SYNC_CAMERA].timestamps
        self.num_steps = len(self.base_time)

    def generate_frames(self) -> Generator[DataFrame, None, None]:
        """Generates structured DataFrame for each frame in the episode.

        This generator function iterates through each processed time step of
        the episode and yields a `DataFrame` object. Each `DataFrame` contains
        all the data for that specific moment, including joint states (robot),
        actions (teleop controller), multiple camera views, and instructional
        data.

        Yields:
            DataFrame: An object representing all data for a single,
                synchronized frame of the episode.
        """
        logger.info(f"Start processing episode: {self.uuid}")
        self.process_data()

        if self.num_steps == 0:
            logger.warning(
                f"Episode {self.uuid} has 0 steps after processing."
            )
            return

        instruction = InstructionData(
            name=self.task_name,
            json_content={
                "name": self.meta_data["task_name"],
                "description": self.meta_data["instruction"],
            },
        )

        parse_config: McapParseConfig = self.pack_config.PARSE_CONFIG
        # --- Joints (Slave Arm) ---
        slave_left = self.batch_joint_dict[parse_config.SLAVE_LEFT_JOINT]
        slave_right = self.batch_joint_dict[parse_config.SLAVE_RIGHT_JOINT]
        slave_right.timestamps = slave_left.timestamps
        joint_states = BatchJointsState.concat(
            [slave_left, slave_right], dim=1
        )

        # --- Actions (Master Arm) ---
        master_left = self.batch_joint_dict[parse_config.MASTER_LEFT_JOINT]
        master_right = self.batch_joint_dict[parse_config.MASTER_RIGHT_JOINT]
        master_right.timestamps = master_left.timestamps
        action_states = BatchJointsState.concat(
            [master_left, master_right], dim=1
        )

        # --- Camera Data ---
        batch_left_ee_pose, batch_right_ee_pose = self.forward_kinematics(
            joint_states
        )
        self.tf_graph.add_tf(batch_left_ee_pose)
        self.tf_graph.add_tf(batch_right_ee_pose)

        # Refresh both color and depth poses from the same TF graph so the
        # exported camera metadata stays aligned.
        update_camera_poses_from_tf_graph(
            tf_graph=self.tf_graph,
            camera_dict=self.batch_image_dict,
            num_steps=self.num_steps,
        )
        update_camera_poses_from_tf_graph(
            tf_graph=self.tf_graph,
            camera_dict=self.batch_depth_dict,
            num_steps=self.num_steps,
        )

        for i in range(self.num_steps):
            features = {"joints": joint_states[i], "actions": action_states[i]}
            for cam_idx, cam_name in enumerate(parse_config.CAMERAS):
                color_topic = parse_config.COLOR_IMAGE_TOPICS[cam_idx]
                depth_topic = parse_config.DEPTH_IMAGE_TOPICS[cam_idx]
                features[cam_name] = get_index_camera(
                    self.batch_image_dict[color_topic], i
                )
                features[f"{cam_name}_depth"] = get_index_camera(
                    self.batch_depth_dict[depth_topic], i
                )

            frame_ts_ns = self.base_time[i]
            yield DataFrame(
                features=features,
                instruction=instruction,
                timestamp_ns_max=frame_ts_ns,
                timestamp_ns_min=frame_ts_ns,
            )

        # Clean up large data to prevent memory leak
        del self.batch_joint_dict
        del self.batch_image_dict
        del self.batch_depth_dict
        del self.tf_graph
        self.batch_joint_dict = None
        self.batch_image_dict = None
        self.batch_depth_dict = None
        self.tf_graph = None


def make_dataset_from_mcap(
    input_path: str,
    output_path: str,
    urdf_path: str,
    pack_config: PackConfig,
    max_shard_size: str | int = "2GB",
    split: hg_datasets.Split | None = None,
    force_overwrite: bool = False,
):
    """Orchestrates conversion of multiple MCAP episodes into an Arrow dataset.

    This function scans the input path for episode directories, initializes a
    `McapEpisodePackaging` instance for each valid episode, and then uses the
    `DatasetPackaging` utility to write all episodes into a sharded Arrow
    dataset compatible with Hugging Face datasets.

    Args:
        input_path (str): A comma-separated string of glob patterns that match
            episode directories.
        output_path (str): The destination path for the output Arrow dataset.
        urdf_path (str): The path to the robot's URDF file.
        pack_config (PackConfig): Configuration parameters for packaging,
        max_shard_size (str | int): The maximum size for each Arrow file shard.
            Defaults to "2GB".
        split (hg_datasets.Split | None): The dataset split to assign (e.g.,
            'train', 'test'). Defaults to None, which the packager typically
            treats as 'train'.
        force_overwrite (bool): If True, the destination directory will be
            overwritten if it already exists. Defaults to False.
    """

    episodes_meta = []
    input_path_list = input_path.strip().split(",")
    logger.info(f"input_paths: {input_path_list}")

    for path_pattern in input_path_list:
        input_files = glob.glob(path_pattern)
        input_files.sort()
        for input_file in input_files:
            episode_path = os.path.dirname(input_file)
            date = os.path.basename(episode_path)
            meta = json.load(
                open(os.path.join(episode_path, "episode_meta.json"), "r")
            )
            user = meta["user_name"]
            task = meta["task_name"]
            episodes_meta.append([episode_path, user, task, date])

    episodes_meta.sort()
    logger.info(f"Found {len(episodes_meta)} potential episodes.")

    episodes = []
    for episode_path, user, task_name, date in episodes_meta:
        try:
            packer = McapEpisodePackaging(
                episode_path=episode_path,
                user=user,
                task_name=task_name,
                date=date,
                urdf_path=urdf_path,
                pack_config=pack_config,
            )
            episodes.append(packer)
        except Exception:
            logger.error(
                f"Failed to process episode at {episode_path}", exc_info=True
            )

    if not episodes:
        logger.error("No valid episodes found to package. Aborting.")
        return
    logger.info(f"Packaging {len(episodes)} valid episodes into Arrow format.")

    packing = DatasetPackaging(features=ArrowDatasetFeatures)
    packing.packaging(
        episodes=episodes,
        dataset_path=output_path,
        max_shard_size=max_shard_size,
        force_overwrite=force_overwrite,
        split=split,
    )
    logger.info(f"Successfully created Arrow dataset at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert MCAP files directly to Arrow format."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path pattern to MCAP episode directories, e.g., '/path/*/*/'",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output Arrow dataset.",
    )
    parser.add_argument(
        "--urdf_path", type=str, required=True, help="Path to the URDF file."
    )
    parser.add_argument(
        "--image_scale_factor",
        type=float,
        default=1.0,
        help="Factor to scale images by.",
    )
    parser.add_argument(
        "--force_overwrite",
        action="store_true",
        help="Overwrite the output directory if it exists.",
    )
    args = parser.parse_args()

    parse_config = McapParseConfig(
        CAMERAS=["middle", "left", "right"],
        SLAVE_LEFT_JOINT="/observation/robot_state/left/joint",
        SLAVE_RIGHT_JOINT="/observation/robot_state/right/joint",
        MASTER_LEFT_JOINT="/observation/robot_state/left_master/joint",
        MASTER_RIGHT_JOINT="/observation/robot_state/right_master/joint",
        COLOR_IMAGE_TOPICS=[
            "/observation/cameras/middle/color_image/image_raw",
            "/observation/cameras/left/color_image/image_raw",
            "/observation/cameras/right/color_image/image_raw",
        ],
        DEPTH_IMAGE_TOPICS=[
            "/observation/cameras/middle/depth_image/image_raw",
            "/observation/cameras/left/depth_image/image_raw",
            "/observation/cameras/right/depth_image/image_raw",
        ],
        COLOR_INFO_TOPICS=[
            "/observation/cameras/middle/color_image/camera_info",
            "/observation/cameras/left/color_image/camera_info",
            "/observation/cameras/right/color_image/camera_info",
        ],
        DEPTH_INFO_TOPICS=[
            "/observation/cameras/middle/depth_image/camera_info",
            "/observation/cameras/left/depth_image/camera_info",
            "/observation/cameras/right/depth_image/camera_info",
        ],
    )

    pack_config = PackConfig(
        SYNC_CAMERA="/observation/cameras/middle/color_image/image_raw",
        IMAGE_SCALE=args.image_scale_factor,
        STATIC_THRESHOLD=1e-3,
        HEAD_TIME_TO_FILTER=None,
        TAIL_TIME_TO_FILTER=None,
        PARSE_CONFIG=parse_config,
        EXTRINSIC_OVERRIDES=None,
    )

    make_dataset_from_mcap(
        input_path=args.input_path,
        output_path=args.output_path,
        urdf_path=args.urdf_path,
        pack_config=pack_config,
        force_overwrite=args.force_overwrite,
    )
