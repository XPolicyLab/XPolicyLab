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

from robo_orchard_lab.dataset.experimental.mcap.batch_encoder import (
    McapBatchEncoderConfig,
    McapBatchFromBatchCameraDataEncodedConfig,
    McapBatchFromBatchFrameTransformConfig,
    McapBatchFromBatchFrameTransformGraphConfig,
    McapBatchFromBatchJointStateConfig,
)


def default_dataset_to_mcap_config() -> dict[str, McapBatchEncoderConfig]:
    """Get the default dataset to MCAP encoder config for Libero dataset."""
    config: dict[str, McapBatchEncoderConfig] = {
        "joints": McapBatchFromBatchJointStateConfig(
            target_topic="/observation/robot_state/joints",
        ),
    }
    config["action_goal_eef"] = McapBatchFromBatchFrameTransformConfig(
        target_topic="/action/goal_eef",
    )

    for camera_name in [
        "agentview",
        "robot0_eye_in_hand",
    ]:
        config[f"{camera_name}_image"] = (
            McapBatchFromBatchCameraDataEncodedConfig(
                calib_topic=f"/observation/cameras/{camera_name}/calib",
                image_topic=f"/observation/cameras/{camera_name}/image",
                tf_topic=f"/observation/cameras/{camera_name}/tf",
            )
        )
        # since depth and rgb using the same camera info, we only need to
        # add depth topic here
        config[f"{camera_name}_depth"] = (
            McapBatchFromBatchCameraDataEncodedConfig(
                image_topic=f"/observation/cameras/{camera_name}/depth",
            )
        )
    config["tf_world"] = McapBatchFromBatchFrameTransformGraphConfig(
        target_topic="/tf",
    )
    return config
