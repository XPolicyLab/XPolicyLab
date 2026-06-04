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
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from robo_orchard_core.utils.config import load_config_class

from robo_orchard_lab.processing.io_processor.base import (
    ClassType_co,
    ModelIOProcessor,
    ModelIOProcessorCfg,
)
from robo_orchard_lab.utils.build import DelayInitDictType, build
from robo_orchard_lab.utils.path import in_cwd

__all__ = ["HoloBrainProcessor", "HoloBrainProcessorCfg"]

TENSOR_TYPE = np.ndarray | torch.Tensor


@dataclass
class MultiArmManipulationInput:
    """Data structure for inputs to a multi-arm manipulation pipeline.

    This class defines the expected inputs for a robotics policy that may use
    various modalities like vision, proprioception, and language instructions.
    """

    image: dict[str, list[TENSOR_TYPE]] | None = None
    """A dictionary mapping camera names to lists of RGB images.
    Allows for multiple viewpoints."""

    depth: dict[str, list[TENSOR_TYPE]] | None = None
    """A dictionary mapping camera names to lists of depth maps."""

    intrinsic: dict[str, TENSOR_TYPE] | None = None
    """A dictionary mapping camera names to their intrinsic
    parameter matrices."""

    t_world2cam: dict[str, TENSOR_TYPE] | None = None
    """A dictionary mapping camera names to their world-to-camera
    transformation matrices (e.g., extrinsic parameters)."""

    t_robot2world: TENSOR_TYPE | None = None
    """The transformation from the robot's base frame to the
    world coordinate frame."""

    t_robot2ego: TENSOR_TYPE | None = None
    """The transformation from the robot's base frame to an
    egocentric frame, if applicable."""

    history_joint_state: list[TENSOR_TYPE] | None = None
    """A list of past joint states, representing the
    robot's proprioceptive history."""

    history_ee_pose: list[TENSOR_TYPE] | None = None
    """A list of past end-effector poses."""

    instruction: str | None = None
    """A natural language command or goal for the task."""

    urdf: str | None = None
    """The URDF (Unified Robot Description Format) of the robot as a
    string, describing its kinematic and dynamic properties."""

    remaining_actions: TENSOR_TYPE | None = None
    """The remaining actions from last pred, if rtc is used."""

    delay_horizon: int | None = None
    """The number of time steps to delay action execution, if rtc is used."""


@dataclass
class MultiArmManipulationOutput:
    """Data structure for outputs from a multi-arm manipulation pipeline.

    This class encapsulates the results produced by the inference pipeline,
    primarily the predicted action for the robot.
    """

    action: TENSOR_TYPE
    """The predicted action tensor. This could represent target joint
    positions, end-effector velocities, or another action space format."""


class Struct2Dict:
    def __init__(
        self,
        load_image: bool,
        load_depth: bool,
        cam_names: Optional[List[str]] = None,
    ):
        self.load_image = load_image
        self.load_depth = load_depth
        self.cam_names = cam_names

    def __call__(self, data: MultiArmManipulationInput) -> dict:
        input_data = dict()

        if self.cam_names is None:
            cam_names = list(data.intrinsic.keys())
        else:
            cam_names = self.cam_names

        assert data.intrinsic is not None
        input_data["intrinsic"] = np.stack(
            [data.intrinsic[x] for x in cam_names]
        )

        if data.t_world2cam is not None:
            input_data["T_world2cam"] = np.stack(
                [data.t_world2cam[x] for x in cam_names]
            )

        assert data.history_joint_state is not None
        input_data["joint_state"] = np.stack(data.history_joint_state)
        input_data["step_index"] = len(data.history_joint_state) - 1

        if self.load_image:
            assert data.image is not None
            input_data["imgs"] = [data.image[x][-1] for x in cam_names]

        if self.load_depth:
            assert data.depth is not None
            input_data["depths"] = [data.depth[x][-1] for x in cam_names]

        input_data["text"] = (
            "" if data.instruction is None else data.instruction
        )
        input_data["cam_names"] = cam_names

        if data.remaining_actions is not None:
            if data.delay_horizon is None:
                raise ValueError(
                    "delay_horizon must be provided when remaining_actions is given."  # noqa: E501
                )
            input_data["remaining_actions"] = data.remaining_actions
            input_data["delay_horizon"] = data.delay_horizon

        return input_data


class HoloBrainProcessor(ModelIOProcessor):
    cfg: "HoloBrainProcessorCfg"  # for type hint

    def __init__(self, cfg: "HoloBrainProcessorCfg"):
        super().__init__(cfg)
        self.struction_to_dict = Struct2Dict(
            load_image=self.cfg.load_image,
            load_depth=self.cfg.load_depth,
            cam_names=self.cfg.cam_names,
        )
        self.transforms = (
            [build(transform) for transform in self.cfg.transforms]
            if self.cfg.transforms is not None
            else []
        )

    def pre_process(self, data: Union[MultiArmManipulationInput, Dict]):
        if isinstance(data, MultiArmManipulationInput):
            data = self.struction_to_dict(data)
        for ts_i in self.transforms:
            data = ts_i(data)
        return data

    def post_process(self, model_outputs, batch) -> MultiArmManipulationOutput:
        # only output one trajectory in joint angle format
        # action shape: num_pred_steps x num_joint
        action = model_outputs[0]["pred_actions"][0][..., 0]
        if self.cfg.valid_action_step is not None:
            action = action[: self.cfg.valid_action_step]
        return MultiArmManipulationOutput(action=action)

    def save(self, path, processor_name, urdf_dir="./urdf"):
        os.makedirs(path, exist_ok=True)
        cfg = copy.deepcopy(self.cfg)
        urdfs = []
        for transform in cfg.transforms:
            if "urdf" in transform:
                urdf_file = transform["urdf"]
                urdfs.append(urdf_file)
                transform["urdf"] = os.path.join(
                    urdf_dir, os.path.basename(urdf_file)
                )

        if len(urdfs) > 0:
            if not os.path.isabs(path):
                target_urdf_path = os.path.join(path, urdf_dir)
            else:
                target_urdf_path = urdf_dir
            os.makedirs(target_urdf_path, exist_ok=True)
            for urdf in urdfs:
                try:
                    shutil.copy2(urdf, target_urdf_path)
                except shutil.SameFileError:
                    pass

        with open(os.path.join(path, processor_name), "w") as fh:
            fh.write(cfg.model_dump_json(indent=4))

    @staticmethod
    def load(path, processor_name):
        processor_cfg = load_config_class(
            open(os.path.join(path, processor_name)).read()
        )
        with in_cwd(path):
            processor = processor_cfg()
        return processor


class HoloBrainProcessorCfg(ModelIOProcessorCfg[HoloBrainProcessor]):
    class_type: ClassType_co[HoloBrainProcessor] = HoloBrainProcessor
    load_image: bool = True
    load_depth: bool = True
    cam_names: Optional[List[str]] = None
    valid_action_step: Optional[int] = None
    transforms: Optional[List[DelayInitDictType]] = None
