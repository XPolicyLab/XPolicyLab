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

from typing import Any, Union

import gymnasium as gym
import numpy as np
from robo_orchard_core.policy.base import ACTType, OBSType
from robo_orchard_core.utils.config import ClassType_co, ConfigInstanceOf

from robo_orchard_lab.envs.robotwin.env import RoboTwinEnvStepReturn
from robo_orchard_lab.models.holobrain.pipeline import (
    HoloBrainInferencePipeline,
    HoloBrainInferencePipelineCfg,
)
from robo_orchard_lab.models.holobrain.processor import (
    MultiArmManipulationInput,
)
from robo_orchard_lab.policy.base import (
    InferencePipelinePolicy,
    InferencePipelinePolicyCfg,
)


class HoloBrainRoboTwinPolicy(InferencePipelinePolicy):
    cfg: "HoloBrainRoboTwinPolicyCfg"
    pipeline: HoloBrainInferencePipeline

    def __init__(
        self,
        cfg: "HoloBrainRoboTwinPolicyCfg",
        observation_space: gym.Space[OBSType] | None = None,
        action_space: gym.Space[ACTType] | None = None,
        pipeline: HoloBrainInferencePipeline | None = None,
    ):
        if (
            pipeline is None
            and cfg.model_dir is not None
            and cfg.pipeline_cfg is not None
        ):
            raise ValueError(
                "Only one of `model_dir` or `pipeline_cfg` can be provided "
                "when pipeline is not passed explicitly."
            )
        if pipeline is None and cfg.model_dir is not None:
            pipeline = HoloBrainInferencePipeline.load_pipeline(
                directory=cfg.model_dir,
                inference_prefix=cfg.inference_prefix,
                device="cpu",
                load_weights=cfg.load_weights,
                load_impl=cfg.load_impl,
                model_prefix=cfg.model_prefix,
            )
        super().__init__(
            cfg,
            observation_space=observation_space,
            action_space=action_space,
            pipeline=pipeline,
        )
        self.pipeline.model.eval()
        self._cached_action: list[np.ndarray] | None = None

    @staticmethod
    def _to_homogeneous_matrix(
        matrix: np.ndarray,
        valid_shapes: tuple[tuple[int, int], ...],
        name: str,
    ) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape == (4, 4):
            return matrix.copy()
        if matrix.shape in valid_shapes:
            padded = np.eye(4, dtype=np.float64)
            padded[: matrix.shape[0], : matrix.shape[1]] = matrix
            return padded
        raise ValueError(
            f"Unsupported RoboTwin raw observation: `{name}` shape "
            f"{matrix.shape} is not supported."
        )

    def _process_raw_obs(self, obs: dict) -> MultiArmManipulationInput:
        required_keys = {"observation", "joint_action", "instructions"}
        missing_keys = required_keys - set(obs.keys())
        if missing_keys:
            raise ValueError(
                "Unsupported RoboTwin raw observation: missing keys "
                f"{sorted(missing_keys)}."
            )

        images = {}
        depths = {}
        t_world2cam = {}
        intrinsic = {}
        for cam_name, camera_data in obs["observation"].items():
            images[cam_name] = [camera_data["rgb"]]
            depths[cam_name] = [camera_data["depth"] / 1000]

            t_world2cam[cam_name] = self._to_homogeneous_matrix(
                camera_data["extrinsic_cv"],
                valid_shapes=((3, 4), (4, 4)),
                name="observation[*].extrinsic_cv",
            )

            intrinsic[cam_name] = self._to_homogeneous_matrix(
                camera_data["intrinsic_cv"],
                valid_shapes=((3, 3),),
                name="observation[*].intrinsic_cv",
            )

        joint_action = obs["joint_action"]
        if not isinstance(joint_action, dict) or "vector" not in joint_action:
            raise ValueError(
                "Unsupported RoboTwin raw observation: "
                '`joint_action["vector"]` is required.'
            )

        instruction = obs["instructions"]
        if instruction is not None and not isinstance(instruction, str):
            raise ValueError(
                "Unsupported RoboTwin raw observation: "
                "`instructions` must be a string or None."
            )

        return MultiArmManipulationInput(
            image=images,
            depth=depths,
            intrinsic=intrinsic,
            t_world2cam=t_world2cam,
            history_joint_state=[joint_action["vector"]],
            instruction=instruction,
        )

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if (
            hasattr(value, "detach")
            and hasattr(value, "cpu")
            and hasattr(value, "numpy")
        ):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @classmethod
    def _camera_pose_to_world2cam(cls, pose: Any) -> np.ndarray:
        if pose is None:
            raise ValueError(
                "Unsupported formatted RoboTwin observation: "
                "camera pose is missing."
            )

        if hasattr(pose, "inverse"):
            pose = pose.inverse()
        else:
            raise ValueError(
                "Unsupported formatted RoboTwin observation: camera pose "
                "does not support inverse()."
            )

        if hasattr(pose, "as_Transform3D_M"):
            pose = pose.as_Transform3D_M()

        if hasattr(pose, "get_matrix"):
            matrix = pose.get_matrix()
        elif hasattr(pose, "matrix"):
            matrix = pose.matrix
        else:
            raise ValueError(
                "Unsupported formatted RoboTwin observation: camera pose "
                "cannot be converted to matrix."
            )

        matrix_np = cls._to_numpy(matrix)
        if matrix_np.ndim == 3:
            matrix_np = matrix_np[0]
        return np.asarray(matrix_np, dtype=np.float64)

    def _process_formatted_obs(self, obs: dict) -> MultiArmManipulationInput:
        required_keys = {"cameras", "joints", "instructions"}
        missing_keys = required_keys - set(obs.keys())
        if missing_keys:
            raise ValueError(
                "Unsupported formatted RoboTwin observation: missing keys "
                f"{sorted(missing_keys)}."
            )

        cameras = obs["cameras"]
        joints = obs["joints"]
        instruction = obs["instructions"]
        if not isinstance(cameras, dict):
            raise ValueError(
                "Unsupported formatted RoboTwin observation: `cameras` "
                "must be a dict."
            )
        if getattr(joints, "position", None) is None:
            raise ValueError(
                "Unsupported formatted RoboTwin observation: "
                "`joints.position` is required."
            )
        if instruction is not None and not isinstance(instruction, str):
            raise ValueError(
                "Unsupported formatted RoboTwin observation: "
                "`instructions` must be a string or None."
            )

        images = {}
        depths = {}
        t_world2cam = {}
        intrinsic = {}
        for cam_name, camera_modalities in cameras.items():
            rgb = camera_modalities.get("rgb")
            depth = camera_modalities.get("depth")
            cam_data = rgb if rgb is not None else depth
            if cam_data is None:
                raise ValueError(
                    "Unsupported formatted RoboTwin observation: "
                    f"camera `{cam_name}` is missing both rgb and depth."
                )

            t_world2cam[cam_name] = self._camera_pose_to_world2cam(
                getattr(cam_data, "pose", None)
            )
            intrinsic_matrix = self._to_numpy(cam_data.intrinsic_matrices)
            if intrinsic_matrix.ndim == 3:
                intrinsic_matrix = intrinsic_matrix[0]
            intrinsic[cam_name] = self._to_homogeneous_matrix(
                intrinsic_matrix,
                valid_shapes=((3, 3), (4, 4)),
                name="cameras[*].intrinsic_matrices",
            )

            if rgb is not None:
                image = self._to_numpy(rgb.sensor_data)
                if image.ndim >= 4:
                    image = image[0]
                images[cam_name] = [image]

            if depth is not None:
                depth_data = self._to_numpy(depth.sensor_data).astype(
                    np.float32
                )
                if depth_data.ndim >= 3:
                    depth_data = depth_data[0]
                depths[cam_name] = [depth_data / 1000.0]

        joint_state = self._to_numpy(joints.position).astype(np.float32)
        if joint_state.ndim == 2:
            joint_state = joint_state[-1]

        return MultiArmManipulationInput(
            image=images or None,
            depth=depths or None,
            intrinsic=intrinsic,
            t_world2cam=t_world2cam,
            history_joint_state=[joint_state],
            instruction=instruction,
        )

    def _process_obs(self, obs: dict) -> MultiArmManipulationInput:
        if "observation" in obs and "joint_action" in obs:
            return self._process_raw_obs(obs)
        if "cameras" in obs and "joints" in obs:
            return self._process_formatted_obs(obs)
        raise ValueError(
            "Unsupported RoboTwin observation: expected raw keys "
            "`observation`/`joint_action` or formatted keys "
            "`cameras`/`joints`."
        )

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._cached_action = None

    def act(self, obs: Union[RoboTwinEnvStepReturn, dict]):
        """Get one action for the current RoboTwin observation."""
        if self._cached_action:
            return self._cached_action.pop(0)

        raw_obs = (
            obs.observations if isinstance(obs, RoboTwinEnvStepReturn) else obs
        )
        if raw_obs is None:
            raise ValueError("Observation is None.")

        data = self._process_obs(raw_obs)
        output = self.pipeline(data)
        action_chunk = output.action.cpu().numpy()  # type: ignore

        max_chunk = min(len(action_chunk), self.cfg.use_action_chunk_size)
        if max_chunk <= 0:
            raise ValueError("Pipeline returned an empty action chunk.")

        self._cached_action = [action_chunk[i] for i in range(max_chunk)]
        return self._cached_action.pop(0)


class HoloBrainRoboTwinPolicyCfg(InferencePipelinePolicyCfg):
    class_type: ClassType_co[HoloBrainRoboTwinPolicy] = HoloBrainRoboTwinPolicy
    """Class type for the policy."""

    pipeline_cfg: ConfigInstanceOf[HoloBrainInferencePipelineCfg] | None = None
    """Configuration for the inference pipeline."""

    inference_prefix: str = "inference"
    """Prefix of the saved inference pipeline config files."""

    model_dir: str | None = None
    """Directory of the exported inference pipeline for lazy loading."""

    load_weights: bool = True
    """Whether to load model weights when building from model_dir."""

    load_impl: str = "native"
    """Implementation used to load model weights."""

    model_prefix: str = "model"
    """Prefix of the exported model files in model_dir."""

    use_action_chunk_size: int = 32
    """The number of predicted actions to cache per pipeline call."""
