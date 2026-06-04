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
from typing import Any

import gymnasium as gym
import numpy as np
from robo_orchard_core.policy.base import ACTType, OBSType
from robo_orchard_core.utils.config import ClassType_co, ConfigInstanceOf

from robo_orchard_lab.dataset.agibot_geniesim.transforms import (
    GenieSim3CalibrationToExtrinsic,
)
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

GENIESIM_CAMERAS = ("hand_left", "hand_right", "top_head")
GENIESIM_ACTION_DIM = 22
GENIESIM_STATE_DIM = 24
GRIPPER_ENCODE_OFFSET = 0.0
GRIPPER_ENCODE_RANGE = 120.0

TASK_NAME_TO_HEAD_STATE = {
    "clean_the_desktop": (0.0, 0.0, 0.11464),
    "hold_pot": (0.0, 0.0, 0.11464),
    "open_door": (0.0, 0.0, 0.11464),
    "place_block_into_box": (0.0, 0.0, 0.11464),
    "pour_workpiece": (0.0, 0.0, 0.11464),
    "scoop_popcorn": (0.0, 0.0, 0.0),
    "sorting_packages": (0.0, 0.0, 0.11464),
    "sorting_packages_continuous": (0.0, 0.0, 0.11464),
    "stock_and_straighten_shelf": (0.0, 0.0, 0.11464),
    "take_wrong_item_shelf": (0.0, 0.0, 0.1745),
}

DEFAULT_CAMERA_INTRINSICS = {
    "hand_left": [
        [486.13733, 0.0, 614.31964, 0.0],
        [0.0, 485.94153, 529.99976, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    "hand_right": [
        [465.1793, 0.0, 630.648, 0.0],
        [0.0, 465.0162, 527.8828, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    "top_head": [
        [306.6911, 0.0, 319.90094, 0.0],
        [0.0, 306.55075, 201.29141, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
}

DEFAULT_CAMERA_CALIBRATION = {
    "hand_left": {
        "position": [-0.089796559162, -0.001158827139, 0.060707139061],
        "orientation": [
            0.25628239463,
            -0.25628239463,
            0.659029084489,
            -0.659029084489,
        ],
    },
    "hand_right": {
        "position": [0.0898, 0.00116, 0.060707139061],
        "orientation": [
            -0.25628239463,
            -0.25628239463,
            0.659029084489,
            0.659029084489,
        ],
    },
    "top_head": {
        "position": [0.10237, 0.02375, 0.10256],
        "orientation": [
            -0.594510412187,
            0.594544262709,
            -0.378729147283,
            0.386831646171,
        ],
    },
}

DEFAULT_TASK_NAME_TO_INSTRUCTION = {
    "hold_pot": (
        "Grasp both handles of the pot with left and right hands, Move the "
        "pot to the stove and put it down"
    ),
    "clean_the_desktop": (
        "Pick up the pen on the left side and place it into the pen holder, "
        "close the laptop, pick up the tissue on the table and place it into "
        "the trash bin on the right size. Then, pick up the mouse and place "
        "it on the right side of the laptop. Finally, straighten the colored "
        "pencil box."
    ),
    "open_door": "Turn the doorknob with the right arm, Push the door",
    "place_block_into_box": (
        "The robot is in front of the table, where 5-10 building blocks and "
        "a block box are placed. The block box has multiple holes of "
        "different shapes."
    ),
    "pour_workpiece": "Pour the workpiece into the box with the right arm.",
    "scoop_popcorn": (
        "Scoop the popcorn with the right arm and pour it into the popcorn "
        "bucket, Scoop the popcorn with the right arm and pour it into the "
        "popcorn bucket, Scoop the popcorn with the right arm and pour it "
        "into the popcorn bucket"
    ),
    "sorting_packages": (
        "Grab the package on the table with right arm black, Turn the waist "
        "right to face the barcode scanner, Place the package on the scanning "
        "table with the barcode facing up, The right arm grabs the package, "
        "Rotate the waist with the right arm, Place the package in the blue "
        "bin, Both arms coordinate and the waist returns to the initial "
        "posture"
    ),
    "sorting_packages_continuous": (
        "Grab the package on the table with right arm black, Turn the waist "
        "right to face the barcode scanner, Place the package on the scanning "
        "table with the barcode facing up, The right arm grabs the package, "
        "Rotate the waist with the right arm, Place the package in the blue "
        "bin, Both arms coordinate and the waist returns to the initial "
        "posture"
    ),
    "stock_and_straighten_shelf": (
        "Pick up the wei-chuan orange juice in the shopping basket and place "
        "it on the shelf with right arm, Straighten the overturned "
        "wei-chuan grape juice with right arm"
    ),
    "take_wrong_item_shelf": (
        "The right arm picks up the incorrectly placed item from the shelf, "
        "Place the misplaced items from the shelf into the shopping basket"
    ),
}

__all__ = [
    "HoloBrainGenieSim3Policy",
    "HoloBrainGenieSim3PolicyCfg",
]


def _as_float_array(
    value: Any, *, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {arr.shape}")
    return arr


def _to_matrix4(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr.copy()
    if arr.shape == (3, 3):
        ret = np.eye(4, dtype=np.float64)
        ret[:3, :3] = arr
        return ret
    raise ValueError(f"Unsupported matrix shape: {arr.shape}")


def _decode_image(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    return np.asarray(arr)[:, :, ::-1]


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


def _decode_depth(camera_name: str, depth: Any) -> np.ndarray:
    """Decode GenieSim payload depth back to meters."""
    arr = np.asarray(depth, dtype=np.float32)
    if camera_name == "top_head":
        arr /= 1000.0
    else:
        arr /= 10000.0
    return arr


def _decode_gripper_state(
    encoded_value: float, gripper_limit: float = 1.0
) -> float:
    """Normalize raw GenieSim3 gripper observations to training scale."""
    scaled = (
        float(encoded_value) - GRIPPER_ENCODE_OFFSET
    ) / GRIPPER_ENCODE_RANGE
    return scaled * gripper_limit


def build_joint_state_from_payload(
    payload_state: Any,
    task_name: str,
    gripper_limit: float = 1.0,
) -> np.ndarray:
    """Convert a GenieSim payload state to HoloBrain joint order.

    Payload gripper observations are raw actuator values and are divided by
    ``120.0`` to match the training-time ``joint_state`` scale.
    """
    state = np.asarray(payload_state, dtype=np.float32).reshape(-1)
    if state.shape[0] < 21:
        raise ValueError(
            "GenieSim payload state must have at least 21 dims, "
            f"got {state.shape[0]}"
        )

    if task_name not in TASK_NAME_TO_HEAD_STATE:
        raise ValueError(
            f"Unknown GenieSim3 task_name `{task_name}` for head state."
        )

    head_state = _as_float_array(
        TASK_NAME_TO_HEAD_STATE[task_name],
        shape=(3,),
    )
    joint_state = np.zeros(GENIESIM_STATE_DIM, dtype=np.float32)
    joint_state[:7] = state[:7]
    joint_state[7] = _decode_gripper_state(state[14], gripper_limit)
    joint_state[8:15] = state[7:14]
    joint_state[15] = _decode_gripper_state(state[15], gripper_limit)
    joint_state[16:19] = head_state
    joint_state[19:24] = state[16:21]
    return joint_state


def convert_actions_to_geniesim(
    actions: Any,
    valid_action_step: int,
    sampling_ratio: float = 1.0,
) -> np.ndarray:
    """Convert HoloBrain joint actions to the GenieSim action layout.

    Args:
        actions (Any): HoloBrain action array of shape ``[T, 24]``.
        valid_action_step (int): Number of output timesteps after resampling.
        sampling_ratio (float, optional): Sampling ratio applied before
            truncation. Integer ratios larger than ``1`` use stride sampling;
            other positive ratios use linear interpolation. Default is ``1``.

    Returns:
        np.ndarray: GenieSim actions of shape ``[valid_action_step, 22]``.
    """
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != GENIESIM_STATE_DIM:
        raise ValueError(
            f"Expected action array of shape [T, {GENIESIM_STATE_DIM}], "
            f"got {arr.shape}"
        )

    valid_action_step = int(valid_action_step)
    if valid_action_step <= 0:
        raise ValueError(
            f"valid_action_step must be > 0, got {valid_action_step}"
        )

    sampling_ratio = float(sampling_ratio)
    if sampling_ratio <= 0:
        raise ValueError(f"sampling_ratio must be > 0, got {sampling_ratio}")

    raw_len = arr.shape[0]
    if raw_len == 0:
        raise ValueError("actions is empty")

    if np.isclose(sampling_ratio, 1.0):
        sampled = arr
    else:
        rounded_ratio = int(round(sampling_ratio))
        if sampling_ratio > 1.0 and np.isclose(sampling_ratio, rounded_ratio):
            sampled = arr[::rounded_ratio]
        else:
            target_len = int(raw_len / sampling_ratio)
            if target_len <= 0:
                raise ValueError(
                    "No actions available after resampling with "
                    f"sampling_ratio={sampling_ratio}, input_steps={raw_len}"
                )
            if raw_len == 1:
                sampled = np.repeat(arr, target_len, axis=0)
            else:
                x_ori = np.linspace(0, raw_len - 1, num=raw_len)
                x_tgt = np.linspace(0, raw_len - 1, num=target_len)
                sampled = np.stack(
                    [
                        np.interp(x_tgt, x_ori, arr[:, dim])
                        for dim in range(arr.shape[1])
                    ],
                    axis=1,
                ).astype(np.float32)

    if sampled.shape[0] < valid_action_step:
        raise ValueError(
            "Resampled action length is shorter than valid_action_step: "
            f"len(sampled)={sampled.shape[0]}, "
            f"valid_action_step={valid_action_step}"
        )

    valid_action = sampled[:valid_action_step]
    ret = np.zeros((valid_action_step, GENIESIM_ACTION_DIM), dtype=np.float32)
    ret[:, :7] = valid_action[:, :7]
    ret[:, 7:14] = valid_action[:, 8:15]
    ret[:, 14] = valid_action[:, 7]
    ret[:, 15] = valid_action[:, 15]
    ret[:, 20] = valid_action[:, 23]

    return ret


class HoloBrainGenieSim3Policy(InferencePipelinePolicy):
    """Deployment policy for running HoloBrain on GenieSim3 payloads."""

    cfg: "HoloBrainGenieSim3PolicyCfg"
    pipeline: HoloBrainInferencePipeline

    def __init__(
        self,
        cfg: "HoloBrainGenieSim3PolicyCfg",
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
                device="cuda",
                load_weights=cfg.load_weights,
                load_impl=cfg.load_impl,
                model_prefix=cfg.model_prefix,
            )

        camera_intrinsics = (
            DEFAULT_CAMERA_INTRINSICS
            if cfg.camera_intrinsics is None
            else cfg.camera_intrinsics
        )
        camera_calibration = (
            DEFAULT_CAMERA_CALIBRATION
            if cfg.camera_calibration is None
            else cfg.camera_calibration
        )

        super().__init__(
            cfg,
            observation_space=observation_space,
            action_space=action_space,
            pipeline=pipeline,
        )
        self.pipeline.model.eval()

        self._camera_intrinsics = {
            cam: _to_matrix4(camera_intrinsics[cam])
            for cam in GENIESIM_CAMERAS
        }
        self._setup_calibration(camera_calibration)

    def _setup_calibration(
        self, camera_calibration: dict[str, Any] | None
    ) -> None:
        """Install static calibration on deploy-time processor transforms."""
        if camera_calibration is None:
            return
        processor = getattr(self.pipeline, "processor", None)
        if processor is None:
            return
        for transform in getattr(processor, "transforms", []):
            if isinstance(transform, GenieSim3CalibrationToExtrinsic):
                transform.calibration = transform.calibration_handler(
                    camera_calibration
                )
                break

    def _resolve_instruction(self, payload: dict[str, Any]) -> str:
        prompt = payload.get("prompt", "")
        task_name = payload.get("task_name", "")
        # Keep the configured canonical instruction authoritative.  The
        # GenieSim3 HoloBrain checkpoint is aligned to these task prompts;
        # payload prompts are used only for tasks without a configured entry.
        return self.cfg.task_name_to_instruction.get(task_name, prompt)

    def data_preprocess(
        self, payload: dict[str, Any]
    ) -> MultiArmManipulationInput:
        """Convert a GenieSim benchmark payload into HoloBrain inference input.

        Args:
            payload: A dict sent by the GenieSim benchmark server with the
                following keys:

                - ``"images"`` (**required**, ``dict[str, array]``): RGB image
                  arrays keyed by camera name.  Expected cameras are
                  ``"hand_left"``, ``"hand_right"``, and ``"top_head"``.
                  Each array should be HWC or CHW uint8.
                - ``"state"`` (**required**, ``array-like``): Joint state
                  vector with at least 21 elements in GenieSim order
                  (left-arm 7, right-arm 7, left-gripper, right-gripper,
                  body 5). Gripper entries are raw actuator observations and
                  are normalized with the same ``/120.0`` convention used by
                  the training dataset loader.
                - ``"task_name"`` (**required**, ``str``): Canonical task
                  name used to look up the head joint initial values and
                  the default instruction text.
                - ``"depth"`` (*optional*, ``dict[str, array]``): Per-camera
                  depth arrays.  Required when ``cfg.use_depth`` is True;
                  ignored (zero-filled) otherwise.
                - ``"prompt"`` (*optional*, ``str``): Fallback free-form
                  instruction.  When ``task_name`` exists in
                  ``cfg.task_name_to_instruction``, the configured canonical
                  instruction is used instead to match training-time prompts.

        Returns:
            MultiArmManipulationInput ready for the HoloBrain pipeline.
        """
        images = payload.get("images")
        if not isinstance(images, dict):
            raise ValueError("Payload `images` must be a dict.")

        payload_state = payload.get("state", [])
        task_name = payload.get("task_name", "")
        joint_state = build_joint_state_from_payload(
            payload_state=payload_state,
            task_name=task_name,
            gripper_limit=self.cfg.gripper_limit,
        )

        image_data: dict[str, list[np.ndarray]] = {}
        depth_data: dict[str, list[np.ndarray]] = {}
        payload_depths = payload.get("depth") or {}

        for cam_name in GENIESIM_CAMERAS:
            if cam_name not in images:
                raise ValueError(f"Payload is missing image for `{cam_name}`.")

            decoded_image = _decode_image(images[cam_name])
            image_data[cam_name] = [decoded_image]

            payload_depth = payload_depths.get(cam_name)
            if self.cfg.use_depth:
                if payload_depth is None:
                    raise ValueError(
                        f"Payload is missing depth for `{cam_name}`."
                    )
                depth_data[cam_name] = [_decode_depth(cam_name, payload_depth)]
            else:
                if payload_depth is not None:
                    black_depth = np.zeros_like(
                        _decode_depth(cam_name, payload_depth),
                        dtype=np.float32,
                    )
                else:
                    black_depth = np.zeros(
                        decoded_image.shape[:2],
                        dtype=np.float32,
                    )
                depth_data[cam_name] = [black_depth]

        return MultiArmManipulationInput(
            image=image_data,
            depth=depth_data,
            intrinsic=self._camera_intrinsics,
            history_joint_state=[joint_state],
            instruction=self._resolve_instruction(payload),
        )

    def get_actions(self, payload: dict[str, Any]) -> np.ndarray:
        """Run HoloBrain inference on a GenieSim payload.

        Args:
            payload: Same contract as :meth:`data_preprocess`.

        Returns:
            np.ndarray: Action chunk of shape
                ``[cfg.valid_action_step, 22]`` in GenieSim joint order.
        """
        output = self.pipeline(self.data_preprocess(payload))
        joint_actions = _to_numpy(output.action)
        return convert_actions_to_geniesim(
            joint_actions,
            self.cfg.valid_action_step,
            sampling_ratio=self.cfg.sampling_ratio,
        )

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        """Return a deploy-time GenieSim action chunk for one payload."""
        return self.get_actions(obs)


class HoloBrainGenieSim3PolicyCfg(InferencePipelinePolicyCfg):
    class_type: ClassType_co[HoloBrainGenieSim3Policy] = (
        HoloBrainGenieSim3Policy
    )
    """Class type for the policy."""

    pipeline_cfg: ConfigInstanceOf[HoloBrainInferencePipelineCfg] | None = None
    """Configuration for the inference pipeline."""

    inference_prefix: str = "agibot_geniesim3_challenge"
    """Prefix of the saved inference pipeline config files."""

    model_dir: str | None = None
    """Directory of the exported inference pipeline for lazy loading."""

    load_weights: bool = True
    """Whether to load model weights when building from model_dir."""

    load_impl: str = "native"
    """Implementation used to load model weights."""

    model_prefix: str = "model"
    """Prefix of the exported model files in model_dir."""

    valid_action_step: int = 32
    """The number of action steps sent to the simulator."""

    sampling_ratio: float = 1.0
    """Sampling ratio used to resample model outputs before truncation."""

    gripper_limit: float = 1.0
    """Scale applied after raw gripper observations are divided by 120.0."""

    use_depth: bool = True
    """Whether payload depth is consumed by the deploy pipeline."""

    camera_intrinsics: dict[str, Any] | None = None
    """Optional per-camera intrinsic matrix overrides."""

    camera_calibration: dict[str, Any] | None = None
    """Optional per-camera static calibration overrides."""

    task_name_to_instruction: dict[str, str] = DEFAULT_TASK_NAME_TO_INSTRUCTION
    """Task-name instruction map that takes priority over payload prompts."""


def build_policy_from_deploy_config(
    deploy_config: dict[str, Any],
) -> HoloBrainGenieSim3Policy:
    """Construct a deploy policy from a YAML-like config dict."""
    cfg = HoloBrainGenieSim3PolicyCfg(
        model_dir=deploy_config["model_dir"],
        inference_prefix=deploy_config.get(
            "inference_prefix",
            "agibot_geniesim3_challenge",
        ),
        model_prefix=deploy_config.get("model_prefix", "model"),
        load_weights=deploy_config.get("load_weights", True),
        load_impl=deploy_config.get("load_impl", "native"),
        valid_action_step=deploy_config.get("valid_action_step", 32),
        sampling_ratio=deploy_config.get("sampling_ratio", 1.0),
        gripper_limit=deploy_config.get("gripper_limit", 1.0),
        use_depth=deploy_config.get("use_depth", False),
        task_name_to_instruction=deploy_config.get(
            "task_name_to_instruction",
            DEFAULT_TASK_NAME_TO_INSTRUCTION,
        ),
    )
    return HoloBrainGenieSim3Policy(cfg=cfg)
