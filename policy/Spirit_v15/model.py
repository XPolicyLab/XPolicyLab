from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from XPolicyLab.model_template import ModelTemplate

_POLICY_DIR = Path(__file__).resolve().parent
if str(_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(_POLICY_DIR))

from spirit_v15.model import SpiritVLAPolicy
from spirit_v15.robochallenge.runner.executor import _post_process_action
from spirit_v15.robochallenge.runner.task_info import (
    TASK_INFO,
    TASKS_USE_LESS_CHUNK_SIZE,
    TASTS_APPLY_GRIPPER_BINARIZATION,
)


TASK_NAME_ALIASES = {
    "stack_bowls_three": "stack_bowls",
    "stack_bowls_two": "stack_bowls",
}


def extract_image(observation: dict[str, Any], candidate_names: list[str]) -> Any:
    if "vision" in observation:
        vision = observation.get("vision", {})
        for candidate_name in candidate_names:
            if candidate_name not in vision:
                continue
            image = vision[candidate_name]
            if isinstance(image, dict):
                for image_key in ("color", "rgb"):
                    if image_key in image:
                        return image[image_key]
            else:
                return image

    images = observation.get("images", {})
    for candidate_name in candidate_names:
        if candidate_name in images:
            return images[candidate_name]

    raise KeyError(f"Could not find any image for candidates: {candidate_names}")


def decode_compressed_image(image_buffer: np.ndarray) -> np.ndarray:
    decoded = cv2.imdecode(np.asarray(image_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Failed to decode compressed image buffer.")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def ensure_hwc_uint8(image: Any) -> np.ndarray:
    if isinstance(image, (bytes, bytearray, memoryview)):
        image = decode_compressed_image(np.frombuffer(bytes(image), dtype=np.uint8))

    image = np.asarray(image)

    if image.ndim == 1 and image.dtype == np.uint8:
        image = decode_compressed_image(image)

    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.shape[-1] in (1, 3):
        return image
    if image.shape[0] in (1, 3):
        return np.transpose(image, (1, 2, 0))
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _normalize_prompt_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    elif isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (list, tuple)):
        for item in value:
            normalized = _normalize_prompt_value(item)
            if normalized is not None:
                return normalized
        return None

    if isinstance(value, str):
        value = value.strip()
        return value or None
    return str(value)


def resolve_prompt(observation: dict[str, Any], default_prompt: str | None) -> str | None:
    for key in ("prompt", "instruction", "task", "language_instruction"):
        prompt = _normalize_prompt_value(observation.get(key))
        if prompt is not None:
            return prompt
    return _normalize_prompt_value(default_prompt)


def quat_wxyz_to_xyzw(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape[0] != 7:
        raise ValueError(f"Expected 7-dim pose, got {pose.shape}")
    return np.concatenate([pose[:3], pose[4:7], pose[3:4]], axis=0).astype(np.float32)


def quat_xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32)
    if quaternion.shape[0] != 4:
        raise ValueError(f"Expected 4-dim quaternion, got {quaternion.shape}")
    return np.concatenate([quaternion[3:4], quaternion[:3]], axis=0).astype(np.float32)


def _to_scalar_gripper(value: Any) -> float:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError("Empty gripper state is not supported.")
    return float(array[0])


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = dict(model_cfg)
        self.device = self._get_device(self.model_cfg.get("device", "auto"))

        checkpoint_path = self.model_cfg.get("checkpoint_path") or self.model_cfg.get("model_path")
        if checkpoint_path is None:
            raise ValueError("checkpoint_path or model_path is required for Spirit_v15.")

        task_name = self.model_cfg.get("task_name")
        self.default_task_name = self._resolve_task_name(task_name) if task_name else None
        self.default_prompt = self.model_cfg.get("prompt", self.default_task_name or task_name)
        self.used_chunk_size = int(self.model_cfg.get("used_chunk_size", 60))
        self.raw_embodiment_stats = None

        raw_stats_path = self.model_cfg.get("raw_embodiment_stats_json_path")
        if raw_stats_path:
            with open(raw_stats_path, "r", encoding="utf-8") as file:
                self.raw_embodiment_stats = json.load(file)

        self.policy = SpiritVLAPolicy.from_pretrained(checkpoint_path)
        self.policy.to(self.device)
        self.policy.eval()
        self.model = self.policy

        self._latest_obs_list: list[dict[str, Any]] | None = None
        self._latest_env_idx_list: list[int] = [0]

    def _get_device(self, device_arg: str):
        if device_arg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        requested = torch.device(device_arg)
        if requested.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return requested

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        self._latest_obs_list = list(obs_list)

    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self._latest_obs_list is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        env_idx_list = env_idx_list or self._latest_env_idx_list
        action_list = []

        for batch_index, _ in enumerate(env_idx_list):
            observation = self._latest_obs_list[batch_index]
            result = self.infer(
                observation=observation,
                instruction=resolve_prompt(observation, self.default_prompt),
                task_name=observation.get("task_name"),
            )
            action_list.append(
                self._decode_action_chunk(
                    result["actions"],
                    result["task_name"],
                    result["action_type"],
                )
            )

        return action_list

    def reset(self):
        self._latest_obs_list = None
        self._latest_env_idx_list = [0]

    def reset_obsrvationwindows(self):
        self.reset()

    def infer(
        self,
        observation: dict[str, Any],
        instruction: str | None = None,
        task_name: str | None = None,
    ) -> dict[str, Any]:
        resolved_task_name = self._resolve_task_name(task_name)
        batch = self._prepare_batch(observation, resolved_task_name)

        used_chunk_size = self.used_chunk_size
        if resolved_task_name in TASKS_USE_LESS_CHUNK_SIZE:
            used_chunk_size = 40

        binarization_threshold = TASTS_APPLY_GRIPPER_BINARIZATION.get(resolved_task_name)
        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext(),
        ):
            action_tensor = self.policy.select_action(batch).cpu()

        actions = _post_process_action(
            action_tensor.squeeze(0).numpy(),
            batch["observation.state.before_norm"].numpy(),
            TASK_INFO[resolved_task_name]["robot_type"],
            used_chunk_size,
            self.raw_embodiment_stats,
            binarization_threshold,
        )
        return {
            "actions": actions,
            "action_type": self._resolve_action_type(resolved_task_name),
            "task_name": resolved_task_name,
            "instruction": instruction,
        }

    def _resolve_task_name(self, task_name: str | None) -> str:
        for candidate in (task_name, self.default_task_name):
            if not candidate:
                continue
            if candidate in TASK_INFO:
                return candidate
            alias = TASK_NAME_ALIASES.get(candidate)
            if alias in TASK_INFO:
                return alias
        available = ", ".join(sorted(TASK_INFO.keys()))
        raise KeyError(f"unsupported Spirit task name: {task_name!r}; available tasks: {available}")

    def _prepare_batch(self, observation: dict[str, Any], task_name: str) -> dict[str, Any]:
        spirit_observation = self._normalize_observation(observation)
        robot_type = TASK_INFO[task_name]["robot_type"]
        item: dict[str, Any] = {
            "task": [TASK_INFO[task_name]["task"]],
            "normalized_in_getitem": torch.tensor([False]),
            "batch_source": "rb",
            "robot_type": [robot_type],
        }

        state_tensor = self._extract_internal_state(spirit_observation, robot_type)
        item["observation.state.before_norm"] = state_tensor.clone()
        item["observation.state"] = state_tensor.unsqueeze(0).to(self.device)

        semantic_images = {
            "high": spirit_observation["observation"]["head_camera"]["rgb"],
            "left_hand": spirit_observation["observation"]["left_camera"]["rgb"],
            "right_hand": spirit_observation["observation"]["right_camera"]["rgb"],
        }
        for key in (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ):
            image = semantic_images[TASK_INFO[task_name][key]]
            item[key] = self._image_to_tensor(image).unsqueeze(0).to(self.device)
        return item

    def _normalize_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if "observation" in observation:
            normalized = dict(observation)
            if "endpose" not in normalized:
                state_dict = observation.get("state")
                if isinstance(state_dict, dict):
                    endpose = self._build_endpose_from_state(state_dict)
                    if endpose is not None:
                        normalized["endpose"] = endpose
            return normalized

        state_dict = observation.get("state")
        normalized = {
            "observation": {
                "head_camera": {
                    "rgb": ensure_hwc_uint8(
                        extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"])
                    )
                },
                "left_camera": {
                    "rgb": ensure_hwc_uint8(
                        extract_image(observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"])
                    )
                },
                "right_camera": {
                    "rgb": ensure_hwc_uint8(
                        extract_image(observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"])
                    )
                },
            }
        }

        if isinstance(state_dict, dict):
            endpose = self._build_endpose_from_state(state_dict)
            if endpose is not None:
                normalized["endpose"] = endpose

            joint_vector = self._build_joint_action_from_state(state_dict)
            if joint_vector is not None:
                normalized["joint_action"] = {"vector": joint_vector}

        return normalized

    def _build_endpose_from_state(self, state_dict: dict[str, Any]) -> dict[str, Any] | None:
        if {"left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state"}.issubset(state_dict):
            return {
                "left_endpose": quat_wxyz_to_xyzw(np.asarray(state_dict["left_ee_pose"], dtype=np.float32)),
                "left_gripper": _to_scalar_gripper(state_dict["left_ee_joint_state"]),
                "right_endpose": quat_wxyz_to_xyzw(np.asarray(state_dict["right_ee_pose"], dtype=np.float32)),
                "right_gripper": _to_scalar_gripper(state_dict["right_ee_joint_state"]),
            }

        pose = state_dict.get("ee_pose")
        gripper = state_dict.get("ee_joint_state")
        if pose is None or gripper is None:
            pose = state_dict.get("left_ee_pose")
            gripper = state_dict.get("left_ee_joint_state")

        if pose is None or gripper is None:
            return None

        return {
            "left_endpose": quat_wxyz_to_xyzw(np.asarray(pose, dtype=np.float32)),
            "left_gripper": _to_scalar_gripper(gripper),
        }

    def _build_joint_action_from_state(self, state_dict: dict[str, Any]) -> np.ndarray | None:
        if "arm_joint_state" in state_dict and "ee_joint_state" in state_dict:
            arm = np.asarray(state_dict["arm_joint_state"], dtype=np.float32).reshape(-1)
            gripper = np.asarray(state_dict["ee_joint_state"], dtype=np.float32).reshape(-1)
            return np.concatenate([arm, gripper], axis=0).astype(np.float32)

        if {"left_arm_joint_state", "left_ee_joint_state", "right_arm_joint_state", "right_ee_joint_state"}.issubset(
            state_dict
        ):
            left_arm = np.asarray(state_dict["left_arm_joint_state"], dtype=np.float32).reshape(-1)
            left_gripper = np.asarray(state_dict["left_ee_joint_state"], dtype=np.float32).reshape(-1)
            right_arm = np.asarray(state_dict["right_arm_joint_state"], dtype=np.float32).reshape(-1)
            right_gripper = np.asarray(state_dict["right_ee_joint_state"], dtype=np.float32).reshape(-1)
            return np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=0).astype(np.float32)

        return None

    def _extract_internal_state(self, observation: dict[str, Any], robot_type: str) -> torch.Tensor:
        endpose = observation.get("endpose") or {}
        if robot_type == "aloha":
            if all(key in endpose for key in ("left_endpose", "left_gripper", "right_endpose", "right_gripper")):
                return self._dual_ee_to_internal_state(
                    left_endpose=np.asarray(endpose["left_endpose"], dtype=np.float32),
                    left_gripper=float(endpose["left_gripper"]),
                    right_endpose=np.asarray(endpose["right_endpose"], dtype=np.float32),
                    right_gripper=float(endpose["right_gripper"]),
                )

        if robot_type in {"ARX5", "Franka", "UR5"}:
            if "left_endpose" in endpose and "left_gripper" in endpose:
                return self._single_ee_to_internal_state(
                    ee_pose=np.asarray(endpose["left_endpose"], dtype=np.float32),
                    gripper=float(endpose["left_gripper"]),
                )

        if "joint_action" in observation and "vector" in observation["joint_action"]:
            return self._robotwin_joint_state_to_internal(
                np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
                robot_type,
            )
        if "action" in observation:
            return self._robotwin_joint_state_to_internal(np.asarray(observation["action"], dtype=np.float32), robot_type)
        raise KeyError("missing usable state in endpose, joint_action.vector, or action")

    @staticmethod
    def _image_to_tensor(image: np.ndarray) -> torch.Tensor:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected RGB image with shape [H, W, 3], got {array.shape}")
        if array.dtype != np.uint8:
            array = np.clip(array, 0.0, 1.0) if np.issubdtype(array.dtype, np.floating) else array
            if array.max() <= 1.0:
                array = (array * 255.0).astype(np.uint8)
            else:
                array = array.astype(np.uint8)
        resized = cv2.resize(array, (320, 240), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(np.asarray(resized, dtype=np.float32)).permute(2, 0, 1) / 255.0

    @staticmethod
    def _robotwin_joint_state_to_internal(state: np.ndarray, robot_type: str) -> torch.Tensor:
        state_tensor = torch.zeros(14, dtype=torch.float32)
        if robot_type == "ARX5":
            if state.shape[0] != 7:
                raise ValueError(f"expected 7-dim ARX5 state, got {state.shape}")
            state_tensor[:3] = torch.from_numpy(state[:3])
            state_tensor[3:6] = torch.tensor(Rotation.from_euler("xyz", state[3:6], degrees=False).as_rotvec())
            state_tensor[6] = torch.tensor(state[6], dtype=torch.float32)
            return state_tensor
        if robot_type == "UR5":
            if state.shape[0] not in {7, 14}:
                raise ValueError(f"expected 7-dim or 14-dim UR5 state, got {state.shape}")
            state_tensor[:7] = torch.from_numpy(state[:7])
            return state_tensor
        if robot_type == "Franka":
            if state.shape[0] not in {8, 16}:
                raise ValueError(f"expected 8-dim or 16-dim Franka state, got {state.shape}")
            state_tensor[:3] = torch.from_numpy(state[:3])
            state_tensor[3:6] = torch.tensor(Rotation.from_quat(state[3:7]).as_rotvec())
            state_tensor[6] = torch.tensor(state[7], dtype=torch.float32)
            return state_tensor
        if robot_type == "aloha":
            raise ValueError(
                f"aloha requires endpose fields in observation; received only joint state with shape {state.shape}"
            )
        raise ValueError(f"unsupported robot type: {robot_type}")

    @staticmethod
    def _single_ee_to_internal_state(ee_pose: np.ndarray, gripper: float) -> torch.Tensor:
        if ee_pose.shape[0] != 7:
            raise ValueError(f"expected 7-dim end-effector pose, got {ee_pose.shape}")

        state_tensor = torch.zeros(14, dtype=torch.float32)
        state_tensor[:3] = torch.from_numpy(ee_pose[:3])
        state_tensor[3:6] = torch.tensor(Rotation.from_quat(ee_pose[3:]).as_rotvec())
        state_tensor[6] = torch.tensor(gripper, dtype=torch.float32)
        return state_tensor

    @staticmethod
    def _dual_ee_to_internal_state(
        left_endpose: np.ndarray,
        left_gripper: float,
        right_endpose: np.ndarray,
        right_gripper: float,
    ) -> torch.Tensor:
        if left_endpose.shape[0] != 7 or right_endpose.shape[0] != 7:
            raise ValueError(
                f"expected 7-dim dual-arm endpose, got left={left_endpose.shape}, right={right_endpose.shape}"
            )

        state_tensor = torch.zeros(14, dtype=torch.float32)
        state_tensor[:3] = torch.from_numpy(left_endpose[:3])
        state_tensor[3:6] = torch.tensor(Rotation.from_quat(left_endpose[3:]).as_rotvec())
        state_tensor[6] = torch.tensor(left_gripper, dtype=torch.float32)
        state_tensor[7:10] = torch.from_numpy(right_endpose[:3])
        state_tensor[10:13] = torch.tensor(Rotation.from_quat(right_endpose[3:]).as_rotvec())
        state_tensor[13] = torch.tensor(right_gripper, dtype=torch.float32)
        return state_tensor

    @staticmethod
    def _resolve_action_type(task_name: str) -> str:
        action_type = TASK_INFO[task_name].get("action_type")
        if action_type == "leftjoint":
            return "joint"
        return "ee"

    def _decode_action_chunk(self, action_chunk: list[list[float]], task_name: str, action_type: str):
        robot_type = TASK_INFO[task_name]["robot_type"]
        if action_type == "joint":
            return [self._decode_joint_action(action, robot_type) for action in action_chunk]
        return [self._decode_ee_action(action, robot_type) for action in action_chunk]

    @staticmethod
    def _decode_joint_action(action: list[float], robot_type: str) -> dict[str, np.ndarray]:
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if robot_type == "UR5":
            return {
                "arm_joint_state": action_array[:6],
                "ee_joint_state": action_array[6:7],
            }
        if robot_type == "aloha":
            return {
                "left_arm_joint_state": action_array[:6],
                "left_ee_joint_state": action_array[6:7],
                "right_arm_joint_state": action_array[7:13],
                "right_ee_joint_state": action_array[13:14],
            }
        raise ValueError(f"Unsupported joint action robot type: {robot_type}")

    @staticmethod
    def _decode_ee_action(action: list[float], robot_type: str) -> dict[str, np.ndarray]:
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)

        if robot_type == "ARX5":
            quat_xyzw = Rotation.from_euler("xyz", action_array[3:6], degrees=False).as_quat().astype(np.float32)
            return {
                "ee_pose": np.concatenate([action_array[:3], quat_xyzw_to_wxyz(quat_xyzw)], axis=0),
                "ee_joint_state": action_array[6:7],
            }

        if robot_type == "Franka":
            return {
                "ee_pose": np.concatenate([action_array[:3], quat_xyzw_to_wxyz(action_array[3:7])], axis=0),
                "ee_joint_state": action_array[7:8],
            }

        if robot_type == "aloha":
            return {
                "left_ee_pose": np.concatenate([action_array[:3], quat_xyzw_to_wxyz(action_array[3:7])], axis=0),
                "left_ee_joint_state": action_array[7:8],
                "right_ee_pose": np.concatenate([action_array[8:11], quat_xyzw_to_wxyz(action_array[11:15])], axis=0),
                "right_ee_joint_state": action_array[15:16],
            }

        if robot_type == "UR5":
            return {
                "arm_joint_state": action_array[:6],
                "ee_joint_state": action_array[6:7],
            }

        raise ValueError(f"Unsupported ee action robot type: {robot_type}")