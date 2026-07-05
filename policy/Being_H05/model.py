from __future__ import annotations

import os
import sys
from typing import Any

import cv2
import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state, unpack_robot_state


BEINGH_ROOT = os.path.join(os.path.dirname(__file__), "Being-H")
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if BEINGH_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(BEINGH_ROOT))

from BeingH.inference.beingh_policy import BeingHPolicy


def _resolve_latest_step_dir(run_dir: str) -> str:
    if os.path.isfile(os.path.join(run_dir, "config.json")):
        return run_dir
    if not os.path.isdir(run_dir):
        return run_dir
    step_dirs = [
        name
        for name in os.listdir(run_dir)
        if name.isdigit() and os.path.isdir(os.path.join(run_dir, name))
    ]
    if not step_dirs:
        return run_dir
    latest = sorted(step_dirs, key=int)[-1]
    return os.path.join(run_dir, latest)


def _resolve_model_path(model_cfg: dict[str, Any]) -> str:
    model_path = model_cfg.get("model_path")
    if model_path:
        return _resolve_latest_step_dir(model_path)

    data_project_name = model_cfg.get("data_project_name") or model_cfg.get("xpolicylab_bench_name")
    ckpt_name = model_cfg.get("ckpt_name")
    env_cfg_type = model_cfg.get("env_cfg_type")
    expert_data_num = model_cfg.get("expert_data_num")
    action_type = model_cfg.get("action_type", "joint")
    seed = model_cfg.get("seed")
    if None in (data_project_name, ckpt_name, env_cfg_type, expert_data_num, seed):
        raise ValueError(
            "deploy config must provide model_path, or data_project_name + ckpt fields for XPolicyLab 6-tuple."
        )

    run_id = f"{data_project_name}-{ckpt_name}-{env_cfg_type}-{expert_data_num}-{action_type}-{seed}"
    candidate = os.path.join(_SCRIPT_DIR, "checkpoints", run_id)
    if not os.path.isdir(candidate):
        raise ValueError(f"checkpoint run dir not found: {candidate}")
    return _resolve_latest_step_dir(candidate)


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.task_name = model_cfg["task_name"]
        self.action_type = model_cfg.get("action_type", "joint")
        self.default_prompt = model_cfg.get("prompt", self.task_name)
        self.robot_action_dim_info = (
            get_robot_action_dim_info(model_cfg["env_cfg_type"]) if model_cfg.get("env_cfg_type") is not None else None
        )
        self._latest_env_idx_list: list[int] = [0]
        self.observation_window: list[dict[str, Any]] | None = None

        if self.action_type != "joint":
            raise ValueError("Being_H05 currently only supports joint/qpos actions in XPolicyLab.")

        model_path = _resolve_model_path(model_cfg)

        data_config_name = model_cfg.get("data_config_name", "robodojo_qpos")
        bench_name = model_cfg.get("bench_name", "robodojo_posttrain")
        embodiment_tag = model_cfg.get("embodiment_tag", "new_embodiment")
        prompt_template = model_cfg.get("prompt_template", "long")
        prop_pos = model_cfg.get("prop_pos", "front")
        max_view_num = int(model_cfg.get("max_view_num", -1))
        use_fixed_view = bool(model_cfg.get("use_fixed_view", False))
        enable_rtc = bool(model_cfg.get("enable_rtc", False))
        device = model_cfg.get("device", "cuda")

        if prompt_template == "short":
            instruction_template = "{task_description}"
        else:
            instruction_template = (
                "According to the instruction '{task_description}', what's the micro-step actions in the next {k} steps?"
            )

        self.policy = BeingHPolicy(
            model_path=model_path,
            data_config_name=data_config_name,
            bench_name=bench_name,
            embodiment_tag=embodiment_tag,
            instruction_template=instruction_template,
            prop_pos=prop_pos,
            max_view_num=max_view_num,
            use_fixed_view=use_fixed_view,
            device=device,
            enable_rtc=enable_rtc,
        )
        self.model = self.policy

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        self.observation_window = [
            encode_obs(obs, self.action_type, self.robot_action_dim_info, self.default_prompt) for obs in obs_list
        ]

    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        env_idx_list = env_idx_list or self._latest_env_idx_list
        action_list = []

        for batch_index, _ in enumerate(env_idx_list):
            beingh_obs = dict(self.observation_window[batch_index])
            result = self.policy.get_action(beingh_obs)
            packed_actions = decode_action_chunk(result)

            if self.robot_action_dim_info is None:
                action_list.append(packed_actions)
            else:
                action_list.append(
                    unpack_robot_state(
                        packed_actions,
                        self.action_type,
                        self.robot_action_dim_info,
                        source_type="obs",
                    )
                )

        return action_list

    def reset(self):
        self.observation_window = None
        self._latest_env_idx_list = [0]

    def reset_obsrvationwindows(self):
        self.reset()


def extract_prompt(observation, default_prompt):
    for key in ("instruction", "instructions", "prompt", "task_instruction"):
        value = observation.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if value is None:
            continue
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        text = str(value).strip()
        if text:
            return text
    return default_prompt


def encode_obs(observation, action_type, robot_action_dim_info, default_prompt):
    prompt = extract_prompt(observation, default_prompt)

    if "images" in observation and "state" in observation:
        packed_state = np.asarray(observation["state"], dtype=np.float32)
        head = ensure_hwc_uint8(observation["images"]["cam_high"])
        left = ensure_hwc_uint8(observation["images"]["cam_left_wrist"])
        right = ensure_hwc_uint8(observation["images"]["cam_right_wrist"])
        return build_beingh_obs(head, left, right, packed_state, prompt)

    if robot_action_dim_info is None:
        raise ValueError("env_cfg_type is required when encoding raw environment observations.")

    head = ensure_hwc_uint8(extract_image(observation, ["cam_head", "head_camera", "cam_high", "top_camera"]))
    left = ensure_hwc_uint8(
        extract_image(observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"])
    )
    right = ensure_hwc_uint8(
        extract_image(observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"])
    )
    packed_state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
    return build_beingh_obs(head, left, right, packed_state, prompt)


def build_beingh_obs(head, left, right, packed_state, prompt):
    qpos = np.asarray(packed_state, dtype=np.float32)
    if qpos.shape[-1] != 14:
        raise ValueError(f"Being_H05 expects 14-dim packed joint state, got shape {qpos.shape}.")

    return {
        "language.instruction": prompt,
        "video.head_view": head,
        "video.right_wrist_view": right,
        "video.left_wrist_view": left,
        "state.left_arm_joint_position": qpos[0:6],
        "state.left_gripper_position": qpos[6:7],
        "state.right_arm_joint_position": qpos[7:13],
        "state.right_gripper_position": qpos[13:14],
    }


def decode_action_chunk(result):
    left_arm = np.asarray(result["action.left_arm_joint_position"], dtype=np.float32)
    left_grip = np.asarray(result["action.left_gripper_position"], dtype=np.float32)
    right_arm = np.asarray(result["action.right_arm_joint_position"], dtype=np.float32)
    right_grip = np.asarray(result["action.right_gripper_position"], dtype=np.float32)
    return np.concatenate([left_arm, left_grip, right_arm, right_grip], axis=-1)


def extract_image(observation, candidate_names):
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
    raise KeyError(f"Could not find any image for candidates: {candidate_names}")


def ensure_hwc_uint8(image):
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


def decode_compressed_image(image_buffer):
    decoded = cv2.imdecode(np.asarray(image_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Failed to decode compressed image buffer.")
    return decoded
