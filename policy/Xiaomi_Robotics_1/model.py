# Copyright (C) 2026 Xiaomi Corporation.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

"""Xiaomi_Robotics_1 policy for XPolicyLab evaluation.

Slot layout (each arm occupies 8 slots of the 60-dim state/action vector):
  Input state is always joint:
    [0:6] left_arm_joint, [7:8] left_gripper,
    [8:14] right_arm_joint, [15:16] right_gripper; every other slot is zero.
  Output action depends on action_type:
    - joint: [0:6] left_arm_joint, [7:8] left_gripper,
             [8:14] right_arm_joint, [15:16] right_gripper (rest ignored).
    - ee:    [0:3] left_xyz, [3:6] left_axis_angle, [7:8] left_gripper,
             [8:11] right_xyz, [11:14] right_axis_angle, [15:16] right_gripper
             (rest ignored; rotation transformed MiBot -> simulator frame).
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info

# RoboDojo -> MiBot EEF local axis redefinition matrix.
# R_mibot = R_robodojo @ P,  P = Rx(+90°) @ Rz(+90°)
# Only rotation changes; position and gripper are unchanged.
EEF_REFRAME_P = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64
)
# Inverse: P^T (orthogonal matrix)
EEF_REFRAME_P_INV = EEF_REFRAME_P.T


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _normalize(x: torch.Tensor, norm_info: dict) -> torch.Tensor:
    mode = norm_info["mode"]
    if mode == "gaussian":
        return (x - norm_info["mean"]) / norm_info["std"]
    elif mode == "quantile":
        q01, q99 = norm_info["q01"], norm_info["q99"]
        denom = q99 - q01
        valid = denom.abs() > 1e-5
        safe_denom = torch.where(valid, denom, torch.ones_like(denom))
        result = 2 * (x - q01) / safe_denom - 1
        return torch.where(valid, result, x)
    return x


def _denormalize(x: torch.Tensor, norm_info: dict) -> torch.Tensor:
    mode = norm_info["mode"]
    if mode == "gaussian":
        return x * norm_info["std"] + norm_info["mean"]
    elif mode == "quantile":
        q01, q99 = norm_info["q01"], norm_info["q99"]
        denom = q99 - q01
        valid = denom.abs() > 1e-5
        safe_denom = torch.where(valid, denom, torch.ones_like(denom))
        result = (x + 1) / 2 * safe_denom + q01
        return torch.where(valid, result, x)
    return x


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


def _center_crop_pil(img: Image.Image, crop_ratio: float) -> Image.Image:
    w, h = 320, 256
    new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return (
        img.resize((w, h), Image.BILINEAR)
        .crop((left, top, left + new_w, top + new_h))
        .resize((w, h), Image.BILINEAR)
    )


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def _ensure_hwc_uint8(image: Any) -> np.ndarray:
    """Convert observation image to HWC uint8 RGB ndarray."""
    if isinstance(image, (bytes, bytearray, memoryview)):
        import cv2
        buf = np.frombuffer(bytes(image), dtype=np.uint8)
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    image = np.asarray(image)
    if image.ndim == 1 and image.dtype == np.uint8:
        import cv2
        decoded = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    if image.ndim == 3:
        if np.issubdtype(image.dtype, np.floating):
            image = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        return image
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _extract_image(obs: dict, cam_keys: list[str]) -> np.ndarray:
    """Extract image from XPolicyLab observation dict."""
    vision = obs.get("vision", {})
    for key in cam_keys:
        if key not in vision:
            continue
        cam = vision[key]
        if isinstance(cam, dict):
            for img_key in ("color", "colors", "rgb"):
                if img_key in cam:
                    return _ensure_hwc_uint8(cam[img_key])
        else:
            return _ensure_hwc_uint8(cam)
    raise KeyError(f"No image found for camera keys: {cam_keys}")


def _ee_pose_sim_to_mibot(xyz_sim: np.ndarray, quat_wxyz_sim: np.ndarray):
    """Convert an ee pose from simulator frame to MiBot frame.

    Only the EEF local axes are redefined (R_mibot = R_sim @ P); the base-frame
    position is unchanged. Returns (pos, rotm) as float64 for downstream math.
    """
    pos_m = np.asarray(xyz_sim, dtype=np.float64).reshape(3)
    q = np.asarray(quat_wxyz_sim, dtype=np.float64).reshape(4)
    rotm_sim = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    rotm_m = rotm_sim @ EEF_REFRAME_P
    return pos_m, rotm_m


def _ee_pose_mibot_to_sim(pos_mibot: np.ndarray, rotm_mibot: np.ndarray):
    """Convert an ee pose from MiBot frame back to simulator frame.

    Inverse of :func:`_ee_pose_sim_to_mibot`: position unchanged, rotation
    mapped by R_sim = R_mibot @ P^T. Returns (xyz, quat_wxyz) as float32.
    """
    xyz = np.asarray(pos_mibot, dtype=np.float32).reshape(3)  # position unchanged
    rotm_sim = np.asarray(rotm_mibot, dtype=np.float64) @ EEF_REFRAME_P_INV
    quat_xyzw = Rotation.from_matrix(rotm_sim).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]].astype(np.float64)
    # Canonicalize: w >= 0
    if quat_wxyz[0] < 0:
        quat_wxyz = -quat_wxyz
    return xyz, quat_wxyz.astype(np.float32)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = model_cfg
        self.action_type = model_cfg.get("action_type", "joint")
        if self.action_type not in ("joint", "ee"):
            raise ValueError(
                f"[Xiaomi_Robotics_1] Unsupported action_type: {self.action_type!r}. "
                "Supported values are 'joint' and 'ee'."
            )
        self.env_cfg_type = model_cfg["env_cfg_type"]
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Config
        self.task_id = model_cfg.get("task_id", "robodojo")
        self.default_prompt = model_cfg.get(
            "default_prompt", model_cfg.get("task_name", "Perform the task.")
        )
        self.action_max_length = model_cfg.get("action_max_length", 60)
        self.action_length = model_cfg.get("action_length", 10)
        self.state_token_length = model_cfg.get("state_token_length", 1)
        self.input_length = model_cfg.get("input_length", 60)
        self.crop_ratio = model_cfg.get("crop_ratio", 0.95)

        from src.server.deploy import helper

        # Resolve model_dir: explicit path > checkpoints/<ckpt_name>/
        import os
        model_dir = model_cfg.get("model_dir")
        if not model_dir:
            policy_dir = os.path.dirname(os.path.abspath(__file__))
            ckpt_name = model_cfg.get("ckpt_name")
            if ckpt_name:
                model_dir = os.path.join(policy_dir, "checkpoints", ckpt_name)
            if not model_dir or not os.path.isdir(model_dir):
                raise ValueError(
                    f"[Xiaomi_Robotics_1] model_dir is not set and fallback "
                    f"checkpoints/{ckpt_name}/ does not exist at {model_dir}"
                )

        # Load model
        class _ModelArgs:
            model = model_dir

        print(f"[Xiaomi_Robotics_1] Loading model from {model_dir}...")
        (
            self.model,
            self.action_norms,
            self.state_norms,
            self.action_composition,
            _,
        ) = helper(_ModelArgs())

        # Build action_dim_mask (same logic as casatwin policy_server.py)
        action_dim = max(
            v[-1] if isinstance(v, (list, tuple)) and isinstance(v[-1], int) else 0
            for v in self.action_composition.values()
            if isinstance(v, (list, tuple))
        )
        self.action_dim_mask = torch.zeros(
            action_dim, dtype=torch.int32, device=self.device
        )
        for component, indexs in self.action_composition.items():
            if not isinstance(indexs, (list, tuple)):
                continue
            if isinstance(indexs[1], (list, tuple)):
                _, (t_start, t_end) = indexs
                self.action_dim_mask[t_start:t_end] = 1
            else:
                start, end = indexs
                if not component.startswith("action_padding"):
                    self.action_dim_mask[start:end] = 1

        # Get action shape from norms
        self.action_shape = None
        for norm in self.action_norms.values():
            if norm["mode"] == "gaussian":
                self.action_shape = norm["mean"].shape
                break
            elif norm["mode"] == "quantile":
                self.action_shape = norm["q01"].shape
                break

        # Load VLM processor (use_fast=True, special tokens include a_i)
        vlm_processor_path = model_cfg.get(
            "vlm_processor_path", "Qwen/Qwen3-VL-4B-Instruct"
        )
        from transformers import AutoProcessor

        special_tokens = {"score": "<score>", "state": "<state>"}
        special_tokens.update({f"a_{i}": f"<a_{i}>" for i in range(self.action_max_length)})
        self.processor = AutoProcessor.from_pretrained(
            vlm_processor_path,
            use_fast=True,
            extra_special_tokens=special_tokens,
        )

        # Internal state
        self._encoded_obs_list: list[dict[str, Any]] = []

        print(f"[Xiaomi_Robotics_1] Model loaded. action_shape={self.action_shape}")

    # ------------------------------------------------------------------
    # Observation preprocessing
    # ------------------------------------------------------------------

    def _encode_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Encode a single XPolicyLab obs into intermediate representation.

        Returns a dict with 'messages', 'state_tensor', 'action_condition_length'.
        """
        # Extract images
        head_img = _extract_image(obs, ["cam_head", "cam_high", "head_camera"])
        left_img = _extract_image(
            obs, ["cam_left_wrist", "left_camera", "wrist_left"]
        )
        right_img = _extract_image(
            obs, ["cam_right_wrist", "right_camera", "wrist_right"]
        )


        pil_images = [
            _center_crop_pil(Image.fromarray(head_img), self.crop_ratio),
            _center_crop_pil(Image.fromarray(left_img), self.crop_ratio),
            _center_crop_pil(Image.fromarray(right_img), self.crop_ratio),
        ]

        # Extract state. Input state is always joint, packed into the sparse
        # per-arm 8-slot layout: [0:6] left_arm_joint, [7:8] left_gripper,
        # [8:14] right_arm_joint, [15:16] right_gripper; rest zero.
        state = obs.get("state", {})
        left_arm_joint = np.asarray(
            state["left_arm_joint_state"], dtype=np.float32
        ).reshape(-1)[:6]
        left_gripper = float(
            np.asarray(state["left_ee_joint_state"]).reshape(-1)[0]
        )
        right_arm_joint = np.asarray(
            state["right_arm_joint_state"], dtype=np.float32
        ).reshape(-1)[:6]
        right_gripper = float(
            np.asarray(state["right_ee_joint_state"]).reshape(-1)[0]
        )

        state_padded = np.zeros(self.input_length, dtype=np.float32)
        state_padded[0:6] = left_arm_joint
        state_padded[7] = left_gripper
        state_padded[8:14] = right_arm_joint
        state_padded[15] = right_gripper
        state_tensor = torch.from_numpy(state_padded).bfloat16()  # [60]

        # The model predicts RELATIVE (delta) actions w.r.t. the current state
        # (see mibot GetJointAction / GetEEActionPos / GetEEActionAA, ref_frame="ee").
        # Stash the current absolute state so _actions_to_xpl_format can restore
        # absolute actions. joint: current joints; ee: current ee pose in MiBot frame.
        current_state: dict[str, Any] = {
            "left_arm_joint": left_arm_joint.copy(),
            "right_arm_joint": right_arm_joint.copy(),
        }
        if self.action_type == "ee":
            left_pose = np.asarray(state["left_ee_pose"], dtype=np.float64).reshape(7)
            right_pose = np.asarray(state["right_ee_pose"], dtype=np.float64).reshape(7)
            l_pos_m, l_rotm_m = _ee_pose_sim_to_mibot(left_pose[:3], left_pose[3:7])
            r_pos_m, r_rotm_m = _ee_pose_sim_to_mibot(right_pose[:3], right_pose[3:7])
            current_state.update({
                "left_ee_pos_mibot": l_pos_m,
                "left_ee_rotm_mibot": l_rotm_m,
                "right_ee_pos_mibot": r_pos_m,
                "right_ee_rotm_mibot": r_rotm_m,
            })

        # Build prompt via apply_chat_template
        instruction = self._get_instruction(obs)

        # Base messages: vision + instruction
        base_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "The following observations are captured from multiple views.\n# Ego View\n"},
                    {"type": "image", "image": pil_images[0]},
                    {"type": "text", "text": "\n# Left-Wrist View\n"},
                    {"type": "image", "image": pil_images[1]},
                    {"type": "text", "text": "\n# Right-Wrist View\n"},
                    {"type": "image", "image": pil_images[2]},
                    {"type": "text", "text": f"\nGenerate robot actions for the task:\n{instruction} /no_cot"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<cot></cot>"}],
            },
        ]

        # Compute action_condition_length from base messages
        base_data = self.processor.apply_chat_template(
            base_messages,
            tokenize=True,
            return_dict=True,
            do_resize=False,
            return_tensors="pt",
        )
        action_condition_length = base_data["input_ids"].size(1)

        # State/action turn
        state_tokens = "".join(
            ["<state>" for _ in range(self.state_token_length)]
        )
        action_tokens = "".join(
            [f"<a_{i}>" for i in range(self.action_length)]
        )
        action_response = f"{action_tokens}<score>"

        action_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Robot state: {state_tokens}"}
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": action_response}],
            },
        ]

        # Full messages (not yet tokenized as batch)
        full_messages = base_messages + action_messages

        return {
            "messages": full_messages,
            "state_tensor": state_tensor,
            "action_condition_length": action_condition_length,
            "current_state": current_state,
        }

    def _build_batch(
        self, encoded_obs_list: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Build a batched model input from a list of encoded observations."""
        batch_size = len(encoded_obs_list)

        # Tokenize all messages as a batch with padding
        all_messages = [item["messages"] for item in encoded_obs_list]
        batch = self.processor.apply_chat_template(
            all_messages,
            tokenize=True,
            return_dict=True,
            do_resize=False,
            return_tensors="pt",
            padding=True,
        )

        # Stack state tensors [B, 1, 60]
        states = torch.stack(
            [item["state_tensor"] for item in encoded_obs_list], dim=0
        ).unsqueeze(1)  # [B, 1, 60]
        batch["state"] = states

        # action_vlm_condition_segments [B, 2]
        batch["action_vlm_condition_segments"] = torch.tensor(
            [[0, item["action_condition_length"]] for item in encoded_obs_list],
            dtype=torch.int64,
        )

        # Metadata
        batch["task_id"] = self.task_id

        return batch

    def _get_instruction(self, obs: dict[str, Any]) -> str:
        for key in ("instruction", "instructions"):
            if key not in obs:
                continue
            val = obs[key]
            if isinstance(val, list):
                val = val[0] if val else ""
            if isinstance(val, str) and val.strip():
                text = val.strip().rstrip(".") + "."
                return text
        return self.default_prompt

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference_batch(self, data: dict[str, Any]) -> np.ndarray:
        """Run model inference on a batch, return raw actions [B, T, action_dim]."""
        task_id = data.pop("task_id")
        if task_id not in self.action_norms:
            task_id = list(self.action_norms.keys())[0]

        data.pop("global_rank", None)
        data.pop("rollout_i", None)
        data.pop("step_i", None)

        # Move tensors to device
        model_input = {
            key: (value.to(self.device) if isinstance(value, torch.Tensor) else value)
            for key, value in data.items()
        }

        # State normalization
        if "state" in model_input and task_id in self.state_norms:
            s_norm = self.state_norms[task_id]
            if s_norm["mode"] is not None:
                model_input["state"] = _normalize(model_input["state"], s_norm)

        # Action placeholder + normalization
        a_norm = self.action_norms[task_id]
        batch_size = model_input["input_ids"].shape[0]
        if "action" not in model_input:
            model_input["action"] = torch.zeros(
                (batch_size, *self.action_shape),
                device=self.device,
                dtype=torch.bfloat16,
            )
        elif a_norm["mode"] is not None:
            normed = _normalize(model_input["action"], a_norm)
            if a_norm["mode"] == "gaussian":
                norm_valid = a_norm["std"] > 1e-5
            elif a_norm["mode"] == "quantile":
                norm_valid = (a_norm["q99"] - a_norm["q01"]).abs() > 1e-5
            model_input["action"] = torch.where(
                norm_valid, normed, model_input["action"]
            )

        # Action mask (keep [1, action_dim] for broadcast over [B, T, D])
        if "action_mask" not in model_input:
            model_input["action_mask"] = self.action_dim_mask[None]

        # Inference
        with torch.no_grad():
            action = self.model.generate(model_input)

        # Denormalize
        if a_norm["mode"] is not None:
            denormed = _denormalize(action, a_norm)
            if a_norm["mode"] == "gaussian":
                norm_valid = a_norm["std"] > 1e-5
            elif a_norm["mode"] == "quantile":
                norm_valid = (a_norm["q99"] - a_norm["q01"]).abs() > 1e-5
            action = torch.where(norm_valid, denormed, action)

        # [B, T, dim] -> [B, T, action_dim]
        raw_actions = action.float().cpu().numpy()
        return raw_actions

    # ------------------------------------------------------------------
    # Action postprocessing
    # ------------------------------------------------------------------

    def _actions_to_xpl_format(
        self, raw_actions: np.ndarray, current_state: dict[str, Any]
    ) -> list[dict[str, np.ndarray]]:
        """Convert raw model actions [T, 16] to XPolicyLab action dicts.

        The model predicts RELATIVE (delta) actions w.r.t. the observation's
        current state, matching mibot's training-time transforms. This method
        restores ABSOLUTE actions before returning them; ``current_state`` holds
        the observation's absolute state captured in _encode_observation.

        Behavior depends on self.action_type:
          - joint (GetJointAction): abs_joint = current_joint + delta.
                delta slots: [0:6] left_arm_joint, [8:14] right_arm_joint.
                Grippers ([7:8], [15:16]) are absolute (GetAbsAction).
          - ee (GetEEActionPos/AA, ref_frame="ee"): the delta is expressed in the
                current ee frame (MiBot). Restore in MiBot frame, then map to sim:
                    abs_pos_m  = current_pos_m + current_rotm_m @ delta_pos
                    abs_rotm_m = current_rotm_m @ Rot(delta_axis_angle)
                delta slots: [0:3]/[8:11] xyz, [3:6]/[11:14] axis-angle.
                Grippers ([7:8], [15:16]) are absolute (GetAbsAction).
        """
        action_list = []
        for t in range(raw_actions.shape[0]):
            a = raw_actions[t]

            if self.action_type == "joint":
                left_arm = current_state["left_arm_joint"] + a[0:6]
                right_arm = current_state["right_arm_joint"] + a[8:14]
                action_list.append({
                    "left_arm_joint_state": left_arm.astype(np.float32),
                    "left_ee_joint_state": a[7:8].astype(np.float32),
                    "right_arm_joint_state": right_arm.astype(np.float32),
                    "right_ee_joint_state": a[15:16].astype(np.float32),
                })
            else:
                left_xyz, left_quat = self._restore_abs_ee(
                    a[0:3], a[3:6],
                    current_state["left_ee_pos_mibot"],
                    current_state["left_ee_rotm_mibot"],
                )
                right_xyz, right_quat = self._restore_abs_ee(
                    a[8:11], a[11:14],
                    current_state["right_ee_pos_mibot"],
                    current_state["right_ee_rotm_mibot"],
                )
                action_list.append({
                    "left_ee_pose": np.concatenate([left_xyz, left_quat]).astype(np.float32),
                    "right_ee_pose": np.concatenate([right_xyz, right_quat]).astype(np.float32),
                    "left_ee_joint_state": a[7:8].astype(np.float32),
                    "right_ee_joint_state": a[15:16].astype(np.float32),
                })

        return action_list

    @staticmethod
    def _restore_abs_ee(
        delta_pos: np.ndarray,
        delta_aa: np.ndarray,
        current_pos_m: np.ndarray,
        current_rotm_m: np.ndarray,
    ):
        """Restore an absolute ee pose (sim frame) from an ee-frame delta.

        Inverse of mibot GetEEActionPos/GetEEActionAA with ref_frame="ee",
        all in the MiBot frame, then converted back to the simulator frame:
            abs_pos_m  = current_pos_m + current_rotm_m @ delta_pos
            abs_rotm_m = current_rotm_m @ Rot(delta_aa)
        Returns (xyz, quat_wxyz) in the simulator frame.
        """
        delta_pos = np.asarray(delta_pos, dtype=np.float64).reshape(3)
        abs_pos_m = current_pos_m + current_rotm_m @ delta_pos
        delta_rotm = Rotation.from_rotvec(
            np.asarray(delta_aa, dtype=np.float64).reshape(3)
        ).as_matrix()
        abs_rotm_m = current_rotm_m @ delta_rotm
        return _ee_pose_mibot_to_sim(abs_pos_m, abs_rotm_m)

    # ------------------------------------------------------------------
    # ModelTemplate interface
    # ------------------------------------------------------------------

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._encoded_obs_list = [
            self._encode_observation(obs) for obs in obs_list
        ]

    def get_action(self, **kwargs):
        if not self._encoded_obs_list:
            raise AssertionError(
                "[Xiaomi_Robotics_1] Call update_obs before get_action."
            )
        return self._predict_action_chunk(self._encoded_obs_list[0])

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if not self._encoded_obs_list:
            raise AssertionError(
                "[Xiaomi_Robotics_1] Call update_obs_batch before get_action_batch."
            )
        return [
            self._predict_action_chunk(encoded_obs)
            for encoded_obs in self._encoded_obs_list
        ]

    def _predict_action_chunk(
        self, encoded_obs: dict[str, Any]
    ) -> list[dict[str, np.ndarray]]:
        """Run inference on a single encoded observation."""
        batch_data = self._build_batch([encoded_obs])
        raw_actions = self._run_inference_batch(batch_data)  # [1, T, 16]
        return self._actions_to_xpl_format(
            raw_actions[0], encoded_obs["current_state"]
        )

    def reset(self):
        self._encoded_obs_list = []
        print("[Xiaomi_Robotics_1] Model reset.")
