"""XPolicyLab adapter for the X-WAM policy.

Bridges the XPolicyLab eval protocol (``ModelTemplate``) to the batched X-WAM
inference wrapper in ``X-WAM/evaluation/deploy_policy.py``.

Data flow per inference:
    XPolicyLab obs  ->  multi-view RGB + raw 16-d proprio + prompt
                    ->  XWAMPolicy.infer_batch (one batched forward)
                    ->  denormalized **delta** EE actions
                    ->  integrated to absolute EE poses (anchored on current state)
                    ->  XPolicyLab ee action dicts.

Batching: ``update_obs_batch`` / ``get_action_batch`` collect every running env,
stack them, and run a single ``infer_batch`` call so the GPU sees a real batch.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info

POLICY_DIR = Path(__file__).resolve().parent
XWAM_ROOT = POLICY_DIR / "X-WAM"
EVAL_DIR = XWAM_ROOT / "evaluation"

# Number of (left/right) end-effector pose dims emitted to the env: xyz + quat_wxyz.
_EE_POSE_DIM = 7
# Per-arm slices inside the X-WAM 16-d proprio / action layouts.
_PROPRIO_ARM_DIM = 8   # xyz(3) + quat_wxyz(4) + gripper(1)
_ACTION_ARM_DIM = 7    # dxyz(3) + daxisangle(3) + dgripper(1)

# EE-frame reframe matrix applied during training data preparation:
#   R_train = R_env @ P,  with P = Rx(+90°) @ Rz(+90°)
# X-WAM was trained on the reframed coordinate system, so the adapter converts
# env poses -> training frame before inference, and converts model outputs back
# env <- training afterwards. Position / gripper are unaffected (rotation only).
# P is a fixed constant (does not change across environments).
_REFRAME_P = np.array(
    [[0.0, -1.0, 0.0],
     [0.0, 0.0, -1.0],
     [1.0, 0.0, 0.0]],
    dtype=np.float64,
)
_REFRAME_P_INV = _REFRAME_P.T  # P is a rotation: inverse == transpose


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in {"", "none", "null"}


def _to_uint8_hwc(image: Any) -> np.ndarray:
    """Decode/standardize an obs image into an HWC uint8 RGB array."""
    arr = np.asarray(image)
    # JPEG/PNG byte buffer.
    if arr.ndim == 1 and arr.dtype == np.uint8:
        arr = cv2.imdecode(arr, cv2.IMREAD_COLOR)[..., ::-1]  # BGR -> RGB
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {arr.shape}")
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(arr)) <= 1.5 else 1.0
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Unsupported image channels: {arr.shape}")
    return np.ascontiguousarray(arr)


def _extract_image(obs: dict, candidates: list[str]) -> np.ndarray:
    vision = obs.get("vision", {})
    for name in candidates:
        if name in vision:
            item = vision[name]
            if isinstance(item, dict):
                for key in ("color", "rgb", "image"):
                    if key in item:
                        return _to_uint8_hwc(item[key])
            return _to_uint8_hwc(item)
        if name in obs:
            return _to_uint8_hwc(obs[name])
    raise KeyError(f"Could not find any camera among {candidates} in obs['vision'].")


def _get_instruction(obs: dict, fallback: str) -> str:
    value = obs.get("task_instruction")
    if value is None:
        value = obs.get("instruction", obs.get("instructions"))
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else fallback
    if value is None:
        return fallback
    if hasattr(value, "item"):
        value = value.item()
    text = str(value).strip()
    return text if text else fallback


def _canonical_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Enforce positive-w canonical form on a wxyz quaternion."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    q = q / norm
    if q[0] < 0:
        q = -q
    return q.astype(np.float32)


def _quat_wxyz_to_rotm(quat_wxyz: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(np.asarray(quat_wxyz, dtype=np.float64)[[1, 2, 3, 0]]).as_matrix()


def _rotm_to_quat_wxyz(rotm: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(rotm).as_quat()
    return _canonical_quat_wxyz(quat_xyzw[[3, 0, 1, 2]])


def _reframe_quat_env_to_train(quat_wxyz: np.ndarray) -> np.ndarray:
    """env-frame quaternion -> training-frame quaternion: R_train = R_env @ P."""
    return _rotm_to_quat_wxyz(_quat_wxyz_to_rotm(quat_wxyz) @ _REFRAME_P)


def _reframe_quat_train_to_env(quat_wxyz: np.ndarray) -> np.ndarray:
    """training-frame quaternion -> env-frame quaternion: R_env = R_train @ P^T."""
    return _rotm_to_quat_wxyz(_quat_wxyz_to_rotm(quat_wxyz) @ _REFRAME_P_INV)


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.action_type = self.model_cfg.get("action_type") or "ee"
        if self.action_type != "ee":
            raise ValueError(
                f"X-WAM is an EE-space policy; action_type must be 'ee', got {self.action_type!r}."
            )
        self.env_cfg_type = self.model_cfg.get("env_cfg_type")
        if not self.env_cfg_type:
            raise ValueError("env_cfg_type is required for X-WAM.")
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.num_arms = len(self.robot_action_dim_info["arm_dim"])
        if self.num_arms not in (1, 2):
            raise ValueError(f"X-WAM supports single- or dual-arm only, got {self.num_arms} arms.")

        # Number of predicted actions to execute before requesting a new obs.
        self.replan_steps = int(self.model_cfg.get("replan_steps") or 0) or None
        self.default_instruction = str(
            self.model_cfg.get("default_instruction")
            or self.model_cfg.get("prompt")
            or "follow the instruction"
        )
        self.cfg_scale = float(self.model_cfg.get("cfg") or self.model_cfg.get("text_cfg_scale") or 0.0)
        self.base_seed = int(self.model_cfg.get("seed") or 0)

        # Camera name candidates: X-WAM views are sorted -> head/left/right.
        self.view_candidates = {
            "head": self.model_cfg.get("head_view_candidates")
            or ["cam_head", "cam_high", "head_camera", "cam_third_view"],
            "left": self.model_cfg.get("left_view_candidates")
            or ["cam_left_wrist", "left_camera", "wrist_left", "cam_left"],
            "right": self.model_cfg.get("right_view_candidates")
            or ["cam_right_wrist", "right_camera", "wrist_right", "cam_right"],
        }

        # Per-rollout step counter -> reproducible per-call seeds.
        self._step_id = 0
        self._batch: dict[int, dict[str, Any]] = {}
        self._order: list[int] = []

        self.allow_dummy_policy = _is_true(self.model_cfg.get("allow_dummy_policy", False))
        self.policy = None
        if self.allow_dummy_policy:
            print("[X-WAM] allow_dummy_policy=true; checkpoint loading skipped (debug only).")
            return

        for path in (str(XWAM_ROOT), str(EVAL_DIR)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from evaluation.deploy_policy import get_model as _get_model

        self.policy = _get_model(self.model_cfg)
        if not self.policy.has_right_arm and self.num_arms == 2:
            raise ValueError("Checkpoint is single-arm but env_cfg is dual-arm.")
        print(
            f"[X-WAM] initialized | arms={self.num_arms} | "
            f"replan_steps={self.replan_steps or 'all'} | cfg={self.cfg_scale}"
        )

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------
    def _pack_proprio(self, obs: dict) -> np.ndarray:
        """Build the X-WAM 16-d proprio vector from an XPolicyLab obs.

        Layout: [l_xyz3, l_quat_wxyz4, l_grip1, r_xyz3, r_quat_wxyz4, r_grip1].
        Single-arm fills the right block with zeros (masked out downstream).
        """
        state = obs.get("state", {})
        vec = np.zeros(2 * _PROPRIO_ARM_DIM, dtype=np.float32)

        if self.num_arms == 1:
            arm_keys = [("ee_pose", "ee_joint_state")]
        else:
            arm_keys = [
                ("left_ee_pose", "left_ee_joint_state"),
                ("right_ee_pose", "right_ee_joint_state"),
            ]

        for i, (pose_key, grip_key) in enumerate(arm_keys):
            if pose_key not in state:
                raise KeyError(f"Missing '{pose_key}' in obs['state'] for X-WAM proprio.")
            pose = np.asarray(state[pose_key], dtype=np.float64).reshape(_EE_POSE_DIM)
            grip = float(np.asarray(state[grip_key], dtype=np.float64).reshape(-1)[0])
            off = i * _PROPRIO_ARM_DIM
            vec[off : off + 3] = pose[:3]                          # xyz (frame-invariant)
            # Reframe orientation into the training coordinate system before it
            # reaches the model.
            vec[off + 3 : off + 7] = _reframe_quat_env_to_train(pose[3:7])
            vec[off + 7] = grip                                    # gripper (frame-invariant)
        return vec

    def _encode_obs(self, obs: dict) -> dict[str, Any]:
        video = np.stack(
            [
                _extract_image(obs, list(self.view_candidates["head"])),
                _extract_image(obs, list(self.view_candidates["left"])),
                _extract_image(obs, list(self.view_candidates["right"])),
            ],
            axis=0,
        )  # [V=3, H, W, C], view order = sorted(head/left/right)
        return {
            "video": video,
            "proprio": self._pack_proprio(obs),
            "prompt": _get_instruction(obs, self.default_instruction),
            "anchor": self._pack_proprio(obs),  # absolute anchor for delta integration
        }

    # ------------------------------------------------------------------
    # XPolicyLab protocol
    # ------------------------------------------------------------------
    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        if isinstance(obs_list, dict):
            obs_list = [obs_list]
        if not obs_list:
            raise ValueError("update_obs_batch received an empty observation list.")
        self._batch = {}
        self._order = []
        for index, obs in enumerate(obs_list):
            env_idx = int(obs.get("env_idx", index))
            self._batch[env_idx] = self._encode_obs(obs)
            self._order.append(env_idx)

    def get_action(self):
        env_idx = self._order[0] if self._order else 0
        return self.get_action_batch([env_idx])[0]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = list(self._order)
        elif isinstance(env_idx_list, np.ndarray):
            env_idx_list = env_idx_list.reshape(-1).tolist()
        elif isinstance(env_idx_list, (int, np.integer)):
            env_idx_list = [int(env_idx_list)]
        else:
            env_idx_list = list(env_idx_list)

        if not self._batch:
            raise ValueError("No observation available. Call update_obs(_batch) before get_action.")

        env_idx_list = [int(e) for e in env_idx_list]
        encoded = [self._batch[e] for e in env_idx_list]

        if self.allow_dummy_policy or self.policy is None:
            actions_batch = [self._zero_actions() for _ in encoded]
        else:
            actions_batch = self._infer_batch(encoded, env_idx_list)

        self._step_id += 1
        return actions_batch

    def reset(self):
        self._step_id = 0
        self._batch = {}
        self._order = []

    # ------------------------------------------------------------------
    # Inference + delta -> absolute reconstruction
    # ------------------------------------------------------------------
    def _seed_for(self, env_idx: int) -> int:
        key = f"{self.base_seed}_{env_idx}_{self._step_id}"
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)

    def _infer_batch(self, encoded: list[dict], env_idx_list: list[int]) -> list[list[dict]]:
        video = np.stack([e["video"] for e in encoded], axis=0)        # [B, V, H, W, C]
        proprios = np.stack([e["proprio"] for e in encoded], axis=0)    # [B, 16]
        prompts = [e["prompt"] for e in encoded]
        seeds = [self._seed_for(e_idx) for e_idx in env_idx_list]

        actions, _ = self.policy.infer_batch(
            video=video,
            proprios=proprios,
            prompts=prompts,
            seeds=seeds,
            cfg=self.cfg_scale,
        )  # actions: [B, Ta, action_dim] delta

        out: list[list[dict]] = []
        for b, enc in enumerate(encoded):
            abs_poses = self._integrate_deltas(actions[b], enc["anchor"])
            out.append(self._poses_to_action_dicts(abs_poses))
        return out

    def _integrate_deltas(self, delta_actions: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        """Integrate per-step delta actions into absolute EE poses.

        delta_actions: [Ta, action_dim] = per arm [dxyz3, daxisangle3, dgrip1].
        anchor:        [16] current absolute proprio (xyz, quat_wxyz, grip per arm).
        Returns absolute poses [Ta, num_arms*8] = per arm [xyz3, quat_wxyz4, grip1].

        Integration (action_skip=1, frame-wise deltas):
            pos[k]  = anchor_pos + sum(dxyz[0..k])
            R[k]    = dR[k] @ ... @ dR[0] @ anchor_R   (left-multiplied accumulation)
            grip[k] = clip(anchor_grip + sum(dgrip[0..k]), 0, 1)
        """
        ta = delta_actions.shape[0]
        n_exec = ta if self.replan_steps is None else min(self.replan_steps, ta)
        out = np.zeros((n_exec, self.num_arms * _PROPRIO_ARM_DIM), dtype=np.float32)

        for arm in range(self.num_arms):
            a_off = arm * _ACTION_ARM_DIM
            p_off = arm * _PROPRIO_ARM_DIM
            cur_pos = anchor[p_off : p_off + 3].astype(np.float64).copy()
            cur_R = Rotation.from_quat(
                anchor[p_off + 3 : p_off + 7][[1, 2, 3, 0]].astype(np.float64)  # wxyz -> xyzw
            ).as_matrix()
            cur_grip = float(anchor[p_off + 7])

            for k in range(n_exec):
                d = delta_actions[k, a_off : a_off + _ACTION_ARM_DIM].astype(np.float64)
                cur_pos = cur_pos + d[:3]
                delta_R = Rotation.from_rotvec(d[3:6]).as_matrix()
                cur_R = delta_R @ cur_R
                cur_grip = float(np.clip(cur_grip + d[6], 0.0, 1.0))

                quat_xyzw = Rotation.from_matrix(cur_R).as_quat()
                quat_wxyz_train = _canonical_quat_wxyz(quat_xyzw[[3, 0, 1, 2]])
                # cur_R is in the training frame (anchor was reframed); convert
                # the absolute orientation back to the env frame for the env.
                out[k, p_off : p_off + 3] = cur_pos
                out[k, p_off + 3 : p_off + 7] = _reframe_quat_train_to_env(quat_wxyz_train)
                out[k, p_off + 7] = cur_grip
        return out

    def _poses_to_action_dicts(self, abs_poses: np.ndarray) -> list[dict]:
        """Convert [Ta, num_arms*8] absolute poses into XPolicyLab ee action dicts."""
        action_list = []
        for row in abs_poses:
            if self.num_arms == 1:
                action_list.append(
                    {
                        "ee_pose": row[:_EE_POSE_DIM].astype(np.float32),
                        "ee_joint_state": row[7:8].astype(np.float32),
                    }
                )
            else:
                action_list.append(
                    {
                        "left_ee_pose": row[:_EE_POSE_DIM].astype(np.float32),
                        "left_ee_joint_state": row[7:8].astype(np.float32),
                        "right_ee_pose": row[8 : 8 + _EE_POSE_DIM].astype(np.float32),
                        "right_ee_joint_state": row[15:16].astype(np.float32),
                    }
                )
        return action_list

    def _zero_actions(self) -> list[dict]:
        n = self.replan_steps or 1
        identity = np.zeros(self.num_arms * _PROPRIO_ARM_DIM, dtype=np.float32)
        for arm in range(self.num_arms):
            identity[arm * _PROPRIO_ARM_DIM + 3] = 1.0  # quat w = 1
        return self._poses_to_action_dicts(np.tile(identity, (n, 1)))


