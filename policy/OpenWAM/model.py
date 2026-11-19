"""XPolicyLab adapter for the OpenWAM policy (RoboDojo dual-arm arx_x5, EE control).

Loads the OpenWAM checkpoint in-process through the standard deploy path
(``openwam.deploy.server.build_server_from_config`` — the same construction
the JSON WebSocket server and ``scripts/verify_batch_equivalence.py`` use),
then serves XPolicyLab's batched eval protocol. Every ``get_action_batch``
call stacks all running envs into ONE ``engine.generate_batch`` forward pass
(true batch inference, verified contamination-free on real weights).

Coordinate contract (must mirror training — ``openwam/dataloader/robodojo.py``):

    obs    env-relative world pose [xyz, quat wxyz] per arm + gripper [0, 1]
        -> robot-base frame       (env_relative_world_to_robot_base, dual-X5
                                   base constants from robodojo_contract)
        -> EEF20 state            [L xyz3, L rot6d6, L grip1,
                                   R xyz3, R rot6d6, R grip1]

    action model emits absolute EEF20 in robot-base frame (physical units)
        -> per-arm pose + gripper (eef20_to_arms)
        -> env-relative world     (robot_base_to_env_relative_world)
        -> XPolicyLab ee dicts    {left_ee_pose, left_ee_joint_state,
                                   right_ee_pose, right_ee_joint_state}

Correctness-first settings are re-forced at load time regardless of the yaml:
dit_cache off, compile off, decode_video off, sync executor.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info

POLICY_DIR = Path(__file__).resolve().parent
# Vendored OpenWAM source tree (weights excluded).
VENDORED_OPENWAM_ROOT = POLICY_DIR / "OpenWAM"
CHECKPOINTS_DIR = POLICY_DIR / "checkpoints"

_EXPECTED_ARM_DIMS = [6, 6]
_EXPECTED_EE_DIMS = [1, 1]
_EEF20_DIM = 20
_QUATERNION_ATOL = 1e-6
# Isaac occasionally reports a closed gripper as ~-3e-17.
_GRIPPER_ATOL = 1e-8
# Fixed obs -> OpenWAM payload camera mapping (matches training camera_layout).
_CAMERA_TO_PAYLOAD = {
    "cam_head": "head_camera",
    "cam_left_wrist": "left_wrist_camera",
    "cam_right_wrist": "right_wrist_camera",
}


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


def _resolve_ckpt_dir(model_cfg: dict) -> Path:
    """Resolve the OpenWAM checkpoint directory via the shared XPolicyLab rules.

    Precedence: explicit ``ckpt_dir`` / path-like ``ckpt_name`` >
    ``checkpoints/<bench>-<ckpt>-<env>-<action>-<seed>/`` >
    ``checkpoints/<ckpt_name>/``. The directory must contain ``config.yaml``.
    """
    root = resolve_checkpoint_root(
        model_cfg,
        CHECKPOINTS_DIR,
        policy_dir=POLICY_DIR,
        explicit_keys=("ckpt_dir", "checkpoint_path", "ckpt_path", "model_dir"),
        must_exist=True,
    )
    if not (root / "config.yaml").is_file():
        raise FileNotFoundError(f"checkpoint dir has no config.yaml: {root}")
    return root


def _resolve_openwam_root(openwam_root: Any) -> Path:
    """Resolve the OpenWAM source root and make it win the import resolution.

    Default is the vendored copy next to this file. The root is prepended to
    sys.path unconditionally: the policy conda env typically carries an
    editable install pointing at the external dev repo, and the vendored /
    explicitly-configured tree must take precedence over it.
    """
    root = VENDORED_OPENWAM_ROOT if _is_none_like(openwam_root) else Path(str(openwam_root)).expanduser()
    if not (root / "openwam" / "__init__.py").is_file():
        raise FileNotFoundError(f"openwam package not found under openwam_root: {root}")
    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)
    if "openwam" in sys.modules:
        loaded = Path(getattr(sys.modules["openwam"], "__file__", "") or "").resolve()
        if not str(loaded).startswith(str(root.resolve())):
            raise RuntimeError(
                f"openwam was already imported from {loaded}, not from openwam_root {root}; "
                "start the server in a fresh process."
            )
    return root


def _decode_instruction(value: Any, fallback: str) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if isinstance(value, np.ndarray):
        value = value.item() if value.shape == () else (value.reshape(-1)[0] if value.size else None)
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8")
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _validated_pose(value: Any, name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64).reshape(-1)
    if pose.shape != (7,):
        raise ValueError(f"{name} must have shape (7,) [xyz, quat wxyz], got {pose.shape}")
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must contain only finite values")
    norm = float(np.linalg.norm(pose[3:7]))
    if norm < 1e-8:
        raise ValueError(f"{name} quaternion must be non-zero")
    # Isaac emits unit wxyz quaternions; normalize instead of rejecting so the
    # synthetic debug client (which sends placeholder np.ones(7) poses) can
    # still exercise the protocol. Large deviations are logged, not fatal.
    if abs(norm - 1.0) > 1e-3:
        print(f"[OpenWAM] warning: {name} quaternion norm {norm:.6g} != 1; normalizing.")
    pose = pose.copy()
    pose[3:7] /= norm
    return pose


def _validated_gripper(value: Any, name: str) -> np.ndarray:
    gripper = np.asarray(value, dtype=np.float64).reshape(-1)
    if gripper.shape != (1,):
        raise ValueError(f"{name} must have shape (1,), got {gripper.shape}")
    if not np.all(np.isfinite(gripper)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any((gripper < -_GRIPPER_ATOL) | (gripper > 1.0 + _GRIPPER_ATOL)):
        raise ValueError(f"{name} gripper value must be within [0, 1], got {gripper}")
    return np.clip(gripper, 0.0, 1.0)


def _to_pil(value: Any, ctx: str):
    """Standardize a decoded obs camera frame into an RGB PIL image."""
    from PIL import Image

    if isinstance(value, Image.Image):
        return value if value.mode == "RGB" else value.convert("RGB")
    arr = np.asarray(value)
    if arr.ndim != 3 or arr.shape[2] < 3 or 0 in arr.shape:
        raise ValueError(f"{ctx}: expected decoded (H, W, >=3) image array, got shape {arr.shape}")
    arr = arr[:, :, :3]
    if np.issubdtype(arr.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(arr)) <= 1.5 else 1.0
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(arr), mode="RGB")


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.benchmark = str(self.model_cfg.get("openwam_benchmark") or "robodojo").lower()
        if self.benchmark not in ("robodojo", "robotwin"):
            raise ValueError("openwam_benchmark must be robodojo or robotwin.")

        action_type = self.model_cfg.get("action_type") or "ee"
        if action_type != "ee":
            raise ValueError(f"OpenWAM RoboDojo is an EE-space policy; action_type must be 'ee', got {action_type!r}.")
        env_cfg_type = self.model_cfg.get("env_cfg_type")
        if not env_cfg_type:
            raise ValueError("env_cfg_type is required for the OpenWAM adapter.")
        dim_info = get_robot_action_dim_info(env_cfg_type)
        self.action_type = action_type
        self.robot_action_dim_info = dim_info
        if list(dim_info.get("arm_dim") or []) != _EXPECTED_ARM_DIMS or list(dim_info.get("ee_dim") or []) != _EXPECTED_EE_DIMS:
            raise ValueError(
                "OpenWAM's RoboDojo checkpoint is dual-X5 only (arm_dim [6, 6], ee_dim [1, 1]); "
                f"env_cfg_type={env_cfg_type!r} resolves to {dim_info!r}."
            )

        # null -> vendored copy at policy/OpenWAM/OpenWAM (X_WAM-style layout).
        self.openwam_root = _resolve_openwam_root(self.model_cfg.get("openwam_root"))

        # Canonical training-side conversion + calibration (single source of truth).
        from openwam.dataloader.robodojo_contract import arx_x5_calibration
        from openwam.dataloader.transforms.multiview import format_prompt_for_inference
        from openwam.dataloader.utils import poses as _poses

        self._poses = _poses
        self._format_prompt = format_prompt_for_inference
        self._calibration = arx_x5_calibration() if self.benchmark == "robodojo" else None

        self.default_instruction = str(self.model_cfg.get("default_instruction") or "follow the instruction")
        replan = self.model_cfg.get("replan_steps")
        self.replan_steps = None if _is_none_like(replan) else int(replan)
        if self.replan_steps is not None and self.replan_steps <= 0:
            raise ValueError(f"replan_steps must be positive or null, got {self.replan_steps}")

        # Per-env stored observations (keyed by env_idx, order preserved).
        self._batch: dict[int, dict] = {}
        self._order: list[int] = []

        self.allow_dummy_policy = _is_true(self.model_cfg.get("allow_dummy_policy", False))
        self._engine = None
        self._wam_policy = None
        self._preprocessor = None
        if self.allow_dummy_policy:
            print("[OpenWAM] allow_dummy_policy=true; checkpoint loading skipped (protocol debug only).")
            return

        ckpt_dir = str(_resolve_ckpt_dir(self.model_cfg))

        # OpenWAM's own deploy defaults; the correctness-critical keys are
        # force-overridden below, so no separate pinned eval yaml is needed.
        deploy_config = self.model_cfg.get("openwam_deploy_config")
        if _is_none_like(deploy_config):
            deploy_config = str(self.openwam_root / "configs" / "deploy.yaml")
        device = str(self.model_cfg.get("device") or "cuda")

        from omegaconf import OmegaConf

        from openwam.deploy.server import _load_deploy_yaml, build_server_from_config

        deploy_cfg = _load_deploy_yaml(deploy_config)
        # Correctness-first: batch inference forbids the single-stream /
        # shape-sensitive acceleration paths. Force them off even if the yaml
        # drifts; engine.generate_batch fails fast if these were re-enabled.
        OmegaConf.update(deploy_cfg, "optimization.dit_cache.enabled", False, merge=False)
        OmegaConf.update(deploy_cfg, "optimization.compile.enabled", False, merge=False)
        OmegaConf.update(deploy_cfg, "optimization.decode_video", False, merge=False)
        OmegaConf.update(deploy_cfg, "inference.inference_mode", "sync", merge=False)

        print(f"[OpenWAM] loading checkpoint from {ckpt_dir} on {device} ...")
        server = build_server_from_config(deploy_cfg, ckpt_dir, device=device)
        server._init_policy()
        self._engine = server.engine
        self._wam_policy = server._policy  # for the binary-dim legality projection
        self._preprocessor = server._obs_preprocessor

        if self.benchmark == "robotwin":
            saved = OmegaConf.load(str(Path(ckpt_dir) / "config.yaml"))
            if OmegaConf.select(saved, "dataloader.type") != "robotwin":
                raise ValueError("Expected a RoboTwin checkpoint.")
            if self._engine.architecture.repr_contract.get("representation") != "eef":
                raise ValueError("RoboTwin inference requires an EEF checkpoint.")
            if self.replan_steps is None:
                self.replan_steps = 1

        if self.replan_steps is None:
            horizon = OmegaConf.select(server.cfg, "inference.inference_horizon", default=None)
            self.replan_steps = None if horizon is None else int(horizon)

        resolved = {
            "denoise_steps": OmegaConf.select(server.cfg, "inference.denoise_steps"),
            "denoise_mode": OmegaConf.select(server.cfg, "inference.denoise_mode"),
            "num_frames": OmegaConf.select(server.cfg, "inference.num_frames"),
            "video_num_frames": OmegaConf.select(server.cfg, "inference.video_num_frames"),
            "replan_steps": self.replan_steps or "full chunk",
            "dit_cache": OmegaConf.select(server.cfg, "optimization.dit_cache.enabled"),
            "compile": OmegaConf.select(server.cfg, "optimization.compile.enabled"),
        }
        print(f"[OpenWAM] initialized | {resolved}")

    # ------------------------------------------------------------------
    # Observation: obs dict -> OpenWAM conditions (world -> base -> EEF20)
    # ------------------------------------------------------------------
    def _state_to_eef20(self, state: Mapping) -> np.ndarray:
        if self.benchmark == "robotwin":
            from .robotwin import state_to_eef20

            return state_to_eef20(state)
        missing = [k for k in ("left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state") if k not in state]
        if missing:
            raise KeyError(f"obs['state'] is missing required field(s): {', '.join(missing)}")
        left = self._calibration["arms"]["left"]
        right = self._calibration["arms"]["right"]
        left_base = self._poses.env_relative_world_to_robot_base(
            _validated_pose(state["left_ee_pose"], "left_ee_pose"),
            left["base_pos_relative_to_env_origin"],
            left["base_quat_wxyz"],
        )
        right_base = self._poses.env_relative_world_to_robot_base(
            _validated_pose(state["right_ee_pose"], "right_ee_pose"),
            right["base_pos_relative_to_env_origin"],
            right["base_quat_wxyz"],
        )
        eef20 = self._poses.arms_to_eef20(
            left_base,
            _validated_gripper(state["left_ee_joint_state"], "left_ee_joint_state"),
            right_base,
            _validated_gripper(state["right_ee_joint_state"], "right_ee_joint_state"),
        )
        if eef20.shape != (_EEF20_DIM,):
            raise RuntimeError(f"state conversion produced {eef20.shape}, expected ({_EEF20_DIM},)")
        return eef20

    def _encode_obs(self, obs: Mapping) -> dict:
        vision = obs.get("vision")
        if not isinstance(vision, Mapping):
            raise ValueError("obs['vision'] must be a mapping of camera name -> {'color': image}")
        images = {}
        for camera_name, payload_name in _CAMERA_TO_PAYLOAD.items():
            if camera_name == "cam_head" and camera_name not in vision:
                camera_name = "cam_high"
            camera = vision.get(camera_name)
            if not isinstance(camera, Mapping) or "color" not in camera:
                raise KeyError(f"missing required camera vision/{camera_name}/color")
            images[payload_name] = _to_pil(camera["color"], ctx=f"vision/{camera_name}/color")

        state = obs.get("state")
        if not isinstance(state, Mapping):
            raise ValueError("obs['state'] must be a mapping")
        eef20 = self._state_to_eef20(state)

        instruction = _decode_instruction(
            obs.get("instruction", obs.get("task_instruction")), self.default_instruction
        )
        return {
            "images": images,
            "prompt": self._format_prompt(instruction),
            "state": eef20.astype(np.float32).tolist(),
        }

    def _conditions(self, payload: dict) -> dict:
        """OpenWAM server-equivalent preprocessing: compose multiview + wrap."""
        obs = self._preprocessor.preprocess(dict(payload))
        cond = {
            "observation": obs,
            "first_frame_image": [obs["image"]],
            "prompt": obs["prompt"],
        }
        if obs.get("state") is not None:
            cond["proprio"] = obs["state"]
        return cond

    # ------------------------------------------------------------------
    # Action: EEF20 chunk (base frame) -> env-relative world ee dicts
    # ------------------------------------------------------------------
    def _eef20_chunk_to_native(self, chunk: np.ndarray) -> list[dict]:
        if self.benchmark == "robotwin":
            from .robotwin import eef20_to_steps

            return eef20_to_steps(chunk)
        left_pose, left_grip, right_pose, right_grip = self._poses.eef20_to_arms(np.asarray(chunk, dtype=np.float64))
        left = self._calibration["arms"]["left"]
        right = self._calibration["arms"]["right"]
        left_world = self._poses.robot_base_to_env_relative_world(
            left_pose, left["base_pos_relative_to_env_origin"], left["base_quat_wxyz"]
        )
        right_world = self._poses.robot_base_to_env_relative_world(
            right_pose, right["base_pos_relative_to_env_origin"], right["base_quat_wxyz"]
        )
        left_grip = np.clip(left_grip, 0.0, 1.0)
        right_grip = np.clip(right_grip, 0.0, 1.0)
        return [
            {
                "left_ee_pose": left_world[t].astype(np.float32),
                "left_ee_joint_state": left_grip[t].astype(np.float32),
                "right_ee_pose": right_world[t].astype(np.float32),
                "right_ee_joint_state": right_grip[t].astype(np.float32),
            }
            for t in range(left_world.shape[0])
        ]

    def _hold_position_chunk(self, payload: dict) -> list[dict]:
        """Dummy-mode chunk: echo the current pose (exercises the full frame chain)."""
        steps = self.replan_steps or 8
        eef20 = np.asarray(payload["state"], dtype=np.float64)
        return self._eef20_chunk_to_native(np.tile(eef20, (steps, 1)))

    # ------------------------------------------------------------------
    # XPolicyLab protocol
    # ------------------------------------------------------------------
    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        if isinstance(obs_list, Mapping):
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
        env_idx_list = [int(e) for e in env_idx_list]
        if not env_idx_list:
            raise ValueError("get_action_batch received an empty env_idx_list.")

        missing = [e for e in env_idx_list if e not in self._batch]
        if missing:
            raise ValueError(f"No stored observation for env_idx {missing}; call update_obs_batch first.")
        payloads = [self._batch[e] for e in env_idx_list]

        if self.allow_dummy_policy:
            return [self._hold_position_chunk(p) for p in payloads]

        conditions = [self._conditions(p) for p in payloads]
        if self.benchmark == "robotwin":
            for condition in conditions:
                condition["seed"] = int(self.model_cfg.get("seed", 42))
        result = self._engine.generate_batch(conditions)
        actions = result["actions"]
        if hasattr(actions, "cpu"):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if actions.ndim != 3 or actions.shape[0] != len(env_idx_list) or actions.shape[2] != _EEF20_DIM:
            raise RuntimeError(
                f"generate_batch returned actions of shape {actions.shape}; "
                f"expected (B={len(env_idx_list)}, T, {_EEF20_DIM})."
            )
        # Same final legality projection the OpenWAM server applies (no-op when
        # the checkpoint declares no binary command dims).
        actions = self._wam_policy._project_binary_dims(actions)

        n_exec = actions.shape[1]
        if self.replan_steps is not None:
            n_exec = min(self.replan_steps, n_exec)
        return [self._eef20_chunk_to_native(actions[b, :n_exec]) for b in range(actions.shape[0])]

    def reset(self):
        self._batch = {}
        self._order = []
