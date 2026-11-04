"""XPolicyLab adapter for FlexWAM (RoboDojo dual-arm EE, delta rot6d).

The vendored tree is ``policy/FlexWAM/flexwam``. This module only translates
the XPolicyLab observation / action contract into that tree's deploy policy.

Real-robot contract for ``arx_x5``, ``piper`` and ``piper_x``. Recorded and live
poses are already in each arm's base, so the base transform is identity.
FlexWAM stores the first two *rows* of the rotation matrix (Zhou et al.),
which is what ``deploy/robot/flexiv.py`` and the training delta transform use.

    obs    robot-base pose [xyz, quat wxyz] + gripper [0, 1]
        -> EEF20 [L xyz3, L rot6d-rows6, L grip1, R xyz3, R rot6d-rows6, R grip1]

    the checkpoint predicts a bimanual pose6d delta
        -> integrate to absolute EEF20 in the robot base
        -> ee dicts in that same base frame

``infer_action`` accepts one environment at a time. ``get_action_batch`` loops.
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
VENDORED_FLEXWAM_ROOT = POLICY_DIR / "flexwam"
CHECKPOINTS_DIR = POLICY_DIR / "checkpoints"

_EXPECTED_ARM_DIMS = [6, 6]
_EXPECTED_EE_DIMS = [1, 1]
_EEF20_DIM = 20
_GRIPPER_ATOL = 1e-8
_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

# cam_head is cam_high in the LeRobot export. Deploy camera names follow the
# task yaml: global, left_wrist, right_wrist.
_CAMERA_TO_CLIENT = {
    "cam_head": "observation/global",
    "cam_left_wrist": "observation/left_wrist",
    "cam_right_wrist": "observation/right_wrist",
}
_TASK_BY_ENV = {
    "arx_x5": "wan/robodojo_arx5",
    "piper": "wan/robodojo_piper",
    "piper_x": "wan/robodojo_piperx",
}
# Real recordings are already in the per-arm base. Identity keeps that frame.
_IDENTITY_ARM = {"pos": (0.0, 0.0, 0.0), "quat": (1.0, 0.0, 0.0, 0.0)}
_CALIBRATIONS = {
    name: {"frame": "identity_robot_base", "arms": {"left": _IDENTITY_ARM, "right": _IDENTITY_ARM}}
    for name in ("arx_x5", "piper", "piper_x")
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


def _optional_int(value: Any) -> int | None:
    if _is_none_like(value):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if _is_none_like(value):
        return None
    return float(value)


def _quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Unit wxyz quaternion to a rotation matrix. Real part is first, matching FlexWAM."""
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        raise ValueError("quaternion must be non-zero")
    q = q / norm
    r, i, j, k = q
    two_s = 2.0 / float(np.dot(q, q))
    return np.array(
        [
            [1.0 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r)],
            [two_s * (i * j + k * r), 1.0 - two_s * (i * i + k * k), two_s * (j * k - i * r)],
            [two_s * (i * k - j * r), two_s * (j * k + i * r), 1.0 - two_s * (i * i + j * j)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrix to a unit wxyz quaternion with non-negative w."""
    rotation = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quat = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        quat = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            ]
        )
    elif rotation[1, 1] > rotation[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        quat = np.array(
            [
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            ]
        )
    else:
        scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
        quat = np.array(
            [
                (rotation[1, 0] - rotation[0, 1]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
            ]
        )
    quat = quat / np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat = -quat
    return quat


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64).reshape(4)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float64).reshape(4)
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _rot6d_rows_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    raw = np.asarray(rot6d, dtype=np.float64).reshape(6)
    row0 = raw[0:3]
    row1 = raw[3:6]
    row0_norm = float(np.linalg.norm(row0))
    if row0_norm < 1e-8:
        raise ValueError("rot6d first row is degenerate")
    row0 = row0 / row0_norm
    row1 = row1 - np.dot(row0, row1) * row0
    row1_norm = float(np.linalg.norm(row1))
    if row1_norm < 1e-8:
        raise ValueError("rot6d second row is degenerate")
    row1 = row1 / row1_norm
    row2 = np.cross(row0, row1)
    return np.stack([row0, row1, row2], axis=0)


def _matrix_to_rot6d_rows(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    return np.concatenate([mat[0], mat[1]]).astype(np.float32)


def _env_to_base(pose: np.ndarray, base_pos: np.ndarray, base_quat: np.ndarray) -> np.ndarray:
    rotation = _quat_wxyz_to_matrix(base_quat)
    position = rotation.T @ (pose[:3] - base_pos)
    inverse_quat = base_quat * np.array([1.0, -1.0, -1.0, -1.0])
    quaternion = _quat_multiply(inverse_quat, pose[3:7])
    quaternion = quaternion / np.linalg.norm(quaternion)
    return np.concatenate([position, quaternion])


def _base_to_env(pose: np.ndarray, base_pos: np.ndarray, base_quat: np.ndarray) -> np.ndarray:
    rotation = _quat_wxyz_to_matrix(base_quat)
    position = rotation @ pose[:3] + base_pos
    quaternion = _quat_multiply(base_quat, pose[3:7])
    quaternion = quaternion / np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return np.concatenate([position, quaternion])


def _arm10_from_pose(pose: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    rot6d = _matrix_to_rot6d_rows(_quat_wxyz_to_matrix(pose[3:7]))
    return np.concatenate([pose[:3].astype(np.float32), rot6d, gripper.astype(np.float32)]).astype(np.float32)


def _pose_from_arm10(arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quaternion = _matrix_to_quat_wxyz(_rot6d_rows_to_matrix(arm[3:9]))
    pose = np.concatenate([arm[:3].astype(np.float64), quaternion])
    return pose, np.asarray(arm[9:10], dtype=np.float32)


def _resolve_flexwam_root(flexwam_root: Any) -> Path:
    root = VENDORED_FLEXWAM_ROOT if _is_none_like(flexwam_root) else Path(str(flexwam_root)).expanduser()
    if not (root / "src" / "flexwam" / "__init__.py").is_file():
        raise FileNotFoundError(f"flexwam package not found under flexwam_root: {root}")
    for entry in (root, root / "src"):
        entry_str = str(entry)
        if entry_str in sys.path:
            sys.path.remove(entry_str)
        sys.path.insert(0, entry_str)
    loaded = sys.modules.get("flexwam")
    if loaded is not None:
        loaded_path = Path(getattr(loaded, "__file__", "") or "").resolve()
        if not str(loaded_path).startswith(str((root / "src").resolve())):
            raise RuntimeError(
                f"flexwam was already imported from {loaded_path}, not from {root}; "
                "start the server in a fresh process."
            )
    loaded_deploy = sys.modules.get("deploy")
    if loaded_deploy is not None:
        loaded_path = Path(getattr(loaded_deploy, "__file__", "") or "").resolve()
        if not str(loaded_path).startswith(str(root.resolve())):
            raise RuntimeError(
                f"deploy was already imported from {loaded_path}, not from {root}; "
                "start the server in a fresh process."
            )
    return root


def _step_index(path: Path) -> int:
    name = path.stem
    marker = "step_"
    if marker not in name:
        return -1
    digits = name.split(marker, 1)[1]
    return int(digits) if digits.isdigit() else -1


def _find_weight_file(root: Path) -> Path:
    if root.is_file():
        if root.suffix != ".pt":
            raise FileNotFoundError(f"checkpoint file must be a .pt weight file, got {root}")
        return root
    found: list[Path] = []
    for directory in (root / "checkpoints" / "weights", root / "weights", root):
        if directory.is_dir():
            found.extend(path for path in directory.glob("*.pt") if path.is_file())
    if not found:
        raise FileNotFoundError(
            f"no FlexWAM weight file (*.pt) under {root}. "
            "Expected checkpoints/weights/step_*.pt next to dataset_stats.json."
        )
    found.sort(key=lambda path: (_step_index(path), path.stat().st_mtime, path.name))
    return found[-1]


def _find_dataset_stats(model_cfg: dict, weight_file: Path, ckpt_root: Path) -> Path:
    explicit = model_cfg.get("dataset_stats_path")
    if not _is_none_like(explicit):
        path = Path(str(explicit)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"dataset_stats_path does not exist: {path}")
        return path
    candidates = [
        ckpt_root / "dataset_stats.json",
        weight_file.parent / "dataset_stats.json",
        weight_file.parent.parent / "dataset_stats.json",
        weight_file.parent.parent.parent / "dataset_stats.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    checked = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"dataset_stats.json not found. Checked:\n  {checked}")


def _resolve_checkpoint(model_cfg: dict) -> tuple[Path, Path]:
    explicit_file = model_cfg.get("checkpoint_path")
    if not _is_none_like(explicit_file):
        weight_file = Path(str(explicit_file)).expanduser()
        if not weight_file.is_file():
            raise FileNotFoundError(f"checkpoint_path does not exist: {weight_file}")
        return weight_file, _find_dataset_stats(model_cfg, weight_file, weight_file.parent)
    root = resolve_checkpoint_root(
        model_cfg,
        CHECKPOINTS_DIR,
        policy_dir=POLICY_DIR,
        explicit_keys=("ckpt_dir", "checkpoint_path", "ckpt_path", "model_dir"),
        must_exist=True,
    )
    weight_file = _find_weight_file(root)
    return weight_file, _find_dataset_stats(model_cfg, weight_file, root if root.is_dir() else weight_file.parent)


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
    if abs(norm - 1.0) > 1e-3:
        print(f"[FlexWAM] warning: {name} quaternion norm {norm:.6g} != 1; normalizing.")
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
    return np.clip(gripper, 0.0, 1.0).astype(np.float32)


def _to_hwc_uint8(value: Any, ctx: str) -> np.ndarray:
    """Accept an already decoded RGB frame. Do not decode JPEG bytes here."""
    if isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype == np.uint8:
        raise ValueError(f"{ctx}: received encoded image bytes. The policy server must decode before model.py.")
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] < 3 or 0 in array.shape:
        raise ValueError(f"{ctx}: expected decoded (H, W, >=3) image array, got shape {array.shape}")
    array = array[:, :, :3]
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(array)) <= 1.5 else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = dict(model_cfg)

        action_type = self.model_cfg.get("action_type") or "ee"
        if action_type != "ee":
            raise ValueError(f"FlexWAM RoboDojo is an EE-space policy; action_type must be 'ee', got {action_type!r}.")
        env_cfg_type = self.model_cfg.get("env_cfg_type")
        if not env_cfg_type:
            raise ValueError("env_cfg_type is required for the FlexWAM adapter.")
        if env_cfg_type not in _CALIBRATIONS:
            raise ValueError(
                f"FlexWAM RoboDojo supports env_cfg_type {sorted(_CALIBRATIONS)}, got {env_cfg_type!r}."
            )
        dim_info = get_robot_action_dim_info(env_cfg_type)
        if list(dim_info.get("arm_dim") or []) != _EXPECTED_ARM_DIMS or list(dim_info.get("ee_dim") or []) != _EXPECTED_EE_DIMS:
            raise ValueError(
                "FlexWAM's RoboDojo checkpoint is a dual 6-DoF gripper robot "
                f"(arm_dim [6, 6], ee_dim [1, 1]); env_cfg_type={env_cfg_type!r} resolves to {dim_info!r}."
            )

        self.env_cfg_type = str(env_cfg_type)
        self.calibration = _CALIBRATIONS[self.env_cfg_type]
        self.flexwam_root = _resolve_flexwam_root(self.model_cfg.get("flexwam_root"))
        self.default_instruction = str(self.model_cfg.get("default_instruction") or "follow the instruction")
        self.replan_steps = _optional_int(self.model_cfg.get("replan_steps"))
        if self.replan_steps is not None and self.replan_steps <= 0:
            raise ValueError(f"replan_steps must be positive or null, got {self.replan_steps}")
        self.delta = _is_true(self.model_cfg.get("delta", True))
        self._batch: dict[int, dict] = {}
        self._order: list[int] = []
        self.allow_dummy_policy = _is_true(self.model_cfg.get("allow_dummy_policy", False))
        self._policy = None

        print(
            f"[FlexWAM] env_cfg_type={self.env_cfg_type} frame={self.calibration['frame']} "
            f"delta={self.delta}"
        )
        if self.allow_dummy_policy:
            print("[FlexWAM] allow_dummy_policy=true; checkpoint loading skipped (protocol debug only).")
            return

        task = self.model_cfg.get("task")
        if _is_none_like(task):
            task = _TASK_BY_ENV[self.env_cfg_type]
        weight_file, stats_path = _resolve_checkpoint(self.model_cfg)
        device = str(self.model_cfg.get("device") or "cuda")

        from deploy.policy import FlexWAMPolicy
        from flexwam.config import compose_task
        from flexwam.utils.config_resolvers import register_default_resolvers

        register_default_resolvers()
        cfg = compose_task(str(task), config_dir=self.flexwam_root / "configs")
        action_horizon = int(cfg.data.train.num_frames) - 1
        num_inference_steps = _optional_int(self.model_cfg.get("num_inference_steps"))
        if num_inference_steps is None:
            num_inference_steps = int(getattr(cfg, "eval_num_inference_steps", 10))
        if action_horizon <= 0 or num_inference_steps <= 0:
            raise ValueError(f"Invalid horizon {action_horizon} or steps {num_inference_steps}")

        print(f"[FlexWAM] loading task={task} weights={weight_file} stats={stats_path} on {device}")
        self._policy = FlexWAMPolicy(
            cfg=cfg,
            checkpoint_path=weight_file,
            dataset_stats_path=stats_path,
            device=device,
            action_horizon=action_horizon,
            num_inference_steps=num_inference_steps,
            sigma_shift=_optional_float(self.model_cfg.get("sigma_shift")),
            seed=_optional_int(self.model_cfg.get("seed")),
            default_prompt=self.default_instruction,
            prompt_template=_PROMPT,
            log_full_io=False,
            action_representation="auto",
            delta_joint_actions=self.delta,
            robot_type="dual_ee",
            cuda_graph=_is_true(self.model_cfg.get("cuda_graph", True)),
        )
        self.model = self._policy
        if self.replan_steps is None:
            self.replan_steps = action_horizon
        print(
            f"[FlexWAM] initialized | horizon={action_horizon} "
            f"replan_steps={self.replan_steps} denoise_steps={num_inference_steps}"
        )

    def _arm_constants(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        spec = self.calibration["arms"][arm]
        return np.asarray(spec["pos"], dtype=np.float64), np.asarray(spec["quat"], dtype=np.float64)

    def _state_to_eef20(self, state: Mapping) -> np.ndarray:
        missing = [key for key in ("left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state") if key not in state]
        if missing:
            raise KeyError(f"obs['state'] is missing required field(s): {', '.join(missing)}")
        left_pos, left_quat = self._arm_constants("left")
        right_pos, right_quat = self._arm_constants("right")
        left_base = _env_to_base(_validated_pose(state["left_ee_pose"], "left_ee_pose"), left_pos, left_quat)
        right_base = _env_to_base(_validated_pose(state["right_ee_pose"], "right_ee_pose"), right_pos, right_quat)
        eef20 = np.concatenate(
            [
                _arm10_from_pose(left_base, _validated_gripper(state["left_ee_joint_state"], "left_ee_joint_state")),
                _arm10_from_pose(right_base, _validated_gripper(state["right_ee_joint_state"], "right_ee_joint_state")),
            ]
        ).astype(np.float32)
        if eef20.shape != (_EEF20_DIM,):
            raise RuntimeError(f"state conversion produced {eef20.shape}, expected ({_EEF20_DIM},)")
        return eef20

    def _encode_obs(self, obs: Mapping) -> dict:
        vision = obs.get("vision")
        if not isinstance(vision, Mapping):
            raise ValueError("obs['vision'] must be a mapping of camera name -> {'color': image}")
        payload = {}
        for camera_name, client_key in _CAMERA_TO_CLIENT.items():
            camera = vision.get(camera_name)
            if not isinstance(camera, Mapping) or "color" not in camera:
                raise KeyError(f"missing required camera vision/{camera_name}/color")
            payload[client_key] = _to_hwc_uint8(camera["color"], ctx=f"vision/{camera_name}/color")
        state = obs.get("state")
        if not isinstance(state, Mapping):
            raise ValueError("obs['state'] must be a mapping")
        payload["observation/state"] = self._state_to_eef20(state)
        payload["prompt"] = _decode_instruction(
            obs.get("instruction", obs.get("task_instruction")),
            self.default_instruction,
        )
        return payload

    def _eef20_chunk_to_native(self, chunk: np.ndarray) -> list[dict]:
        actions = np.asarray(chunk, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != _EEF20_DIM:
            raise ValueError(f"expected action chunk [T, {_EEF20_DIM}], got {actions.shape}")
        left_pos, left_quat = self._arm_constants("left")
        right_pos, right_quat = self._arm_constants("right")
        native = []
        for row in actions:
            left_pose, left_grip = _pose_from_arm10(row[:10])
            right_pose, right_grip = _pose_from_arm10(row[10:])
            left_world = _base_to_env(left_pose, left_pos, left_quat)
            right_world = _base_to_env(right_pose, right_pos, right_quat)
            native.append(
                {
                    "left_ee_pose": left_world.astype(np.float32),
                    "left_ee_joint_state": np.clip(left_grip, 0.0, 1.0).astype(np.float32),
                    "right_ee_pose": right_world.astype(np.float32),
                    "right_ee_joint_state": np.clip(right_grip, 0.0, 1.0).astype(np.float32),
                }
            )
        return native

    def _hold_position_chunk(self, payload: dict) -> list[dict]:
        steps = self.replan_steps or 8
        eef20 = np.asarray(payload["observation/state"], dtype=np.float32)
        return self._eef20_chunk_to_native(np.tile(eef20, (steps, 1)))

    def _infer_one(self, payload: dict) -> list[dict]:
        if self.allow_dummy_policy or self._policy is None:
            return self._hold_position_chunk(payload)
        result = self._policy.infer(payload)
        actions = np.asarray(result["actions"], dtype=np.float32)
        n_exec = actions.shape[0]
        if self.replan_steps is not None:
            n_exec = min(self.replan_steps, n_exec)
        return self._eef20_chunk_to_native(actions[:n_exec])

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
        env_idx_list = [int(item) for item in env_idx_list]
        if not env_idx_list:
            raise ValueError("get_action_batch received an empty env_idx_list.")
        missing = [item for item in env_idx_list if item not in self._batch]
        if missing:
            raise ValueError(f"No stored observation for env_idx {missing}; call update_obs_batch first.")
        return [self._infer_one(self._batch[item]) for item in env_idx_list]

    def reset(self):
        self._batch = {}
        self._order = []
        if self._policy is not None:
            self._policy.reset()
