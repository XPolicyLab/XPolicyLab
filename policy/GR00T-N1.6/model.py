from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

_POLICY_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _POLICY_DIR.parents[2]
_GR00T_DIR = _POLICY_DIR / "Isaac-GR00T"
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
if str(_GR00T_DIR) not in sys.path:
    sys.path.insert(0, str(_GR00T_DIR))

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


def _find_latest_checkpoint(run_dir: Path) -> str | None:
    if not run_dir.is_dir():
        return None
    candidates = [
        p for p in run_dir.iterdir()
        if p.is_dir() and p.name.startswith("checkpoint-")
    ]
    if not candidates:
        if (run_dir / "config.json").exists():
            return str(run_dir)
        return None

    def _step(path: Path) -> int:
        parts = path.name.split("-")
        for item in reversed(parts):
            if item.isdigit():
                return int(item)
        return -1

    return str(sorted(candidates, key=_step)[-1])


def _resolve_embodiment_tag(value: Any) -> EmbodimentTag:
    if isinstance(value, EmbodimentTag):
        return value
    if value is None:
        return EmbodimentTag.NEW_EMBODIMENT

    text = str(value)
    if text in EmbodimentTag.__members__:
        return EmbodimentTag[text]
    for item in EmbodimentTag:
        if text == item.value:
            return item
    raise ValueError(
        f"Unknown GR00T embodiment_tag={value!r}. "
        f"Use an enum name such as NEW_EMBODIMENT or an enum value such as new_embodiment."
    )


def _default_pretrained_model_path() -> Path:
    return (_POLICY_DIR / "../../../../models/GR00T-N1.6-3B").resolve()


def _state_modality_layout(action_type: str, dim_info: dict) -> list[tuple[str, int]]:
    arm_dims = dim_info["arm_dim"]
    ee_dims = dim_info["ee_dim"]
    if len(arm_dims) == 1:
        arm_name = "arm" if action_type == "joint" else "ee_pose"
        return [(arm_name, int(arm_dims[0])), ("gripper", int(ee_dims[0]))]
    if len(arm_dims) == 2:
        arm_suffix = "arm" if action_type == "joint" else "ee_pose"
        return [
            (f"left_{arm_suffix}", int(arm_dims[0])),
            ("left_gripper", int(ee_dims[0])),
            (f"right_{arm_suffix}", int(arm_dims[1])),
            ("right_gripper", int(ee_dims[1])),
        ]
    raise ValueError(f"Unsupported arm count: {len(arm_dims)}")


def _normalize_single_arm_aliases(observation: dict, action_type: str, dim_info: dict) -> dict:
    if action_type != "joint" or len(dim_info["arm_dim"]) != 1:
        return observation
    state = observation.get("state")
    if isinstance(state, dict) and "joint_state" not in state and "arm_joint_state" in state:
        state["joint_state"] = state["arm_joint_state"]
    return observation


class _DummyPolicy:
    def __init__(self, action_keys: list[str], action_dims: dict[str, int], horizon: int):
        self._action_keys = action_keys
        self._action_dims = action_dims
        self._horizon = horizon
        self.modality_configs = {
            "action": type("Cfg", (), {"modality_keys": action_keys, "delta_indices": list(range(horizon))})(),
            "state": type("Cfg", (), {"modality_keys": action_keys, "delta_indices": [0]})(),
            "video": type("Cfg", (), {"modality_keys": ["cam_head"], "delta_indices": [0]})(),
            "language": type("Cfg", (), {"modality_keys": ["annotation.human.task_description"], "delta_indices": [0]})(),
        }
        self.language_key = "annotation.human.task_description"

    def get_modality_config(self):
        return self.modality_configs

    def get_action(self, observation):
        batch_size = next(iter(observation["state"].values())).shape[0]
        return {
            key: np.zeros((batch_size, self._horizon, self._action_dims[key]), dtype=np.float32)
            for key in self._action_keys
        }, {}

    def reset(self):
        return {}


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        self.env_cfg_type = model_cfg["env_cfg_type"]
        self.task_name = model_cfg.get("task_name", "")
        self.default_prompt = model_cfg.get("prompt") or self.task_name.replace("_", " ") or "Do your job."
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.layout = _state_modality_layout(self.action_type, self.robot_action_dim_info)
        self.action_chunk_size = int(model_cfg.get("action_chunk_size", 16))
        self.execution_horizon = int(model_cfg.get("execution_horizon", self.action_chunk_size))
        self.device = model_cfg.get("device", "cuda:0")
        self.embodiment_tag = _resolve_embodiment_tag(model_cfg.get("embodiment_tag", "NEW_EMBODIMENT"))
        self.strict = bool(model_cfg.get("strict", True))

        self.model = self._load_policy(model_cfg)
        self.modality_configs = self.model.get_modality_config()
        self.language_key = getattr(
            self.model,
            "language_key",
            self.modality_configs["language"].modality_keys[0],
        )

        self._obs_buffer_batch: dict[int, dict[str, Any]] = {}
        self._latest_env_idx_list = [0]
        print(
            f"[GR00T Model] Initialized | action_type={self.action_type} | "
            f"embodiment_tag={self.embodiment_tag.name} | execution_horizon={self.execution_horizon}"
        )

    def _load_policy(self, model_cfg):
        action_dims = {key: dim for key, dim in self.layout}
        action_keys = [key for key, _dim in self.layout]
        if bool(model_cfg.get("dummy_policy", False)):
            print("[GR00T Model] Using dummy zero-action policy.")
            return _DummyPolicy(action_keys, action_dims, self.execution_horizon)

        model_path = self._resolve_model_path(model_cfg)
        print(f"[GR00T Model] Loading checkpoint from: {model_path}")
        return Gr00tPolicy(
            embodiment_tag=self.embodiment_tag,
            model_path=model_path,
            device=self.device,
            strict=self.strict,
        )

    def _resolve_model_path(self, model_cfg) -> str:
        model_path = model_cfg.get("model_path") or None
        if model_path:
            candidate = _find_latest_checkpoint(Path(model_path))
            return candidate or model_path

        task_name = model_cfg.get("task_name", self.task_name)
        expert_data_num = model_cfg.get("expert_data_num", "")
        seed = model_cfg.get("seed", "0")
        run_basename = f"{task_name}-gr00t-{self.action_type}-{expert_data_num}eps-seed{seed}"
        latest_file = _POLICY_DIR / "checkpoints" / f"{run_basename}.latest"
        if latest_file.exists():
            latest = latest_file.read_text(encoding="utf-8").strip()
            candidate = _find_latest_checkpoint(Path(latest))
            if candidate:
                return candidate

        pattern = f"{run_basename}-*"
        runs = sorted((_POLICY_DIR / "checkpoints").glob(pattern), key=lambda p: p.stat().st_mtime)
        for run_dir in reversed(runs):
            candidate = _find_latest_checkpoint(run_dir)
            if candidate:
                return candidate

        pretrained = (
            model_cfg.get("pretrained_model_path")
            or os.environ.get("BASE_MODEL_PATH")
            or str(_default_pretrained_model_path())
        )
        if Path(pretrained).exists():
            return str(pretrained)
        raise FileNotFoundError(
            "No GR00T checkpoint found. Train first or pass MODEL_PATH to eval.sh. "
            f"Checked latest file {latest_file} and pretrained path {pretrained}."
        )

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", i) for i, obs in enumerate(obs_list)]
        for obs in obs_list:
            env_idx = obs.get("env_idx", 0)
            self._obs_buffer_batch[env_idx] = self._encode_obs(obs)

    def get_action(self):
        return self.get_action_batch([self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list

        observations = []
        for env_idx in env_idx_list:
            if env_idx not in self._obs_buffer_batch:
                raise RuntimeError(f"No observation buffered for env_idx={env_idx}. Call update_obs first.")
            observations.append(self._obs_buffer_batch[env_idx])

        batched_obs = _stack_gr00t_observations(observations)
        action_dict, _info = self.model.get_action(batched_obs)
        return self._decode_action_batch(action_dict, len(env_idx_list))

    def reset(self):
        self._obs_buffer_batch.clear()
        self._latest_env_idx_list = [0]
        if hasattr(self.model, "reset"):
            self.model.reset()
        print("[GR00T Model] Reset")

    def _encode_obs(self, observation):
        observation = _normalize_single_arm_aliases(
            observation,
            self.action_type,
            self.robot_action_dim_info,
        )
        packed_state = pack_robot_state(
            observation,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
        ).astype(np.float32)
        state = {}
        offset = 0
        for key, dim in self.layout:
            state[key] = packed_state[offset: offset + dim].reshape(1, dim).astype(np.float32)
            offset += dim

        video = {}
        for video_key in self.modality_configs["video"].modality_keys:
            video[video_key] = _extract_video_frame(observation.get("vision", {}), video_key)

        instruction = observation.get("instruction", observation.get("instructions", self.default_prompt))
        if isinstance(instruction, (list, tuple)):
            instruction = instruction[0] if instruction else self.default_prompt
        instruction = str(instruction or self.default_prompt)

        return {
            "video": video,
            "state": state,
            "language": {self.language_key: [instruction]},
        }

    def _decode_action_batch(self, action_dict: dict[str, np.ndarray], batch_size: int):
        action_keys = self.modality_configs["action"].modality_keys
        horizon = min(
            self.execution_horizon,
            min(int(action_dict[key].shape[1]) for key in action_keys),
        )
        result = []
        for batch_idx in range(batch_size):
            env_actions = []
            for step_idx in range(horizon):
                packed = np.concatenate(
                    [np.asarray(action_dict[key][batch_idx, step_idx], dtype=np.float32) for key in action_keys],
                    axis=-1,
                )
                env_actions.append(
                    unpack_robot_state(
                        packed,
                        self.action_type,
                        self.robot_action_dim_info,
                        source_type="obs",
                    )
                )
            result.append(env_actions)
        return result


def _stack_gr00t_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
    first = observations[0]
    return {
        "video": {
            key: np.stack([obs["video"][key] for obs in observations], axis=0)
            for key in first["video"]
        },
        "state": {
            key: np.stack([obs["state"][key] for obs in observations], axis=0)
            for key in first["state"]
        },
        "language": {
            key: [obs["language"][key] for obs in observations]
            for key in first["language"]
        },
    }


def _extract_video_frame(vision: dict, video_key: str) -> np.ndarray:
    candidates = {
        "cam_head": ["cam_head"],
        "cam_wrist": ["cam_wrist", "cam_left_wrist", "cam_right_wrist", "cam_head"],
        "cam_left_wrist": ["cam_left_wrist", "cam_wrist", "cam_head"],
        "cam_right_wrist": ["cam_right_wrist", "cam_wrist", "cam_head"],
    }.get(video_key, [video_key, "cam_head", "cam_left_wrist", "cam_right_wrist", "cam_wrist"])

    image = None
    for name in candidates:
        cam = vision.get(name)
        if cam is None:
            continue
        image = cam.get("color") if isinstance(cam, dict) else cam
        if image is not None:
            break
    if image is None:
        raise KeyError(f"Cannot find image for GR00T video key '{video_key}'.")

    image = np.asarray(image)
    if image.ndim == 1 and image.dtype == np.uint8:
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 image for '{video_key}', got {image.shape}")

    image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.uint8).reshape(1, 240, 320, 3)
