import copy
import importlib.util
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_A1_DIR = _SCRIPT_DIR / "A1"
if str(_A1_DIR) not in sys.path:
    sys.path.insert(0, str(_A1_DIR))

_INFER_VLA_PATH = _A1_DIR / "deploy" / "infer_vla.py"
_INFER_VLA_SPEC = importlib.util.spec_from_file_location("a1_deploy_infer_vla", _INFER_VLA_PATH)
if _INFER_VLA_SPEC is None or _INFER_VLA_SPEC.loader is None:
    raise ImportError(f"Unable to load A1 infer_vla from {_INFER_VLA_PATH}")
_INFER_VLA_MODULE = importlib.util.module_from_spec(_INFER_VLA_SPEC)
_INFER_VLA_SPEC.loader.exec_module(_INFER_VLA_MODULE)
run_inference = _INFER_VLA_MODULE.run_inference
from a1.config import TrainConfig  # noqa: E402
from a1.data.vla.utils import NormalizationType  # noqa: E402
from a1.torch_util import get_local_rank, seed_all  # noqa: E402
from a1.util import resource_path  # noqa: E402
from a1.vla.affordvla import AffordVLA  # noqa: E402

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


DEFAULT_MODEL_PATH = str((_SCRIPT_DIR / "../../../../models/a1-pretrain").resolve())


def _quat_wxyz_to_rpy(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1.0, np.sign(sinp) * (np.pi / 2.0), np.arcsin(sinp))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack([roll, pitch, yaw], axis=-1).astype(np.float32)


def _pose7_to_pose6(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape[-1] != 7:
        return pose
    return np.concatenate([pose[..., :3], _quat_wxyz_to_rpy(pose[..., 3:7])], axis=-1).astype(np.float32)


def _rpy_to_quat_wxyz(rpy: np.ndarray) -> np.ndarray:
    rpy = np.asarray(rpy, dtype=np.float64)
    roll, pitch, yaw = rpy[..., 0], rpy[..., 1], rpy[..., 2]

    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.stack([w, x, y, z], axis=-1).astype(np.float32)


def _pose6_to_pose7(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape[-1] != 6:
        return pose
    return np.concatenate([pose[..., :3], _rpy_to_quat_wxyz(pose[..., 3:6])], axis=-1).astype(np.float32)


def _prepare_ee_obs_schema(obs: dict) -> dict:
    obs = copy.deepcopy(obs)
    state = obs.get("state", {})
    for key in ("ee_pose", "left_ee_pose", "right_ee_pose"):
        if key in state:
            state[key] = _pose7_to_pose6(state[key])
    return obs


def _prepare_ee_action_schema(action: dict) -> dict:
    action = copy.deepcopy(action)
    for key in ("ee_pose", "left_ee_pose", "right_ee_pose"):
        if key in action:
            action[key] = _pose6_to_pose7(action[key])
    return action


def _find_latest_unsharded(run_dir: str | os.PathLike | None) -> str | None:
    if not run_dir or not os.path.isdir(run_dir):
        return None
    candidates = []
    for name in os.listdir(run_dir):
        path = os.path.join(run_dir, name)
        if name.endswith("-unsharded") and os.path.isdir(path) and os.path.isfile(os.path.join(path, "model.pt")):
            candidates.append(path)
    if not candidates:
        latest = os.path.join(run_dir, "latest-unsharded")
        if os.path.isdir(latest) and os.path.isfile(os.path.join(latest, "model.pt")):
            return latest
        return None
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def _find_config_path(checkpoint_path: str | os.PathLike) -> Path:
    checkpoint_path = Path(checkpoint_path)
    candidates = [checkpoint_path / "config.yaml", checkpoint_path.parent / "config.yaml"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"config.yaml not found in {checkpoint_path} or {checkpoint_path.parent}")


def _load_json(path: str | os.PathLike):
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_a1_model(checkpoint_path: str, seed: int):
    config = TrainConfig.load(_find_config_path(checkpoint_path), validate_paths=False)
    model_cfg = config.model

    if model_cfg.vit_load_path:
        model_cfg.vit_load_path = os.path.join(
            os.environ.get("DATA_DIR", ""),
            "pretrained_image_encoders",
            os.path.basename(model_cfg.vit_load_path),
        )
    if model_cfg.llm_load_path:
        model_cfg.llm_load_path = os.path.join(
            os.environ.get("DATA_DIR", ""),
            "pretrained_llms",
            os.path.basename(model_cfg.llm_load_path),
        )
    model_cfg.tokenizer.tokenizer_dir = os.environ.get("HF_HOME", "")

    if not torch.cuda.is_available():
        raise RuntimeError("A1 inference requires CUDA in the current implementation.")
    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    device = torch.device("cuda")

    model = AffordVLA(model_cfg)
    base_model_state_dict_path = resource_path(DEFAULT_MODEL_PATH, "model.pt")
    if os.path.exists(base_model_state_dict_path) and os.path.abspath(checkpoint_path) != os.path.abspath(DEFAULT_MODEL_PATH):
        base_state_dict = torch.load(base_model_state_dict_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(base_state_dict, strict=False)
        print(f"[A1 Model] Loaded base checkpoint | missing={len(missing)} | unexpected={len(unexpected)}")
        del base_state_dict

    model_state_dict_path = resource_path(checkpoint_path, "model.pt")
    if not os.path.exists(model_state_dict_path):
        raise FileNotFoundError(f"A1 model.pt not found at {model_state_dict_path}")
    model_state_dict = torch.load(model_state_dict_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
    if missing or unexpected:
        print(
            f"[A1 Model] Loaded finetuned checkpoint with strict=False | "
            f"missing={len(missing)} {missing[:5]} | unexpected={len(unexpected)} {unexpected[:5]}"
        )
    del model_state_dict
    model = model.to(device)
    model.eval()
    seed_all(seed)
    return model, model_cfg


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        self.env_cfg_type = model_cfg["env_cfg_type"]
        self.task_name = model_cfg.get("task_name", "")
        self.default_prompt = model_cfg.get("prompt") or self.task_name or "Do your job."

        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        assert len(self.robot_action_dim_info["arm_dim"]) == len(self.robot_action_dim_info["ee_dim"]), (
            "Arm and EE action dimensions must match"
        )

        self.action_dim = int(model_cfg.get("action_dim") or (
            sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"])
        ))
        self.action_chunk_size = int(model_cfg.get("action_chunk_size", 50))
        self.sequence_length = int(model_cfg.get("sequence_length", 600))
        self.no_norm = bool(model_cfg.get("no_norm", True))
        self.normalization_type = NormalizationType(model_cfg.get("normalization_type", "bounds"))
        self.use_wrist_image = bool(model_cfg.get("use_wrist_image", True))
        self.seed = int(model_cfg.get("seed") or 6198)

        self.model_path = self._resolve_model_path(model_cfg)
        self.norm_stats = self._load_norm_stats(model_cfg)
        self.model, self.a1_model_cfg = _load_a1_model(self.model_path, self.seed)

        self._obs_buffer_batch = {}
        self._latest_env_idx_list = [0]

        print(
            f"[A1 Model] Initialized | model_path={self.model_path} | action_type={self.action_type} | "
            f"action_dim={self.action_dim} | action_chunk_size={self.action_chunk_size} | no_norm={self.no_norm}"
        )

    def _resolve_model_path(self, model_cfg):
        model_path = model_cfg.get("model_path") or None
        if model_path:
            latest = _find_latest_unsharded(model_path)
            return latest or model_path

        task_name = model_cfg.get("task_name", "")
        expert_data_num = model_cfg.get("expert_data_num", "")
        seed = model_cfg.get("seed", "")
        run_base = f"{task_name}-a1-{self.action_type}-{expert_data_num}eps-seed{seed}"
        latest_file = _SCRIPT_DIR / "checkpoints" / f"{run_base}.latest"
        if latest_file.is_file():
            latest = _find_latest_unsharded(latest_file.read_text().strip())
            if latest:
                return latest

        checkpoints_dir = _SCRIPT_DIR / "checkpoints"
        if checkpoints_dir.is_dir():
            matches = sorted(checkpoints_dir.glob(f"{run_base}-*"), key=lambda p: p.stat().st_mtime, reverse=True)
            for match in matches:
                latest = _find_latest_unsharded(match)
                if latest:
                    return latest

        return DEFAULT_MODEL_PATH

    def _load_norm_stats(self, model_cfg):
        if self.no_norm:
            return None
        stats_path = model_cfg.get("norm_stats_json_path") or model_cfg.get("data_stats_path")
        if not stats_path:
            for candidate in (
                Path(self.model_path) / "dataset_statistics.json",
                Path(self.model_path) / "dataset_stats.json",
            ):
                if candidate.is_file():
                    stats_path = str(candidate)
                    break
        if not stats_path:
            raise ValueError("A1 normalization is enabled, but no norm stats json path was provided.")
        return _load_json(stats_path)

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", i) for i, obs in enumerate(obs_list)]
        for obs in obs_list:
            env_idx = obs.get("env_idx", 0)
            self._obs_buffer_batch[env_idx] = _encode_obs(
                obs,
                self.action_type,
                self.robot_action_dim_info,
                self.default_prompt,
            )

    def get_action(self):
        return self.get_action_batch([self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list

        action_batch = []
        for env_idx in env_idx_list:
            if env_idx not in self._obs_buffer_batch:
                raise RuntimeError(f"No observation buffered for env_idx={env_idx}. Call update_obs first.")

            input_data = self._obs_buffer_batch[env_idx]
            with torch.inference_mode():
                results = run_inference(
                    self.model,
                    input_data,
                    self.sequence_length,
                    self.norm_stats,
                    self.normalization_type,
                    use_proprio=True,
                    use_wrist_image=self.use_wrist_image,
                    no_norm=self.no_norm,
                )

            actions = np.asarray(results["predicted_actions"], dtype=np.float32).squeeze()
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)
            actions = actions[: self.action_chunk_size, : self.action_dim]

            action_steps = []
            for step_action in actions:
                action = unpack_robot_state(
                    step_action,
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                )
                if self.action_type == "ee":
                    action = _prepare_ee_action_schema(action)
                action_steps.append(action)
            action_batch.append(action_steps)

        return action_batch

    def reset(self):
        self._obs_buffer_batch = {}
        self._latest_env_idx_list = [0]
        print("[A1 Model] Reset")


def _encode_obs(observation, action_type, robot_action_dim_info, default_prompt):
    if action_type == "ee":
        observation = _prepare_ee_obs_schema(observation)

    vision = observation.get("vision", {})
    images = []
    for camera_name in ("cam_head", "cam_right_wrist", "cam_left_wrist"):
        img = _extract_rgb_image(vision, camera_name)
        if img is not None:
            images.append(img)

    if not images:
        raise ValueError("A1 requires at least one RGB camera image in observation['vision'].")

    instruction = observation.get("instruction", observation.get("instructions", default_prompt))
    if isinstance(instruction, (list, tuple)):
        instruction = instruction[0] if instruction else default_prompt

    state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs")
    return {
        "images": images,
        "instruction": str(instruction),
        "proprio": torch.tensor(state.reshape(1, -1), dtype=torch.float32),
    }


def _extract_rgb_image(vision, camera_name):
    camera_data = vision.get(camera_name)
    if camera_data is None:
        return None
    if isinstance(camera_data, dict):
        img = camera_data.get("color", camera_data.get("rgb"))
    else:
        img = camera_data
    if img is None:
        return None

    img = np.asarray(img)
    if img.ndim == 1 and img.dtype == np.uint8:
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if img is None:
            return None
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim != 3 or img.shape[-1] != 3:
        return None

    img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
