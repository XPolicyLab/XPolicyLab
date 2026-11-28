"""XPolicyLab observation/action adapter for the public SWAM RoboTwin policy.

Core inference comes from SWAM_ROOT/groot/vla/model/n1_5/sim_policy.py;
GrootSimPolicy calls the DreamZero VLA and WANPolicyHead in the SWAM checkout.
See README.md for the pinned upstream source and checkpoint revisions.
"""

from __future__ import annotations

import json
import os
import random
import time
import uuid
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.distributed as dist
from tianshou.data import Batch

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import get_action_dim, get_robot_action_dim_info, unpack_robot_state


POLICY_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = POLICY_DIR / "checkpoints"



def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _ensure_dist_initialized() -> None:
    if dist.is_available() and not dist.is_initialized():
        rendezvous_file = Path(os.environ.get("SWAM_DIST_INIT_FILE", f"/tmp/swam_xpolicylab_dist_{os.getpid()}"))
        rendezvous_file.parent.mkdir(parents=True, exist_ok=True)
        if rendezvous_file.exists():
            rendezvous_file.unlink()
        dist.init_process_group(backend="gloo", init_method=f"file://{rendezvous_file}", world_size=1, rank=0)


def _validate_checkpoint(path: Path) -> None:
    """Reject incomplete downloads before the upstream non-strict weight loader."""
    required = [path / name for name in (
        "config.json", "experiment_cfg/conf.yaml", "experiment_cfg/metadata.json"
    )]
    index = path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text()).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map or not all(
            isinstance(name, str) and name.strip() for name in weight_map.values()
        ):
            raise ValueError(f"Invalid or empty weight_map in {index}")
        required.extend(path / name for name in sorted(set(weight_map.values())))
    else:
        required.append(path / "model.safetensors")
    missing = [str(file) for file in required if not file.is_file() or file.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Incomplete SWAM checkpoint; missing or empty files:\n  " + "\n  ".join(missing))


def _resolve_model_path(model_cfg: dict[str, Any]) -> Path:
    cfg = dict(model_cfg)
    explicit = cfg.get("model_path") or cfg.get("checkpoint_path") or os.environ.get("SWAM_MODEL_PATH")
    if explicit:
        cfg["model_path"] = explicit
    candidates = candidate_checkpoint_roots(
        cfg, CHECKPOINTS_DIR, policy_dir=POLICY_DIR, explicit_keys=("model_path", "checkpoint_path")
    )
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "experiment_cfg" / "conf.yaml").is_file():
            _validate_checkpoint(candidate)
            return candidate.resolve()
    checked = "\n  ".join(str(path) for path in candidates) or "  <no checkpoint candidates>"
    raise FileNotFoundError(f"SWAM checkpoint not found. Set SWAM_MODEL_PATH or deploy.yml model_path. Checked:\n  {checked}")


def _import_swam(root_value: str) -> tuple[Any, Any]:
    SWAM_ROOT = Path(root_value).expanduser()
    if not root_value or not (SWAM_ROOT / "groot").is_dir():
        raise FileNotFoundError("SWAM_ROOT must point to the SWAM/DreamZero checkout containing groot/ and eval_utils/.")
    root = str(SWAM_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from groot.vla.data.schema import EmbodimentTag
    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
    return EmbodimentTag, GrootSimPolicy


def _component_overrides(model_cfg: dict[str, Any]) -> tuple[list[str], str]:
    """Use the upstream loader's in-memory overrides; never rewrite checkpoint files."""
    paths = {}
    for key, env in (("wan_path", "SWAM_WAN_PATH"),
                     ("tokenizer_path", "SWAM_TOKENIZER_PATH"),
                     ("image_encoder_path", "SWAM_IMAGE_ENCODER_PATH")):
        value = model_cfg.get(key) or os.environ.get(env)
        if not value:
            raise ValueError(f"Set {key} in deploy.yml or {env}; see README Model Assets.")
        paths[key] = Path(value).expanduser().resolve()
    wan, tokenizer, encoder = (paths[k] for k in ("wan_path", "tokenizer_path", "image_encoder_path"))
    for path in (wan / "Wan2.2_VAE.pth", wan / "models_t5_umt5-xxl-enc-bf16.pth", encoder):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not tokenizer.is_dir():
        raise FileNotFoundError(tokenizer)
    overrides = {"tokenizer_path": str(tokenizer)}
    for component, path in (("diffusion_model", wan),
                            ("text_encoder", wan / "models_t5_umt5-xxl-enc-bf16.pth"),
                            ("vae", wan / "Wan2.2_VAE.pth"),
                            ("image_encoder", encoder)):
        name = f"{component}_pretrained_path"
        overrides[name] = str(path)
        overrides[f"action_head_cfg.config.{component}_cfg.{name}"] = str(path)
    return [f"{key}={json.dumps(value)}" for key, value in overrides.items()], str(tokenizer)


def _image(obs: dict[str, Any], names: tuple[str, ...]) -> np.ndarray:
    vision = obs.get("vision", {})
    for name in names:
        if name not in vision:
            continue
        value = vision[name]
        if isinstance(value, dict):
            value = value.get("color", value.get("rgb", value.get("image")))
        if value is None:
            continue
        array = np.asarray(value)
        if array.ndim != 3:
            raise ValueError(f"Camera {name!r} must be HWC, got {array.shape}")
        if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.shape[-1] != 3:
            raise ValueError(f"Camera {name!r} must have 3 channels, got {array.shape}")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.shape[:2] != (256, 320):
            array = cv2.resize(array, (320, 256), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(array)
    raise KeyError(f"Missing camera; tried {names!r} in obs['vision']")


def _prompt(obs: dict[str, Any], default: str) -> str:
    value = obs.get("instruction", obs.get("instructions", obs.get("task_instruction")))
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if hasattr(value, "item"):
        value = value.item()
    text = str(value).strip() if value is not None else ""
    return text or default


class Model(ModelTemplate):
    """Single-environment SWAM policy with RoboTwin 2.0 EEF semantics."""

    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = dict(model_cfg)
        self.action_type = self.model_cfg.get("action_type") or "ee"
        self.env_cfg_type = self.model_cfg.get("env_cfg_type") or "arx_x5"
        if self.env_cfg_type not in {"arx_x5", "aloha_agilex"}:
            raise ValueError("Use arx_x5 (official RoboTwin action profile) or aloha_agilex with the native aloha-agilex task configuration.")
        if self.action_type != "ee":
            raise ValueError("SWAM RoboTwin adapter only supports action_type=ee.")
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        if self.robot_action_dim_info["arm_dim"] != [6, 6] or self.robot_action_dim_info["ee_dim"] != [1, 1]:
            raise ValueError(f"SWAM RoboTwin EEF expects dual 6-DOF arms, got {self.robot_action_dim_info!r}")
        self.action_dim = get_action_dim(self.env_cfg_type, "ee")
        if self.action_dim != 16:
            raise ValueError("This checkpoint requires 16D EEF actions.")
        configured_dim = self.model_cfg.get("action_dim")
        if configured_dim is not None and int(configured_dim) != self.action_dim:
            raise ValueError(f"RoboTwin EEF action_dim must be 16, got {configured_dim!r}.")
        self.action_horizon = int(self.model_cfg.get("action_horizon", 48))
        if self.action_horizon != 48:
            raise ValueError("This SWAM checkpoint contract requires action_horizon=48.")
        self.video_history = int(self.model_cfg.get("video_history", 8))
        if self.video_history != 8:
            raise ValueError("SWAM requires 8 frames sampled at stride 6.")
        self._first_call = True
        self._session_id = uuid.uuid4().hex
        self.default_prompt = str(self.model_cfg.get("prompt") or self.model_cfg.get("task_name") or "")
        self._frames: dict[str, list[np.ndarray]] = {
            "video.cam_high": [], "video.cam_left_wrist": [], "video.cam_right_wrist": []
        }
        self._observation: dict[str, Any] | None = None
        self._latest_env_idx_list = [0]
        if _as_bool(self.model_cfg.get("eval_batch"), False):
            raise ValueError("SWAM adapter is stateful and requires eval_batch=false.")

        model_path = _resolve_model_path(self.model_cfg)
        model_overrides, tokenizer_path = _component_overrides(self.model_cfg)
        embodiment_tag_cls, groot_policy_cls = _import_swam(str(self.model_cfg.get("swam_root") or os.environ.get("SWAM_ROOT", "")))
        _ensure_dist_initialized()
        os.environ.setdefault("ATTENTION_BACKEND", "TE")
        try:
            torch._dynamo.config.recompile_limit = 800
        except AttributeError:
            torch._dynamo.config.cache_size_limit = 800
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[SWAM] loading {model_path} on {device}; first load may take 8-10 minutes", flush=True)
        self.policy = groot_policy_cls(
            embodiment_tag=embodiment_tag_cls.ROBOTWIN, model_path=str(model_path), device=device,
            model_config_overrides=model_overrides, tokenizer_path_override=tokenizer_path,
            device_mesh=None,
        )
        train_cfg = getattr(self.policy, "train_cfg", {})
        relative = bool(train_cfg.get("relative_action", False))
        if relative:
            raise ValueError("The selected checkpoint uses relative_action=true; this adapter requires absolute EEF actions.")
        # CUDA Graph caches are thread-local; warmup and RPC inference must share a worker.
        self._inference_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="swam-inference")
        try:
            self._inference_worker.submit(self._warmup).result()
        except BaseException:
            self._inference_worker.shutdown(wait=True)
            raise
        print("[SWAM] initialized | embodiment=robotwin | action=ee16 | horizon=48", flush=True)

    def _warmup(self) -> None:
        """Compile first-frame and continuation paths before the server accepts RPCs."""
        started = time.monotonic()
        print("[SWAM] startup warmup: first frame and 8-frame continuation", flush=True)
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        head = self.policy.trained_model.action_head
        obs = {
            "vision": {name: {"color": np.zeros((256, 320, 3), dtype=np.uint8)}
                       for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")},
            "state": {"eef_position": np.array([0, 0, 0, 1, 0, 0, 0, 0] * 2, dtype=np.float32)},
            "instruction": "Move the object.",
        }
        with torch.random.fork_rng(devices=devices):
            try:
                self.update_obs(obs)
                self._get_action_batch(self._latest_env_idx_list)
                print("[SWAM] startup warmup: first frame complete", flush=True)
                for _ in range(self.action_horizon):
                    self.update_obs(obs)
                self._get_action_batch(self._latest_env_idx_list)
                if devices:
                    torch.cuda.synchronize(devices[0])
            finally:
                # These are the episode caches of the pinned public SWAM action head.
                for name in ("language", "clip_feas", "ys", "kv_cache1", "kv_cache_neg",
                             "crossattn_cache", "crossattn_cache_neg", "past_action_history"):
                    setattr(head, name, None)
                head.current_start_frame = 0
                self.reset()
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
        print(f"[SWAM] startup warmup complete in {time.monotonic() - started:.1f}s; "
              "episode state cleared and RNG restored", flush=True)

    def update_obs(self, obs: dict[str, Any]) -> None:
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list: list[dict[str, Any]]) -> None:
        if isinstance(obs_list, dict):
            obs_list = [obs_list]
        if len(obs_list) != 1:
            raise ValueError("SWAM RoboTwin adapter supports one active environment (eval_batch=false).")
        obs = obs_list[0]
        self._latest_env_idx_list = [int(obs.get("env_idx", 0))]
        camera_groups = (
            ("video.cam_high", ("cam_high", "cam_head", "head_camera", "top_camera")),
            ("video.cam_left_wrist", ("cam_left_wrist", "left_camera", "left_wrist", "wrist_left")),
            ("video.cam_right_wrist", ("cam_right_wrist", "right_camera", "right_wrist", "wrist_right")),
        )
        for key, names in camera_groups:
            self._frames[key].append(_image(obs, names))
            self._frames[key] = self._frames[key][-49:]
        state_dict = obs.get("state", {})
        state = np.asarray(state_dict.get("eef_position", []), dtype=np.float32).reshape(-1)
        if state.size != 16:
            state = np.concatenate([
                np.asarray(state_dict["left_ee_pose"], dtype=np.float32).reshape(7),
                np.asarray(state_dict["left_ee_joint_state"], dtype=np.float32).reshape(1),
                np.asarray(state_dict["right_ee_pose"], dtype=np.float32).reshape(7),
                np.asarray(state_dict["right_ee_joint_state"], dtype=np.float32).reshape(1),
            ])
        self._observation = {
            "video.cam_high": self._stack(self._frames["video.cam_high"]),
            "video.cam_left_wrist": self._stack(self._frames["video.cam_left_wrist"]),
            "video.cam_right_wrist": self._stack(self._frames["video.cam_right_wrist"]),
            "state.eef_position": state.astype(np.float64).reshape(1, 16),
            "annotation.language.action_text": _prompt(obs, self.default_prompt),
            "session_id": self._session_id,
        }

    def get_action(self) -> list[dict[str, np.ndarray]]:
        return self.get_action_batch(self._latest_env_idx_list)[0]

    def get_action_batch(self, env_idx_list=None) -> list[list[dict[str, np.ndarray]]]:
        return self._inference_worker.submit(self._get_action_batch, env_idx_list).result()

    def _get_action_batch(self, env_idx_list=None) -> list[list[dict[str, np.ndarray]]]:
        if env_idx_list is not None:
            ids = np.asarray(env_idx_list).reshape(-1).tolist()
            if ids != [self._latest_env_idx_list[0]]:
                raise ValueError("SWAM RoboTwin adapter only supports one active environment.")
        if self._observation is None:
            raise RuntimeError("Call update_obs before get_action.")
        result, _ = self.policy.lazy_joint_forward_causal(Batch(obs=dict(self._observation)))
        action_dict = self._action_dict(result.act)
        packed = np.asarray(action_dict["action.eef_position"], dtype=np.float32).reshape(-1, self.action_dim)
        if packed.shape[0] < self.action_horizon:
            raise ValueError(f"SWAM returned {packed.shape[0]} actions, expected {self.action_horizon}.")
        actions = unpack_robot_state(packed[: self.action_horizon], "ee", self.robot_action_dim_info, source_type="obs")
        if not np.isfinite(packed).all():
            raise ValueError("SWAM returned non-finite actions.")
        self._first_call = False
        return [actions]

    def reset(self) -> None:
        self._first_call = True
        self._session_id = uuid.uuid4().hex
        for key in self._frames:
            self._frames[key] = []
        self._observation = None
        self._latest_env_idx_list = [0]

    def _stack(self, frames: list[np.ndarray]) -> np.ndarray:
        if not frames:
            raise RuntimeError("No camera frame buffered.")
        if self._first_call:
            return np.stack(frames[-1:], axis=0)
        # Native SWAM schedule: observations at steps 6,12,...,48.
        return np.stack([frames[max(0, len(frames) - 1 + offset)]
                         for offset in range(-42, 1, 6)], axis=0)

    @staticmethod
    def _action_dict(action: Any) -> dict[str, Any]:
        if isinstance(action, dict):
            result = {str(k): v for k, v in action.items() if str(k).startswith("action.")}
        else:
            result = {key: getattr(action, key) for key in dir(action) if key.startswith("action.")}
        if "action.eef_position" not in result:
            raise KeyError(f"SWAM output did not contain action.eef_position; keys={sorted(result)}")
        return result
