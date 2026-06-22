import inspect
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from XPolicyLab.model_template import ModelTemplate


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in {"", "none", "null"}


def _parse_optional_int(value: Any) -> int | None:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> float | None:
    if _is_none_like(value):
        return None
    return float(value)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _model_dtype(mixed_precision: str) -> torch.dtype:
    key = str(mixed_precision).strip().lower()
    if key == "no":
        return torch.float32
    if key == "fp16":
        return torch.float16
    if key == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported mixed_precision: {mixed_precision}")


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB HWC image, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(image, mode="RGB")
    return np.asarray(pil_image.resize(size_wh, resample=Image.BILINEAR), dtype=np.uint8)


def _get_instruction(obs: dict, fallback: str) -> str:
    value = obs.get("task_instruction")
    if value is None:
        value = obs.get("instruction", obs.get("instructions"))
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if hasattr(value, "item"):
        value = value.item()
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _load_simple_yaml(path: Path) -> dict:
    try:
        import yaml

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        data: dict[str, Any] = {}
        stack: list[tuple[int, dict]] = [(-1, data)]
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.split("#", 1)[0].rstrip()
                if not raw.strip() or ":" not in raw:
                    continue
                indent = len(raw) - len(raw.lstrip(" "))
                key, value = raw.strip().split(":", 1)
                value = value.strip().strip("\"'")
                while stack and indent <= stack[-1][0]:
                    stack.pop()
                parent = stack[-1][1]
                if value:
                    parent[key] = value
                else:
                    child: dict[str, Any] = {}
                    parent[key] = child
                    stack.append((indent, child))
        return data


def _get_env_cfg_root(model_cfg: dict[str, Any]) -> Path:
    configured = model_cfg.get("env_cfg_root") or os.environ.get("AHA_WAM_ENV_CFG_ROOT")
    if configured:
        return Path(str(configured)).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]
    repo_env_cfg = repo_root / "env_cfg"
    if repo_env_cfg.exists():
        return repo_env_cfg
    return Path("/mnt/petrelfs/caijisong/env_cfg")


def _get_robot_action_dim_info(env_cfg_type: str, env_cfg_root: Path) -> dict:
    env_cfg = _load_simple_yaml(env_cfg_root / f"{env_cfg_type}.yml")
    robot_name = env_cfg["config"]["robot"]
    with (env_cfg_root / "robot" / "_robot_info.json").open("r", encoding="utf-8") as f:
        return json.load(f)[robot_name]


def _validate_state_config(action_type: str, robot_action_dim_info: dict) -> tuple[list[int], list[int]]:
    if action_type not in {"joint", "ee"}:
        raise ValueError(f"Unsupported action_type: {action_type!r}.")
    arm_dims = list(robot_action_dim_info["arm_dim"])
    ee_dims = list(robot_action_dim_info["ee_dim"])
    if len(arm_dims) != len(ee_dims) or len(arm_dims) not in {1, 2}:
        raise ValueError(f"Unsupported robot action dimensions: {robot_action_dim_info}")
    return arm_dims, ee_dims


def _state_keys(action_type: str, num_arms: int) -> tuple[list[str], list[str]]:
    if num_arms == 1:
        arm_keys = ["joint_state"] if action_type == "joint" else ["ee_pose"]
        return arm_keys, ["ee_joint_state"]
    if action_type == "joint":
        arm_keys = ["left_arm_joint_state", "right_arm_joint_state"]
    else:
        arm_keys = ["left_ee_pose", "right_ee_pose"]
    return arm_keys, ["left_ee_joint_state", "right_ee_joint_state"]


def _pack_robot_state(obs: dict, action_type: str, robot_action_dim_info: dict) -> np.ndarray:
    arm_dims, ee_dims = _validate_state_config(action_type, robot_action_dim_info)
    arm_keys, ee_keys = _state_keys(action_type, len(arm_dims))
    state = obs["state"]
    parts = []
    for arm_key, ee_key, arm_dim, ee_dim in zip(arm_keys, ee_keys, arm_dims, ee_dims):
        arm = np.asarray(state[arm_key])
        ee = np.asarray(state[ee_key])
        if arm.shape[-1] != arm_dim or ee.shape[-1] != ee_dim:
            raise ValueError(
                f"State shape mismatch for {arm_key}/{ee_key}: "
                f"got {arm.shape[-1]}/{ee.shape[-1]}, expected {arm_dim}/{ee_dim}."
            )
        parts.append(np.concatenate([arm, ee], axis=-1))
    return np.concatenate(parts, axis=-1)


def _unpack_robot_state(actions, action_type: str, robot_action_dim_info: dict) -> list[dict]:
    arm_dims, ee_dims = _validate_state_config(action_type, robot_action_dim_info)
    arm_keys, ee_keys = _state_keys(action_type, len(arm_dims))
    packed = np.asarray(actions)
    expected_dim = sum(arm_dims) + sum(ee_dims)
    if packed.shape[-1] != expected_dim:
        raise ValueError(f"Action dim mismatch: got {packed.shape[-1]}, expected {expected_dim}.")
    if packed.ndim == 1:
        packed = packed[None]

    result = []
    for action in packed:
        item = {}
        offset = 0
        for arm_key, ee_key, arm_dim, ee_dim in zip(arm_keys, ee_keys, arm_dims, ee_dims):
            item[arm_key] = action[offset : offset + arm_dim]
            offset += arm_dim
            item[ee_key] = action[offset : offset + ee_dim]
            offset += ee_dim
        result.append(item)
    return result


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.action_type = str(self.model_cfg["action_type"])
        self.env_cfg_type = str(self.model_cfg["env_cfg_type"])
        if self.action_type != "joint":
            raise ValueError("aha-wam was trained for joint/qpos actions; use action_type=joint.")

        self.env_cfg_root = _get_env_cfg_root(self.model_cfg)
        self.robot_action_dim_info = _get_robot_action_dim_info(self.env_cfg_type, self.env_cfg_root)
        self.default_instruction = str(self.model_cfg.get("default_instruction") or "follow the instruction")
        self.pending_actions: deque[np.ndarray] = deque()
        self.last_obs = None
        self.last_instruction = self.default_instruction
        self._batch_obs: dict[int, dict] = {}
        self._batch_instruction: dict[int, str] = {}
        self.allow_dummy_policy = _parse_bool(self.model_cfg.get("allow_dummy_policy", False))
        self._chunks_since_video_prefill = 0

        self.elava_root = Path(str(self.model_cfg["elava_root"])).expanduser().resolve()
        self.elava_src = self.elava_root / "src"
        self.checkpoint_path = Path(str(self.model_cfg["checkpoint_path"])).expanduser().resolve()
        self.dataset_stats_path = Path(str(self.model_cfg["dataset_stats_path"])).expanduser().resolve()
        if not self.allow_dummy_policy and not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not self.allow_dummy_policy and not self.dataset_stats_path.exists():
            raise FileNotFoundError(f"dataset_stats.json not found: {self.dataset_stats_path}")

        diffsynth_base = self.model_cfg.get("diffsynth_model_base_path")
        if not _is_none_like(diffsynth_base):
            os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(Path(str(diffsynth_base)).expanduser().resolve())

        for path in (self.elava_root, self.elava_src):
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)

        from ahawam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

        self.default_prompt_template = DEFAULT_PROMPT
        if self.allow_dummy_policy:
            self.policy = None
            self.action_horizon = int(self.model_cfg.get("action_horizon") or 64)
            self.replan_steps = int(self.model_cfg.get("replan_steps") or 16)
            self.chunks_per_video_prefill = int(
                self.model_cfg.get("chunks_per_video_prefill") or max(1, self.action_horizon // self.replan_steps)
            )
            self.video_prefill_action_horizon = self._resolve_video_prefill_action_horizon(self.replan_steps)
            print("[aha-wam] allow_dummy_policy=true; real model loading is skipped.")
        else:
            self.policy = self._load_policy()
            self.action_horizon = int(self.model_cfg.get("action_horizon") or self.policy.action_horizon)
            self.replan_steps = int(self.model_cfg.get("replan_steps") or self.policy.model.action_chunk_size)
            self.replan_steps = min(self.replan_steps, int(self.policy.model.action_chunk_size))
            self.chunks_per_video_prefill = self._resolve_chunks_per_video_prefill()
            self.video_prefill_action_horizon = self._resolve_video_prefill_action_horizon(
                int(self.policy.model.action_chunk_size)
            )
            self._warmup()
        print(
            "[aha-wam] initialized "
            f"ckpt={self.checkpoint_path} stats={self.dataset_stats_path} "
            f"horizon={self.action_horizon} replan_steps={self.replan_steps} "
            f"num_chunks={getattr(self.policy, 'num_chunks', self.action_horizon // max(self.replan_steps, 1))} "
            f"chunks_per_video_prefill={self.chunks_per_video_prefill} "
            f"video_prefill_horizon={self.video_prefill_action_horizon}"
        )

    def _compose_cfg(self):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        configs_root = self.elava_root / "configs"
        task_name = str(self.model_cfg.get("task_config") or self.model_cfg.get("sim_task"))
        overrides = [f"task={task_name}"]
        if "prepend_episode_first_frame" in self.model_cfg:
            overrides.append(
                "model.prepend_episode_first_frame="
                f"{str(_parse_bool(self.model_cfg.get('prepend_episode_first_frame'))).lower()}"
            )
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
            return compose(config_name="train.yaml", overrides=overrides)

    def _load_policy(self):
        from hydra.utils import instantiate
        from ahawam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        from omegaconf import OmegaConf

        cfg = self._compose_cfg()
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.load_text_encoder = True
        device = str(self.model_cfg.get("device") or "cuda")
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[aha-wam] CUDA is unavailable; falling back to cpu.")
            device = "cpu"

        model = instantiate(
            model_cfg,
            model_dtype=_model_dtype(str(self.model_cfg.get("mixed_precision") or "bf16")),
            device=device,
        )
        model.load_checkpoint(str(self.checkpoint_path))
        model = model.to(device).eval()

        processor = instantiate(cfg.data.train.processor).eval()
        processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(self.dataset_stats_path)))

        class _Policy:
            pass

        policy = _Policy()
        policy.model = model
        policy.processor = processor
        policy.action_horizon = int(self.model_cfg.get("action_horizon") or int(cfg.data.train.num_frames) - 1)
        policy.num_inference_steps = _parse_optional_int(self.model_cfg.get("num_inference_steps"))
        policy.sigma_shift = _parse_optional_float(self.model_cfg.get("sigma_shift"))
        policy.seed = _parse_optional_int(self.model_cfg.get("seed"))
        policy.text_cfg_scale = float(self.model_cfg.get("text_cfg_scale", 1.0))
        policy.negative_prompt = str(self.model_cfg.get("negative_prompt", ""))
        policy.rand_device = str(self.model_cfg.get("rand_device", "cpu"))
        policy.tiled = _parse_bool(self.model_cfg.get("tiled", False))
        policy.timing_enabled = _parse_bool(self.model_cfg.get("timing_enabled", False))
        policy.num_chunks = int(policy.action_horizon // model.action_chunk_size)
        policy.episode_prefilled = False
        return policy

    def _resolve_chunks_per_video_prefill(self) -> int:
        configured = self.model_cfg.get("chunks_per_video_prefill")
        if _is_none_like(configured):
            chunks = int(self.policy.num_chunks)
        else:
            chunks = int(configured)
        if chunks <= 0:
            raise ValueError(f"chunks_per_video_prefill must be positive, got {chunks}.")
        if chunks > int(self.policy.num_chunks):
            raise ValueError(
                "chunks_per_video_prefill cannot exceed chunks in action_horizon: "
                f"{chunks} > {self.policy.num_chunks}."
            )
        return chunks

    def _resolve_video_prefill_action_horizon(self, chunk_size: int) -> int:
        horizon = int(self.chunks_per_video_prefill) * int(chunk_size)
        if horizon <= 0:
            raise ValueError(f"video_prefill_action_horizon must be positive, got {horizon}.")
        if horizon > int(self.action_horizon):
            raise ValueError(
                "video_prefill_action_horizon cannot exceed action_horizon: "
                f"{horizon} > {self.action_horizon}."
            )
        return horizon

    def _warmup(self) -> None:
        dummy_image = torch.zeros(
            (1, 3, 384, 320),
            device=self.policy.model.device,
            dtype=self.policy.model.torch_dtype,
        )
        dummy_proprio = None
        if self.policy.model.proprio_dim is not None:
            dummy_proprio = torch.zeros(
                (1, self.policy.model.proprio_dim),
                device=self.policy.model.device,
                dtype=torch.float32,
            )
        params = inspect.signature(self.policy.model.infer_action).parameters
        prompt = self.default_prompt_template.format(task="warmup")
        with torch.no_grad():
            if "phase" in params:
                self.policy.model.infer_action(
                    prompt=prompt,
                    input_image=dummy_image,
                    action_horizon=self.video_prefill_action_horizon,
                    negative_prompt=self.policy.negative_prompt,
                    text_cfg_scale=self.policy.text_cfg_scale,
                    num_inference_steps=self.policy.num_inference_steps,
                    sigma_shift=self.policy.sigma_shift,
                    seed=0,
                    rand_device=self.policy.rand_device,
                    tiled=self.policy.tiled,
                    phase="video",
                )
                self.policy.model.infer_action(
                    prompt=None,
                    input_image=dummy_image,
                    action_horizon=self.video_prefill_action_horizon,
                    chunk_obs_image=dummy_image,
                    chunk_proprio=dummy_proprio,
                    negative_prompt=self.policy.negative_prompt,
                    text_cfg_scale=self.policy.text_cfg_scale,
                    num_inference_steps=self.policy.num_inference_steps,
                    sigma_shift=self.policy.sigma_shift,
                    seed=0,
                    rand_device=self.policy.rand_device,
                    tiled=self.policy.tiled,
                    phase="action",
                )
                if hasattr(self.policy.model, "_inference_state"):
                    self.policy.model._inference_state = None
                if hasattr(self.policy.model, "reset_history"):
                    self.policy.model.reset_history()

    def _build_image_tensor(self, obs: dict) -> torch.Tensor:
        vision = obs["vision"]
        head = _resize_rgb(vision["cam_head"]["color"], (320, 256))
        left = _resize_rgb(vision["cam_left_wrist"]["color"], (160, 128))
        right = _resize_rgb(vision["cam_right_wrist"]["color"], (160, 128))
        image = np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=self.policy.model.device,
            dtype=self.policy.model.torch_dtype,
        )
        return tensor * (2.0 / 255.0) - 1.0

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.policy.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected one merged state key in processor shape_meta.")
        state_key = state_meta[0]["key"]
        batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        batch = self.policy.processor.action_state_transform(batch)
        batch = self.policy.processor.normalizer.forward(batch)
        return batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        action_meta = self.policy.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected one merged action key in processor shape_meta.")
        action_key = action_meta[0]["key"]
        normalizer = self.policy.processor.normalizer.normalizers["action"][action_key]
        return normalizer.backward(action.to(dtype=torch.float32, device="cpu")).numpy()

    def _prefill_episode(self, prompt: str, image_tensor: torch.Tensor) -> None:
        self.policy.model.infer_action(
            prompt=prompt,
            input_image=image_tensor,
            action_horizon=self.video_prefill_action_horizon,
            negative_prompt=self.policy.negative_prompt,
            text_cfg_scale=self.policy.text_cfg_scale,
            seed=self.policy.seed,
            rand_device=self.policy.rand_device,
            tiled=self.policy.tiled,
            phase="video",
        )
        self.policy.episode_prefilled = True
        self._chunks_since_video_prefill = 0

    def _model_num_history_frames(self) -> int:
        getter = getattr(self.policy.model, "_configured_num_history_frames", None)
        if callable(getter):
            return int(getter())
        return int(getattr(self.policy.model, "num_history_frames", 0))

    def _reset_chunk_rollout_state(self) -> None:
        if hasattr(self.policy.model, "reset_history"):
            self.policy.model.reset_history()
        if hasattr(self.policy.model, "_inference_state"):
            self.policy.model._inference_state = None
        self.policy.episode_prefilled = False
        self._chunks_since_video_prefill = 0

    def _soft_reset_for_new_observation(self) -> None:
        if self._model_num_history_frames() <= 0:
            self._reset_chunk_rollout_state()
            return
        self.policy.episode_prefilled = False
        self._chunks_since_video_prefill = 0

    def _predict_chunk(self, obs: dict, instruction: str) -> np.ndarray:
        image_tensor = self._build_image_tensor(obs)
        state = _pack_robot_state(obs, self.action_type, self.robot_action_dim_info).astype(np.float32)
        proprio = self._normalize_state(state)
        prompt = self.default_prompt_template.format(task=instruction)

        with torch.no_grad():
            if not self.policy.episode_prefilled:
                self._prefill_episode(prompt=prompt, image_tensor=image_tensor)
            kwargs = {
                "chunk_obs_image": image_tensor,
                "chunk_proprio": proprio,
                "sigma_shift": self.policy.sigma_shift,
                "tiled": self.policy.tiled,
                "phase": "action",
            }
            if self.policy.num_inference_steps is not None:
                kwargs["num_inference_steps"] = self.policy.num_inference_steps
            pred = self.policy.model.infer_action(**kwargs)

        return self._denormalize_action(pred["action_chunk"].unsqueeze(0))[0]

    def _fill_action_queue(self, obs: dict, instruction: str) -> None:
        if self.allow_dummy_policy:
            dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"])
            for action in np.zeros((self.replan_steps, dim), dtype=np.float32):
                self.pending_actions.append(action)
            return

        state = getattr(self.policy.model, "_inference_state", None)
        next_chunk_index = 0 if state is None else int(state.get("next_chunk_index", 0))
        if (
            next_chunk_index >= self.chunks_per_video_prefill
            or self._chunks_since_video_prefill >= self.chunks_per_video_prefill
        ):
            self._soft_reset_for_new_observation()
        chunk = self._predict_chunk(obs, instruction)
        self._chunks_since_video_prefill += 1
        chunk = chunk[: self.replan_steps]
        for action in chunk:
            self.pending_actions.append(np.asarray(action, dtype=np.float32))

    def update_obs(self, obs):
        self.last_obs = obs
        self.last_instruction = _get_instruction(obs, self.default_instruction)

    def update_obs_batch(self, obs_list):
        self._batch_obs = {}
        self._batch_instruction = {}
        for obs in obs_list:
            env_idx = int(obs["env_idx"])
            self._batch_obs[env_idx] = obs
            self._batch_instruction[env_idx] = _get_instruction(obs, self.default_instruction)

    def get_action(self):
        if self.last_obs is None:
            raise ValueError("No observation available. Call update_obs before get_action.")
        if not self.pending_actions:
            self._fill_action_queue(self.last_obs, self.last_instruction)
        actions = list(self.pending_actions)
        self.pending_actions.clear()
        return _unpack_robot_state(actions, self.action_type, self.robot_action_dim_info)

    def get_action_batch(self, env_idx_list):
        # The underlying AHAWAM history state is single-rollout stateful. Use one process per
        # environment for true parallel eval; this fallback is for debug compatibility only.
        results = []
        for env_idx in env_idx_list:
            self.update_obs(self._batch_obs[int(env_idx)])
            results.append(self.get_action())
        return results

    def reset(self):
        self.pending_actions.clear()
        self.last_obs = None
        self.last_instruction = self.default_instruction
        self._chunks_since_video_prefill = 0
        if self.allow_dummy_policy:
            return
        self._reset_chunk_rollout_state()
