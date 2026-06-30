import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_CUR_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CUR_DIR.parents[3]
_WAN_VA_ROOT = _CUR_DIR / "lingbot_va" / "wan_va"

for _path in (str(_REPO_ROOT), str(_WAN_VA_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    decode_image_bit,
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

from .lingbot_va.wan_va.configs import VA_CONFIGS
from .lingbot_va.wan_va.distributed.util import init_distributed
from .lingbot_va.wan_va.wan_va_server import VA_Server


DEFAULT_CONFIG_NAME = "robotwin30_train"


def resolve_lingbot_wan_paths(
    checkpoint_path: str,
    base_model_path: str,
) -> tuple[str, str]:
    """Resolve base (vae/tokenizer/text_encoder) and finetuned transformer paths."""
    if not checkpoint_path:
        raise ValueError("checkpoint_path is required (set in deploy.yml or eval overrides).")
    if not base_model_path:
        raise ValueError("base_model_path is required (set in deploy.yml).")

    ckpt_root = Path(checkpoint_path).expanduser().resolve()
    if not ckpt_root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_root}")

    transformer_candidates = (
        ckpt_root / "checkpoints" / "transformer",
        ckpt_root / "transformer",
    )
    for transformer_path in transformer_candidates:
        if (transformer_path / "config.json").exists():
            base_root = Path(base_model_path).expanduser().resolve()
            if not (base_root / "vae").is_dir():
                raise FileNotFoundError(
                    f"Base model directory missing vae/: {base_root}. "
                    "Set base_model_path in deploy.yml."
                )
            return str(base_root), str(transformer_path)

    if (ckpt_root / "vae").is_dir():
        transformer_path = ckpt_root / "transformer"
        if not (transformer_path / "config.json").exists():
            raise FileNotFoundError(f"Transformer checkpoint not found under: {ckpt_root}")
        return str(ckpt_root), str(transformer_path)

    raise FileNotFoundError(
        f"Unrecognized checkpoint layout under {ckpt_root}. "
        "Expected checkpoints/transformer/ (SFT export) or vae/ + transformer/ (full bundle)."
    )

JOINT_CONTROL_INDICES = np.array([
    14, 15, 16, 17, 18, 19,
    28,
    21, 22, 23, 24, 25, 26,
    29,
], dtype=np.int64)


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
        image = decode_image_bit(np.frombuffer(bytes(image), dtype=np.uint8))

    image = np.asarray(image)
    if image.ndim == 1 and image.dtype == np.uint8:
        image = decode_image_bit(image)

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


def encode_obs(observation, action_type, robot_action_dim_info, default_prompt):
    images = {
        "cam_high": ensure_hwc_uint8(
            extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"])
        ),
        "cam_left_wrist": ensure_hwc_uint8(
            extract_image(observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"])
        ),
        "cam_right_wrist": ensure_hwc_uint8(
            extract_image(observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"])
        ),
    }

    if robot_action_dim_info is None:
        state = np.zeros((1,), dtype=np.float32)
    else:
        state = pack_robot_state(
            observation,
            action_type,
            robot_action_dim_info,
            source_type="obs",
        ).astype(np.float32)

    prompt = observation.get("prompt", default_prompt)

    return {
        "observation.images.cam_high": images["cam_high"],
        "observation.images.cam_left_wrist": images["cam_left_wrist"],
        "observation.images.cam_right_wrist": images["cam_right_wrist"],
        "observation.state": state,
        "task": prompt,
    }


class Model(ModelTemplate):
    def __init__(self, model_cfg) -> None:
        self.model_cfg = dict(model_cfg)

        self.model_cfg.setdefault("config_name", DEFAULT_CONFIG_NAME)

        self.task_name = self.model_cfg.get("task_name", "default_task")
        self.action_type = self.model_cfg["action_type"]
        self.default_prompt = self.model_cfg.get("prompt", self.task_name)

        env_cfg = self.model_cfg.get("env_cfg")
        if env_cfg is None:
            self.robot_action_dim_info = None
        else:
            try:
                self.robot_action_dim_info = get_robot_action_dim_info(env_cfg)
            except FileNotFoundError:
                print(f"[WARN] env_cfg '{env_cfg}' not found, fallback to raw action mode.")
                self.robot_action_dim_info = None

        self.observation_window: list[dict[str, Any]] | None = None
        self._latest_env_idx_list: list[int] = [0]
        # Per-env streaming state. LingBot_VA generates action chunks autoregressively
        # and must refresh its KV cache from the executed-step observations before the
        # next chunk. The standard eval loop only calls update_obs/get_action, so we
        # buffer those observations here and drive the KV-cache update internally.
        self._env_states: dict[int, dict[str, Any]] = {}

        self.vla = self.get_model(self.model_cfg)

    def get_model(self, model_cfg):
        config_name = model_cfg.get("config_name", DEFAULT_CONFIG_NAME)
        if config_name not in VA_CONFIGS:
            raise KeyError(f"Unknown config_name: {config_name}")

        job_config = copy.deepcopy(VA_CONFIGS[config_name])

        if model_cfg.get("save_root") is not None:
            job_config.save_root = model_cfg["save_root"]

        checkpoint_path = (
            model_cfg.get("checkpoint_path")
            or model_cfg.get("wan22_pretrained_model_name_or_path")
        )
        base_model_path = model_cfg.get("base_model_path")
        base_path, transformer_path = resolve_lingbot_wan_paths(
            str(checkpoint_path) if checkpoint_path else "",
            str(base_model_path) if base_model_path else "",
        )

        model_cfg["checkpoint_path"] = transformer_path
        job_config.wan22_pretrained_model_name_or_path = base_path
        job_config.transformer_pretrained_path = transformer_path
        print(
            f"[LingBot_VA] base_model={base_path}, transformer={transformer_path}",
            flush=True,
        )
        if hasattr(job_config, "infer_mode"):
            job_config.infer_mode = "server"

        rank = int(os.getenv("RANK", 0))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))

        if world_size > 1:
            init_distributed(world_size, local_rank, rank)

        job_config.rank = rank
        job_config.local_rank = local_rank
        job_config.world_size = world_size

        model = VA_Server(job_config)
        model._reset(prompt=self.default_prompt)
        return model

    def _to_engine_obs(self, observation):
        image_dict = {
            key: observation[key]
            for key in self.vla.job_config.obs_cam_keys
        }
        return {
            "obs": [image_dict],
            "state": observation["observation.state"],
            "prompt": observation["task"],
        }

    def _to_engine_obs_batch(self, observation_list, state=None):
        obs_batch = []
        prompt = self.default_prompt

        for observation in observation_list:
            image_dict = {
                key: observation[key]
                for key in self.vla.job_config.obs_cam_keys
            }
            obs_batch.append(image_dict)
            prompt = observation.get("task", prompt)

        payload = {
            "obs": obs_batch,
            "prompt": prompt,
        }
        if state is not None:
            payload["state"] = state
        return payload

    def _format_action_chunk(self, action):
        action = np.asarray(action)

        if action.ndim == 3:
            action = np.transpose(action, (1, 2, 0)).reshape(-1, action.shape[0])
        elif action.ndim == 2 and action.shape[0] == self.vla.job_config.action_dim:
            action = action.T

        return action

    def _convert_to_joint_control_chunk(self, action_chunk):
        action_chunk = np.asarray(action_chunk)

        if action_chunk.ndim != 2:
            raise ValueError(f"Expected action chunk with ndim=2, got shape {action_chunk.shape}.")

        if action_chunk.shape[1] == len(JOINT_CONTROL_INDICES):
            return action_chunk

        if action_chunk.shape[1] != 30:
            raise ValueError(
                "LingBot_VA joint-control conversion expects raw action dim 30 or already-converted dim 14, "
                f"got {action_chunk.shape[1]}."
            )

        return action_chunk[:, JOINT_CONTROL_INDICES]

    def _new_env_state(self):
        return {
            "first_obs": None,            # anchor observation reused for streaming infer
            "raw_action": None,           # last raw (C, F, H) action chunk for KV-cache state
            "skip_leading": True,         # trim the conditioning frame from the first chunk
            "inferred_once": False,       # whether the first chunk has been generated
            "cache_buffer": [],           # executed-step observations awaiting KV-cache update
        }

    def _get_env_state(self, env_idx):
        state = self._env_states.get(env_idx)
        if state is None:
            state = self._new_env_state()
            self._env_states[env_idx] = state
        return state

    def _select_keyframes(self, cache_buffer):
        """Pick one observation per generated frame (every action_per_frame steps).

        The streaming VAE/KV-cache expects keyframes aligned with the action frames
        (frame_chunk_size); the executed-step observations arrive one per action step,
        so we keep the observation at each action-frame boundary.
        """
        if not cache_buffer:
            return []
        step = int(getattr(self.vla.job_config, "action_per_frame", 0)) or 1
        keyframes = cache_buffer[step - 1::step]
        if not keyframes:
            keyframes = [cache_buffer[-1]]
        return keyframes

    def _maybe_trim_initial_action_chunk(self, state, action_chunk):
        action_chunk = np.asarray(action_chunk)

        if not state["skip_leading"]:
            return action_chunk

        skip_count = int(getattr(self.vla.job_config, "action_per_frame", 0))
        if skip_count <= 0:
            return action_chunk

        if action_chunk.shape[0] <= skip_count:
            raise ValueError(
                "Initial-action trimming would remove the whole chunk: "
                f"chunk_len={action_chunk.shape[0]}, skip_count={skip_count}."
            )

        state["skip_leading"] = False
        return action_chunk[skip_count:]

    def _infer_env(self, env_idx, encoded_obs):
        """Generate one action chunk for a single env, advancing its KV cache first."""
        state = self._get_env_state(env_idx)

        if not state["inferred_once"]:
            # First chunk: anchor on this observation; VA_Server encodes it (frame_st_id=0).
            state["first_obs"] = encoded_obs
        else:
            # Subsequent chunk: refresh the KV cache with the executed-step observations
            # before generating. Skipping this would re-encode frame_st_id=0 every call,
            # which crashes the causal streaming VAE (conv3d kernel > input frames).
            keyframes = self._select_keyframes(state["cache_buffer"])
            if keyframes and state["raw_action"] is not None:
                cache_obs = self._to_engine_obs_batch(keyframes, state=state["raw_action"])
                cache_obs["compute_kv_cache"] = True
                self.vla.infer(cache_obs)
            state["cache_buffer"] = []

        model_obs = self._to_engine_obs(state["first_obs"])
        result = self.vla.infer(model_obs)
        state["raw_action"] = np.asarray(result["action"])
        state["inferred_once"] = True

        action = self._format_action_chunk(result["action"])
        action = self._convert_to_joint_control_chunk(action)
        action = self._maybe_trim_initial_action_chunk(state, action)
        return action

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        encoded_obs_list = [
            encode_obs(obs, self.action_type, self.robot_action_dim_info, self.default_prompt)
            for obs in obs_list
        ]
        self.observation_window = encoded_obs_list

        # Buffer executed-step observations so the next get_action can refresh the KV cache.
        for index, env_idx in enumerate(self._latest_env_idx_list):
            state = self._env_states.get(env_idx)
            if state is not None and state["inferred_once"]:
                state["cache_buffer"].append(encoded_obs_list[index])

    def get_action(self, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        env_idx = self._latest_env_idx_list[0] if self._latest_env_idx_list else 0
        action_chunk = self._infer_env(env_idx, self.observation_window[0])

        if self.robot_action_dim_info is None:
            return action_chunk

        return unpack_robot_state(
            action_chunk,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
        )

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        env_idx_list = env_idx_list or self._latest_env_idx_list
        action_list = []

        # VA_Server holds a single streaming state, so envs are stepped sequentially
        # (brute-force) rather than truly batched. Correct for one running env at a time.
        for batch_index, env_idx in enumerate(env_idx_list):
            action_chunk = self._infer_env(env_idx, self.observation_window[batch_index])

            if self.robot_action_dim_info is None:
                action_list.append(action_chunk)
            else:
                action_list.append(
                    unpack_robot_state(
                        action_chunk,
                        self.action_type,
                        self.robot_action_dim_info,
                        source_type="obs",
                    )
                )

        return action_list

    def get_action_per_frame(self):
        return int(getattr(self.vla.job_config, "action_per_frame", 0))

    def reset(self, checkpoint_path=None) -> None:
        if checkpoint_path is not None:
            self.model_cfg["checkpoint_path"] = checkpoint_path
            self.vla = self.get_model(self.model_cfg)
        else:
            self.vla._reset(prompt=self.default_prompt)

        self.observation_window = None
        self._latest_env_idx_list = [0]
        self._env_states = {}


def _make_fake_obs(prompt: str):
    h, w = 224, 224
    return {
        "vision": {
            "cam_high": np.random.randint(0, 255, (h, w, 3), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(0, 255, (h, w, 3), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(0, 255, (h, w, 3), dtype=np.uint8),
        },
        "joint_action": {
            "left_gripper": 0.0,
            "left_arm": [0.0] * 6,
            "right_gripper": 0.0,
            "right_arm": [0.0] * 6,
        },
        "endpose": {
            "left_endpose": [0.0] * 6,
            "right_endpose": [0.0] * 6,
        },
        "prompt": prompt,
        "env_idx": 0,
    }