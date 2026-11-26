"""XPolicyLab adapter: one local model, independent state and RNG per episode."""

import hashlib
import json
import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info

from .attention import attention_runtime, validate_backend
from .network import load_network
from .processing import (CAMERAS, Checkpoint, camera_geometry, decode_actions, encode_state,
                         nonnegative_integer)


def boolean_option(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean (true or false), not {value!r}")
    return value


def episode_identity(obs):
    value = obs.get("episode_identity")
    if not isinstance(value, dict) or not isinstance(value.get("task"), str) or not value["task"].strip():
        raise ValueError("Observations require episode_identity with task, eval_seed and layout_id")
    return (value["task"], nonnegative_integer(value.get("eval_seed"), "eval_seed"),
            nonnegative_integer(value.get("layout_id"), "layout_id"))


def request_seed(policy_seed, identity, control_step):
    key = json.dumps(["episode_control_step_v1", policy_seed, *identity, control_step],
                     ensure_ascii=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)


@dataclass
class Episode:
    identity: tuple
    first_step: int
    control_step: int
    history: deque
    first_frame: tuple
    instruction: str = ""
    world_to_camera: np.ndarray | None = None
    state: np.ndarray | None = None
    state_mask: np.ndarray | None = None
    fov: np.ndarray | None = None


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        self.steps = nonnegative_integer(model_cfg.get("num_inference_steps", 10), "num_inference_steps")
        self.execute_steps = nonnegative_integer(model_cfg.get("execute_steps", 20), "execute_steps")
        self.auto_cast_bf16 = boolean_option(model_cfg.get("auto_cast_bf16", True), "auto_cast_bf16")
        self.use_ema = boolean_option(model_cfg.get("use_ema", False), "use_ema")
        self.attn_backend = validate_backend(model_cfg.get("attn_backend", "fa4"))
        if self.steps < 1 or self.execute_steps < 1:
            raise ValueError("num_inference_steps and execute_steps must be positive")
        if model_cfg.get("env_cfg_type") != "arx_x5" or model_cfg.get("action_type") != "ee":
            raise ValueError("CogACT2 currently supports RoboDojo arx_x5 with action_type=ee")
        if str(model_cfg.get("bench_name", "RoboDojo")).lower() != "robodojo":
            raise ValueError("This adapter uses RoboDojo camera and gripper conventions")
        dims = get_robot_action_dim_info(model_cfg["env_cfg_type"])
        if len(dims["arm_dim"]) != 2 or dims["ee_dim"] != [1, 1]:
            raise ValueError("Expected two arms with one gripper value per arm")
        self.policy_seed = nonnegative_integer(model_cfg.get("policy_seed", 42), "policy_seed")
        if self.policy_seed >= 2**63:
            raise ValueError("policy_seed must be in [0, 2**63)")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        random.seed(self.policy_seed)
        np.random.seed(self.policy_seed % 2**32)
        torch.manual_seed(self.policy_seed)
        torch.cuda.manual_seed_all(self.policy_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        policy_dir = Path(__file__).resolve().parent
        directory = resolve_checkpoint_root(model_cfg, policy_dir / "checkpoints", policy_dir=policy_dir)
        self.checkpoint = Checkpoint(directory, model_cfg.get("unnorm_key"))
        if self.execute_steps > self.checkpoint.robot_chunk_size:
            raise ValueError("execute_steps exceeds the checkpoint's robot-action horizon")
        self.reset()
        self.model, self.processor = load_network(
            self.checkpoint, backbone_path=model_cfg.get("backbone_path"), use_ema=self.use_ema,
            attn_backend=self.attn_backend, auto_cast_bf16=self.auto_cast_bf16)
        attention_runtime(self.model)
        self._warmup()
        print("[CogACT2] startup inference passed; model is ready", flush=True)

    def reset(self):
        # Reset changes memory only; sampling is keyed by simulator metadata.
        self._episodes = {}
        self._active_ids = []

    def prepare_case(self, case_meta=None):
        self.reset()

    def _warmup(self):
        """Run the actual image/state/memory/sampler/decode path before serving."""
        print("[CogACT2] running startup inference", flush=True)
        pose = np.eye(4)
        pose[2, 3] = 1.2
        obs = {
            "env_idx": 0, "env_origin": [0, 0, 0], "control_step": 0,
            "episode_identity": {"task": "__cogact2_warmup__", "eval_seed": 0, "layout_id": 0},
            "instruction": "Stack the bowls.",
            "vision": {name: {
                "color": np.full((480, 640, 3), 127, dtype=np.uint8),
                "intrinsic_matrix": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                "extrinsics_matrix": pose.copy(),
            } for name in CAMERAS},
            "state": {f"{side}_{field}": value for side, y in (("left", .2), ("right", -.2))
                      for field, value in (("ee_pose", [.4, y, .4, 1, 0, 0, 0]), ("ee_joint_state", [.5]))},
        }
        try:
            self.update_obs(obs)
            self._predict(self._episodes[0])
        finally:
            self.reset()

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        ids = [nonnegative_integer(obs.get("env_idx", 0), "env_idx") for obs in obs_list]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate environment indices in observation batch")
        prepared = []
        for env_idx, obs in zip(ids, obs_list):
            identity = episode_identity(obs)
            step = nonnegative_integer(obs.get("control_step"), "control_step")
            previous = self._episodes.get(env_idx)
            if previous is not None and previous.identity == identity:
                if step < previous.control_step:
                    raise ValueError("Control step moved backwards; reset before replaying an episode")
                if self.checkpoint.memory.offsets and step > previous.control_step + 1:
                    raise ValueError("Image history requires an observation after every control step")
            images = [self.checkpoint.image(obs["vision"][camera]["color"]) for camera in CAMERAS]
            transform, fov = camera_geometry(obs)
            state, mask = encode_state(obs, transform, self.checkpoint)
            instruction = self.checkpoint.instruction(obs)
            prepared.append((env_idx, identity, step, images, transform, fov, state, mask, instruction))
        # Validate the entire batch before changing any episode's history.
        for env_idx, identity, step, images, transform, fov, state, mask, instruction in prepared:
            episode = self._episodes.get(env_idx)
            if episode is None or episode.identity != identity:
                episode = Episode(identity, step, step, deque(maxlen=self.checkpoint.memory.history_length),
                                  tuple(image.copy() for image in images))
                self._episodes[env_idx] = episode
            if episode.history and episode.control_step == step:
                episode.history[-1] = images
            else:
                episode.history.append(images)
            episode.control_step, episode.instruction = step, instruction
            episode.world_to_camera, episode.fov = transform, fov
            episode.state, episode.state_mask = state, mask
        self._active_ids = ids

    def _predict(self, episode):
        images, kinds = self.checkpoint.memory.assemble(episode.first_frame, list(episode.history))
        tokens = self.processor(images, kinds, episode.instruction)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.auto_cast_bf16,
                            cache_enabled=False):
            values = self.model.sample(*tokens, episode.state, episode.state_mask, episode.fov,
                                       request_seed(self.policy_seed, episode.identity, episode.control_step), self.steps,
                                       first_frame_offset=episode.first_step - episode.control_step)
        return decode_actions(values, episode.world_to_camera, self.checkpoint)

    def get_action(self):
        if len(self._active_ids) != 1:
            raise ValueError("get_action requires one observation; use get_action_batch for multiple environments")
        return self.get_action_batch(self._active_ids)[0]

    def get_action_batch(self, env_idx_list=None):
        ids = self._active_ids if env_idx_list is None else [nonnegative_integer(i, "env_idx") for i in env_idx_list]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Expected distinct active environment indices")
        episodes = [self._episodes[i] for i in ids]
        # Fixed single-sample execution also avoids batch-shape-dependent floating
        # point differences when environments finish at different times.
        predictions = [self._predict(episode) for episode in episodes]
        return [actions[:self.execute_steps] for actions in predictions]
