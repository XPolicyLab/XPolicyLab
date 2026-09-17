"""Standalone, eval-only SIPAI adapter for XPolicyLab."""

from pathlib import Path
import random

import numpy as np
from safetensors.torch import load_file
import sentencepiece
import torch

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info

from .history import CameraHistory
from .network import SIPAINetwork
from .processing import CAMERAS, action_dict, camera_image, decode_actions, prepare

POLICY_DIR = Path(__file__).resolve().parent


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        if model_cfg.get("action_type") != "joint":
            raise ValueError("SIPAI requires action_type=joint")
        dims = get_robot_action_dim_info(model_cfg["env_cfg_type"])
        if list(dims["arm_dim"]) != [6, 6] or list(dims["ee_dim"]) != [1, 1]:
            raise ValueError("Checkpoint requires two six-joint arms and scalar grippers")
        self.robot_action_dim_info = dims
        root = resolve_checkpoint_root(
            model_cfg,
            POLICY_DIR / "checkpoints",
            policy_dir=POLICY_DIR,
            explicit_keys=("checkpoint_path",),
        )
        cfg = self.model_cfg
        if cfg["architecture"] != "sipai_pi05_memory_joint_v1" or cfg["dtype"] != "float32":
            raise ValueError("Unsupported SIPAI deployment architecture or precision")
        self.device = torch.device(model_cfg.get("device", "cuda:0"))
        self.exec_chunk_size = int(model_cfg.get("exec_chunk_size", 10))
        if not 1 <= self.exec_chunk_size <= 50:
            raise ValueError("exec_chunk_size must be between 1 and 50")
        self.token_length = int(cfg["max_token_len"])
        self.steps = int(cfg["num_inference_steps"])
        if self.token_length <= 0 or self.steps <= 0 or not 1 <= cfg["history_pool_size"] <= 16:
            raise ValueError("Invalid inference dimensions")
        self.history = CameraHistory(cfg["history_frames"], cfg["history_interval"])
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(root / "tokenizer.model"))
        with torch.device("meta"):
            self.network = SIPAINetwork(cfg["history_pool_size"])
        weights = load_file(str(root / "model.safetensors"), device="cpu")
        if any(value.dtype != torch.float32 for value in weights.values()):
            raise ValueError("This release expects FP32 weights")
        self.network.load_state_dict(weights, strict=True, assign=True)
        self.network.to(self.device).eval().requires_grad_(False)
        seed = int(model_cfg.get("seed") or 0)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.observations, self.latest = {}, {}
        print(
            f"[SIPAI] standalone; history={cfg['history_frames']}"
            f"x{cfg['history_interval']} steps; "
            f"exec={self.exec_chunk_size}; fp32; gripper compensation=0",
            flush=True,
        )

    @staticmethod
    def _scope(scope):
        if scope is None:
            return None
        value = scope.get("evaluation_id")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Scope requires a non-empty evaluation_id")
        return value

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        rows = list(obs_list)
        if not rows:
            raise ValueError("Observation batch cannot be empty")
        scope = rows[0].get("evaluation_id")
        if scope is not None:
            self._scope(rows[0])
        indices = [int(row.get("env_idx", i)) for i, row in enumerate(rows)]
        if len(set(indices)) != len(indices) or any(row.get("evaluation_id") != scope for row in rows):
            raise ValueError("Batch requires unique environment IDs and one evaluation ID")
        images = [camera_image(row, CAMERAS[0]) for row in rows]
        for row, index, image in zip(rows, indices, images, strict=True):
            self.observations[scope, index] = dict(row)
            self.history.update(image, evaluation_id=scope, env_idx=index)
        self.latest[scope] = indices

    def _infer(self, scope, index):
        if (scope, index) not in self.observations:
            raise RuntimeError("Call update_obs before requesting actions")
        history = self.history.frames_for(evaluation_id=scope, env_idx=index)
        prepared, state = prepare(
            self.observations[scope, index],
            history,
            self.tokenizer,
            self.token_length,
            self.device,
        )
        normalized = self.network(prepared, steps=self.steps)[0].cpu().numpy()
        if not np.isfinite(normalized).all():
            raise FloatingPointError("SIPAI produced non-finite actions")
        actions = decode_actions(normalized, state)
        return [action_dict(row, self.robot_action_dim_info) for row in actions[: self.exec_chunk_size]]

    def get_action(self, scope=None):
        scope = self._scope(scope)
        return self._infer(scope, self.latest.get(scope, [0])[0])

    def get_action_batch(self, env_idx_list=None):
        scope = self._scope(env_idx_list) if isinstance(env_idx_list, dict) else None
        indices = env_idx_list.get("env_idx_list") if isinstance(env_idx_list, dict) else env_idx_list
        if indices is None:
            indices = self.latest.get(scope, [0])
        # Follow the shared sequential-batch convention; keep requested ID order.
        return [self._infer(scope, int(index)) for index in indices]

    def reset(self):
        self.observations.clear()
        self.latest.clear()
        self.history.reset()

    def reset_evaluation(self, scope):
        scope = self._scope(scope)
        if scope is None:
            raise ValueError("reset_evaluation requires an evaluation ID")
        for key in list(self.observations):
            if key[0] == scope:
                del self.observations[key]
        self.latest.pop(scope, None)
        self.history.reset(scope)
