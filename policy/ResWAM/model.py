"""XPolicyLab adapter for ResWAM on RoboDojo (arx_x5, absolute joint targets).

The policy itself lives in the upstream repository (``policy/ResWAM/ResWAM``, cloned at a
pinned commit by ``install.sh``); this file only maps the XPolicyLab Model contract onto
``RoboDojoPolicy`` from ``ResWAM/scripts/eval/eval_robodojo.py``, which is the exact code
path behind the paper's RoboDojo numbers. Images arrive already decoded (RGB arrays).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info

POLICY_DIR = Path(__file__).resolve().parent
UPSTREAM_DIR = Path(os.environ.get("RESWAM_REPO") or POLICY_DIR / "ResWAM").resolve()
CHECKPOINTS_DIR = POLICY_DIR / "checkpoints"


def _load_upstream():
    # Load by file path: the RoboDojo workspace also has a top-level ``scripts`` package.
    path = UPSTREAM_DIR / "scripts" / "eval" / "eval_robodojo.py"
    if not path.is_file():
        raise FileNotFoundError(f"ResWAM source not found at {UPSTREAM_DIR}; run install.sh or set RESWAM_REPO")
    spec = importlib.util.spec_from_file_location("reswam_eval_robodojo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _none(value):
    return value is None or str(value).strip().lower() in {"", "null", "none"}


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        if self.model_cfg["action_type"] != "joint":
            raise ValueError(f"ResWAM is trained on absolute joint targets, got action_type={self.model_cfg['action_type']!r}")
        dims = get_robot_action_dim_info(self.model_cfg["env_cfg_type"])
        if list(dims["arm_dim"]) != [6, 6] or list(dims["ee_dim"]) != [1, 1]:
            raise ValueError(f"ResWAM-RoboDojo expects a 2x(6+1) joint robot (arx_x5), got {dims}")

        upstream = _load_upstream()
        self._derive_seed = upstream.derive_seed
        self._unpack_action = upstream.unpack_action
        self.execute_horizon = int(self.model_cfg.get("execute_horizon") or 10)
        if not 1 <= self.execute_horizon <= upstream.HORIZON:
            raise ValueError(f"execute_horizon must be in [1, {upstream.HORIZON}]")

        root = resolve_checkpoint_root(self.model_cfg, CHECKPOINTS_DIR, policy_dir=POLICY_DIR)
        checkpoint = root / "model.pt" if root.is_dir() else root
        if not checkpoint.is_file():
            raise FileNotFoundError(f"No model.pt under {root}; run download_checkpoint.sh")
        if not _none(self.model_cfg.get("checkpoint_sha256")):
            os.environ["RESWAM_CHECKPOINT_SHA256"] = str(self.model_cfg["checkpoint_sha256"])
        codec = self.model_cfg.get("codec_statistics_path")
        self.policy = upstream.RoboDojoPolicy(
            checkpoint,
            codec_path=None if _none(codec) else Path(codec),
            execute_horizon=self.execute_horizon,
        )
        base_seed = self.model_cfg.get("base_seed")
        # Per-chunk flow-matching noise seeds follow the official ResWAM RoboDojo protocol.
        self.seed_fields = {
            "base_seed": int(upstream.DEFAULT_BASE_SEED if _none(base_seed) else base_seed),
            "task": str(self.model_cfg.get("task_name", "")),
            "eval_seed": str(self.model_cfg.get("seed", 0)),
            "checkpoint_sha256": self.policy.checkpoint_sha256,
        }
        self.observations: dict[int, dict] = {}
        self.calls: dict[int, int] = {}
        self.resets = 0
        print(f"[ResWAM] loaded {checkpoint} (sha256 {self.policy.checkpoint_sha256[:12]}), "
              f"execute_horizon={self.execute_horizon}", flush=True)

    def update_obs(self, obs):
        self.observations[int(obs.get("env_idx", 0))] = obs

    def update_obs_batch(self, obs_list):
        for obs in obs_list:
            self.update_obs(obs)

    def _act(self, env_indices):
        observations = [self.observations[index] for index in env_indices]
        seeds = []
        for index in env_indices:
            call = self.calls.get(index, 0)
            self.calls[index] = call + 1
            seeds.append(self._derive_seed(**self.seed_fields, reset=self.resets, env_idx=index, call=call))
        chunks = self.policy.predict(observations, seeds)
        return [[self._unpack_action(row) for row in chunk[: self.execute_horizon]] for chunk in chunks]

    def get_action(self):
        if len(self.observations) != 1:
            raise RuntimeError("get_action needs exactly one observed environment; use get_action_batch")
        return self._act(list(self.observations))[0]

    def get_action_batch(self, env_idx_list=None):
        indices = list(env_idx_list) if env_idx_list is not None else sorted(self.observations)
        return self._act([int(index) for index in indices])

    def reset(self):
        self.observations.clear()
        self.calls.clear()
        self.resets += 1
