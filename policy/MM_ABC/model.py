"""MM-ABC adapter using checkpoint metadata for the mobile action layout."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root

POLICY_DIR = Path(__file__).resolve().parent
# The vendored MM-ABC source tree (like policy/G05/G05); MMABC_ROOT overrides it.
DEFAULT_MMABC_ROOT = Path(os.environ.get("MMABC_ROOT", POLICY_DIR / "MM_ABC"))

def _clean(value: Any) -> Any:
    return None if value in (None, "", "null", "None") else value

def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if _clean(value) is None:
        return default
    return str(value).strip().lower() in ("1", "true", "t", "yes", "y", "on")

def resolve_milestone(root: Path, ckpt_step: Any = None) -> Path:
    """A run directory -> one of its milestones; a milestone/resume dir -> itself."""
    root = Path(root)
    if (root / ".metadata").exists():
        return root
    milestones = root / "milestones"
    if not milestones.is_dir():
        raise FileNotFoundError(f"{root} is neither a checkpoint nor an MM-ABC run directory")
    if _clean(ckpt_step) is not None:
        path = milestones / f"step_{int(ckpt_step):08d}"
        if not (path / ".metadata").exists():
            raise FileNotFoundError(f"milestone {path} not found")
        return path
    done = sorted(p for p in milestones.glob("step_*") if (p / ".metadata").exists())
    if not done:
        raise FileNotFoundError(f"no finished milestone under {milestones}")
    return done[-1]

class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = dict(model_cfg)
        self.action_type = self.model_cfg.get("action_type") or "joint"
        if self.action_type != "joint":
            raise ValueError(f"MM_ABC supports action_type=joint only, got {self.action_type!r}")
        self.mock = _as_bool(self.model_cfg.get("mock"))

        mmabc_root = Path(_clean(self.model_cfg.get("mmabc_root")) or DEFAULT_MMABC_ROOT).resolve()
        if str(mmabc_root) not in sys.path:
            sys.path.insert(0, str(mmabc_root))
        from mmabc.embodiments import mobile
        from mmabc.eval.adapters.mobile import MobileAdapter

        self.mobile = mobile
        self.adapter = MobileAdapter(
            allow_missing_state=_as_bool(self.model_cfg.get("allow_missing_state_keys")),
            action_key_style=str(self.model_cfg.get("action_key_style") or "singular"),
        )

        gpu_id = _clean(self.model_cfg.get("gpu_id"))
        if gpu_id is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        execute = _clean(self.model_cfg.get("execute_steps"))
        self.policy = None
        if self.mock:
            self.chunk_size = 32
            self.execute_steps = int(execute or 16)
            if not 1 <= self.execute_steps <= self.chunk_size:
                raise ValueError("execute_steps must be between 1 and 32 in mock mode")
            self.checkpoint = None
        else:
            root = resolve_checkpoint_root(self.model_cfg, POLICY_DIR / "checkpoints", policy_dir=POLICY_DIR)
            self.checkpoint = resolve_milestone(root, self.model_cfg.get("ckpt_step"))
            from mmabc.eval.policy import MMABCInferencePolicy

            steps = _clean(self.model_cfg.get("num_inference_steps"))
            seed = _clean(self.model_cfg.get("seed"))
            self.policy = MMABCInferencePolicy.from_run(
                self.checkpoint,
                device=str(self.model_cfg.get("device") or "cuda"),
                repo_root=str(mmabc_root),
                num_inference_steps=int(steps) if steps else None,
                execute_steps=int(execute) if execute else None,
                seed=int(seed) if seed is not None else None,
            )
            self.chunk_size = self.policy.chunk_size
            self.execute_steps = self.policy.execute_steps
            # Pretrain-mode checkpoints use the 80-d canonical layout.
            self.adapter.variant = (self.policy.contract or {}).get("variant", "m75")
        self.default_instruction = str(self.model_cfg.get("task_name") or "")

        self.obs_by_env: dict[int, dict] = {}
        self._latest_env_idx_list = [0]
        print(
            f"[MM_ABC] ready (mock={self.mock}) checkpoint={self.checkpoint} layout={self.adapter.variant} "
            f"chunk={self.chunk_size} execute_steps={self.execute_steps} "
            f"cameras={list(mobile.CAMERAS.values())}",
            flush=True,
        )

    def _instruction(self, obs: dict) -> str:
        text = obs.get("instruction") or obs.get("instructions") or ""
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        return str(text) or self.default_instruction

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        from mmabc.eval.policy import Observation

        indices = [int(obs.get("env_idx", i)) for i, obs in enumerate(obs_list)]
        if len(indices) != len(set(indices)):
            raise ValueError("batch observations must have distinct env_idx values")
        self._latest_env_idx_list = []
        for env_idx, obs in zip(indices, obs_list):
            self._latest_env_idx_list.append(env_idx)
            self.obs_by_env[env_idx] = Observation(
                state=self.adapter.to_canonical_state(obs),
                images=self.adapter.to_canonical_images(obs),
                instruction=self._instruction(obs),
            )

    def get_action(self):
        if not self._latest_env_idx_list:
            raise RuntimeError("update_obs must provide an observation before get_action")
        return self.get_action_batch([self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        env_idx_list = [int(i) for i in (self._latest_env_idx_list if env_idx_list is None else env_idx_list)]
        if not env_idx_list:
            return []
        missing = [i for i in env_idx_list if i not in self.obs_by_env]
        if missing:
            raise AssertionError(f"update_obs must be called before get_action (envs {missing})")
        observations = [self.obs_by_env[i] for i in env_idx_list]
        if self.mock:
            # Hold the current state: a well-formed, deterministic chunk.
            chunks = [np.repeat(o.state[None], self.execute_steps, axis=0) for o in observations]
        else:
            chunks = [r["action_canonical"] for r in self.policy.act_batch(observations)]
        return [self.adapter.from_canonical_action(c) for c in chunks]

    def reset(self):
        self.obs_by_env.clear()
        self._latest_env_idx_list = [0]
