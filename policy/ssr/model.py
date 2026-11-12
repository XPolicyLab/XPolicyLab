from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE / "_lib") not in sys.path:
    sys.path.insert(0, str(_HERE / "_lib"))

try:
    from XPolicyLab.model_template import ModelTemplate
except ImportError:
    _XPL = False

    class ModelTemplate:
        pass
else:
    _XPL = True

import rsi
from rsi.core.runtime import CALL_BUDGET_S, EpisodeAgent
from rsi.perception import models

ALLOWED_CFG = ("action_type", "env_cfg_type", "policy_name", "ckpt_name")


def check_action_dims(env_cfg_type: str | None) -> dict | None:
    if not _XPL or not env_cfg_type:
        return None
    from XPolicyLab.utils.process_data import get_robot_action_dim_info

    info = get_robot_action_dim_info(env_cfg_type)
    if len(info.get("arm_dim") or []) != 2 or list(info.get("ee_dim") or []) != [1, 1]:
        raise ValueError(f"rsi drives two arms with one gripper value each; {env_cfg_type} registers {info}")
    return info


def checkpoint_dir(cfg: dict, policy_dir: Path = _HERE) -> Path | None:
    if not _XPL or not cfg.get("ckpt_name"):
        return None
    from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root

    p = resolve_checkpoint_root({k: cfg[k] for k in ALLOWED_CFG if k in cfg}, policy_dir / "checkpoints",
                                policy_dir=policy_dir, include_seed=False, must_exist=False)
    return p if p.is_dir() else None


def _torch_settled() -> None:
    if "torch" in sys.modules:
        try:
            import torch
        except Exception:
            pass


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.cfg = {k: model_cfg.get(k) for k in ALLOWED_CFG if k in (model_cfg or {})}
        if (self.cfg.get("action_type") or "ee") != "ee":
            raise ValueError("rsi runs in absolute end-effector mode (action_type: ee)")
        check_action_dims(self.cfg.get("env_cfg_type"))
        ckpt = checkpoint_dir(self.cfg)
        if ckpt is not None:
            models.set_root(ckpt)
            models.verify_all()
        self.agents: dict[int, EpisodeAgent] = {}
        self._last_keys: list[int] = []
        self._t_obs: float | None = None
        print("[rsi] Model ready", flush=True)

    def update_obs(self, obs):
        self._t_obs = time.monotonic()
        self._agent(0).observe(obs)

    def get_action(self):
        a = self._agent(0)
        t0 = self._t_obs or time.monotonic()
        out = a.act(t0 + a.call_budget_s)
        _torch_settled()
        self._t_obs = None
        return out

    def update_obs_batch(self, obs_list):
        self._t_obs = time.monotonic()
        keys = []
        for pos, o in enumerate(obs_list or []):
            key = int(o.get("env_idx", pos)) if isinstance(o, dict) else pos
            self._agent(key).observe(o)
            keys.append(key)
        self._last_keys = keys

    def get_action_batch(self, env_idx_list=None):
        keys = [int(e) for e in env_idx_list] if env_idx_list is not None else list(self._last_keys)
        t0 = self._t_obs or time.monotonic()
        budget = min((self._agent(k).call_budget_s for k in keys), default=CALL_BUDGET_S)
        deadline = t0 + budget
        for k in keys:
            self._agent(k).prefetch()
        chunks = [self._agent(k).act(deadline) for k in keys]
        _torch_settled()
        self._t_obs = None
        n = min((len(c) for c in chunks), default=0)
        out = []
        for k, c in zip(keys, chunks):
            self._agent(k).push_back(c[n:])
            out.append(c[:n])
        return out

    def reset(self):
        self.agents = {}
        self._last_keys = []

    def _agent(self, key: int) -> EpisodeAgent:
        if key not in self.agents:
            self.agents[key] = EpisodeAgent()
        return self.agents[key]
