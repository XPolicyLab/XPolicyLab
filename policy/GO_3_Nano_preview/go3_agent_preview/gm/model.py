from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info

from . import runtime

CHUNK_SIZE = 1


class GmModel(ModelTemplate):
    def __init__(self, model_cfg, *, respect_switch=True):
        self.respect_switch = bool(respect_switch)
        self.model_cfg = model_cfg
        self.action_type = str(model_cfg.get("action_type", "joint"))
        self.env_cfg_type = model_cfg.get("env_cfg_type")
        self.task_name = model_cfg.get("task_name")
        if self.action_type != "joint":
            raise NotImplementedError(
                f"gm programs command joint positions, got action_type={self.action_type!r}"
            )

        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        arm_dims = self.robot_action_dim_info["arm_dim"]
        ee_dims = self.robot_action_dim_info["ee_dim"]
        assert len(arm_dims) == len(ee_dims), "Arm and EE action dimensions must match"
        self.num_arms = len(arm_dims)
        if self.num_arms != 2:
            raise NotImplementedError(f"gm programs assume two arms, got {self.num_arms}")

        cfg = model_cfg.get("code") or {}
        if not isinstance(cfg, Mapping):
            raise TypeError("code must be a mapping in deploy.yml")
        out = cfg.get("out_dir")
        self.out_root = Path(out) if out else None

        self._sessions: dict[int, runtime.Session] = {}
        self._episode: dict[int, int] = {}
        self._last: dict[int, dict[str, np.ndarray]] = {}

        print(
            f"[gm] ready: task={self.task_name} program={self.program_name()} "
            f"env_cfg_type={self.env_cfg_type} arms={self.num_arms}",
            flush=True,
        )

    def program_name(self) -> str:
        name = self.task_name
        if not name:
            raise ValueError("gm needs task_name to pick a program")
        return str(name)

    def _load(self, name: str):
        prefix = f"{__package__}.programs"
        for mod in [m for m in list(sys.modules)
                    if m == prefix or m.startswith(prefix + ".")]:
            del sys.modules[mod]
        try:
            mod = importlib.import_module(f"{prefix}.{name}")
        except ModuleNotFoundError as exc:
            base = name[: -len("_random")] if name.endswith("_random") else None
            if base is None or exc.name != f"{prefix}.{name}":
                raise
            mod = importlib.import_module(f"{prefix}.{base}")
        if not hasattr(mod, "run"):
            raise ImportError(f"{prefix}.{name} has no run()")
        return mod

    def _session(self, key: int) -> runtime.Session:
        sess = self._sessions.get(key)
        if sess is None:
            episode = self._episode.get(key, -1) + 1
            self._episode[key] = episode
            out = None
            if self.out_root is not None:
                out = self.out_root / f"env-{key:03d}" / f"episode-{episode:06d}"
                out.mkdir(parents=True, exist_ok=True)
            mod = self._load(self.program_name())
            sess = self._sessions[key] = runtime.Session(self.task_name, mod.run, out, respect_switch=self.respect_switch)
        return sess

    @staticmethod
    def _key(obs, fallback: int) -> int:
        return int(obs.get("env_idx", fallback))

    def update_obs(self, obs):
        self._session(self._key(obs, 0)).observe(obs)

    def update_obs_batch(self, obs_list):
        for i, obs in enumerate(obs_list):
            self._session(self._key(obs, i)).observe(obs)

    def _chunk(self, key: int):
        sess = self._sessions.get(key)
        if sess is None:
            raise RuntimeError(f"get_action called before update_obs for env {key}")
        action = sess.next_action()
        if sess.finished:
            held = self._last.get(key)
            if held is not None:
                action = held
        self._last[key] = action
        return [action for _ in range(CHUNK_SIZE)]

    def get_action(self):
        return self._chunk(0)

    def get_action_batch(self, env_idx_list=None):
        if not env_idx_list:
            return self._chunk(0)
        return [self._chunk(int(i)) for i in env_idx_list]

    def reset(self):
        self._close_sessions()

    def status(self, env_idx: int) -> dict:
        """Policy-visible signals only; no simulator success, reward or truth."""
        sess = self._sessions.get(int(env_idx))
        if sess is None:
            return {"step": 0, "done": False, "failure": None, "events": []}
        ctx = sess.ctx
        tracking = {arm: float(np.max(np.abs(ctx.q(arm) - ctx.cmd_q[arm])))
                    for arm in runtime.session.ARMS}
        return {"step": ctx.step, "done": bool(ctx.done), "failure": ctx.failure,
                "stage": ctx.stage, "events": list(ctx.online_events),
                "skill_switch_request": ctx.switch_request.record() if ctx.switch_request else None,
                "switch_paused": sess.switch_paused,
                "tracking_error_rad": tracking,
                "trace_ref": str(ctx.out / "tape.jsonl") if ctx.out else None}

    def drop_env(self, env_idx: int) -> None:
        """Drop only an idle session; running calls are isolated by the worker."""
        key = int(env_idx)
        sess = self._sessions.pop(key, None)
        if sess is not None:
            sess.ctx.close()
        self._last.pop(key, None)

    def close(self):
        self._close_sessions()

    def _close_sessions(self):
        for sess in self._sessions.values():
            try:
                sess.ctx.close()
            except Exception:
                pass
        self._sessions.clear()
        self._last.clear()
