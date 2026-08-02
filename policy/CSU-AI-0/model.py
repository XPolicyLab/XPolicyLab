"""Official XPolicyLab model entry point for the CSU-AI-0 routed policy.

The router itself is lightweight.  The selected expert runs in its own local
policy-server process so Xiaomi, Spatial Forcing, and Hunyuan keep their
mutually incompatible dependency environments.
"""

from __future__ import annotations

import os
from typing import Any

from XPolicyLab.model_template import ModelTemplate

from .expert_proxy import ExpertProcessProxy
from .route import append_audit, infer


class Model(ModelTemplate):
    """Route once per task and proxy the standard ModelTemplate interface."""

    def __init__(self, model_cfg: dict[str, Any]):
        super().__init__()
        self.model_cfg = dict(model_cfg)
        task_name = str(self.model_cfg.get("task_name") or "").strip()
        if not task_name:
            raise ValueError("CSU-AI-0 requires a non-empty task_name")

        self.route_result = infer(task_name)
        audit_path = self.model_cfg.get("route_audit_path") or os.environ.get(
            "CSU_ROUTE_AUDIT"
        )
        if audit_path:
            from pathlib import Path

            append_audit(Path(str(audit_path)).expanduser().resolve(), self.route_result)

        self.proxy = ExpertProcessProxy(self.route_result, self.model_cfg)
        self.model = self.proxy
        self._latest_obs: Any | None = None
        self._latest_obs_batch: list[Any] | None = None

    def update_obs(self, obs: Any) -> None:
        self._latest_obs = obs
        self._latest_obs_batch = None

    def update_obs_batch(self, obs_list: list[Any]) -> None:
        self._latest_obs_batch = list(obs_list)
        self._latest_obs = None

    def get_action(self):
        if self._latest_obs is None:
            raise AssertionError("update_obs() must be called before get_action()")
        return self.proxy.get_action(self._latest_obs)

    def get_action_batch(self, env_idx_list=None):
        del env_idx_list  # order is already fixed by update_obs_batch
        if self._latest_obs_batch is None:
            raise AssertionError(
                "update_obs_batch() must be called before get_action_batch()"
            )
        return self.proxy.get_action_batch(self._latest_obs_batch)

    def reset(self):
        self._latest_obs = None
        self._latest_obs_batch = None
        return self.proxy.reset()

    def prepare_case(self, case_meta=None):
        return self.proxy.prepare_case(case_meta or {})

    def on_trial_end(self, result=None):
        return self.proxy.trial_end(result or {})

    def close(self) -> None:
        self.proxy.close()


__all__ = ["Model", "infer"]
