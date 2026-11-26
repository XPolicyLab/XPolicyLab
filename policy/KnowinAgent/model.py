"""KnowinAgent serves no learned policy: the agent runs in the RoboDojo client (``deploy.py``).

The XPolicyLab policy server loads a ``Model`` for every adapter. Simulation answers ``reset`` and refuses
served actions because the client-side Agent owns them. Only ``EVAL_ENV_TYPE=debug`` serves joint holds,
using the official server's decoded observations to check the websocket roundtrip without model inference.
"""

from __future__ import annotations

import os

from typing import Any

from XPolicyLab.model_template import ModelTemplate


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]) -> None:
        self.model_cfg = model_cfg
        self.debug = os.environ.get("EVAL_ENV_TYPE") == "debug"
        if not self.debug and os.environ.get("KNOWIN_AGENT_DRY_RUN", "0") == "1":
            raise RuntimeError("KNOWIN_AGENT_DRY_RUN=1 requires EVAL_ENV_TYPE=debug")
        self._hold = None

    def reset(self) -> None:
        self._hold = None

    def update_obs(self, obs: dict[str, Any]) -> None:
        if not self.debug:
            return
        # XPolicyLab's server decodes all incoming images before calling this method.
        # Checking their decoded shape makes the raw/encoded debug roundtrip meaningful.
        for camera in obs.get("vision", {}).values():
            color = camera.get("color")
            if getattr(color, "ndim", None) != 3 or color.shape[-1] != 3:
                raise ValueError("debug expects official decoded HxWx3 camera colors")
        state = obs["state"]
        self._hold = {name: value for name, value in state.items()
                      if name.endswith("arm_joint_state") or name.endswith("ee_joint_state")}
        if not self._hold:
            raise ValueError("debug observation has no joint state to hold")

    def update_obs_batch(self, obs_list: list[dict[str, Any]]) -> None:
        del obs_list

    def get_action(self) -> Any:
        if self.debug:
            if self._hold is None:
                raise RuntimeError("debug get_action requires update_obs first")
            return self._hold
        raise RuntimeError("KnowinAgent has no served policy: actions come from the agent in deploy.py")

    def get_action_batch(self, env_idx_list: list[int] | None = None) -> Any:
        raise RuntimeError("KnowinAgent has no served policy: actions come from the agent in deploy.py")
