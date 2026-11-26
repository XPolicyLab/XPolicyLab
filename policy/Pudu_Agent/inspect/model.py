"""RoboDojo_Agent_L4_Inspect serves no VLA.

The harness expects every adapter to expose a ``Model``, so this one exists to
answer that contract and to fail loudly if anything ever tries to ask it for an
action: at L4 inspect the only thing that produces actions is the LLM in
``policy.py``.
"""

from __future__ import annotations

from typing import Any

from XPolicyLab.model_template import ModelTemplate


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]) -> None:
        self.model_cfg = model_cfg

    def reset(self) -> None:
        return None

    def update_obs(self, obs: dict[str, Any]) -> None:
        del obs

    def update_obs_batch(self, obs_list: list[dict[str, Any]]) -> None:
        del obs_list

    def get_action(self) -> Any:
        raise RuntimeError(
            "RoboDojo_Agent_L4_Inspect has no VLA to call. Actions come from the "
            "LLM in policy.py; a condition that calls a served policy here is "
            "L1 or L2, not L4 inspect."
        )

    def get_action_batch(self, env_idx_list: list[int] | None = None) -> Any:
        del env_idx_list
        raise RuntimeError(
            "RoboDojo_Agent_L4_Inspect has no batched VLA action path. Each "
            "RoboDojo environment needs its own L4 inspect conversation."
        )
