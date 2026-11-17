"""Receding-horizon slice for closed-loop eval.

The policy still predicts the full action chunk (usually 50). Setting
``FOCUS_VLWA_REPLAN_STEPS=N`` keeps only the first N steps so Isaac replans sooner.
``0`` executes the full chunk; the released Focus-VLWA policy uses 25 steps.
"""

from __future__ import annotations

import os
from typing import Any


def replan_steps_from_env(env: dict[str, str] | None = None) -> int:
    raw = (os.environ if env is None else env).get("FOCUS_VLWA_REPLAN_STEPS", "25")
    try:
        n = int(str(raw).strip() or 0)
    except ValueError as error:
        raise ValueError("FOCUS_VLWA_REPLAN_STEPS must be an integer") from error
    if n < 0:
        raise ValueError("FOCUS_VLWA_REPLAN_STEPS cannot be negative")
    return n


def slice_action_chunk(actions: Any, n: int) -> Any:
    if n <= 0 or actions is None:
        return actions
    if hasattr(actions, "shape"):
        if len(actions) <= n:
            return actions
        return actions[:n]
    if isinstance(actions, (list, tuple)):
        if len(actions) <= n:
            return actions
        return actions[:n]
    return actions
