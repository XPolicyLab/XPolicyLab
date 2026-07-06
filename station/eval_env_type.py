"""Resolve evaluation environment type (aligned with shell ``EVAL_ENV_TYPE``)."""

from __future__ import annotations

import os
from typing import Literal

EvalEnvType = Literal["debug", "sim", "real_world"]

DEFAULT_EVAL_ENV_TYPE: EvalEnvType = "sim"


def normalize_eval_env_type(raw: str | None) -> EvalEnvType:
    value = (raw or "").strip()
    if value in ("", "sim"):
        return "sim"
    if value == "debug":
        return "debug"
    if value in ("real", "real_world"):
        return "real_world"
    raise ValueError(
        f"Unknown eval env type: {raw!r} (expected: sim, debug, real)"
    )


def resolve_eval_env_type(raw: str | None = None) -> EvalEnvType:
    if raw is not None:
        return normalize_eval_env_type(raw)
    return normalize_eval_env_type(os.environ.get("EVAL_ENV_TYPE"))


def is_real_world(eval_env_type: str) -> bool:
    return eval_env_type == "real_world"


def is_debug(eval_env_type: str) -> bool:
    return eval_env_type == "debug"
