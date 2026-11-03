"""Platform-independent WorldScape rollout runtime."""

from typing import Any

from worldscape_policy.rollout.session import PolicyRuntime

__all__ = [
    "PolicyRuntime",
    "RolloutConfig",
    "RolloutResult",
    "RolloutRunner",
]


def __getattr__(name: str) -> Any:
    if name in {"RolloutConfig", "RolloutResult", "RolloutRunner"}:
        from worldscape_policy.rollout import runner

        return getattr(runner, name)
    raise AttributeError(name)
