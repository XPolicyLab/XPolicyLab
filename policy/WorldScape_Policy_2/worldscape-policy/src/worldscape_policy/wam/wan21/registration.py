from __future__ import annotations

from typing import Any

from worldscape_policy.wam.registry import WAMRegistry
from worldscape_policy.wam.wan21.plugin import Wan21WAMConfig, Wan21WAMPlugin


def _build_wan21(
    *,
    config: Wan21WAMConfig,
    **dependencies: Any,
) -> Wan21WAMPlugin:
    del config
    if dependencies:
        raise TypeError(
            f"Unexpected Wan2.1 dependencies: {', '.join(sorted(dependencies))}"
        )
    return Wan21WAMPlugin()


def register_wan21(registry: WAMRegistry) -> None:
    registry.register(
        name="wan21",
        version="2.1",
        capabilities={"protocol_adapter"},
        config_type=Wan21WAMConfig,
        factory=_build_wan21,
    )


__all__ = ["register_wan21"]
