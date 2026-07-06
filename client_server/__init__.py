"""Env↔policy transport over WebSocket (default) and legacy TCP."""

from __future__ import annotations

from typing import Any

from client_server.ws.model_client import WsModelClient

ModelClient = WsModelClient

__all__ = [
    "ModelClient",
    "ModelServer",
    "ModelServerConfig",
    "PolicyServer",
    "PolicyServerConfig",
    "WsModelClient",
]


def __getattr__(name: str) -> Any:
    if name in ("ModelServer", "PolicyServer"):
        from client_server.ws.model_server import PolicyServer

        if name == "ModelServer":
            return PolicyServer
        return PolicyServer
    if name in ("ModelServerConfig", "PolicyServerConfig"):
        from client_server.ws.model_server import PolicyServerConfig

        if name == "ModelServerConfig":
            return PolicyServerConfig
        return PolicyServerConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
