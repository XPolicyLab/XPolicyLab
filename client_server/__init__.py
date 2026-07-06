"""Env↔policy transport over TCP and WebSocket."""

from __future__ import annotations

from typing import Any

from client_server.tcp import ModelClient, ModelServer

__all__ = ["ModelClient", "ModelServer", "WsModelClient"]


def __getattr__(name: str) -> Any:
    if name == "WsModelClient":
        from client_server.ws import WsModelClient

        return WsModelClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
