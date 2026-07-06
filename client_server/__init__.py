"""Env↔policy transport over WebSocket (default) and legacy TCP."""

from client_server.ws.model_client import WsModelClient
from client_server.ws.model_server import PolicyServer, PolicyServerConfig

ModelClient = WsModelClient
ModelServer = PolicyServer
ModelServerConfig = PolicyServerConfig

__all__ = [
    "ModelClient",
    "ModelServer",
    "ModelServerConfig",
    "PolicyServer",
    "PolicyServerConfig",
    "WsModelClient",
]
