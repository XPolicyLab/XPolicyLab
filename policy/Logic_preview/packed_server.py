"""Serve independent RoboDojo shard sessions with one resident Logic_preview model."""
import asyncio
from collections import OrderedDict
from importlib import import_module

from websockets.exceptions import ConnectionClosed

from client_server.ws.model_server import PolicyServer, PolicyServerConfig
from client_server.ws.protocol.codec import decode_envelope
from client_server.ws.protocol.exceptions import WsError
from client_server.ws.protocol.messages import MessageType


class _Connection:
    """Pass the routing HELLO and the rest of its socket to the official handler."""

    def __init__(self, socket, hello):
        self.socket = socket
        self.hello = hello

    async def __aiter__(self):
        yield self.hello
        async for packet in self.socket:
            yield packet

    def __getattr__(self, name):
        return getattr(self.socket, name)


class PackedPolicyServer(PolicyServer):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.sessions = OrderedDict()
        self.connections = {}

    def _trim_idle_sessions(self):
        # Keep recent disconnects for native request replay. An evicted session
        # receives a new instance ID on reconnect, as after a server restart.
        idle = [key for key, server in self.sessions.items()
                if not self.connections[key] and not server._inflight]
        for key in idle[:-4]:
            del self.sessions[key], self.connections[key]

    async def _handle_connection(self, socket):
        try:
            hello = await socket.recv()
            frame = decode_envelope(hello) if isinstance(hello, bytes) else None
            if frame is None or frame.message_type != MessageType.HELLO:
                await socket.close(1002, "Expected a RoboDojo evaluation HELLO")
                return
        except ConnectionClosed:
            return
        except WsError:
            await socket.close(1002, "Invalid evaluation HELLO")
            return
        key = frame.evaluation_id
        if key not in self.sessions:
            session = PolicyServer(self.model, self.config)
            session._model_lock = self._model_lock
            self.sessions[key] = session
            self.connections[key] = 0
        self.sessions.move_to_end(key)
        self.connections[key] += 1
        try:
            await self.sessions[key]._handle_connection(_Connection(socket, hello))
        finally:
            self.connections[key] -= 1
            self._trim_idle_sessions()


if __name__ == "__main__":
    from setup_policy_server import parse_args_and_config

    cfg = parse_args_and_config()
    model = import_module(f"XPolicyLab.policy.{cfg['policy_name']}.model").Model(cfg)
    config = PolicyServerConfig(host=cfg["host"], port=int(cfg["port"]),
                                ws_ping_interval_s=cfg.get("ws_ping_interval_s", 20),
                                ws_ping_timeout_s=cfg.get("ws_ping_timeout_s", 20))
    asyncio.run(PackedPolicyServer(model, config).serve_forever())
