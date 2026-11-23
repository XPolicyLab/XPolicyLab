# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Modifications Copyright © 2026 Martingale. All Rights Reserved.
# This file has been modified by Martingale.


from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any

SERVER_NAME = "martingale"


def _result_to_mcp(result: Any) -> tuple[list[Any], bool]:
    from mcp import types

    blocks = getattr(result, "content_blocks", None)
    if blocks is None:
        return [types.TextContent(type="text", text=str(result))], False
    out: list[Any] = []
    for block in blocks:
        if block.get("type") == "text":
            out.append(types.TextContent(type="text", text=block.get("text", "")))
        elif block.get("type") == "image":
            src = block.get("source", {})
            out.append(
                types.ImageContent(
                    type="image",
                    data=src.get("data", ""),
                    mimeType=src.get("media_type", "image/png"),
                )
            )
    payload = getattr(result, "result", None)
    is_error = isinstance(payload, dict) and bool(payload.get("error"))
    return out, is_error


def _build_asgi_app(toolkit: Any, *, annotations: dict[str, Any] | None) -> Any:
    from mcp import types
    from mcp.server.lowlevel import Server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    app: Server = Server(SERVER_NAME, version="0.2.0")
    lock = asyncio.Lock()

    @app.list_tools()
    async def _list_tools() -> list[types.Tool]:
        tools = []
        for spec in toolkit.get_tools_spec():
            kwargs: dict[str, Any] = {
                "name": str(spec["name"]),
                "description": str(spec.get("description", "")),
                "inputSchema": spec.get("input_schema", {"type": "object"}),
            }
            if annotations:
                kwargs["annotations"] = types.ToolAnnotations(**annotations)
            tools.append(types.Tool(**kwargs))
        return tools

    @app.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        lookup = name.split("__")[-1] if name.startswith("mcp__") else name
        async with lock:
            result = await asyncio.get_running_loop().run_in_executor(
                None, toolkit.execute_tool, lookup, arguments or {}
            )
        content, is_error = _result_to_mcp(result)
        return types.CallToolResult(content=content, isError=is_error)

    sessions = StreamableHTTPSessionManager(app=app, stateless=True, json_response=True)

    async def asgi(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            while True:
                event = await receive()
                if event["type"] == "lifespan.startup":
                    async with sessions.run():
                        await send({"type": "lifespan.startup.complete"})
                        shutdown = await receive()
                        if shutdown["type"] != "lifespan.shutdown":
                            break
                        await send({"type": "lifespan.shutdown.complete"})
                    break
                if event["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    break
            return
        if scope["type"] == "http":
            await sessions.handle_request(scope, receive, send)

    return asgi


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


class ArmMcpServer:
    """Serve an :class:`ArmToolkit` over streamable HTTP on a daemon thread."""

    def __init__(
        self,
        toolkit: Any,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        annotations: dict[str, Any] | None = None,
    ) -> None:
        self._toolkit = toolkit
        self._host = host
        self._port = port or _free_port(host)
        self._annotations = annotations
        self._server: Any = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}/mcp/"

    def start(self, *, ready_timeout_s: float = 30.0) -> str:
        import httpx
        import uvicorn

        app = _build_asgi_app(self._toolkit, annotations=self._annotations)
        config = uvicorn.Config(app, host=self._host, port=self._port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True, name="arm-mcp")
        self._thread.start()
        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "hc", "version": "0"},
            },
        }
        with httpx.Client(
            transport=httpx.HTTPTransport(retries=10),
            timeout=httpx.Timeout(ready_timeout_s, connect=2),
            trust_env=False,
        ) as client:
            response = client.post(self.url, json=body, headers={"Accept": "application/json"})
            if not (response.is_success and "result" in response.json()):
                raise RuntimeError(f"MCP server not ready: {response.status_code}")
        return self.url

    def stop(self, *, timeout_s: float = 5.0) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)


__all__ = ["ArmMcpServer", "SERVER_NAME"]
