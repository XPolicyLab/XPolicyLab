"""Stdio MCP server exposing one KnowinAgent role's tools; every call is forwarded to the episode host.

Stdlib only, so any Python can run it: python -m knowin_agent.bridge.mcp --socket PATH --role task|local
"""

from __future__ import annotations

import argparse
import json
import socket
import sys

from knowin_agent import __version__

PROTOCOL = "2025-06-18"


class HostClient:
    def __init__(self, path: str):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(path)
        self.reader = self.sock.makefile("rb")

    def request(self, payload: dict) -> dict:
        self.sock.sendall(json.dumps(payload).encode() + b"\n")
        line = self.reader.readline()
        if not line:
            raise ConnectionError("episode host closed the connection")
        return json.loads(line)


def tool_result(reply: dict) -> dict:
    if "error" in reply and "result" not in reply:
        return {"content": [{"type": "text", "text": reply["error"]}], "isError": True}
    content = [{"type": "text", "text": reply["result"]}, {"type": "text", "text": reply.get("context", "")}]
    content += [{"type": "image", "data": data, "mimeType": "image/jpeg"} for data in reply.get("images", [])]
    if reply.get("episode_done"):
        content.append({"type": "text", "text": "The episode is over. Stop calling tools and end your turn."})
    return {"content": content, "isError": not reply.get("ok", False)}


def serve(client: HostClient, role: str, stdin=sys.stdin, stdout=sys.stdout) -> None:
    def send(message: dict) -> None:
        stdout.write(json.dumps(message) + "\n")
        stdout.flush()

    for line in stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
            continue
        method, ident = message.get("method"), message.get("id")
        if ident is None:
            continue  # notifications (initialized, cancelled) need no answer
        try:
            if method == "initialize":
                version = (message.get("params") or {}).get("protocolVersion") or PROTOCOL
                result = {"protocolVersion": version, "capabilities": {"tools": {"listChanged": False}},
                          "serverInfo": {"name": f"knowin_agent-{role}", "version": __version__},
                          "instructions": f"KnowinAgent {role} agent robot tools"}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": client.request({"op": "tools", "role": role})["tools"]}
            elif method == "tools/call":
                params = message.get("params") or {}
                reply = client.request({"op": "call", "role": role, "name": params.get("name"),
                                        "arguments": params.get("arguments") or {}})
                result = tool_result(reply)
            else:
                send({"jsonrpc": "2.0", "id": ident, "error": {"code": -32601, "message": f"unknown method {method}"}})
                continue
            send({"jsonrpc": "2.0", "id": ident, "result": result})
        except Exception as exc:  # noqa: BLE001 - always answer the client
            send({"jsonrpc": "2.0", "id": ident, "error": {"code": -32000, "message": f"{type(exc).__name__}: {exc}"}})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--role", choices=["task", "local"], required=True)
    args = parser.parse_args(argv)
    serve(HostClient(args.socket), args.role)
    return 0


if __name__ == "__main__":
    sys.exit(main())
