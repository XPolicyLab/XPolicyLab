"""A small MCP tool server for trying the mcp skill: streamable HTTP with JSON answers,
one tool, convert, between units of length, mass and angle (services/mcp-units.toml).

  python services/servers/serve_mcp_example.py --port 8860
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import uuid


UNITS = {"m": ("length", 1.0), "cm": ("length", 0.01), "mm": ("length", 0.001), "in": ("length", 0.0254),
         "kg": ("mass", 1.0), "g": ("mass", 0.001), "lb": ("mass", 0.45359237),
         "rad": ("angle", 1.0), "deg": ("angle", 0.017453292519943295)}
TOOLS = [{
    "name": "convert",
    "description": "Convert a value between units of length (m, cm, mm, in), mass (kg, g, lb) or angle (rad, deg).",
    "inputSchema": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"},
                                                     "to_unit": {"type": "string"}},
                    "required": ["value", "from_unit", "to_unit"]},
}]


def convert(value, from_unit, to_unit):
    if from_unit not in UNITS or to_unit not in UNITS or UNITS[from_unit][0] != UNITS[to_unit][0]:
        raise ValueError(f"cannot convert {from_unit} to {to_unit}")
    return float(value) * UNITS[from_unit][1] / UNITS[to_unit][1]


def answer(message):
    """The JSON-RPC response to one request (None for a notification)."""
    method, params, request_id = message.get("method"), message.get("params") or {}, message.get("id")
    if request_id is None:
        return None
    if method == "initialize":
        result = {"protocolVersion": params.get("protocolVersion", "2025-06-18"), "capabilities": {"tools": {}},
                  "serverInfo": {"name": "units", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        arguments = params.get("arguments") or {}
        try:
            value = convert(arguments["value"], arguments["from_unit"], arguments["to_unit"])
            result = {"content": [{"type": "text", "text": f"{value:g}"}],
                      "structuredContent": {"value": value, "unit": arguments["to_unit"]}}
        except (KeyError, ValueError) as error:
            result = {"content": [{"type": "text", "text": str(error)}], "isError": True}
    else:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"unknown method {method}"}}
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        message = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        reply = answer(message)
        if reply is None:
            self.send_response(202)
            self.end_headers()
            return
        body = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if message.get("method") == "initialize":
            self.send_header("Mcp-Session-Id", uuid.uuid4().hex)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def server(port, host="127.0.0.1"):
    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8860)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    server(args.port, args.host).serve_forever()
