"""Client of tool servers that speak the Model Context Protocol (MCP) over streamable HTTP.

The mcp skill (skills/mcp) lists the tools of the MCP servers an agent names and calls
them from programs. Messages are JSON-RPC 2.0 POSTed to the server's endpoint; the answer
comes as JSON or as a text/event-stream whose data lines carry it. Only the HTTP
transport: programs run isolated and may not start processes, so a stdio-only server is
put behind an HTTP bridge as a service.
"""

import json


PROTOCOL_VERSION = "2025-06-18"


class McpError(RuntimeError):
    """An MCP server answered with an error, or a tool reported that it failed."""


class _SessionGone(Exception):
    """HTTP 404 to a request with a session id: the server no longer knows the session."""


class McpClient:
    """One MCP server. The session starts on first use (initialize, then the initialized
    notification), and the server's Mcp-Session-Id goes back on every later request."""

    def __init__(self, url, timeout=60.0, on_demand=None):
        self.url = url
        self.timeout = float(timeout)
        # A server placed on demand is started by the deploy's broker before a call.
        self.on_demand = on_demand
        self.session_id = None
        self._ready = False
        self._next_id = 0

    def _post(self, message):
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen

        from .broker import ensure

        ensure({"on_demand": self.on_demand})
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                   "MCP-Protocol-Version": PROTOCOL_VERSION}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        request = Request(self.url, data=json.dumps(message, default=_jsonable).encode(), headers=headers,
                          method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                self.session_id = response.headers.get("Mcp-Session-Id") or self.session_id
                content_type = response.headers.get("Content-Type", "")
                body = response.read().decode("utf-8", errors="replace")
        except HTTPError as error:
            with error:
                detail = error.read().decode("utf-8", errors="replace").strip()[:300]
            if error.code == 404 and self.session_id:
                raise _SessionGone() from None
            raise McpError(f"HTTP {error.code} from {self.url}: {detail or error.reason}") from None
        if "id" not in message:
            return None  # a notification: accepted, no answer
        return _response(body, content_type, message["id"])

    def request(self, method, params=None):
        """Send one JSON-RPC request and return its result. When the server no longer knows
        the session (it restarted, say after the broker stopped it idle), a new one starts
        with initialize, as the protocol requires, and the request is sent again."""
        try:
            return self._request(method, params)
        except _SessionGone:
            self.session_id, self._ready = None, False
            return self._request(method, params)

    def _request(self, method, params):
        if not self._ready and method != "initialize":
            self._request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                                         "clientInfo": {"name": "the-harness", "version": "1"}})
            self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
            self._ready = True
        self._next_id += 1
        message = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            message["params"] = params
        return self._post(message)

    def tools(self):
        """The server's tools: [{"name", "description", "inputSchema", ...}], every page."""
        tools, cursor = [], None
        while True:
            result = self.request("tools/list", {"cursor": cursor} if cursor else {})
            tools += list(result.get("tools", []))
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    def call(self, name, arguments=None):
        """Call tool `name`; its structured content when it gives one, else its text."""
        result = self.request("tools/call", {"name": name, "arguments": dict(arguments or {})})
        if result.get("isError"):
            raise McpError(f"tool {name}: {_text(result) or 'failed'}")
        if result.get("structuredContent") is not None:
            return result["structuredContent"]
        return _text(result)


def _jsonable(value):
    """numpy arrays and scalars (a program's positions, say) as JSON values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _text(result):
    return "\n".join(str(item.get("text", "")) for item in result.get("content", [])
                     if isinstance(item, dict) and item.get("type") == "text")


def _response(body, content_type, request_id):
    """The result of request `request_id` in a JSON body or an event stream."""
    if "text/event-stream" in content_type:
        messages, data = [], []
        for line in body.splitlines():
            if line.startswith("data:"):
                data.append(line[5:].strip())
            elif not line.strip() and data:
                messages.append("\n".join(data))
                data = []
        if data:
            messages.append("\n".join(data))
    else:
        messages = [body]
    for text in messages:
        try:
            reply = json.loads(text)
        except ValueError:
            continue
        for item in reply if isinstance(reply, list) else [reply]:
            if isinstance(item, dict) and item.get("id") == request_id:
                if "error" in item:
                    error = item["error"]
                    raise McpError(f"{error.get('code')}: {error.get('message')}")
                return item.get("result", {})
    raise McpError("the server's reply has no answer to the request")
