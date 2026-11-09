"""MCP tool servers: the client (services/mcp.py) against the example server, the mcp
skill, and the wiring of kind mcp through the services a kernel gets."""
from pathlib import Path
import sys
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "services" / "servers")]

import serve_mcp_example  # noqa: E402
from harness_policy.runtime.skills import available_services  # noqa: E402
from harness_policy.services.mcp import McpClient, McpError, _response  # noqa: E402
from harness_policy.services.registry import load_services  # noqa: E402
from skills.mcp import McpTools  # noqa: E402


class SessionHandler(serve_mcp_example.Handler):
    """The example server, keeping its sessions as a stateful server does: a request with
    an id it did not give out (it restarted) gets 404."""
    sessions = set()

    def do_POST(self):
        session = self.headers.get("Mcp-Session-Id")
        if session and session not in self.sessions:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        super().do_POST()

    def send_header(self, keyword, value):
        if keyword == "Mcp-Session-Id":
            self.sessions.add(value)
        super().send_header(keyword, value)


class ExampleServer:
    def __init__(self, handler=None):
        self.handler = handler

    def __enter__(self):
        if self.handler:
            self.server = serve_mcp_example.ThreadingHTTPServer(("127.0.0.1", 0), self.handler)
        else:
            self.server = serve_mcp_example.server(0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.server.server_address[1]}/mcp"

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


class ClientTests(unittest.TestCase):
    def test_lists_and_calls_tools(self):
        with ExampleServer() as url:
            client = McpClient(url)
            self.assertEqual([tool["name"] for tool in client.tools()], ["convert"])
            self.assertIsNotNone(client.session_id)  # from initialize, sent back since
            self.assertEqual(client.call("convert", {"value": 25, "from_unit": "cm", "to_unit": "m"}),
                             {"value": 0.25, "unit": "m"})
            with self.assertRaises(McpError):
                client.call("convert", {"value": 1, "from_unit": "kg", "to_unit": "m"})

    def test_new_session_after_the_server_forgets_it(self):
        with ExampleServer(SessionHandler) as url:
            client = McpClient(url)
            client.tools()
            first = client.session_id
            SessionHandler.sessions.clear()  # the server restarted
            self.assertEqual(client.call("convert", {"value": 1, "from_unit": "m", "to_unit": "cm"})["value"], 100.0)
            self.assertNotEqual(client.session_id, first)

    def test_numpy_arguments(self):
        import numpy as np

        with ExampleServer() as url:
            result = McpClient(url).call("convert", {"value": np.float32(2.0), "from_unit": "m", "to_unit": "cm"})
            self.assertEqual(result["value"], 200.0)

    def test_event_stream_answers(self):
        body = 'event: message\ndata: {"jsonrpc": "2.0", "id": 3, "result": {"tools": []}}\n\n'
        self.assertEqual(_response(body, "text/event-stream", 3), {"tools": []})
        with self.assertRaises(McpError):
            _response('{"jsonrpc": "2.0", "id": 4, "error": {"code": -1, "message": "no"}}', "application/json", 4)


class Session(McpTools):
    def __init__(self, servers):
        self.mcp_servers = servers


class SkillTests(unittest.TestCase):
    def test_tools_by_name_and_shared_names_by_server(self):
        with ExampleServer() as first, ExampleServer() as second:
            one = Session({"units": {"url": first}})
            self.assertEqual([tool["name"] for tool in one.mcp_tools()], ["convert"])
            self.assertAlmostEqual(one.mcp_call("convert", value=90, from_unit="deg", to_unit="rad")["value"],
                                   1.5707963, places=6)
            two = Session({"a": {"url": first}, "b": {"url": second}})
            self.assertEqual([tool["name"] for tool in two.mcp_tools()], ["a.convert", "b.convert"])
            self.assertEqual(two.mcp_call("b.convert", value=1, from_unit="m", to_unit="mm")["value"], 1000.0)
            with self.assertRaises(ValueError):
                two.mcp_call("convert", value=1, from_unit="m", to_unit="mm")

    def test_a_server_down_leaves_the_others(self):
        with ExampleServer() as url:
            down = "http://127.0.0.1:9/mcp"  # nothing listens there
            both = Session({"units": {"url": url, "timeout_seconds": 5}, "down": {"url": down, "timeout_seconds": 5}})
            self.assertEqual([tool["name"] for tool in both.mcp_tools()], ["convert"])
            self.assertEqual(both.mcp_call("convert", value=1, from_unit="m", to_unit="mm")["value"], 1000.0)
            with self.assertRaises(McpError):
                Session({"down": {"url": down, "timeout_seconds": 5}}).mcp_tools()


class WiringTests(unittest.TestCase):
    def test_kind_mcp(self):
        self.assertEqual(load_services()["mcp-units"].kind, "mcp")
        self.assertIn("mcp", available_services({"mcp": {"mcp-units": {"url": "http://127.0.0.1:8860/mcp"}}}))
        self.assertNotIn("mcp", available_services({"mcp": {}}))


if __name__ == "__main__":
    unittest.main()
