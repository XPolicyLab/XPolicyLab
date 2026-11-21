"""Exercise model headers and robot routing through local HTTP listeners."""
from contextlib import contextmanager
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit
import uuid


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("joyai_egress", ROOT / "agent/api/egress_proxy.py")
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)


@contextmanager
def listener(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def post(address, path, body, headers):
    connection = http.client.HTTPConnection(*address, timeout=5)
    try:
        connection.request("POST", path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


class EgressProxy(unittest.TestCase):
    def setUp(self):
        self.received = []
        received = self.received

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                received.append((self.path, self.headers,
                                 self.rfile.read(int(self.headers.get("Content-Length", "0")))))
                body = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.upstream = self.enterContext(listener(Upstream))
        self.logs = []
        self.enterContext(patch.object(proxy, "log", side_effect=self.logs.append))

    def proxy_address(self, key_file=None):
        handler = type("Handler", (proxy.Proxy,), {
            "upstream": urlsplit(f"http://{self.upstream[0]}:{self.upstream[1]}"),
            "allowed_prefixes": ("/v1/chat/completions",) if key_file else ("/v1/cmd",),
            "key_file": key_file,
        })
        return self.enterContext(listener(handler))

    def test_model_requests_get_private_auth_and_unique_trace_ids(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        key_file = Path(temporary) / "key"
        key_file.write_text("synthetic-private-key\n")
        address = self.proxy_address(str(key_file))
        body = json.dumps({"model": "joyai_vl_testing", "stream": False,
                           "messages": [{"role": "user", "content": "你好"}]}).encode()
        headers = {"Content-Type": "application/json", "Authorization": "Bearer ignored",
                   "api-key": "ignored", "X-API-Key": "ignored",
                   "trace-id": "old-id", "Trace-ID": "duplicate-old-id"}
        for _ in range(2):
            self.assertEqual(post(address, "/v1/chat/completions", body, headers),
                             (200, b'{"ok":true}'))
        trace_ids = []
        for path, forwarded, received_body in self.received:
            self.assertEqual(path, "/v1/chat/completions")
            self.assertEqual(received_body, body)
            self.assertEqual(forwarded.get_all("Authorization"), ["Bearer synthetic-private-key"])
            self.assertNotIn("api-key", forwarded)
            self.assertNotIn("x-api-key", forwarded)
            self.assertEqual(forwarded["Content-Type"], "application/json")
            self.assertEqual(len(forwarded.get_all("Trace-Id")), 1)
            trace_id = forwarded["Trace-Id"]
            self.assertEqual(uuid.UUID(trace_id).version, 4)
            trace_ids.append(trace_id)
        self.assertEqual(len(set(trace_ids)), 2)
        # Request-start logs are written before the response; completion logs follow
        # the final chunk and may reach the reader just after the response does.
        for trace_id in trace_ids:
            self.assertTrue(any(line.startswith("TRACE ") and trace_id in line for line in self.logs))
        self.assertNotIn("synthetic-private-key", "\n".join(self.logs))

    def test_robot_requests_keep_body_and_route_boundary(self):
        address = self.proxy_address()
        body = b'{"cmd":"wait"}'
        self.assertEqual(post(address, "/v1/cmd", body, {"Content-Type": "application/json"}),
                         (200, b'{"ok":true}'))
        self.assertEqual(post(address, "/admin", body, {})[0], 403)
        self.assertEqual(len(self.received), 1)
        path, forwarded, received_body = self.received[0]
        self.assertEqual(path, "/v1/cmd")
        self.assertEqual(received_body, body)
        self.assertNotIn("Authorization", forwarded)
        self.assertNotIn("Trace-Id", forwarded)


if __name__ == "__main__":
    unittest.main()
