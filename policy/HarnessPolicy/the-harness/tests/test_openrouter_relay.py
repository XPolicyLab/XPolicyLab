"""scripts/openrouter_relay.py against a fake OpenRouter on localhost. No network or API key."""

import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import unittest
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def _relay_module():
    spec = importlib.util.spec_from_file_location("openrouter_relay", ROOT / "scripts" / "openrouter_relay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeOpenRouter(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOpenRouter.seen.append({"path": self.path, "authorization": self.headers.get("Authorization"),
                                    "title": self.headers.get("X-Title"), "body": body})
        if body.get("model") == "bad/model":
            payload = json.dumps({"error": {"code": 400, "message": "bad model"}}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            delta = json.dumps({"id": "gen-1", "choices": [{"delta": {"content": "hello"}, "finish_reason": "stop"}]})
            for line in (b": OPENROUTER PROCESSING\n\n", f"data: {delta}\n\n".encode(), b"data: [DONE]\n\n"):
                self.wfile.write(line)
                self.wfile.flush()
            return
        payload = json.dumps({"id": "gen-1", "model": body["model"],
                              "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


def _serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class RelayTests(unittest.TestCase):
    TOKEN = "relay-token-0123456789"

    def setUp(self):
        FakeOpenRouter.seen = []
        self.upstream = _serve(FakeOpenRouter)
        handler = _relay_module().make_handler(self.TOKEN, "sk-or-upstream",
                                               upstream=f"http://127.0.0.1:{self.upstream.server_port}")
        self.relay = _serve(handler)
        self.base = f"http://127.0.0.1:{self.relay.server_port}"
        self._stderr, sys.stderr = sys.stderr, open("/dev/null", "w")

    def tearDown(self):
        sys.stderr.close()
        sys.stderr = self._stderr
        for server in (self.relay, self.upstream):
            server.shutdown()
            server.server_close()

    def post(self, path, payload, token=TOKEN):
        request = urllib.request.Request(self.base + path, data=json.dumps(payload).encode(), method="POST",
                                         headers={"Content-Type": "application/json", "X-Title": "harness",
                                                  "Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(request, timeout=10) as reply:
            return reply.status, reply.headers.get("Content-Type"), reply.read()

    def test_swaps_the_token_for_the_key_and_forwards_the_body(self):
        status, kind, body = self.post("/api/v1/responses", {"model": "openai/gpt-6-astra", "input": "hi"})
        self.assertEqual((status, kind), (200, "application/json"))
        self.assertEqual(json.loads(body)["model"], "openai/gpt-6-astra")
        [seen] = FakeOpenRouter.seen
        self.assertEqual(seen["path"], "/api/v1/responses")
        self.assertEqual(seen["authorization"], "Bearer sk-or-upstream")
        self.assertEqual(seen["title"], "harness")
        self.assertEqual(seen["body"], {"model": "openai/gpt-6-astra", "input": "hi"})

    def test_streams_server_sent_events(self):
        status, kind, body = self.post("/api/v1/chat/completions", {"model": "m", "stream": True})
        self.assertEqual((status, kind), (200, "text/event-stream"))
        self.assertTrue(body.endswith(b"data: [DONE]\n\n"))

    def test_passes_openrouter_errors_through(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post("/api/v1/chat/completions", {"model": "bad/model"})
        self.assertEqual(caught.exception.code, 400)
        self.assertEqual(json.loads(caught.exception.read())["error"]["message"], "bad model")

    def test_refuses_a_wrong_token_and_other_paths(self):
        for path, token, code in (("/api/v1/chat/completions", "wrong", 401), ("/other", self.TOKEN, 404)):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.post(path, {"model": "m"}, token=token)
            self.assertEqual(caught.exception.code, code)
            self.assertTrue(json.loads(caught.exception.read())["error"]["relay"])
        self.assertEqual(FakeOpenRouter.seen, [])

    def test_the_program_chat_model_calls_through_the_relay(self):
        import os
        from unittest import mock

        from harness_policy.runtime.chat import ChatModel

        for stream in (False, True):
            with self.subTest(stream=stream), mock.patch.dict(os.environ, {"OPENROUTER_RELAY_TOKEN": self.TOKEN}):
                FakeOpenRouter.seen = []
                model = ChatModel({"backend": "openrouter", "model": "x-ai/grok-4.7", "stream": stream,
                                   "base_url": self.base + "/api/v1", "api_key_env": "OPENROUTER_RELAY_TOKEN"})
                reply = model.complete("system", [{"role": "user", "content": "hi"}])
                self.assertEqual(reply.text, "hello")
                [seen] = FakeOpenRouter.seen
                self.assertEqual((seen["path"], seen["authorization"]),
                                 ("/api/v1/chat/completions", "Bearer sk-or-upstream"))
                self.assertEqual(seen["body"]["usage"], {"include": True})  # OpenRouter fields kept


RELAY = "http://127.0.0.1:8850/api/v1"
# Old harness names -> (agent configuration, model file) of configs/.
HARNESS = {f"r{rung}_openrouter.toml": (f"r{rung}", "grok-medium") for rung in range(4)}
HARNESS.update({"r4_openrouter.toml": ("default", "grok-medium"), "r4_local.toml": ("default", "qwen3-vl-32b")})


def load_harness(name):
    sys.path.insert(0, str(ROOT / "tests"))
    from helpers import task_config

    agent, model = HARNESS[name]
    return task_config(agent, model)


class RouteTests(unittest.TestCase):
    """OPENROUTER_RELAY_URL sends every OpenRouter model of a harness through the relay."""

    def load(self, harness, relay=RELAY):
        from unittest import mock

        env = {"OPENROUTER_RELAY_URL": relay} if relay else {}
        with mock.patch.dict("os.environ", env):
            if not relay:
                import os
                os.environ.pop("OPENROUTER_RELAY_URL", None)
            return load_harness(harness)

    def test_r0_to_r4_route_the_agent_and_helper_vlm(self):
        from harness_policy.runtime.isolation import endpoints, kernel_environment

        for rung in range(5):
            with self.subTest(rung=rung):
                config = self.load(f"r{rung}_openrouter.toml")
                models = [config.main] + ([config.runtime["vlm"]] if "vlm" in config.runtime else [])
                self.assertEqual(len(models), 1 if rung == 0 else 2)
                for model in models:
                    self.assertEqual((model["backend"], model["base_url"], model["api_key_env"]),
                                     ("openrouter", RELAY, "OPENROUTER_RELAY_TOKEN"))
                self.assertEqual(config.main["model"], "x-ai/grok-4.7")
                if "pointer" in config.runtime:  # self-hosted Molmo2 stays where it is
                    self.assertEqual(config.runtime["pointer"]["base_url"], "http://127.0.0.1:8830/v1")
                if rung:
                    # The program kernel's network guard admits the relay and gets the token.
                    options = {"vlm": config.runtime["vlm"]}
                    self.assertIn(("127.0.0.1", 8850), endpoints(options))
                    self.assertEqual(kernel_environment(options, {"OPENROUTER_RELAY_TOKEN": "t"})
                                     ["OPENROUTER_RELAY_TOKEN"], "t")

    def test_unset_or_self_hosted_is_unchanged(self):
        self.assertNotIn("base_url", self.load("r4_openrouter.toml", relay=None).main)
        local = self.load("r4_local.toml")
        self.assertEqual(local.main["base_url"], "http://127.0.0.1:8800/v1")
        self.assertNotIn("api_key_env", local.main)

class BaseUrlTests(unittest.TestCase):
    """OPENROUTER_BASE_URL replaces the OpenRouter URL and keeps OPENROUTER_API_KEY."""

    URL = "https://my-proxy.example/api/v1"

    def load(self, harness, **env):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, env):
            for name in ("OPENROUTER_BASE_URL", "OPENROUTER_RELAY_URL"):
                if name not in env:
                    os.environ.pop(name, None)
            return load_harness(harness)

    def test_r0_to_r4_call_the_new_url_with_the_openrouter_key(self):
        from harness_policy.core.http import chat_url as _chat_url

        for rung in range(5):
            with self.subTest(rung=rung):
                config = self.load(f"r{rung}_openrouter.toml", OPENROUTER_BASE_URL=self.URL)
                models = [config.main] + ([config.runtime["vlm"]] if "vlm" in config.runtime else [])
                for model in models:
                    self.assertEqual(model["base_url"], self.URL)
                    self.assertNotIn("api_key_env", model)  # still OPENROUTER_API_KEY
                    self.assertEqual(_chat_url(model["backend"], model["base_url"]), self.URL + "/chat/completions")
                if "pointer" in config.runtime:
                    self.assertEqual(config.runtime["pointer"]["base_url"], "http://127.0.0.1:8830/v1")

    def test_unset_keeps_openrouter_ai(self):
        from harness_policy.core.http import chat_url as _chat_url

        agent = self.load("r0_openrouter.toml").main
        self.assertEqual(_chat_url(agent["backend"], agent.get("base_url")),
                         "https://openrouter.ai/api/v1/chat/completions")

    def test_base_url_and_relay_together_are_refused(self):
        with self.assertRaises(ValueError):
            self.load("r0_openrouter.toml", OPENROUTER_BASE_URL=self.URL, OPENROUTER_RELAY_URL=RELAY)


if __name__ == "__main__":
    unittest.main()
