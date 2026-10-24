"""On-demand services: the broker starts a service when a client asks, keeps one service
of a swap group loaded, and waits while a peer is in use."""

from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from harness_policy.services import broker as brokers  # noqa: E402
from harness_policy.services import supervisor  # noqa: E402
from harness_policy.services.deploy import client_config, load_deploy  # noqa: E402


def free_port():
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return handle.getsockname()[1]


class BrokerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        services = root / "services"
        services.mkdir()
        for name in ("a", "b"):
            (services / f"{name}.toml").write_text(f"""
kind = "policy"
protocol = "http"
command = ["{{python}}", "-m", "http.server", "{{port}}", "--bind", "127.0.0.1"]
port = {free_port()}
vram_gb = 0
load_seconds = 10
client = {{ url = "{{url}}" }}
[[weights]]
dir = "none"
local = "nothing"
""")
        (root / "deploy.toml").write_text(f"""
state = "{root / 'state'}"
broker_port = {free_port()}
[services.a]
mode = "on_demand"
swap = "g"
[services.b]
mode = "on_demand"
swap = "g"
""")
        self.deploy = load_deploy(root / "deploy.toml", services_root=services)
        self.broker = brokers.Broker(self.deploy, log=lambda message: None, swap_grace=0.5)
        self.server = self.broker.serve()
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.broker.stopped = True
        self.server.shutdown()
        self.server.server_close()
        supervisor.down(self.deploy, ["a", "b"], log=lambda message: None)
        self.directory.cleanup()

    def up(self, name):
        return supervisor.listening(int(client_config(self.deploy, name)["url"].rsplit(":", 1)[1]))

    def test_client_config_names_the_broker(self):
        config = client_config(self.deploy, "a")
        self.assertEqual(config["on_demand"], {"url": f"http://127.0.0.1:{self.deploy.broker_port}", "service": "a"})

    def test_ensure_starts_and_swaps(self):
        self.assertFalse(self.up("a"))
        brokers.ensure(client_config(self.deploy, "a"))
        self.assertTrue(self.up("a"))
        time.sleep(0.6)  # a is idle beyond the swap grace
        brokers._LAST.clear()
        brokers.ensure(client_config(self.deploy, "b"))
        self.assertTrue(self.up("b"))
        self.assertFalse(self.up("a"))

    def test_a_peer_in_use_is_not_swapped_out(self):
        self.broker.ensure("a")
        done = []

        def want_b():
            self.broker.ensure("b")
            done.append(time.monotonic())

        thread = threading.Thread(target=want_b)
        start = time.monotonic()
        thread.start()
        for _ in range(4):  # a stays in use for about 0.8 s
            self.broker.ensure("a")
            time.sleep(0.2)
        thread.join(timeout=30)
        self.assertTrue(done)
        self.assertGreater(done[0] - start, 0.8)
        self.assertTrue(self.up("b"))

    def test_without_on_demand_ensure_does_nothing(self):
        brokers.ensure({"url": "http://127.0.0.1:1"})
        brokers.ensure(None)


if __name__ == "__main__":
    unittest.main()
