"""Exercise real pi0.5 sockets and owned child processes with CPU-only policies.

Run on a worker that permits localhost listeners:
    python -m unittest discover -s tests -p test_pi05_parallel.py -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.request import ProxyHandler, build_opener

import msgpack
import numpy as np
import websockets
from websockets.sync.client import connect

POLICY_DIR = Path(__file__).resolve().parents[1]


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


with patch.object(sys, "path", [str(POLICY_DIR), *sys.path]):
    runtime = _module("_pi05_parallel_runtime", POLICY_DIR / "pi05_runtime.py")
protocol = _module(
    "_pi05_parallel_protocol", POLICY_DIR / "vendor/pi05/server/protocol.py"
)


class Pi05ParallelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pi05-parallel-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.policy = self.root / "policy"
        python = self.policy / "runtime/pi05-env/bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        entry = POLICY_DIR / "pi05_server.py"
        self.assertTrue(entry.is_file(), "The production pi0.5 child entry is required")
        dependency_roots = sorted(
            {
                str(Path(module.__file__).resolve().parent.parent)
                for module in (np, msgpack, websockets)
            }
        )
        # Only the expensive model factory/adapter are replaced. The production
        # entry, original WS handler, dynamic batcher and HTTP route remain real.
        (self.policy / "pi05_server.py").write_text(
            "import runpy, sys\n"
            "from pathlib import Path\n"
            "from types import SimpleNamespace\n"
            f"sys.path[:0] = {[str(POLICY_DIR / 'vendor/pi05'), *dependency_roots]!r}\n"
            "import numpy as np\n"
            "from server import openpi_batch_server as server\n"
            "def create_policy(*, config_name, checkpoint_dir):\n"
            "    return SimpleNamespace(metadata={'instance_id': Path(checkpoint_dir).name})\n"
            "class CpuBatchPolicy:\n"
            "    def __init__(self, policy):\n"
            "        self.metadata = policy.metadata\n"
            "    def infer_batch(self, observations, *, pad_to=None):\n"
            "        marker = 11 if self.metadata['instance_id'] == 'instance-a' else 22\n"
            "        return [{'instance_id': self.metadata['instance_id'],\n"
            "                 'actions': np.asarray(obs['state'], dtype=np.float32) + marker}\n"
            "                for obs in observations]\n"
            "server.create_policy = create_policy\n"
            "server.OpenPIBatchPolicy = CpuBatchPolicy\n"
            f"runpy.run_path({str(entry)!r}, run_name='__main__')\n"
        )
        for name in (
            "check_pi05_runtime",
            "validate_pi05_checkpoint",
            "validate_pi05_tokenizer",
        ):
            selected = patch.object(runtime, name, return_value={})
            selected.start()
            self.addCleanup(selected.stop)
        selected = patch.object(runtime, "_service_env", side_effect=self._environment)
        selected.start()
        self.addCleanup(selected.stop)

    @staticmethod
    def _environment(_policy, _data, output, _gpu):
        output.mkdir(parents=True, exist_ok=True)
        return {
            "PATH": os.defpath,
            "HOME": str(output),
            "LANG": "C.UTF-8",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

    def _server(self, identity, *, port=0):
        server = runtime.Pi05Server(
            self.root / identity,
            self.root / "data",
            self.root / (identity + "-output"),
            port=port,
            policy_dir=self.policy,
            startup_timeout_s=20,
        )
        self.addCleanup(server.close)
        return server

    def _client(self, server):
        return connect(
            f"ws://127.0.0.1:{server.port}",
            proxy=None,
            open_timeout=5,
            close_timeout=2,
        )

    def _request(self, client, identity, marker, *, offset=0):
        state = np.arange(6, dtype=np.float32).reshape(2, 3) + offset
        client.send(protocol.pack_message({"state": state}))
        action = protocol.unpack_message(client.recv(timeout=5))
        self.assertEqual(action["instance_id"], identity)
        np.testing.assert_array_equal(action["actions"], state + marker)
        self.assertEqual(action["server_timing"]["batch_size"], 1)

    def test_parallel_ephemeral_ports_and_independent_shutdown(self):
        first, second = self._server("instance-a"), self._server("instance-b")
        self.assertEqual((first.port, second.port), (0, 0))
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(server.start) for server in (first, second)]
            for future in futures:
                future.result(timeout=30)
        self.assertGreater(first.port, 0)
        self.assertGreater(second.port, 0)
        self.assertNotEqual(first.port, second.port)
        self.assertNotEqual(first.process.pid, second.process.pid)
        with self._client(second) as second_client:
            metadata = protocol.unpack_message(second_client.recv(timeout=5))
            self.assertEqual(metadata["instance_id"], "instance-b")
            with self._client(first) as first_client:
                metadata = protocol.unpack_message(first_client.recv(timeout=5))
                self.assertEqual(metadata["instance_id"], "instance-a")
                self._request(first_client, "instance-a", 11)
                self._request(second_client, "instance-b", 22)
            first.close()
            self.assertIsNotNone(first.process.poll())
            self.assertIsNone(second.process.poll())
            self.assertTrue(second._ready())
            self._request(second_client, "instance-b", 22, offset=100)
        second.close()
        self.assertIsNotNone(first.process.poll())
        self.assertIsNotNone(second.process.poll())

    def test_occupied_fixed_port_never_accepts_or_stops_the_external_service(self):
        requests = []

        class ExternalHealth(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK\n")

            def log_message(self, _format, *args):
                pass

        external = ThreadingHTTPServer(("127.0.0.1", 0), ExternalHealth)
        thread = threading.Thread(target=external.serve_forever, daemon=True)
        thread.start()

        def close_external():
            external.shutdown()
            external.server_close()
            thread.join(timeout=3)

        self.addCleanup(close_external)
        occupied_port = external.server_address[1]
        server = self._server("occupied", port=occupied_port)
        with self.assertRaises(RuntimeError):
            server.start()
        self.assertIsNotNone(server.process.poll())
        self.assertEqual(requests, [], "Readiness must come from the owned child")
        server.close()
        self.assertTrue(thread.is_alive())
        opener = build_opener(ProxyHandler({}))
        with opener.open(
            f"http://127.0.0.1:{occupied_port}/healthz", timeout=3
        ) as reply:
            self.assertEqual(reply.status, 200)
            self.assertEqual(reply.read(), b"OK\n")
        self.assertEqual(requests, ["/healthz"])


if __name__ == "__main__":
    unittest.main()
