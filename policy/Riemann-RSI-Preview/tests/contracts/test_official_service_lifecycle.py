"""Official WS service shutdown; synthetic public actor, no model or GPU."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from tools import serve_official_policy as service

ROOT = Path(__file__).resolve().parents[2]


class OfficialServiceLifecycle(unittest.TestCase):
    def test_owned_sigterm_closes_the_real_server_without_an_error(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / 'deploy.json'
            config.write_text(json.dumps({'debug_hold': True}))
            with (root / 'server.log').open('w') as log:
                child = subprocess.Popen(
                    [sys.executable, '-B', str(ROOT / 'tools/serve_official_policy.py'),
                     '--mode', 'insert_key', '--port', '0', '--config', str(config),
                     '--output', str(root / 'policy')], cwd=ROOT,
                    env=dict(os.environ, EVAL_ENV_TYPE='debug', CUDA_VISIBLE_DEVICES=''),
                    stdout=log, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + 30
                    while child.poll() is None:
                        if 'official_rule_policy_ready' in (root / 'server.log').read_text():
                            break
                        if time.monotonic() >= deadline:
                            self.fail('Official WS server did not become ready')
                        time.sleep(.05)
                    self.assertIsNone(child.poll(), (root / 'server.log').read_text())
                    child.terminate()
                    self.assertEqual(child.wait(timeout=15), 0)
                finally:
                    if child.poll() is None:
                        child.kill()
                        child.wait(timeout=15)
            self.assertNotIn('Traceback', (root / 'server.log').read_text())

    def test_unrequested_cancellation_is_still_an_error(self):
        service.configure_imports()
        from client_server.ws.model_server import PolicyServer

        async def unexpected_cancellation(server):
            raise asyncio.CancelledError()

        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / 'deploy.json'
            config.write_text(json.dumps({'debug_hold': True}))
            with patch.dict(os.environ, {'EVAL_ENV_TYPE': 'debug'}), \
                    patch.object(PolicyServer, 'serve_forever', unexpected_cancellation):
                with self.assertRaises(asyncio.CancelledError):
                    service.main(['--mode', 'insert_key', '--port', '0',
                                  '--config', str(config), '--output', str(Path(folder) / 'policy')])
