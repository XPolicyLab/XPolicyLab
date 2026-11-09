"""The isolated kernel (App. A): what a program can see and reach."""

from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness_policy.runtime.isolation import endpoints, hidden_roots
from harness_policy.runtime.kernel import Kernel


def _namespaces_work() -> bool:
    if not shutil.which("unshare"):
        return False
    probe = subprocess.run(["unshare", "--user", "--map-root-user", "--mount", "true"], capture_output=True)
    return probe.returncode == 0


class EndpointTests(unittest.TestCase):
    def test_every_service_in_the_options_is_allowed(self):
        options = {"endpoint": "http://127.0.0.1:8795", "segmenter": {"url": "http://127.0.0.1:8810"},
                   "policies": {"g05": {"url": "ws://127.0.0.1:19005"}},
                   "vlm": {"backend": "openrouter", "model": "m"},
                   "verifier": {"backend": "openai_chat", "base_url": "http://10.0.0.2:8000/v1", "model": "m"}}
        self.assertEqual(endpoints(options), {("127.0.0.1", 8795), ("127.0.0.1", 8810), ("127.0.0.1", 19005),
                                              ("openrouter.ai", 443), ("10.0.0.2", 8000)})

    def test_hidden_roots_cover_home_and_owned_ancestors(self):
        roots = hidden_roots([Path.home() / "x" / "y"])
        self.assertIn(Path.home().resolve(), roots)
        self.assertFalse(any(Path.home().resolve() in root.parents for root in roots))


@unittest.skipUnless(sys.platform.startswith("linux"), "the guard is tested on Linux")
class GuardWithoutSandboxTests(unittest.TestCase):
    """isolation = "auto" where user namespaces are not available: no filesystem
    sandbox, but programs still reach only the services in their options."""

    def test_the_guard_stays(self):
        from unittest import mock

        from harness_policy.runtime import kernel as kernels

        server = socket.create_server(("127.0.0.1", 0))
        allowed = server.getsockname()[1]
        with mock.patch.object(kernels, "sandbox_available", return_value=False):
            kernel = Kernel("harness_policy.runtime.kernel:PlainSession",
                            {"isolation": "auto", "endpoint": f"http://127.0.0.1:{allowed}"})
        try:
            result = kernel.run("import socket\nsocket.getaddrinfo('example.com', 443)", timeout=60)
            self.assertEqual(result["status"], "error")
            self.assertIn("PermissionError", result["error"])
            result = kernel.run("import subprocess\nsubprocess.run(['true'])", timeout=60)
            self.assertEqual(result["status"], "error")
        finally:
            kernel.close()
            server.close()


@unittest.skipUnless(sys.platform.startswith("linux") and _namespaces_work(), "needs unprivileged user namespaces")
class IsolatedKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A service the kernel may reach, and one it may not.
        cls.servers = [socket.create_server(("127.0.0.1", 0)) for _ in range(2)]
        for server in cls.servers:
            threading.Thread(target=cls._accept, args=(server,), daemon=True).start()
        cls.allowed, cls.other = (server.getsockname()[1] for server in cls.servers)
        cls.secret_dir = Path(tempfile.mkdtemp(dir=Path.home()))
        (cls.secret_dir / "secret.txt").write_text("task checker")
        cls.kernel = Kernel("harness_policy.runtime.kernel:PlainSession",
                            {"isolation": True, "endpoint": f"http://127.0.0.1:{cls.allowed}"})

    @staticmethod
    def _accept(server):
        while True:
            try:
                connection, _ = server.accept()
                connection.close()
            except OSError:
                return

    @classmethod
    def tearDownClass(cls):
        cls.kernel.close()
        for server in cls.servers:
            server.close()
        shutil.rmtree(cls.secret_dir)

    def run_cell(self, code):
        result = self.kernel.run(code, timeout=60)
        return result["status"], result["stdout"], result.get("error") or ""

    def test_user_files_are_hidden(self):
        status, stdout, _ = self.run_cell(
            f"import os\nprint(os.path.exists({str(self.secret_dir / 'secret.txt')!r}), sorted(os.listdir(os.path.expanduser('~'))))")
        self.assertEqual(status, "ok")
        self.assertTrue(stdout.startswith("False"), stdout)

    def test_source_is_read_only_and_the_workspace_writable(self):
        source = Path(__file__).resolve().parents[1] / "src" / "harness_policy"
        status, stdout, _ = self.run_cell(
            "import os\n"
            f"print(os.path.isdir({str(source)!r}))\n"
            "open('notes.txt', 'w').write('x')\n"
            "print(os.getcwd())\n"
            "try:\n"
            f"    open({str(source / 'x.py')!r}, 'w')\n"
            "except OSError as error:\n"
            "    print('refused', error.errno)\n")
        self.assertEqual(status, "ok")
        self.assertEqual(stdout.split(), ["True", "/tmp/workspace", "refused", "30"])  # EROFS

    def test_only_allowed_services_are_reachable(self):
        status, stdout, error = self.run_cell(
            f"import socket\nsocket.create_connection(('127.0.0.1', {self.allowed}), timeout=5).close()\nprint('ok')")
        self.assertEqual((status, stdout.strip()), ("ok", "ok"), error)
        status, _, error = self.run_cell(f"import socket\nsocket.create_connection(('127.0.0.1', {self.other}), timeout=5)")
        self.assertEqual(status, "error")
        self.assertIn("PermissionError", error)
        status, _, error = self.run_cell("import socket\nsocket.getaddrinfo('example.com', 443)")
        self.assertIn("PermissionError", error)

    def test_policy_client_is_importable_but_its_checkout_hidden(self):
        from harness_policy.runtime.isolation import package_dirs

        dirs = package_dirs()
        if not dirs:
            self.skipTest("the policy client (XPolicyLab client_server) is not installed")
        status, stdout, error = self.run_cell(
            "import os, client_server.ws.model_client as c\n"
            f"print(sorted(os.listdir({str(dirs[0].parent)!r})))")
        self.assertEqual(status, "ok", error)
        self.assertEqual(stdout.strip(), "['client_server']")  # the rest of the checkout is not visible

    def test_processes_cannot_be_started_or_seen(self):
        status, _, error = self.run_cell("import subprocess\nsubprocess.run(['ls'])")
        self.assertIn("PermissionError", error)
        status, _, error = self.run_cell("import os\nos.system('ls')")
        self.assertIn("PermissionError", error)
        status, stdout, _ = self.run_cell("import os\nprint(sorted(int(p) for p in os.listdir('/proc') if p.isdigit()))")
        self.assertEqual(status, "ok")
        self.assertLessEqual(len(eval(stdout)), 2)  # the kernel (and the shell it replaced)


if __name__ == "__main__":
    unittest.main()
