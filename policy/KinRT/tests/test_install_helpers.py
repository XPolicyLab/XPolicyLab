"""Portable tests for frozen-lock mirrors and tokenizer cache integrity."""

import sys
import unittest

if sys.version_info < (3, 11):
    raise unittest.SkipTest("KinRT installation helpers require Python 3.11 or newer")

import hashlib
import importlib.util
import io
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import tomllib
from unittest import mock


def load_helper(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dependencies = load_helper("sync_full35_dependencies")
tokenizer = load_helper("prepare_tokenizer")


class LockedMirrorTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.original = b'''version = 1
requires-python = ">=3.11"
[[package]]
name = "numpy"
version = "1.26.4"
source = { registry = "https://pypi.tuna.tsinghua.edu.cn/simple" }
wheels = [{ url = "https://pypi.tuna.tsinghua.edu.cn/packages/aa/bb/numpy.whl", hash = "sha256:1234", size = 99 }]
[[package]]
name = "lerobot"
version = "0.1.0"
source = { git = "https://github.com/huggingface/lerobot?rev=original#commit" }
'''
        self.lock = self.root / "uv.lock"
        self.lock.write_bytes(self.original)
        patch = mock.patch.object(dependencies, "LOCK_SHA256", hashlib.sha256(self.original).hexdigest())
        patch.start()
        self.addCleanup(patch.stop)

    def test_mirror_preserves_versions_hashes_and_git_source(self):
        original = tomllib.loads(self.original.decode())
        for mirror, registry, artifact in (
            ("pypi", "https://pypi.org/simple", "https://files.pythonhosted.org/packages/aa/bb/numpy.whl"),
            ("tencent", "https://mirrors.tencent.com/pypi/simple", "https://mirrors.tencent.com/pypi/packages/aa/bb/numpy.whl"),
        ):
            with self.subTest(mirror=mirror):
                converted = tomllib.loads(dependencies.mirrored_lock(self.original, mirror).decode())
                self.assertEqual(converted["package"][0]["source"]["registry"], registry)
                self.assertEqual(converted["package"][0]["wheels"][0]["url"], artifact)
                converted["package"][0]["source"] = original["package"][0]["source"]
                converted["package"][0]["wheels"][0]["url"] = original["package"][0]["wheels"][0]["url"]
                self.assertEqual(converted, original)
        self.assertEqual(dependencies.mirrored_lock(self.original, "original"), self.original)

    def test_unknown_lock_rejected_before_uv(self):
        changed = self.original.replace(b"1.26.4", b"2.0.0")
        self.lock.write_bytes(changed)
        with mock.patch.object(dependencies.subprocess, "Popen") as command:
            with self.assertRaises(ValueError):
                dependencies.sync(self.root, "uv", "pypi")
            command.assert_not_called()
        self.assertEqual(self.lock.read_bytes(), changed)

    def test_frozen_sync_and_restore_after_nonzero_exit(self):
        def command(arguments, **kwargs):
            self.assertIn(b"https://files.pythonhosted.org", self.lock.read_bytes())
            self.assertEqual(arguments, ["uv", "sync", "--frozen", "--no-default-groups", "--python", str(self.root / ".venv/bin/python")])
            self.assertEqual(kwargs["cwd"], self.root.resolve())
            process = mock.Mock()
            process.wait.return_value = 7
            process.poll.return_value = 7
            return process

        with mock.patch.object(dependencies.subprocess, "Popen", side_effect=command):
            self.assertEqual(dependencies.sync(self.root, "uv", "pypi"), 7)
        self.assertEqual(self.lock.read_bytes(), self.original)

    def test_restore_after_process_launch_exception(self):
        with mock.patch.object(dependencies.subprocess, "Popen", side_effect=FileNotFoundError("uv")):
            with self.assertRaises(FileNotFoundError):
                dependencies.sync(self.root, "uv", "tencent")
        self.assertEqual(self.lock.read_bytes(), self.original)

    def test_sigterm_during_successful_wait_returns_signal_status(self):
        previous_handler = signal.getsignal(signal.SIGTERM)
        process = mock.Mock()
        process.poll.return_value = 0

        def finish(**kwargs):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return 0

        def stopped(child):
            self.assertIs(child, process)
            self.assertIn(b"https://files.pythonhosted.org", self.lock.read_bytes())

        process.wait.side_effect = finish
        with mock.patch.object(dependencies.subprocess, "Popen", return_value=process), \
                mock.patch.object(dependencies, "stop_uv", side_effect=stopped) as stop:
            self.assertEqual(dependencies.sync(self.root, "uv", "pypi"), 128 + signal.SIGTERM)
            stop.assert_called_once_with(process)
        self.assertEqual(self.lock.read_bytes(), self.original)
        self.assertIs(signal.getsignal(signal.SIGTERM), previous_handler)

    @unittest.skipUnless(os.name == "posix", "Requires POSIX SIGTERM and process groups")
    def test_sigterm_stops_child_before_restoring_lock(self):
        self.assert_sigterm_cleanup()

    @unittest.skipUnless(os.name == "posix", "Requires POSIX SIGTERM and process groups")
    def test_sigterm_kills_term_ignoring_grandchild_before_restoring_lock(self):
        self.assert_sigterm_cleanup(with_grandchild=True)

    def assert_sigterm_cleanup(self, with_grandchild=False):
        grandchild_source = (
            "import os, signal, time\n"
            "from pathlib import Path\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "Path('grandchild.pid').write_text(str(os.getpid()))\n"
            "while True:\n"
            "    if 'files.pythonhosted.org' not in Path('uv.lock').read_text():\n"
            "        Path('grandchild-saw-original').touch()\n"
            "    time.sleep(0.01)\n"
        )
        fake_uv = self.root / "fake-uv"
        fake_uv.write_text(
            "#!/usr/bin/env python3\n"
            "import os, signal, subprocess, sys, time\n"
            "from pathlib import Path\n"
            "def stopped(signum, frame):\n"
            "    Path('child-saw-mirror').write_text(str('files.pythonhosted.org' in Path('uv.lock').read_text()))\n"
            "    raise SystemExit(0)\n"
            "signal.signal(signal.SIGTERM, stopped)\n"
            f"if {with_grandchild!r}:\n"
            f"    subprocess.Popen([sys.executable, '-c', {grandchild_source!r}])\n"
            "    while not Path('grandchild.pid').exists(): time.sleep(0.01)\n"
            "Path('child.pid').write_text(str(os.getpid()))\n"
            "while True: time.sleep(1)\n",
            encoding="utf-8",
        )
        fake_uv.chmod(0o755)
        runner = (
            "import importlib.util, sys; from pathlib import Path; "
            "spec=importlib.util.spec_from_file_location('dependencies',sys.argv[1]); "
            "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "module.LOCK_SHA256=sys.argv[4]; "
            "module.TERMINATE_GRACE_SECONDS=0.2; module.KILL_GRACE_SECONDS=3; "
            "sys.exit(module.sync(Path(sys.argv[2]),sys.argv[3],'pypi'))"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", runner, dependencies.__file__, str(self.root), str(fake_uv), hashlib.sha256(self.original).hexdigest()],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            deadline = time.monotonic() + 10
            pid_path = self.root / "child.pid"
            while (not pid_path.exists() or pid_path.stat().st_size == 0) and time.monotonic() < deadline and process.poll() is None:
                time.sleep(0.05)
            self.assertTrue(pid_path.exists(), "Fake uv did not start")
            child_pid = int(pid_path.read_text())
            process.send_signal(signal.SIGTERM)
            output, _ = process.communicate(timeout=15)
            self.assertEqual(process.returncode, 128 + signal.SIGTERM, output)
            self.assertEqual((self.root / "child-saw-mirror").read_text(), "True")
            self.assertEqual(self.lock.read_bytes(), self.original)
            self.assertFalse((self.root / "grandchild-saw-original").exists())
            with self.assertRaises(ProcessLookupError):
                os.killpg(child_pid, 0)
        finally:
            pid_path = self.root / "child.pid"
            if pid_path.exists() and pid_path.stat().st_size:
                try:
                    os.killpg(int(pid_path.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    @unittest.skipUnless(os.name == "posix", "Requires POSIX SIGTERM and process groups")
    def test_cleanup_kills_grandchild_after_uv_already_exited(self):
        child_source = (
            "import os, signal, time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "Path('grandchild.pid').write_text(str(os.getpid())); "
            "time.sleep(60)"
        )
        leader_source = (
            "import subprocess, sys, time\n"
            "from pathlib import Path\n"
            f"subprocess.Popen([sys.executable, '-c', {child_source!r}])\n"
            "while not Path('grandchild.pid').exists(): time.sleep(0.01)\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", leader_source], cwd=self.root,
            start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            self.assertEqual(process.wait(timeout=5), 0)
            os.killpg(process.pid, 0)
            with mock.patch.object(dependencies, "TERMINATE_GRACE_SECONDS", 0.2), \
                    mock.patch.object(dependencies, "KILL_GRACE_SECONDS", 3):
                dependencies.stop_uv(process)
            with self.assertRaises(ProcessLookupError):
                os.killpg(process.pid, 0)
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)


class TokenizerCacheTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "cache"
        self.data = b"test tokenizer bytes"
        self.destination = self.cache / tokenizer.TOKENIZER_PATH
        patch = mock.patch.object(tokenizer, "TOKENIZER_SHA256", hashlib.sha256(self.data).hexdigest())
        patch.start()
        self.addCleanup(patch.stop)

    def test_download_and_cached_verification(self):
        with mock.patch.object(tokenizer.urllib.request, "urlopen", return_value=io.BytesIO(self.data)) as download:
            self.assertEqual(tokenizer.prepare(self.cache), self.destination)
            download.assert_called_once_with(tokenizer.TOKENIZER_URL, timeout=120)
        with mock.patch.object(tokenizer.urllib.request, "urlopen") as download:
            self.assertEqual(tokenizer.prepare(self.cache, check=True), self.destination)
            download.assert_not_called()
        self.assertEqual(self.destination.read_bytes(), self.data)

    def test_bad_download_leaves_no_cached_or_partial_file(self):
        with mock.patch.object(tokenizer.urllib.request, "urlopen", return_value=io.BytesIO(b"bad download")):
            with self.assertRaises(ValueError):
                tokenizer.prepare(self.cache)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.destination.parent.iterdir()), [])

    def test_corrupt_existing_cache_is_reported_without_overwrite(self):
        self.destination.parent.mkdir(parents=True)
        self.destination.write_bytes(b"existing unrelated bytes")
        with mock.patch.object(tokenizer.urllib.request, "urlopen") as download:
            with self.assertRaises(ValueError):
                tokenizer.prepare(self.cache)
            download.assert_not_called()
        self.assertEqual(self.destination.read_bytes(), b"existing unrelated bytes")

    def test_local_source_and_missing_check(self):
        with self.assertRaises(FileNotFoundError):
            tokenizer.prepare(self.cache, check=True)
        self.assertFalse(self.cache.exists())
        source = self.root / "local.model"
        source.write_bytes(self.data)
        with mock.patch.object(tokenizer.urllib.request, "urlopen") as download:
            tokenizer.prepare(self.cache, source_file=source)
            download.assert_not_called()
        self.assertEqual(self.destination.read_bytes(), self.data)


if __name__ == "__main__":
    unittest.main()
