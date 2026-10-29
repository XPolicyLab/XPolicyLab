import contextlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from harness_policy.services import policy_server as session
from harness_policy.services.policy_server import policy_lock_path, policy_server_lock


class FakePolicyServer:
    """An XPolicyLab server's one current observation: get_action answers from the last update_obs."""

    def __init__(self):
        self.observation, self.mixups = None, 0

    def update_obs(self, obs):
        self.observation = obs

    def get_action(self, obs):
        time.sleep(0.01)  # a client that sends its pair unlocked is overtaken here
        if self.observation != obs:
            self.mixups += 1


def episodes(server, url, locked):
    def episode(name):
        for step in range(20):
            obs = f"{name}-{step}"
            if locked:
                with policy_server_lock(url):
                    server.update_obs(obs)
                    time.sleep(0.005)
                    server.get_action(obs)
            else:
                server.update_obs(obs)
                time.sleep(0.005)
                server.get_action(obs)

    threads = [threading.Thread(target=episode, args=(name,)) for name in ("a", "b", "c")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


class PolicyLockTests(unittest.TestCase):
    def setUp(self):
        self.locks = tempfile.TemporaryDirectory()
        self.directory = mock.patch.object(session, "POLICY_LOCK_DIR", self.locks.name)
        self.directory.start()

    def tearDown(self):
        self.directory.stop()
        self.locks.cleanup()

    def test_parallel_episodes_get_their_own_actions(self):
        unlocked, locked = FakePolicyServer(), FakePolicyServer()
        episodes(unlocked, "ws://127.0.0.1:19005", locked=False)
        episodes(locked, "ws://127.0.0.1:19005", locked=True)
        self.assertGreater(unlocked.mixups, 0)  # the race the lock exists for
        self.assertEqual(locked.mixups, 0)

    def test_spellings_of_one_server_share_a_lock(self):
        with policy_server_lock("ws://127.0.0.1:19005"):
            acquired = threading.Event()

            def same_server():
                with policy_server_lock("ws://localhost:19005/"):
                    acquired.set()

            thread = threading.Thread(target=same_server)
            thread.start()
            self.assertFalse(acquired.wait(0.3))
        self.assertTrue(acquired.wait(5))
        thread.join()

    def test_an_existing_lock_file_is_opened_without_create(self):
        # In a sticky directory the kernel refuses O_CREAT on another account's file
        # (fs.protected_regular), so an existing lock file must be opened without it.
        url = "ws://127.0.0.1:19005"
        with policy_server_lock(url):
            pass
        real_open = os.open

        def protected_open(path, flags, *args):
            if flags & os.O_CREAT and os.path.exists(path):
                raise PermissionError(13, "Permission denied", path)
            return real_open(path, flags, *args)

        with mock.patch.object(session.os, "open", protected_open):
            with policy_server_lock(url):
                pass

    def test_a_dangling_symlink_fails_instead_of_spinning(self):
        url = "ws://127.0.0.1:19005"
        os.symlink(os.path.join(self.locks.name, "missing"), session.policy_lock_path(url))
        with self.assertRaises(OSError):
            with policy_server_lock(url):
                pass

    def test_lock_is_per_server(self):
        with policy_server_lock("ws://127.0.0.1:19005"):
            acquired = threading.Event()

            def other_server():
                with policy_server_lock("ws://127.0.0.1:19006"):
                    acquired.set()

            thread = threading.Thread(target=other_server)
            thread.start()
            self.assertTrue(acquired.wait(5))
            thread.join()


def _namespaces_work() -> bool:
    if not shutil.which("unshare"):
        return False
    probe = subprocess.run(["unshare", "--user", "--map-root-user", "--mount", "true"], capture_output=True)
    return probe.returncode == 0


@unittest.skipUnless(sys.platform.startswith("linux") and _namespaces_work(), "needs unprivileged user namespaces")
class IsolatedKernelLockTests(unittest.TestCase):
    """run_policy takes the lock inside the episode's isolated kernel, which has its own
    empty home and /tmp: two kernels must still wait for each other."""

    def test_two_isolated_kernels_share_the_lock(self):
        import fcntl

        from harness_policy.runtime.kernel import Kernel

        # A server no real run uses, so the test never waits on (or blocks) a live suite.
        url = "ws://127.0.0.1:1/isolated-lock-test"
        kernels = []
        try:
            for _ in range(2):
                kernels.append(Kernel("harness_policy.runtime.kernel:PlainSession", {"isolation": True}))
            for kernel in kernels:
                result = kernel.run("from harness_policy.services.policy_server import policy_server_lock\nimport time",
                                    timeout=120)
                self.assertEqual(result["status"], "ok", result)
            holder = threading.Thread(target=kernels[0].run, args=(
                f"with policy_server_lock({url!r}):\n    print('held', flush=True)\n    time.sleep(4)",),
                kwargs={"timeout": 60})
            holder.start()
            # The host sees the first kernel's lock (the same file), so the second must wait.
            deadline, held = time.time() + 20, False
            while not held and time.time() < deadline:
                try:
                    with open(policy_lock_path(url), "rb") as handle:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fcntl.flock(handle, fcntl.LOCK_UN)
                    time.sleep(0.1)
                except FileNotFoundError:
                    time.sleep(0.1)
                except BlockingIOError:
                    held = True
            self.assertTrue(held, "the host never saw the first kernel's lock")
            waited = kernels[1].run(f"start = time.time()\nwith policy_server_lock({url!r}):\n    pass\n"
                                    "print(round(time.time() - start, 1))", timeout=60)
            holder.join()
            self.assertEqual(waited["status"], "ok", waited)
            self.assertGreaterEqual(float(waited["stdout"].strip()), 1.0, waited["stdout"])
        finally:
            for kernel in kernels:
                kernel.close()
            with contextlib.suppress(OSError):
                os.unlink(policy_lock_path(url))


if __name__ == "__main__":
    unittest.main()
