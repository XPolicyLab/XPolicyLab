"""Storage escapes and lifecycle regressions; no model, GPU, or simulator is loaded."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

POLICY = Path(__file__).resolve().parents[1]


def alive(pid):
    result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return result.returncode == 0 and not result.stdout.strip().startswith("Z")


class EntrypointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rap-wam-entrypoint-")
        self.base = Path(self.temporary.name).resolve()
        self.work = self.base / "work"
        self.env = dict(os.environ, RAP_WAM_WORK_DIR=str(self.work), RAP_WAM_BOOTSTRAP_PYTHON=sys.executable)
        self.env.pop("RAP_WAM_ENV", None)

    def tearDown(self):
        self.temporary.cleanup()

    def source(self, tail="true"):
        return subprocess.run(["bash", "-ec", 'source "$1"; ' + tail, "test", str(POLICY / "runtime_env.sh")],
                              env=self.env, text=True, capture_output=True)

    def test_all_exported_output_paths_are_contained_and_home_unchanged(self):
        self.env.update(TMPDIR="/tmp/wrong", HF_HOME="/tmp/wrong", TORCH_HOME="/tmp/wrong")
        result = self.source('env')
        self.assertEqual(result.returncode, 0, result.stderr)
        env = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        for key in ("TMPDIR", "TMP", "TEMP", "PIP_CACHE_DIR", "HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE",
                    "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "CUDA_CACHE_PATH", "TRITON_CACHE_DIR",
                    "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "RAP_WAM_ENV"):
            Path(env[key]).resolve().relative_to(self.work)
        self.assertEqual(env.get("HOME"), self.env.get("HOME"))
        self.assertFalse(list(self.work.rglob(".rap-wam-storage-check-*")))

    def test_cache_symlink_escape_fails_before_any_outside_write(self):
        outside = self.base / "outside"
        outside.mkdir()
        self.work.mkdir()
        (self.work / "cache").symlink_to(outside, target_is_directory=True)
        result = self.source()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("escapes", result.stderr)
        self.assertEqual(list(outside.iterdir()), [])
        self.assertFalse((self.work / "tmp").exists())

    def test_environment_symlink_escape_is_rejected(self):
        self.work.mkdir()
        (self.work / "env").symlink_to(self.base / "other_env", target_is_directory=True)
        result = self.source()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("environment escapes", result.stderr)

    def test_relative_work_root_is_rejected(self):
        self.env["RAP_WAM_WORK_DIR"] = "relative"
        result = self.source()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("absolute", result.stderr)

    def installer(self):
        policy = self.work / "checkout" / "XPolicyLab" / "policy" / "RAP_WAM"
        policy.mkdir(parents=True)
        for name in ("install.sh", "runtime_env.sh"):
            shutil.copy2(POLICY / name, policy / name)
        self.env["PYTHON"] = sys.executable
        return policy / "install.sh"

    def test_installer_refuses_nonempty_environment_by_default(self):
        script = self.installer()
        env_dir = self.work / "env"
        env_dir.mkdir()
        marker = env_dir / "preserve.txt"
        marker.write_text("existing environment")
        result = subprocess.run(["bash", str(script)], env=self.env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("non-empty environment", result.stderr)
        self.assertEqual(marker.read_text(), "existing environment")

    def capturing_installer(self):
        """Record shell invocations without creating a real venv or running pip."""
        if not (3, 10) <= sys.version_info[:2] < (3, 12):
            self.skipTest("Installer command capture requires supported Python 3.10 or 3.11")
        script = self.installer()
        # Test captures must not persist unrelated credentials from the caller.
        for name in ("PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "HTTPS_PROXY", "PIP_CERT", "RAP_WAM_TORCH_INDEX_URL"):
            self.env.pop(name, None)
        # Use the existing executable: generated scripts may lack execute permission
        # on shared storage. Only these module bodies substitute for venv and pip.
        modules = self.work / "capture_modules"
        pip_module = modules / "pip"
        pip_module.mkdir(parents=True)
        (modules / "venv.py").write_text('''
import pathlib, sys
if len(sys.argv) != 2:
    raise SystemExit("Unexpected fake venv invocation: " + repr(sys.argv))
target = pathlib.Path(sys.argv[1]) / "bin" / "python"
target.parent.mkdir(parents=True)
target.symlink_to(sys.executable)
''')
        (pip_module / "__init__.py").write_text("")
        (pip_module / "__main__.py").write_text('''
import json, os, pathlib, sys
if sys.argv[1:2] != ["install"]:
    raise SystemExit("Unexpected fake pip invocation: " + repr(sys.argv))
with open(pathlib.Path(os.environ["RAP_WAM_WORK_DIR"]) / "pip_calls.jsonl", "a") as handle:
    names = ("PIP_TARGET", "PIP_PREFIX", "PIP_ROOT", "PIP_USER", "PIP_PYTHON",
             "PIP_LOG", "PIP_SRC", "PIP_BUILD_TRACKER", "PIP_REPORT", "PIP_CONFIG_FILE",
             "PIP_CACHE_DIR", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "HTTPS_PROXY",
             "PIP_CERT", "RAP_WAM_TORCH_INDEX_URL")
    selected = {name: os.environ[name] for name in names if name in os.environ}
    handle.write(json.dumps({"argv": ["-m", "pip", *sys.argv[1:]], "env": selected}) + "\\n")
''')
        self.env["PYTHON"] = sys.executable
        self.env["PYTHONPATH"] = str(modules)
        return script

    def test_installer_clears_pip_destinations_but_keeps_network_and_local_report(self):
        script = self.capturing_installer()
        outside = self.base / "outside"
        blocked = ("PIP_TARGET", "PIP_PREFIX", "PIP_ROOT", "PIP_USER", "PIP_PYTHON",
                   "PIP_LOG", "PIP_SRC", "PIP_BUILD_TRACKER")
        self.env.update({name: "1" if name == "PIP_USER" else str(outside / name) for name in blocked})
        preserved = {
            "PIP_INDEX_URL": "https://example.invalid/simple",
            "PIP_EXTRA_INDEX_URL": "https://extra.example.invalid/simple",
            "HTTPS_PROXY": "http://proxy.example.invalid:8080",
            "PIP_CERT": str(self.base / "cert.pem"),
            "RAP_WAM_TORCH_INDEX_URL": "https://torch.example.invalid/cu128",
            "PIP_REPORT": str(self.work / "logs" / "install.json"),
        }
        self.env.update(preserved)
        result = subprocess.run(["bash", str(script)], env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in (self.work / "pip_calls.jsonl").read_text().splitlines()]
        self.assertEqual(len(calls), 3)
        for call in calls:
            for name in blocked:
                self.assertNotIn(name, call["env"])
            for name, value in preserved.items():
                self.assertEqual(call["env"][name], value)
            self.assertEqual(call["env"]["PIP_CONFIG_FILE"], "/dev/null")
            Path(call["env"]["PIP_CACHE_DIR"]).relative_to(self.work)
        self.assertIn(preserved["RAP_WAM_TORCH_INDEX_URL"], calls[1]["argv"])
        self.assertFalse(outside.exists())

    def test_installer_rejects_external_or_relative_pip_report_before_venv_creation(self):
        script = self.installer()
        outside = self.base / "outside"
        outside.mkdir()
        (self.work / "report_link").symlink_to(outside, target_is_directory=True)
        reports = (str(outside / "install.json"), str(self.work / "report_link" / "install.json"),
                   "relative-report.json")
        for report in reports:
            with self.subTest(report=report):
                self.env["PIP_REPORT"] = report
                result = subprocess.run(["bash", str(script)], env=self.env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("pip report", result.stderr)
                self.assertFalse((self.work / "env").exists())
                self.assertEqual(list(outside.iterdir()), [])

    def test_installer_keeps_stdout_pip_report(self):
        script = self.capturing_installer()
        self.env["PIP_REPORT"] = "-"
        result = subprocess.run(["bash", str(script)], env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in (self.work / "pip_calls.jsonl").read_text().splitlines()]
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(call["env"]["PIP_REPORT"] == "-" for call in calls))

    def test_resume_refuses_missing_venv_metadata(self):
        script = self.installer()
        result = subprocess.run(["bash", str(script), "--resume"], env=self.env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pyvenv.cfg", result.stderr)
        self.assertFalse((self.work / "env").exists())

    def test_resume_refuses_system_site_packages(self):
        script = self.installer()
        env_dir = self.work / "env"
        env_dir.mkdir()
        (env_dir / "pyvenv.cfg").write_text("include-system-site-packages = true\nversion = 3.10.12\n")
        result = subprocess.run(["bash", str(script), "--resume"], env=self.env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("include-system-site-packages=false", result.stderr)

    def test_resume_refuses_environment_outside_work_root(self):
        script = self.installer()
        self.env["RAP_WAM_ENV"] = str(self.base / "outside")
        result = subprocess.run(["bash", str(script), "--resume"], env=self.env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("environment escapes", result.stderr)
        self.assertFalse((self.base / "outside").exists())

    def fixture(self, mode):
        root = self.work / "checkout" / "XPolicyLab"
        policy, utils = root / "policy" / "RAP_WAM", root / "utils"
        policy.mkdir(parents=True)
        utils.mkdir()
        for name in ("eval.sh", "runtime_env.sh", "owned_process.py"):
            shutil.copy2(POLICY / name, policy / name)
        (utils / "get_free_port.sh").write_text("#!/bin/bash\necho 12345\n")
        (utils / "wait_for_policy_server.sh").write_text("""#!/bin/bash
for attempt in $(seq 1 100); do
  kill -0 "$3" 2>/dev/null || exit 1
  [[ -f "$RAP_WAM_WORK_DIR/server.pid" ]] && exit 0
  sleep 0.02
done
exit 1
""")
        (policy / "setup_eval_policy_server.sh").write_text("""#!/bin/bash
exec "$RAP_WAM_BOOTSTRAP_PYTHON" -B -c '
import os,pathlib,time
if os.environ["TEST_MODE"] == "server_fail": raise SystemExit(7)
pathlib.Path(os.environ["RAP_WAM_WORK_DIR"], "server.pid").write_text(str(os.getpid()))
time.sleep(120)
'
""")
        (policy / "setup_eval_env_client.sh").write_text("""#!/bin/bash
exec "$RAP_WAM_BOOTSTRAP_PYTHON" -B -c '
import os,pathlib,subprocess,sys,time
root=pathlib.Path(os.environ["RAP_WAM_WORK_DIR"])
root.joinpath("client.pid").write_text(str(os.getpid()))
mode=os.environ["TEST_MODE"]
child_code="import time; time.sleep(120)"
if mode=="stubborn":
    child_code="import os,pathlib,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path(os.environ[\\"RAP_WAM_WORK_DIR\\"],\\"stubborn.ready\\").touch(); time.sleep(120)"
child=subprocess.Popen([sys.executable,"-B","-c",child_code])
root.joinpath("descendant.pid").write_text(str(child.pid))
if mode=="stubborn":
    while not root.joinpath("stubborn.ready").exists(): time.sleep(0.01)
if mode in ("interrupt", "parent_kill"): time.sleep(120)
raise SystemExit(9 if mode=="client_fail" else 0)
'
""")
        self.env["TEST_MODE"] = mode
        return policy / "eval.sh"

    def run_lifecycle(self, mode, expected):
        script = self.fixture(mode)
        observer = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(120)"])
        command = ["bash", str(script), "RoboDojo", "stack_bowls", "bundle", "arx_x5", "joint", "0", "0", "1", "unused", "unused"]
        proc = subprocess.Popen(command, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            if mode in ("interrupt", "parent_kill"):
                deadline = time.monotonic() + 10
                while not (self.work / "descendant.pid").exists():
                    if time.monotonic() > deadline:
                        self.fail("client did not start")
                    time.sleep(0.03)
                proc.send_signal(signal.SIGKILL if mode == "parent_kill" else signal.SIGTERM)
            output, _ = proc.communicate(timeout=30)
            self.assertEqual(proc.returncode, expected, output)
            self.assertEqual("[MAIN] eval finished" in output, expected == 0, output)
            for pid_file in self.work.glob("*.pid"):
                self.assertFalse(alive(int(pid_file.read_text())), f"Orphan: {pid_file}: {output}")
            self.assertIsNone(observer.poll(), "Unrelated process was terminated")
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=30)
            observer.terminate()
            observer.wait(timeout=10)

    def test_success_reaps_server_and_client_descendants(self):
        self.run_lifecycle("success", 0)

    def test_client_failure_preserves_exit_status_and_cleans_server(self):
        self.run_lifecycle("client_fail", 9)

    def test_server_failure_does_not_run_client(self):
        self.run_lifecycle("server_fail", 1)
        self.assertFalse((self.work / "client.pid").exists())

    def test_termination_cleans_owned_groups_only(self):
        self.run_lifecycle("interrupt", 143)

    def test_parent_death_is_detected_without_signaling_recorded_ids(self):
        self.run_lifecycle("parent_kill", -signal.SIGKILL)

    def test_sigterm_ignoring_descendant_is_forcibly_cleaned(self):
        self.run_lifecycle("stubborn", 0)
        status = list((self.work / "run").glob("client.*/status.json"))
        self.assertEqual(len(status), 1)
        self.assertEqual(json.loads(status[0].read_text())["cleanup"], "group_sigkill_requested")


if __name__ == "__main__":
    unittest.main()
