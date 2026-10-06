"""Exercise the actual workspace CLI, worker socket and persistent action bridge."""

import json
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from XPolicyLab.policy.EmbodiedRSI.runtime.workspace import POLICY_DIR, prepare_run
from XPolicyLab.policy.EmbodiedRSI.runtime.worker import run_worker

# A deterministic coding agent uses the same filesystem and CLI as the real one.
# Only the model process and Docker cleanup are replaced in this CPU regression.
AGENT = r'''
import json, os, subprocess, sys
from pathlib import Path
socket, workspace = sys.argv[1:]
os.environ["EAHARNESS_SOCKET"] = socket
os.chdir(workspace)

def call(op, code=None, ok=True):
    args = [sys.executable, "scripts/env.py", op]
    if code is not None:
        Path("submission/solution.py").write_text(code)
        args.append("submission/solution.py")
    result = subprocess.run(args, capture_output=True, text=True, timeout=20)
    assert result.returncode == (0 if ok else 1), result.stderr + result.stdout
    return json.loads(result.stdout)

assert call("status")["executions_used"] == 0
first = call("exec", "marker = 7\na = {k:v for k,v in get_observation()['state'].items() if k.endswith('joint_state')}\nstep(a)")
assert first["native_steps_used"] == 1
second = call("exec", "marker += 1\na = {k:v for k,v in get_observation()['state'].items() if k.endswith('ee_pose') or k.endswith('ee_joint_state')}\nstep(a)\nprint(marker)")
assert second["stdout"].strip() == "8", second
assert second["native_steps_used"] == 2
assert "forbidden in Test" in call("reset", ok=False)["error"]
assert call("status")["executions_used"] == 2
last = call("exec", "assert marker == 8\nstep(a)\nprint(get_observation()['state']['ee_joint_state'])")
assert last["success"] and last["native_steps_used"] == 3, last
'''


class ExecutionTests(unittest.TestCase):
    def test_workspace_programs_receive_live_and_terminal_feedback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = yaml.safe_load((POLICY_DIR / "deploy.yml").read_text())
            cfg.update(task_name="stack_bowls", run_dir=str(root / "run"), diagnostic=False)
            with patch.dict(os.environ, EMBODIEDRSI_RUN_DIR="", EMBODIEDRSI_CODEX_HOME=str(root)):
                spec, _ = prepare_run(cfg)
            observation = {
                "instruction": "Exercise workspace control",
                "vision": {"cam_head": {"color": np.zeros((16, 16, 3), dtype=np.uint8)}},
                "state": {"arm_joint_state": np.zeros(6), "ee_joint_state": np.zeros(1),
                          "ee_pose": np.array([0., 0., 0., 1., 0., 0., 0.])},
            }
            parent, worker = multiprocessing.Pipe()
            messages = []

            def evaluator():
                for step in range(1, 5):
                    if not parent.poll(30):
                        return
                    message = parent.recv()
                    messages.append(message)
                    if message["kind"] != "action":
                        return
                    observation["state"]["ee_joint_state"][:] = step / 10
                    parent.send({"kind": "transition", "observation": observation,
                                 "success": step == 3, "terminated": step == 3, "truncated": False})

            def launch(cfg, run, socket_dir, phase):
                return "test-agent", [sys.executable, "-c", AGENT,
                                      str(socket_dir / "session.sock"), str(run / "workspace")]

            thread = threading.Thread(target=evaluator, daemon=True)
            thread.start()
            previous = signal.getsignal(signal.SIGTERM)
            try:
                with patch("XPolicyLab.policy.EmbodiedRSI.runtime.worker.agent_command", side_effect=launch), \
                     patch("XPolicyLab.policy.EmbodiedRSI.runtime.worker.subprocess.run",
                           return_value=subprocess.CompletedProcess(["docker"], 0)):
                    run_worker(worker, spec, observation, 3, 0)
            finally:
                signal.signal(signal.SIGTERM, previous)
                thread.join(timeout=35)
                parent.close()
            episode = root / "run/episodes/episode_0000"
            status = json.loads((episode / "runtime.json").read_text())
            self.assertEqual(status["status"], "finished", status)
            self.assertEqual(status["reason"], "official_episode_end")
            self.assertEqual([m["kind"] for m in messages], ["action"] * 3 + ["finished"])
            self.assertEqual(status["native_steps"], 3)
            self.assertEqual((episode / "workspace/observations/000003/stdout.txt").read_text().strip(), "[0.3]")
            self.assertTrue((episode / "workspace/observations/000003/current_cam_head.png").is_file())
            self.assertFalse((episode / "workspace/ledger.jsonl").exists())
