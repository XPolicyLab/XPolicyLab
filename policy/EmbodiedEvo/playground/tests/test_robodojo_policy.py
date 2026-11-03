"""Real XPolicyLab transport with a small deterministic code worker, no GPU."""

import asyncio
import json
import multiprocessing
import socket
import subprocess
import sys
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

try:
    import websockets.asyncio.server
except ImportError:
    raise unittest.SkipTest("Run with .venv-robodojo-policy (websockets >=15)")

from integrations.robodojo_policy.bootstrap import imports, VENDOR
if not (VENDOR / "XPolicyLab/client_server/ws/model_server.py").is_file():
    raise unittest.SkipTest("Provision repository-local RoboDojo/XPolicyLab for transport tests")
imports()
from client_server.ws.model_client import WsModelClient
from client_server.ws.model_server import PolicyServer, PolicyServerConfig
from integrations.robodojo_policy.bridge import BridgeTask, policy_observation, validate_action
from integrations.robodojo_policy.run import pin
from orchestrator.config import config_dict, load_task
from orchestrator.files import hashes
from XPolicyLab.policy.EmbodiedEvo.model import Model
from XPolicyLab.policy.EmbodiedEvo.deploy import AdapterFailure, eval_one_episode, playground_one_session
from integrations.robodojo_policy.transport import forward


def observation(value=0):
    return {"instruction": "Test", "state": {
        "left_arm_joint_state": np.full(6, value), "right_arm_joint_state": np.zeros(6),
        "left_ee_joint_state": np.ones(1), "right_ee_joint_state": np.ones(1),
        "left_ee_pose": np.array([0., 0., 0., 1., 0., 0., 0.]),
        "right_ee_pose": np.array([0., 0., 0., 1., 0., 0., 0.]),
    }, "vision": {"cam_head": {"color": np.zeros((8, 8, 3), dtype=np.uint8)}}}


def code_worker(connection, spec, obs, limit, episode):
    env = BridgeTask(connection, obs, limit)
    env.reset(seed=0)
    env.step("remember = 7\na = {k:v for k,v in get_observation()['state'].items() if k.endswith('joint_state')}\nstep(a)")
    # Second submission sees both persistent Python state and fresh observation.
    env.step("assert remember == 7\na['left_arm_joint_state'] = get_observation()['state']['left_arm_joint_state'] + remember\nstep(a)")
    connection.send({"kind": "finished", "success": env.success})


def stopped_worker(connection, *_args):
    connection.send({"kind": "finished", "success": False})
    connection.close()


def playground_worker(connection, spec, obs, limit, episode):
    env = BridgeTask(connection, obs, limit, phase="playground")
    env.execution_phase = "playground"
    env.reset(seed=0)
    env.step("remember = 7\na = {k:v for k,v in get_observation()['state'].items() if k.endswith('joint_state')}\nstep(a)")
    assert env.done and env.control_steps == 1
    env.step("reset()\nassert remember == 7\na['left_arm_joint_state'] = get_observation()['state']['left_arm_joint_state'] + remember\nstep(a)")
    assert env.done and env.control_steps == 2
    assert env.max_steps == 2 and env.native_step_limit == 1
    connection.send({"kind": "finished", "success": env.success})
    connection.close()


class AdapterTests(unittest.TestCase):
    def test_numpy_formatting_in_fresh_restricted_worker(self):
        result = subprocess.run([sys.executable, "-B", "-c", """
from tests.fixtures import observation
from integrations.robodojo_policy.bridge import BridgeTask
env = BridgeTask(None, observation(), 2)
env.reset(seed=0)
info = env.step("a = get_observation()['state']['left_ee_pose']; print(a); print({'pose': a})")[-1]
assert info['ok'], info['stderr']
assert 'array(' in info['stdout']
info = env.step('import os')[-1]
assert not info['ok']
"""], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_playground_reset_over_websocket_and_unix_transport(self):
        class Env:
            step_lim = 1
            env_seeds = [0]
            take_action_cnt = [0]
            end_flag = [False]
            success = [False]
            resets = 0
            total_actions = 0
            reward_ready = False
            reward_updates = 0
            obs_manager = SimpleNamespace(instruction=["Test"])

            def run_reward(self):
                self.reward_ready = True
                self.reward_updates += 1

            def is_episode_end(self):
                self.end_flag = [not self.reward_ready or self.take_action_cnt[0] >= self.step_lim]
                return self.end_flag[0]

            def get_obs(self):
                return observation(3 if self.resets else 0)

            def get_obs_batch(self, *_args, **_kwargs):
                return [self.get_obs()]

            def take_action(self, action):
                expected = 10 if self.resets else 0
                np.testing.assert_array_equal(action["left_arm_joint_state"], np.full(6, expected))
                self.take_action_cnt[0] += 1
                self.total_actions += 1

            def reset(self, seed):
                assert seed == [0]
                self.resets += 1
                self.reward_ready = False
                self.take_action_cnt = [0]
                self.end_flag = [False]

        with tempfile.TemporaryDirectory() as temporary, patch(
            "XPolicyLab.policy.EmbodiedEvo.model.run_worker", playground_worker
        ):
            model = Model({"run_dir": temporary, "phase": "playground"})
            loop = asyncio.new_event_loop()
            thread = threading.Thread(target=loop.run_forever)
            thread.start()
            server = PolicyServer(model, PolicyServerConfig(host="127.0.0.1", port=0))
            asyncio.run_coroutine_threadsafe(server.start(), loop).result(10)
            unix_path = str(Path(temporary) / "policy.sock")
            port = int(server.url.rsplit(":", 1)[1])
            try:
                with forward(unix_path, (socket.AF_INET, ("127.0.0.1", port)), unix=True):
                    with forward(("127.0.0.1", 0), (socket.AF_UNIX, unix_path)) as address:
                        client = WsModelClient(url=f"ws://127.0.0.1:{address[1]}",
                                               evaluation_id="pg", trial_id="trial",
                                               action_case_id="case", request_timeout_s=10)
                        try:
                            env = Env()
                            playground_one_session(env, client)
                            self.assertEqual(env.resets, 1)
                            self.assertEqual(env.total_actions, 2)
                            self.assertEqual(env.reward_updates, 2)
                            self.assertEqual(model.episode, 0)
                        finally:
                            client.close()
            finally:
                model.close()
                asyncio.run_coroutine_threadsafe(server.stop(), loop).result(10)
                loop.call_soon_threadsafe(loop.stop)
                thread.join()
                loop.close()

    def test_pin_preserves_source_model_budget_and_only_frozen_knowledge(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "inputs/primitives").mkdir(parents=True)
            (source / "inputs/test.md").write_text("Solve the task. There is no action-count budget.")
            (source / "inputs/primitives/primitives.md").write_text("Fixed four primitives")
            (source / "frozen_harness/skills").mkdir(parents=True)
            (source / "frozen_harness/skills/skill.py").write_text("def learned(): pass\n")
            (source / "workspace/submission").mkdir(parents=True)
            (source / "workspace/submission/playground.py").write_text("not transferable")
            cfg = config_dict(load_task("RoboDojo", env_task="push_T"))
            cfg["agent"]["model"] = "source-model"
            cfg["test"]["budget"] = 7
            state = {"phase": "frozen", "playground_valid": True,
                     "harness": hashes(source / "frozen_harness"), "definition": {"config": cfg}}
            (source / "protocol.json").write_text(json.dumps(state))
            repo = root / "repo"
            (repo / "integrations").mkdir(parents=True)
            (repo / "vendor/robodojo").mkdir(parents=True)
            provision = {"commit": "test-source-commit"}
            (repo / "vendor/robodojo/.embodiedevo-provisioned.json").write_text(
                json.dumps(provision)
            )
            with patch("integrations.robodojo_policy.run.REPO", repo):
                _, run, spec = pin(SimpleNamespace(source_run=source, run_dir=root / "new-run",
                                                  env_task="push_T", seed=1, episodes=1, gpu="0"))
            self.assertEqual(spec["vendor_provision"], provision)
            self.assertEqual(spec["task_config"]["agent"]["model"], "source-model")
            self.assertEqual(spec["task_config"]["test"]["budget"], 7)
            self.assertEqual(hashes(run / "frozen_harness"), state["harness"])
            self.assertFalse(list(run.rglob("playground.py")))
            self.assertEqual((run / "inputs/test.md").read_text(),
                             "Solve the task. The official action-count budget applies.")

    def test_action_validation_and_no_privileged_fields(self):
        obs = observation()
        obs["score"] = 99
        obs["layout_id"] = 123
        public = policy_observation(obs)
        self.assertNotIn("score", public)
        self.assertNotIn("layout_id", public)
        action = {k:v for k,v in obs["state"].items() if k.endswith("joint_state")}
        validate_action(action, obs["state"])
        action["left_ee_pose"] = obs["state"]["left_ee_pose"]
        with self.assertRaises(ValueError):
            validate_action(action, obs["state"])

    def test_reset_forbidden_and_terminal_actions_do_not_advance(self):
        parent, child = multiprocessing.Pipe()
        env = BridgeTask(child, observation(), 2)
        env.reset(seed=0)
        with self.assertRaises(PermissionError):
            env.reset_from_program()
        env.terminated, env.success = True, True
        env.control_step({})
        self.assertFalse(parent.poll())
        self.assertEqual(env.control_steps, 0)
        parent.close()
        child.close()

    def test_official_websocket_closed_loop_and_new_episode_state(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "XPolicyLab.policy.EmbodiedEvo.model.run_worker", code_worker
        ):
            model = Model({"run_dir": temporary})
            loop = asyncio.new_event_loop()
            thread = threading.Thread(target=loop.run_forever)
            thread.start()
            server = PolicyServer(model, PolicyServerConfig(host="127.0.0.1", port=0))
            asyncio.run_coroutine_threadsafe(server.start(), loop).result(10)
            client = WsModelClient(url=server.url, evaluation_id="test", trial_id="trial",
                                   action_case_id="case", request_timeout_s=10)
            try:
                for _ in range(2):
                    client.call("reset")
                    client.call("prepare_case", obs={"native_step_limit": 2})
                    client.call("update_obs", obs=observation())
                    first = self.next_action(client)
                    np.testing.assert_array_equal(first[0]["left_arm_joint_state"], np.zeros(6))
                    client.call("update_obs", obs=observation(3))
                    second = self.next_action(client)
                    np.testing.assert_array_equal(second[0]["left_arm_joint_state"], np.full(6, 10))
                    client.call("update_obs", obs=observation(10))
                    client.call("trial_end", obs={"success": True, "terminated": True, "truncated": False})
                self.assertEqual(model.episode, 1)
                # An agent may stop before the environment. The adapter must
                # hold safely and tolerate the already closed worker pipe.
                with patch("XPolicyLab.policy.EmbodiedEvo.model.run_worker", stopped_worker):
                    client.call("reset")
                    client.call("prepare_case", obs={"native_step_limit": 2})
                    obs = observation()
                    obs["state"]["left_ee_joint_state"] = [1.0]
                    client.call("update_obs", obs=obs)
                    hold = self.next_action(client)
                    np.testing.assert_array_equal(hold[0]["left_ee_pose"], obs["state"]["left_ee_pose"])
                    client.call("trial_end", obs={"success": False, "terminated": False, "truncated": True})
                    self.assertEqual(model.episode, 2)
            finally:
                client.close()
                model.close()
                asyncio.run_coroutine_threadsafe(server.stop(), loop).result(10)
                loop.call_soon_threadsafe(loop.stop)
                thread.join()
                loop.close()

    @staticmethod
    def next_action(client):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            actions = client.call("get_action")
            if actions:
                return actions
        raise AssertionError("Timed out waiting for code worker")

    def test_deploy_waits_for_native_end_and_sends_no_score(self):
        class Env:
            step_lim = 3
            take_action_cnt = [0]
            end_flag = [False]
            success = [True]  # Native initial viability flag is not completion.
            def is_episode_end(self):
                if self.take_action_cnt[0] >= self.step_lim:
                    self.end_flag, self.success = [True], [False]
                return self.end_flag[0]
            def get_obs(self):
                return observation()
            def get_obs_batch(self, *_args, **_kwargs):
                return [observation()]
            def take_action(self, action):
                self.take_action_cnt[0] += 1
        class Client:
            result = None
            calls = 0
            def call(self, func_name, obs=None):
                if func_name == "get_action":
                    self.calls += 1
                    return [] if self.calls == 1 else [{}]
                if func_name == "trial_end":
                    self.result = obs
        env, client = Env(), Client()
        eval_one_episode(env, client)
        self.assertEqual(env.take_action_cnt[0], 3)
        self.assertEqual(client.result, {"success": False, "terminated": False, "truncated": True})

    def test_policy_errors_abort_instead_of_skipping_to_another_layout(self):
        class Client:
            def call(self, *args, **kwargs):
                raise RuntimeError("broken policy")

        with self.assertRaises(AdapterFailure) as caught:
            eval_one_episode(object(), Client())
        # Upstream's per-layout `except Exception` must not swallow this.
        self.assertNotIsInstance(caught.exception, Exception)
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_finished_policy_only_holds_current_pose(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Model({"run_dir": temporary})
            model.prepare_case({"native_step_limit": 3})
            obs = observation()
            # Native gripper observations are lists, whereas poses are arrays.
            obs["state"]["left_ee_joint_state"] = [1.0]
            model.update_obs(obs)
            model.worker = object()
            model.finished = True
            actions = model.get_action()
            expected = {key: value for key, value in observation()["state"].items()
                        if key.endswith("ee_pose") or key.endswith("ee_joint_state")}
            self.assertEqual(set(actions[0]), set(expected))
            for key in expected:
                np.testing.assert_array_equal(actions[0][key], expected[key])


if __name__ == "__main__":
    unittest.main()
