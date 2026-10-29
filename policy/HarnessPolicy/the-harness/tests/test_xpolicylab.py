"""XPolicyLab support: the package's deploy, the bridge under XPolicyLab's episode loop,
the export, and an exported policy end to end with scripted model replies (no
simulator, no model calls)."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from harness_policy.config.agent import resolve_agent  # noqa: E402
from harness_policy.embodiments import load_embodiment  # noqa: E402

TEST_DEPLOY = ROOT / "tests" / "fixtures" / "deploy-test.toml"
DEFAULT = ROOT / "configs" / "default.toml"
ABLATIONS = ROOT / "configs" / "ablations"
HAS_IMAGES = all(importlib.util.find_spec(name) for name in ("numpy", "cv2"))
EMBODIMENT = load_embodiment("arx_x5")


class PackageDeployTests(unittest.TestCase):
    SERVICES = ["g05", "kinematics", "molmo2", "openwam", "sam3"]

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.package = Path(self.directory.name) / "RoboDojo" / "XPolicyLab" / "policy" / "HarnessPolicy"
        self.package.mkdir(parents=True)
        (self.package / "deploy.toml").write_text('weights = "checkpoints"\n')

    def tearDown(self):
        self.directory.cleanup()

    def deploy(self, gpus, free_gb=24.0):
        from harness_policy.connectors.package import package_deploy

        rows = [{"index": gpu, "name": "test", "used_gb": 24.0 - free_gb, "total_gb": 24.0} for gpu in gpus]
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))}):
            return package_deploy(self.package, self.SERVICES, self.package / "results" / "i", log=lambda m: None,
                                  gpu_rows=rows)

    def modes(self, deploy):
        return {name: (deploy.placement(name).mode, deploy.placement(name).gpu) for name in self.SERVICES}

    def test_all_resident_when_they_fit(self):
        deploy = self.deploy([2, 3, 4, 5])
        self.assertTrue(all(mode == "resident" for mode, _ in self.modes(deploy).values()))
        ports = [deploy.placement(name).port for name in self.SERVICES]
        self.assertEqual(len(set(ports)), len(ports))
        self.assertEqual(deploy.weights, (self.package / "checkpoints").resolve())
        self.assertEqual(deploy.robodojo, self.package.parents[2])
        self.assertEqual(deploy.xpolicylab, self.package.parents[1])

    def test_two_gpus_swap_the_learned_models(self):
        modes = self.modes(self.deploy([2, 3]))
        self.assertEqual(modes["g05"][0], "on_demand")
        self.assertEqual(modes["g05"][1], modes["openwam"][1])   # one swap group shares a GPU
        self.assertEqual(modes["sam3"][0], "resident")
        self.assertEqual(modes["molmo2"][0], "resident")

    def test_one_gpu_swaps_every_model_but_the_kinematics(self):
        modes = self.modes(self.deploy([5]))
        self.assertEqual(modes["kinematics"], ("resident", 5))
        self.assertEqual({modes[name][0] for name in ("g05", "openwam", "molmo2", "sam3")}, {"on_demand"})

    def test_too_little_memory_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "one model at a time"):
            self.deploy([5], free_gb=10.0)

    def test_the_package_may_name_external_services(self):
        (self.package / "deploy.toml").write_text(
            'weights = "/w"\n[services.molmo2]\nmode = "external"\nurl = "http://10.0.0.7:8830"\n')
        deploy = self.deploy([2])
        self.assertEqual(deploy.placement("molmo2").mode, "external")
        self.assertEqual(deploy.placement("molmo2").url, "http://10.0.0.7:8830")
        self.assertEqual(deploy.weights, Path("/w"))

    def test_no_two_instances_share_ports(self):
        first, second = self.deploy([2, 3, 4, 5]), self.deploy([2, 3, 4, 5])
        self.assertNotEqual(first.placement("sam3").port, second.placement("sam3").port)
        self.assertNotEqual(first.broker_port, second.broker_port)


class FakeKinematics:
    """/fk of the kinematics service: both link6s in the workspace, wherever the joints are."""

    def __init__(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body = json.dumps({"result": {"left_arm": [-0.3, -0.2, 1.0, 1, 0, 0, 0],
                                              "right_arm": [0.3, -0.2, 1.0, 1, 0, 0, 0]}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.send_response(200)
                self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class FakeHost:
    """XPolicyLab's side: its debug environment's observations and the standard episode
    loop of policy/demo_policy/deploy.py."""

    def __init__(self, model, step_limit=30):
        import numpy as np

        self.np = np
        self.model = model
        self.step_limit = step_limit
        self.actions = []
        self.empty_chunks = 0

    def obs(self):
        np = self.np
        camera = {"color": np.zeros((480, 640, 3), dtype=np.uint8), "shape": (480, 640)}
        state = {}
        for prefix in ("left_", "right_"):
            state[f"{prefix}arm_joint_state"] = np.zeros(6, dtype=np.float32)
            state[f"{prefix}ee_joint_state"] = np.zeros(1, dtype=np.float32)
            state[f"{prefix}ee_pose"] = np.array([0.0, -0.2, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return {"vision": {name: dict(camera) for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
                "instruction": "stack the bowls", "state": state, "additional_info": {"frequency": 25},
                "data_format_version": "v1.0", "env_idx": 0}

    def take_action(self, action):
        self.assertions(action)
        self.actions.append(action)
        self.step += 1

    def assertions(self, action):
        expected = {"left_arm_joint_state": 6, "left_ee_joint_state": 1, "right_arm_joint_state": 6,
                    "right_ee_joint_state": 1}
        if {key: len(value) for key, value in action.items()} != expected:
            raise AssertionError(f"action keys/dims {action}")

    def episode(self, double_reset=False):
        self.step = 0
        self.model.reset()
        if double_reset:      # XPolicyLab's debug client resets twice per episode
            self.model.reset()
        while self.step < self.step_limit:
            self.model.update_obs(self.obs())
            actions = self.model.get_action()
            if not actions:
                self.empty_chunks += 1
            for index, action in enumerate(actions):
                self.take_action(action)
                if self.step >= self.step_limit or index + 1 == len(actions):
                    break
                self.model.update_obs(self.obs())


@unittest.skipUnless(HAS_IMAGES, "numpy and cv2 are needed")
class BridgeTests(unittest.TestCase):
    def setUp(self):
        from harness_policy.connectors.xpolicylab import Bridge

        self.kinematics = FakeKinematics()
        self.bridge = Bridge(EMBODIMENT, kinematics_url=self.kinematics.url, poll_seconds=0.2, obs_timeout=20)

    def tearDown(self):
        self.bridge.close()
        self.kinematics.close()

    def program(self, ticks, think=0.0, results=None):
        observation = self.bridge.reset("task", 0)
        time.sleep(think)
        results = results if results is not None else []
        for step in range(ticks):
            result = self.bridge.call("tick", {"episode_id": observation.episode_id, "step": step,
                                               "command": [0.0] * 14})
            results.append(result)
            if result["state"]["ended"]:
                break
        self.bridge.program_over()
        return results

    def test_actions_follow_the_ticks_then_the_arms_hold(self):
        host = FakeHost(self.bridge_model(), step_limit=12)
        results = []
        thread = threading.Thread(target=self.program, args=(5, 0.5, results))
        self.bridge.host_reset()
        self.bridge.host_update_obs(host.obs())
        thread.start()
        host.step = 0
        while host.step < host.step_limit:
            actions = self.bridge.host_get_action()
            if not actions:
                host.empty_chunks += 1
            for action in actions:
                host.take_action(action)
            self.bridge.host_update_obs(host.obs())
        thread.join(10)
        self.assertEqual(len(results), 5)
        self.assertTrue(all(r["executed"] for r in results))
        self.assertEqual([r["state"]["step"] for r in results], [1, 2, 3, 4, 5])
        self.assertNotIn("step_limit", results[0]["state"])
        self.assertGreater(host.empty_chunks, 0)          # the model was thinking: no action, no step
        self.assertEqual(len(host.actions), 12)           # 5 from the program, then holds

    def bridge_model(self):
        return mock.Mock(reset=self.bridge.host_reset, update_obs=self.bridge.host_update_obs,
                         get_action=self.bridge.host_get_action)

    def test_a_host_reset_ends_a_waiting_tick(self):
        self.bridge.host_reset()
        self.bridge.host_update_obs(FakeHost(None).obs())
        results = []
        thread = threading.Thread(target=self.program, args=(3, 0.0, results))
        thread.start()
        time.sleep(0.5)            # the first tick waits for the host to take its action
        self.bridge.host_reset()   # ... which ends the episode instead
        thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertTrue(results[-1]["state"]["ended"])

    def test_profile_and_outcome_follow_the_contract(self):
        profile = self.bridge.call("profile", {})
        self.assertEqual(profile["observation_contract"], "xpolicylab")
        self.assertNotIn("step_limit", profile)
        self.assertNotIn("task", profile)
        self.assertIsNone(self.bridge.call("outcome", {"token": "x"})["success"])


@unittest.skipUnless(HAS_IMAGES and shutil.which("git") and (ROOT / ".git").exists(),
                     "numpy, cv2 and a git checkout are needed")
class ExportTests(unittest.TestCase):
    """`harness export xpolicylab`, and the exported policy under XPolicyLab's loop."""

    @classmethod
    def setUpClass(cls):
        from harness_policy.export.xpolicylab import STANDARD_SCRIPTS, export

        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        demo = root / "RoboDojo" / "XPolicyLab" / "policy" / "demo_policy"
        demo.mkdir(parents=True)
        for script in STANDARD_SCRIPTS:
            (demo / script).write_text("#!/bin/bash\ntrue\n" if script.endswith(".sh") else "\n")
        cls.package = demo.parent / "HarnessPolicy"
        cls.result = export(DEFAULT, name="HarnessPolicy", out=cls.package,
                            deploy=str(TEST_DEPLOY), xpolicylab=demo.parents[1], allow_dirty=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_sources_resolve_to_the_same_identity(self):
        agent = resolve_agent(DEFAULT, deploy=str(TEST_DEPLOY), name="default", strict=True)
        again = resolve_agent(agent.sources["agent"], model=agent.sources["model"], deploy=str(TEST_DEPLOY),
                              strict=True, name=agent.name, runtime_overrides=agent.sources["runtime_overrides"])
        self.assertEqual(again.identity, agent.identity)

    def test_the_folder_follows_xpolicylab(self):
        names = {path.name for path in self.package.iterdir()}
        self.assertTrue({"model.py", "deploy.py", "deploy.yml", "eval.sh", "setup_eval_policy_server.sh",
                         "setup_eval_env_client.sh", "__init__.py", "README.md", "install.sh",
                         "download_checkpoints.sh", "agent.json", "deploy.toml"} <= names)
        deploy_yml = (self.package / "deploy.yml").read_text()
        for key in ("policy_name: HarnessPolicy", "protocol: ws", "action_type: joint", "eval_batch: true",
                    "poll_seconds:"):
            self.assertIn(key, deploy_yml)
        self.assertIn(self.result["identity"], (self.package / "README.md").read_text())
        self.assertNotIn("imdecode", (self.package / "model.py").read_text())
        self.assertIn("sam3 simate", (self.package / "download_checkpoints.sh").read_text())
        self.assertEqual(self.result["services"], ["kinematics", "sam3", "simate"])
        self.assertNotIn("Molmo2", (self.package / "README.md").read_text())
        readme = (self.package / "README.md").read_text()
        self.assertIn("{name} arx_x5 joint <seed>".format(name="HarnessPolicy"), readme)
        self.assertIn("(`the-harness/embodiments/arx_x5`, RoboDojo's own cuRobo planner)", readme)

    def test_agent_json_is_machine_independent(self):
        from harness_policy.connectors.package import relocate

        document = json.loads((self.package / "agent.json").read_text())
        self.assertEqual(document["embodiment"], "arx_x5")  # RoboDojo's, by default
        self.assertNotIn(str(ROOT), json.dumps(document["sources"]))
        sources = relocate(document["sources"], ROOT)
        agent = resolve_agent(sources["agent"], model=sources["model"], deploy=str(TEST_DEPLOY), strict=True,
                              name=document["name"], runtime_overrides=sources["runtime_overrides"])
        self.assertEqual(agent.identity, document["identity"])

    def test_only_contract_agents_export(self):
        from harness_policy.export.xpolicylab import export

        with self.assertRaisesRegex(ValueError, "contract"):
            export(DEFAULT, sets=["runtime.contract=native"], name="X", out=Path(self.directory.name) / "x",
                   deploy=str(TEST_DEPLOY), xpolicylab=self.package.parents[1], allow_dirty=True)
        with self.assertRaisesRegex(ValueError, "task description"):
            export(ABLATIONS / "with-task-description.toml", name="X", out=Path(self.directory.name) / "x",
                   deploy=str(TEST_DEPLOY), xpolicylab=self.package.parents[1], allow_dirty=True)
        with self.assertRaisesRegex(ValueError, "identifier"):
            export(DEFAULT, name="my-policy", out=Path(self.directory.name) / "y",
                   deploy=str(TEST_DEPLOY), allow_dirty=True)

    def test_the_policy_runs_episodes_under_xpolicylab_s_loop(self):
        from harness_policy.connectors.package import HarnessModel

        kinematics = FakeKinematics()
        template = self.package / "deploy.toml"
        original = template.read_text()
        # Every service runs elsewhere: nothing to start, nothing on a GPU.
        template.write_text('weights = "checkpoints"\n' + "".join(
            f'[services.{name}]\nmode = "external"\nurl = "{kinematics.url}"\n'
            for name in self.result["services"]))
        script = Path(self.directory.name) / "script.json"
        script.write_text(json.dumps([
            "<python>\ns = get_state()\nprint('step_limit' in s, s['joint_position'][6])\n"
            "obs = get_observation(['cam_head', 'cam_left_wrist'])\nprint(obs.images['cam_head'].shape)\n"
            "print(hold(0.2))\n</python>",
            "<finish>done</finish>"]))
        messages = []
        handler = __import__("signal").getsignal(__import__("signal").SIGTERM)
        try:
            # The hosted VLM's client is built in every kernel, though no model is called.
            with mock.patch.dict(os.environ, {"HARNESS_SCRIPT": str(script), "CUDA_VISIBLE_DEVICES": "",
                                              "QWEN_BASE_URL": "http://127.0.0.1:9/v1"}):
                model = HarnessModel({"task_name": "stack_bowls", "poll_seconds": 0.5,
                                      "result_dir": str(Path(self.directory.name) / "results")},
                                     self.package, log=messages.append)
                host = FakeHost(model, step_limit=30)
                for _ in range(2):
                    host.episode(double_reset=True)
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline and sum("episode" in m and "calls=" in m for m in messages) < 2:
                    time.sleep(0.2)
                # A third episode the host stops after two actions, mid-program, as the
                # client does when it shuts the policy server down.
                model.reset()
                taken, deadline = 0, time.monotonic() + 60
                while taken < 2 and time.monotonic() < deadline:
                    model.update_obs(host.obs())
                    taken += len(model.get_action())
                model.close()
        finally:
            __import__("signal").signal(__import__("signal").SIGTERM, handler)
            template.write_text(original)
            kinematics.close()
        finished = [m for m in messages if "calls=" in m]
        self.assertEqual(len(finished), 3, messages)
        self.assertTrue(all(" finished " in m for m in finished[:2]), finished)
        self.assertIn(" episode_ended ", finished[2])
        self.assertEqual(len(host.actions), 2 * 30)       # every step of both episodes
        self.assertEqual(model.episodes, 3)   # double resets start no extra episode
        summaries = [json.loads((model.output / f"episode-{n:03d}" / "summary.json").read_text()) for n in (1, 2, 3)]
        self.assertEqual([s["status"] for s in summaries], ["finished", "finished", "episode_ended"])
        transcript = "".join(path.read_text(errors="replace") for path in model.output.rglob("*.jsonl"))
        self.assertIn("False 0.0", transcript)   # no step limit; the gripper reads its last command

    def test_a_batch_keeps_every_chunk_the_same_length(self):
        from harness_policy.connectors.package import HarnessModel

        kinematics = FakeKinematics()
        template = self.package / "deploy.toml"
        original = template.read_text()
        template.write_text('weights = "checkpoints"\n' + "".join(
            f'[services.{name}]\nmode = "external"\nurl = "{kinematics.url}"\n'
            for name in self.result["services"]))
        script = Path(self.directory.name) / "batch-script.json"
        script.write_text(json.dumps([
            "<python>\nprint(hold(0.04))\n</python>",
            "<finish>done</finish>"]))
        handler = __import__("signal").getsignal(__import__("signal").SIGTERM)
        try:
            with mock.patch.dict(os.environ, {"HARNESS_SCRIPT": str(script), "CUDA_VISIBLE_DEVICES": "",
                                              "QWEN_BASE_URL": "http://127.0.0.1:9/v1"}):
                model = HarnessModel({"task_name": "stack_bowls", "poll_seconds": 2,
                                      "result_dir": str(Path(self.directory.name) / "batch-results")},
                                     self.package, log=lambda _message: None)
                hosts = [FakeHost(model) for _ in range(2)]
                for host in hosts:
                    host.step = 0
                model.reset()
                model.reset()
                seen = [0, 0]
                deadline = time.monotonic() + 40
                while time.monotonic() < deadline and min(seen) < 1:
                    model.update_obs_batch([host.obs() for host in hosts])
                    chunks = model.get_action_batch([0, 1])
                    self.assertEqual(len(chunks), 2)
                    self.assertEqual(len(chunks[0]), len(chunks[1]))
                    if not chunks[0]:
                        continue
                    for index, chunk in enumerate(chunks):
                        hosts[index].assertions(chunk[0])
                        seen[index] += 1
                # A tick that is ready stays ready while another environment is still
                # computing: the client cannot step one of them.
                ready = model._slot(0)
                waiting = model._slot(1)
                planted = hosts[0].obs()["state"]["left_arm_joint_state"]
                with ready.bridge.lock:
                    ready.bridge.pending = [{"left_arm_joint_state": planted}]
                    ready.bridge.idle = False
                waiting.bridge.host_status = lambda: "wait"
                model.poll_seconds = 0.05
                model.update_obs_batch([host.obs() for host in hosts])
                held = model.get_action_batch([0, 1])
                self.assertEqual(held, [[], []])
                self.assertIsNotNone(ready.bridge.pending)
                model.close()
        finally:
            __import__("signal").signal(__import__("signal").SIGTERM, handler)
            template.write_text(original)
            kinematics.close()
        self.assertGreaterEqual(min(seen), 1, seen)
        self.assertEqual(model.episodes, 2)


if __name__ == "__main__":
    unittest.main()
