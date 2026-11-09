"""WALL-OSS ablation only: native EE, fixed routes, no simulator/model/processes."""

import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

from harness_policy.config import load_layered
from harness_policy.connectors.xpolicylab import Bridge
from harness_policy.connectors.package import HarnessModel
from harness_policy.core.worker import WorkerAPI
from harness_policy.embodiments import load_embodiment
from harness_policy.runtime.events import EpisodeEnded
from harness_policy.runtime.chat import ScriptedChat
from harness_policy.runtime.routing import task_route, validate_task_routing
from harness_policy.runtime.driver import run_program_episode
from harness_policy.services import policy_server
from test_run_policy import Policy, Worker, session
from test_program_driver import FakeEnvironment, _config
from skills.vla import route_instruction

# A fixture for the validator only. The shipped ablation does not route by task name.
ROUTING = {"default": "wall-oss", "tasks": {task: "own" for task in (
    "press_by_number", "solve_equation", "arrange_largest_number", "general_pickup")}}


def ee_action(x=0.123456789):
    return {"left_ee_pose": [x, -0.2, 1.0, 0.75, 0.1, 0.2, 0.3],
            "right_ee_pose": [-x, -0.2, 1.0, 0.75, -0.1, -0.2, -0.3],
            "left_ee_joint_state": [-0.3], "right_ee_joint_state": [1.4]}


class NativeWorker(Worker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.actions = []

    def call(self, name, **arguments):
        if name == "tick" and isinstance(arguments["command"], dict):
            action = arguments["command"]
            self.actions.append(action)
            self.step += 1
            self.joints[0] = 0.1  # moves once, then stays still: must not stall out
            return {"executed": True, "shield": {"verdict": "pass", "reasons": [], "command": action},
                    "state": self.state()}
        return super().call(name, **arguments)


class KFPolicy:
    def __init__(self, size=32):
        self.size, self.requests, self.calls = size, [], []

    def call(self, name, obs=None):
        self.calls.append(name)
        if name == "reset":
            return {"version": 1, "prev_sim_frame": None, "mem_frame_ids": [], "rng_state": None}
        if name != "infer":
            raise AssertionError(name)
        self.requests.append(obs)
        frame = obs["sim_frame"]
        return {"actions": [ee_action(frame + i) for i in range(self.size)],
                "kfmem_state": {"version": 1, "prev_sim_frame": frame, "mem_frame_ids": [],
                                "last_sel_frame": 0, "consec_skip": 0, "dump_count": 0, "rng_state": None}}


def native_session(worker, policy):
    s = session(worker, policy)
    s.policies = {"wall-oss": {"url": "ws://127.0.0.1:19999", "memory": "kfmem",
                               "execute_steps": 16, "action_chunk_length": 32}}
    s._policy_clients = {"wall-oss": policy}
    s._policy_action_mode = "ee_passthrough"
    s._fixed_task_route = "wall-oss"
    s.ik = mock.Mock(side_effect=AssertionError("harness IK is forbidden"))
    return s


class CellSpanTests(unittest.TestCase):
    def test_the_program_keeps_the_episode_and_the_policy_is_not_started(self):
        worker, policy = Worker(instruction="Stack the bowls."), Policy()
        s = session(worker, policy)
        s.policies = {"wall-oss": {"url": "ws://wall"}}
        s._policy_clients = {"wall-oss": policy}
        s._policy_span = "cell"
        self.assertIsNone(s.start("B", "stack"))
        self.assertEqual(s._strategy["strategy"], "B")
        self.assertIsNone(s.start("A", "hand the bowls over"))
        self.assertEqual(s._strategy["strategy"], "A")
        self.assertFalse(s.__dict__.get("_handover_locked", False))
        self.assertEqual(policy.calls, [])
        self.assertIsNone(s.__policy_to_end__())

    def test_a_read_still_stays_on_own_tools(self):
        worker, policy = Worker(instruction="Read the number cards."), Policy()
        s = session(worker, policy)
        s._policy_span = "cell"
        self.assertIsNone(s.start("A", "a read"))
        self.assertEqual(s._strategy["strategy"], "B")

    def test_10cm_pickup_does_not_force_cell_owner_b(self):
        worker, policy = Worker(instruction="Pick the block up by 10 cm."), Policy()
        s = session(worker, policy)
        s._policy_span = "cell"
        self.assertIsNone(s.start("A", "pickup"))
        self.assertEqual(s._strategy["strategy"], "A")
        self.assertEqual(policy.calls, [])


class NativeRunPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.patch = mock.patch.object(policy_server, "POLICY_LOCK_DIR", self.temp.name)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def test_full_32_chunk_replans_every_16_native_ticks_to_episode_end(self):
        worker, policy = NativeWorker(step_limit=180, instruction="Read the number cards."), KFPolicy()
        s = native_session(worker, policy)
        s._policy_progress_seconds = 0.01
        s._tracked_positions = mock.Mock(side_effect=AssertionError("whole task has no progress watch"))
        with self.assertRaises(EpisodeEnded):
            s.start("B", "model requested own tools", policy="wrong-policy")
        self.assertEqual(s._strategy["strategy"], "A")
        self.assertEqual(s._strategy["policy"], "wall-oss")
        self.assertEqual(worker.step, 180)  # exceeds the 5-second default stall window
        self.assertEqual([obs["sim_frame"] for obs in policy.requests], list(range(0, 180, 16)))
        self.assertEqual(policy.calls.count("reset"), 1)
        self.assertEqual(worker.actions, [ee_action(i) for i in range(180)])
        self.assertTrue(all(obs["instruction"] == worker.instruction for obs in policy.requests))
        self.assertEqual([frame["sim_frame"] for frame in policy.requests[1]["kfmem_history"]], list(range(1, 17)))
        s.ik.assert_not_called()
        s._tracked_positions.assert_not_called()

    def test_bad_horizon_fails_before_motion_and_is_not_retried(self):
        worker, policy = NativeWorker(), KFPolicy(size=16)
        s = native_session(worker, policy)
        with self.assertRaisesRegex(RuntimeError, "full 32-step"):
            s.start("A", "fixed")
        self.assertEqual(worker.actions, [])
        self.assertEqual(policy.calls, ["reset", "infer"])

    def test_native_cannot_be_silently_converted_for_one_arm(self):
        s = native_session(NativeWorker(), KFPolicy())
        with self.assertRaisesRegex(ValueError, "both arms"):
            s.run_policy("wall-oss", arms="left_arm")

    def test_missing_action_chunk_fails_explicitly_without_motion(self):
        class MissingActions(KFPolicy):
            def call(self, name, obs=None):
                result = super().call(name, obs)
                if name == "infer":
                    result.pop("actions")
                return result

        worker, policy = NativeWorker(), MissingActions()
        s = native_session(worker, policy)
        with self.assertRaisesRegex(RuntimeError, "did not return an action chunk"):
            s.start("A", "fixed")
        self.assertEqual(worker.actions, [])
        self.assertEqual(policy.calls, ["reset", "infer"])

    def test_own_motion_gap_rebases_wire_frames_and_reports_restart_once(self):
        worker, policy = NativeWorker(step_limit=500), KFPolicy()
        s = native_session(worker, policy)
        s._fixed_task_route = None
        s._policy_span = "cell"
        s.start("B", "bounded policy")
        s.run_policy("wall-oss", max_seconds=0.2, stall_seconds=None)
        before = len(worker.actions)
        # A real own-tool interval invalidates the unused policy actions.
        worker.step += 100
        worker.joints[0] += 0.2
        s.state = worker.state()
        trace = s.run_policy("wall-oss", max_seconds=0.2, from_anywhere=True, stall_seconds=None)
        self.assertEqual(policy.requests[-1]["sim_frame"], 0)
        self.assertEqual(len(worker.actions) - before, trace.ticks)
        self.assertEqual(trace.event.detail["memory_restarts"][0]["host_step"], 105)
        self.assertEqual(policy.calls.count("reset"), 1)  # no global service reset or retry
        self.assertGreaterEqual(trace.end_tick, 105)

    def test_own_route_keeps_existing_tools_and_forbids_policy_handover(self):
        worker, policy = Worker(instruction="Pick up the object."), Policy()
        s = session(worker, policy)
        s._fixed_task_route = "own"
        self.assertIsNone(s.start("A", "hand it over"))
        self.assertIsNone(s.start("C", "try another policy", policy="g05"))
        self.assertEqual(s._strategy["strategy"], "B")
        with self.assertRaisesRegex(ValueError, "own tools only"):
            s.run_policy("g05")
        s.plan_path = mock.Mock(return_value=[[0.1] + [0.0] * 5])
        self.assertTrue(s.move_joints("left_arm", [0.1] + [0.0] * 5, max_seconds=1).ok)
        self.assertTrue(s.move_eef("left_arm", worker.state()["left_ee_pose"], max_seconds=1).ok)
        self.assertEqual(policy.calls, [])


class NativeBridgeTests(unittest.TestCase):
    def setUp(self):
        self.bridge = Bridge(load_embodiment("arx_x5"), kinematics_url=None,
                             ee_passthrough=True, poll_seconds=0.1, obs_timeout=2)
        self.obs = {"state": {"left_arm_joint_state": [0.] * 6, "right_arm_joint_state": [0.] * 6,
                              **ee_action()}, "vision": {}, "instruction": "Task."}
        self.bridge.host_reset()
        self.bridge.host_update_obs(self.obs)
        self.episode = self.bridge.reset("stack_bowls", 0).episode_id

    def tearDown(self):
        self.bridge.close()

    def test_worker_rpc_to_host_preserves_all_four_keys_and_values(self):
        action = ee_action()
        results = []
        self.bridge.shield.check = mock.Mock(side_effect=AssertionError("no joint shield for EE"))
        self.bridge._kinematics = mock.Mock(side_effect=AssertionError("no harness IK/FK"))
        api = WorkerAPI(self.bridge)

        def tick():
            results.append(api.dispatch("call", {"name": "tick", "arguments": {
                "episode_id": self.episode, "step": 0, "command": action, "source": "policy"}}))

        thread = threading.Thread(target=tick, daemon=True)
        thread.start()
        sent = self.bridge.host_get_action()
        self.bridge.host_update_obs(self.obs)
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(sent, [action])
        self.assertEqual(results[0]["shield"]["command"], action)
        self.assertTrue(results[0]["executed"])
        self.assertEqual(results[0]["state"]["step"], 1)
        self.bridge._kinematics.assert_not_called()
        self.bridge.shield.check.assert_not_called()
        with self.assertRaisesRegex(ValueError, "Stale tick"):
            self.bridge._call_tick(self.episode, 0, action, source="policy")
        self.bridge.program_over()
        self.assertEqual(set(self.bridge.host_get_action()[0]), set(action))

    def test_native_rejects_bad_shapes_nonfinite_mixed_keys_or_nonpolicy_source(self):
        invalid = [dict(ee_action(), left_ee_pose=[0.]),
                   dict(ee_action(), left_ee_joint_state=[float("nan")]),
                   dict(ee_action(), left_arm_joint_state=[0.] * 6),
                   dict(ee_action(), left_ee_joint_state=[True])]
        for command in invalid:
            with self.subTest(command=command), self.assertRaises(ValueError):
                self.bridge._call_tick(self.episode, 0, command, source="policy")
        with self.assertRaises(ValueError):
            self.bridge._call_tick(self.episode, 0, ee_action())
        self.bridge.ee_passthrough = False
        with self.assertRaises(ValueError):
            self.bridge._call_tick(self.episode, 0, ee_action(), source="policy")
        self.assertIsNone(self.bridge.pending)

    def test_reset_rejects_old_episode_without_a_motion_retry(self):
        self.bridge.host_reset()
        result = self.bridge._call_tick(self.episode, 0, ee_action(), source="policy")
        self.assertFalse(result["executed"])
        self.assertTrue(result["state"]["ended"])
        self.assertIsNone(self.bridge.pending)

    def test_exported_model_enables_bridge_mode_only_for_explicit_ablation(self):
        for mode in (None, "ee_passthrough"):
            model = object.__new__(HarnessModel)
            model._embodiment, model._kinematics_url, model.poll_seconds = load_embodiment("arx_x5"), None, 0.1
            model.agent = mock.Mock(runtime={} if mode is None else {"policy_action_mode": mode})
            server = mock.Mock(server_port=12345)
            with mock.patch("harness_policy.core.worker.create_server", return_value=server), \
                 mock.patch("harness_policy.connectors.package.threading.Thread"):
                slot = model._make_slot()
            self.assertEqual(slot.bridge.ee_passthrough, mode == "ee_passthrough")
            slot.bridge.close()


class RoutingTests(unittest.TestCase):
    BASKET = ("Put fruit objects into the left basket, kitchen utensil objects into "
              "the middle basket, and toy objects into the right basket, then reset the robot arm.")
    POUR = ("Pour the liquid from the first turquoise bottle into the first brown bowl, "
            "from the second violet bottle into the second black bowl, and from the third "
            "red bottle into the third white bowl. Then reset the robot arm.")

    def test_wall_language_gates_and_default_isolation(self):
        positive = (self.BASKET, self.POUR,
                    "Place metal items in the centre basket.",
                    "Pour liquid from the green bottle into the purple bowl.")
        for instruction in positive:
            for text in (instruction, instruction.upper(), instruction.replace(" ", "\n")):
                with self.subTest(instruction=text):
                    self.assertEqual(route_instruction(text, wall_only=True), "own")
                    self.assertEqual(route_instruction(text), "simate")
        for instruction in ("", "classify_objects_by_language", "pour_by_language",
                            "Sort the objects by category into baskets.",
                            "Put fruit objects into a basket.",
                            "Put objects into the left basket.",
                            "Pour the liquid from the bottle into the bowl.",
                            "Pour from the red bottle into the bowl.",
                            "Put the red bottle in the white bowl.",
                            "Place all stacking toy pieces onto the correct pegs."):
            with self.subTest(instruction=instruction):
                self.assertEqual(route_instruction(instruction, wall_only=True), "simate")

    def test_wall_binding_rejects_policy_escape_before_service_or_motion(self):
        for instruction in (self.BASKET, self.POUR):
            worker, policy = Worker(instruction=instruction), Policy()
            s = session(worker, policy)
            s.policies = {"wall-oss": {"url": "ws://wall"}}
            s._policy_clients = {"wall-oss": policy}
            s._policy_span = "cell"
            s._fixed_task_route = "wall-oss"  # original instruction gate takes precedence
            with mock.patch("skills.vla.ensure") as ensure:
                for strategy in ("A", "B", "C", "A"):
                    self.assertIsNone(s.start(strategy, "attempt handover"))
                    self.assertEqual(s._strategy["strategy"], "B")
                for kwargs in ({}, {"instruction": "Sort the objects.", "reset": True},
                               {"from_anywhere": True, "max_seconds": 1}):
                    with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "own tools only"):
                        s.run_policy("wall-oss", **kwargs)
                worker.instruction = ""  # lost observation does not erase the original binding
                self.assertIsNone(s.start("A", "try after missing instruction"))
                with self.assertRaisesRegex(ValueError, "own tools only"):
                    s.run_policy("wall-oss")
                ensure.assert_not_called()
            self.assertEqual(policy.calls, [])
            self.assertEqual(worker.step, 0)

    def test_direct_wall_call_also_checks_language_before_start(self):
        s = session(Worker(instruction=self.POUR), Policy())
        s.policies, s._policy_span = {"wall-oss": {"url": "ws://wall"}}, "cell"
        with mock.patch("skills.vla.ensure") as ensure, self.assertRaisesRegex(ValueError, "own tools only"):
            s.run_policy("wall-oss", instruction="A generic pour.")
        ensure.assert_not_called()

    def test_wall_gate_is_not_enabled_for_episode_span_or_other_policies(self):
        for span, policies in (("episode", {"wall-oss": {}}),
                               ("cell", {"simate": {}}),
                               ("cell", {"wall-oss": {}, "simate": {}})):
            s = session(Worker(instruction=self.BASKET), Policy())
            s._policy_span, s.policies = span, policies
            s.run_policy = mock.Mock()
            s.start("A", "ordinary handover")
            self.assertEqual(s._strategy["strategy"], "A")

    def test_plain_classification_can_still_call_a_wall_slice(self):
        worker, policy = NativeWorker(instruction="Sort the objects by category."), KFPolicy()
        s = native_session(worker, policy)
        s._fixed_task_route, s._policy_span = None, "cell"
        s.start("B", "ordinary classification")
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(policy_server, "POLICY_LOCK_DIR", temp):
            trace = s.run_policy("wall-oss", max_seconds=0.2, stall_seconds=None)
        self.assertGreater(trace.ticks, 0)
        self.assertEqual(policy.calls, ["reset", "infer"])

    def test_only_four_fixed_own_routes_and_random_inheritance(self):
        validate_task_routing(ROUTING, {"wall-oss": {}})
        for task in ROUTING["tasks"]:
            self.assertEqual(task_route(ROUTING, task), "own")
            self.assertEqual(task_route(ROUTING, task + "_random"), "own")
        for task in ("stack_bowls", "stack_bowls_random", "unlisted", "number cards"):
            self.assertEqual(task_route(ROUTING, task), "wall-oss")
            self.assertIsNone(task_route(None, task))

    def test_invalid_routes_cannot_switch_policy_or_override_random(self):
        for routing in ({"default": "other", "tasks": {}},
                        {"default": "wall-oss", "tasks": {"stack_bowls": "other"}},
                        {"default": "wall-oss", "tasks": {"general_pickup_random": "wall-oss"}}):
            with self.subTest(routing=routing), self.assertRaises(ValueError):
                validate_task_routing(routing, {"wall-oss": {}})

    def test_default_uses_four_semantic_instruction_gates_not_task_ids(self):
        for gate in ("number cards", "required number of times", "missing number or operator",
                     "largest possible number"):
            self.assertEqual(route_instruction(gate.upper()), "own")
        for wall_only in (False, True):
            for text in ("by 10 cm", "Pick up the speckled ceramic dove by 10 cm."):
                self.assertEqual(route_instruction(text, wall_only=wall_only), "simate")
        for task in ROUTING["tasks"]:
            self.assertEqual(route_instruction(task), "simate")
        for task in ("solve_equation", "stack_bowls"):
            worker, policy = Worker(instruction="Stack the bowls."), Policy()
            original = worker.call
            worker.call = lambda name, **args: dict(original(name, **args), task_id=task) if name == "policy_obs" else original(name, **args)
            s = session(worker, policy)
            s.run_policy = mock.Mock(return_value=None)
            s.start("B", "default")
            self.assertEqual(s._strategy["strategy"], "A")

    def test_driver_routes_before_model_with_mocked_kernel_no_process(self):
        for task, own in (("unlisted", False), ("general_pickup_random", True)):
            config = _config()
            config.task = task
            config.runtime.update(task_routing=ROUTING, policies={"wall-oss": {}},
                                  policy_action_mode="ee_passthrough", verifier=False)
            kernel = mock.Mock()
            kernel.run.return_value = {"status": "ok" if own else "episode_ended", "strategy": {
                "strategy": "B" if own else "A", "policy": None if own else "wall-oss"}}
            chat = mock.Mock(wraps=ScriptedChat(["<finish>done</finish>"]))
            with tempfile.TemporaryDirectory() as temp, mock.patch("harness_policy.runtime.driver.Kernel", return_value=kernel) as factory:
                report = run_program_episode(config, 0, Path(temp) / "run", chat=chat, environment=FakeEnvironment())
                events = [json.loads(line) for line in (Path(temp) / "run/events.jsonl").read_text().splitlines()]
            self.assertEqual(factory.call_args.args[1]["fixed_task_route"], "own" if own else "wall-oss")
            self.assertEqual(events[1]["event"], "task_route")
            self.assertIn("start('B'" if own else "start('A'", kernel.run.call_args_list[0].args[0])
            if own:
                self.assertEqual(report["model_calls"], 1)
                self.assertEqual(report["status"], "finished")
            else:
                chat.assert_not_called()
                chat.complete.assert_not_called()
                self.assertEqual(report["status"], "episode_ended")
                self.assertEqual(report["model_calls"], 0)

    def test_driver_does_not_retry_or_call_model_when_fixed_policy_errors(self):
        config = _config()
        config.task = "unlisted"
        config.runtime.update(task_routing=ROUTING, policies={"wall-oss": {}}, verifier=False)
        kernel = mock.Mock()
        kernel.run.return_value = {"status": "error", "error": "possibly executed motion"}
        chat = mock.Mock()
        with tempfile.TemporaryDirectory() as temp, mock.patch("harness_policy.runtime.driver.Kernel", return_value=kernel):
            report = run_program_episode(config, 0, Path(temp) / "run", chat=chat, environment=FakeEnvironment())
        self.assertEqual(report["status"], "error")
        self.assertIn("possibly executed motion", report["error"])
        self.assertEqual(kernel.run.call_count, 1)
        chat.complete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
