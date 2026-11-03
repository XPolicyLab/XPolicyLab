"""CPU regressions for the execution guard; no Isaac Sim imports or startup."""

import ast
from copy import deepcopy
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

import importlib.util
_guard_path=Path(__file__).resolve().parents[1]/"sdk_sources/local/src/eval_client/ik_guard.py"
_spec=importlib.util.spec_from_file_location("packaged_ik_guard",_guard_path)
_guard=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_guard)
IKGuard=_guard.IKGuard


class IKGuardTests(unittest.TestCase):
    def test_accepts_small_motion_and_exact_threshold(self):
        current = np.array([0.1, -0.3, 0.0])
        target = np.array([0.1, -0.2, 0.5])
        safe, reason = IKGuard().filter_result(current, {"status": "Success", "joint_value": target})
        self.assertIsNone(reason)
        np.testing.assert_array_equal(safe, target)

    def test_rejects_whole_arm_for_jump_and_raw_full_turn(self):
        current = np.array([0.1, -0.3, 0.0])
        for target in ([0.2, -0.3, 0.5001], [0.2, -0.3, 2 * np.pi], [1e300, -0.3, 0]):
            with self.subTest(target=target):
                safe, reason = IKGuard().filter_result(current, {"status": "Success", "joint_value": target})
                self.assertIn("delta=", reason)
                np.testing.assert_array_equal(safe, current)
                self.assertFalse(np.shares_memory(safe, current))

    def test_invalid_results_hold_without_mutating_solver_output(self):
        current = np.array([0.1, 0.2, 0.3])
        cases = [
            None, {}, {"status": "Fail"}, {"status": "Success"},
            *({"status": "Success", "joint_value": target} for target in (
                [np.nan, 0, 0], [np.inf, 0, 0], [-np.inf, 0, 0],
                [0, 0], [[0, 0, 0]], ["invalid", 0, 0], None,
            )),
        ]
        for result in cases:
            with self.subTest(result=result):
                before = repr(result)
                safe, reason = IKGuard().filter_result(current, result)
                self.assertIsNotNone(reason)
                np.testing.assert_array_equal(safe, current)
                self.assertEqual(repr(result), before)

    def test_invalid_current_cannot_produce_a_hold_command(self):
        for current in (None, [], [[0, 0]], [np.nan], [np.inf]):
            with self.subTest(current=current), self.assertRaisesRegex(ValueError, "Cannot hold arm"):
                IKGuard().filter_result(current, {"status": "Fail"})

    def test_threshold_must_be_finite_and_positive(self):
        for threshold in (0, -1, np.nan, np.inf):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                IKGuard(threshold)
        safe, reason = IKGuard(0.2).filter_result([0], {"status": "Success", "joint_value": [0.3]})
        np.testing.assert_array_equal(safe, [0])
        self.assertIsNotNone(reason)


def execution_class():
    """Load the actual execution/interpolation methods without simulator imports."""
    path = Path(__file__).resolve().parents[1] / "sdk_sources/local/src/eval_client/eval_env.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "EvalEnv")
    methods = {"take_action", "take_action_batch", "process_control_info"}
    cls.bases = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    namespace = {
        "np": np, "deepcopy": deepcopy, "os": SimpleNamespace(environ={}),
        "logger": logging.getLogger("test_ik_guard"), "UnStableError": RuntimeError,
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["EvalEnv"]


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.env = execution_class()()
        self.env.num_envs = 3
        self.env.ik_guard = IKGuard()
        self.env.ik_rejections = [{}, {}, {}]
        self.env.physx_monitor_enabled = False
        self.env.take_action_cnt = [0, 0, 0]
        self.env.end_flag = [False, False, False]
        self.env.step_lim = 100
        self.env.obs_manager = SimpleNamespace(collect_interval=5)
        self.env.validate_action_dict = lambda action: None
        self.env.get_action_type = lambda action: "ee"
        self.env.have_empty = lambda ids: True
        self.env.is_episode_end = lambda: False
        self.env.reward_manager = SimpleNamespace(step=lambda **kwargs: None)
        self.measured = {
            (idx, side): np.array([0.1 * idx, -0.2])
            for idx in range(3) for side in ("left_arm", "right_arm")
        }
        self.results = {
            key: {"status": "Success", "joint_value": current + 0.1}
            for key, current in self.measured.items()
        }
        robots = [SimpleNamespace(
            type="target", arm_name=f"{side}_arm", gripper_name=f"{side}_ee", ee_type="gripper",
            gripper_move={"sign": 1, "mimic": [0, -1, 0]}, gripper_scale=[0, 0.04],
        ) for side in ("left", "right")]
        self.pushed = []
        self.env.robot_manager = SimpleNamespace(
            robot_list=robots,
            process_name=lambda name: name + "_joint_state",
            get_joint=lambda robot, env_idx_list: {
                idx: self.measured[idx, robot.arm_name] for idx in env_idx_list
            },
            get_end_effector_real_val=lambda robot, env_idx_list: {
                idx: [0.02, -0.02] for idx in env_idx_list
            },
            solve_ik=lambda target_pose, env_idx, robot: self.results[env_idx, robot.arm_name],
            control_manager=SimpleNamespace(push=lambda ids, seqs: self.pushed.append((ids, seqs))),
        )
        self.action = {
            "left_ee_pose": [0, 0, 0, 1, 0, 0, 0], "right_ee_pose": [0, 0, 0, 1, 0, 0, 0],
            "left_ee_joint_state": [0.4], "right_ee_joint_state": [0.6],
        }

    def test_batch_rejection_holds_arm_through_every_interpolation_step(self):
        # Noncontiguous env ids catch accidental use of the batch-list position.
        self.results[2, "left_arm"]["joint_value"] = [2.0, -0.1]
        with self.assertLogs("test_ik_guard", level="WARNING") as captured:
            self.env.take_action_batch([self.action, self.action], [2, 0])
        ids, sequences = self.pushed[0]
        self.assertEqual(ids, [2, 0])
        self.assertEqual(len(sequences[0]), 5)
        self.assertIn("env=2 arm=left_arm", captured.output[0])
        self.assertEqual(self.env.ik_rejections, [{}, {}, {"left_arm": 1}])
        for control in sequences[0]:
            np.testing.assert_allclose(
                control["left_arm_joint_state"]["position"], self.measured[2, "left_arm"], rtol=0, atol=1e-15,
            )
        np.testing.assert_allclose(sequences[0][-1]["right_arm_joint_state"]["position"], [0.3, -0.1])
        np.testing.assert_allclose(sequences[1][-1]["left_arm_joint_state"]["position"], [0.1, -0.1])
        np.testing.assert_allclose(sequences[0][-1]["left_ee_joint_state"]["position"], [0.016, -0.016])
        self.assertEqual(self.env.take_action_cnt, [1, 0, 1])

    def test_single_env_failure_and_nonfinite_result_explicitly_hold(self):
        self.results[0, "left_arm"] = {"status": "Fail"}
        self.results[0, "right_arm"]["joint_value"] = [np.nan, 0]
        with self.assertLogs("test_ik_guard", level="WARNING"):
            self.env.take_action(self.action)
        for side in ("left_arm", "right_arm"):
            for control in self.pushed[0][1][0]:
                np.testing.assert_allclose(
                    control[f"{side}_joint_state"]["position"], self.measured[0, side], rtol=0, atol=1e-15,
                )

    def test_comparison_uses_live_measurement_each_action(self):
        self.env.take_action(self.action)
        self.measured[0, "left_arm"] = np.array([1.0, -0.2])
        with self.assertLogs("test_ik_guard", level="WARNING"):
            self.env.take_action(self.action)
        np.testing.assert_array_equal(self.pushed[1][1][0][-1]["left_arm_joint_state"]["position"], [1.0, -0.2])

    def test_invalid_measurement_aborts_before_any_batch_control_push(self):
        self.measured[0, "right_arm"][0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "Cannot hold arm"):
            self.env.take_action_batch([self.action, self.action], [2, 0])
        self.assertFalse(self.pushed)


if __name__ == "__main__":
    unittest.main()
