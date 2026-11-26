"""Mem_0 prediction validation without importing its GPU models or checkpoints.

Run the adapter and upstream smoothing method bodies, including real layout,
normalization and robot-action conversion. Only neural inference and media I/O
are replaced; malformed predictions must never enter the smoothing history.
"""

import ast
import copy
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import unpack_robot_state


POLICY = Path(__file__).resolve().parents[1] / "policy" / "Mem_0"


def load_methods(path, class_name, names, namespace):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    functions = [node for node in cls.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise AssertionError("A tested Mem_0 method is missing")
    for function in functions:
        function.decorator_list = []  # torch.inference_mode does not change CPU routing.
    exec(compile(ast.Module(functions, type_ignores=[]), str(path), "exec"), namespace)
    return {function.name: namespace[function.name] for function in functions}


class Mem0ActionValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helpers = POLICY / "Mem_0" / "scripts" / "tools_for_deploy"
        namespace = dict(np=np, unpack_robot_state=unpack_robot_state,
                         ARM_NORM_DIMS=14, TMP_VISUAL="unused-test-media", cprint=lambda *args: None)
        namespace.update(runpy.run_path(str(helpers / "layout_utils.py")))
        namespace.update(runpy.run_path(str(helpers / "normlization.py")))
        cls.Agent = type("SmoothingAgent", (), load_methods(
            POLICY / "Mem_0" / "source" / "agent" / "memorymatters_agent.py",
            "MemoryMattersAgent", {"accumulate_actions_chunk", "get_smoothed_actions"}, namespace,
        ))
        cls.Model = type("Mem0Adapter", (ModelTemplate,), load_methods(
            POLICY / "model.py", "Model", {
                "_validated_action_prediction", "_start_macro_chunk", "_postprocess_chunk",
                "_env16_to_packed14", "_denormalize_action", "_advance_macro_iter",
                "_handle_subtask_transition", "step", "get_action",
            }, namespace,
        ))
        cls.deploy = runpy.run_path(str(POLICY / "deploy.py"))

    def make_model(self, result):
        agent = self.Agent()
        agent.action_horizon = agent.action_strip = 3
        agent.executor = SimpleNamespace(action_model=SimpleNamespace(action_dim=16))
        agent.iter, agent.stage, agent.action_count = 4, 1, 1
        agent.end_signal_count, agent.threshold = 0, 2
        agent._time_action_history = {4: [np.arange(16, dtype=np.float32) / 10]}
        agent.get_action = Mock(return_value=result)
        agent.update_obs = Mock(return_value=0)
        model = self.Model()
        model.agent = agent
        model.action_type = "joint"
        model.robot_action_dim_info = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
        model.norm_way = "meanstd"
        model._stats = {
            "action_mean": np.arange(16, dtype=np.float32) / 5,
            "action_std": np.full(16, 0.25, dtype=np.float32),
        }
        model._chunk_queue = []
        model._macro_started = model._need_obs_update = False
        model._steps_in_chunk = 0
        model.task_type = "M1"
        model.encode_obs = lambda obs: {"_rgb": np.zeros((2, 2, 3), np.uint8)}
        model._write_ffmpeg = Mock()
        return model

    def assert_history_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for time, rows in expected.items():
            self.assertEqual(len(actual[time]), len(rows))
            for actual_row, expected_row in zip(actual[time], rows):
                np.testing.assert_array_equal(actual_row, expected_row)

    def assert_chunks_equal(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for actual_step, expected_step in zip(actual, expected):
            self.assertEqual(set(actual_step), set(expected_step))
            for key in expected_step:
                np.testing.assert_array_equal(actual_step[key], expected_step[key])

    def test_valid_predictions_preserve_upstream_smoothing_and_postprocessing(self):
        for batched in (False, True):
            for horizon in (3, 5):
                for dtype in (np.float16, np.float32, np.float64):
                    with self.subTest(batched=batched, horizon=horizon, dtype=dtype):
                        prediction = (np.arange(horizon * 16).reshape(horizon, 16) / 100).astype(dtype)
                        wire_prediction = prediction[None] if batched else prediction
                        model = self.make_model({"normalized_actions": wire_prediction})
                        baseline = self.make_model(None)
                        # The original implementation passes predictions directly
                        # to these same upstream methods before postprocessing.
                        baseline.agent.accumulate_actions_chunk(wire_prediction)
                        smoothed = baseline.agent.get_smoothed_actions(4, 3)
                        expected = baseline._postprocess_chunk(smoothed)
                        model._start_macro_chunk({})
                        self.assert_chunks_equal(model._chunk_queue, expected)
                        self.assert_history_equal(model.agent._time_action_history,
                                                  baseline.agent._time_action_history)
                        self.assertEqual(model.agent.iter, 4)
                        self.assertEqual(model._steps_in_chunk, 3)
                        # The compatibility RPC still returns the full prediction.
                        self.assert_chunks_equal(model.get_action(), baseline._postprocess_chunk(prediction))

    def test_malformed_predictions_do_not_modify_action_history_or_queue(self):
        good = np.zeros((1, 3, 16), np.float32)
        invalid = [None, {}, [], {"actions": good}]
        arrays = [None, [], np.zeros((1, 0, 16)), np.zeros((0, 3, 16)),
                  np.zeros((2, 3, 16)), np.zeros((3, 0)), np.zeros((1, 2, 16)),
                  np.zeros((3, 15)), np.zeros((3, 17)), np.zeros(16),
                  np.zeros((1, 1, 3, 16)), good.astype(bool), good.astype(complex),
                  good.astype(str), good.astype(object), [[1], [1, 2]],
                  np.full((1, 3, 16), 1e300)]
        for value in (np.nan, np.inf, -np.inf):
            array = good.copy()
            array[0, -1, -1] = value
            arrays.append(array)
        invalid.extend({"normalized_actions": value} for value in arrays)
        for result in invalid:
            with self.subTest(result_type=type(result).__name__, prediction=str(result)[:70]):
                model = self.make_model(result)
                history = copy.deepcopy(model.agent._time_action_history)
                queue = model._chunk_queue
                with self.assertRaises((ValueError, TypeError)):
                    model._start_macro_chunk({})
                self.assertIs(model._chunk_queue, queue)
                self.assertEqual(queue, [])
                self.assertFalse(model._macro_started)
                self.assertFalse(model._need_obs_update)
                self.assertEqual(model._steps_in_chunk, 0)
                self.assertEqual((model.agent.iter, model.agent.stage, model.agent.action_count), (4, 1, 1))
                self.assert_history_equal(model.agent._time_action_history, history)
                with self.assertRaises((ValueError, TypeError, RuntimeError)):
                    model.get_action()

    def test_execution_horizon_and_model_dimension_are_checked_before_accumulation(self):
        for strip, horizon, dimension in ((0, 3, 16), (-1, 3, 16), (4, 3, 16), (3, 3, 17)):
            with self.subTest(strip=strip, horizon=horizon, dimension=dimension):
                model = self.make_model({"normalized_actions": np.zeros((5, 16))})
                model.agent.action_strip, model.agent.action_horizon = strip, horizon
                model.agent.executor.action_model.action_dim = dimension
                history = copy.deepcopy(model.agent._time_action_history)
                with self.assertRaises(ValueError):
                    model._start_macro_chunk({})
                self.assert_history_equal(model.agent._time_action_history, history)
                self.assertEqual(model.agent.iter, 4)

    def test_empty_prediction_cannot_turn_into_executed_zero_actions(self):
        model = self.make_model({"normalized_actions": np.zeros((1, 0, 16))})
        history = copy.deepcopy(model.agent._time_action_history)
        env = SimpleNamespace(get_obs=lambda: {}, is_episode_end=lambda: False, take_action=Mock())

        def call(func_name, obs=None):
            if func_name == "step":
                return model.step(obs)

        with self.assertRaises(ValueError):
            self.deploy["eval_one_episode"](env, SimpleNamespace(call=call))
        env.take_action.assert_not_called()
        model.agent.get_action.assert_called_once()
        self.assertEqual(model.agent.iter, 4)
        self.assert_history_equal(model.agent._time_action_history, history)

    def test_subtask_transition_still_returns_none_and_client_keeps_waiting(self):
        model = self.make_model({"normalized_actions": np.ones((1, 3, 16))})
        model.task_type = "Mn"
        model._need_obs_update = model._macro_started = True
        model._steps_in_chunk = 3
        model.agent.end_signal_count = model.agent.threshold
        model.agent.update_obs.return_value = 1
        model._handle_subtask_transition = Mock()
        self.assertIsNone(model.step({}))
        model._handle_subtask_transition.assert_called_once()
        model.agent.get_action.assert_not_called()
        self.assertEqual(model.agent.iter, 7)  # Only the previously executed chunk advances.

        action = {"left_arm_joint_state": np.zeros(6)}
        replies = iter((None, action))
        env = SimpleNamespace(get_obs=lambda: {}, take_action=Mock())
        env.is_episode_end = lambda: env.take_action.call_count == 1
        client = SimpleNamespace(call=lambda func_name, obs=None: next(replies) if func_name == "step" else None)
        self.deploy["eval_one_episode"](env, client)
        env.take_action.assert_called_once_with(action)

    def test_two_stages_keep_action_order_smoothing_and_iteration(self):
        first = np.arange(48, dtype=np.float32).reshape(1, 3, 16) / 100
        second = first + 1
        model = self.make_model(None)
        model.task_type = "Mn"
        model.agent.get_action.side_effect = [
            {"normalized_actions": first}, {"normalized_actions": second},
        ]
        model.agent.executor.memory_bank = SimpleNamespace(reset=Mock())
        model._save_stage_image = Mock()
        updates = []

        def update_obs(payload):
            updates.append(payload)
            if len(updates) == 3:
                model.agent.end_signal_count = model.agent.threshold
                return 1
            return 0

        def update_high_observation():
            model.agent.stage += 1

        model.agent.update_obs.side_effect = update_obs
        model.agent.update_high_observation = Mock(side_effect=update_high_observation)
        baseline = self.make_model(None)
        baseline.agent.accumulate_actions_chunk(first)
        expected_first = baseline._postprocess_chunk(baseline.agent.get_smoothed_actions(4, 3))
        baseline.agent.iter = 7
        baseline.agent.accumulate_actions_chunk(second)
        expected_second = baseline._postprocess_chunk(baseline.agent.get_smoothed_actions(7, 3))

        self.assert_chunks_equal([model.step({}) for _ in range(3)], expected_first)
        self.assertIsNone(model.step({}))
        self.assertEqual((model.agent.stage, model.agent.iter), (2, 7))
        self.assertEqual(model.agent.end_signal_count, 0)
        model.agent.executor.memory_bank.reset.assert_called_once()
        model.agent.update_high_observation.assert_called_once()
        self.assert_chunks_equal([model.step({}) for _ in range(3)], expected_second)
        self.assertEqual(model.agent.iter, 7)
        self.assertEqual(model.agent.get_action.call_count, 2)
        self.assert_history_equal(model.agent._time_action_history,
                                  baseline.agent._time_action_history)


if __name__ == "__main__":
    unittest.main()
