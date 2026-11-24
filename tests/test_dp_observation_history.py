"""Exercise DP's real history and action assembly without CUDA or weights."""
from __future__ import annotations

import ast
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class Tensor:
    """Only the device-transfer API used by DPRunner; values stay on CPU."""
    def __init__(self, array):
        self.array = np.asarray(array)

    def to(self, **kwargs):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


def make_runner(n_obs_steps=3):
    path = ROOT / "policy/DP/diffusion_policy/env_runner/dp_runner.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    helper_path = ROOT / "policy/DP/diffusion_policy/common/pytorch_util.py"
    helper = next(node for node in ast.parse(helper_path.read_text()).body
                  if isinstance(node, ast.FunctionDef) and node.name == "dict_apply")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module([future, helper, cls], type_ignores=[]))
    torch = SimpleNamespace(
        Tensor=Tensor, from_numpy=Tensor, no_grad=nullcontext,
        stack=lambda items, dim: Tensor(np.stack([item.array for item in items], axis=dim)),
    )
    namespace = dict(np=np, deque=deque, torch=torch)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["DPRunner"](n_obs_steps=n_obs_steps, n_action_steps=2)


def observation(marker):
    return {name: np.array([marker], dtype=np.float32)
            for name in ("head_cam", "left_cam", "right_cam", "agent_pos")}


class DPObservationHistoryTests(unittest.TestCase):
    def test_sparse_ids_and_more_than_100_environments_keep_separate_history(self):
        runner = make_runner()
        ids = [*range(128), 10001]
        runner.update_obs([observation(i) for i in ids], ids)
        selected = [10001, 127, 100, 0]
        for env_id, obs in zip(selected, runner.get_n_steps_obs(selected)):
            for values in obs.values():
                np.testing.assert_array_equal(values[:, 0], [env_id] * 3)
        runner.update_obs([observation(999)], [100])
        np.testing.assert_array_equal(runner.get_n_steps_obs([100])[0]["agent_pos"][:, 0],
                                      [100, 100, 999])
        np.testing.assert_array_equal(runner.get_n_steps_obs([127])[0]["agent_pos"][:, 0],
                                      [127, 127, 127])

    def test_padding_truncation_and_normal_action_assembly_are_unchanged(self):
        runner = make_runner()
        seen = []

        def predict(inputs):
            seen.append(inputs)
            # A deterministic weight substitute uses the actual stacked input.
            return {"action": Tensor(inputs["agent_pos"].array + 0.25)}

        policy = SimpleNamespace(device="cpu", predict_action=predict)
        history = {0: [], 1: []}
        for step in range(6):
            runner.update_obs([observation(step), observation(10 + step)], [0, 1])
            for env_id in history:
                history[env_id].append(10 * env_id + step)
            actions = runner.get_action(policy, [1, 0])
            for row, env_id in enumerate([1, 0]):
                expected = history[env_id][-3:]
                expected = [expected[0]] * (3 - len(expected)) + expected
                for value in seen[-1].values():
                    np.testing.assert_array_equal(value.array[row, :, 0], expected)
                np.testing.assert_array_equal(actions[row, :, 0], np.array(expected[:2]) + 0.25)
        self.assertLessEqual(len(runner.obs_list[0]), 4)

    def test_reset_removes_all_ids_and_unknown_reads_do_not_create_history(self):
        runner = make_runner()
        runner.update_obs([observation(1), observation(2)], [0, 127])
        runner.reset_obs()
        for env_id in [0, 127, 10001]:
            with self.assertRaisesRegex(ValueError, "No observation"):
                runner.get_n_steps_obs([env_id])
        self.assertEqual(len(runner.obs_list), 0)
        runner.update_obs([observation(3)], [127])
        np.testing.assert_array_equal(runner.get_n_steps_obs([127])[0]["agent_pos"][:, 0], [3, 3, 3])


if __name__ == "__main__":
    unittest.main()
