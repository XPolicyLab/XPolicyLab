"""Exercise the real adapter with synthetic predictions, without loading weights."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from XPolicyLab.policy.XBrain_v1.model import Model


POLICY_ROOT = Path(__file__).resolve().parents[1]


class FakePipeline:
    device = torch.device("cpu")

    def __init__(self):
        self.prediction = torch.arange(50 * 14, dtype=torch.float32).reshape(50, 14) / 100
        self.prediction[:, 6] = 0.35
        self.prediction[:, 13] = 0.30
        self.calls = 0

    def __call__(self, images, prompt, state, **kwargs):
        self.calls += 1
        return self.prediction


class ActionOutputTests(unittest.TestCase):
    def make_model(self, robot="piper_x", horizon=None):
        config = yaml.safe_load((POLICY_ROOT / "deploy.yml").read_text())
        config["env_cfg_type"] = robot
        if horizon is not None:
            config["action_horizon"] = horizon
        pipeline = FakePipeline()
        with patch.object(Model, "_load_pipeline", return_value=pipeline), \
             patch("XPolicyLab.policy.XBrain_v1.model.get_robot_action_dim_info",
                   return_value={"arm_dim": [6, 6], "ee_dim": [1, 1]}):
            model = Model(config)
        return model, pipeline

    @staticmethod
    def observe(model, prompt):
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        model.update_obs(dict(
            vision={name: {"color": image} for name in
                    ("cam_head", "cam_left_wrist", "cam_right_wrist")},
            state=np.zeros(14, dtype=np.float32), instruction=prompt,
        ))

    @staticmethod
    def rows(actions):
        return np.array([np.concatenate([
            action["left_arm_joint_state"], action["left_ee_joint_state"],
            action["right_arm_joint_state"], action["right_ee_joint_state"],
        ]) for action in actions])

    def test_first_20_rows_and_task_grippers_are_sent_once(self):
        model, pipeline = self.make_model()
        prompt = next(rule.prompt for rule in model._gripper_thresholds.values()
                      if rule.task == "sweep_blocks")
        self.observe(model, prompt)
        original = pipeline.prediction.numpy().copy()
        expected = original[:20].copy()
        expected[:, 6] = 0.455
        expected[:, 13] = 0.39
        first = self.rows(model.get_action())
        second = self.rows(model.get_action())
        self.assertEqual(first.shape, (20, 14))
        np.testing.assert_allclose(first, expected)
        np.testing.assert_allclose(second, expected)
        np.testing.assert_array_equal(pipeline.prediction.numpy(), original)

    def test_all_18_task_rules_apply_independently(self):
        for robot in ("piper_x", "piper", "arx_x5"):
            model, pipeline = self.make_model(robot)
            for rule in model._gripper_thresholds.values():
                with self.subTest(robot=robot, task=rule.task):
                    pipeline.prediction[:, 6] = rule.left / 2
                    pipeline.prediction[:, 13] = rule.right * 2
                    self.observe(model, rule.prompt)
                    result = self.rows(model.get_action())
                    self.assertEqual(result.shape, (20, 14))
                    np.testing.assert_array_equal(result[:, 6], np.zeros(20))
                    # Predictions and postprocessing use float32 throughout.
                    np.testing.assert_allclose(result[:, 13], rule.right * 2.6, rtol=1e-6)

    def test_unknown_prompt_preserves_both_grippers(self):
        model, pipeline = self.make_model()
        self.observe(model, "An unconfigured task prompt.")
        np.testing.assert_array_equal(self.rows(model.get_action()), pipeline.prediction.numpy()[:20])

    def test_new_observation_requests_a_fresh_plan(self):
        model, pipeline = self.make_model()
        self.observe(model, "An unconfigured task prompt.")
        first = self.rows(model.get_action())
        pipeline.prediction += 1.0
        self.observe(model, "An unconfigured task prompt.")
        second = self.rows(model.get_action())
        self.assertEqual(pipeline.calls, 2)
        np.testing.assert_allclose(second, first + 1)

    def test_20_is_the_default_and_maximum(self):
        with patch.object(Model, "_load_pipeline", return_value=FakePipeline()), \
             patch("XPolicyLab.policy.XBrain_v1.model.get_robot_action_dim_info",
                   return_value={"arm_dim": [6, 6], "ee_dim": [1, 1]}):
            model = Model(dict(env_cfg_type="piper_x", action_type="joint"))
        self.assertEqual(model.action_horizon, 20)
        for horizon in (0, 21, 30):
            with self.subTest(horizon=horizon), self.assertRaisesRegex(ValueError, r"\[1, 20\]"):
                self.make_model(horizon=horizon)


if __name__ == "__main__":
    unittest.main()
