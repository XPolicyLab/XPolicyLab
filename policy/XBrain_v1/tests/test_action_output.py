"""Exercise the real adapter with synthetic predictions, without loading weights."""

from pathlib import Path
import asyncio
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from XPolicyLab.policy.XBrain_v1.model import Model


POLICY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HORIZONS = {
    "piper_x": {"cap_pen": 30, "classify_objects": 20, "disassemble_LEGO": 30,
                "hang_mugs": 30, "pack_objects_into_backpack": 20, "sweep_blocks": 30},
    "piper": {"fill_pen_holder": 20, "insert_charger": 20, "put_objects_into_basket": 20,
              "stack_and_cover_blocks": 40, "stack_bowls": 20, "stand_up_bottles": 20},
    "arx_x5": {"cover_blocks": 20, "insert_tubes": 20, "make_bread": 30,
               "make_food": 20, "pack_and_pour_fruit": 20, "store_in_safe": 20},
}


class FakePipeline:
    device = torch.device("cpu")

    def __init__(self):
        self.prediction = torch.arange(50 * 14, dtype=torch.float32).reshape(50, 14) / 100
        self.prediction[:, 6] = 0.35
        self.prediction[:, 13] = 0.30
        self.calls = 0
        self.last_state = None

    def __call__(self, images, prompt, state, **kwargs):
        self.calls += 1
        self.last_state = state.detach().cpu().numpy().copy()
        return self.prediction

    def reset_observation_memory(self):
        self.last_state = None


class ActionOutputTests(unittest.TestCase):
    def make_model(self, robot="piper_x", horizon=None, output_action_type=None):
        config = yaml.safe_load((POLICY_ROOT / "deploy.yml").read_text())
        config["env_cfg_type"] = robot
        if output_action_type is not None:
            config["output_action_type"] = output_action_type
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

    def test_first_30_sweep_rows_and_task_grippers_are_sent_once(self):
        model, pipeline = self.make_model()
        prompt = next(rule.prompt for rule in model._gripper_thresholds.values()
                      if rule.task == "sweep_blocks")
        self.observe(model, prompt)
        original = pipeline.prediction.numpy().copy()
        expected = original[:30].copy()
        expected[:, 6] = 0.455
        expected[:, 13] = 0.39
        first = self.rows(model.get_action())
        second = self.rows(model.get_action())
        self.assertEqual(first.shape, (30, 14))
        np.testing.assert_allclose(first, expected)
        np.testing.assert_allclose(second, expected)
        np.testing.assert_array_equal(pipeline.prediction.numpy(), original)

    def test_all_18_task_rules_apply_independently(self):
        for robot in ("piper_x", "piper", "arx_x5"):
            model, pipeline = self.make_model(robot)
            self.assertEqual({rule.task for rule in model._gripper_thresholds.values()},
                             set(EXPECTED_HORIZONS[robot]))
            for rule in model._gripper_thresholds.values():
                with self.subTest(robot=robot, task=rule.task):
                    horizon = EXPECTED_HORIZONS[robot][rule.task]
                    pipeline.prediction[:, 6] = rule.left / 2
                    pipeline.prediction[:, 13] = rule.right * 2
                    self.observe(model, rule.prompt)
                    result = self.rows(model.get_action())
                    self.assertEqual(result.shape, (horizon, 14))
                    np.testing.assert_array_equal(result[:, 6], np.zeros(horizon))
                    arm_columns = list(range(6)) + list(range(7, 13))
                    np.testing.assert_array_equal(result[:, arm_columns],
                                                  pipeline.prediction.numpy()[:horizon, arm_columns])
                    # Predictions and postprocessing use float32 throughout.
                    np.testing.assert_allclose(result[:, 13], rule.right * 2 * rule.right_scale, rtol=1e-6)

    def test_unknown_prompt_preserves_both_grippers(self):
        model, pipeline = self.make_model()
        self.observe(model, "An unconfigured task prompt.")
        np.testing.assert_array_equal(self.rows(model.get_action()), pipeline.prediction.numpy()[:20])

    def test_task_horizons_follow_normalized_prompt_without_sticking(self):
        for robot in EXPECTED_HORIZONS:
            model, _ = self.make_model(robot)
            rules = list(model._gripper_thresholds.values())
            for rule in rules + list(reversed(rules)):
                for field in ("instruction", "task_instruction"):
                    with self.subTest(robot=robot, task=rule.task, field=field):
                        prompt = "  " + rule.prompt.upper().replace(" ", "  ").rstrip(".") + "  "
                        self.observe(model, prompt)
                        if field == "task_instruction":
                            obs = dict(model._obs)
                            obs[field] = obs.pop("instruction")
                            model.update_obs(obs)
                        self.assertEqual(len(model.get_action()), EXPECTED_HORIZONS[robot][rule.task])
                        self.assertEqual(model.action_horizon, 20)
                        self.observe(model, "An unconfigured task prompt.")
                        self.assertEqual(len(model.get_action()), 20)

    def test_task_horizon_overrides_only_the_matched_tasks_fallback(self):
        model, _ = self.make_model("piper", horizon=10)
        prompts = {rule.task: rule.prompt for rule in model._gripper_thresholds.values()}
        for prompt, horizon in ((prompts["stack_and_cover_blocks"], 40),
                                (prompts["stack_bowls"], 10),
                                ("An unconfigured task prompt.", 10)):
            self.observe(model, prompt)
            self.assertEqual(len(model.get_action()), horizon)
        self.assertEqual(model.action_horizon, 10)

    def test_new_observation_requests_a_fresh_plan(self):
        model, pipeline = self.make_model()
        self.observe(model, "An unconfigured task prompt.")
        first = self.rows(model.get_action())
        pipeline.prediction += 1.0
        self.observe(model, "An unconfigured task prompt.")
        second = self.rows(model.get_action())
        self.assertEqual(pipeline.calls, 2)
        np.testing.assert_allclose(second, first + 1)

    def test_20_is_the_fallback_default_and_maximum(self):
        with patch.object(Model, "_load_pipeline", return_value=FakePipeline()), \
             patch("XPolicyLab.policy.XBrain_v1.model.get_robot_action_dim_info",
                   return_value={"arm_dim": [6, 6], "ee_dim": [1, 1]}):
            model = Model(dict(env_cfg_type="piper_x", action_type="joint"))
        self.assertEqual(model.action_horizon, 20)
        self.assertEqual(model.output_action_type, "joint")
        for horizon in (0, 21, 30):
            with self.subTest(horizon=horizon), self.assertRaisesRegex(ValueError, r"\[1, 20\]"):
                self.make_model(horizon=horizon)

    def test_piperx_endpose_output_uses_official_action_keys(self):
        model, pipeline = self.make_model()
        model.output_action_type = "ee"
        self.observe(model, "An unconfigured task prompt.")
        actions = model.get_action()
        self.assertEqual(len(actions), 20)
        self.assertEqual(
            set(actions[0]),
            {"left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state"},
        )
        self.assertEqual(np.asarray(actions[0]["left_ee_pose"]).shape, (7,))
        self.assertEqual(np.asarray(actions[0]["right_ee_pose"]).shape, (7,))
        np.testing.assert_allclose(actions[0]["left_ee_joint_state"], [0.35])
        np.testing.assert_allclose(actions[0]["right_ee_joint_state"], [0.30])
        for action in actions:
            np.testing.assert_allclose(np.linalg.norm(action["left_ee_pose"][3:]), 1.0, atol=1e-5)
            np.testing.assert_allclose(np.linalg.norm(action["right_ee_pose"][3:]), 1.0, atol=1e-5)

    def test_piperx_backpack_skips_fk_and_z_processing(self):
        prompt = "Place all the objects on the table into the backpack."
        for output_type in ("joint", "auto", "ee"):
            model, pipeline = self.make_model(output_action_type=output_type)
            for field in ("instruction", "task_instruction"):
                with self.subTest(output_type=output_type, prompt=prompt, field=field):
                    self.observe(model, "  " + prompt.upper().replace(" ", "  ").rstrip(".") + "  ")
                    if field == "task_instruction":
                        observation = dict(model._obs)
                        observation["task_instruction"] = observation.pop("instruction")
                        model.update_obs(observation)
                    expected = pipeline.prediction.numpy()[:20].copy()
                    expected[:, 6] = 0.385
                    expected[:, 13] = 0.33
                    with patch("XPolicyLab.policy.XBrain_v1.model.joints14_to_endpose16",
                               side_effect=AssertionError("Backpack must not run FK")), \
                         patch("XPolicyLab.policy.XBrain_v1.model.adjust_piperx_endpose_z",
                               side_effect=AssertionError("Backpack must not adjust Z")):
                        actions = model.get_action()
                    self.assertEqual(set(actions[0]), {
                        "left_arm_joint_state", "left_ee_joint_state",
                        "right_arm_joint_state", "right_ee_joint_state",
                    })
                    self.assertEqual(len(actions), 20)
                    np.testing.assert_allclose(self.rows(actions), expected)
                    self.assertEqual(model.output_action_type, "ee" if output_type == "ee" else "joint")

    def test_explicit_endpose_keeps_backpack_joint_exception(self):
        model, _ = self.make_model(output_action_type="ee")
        for rule in model._gripper_thresholds.values():
            with self.subTest(task=rule.task):
                self.observe(model, rule.prompt)
                actions = model.get_action()
                expected_arm_keys = ({"left_arm_joint_state", "right_arm_joint_state"}
                                     if rule.task == "pack_objects_into_backpack"
                                     else {"left_ee_pose", "right_ee_pose"})
                self.assertEqual(len(actions), EXPECTED_HORIZONS["piper_x"][rule.task])
                for action in actions:
                    self.assertEqual(set(action), expected_arm_keys | {
                        "left_ee_joint_state", "right_ee_joint_state"})

    def test_piperx_output_switches_per_request_without_reset(self):
        model, _ = self.make_model(output_action_type="ee")
        for prompt, key, horizon in (
            ("Place all the objects on the table into the backpack.", "left_arm_joint_state", 20),
            ("Hang the mugs on the mug rack.", "left_ee_pose", 30),
            ("Place all the fruits into the blue bowl, then pour the fruits from the blue bowl into the large white bowl.", "left_ee_pose", 20),
            ("An unconfigured task prompt.", "left_ee_pose", 20),
        ):
            self.observe(model, prompt)
            actions = model.get_action()
            self.assertEqual(len(actions), horizon)
            self.assertIn(key, actions[0])

    def test_batch_uses_each_observations_prompt(self):
        model, _ = self.make_model(output_action_type="ee")
        observations = []
        for prompt in ("Hang the mugs on the mug rack.",
                       "Place all the objects on the table into the backpack.",
                       "Hang the mugs on the mug rack."):
            self.observe(model, prompt)
            observations.append(model._obs)
        model.update_obs_batch(observations)
        actions = model.get_action_batch()
        self.assertEqual([len(chunk) for chunk in actions], [30, 20, 30])
        self.assertIn("left_ee_pose", actions[0][0])
        self.assertIn("left_arm_joint_state", actions[1][0])
        self.assertIn("left_ee_pose", actions[2][0])

    def test_other_robots_keep_joint_output(self):
        for robot in ("piper", "arx_x5"):
            model, _ = self.make_model(robot, output_action_type="auto")
            for prompt in ("Place all the objects on the table into the backpack.",
                           "Place all the fruits into the blue bowl, then pour the fruits from the blue bowl into the large white bowl.",
                           "An unconfigured task prompt."):
                with self.subTest(robot=robot, prompt=prompt):
                    self.observe(model, prompt)
                    actions = model.get_action()
                    self.assertEqual(len(actions), 20)
                    self.assertIn("left_arm_joint_state", actions[0])
                    self.assertNotIn("left_ee_pose", actions[0])

    def test_joint_observation_is_used_for_piperx_endpose_output(self):
        model, pipeline = self.make_model(output_action_type="ee")
        self.observe(model, "Hang the mugs on the mug rack.")
        observation = dict(model._obs)
        joint_state = np.arange(14, dtype=np.float32) / 100
        observation["state"] = {
            "left_arm_joint_state": joint_state[:6],
            "left_ee_joint_state": joint_state[6:7],
            "right_arm_joint_state": joint_state[7:13],
            "right_ee_joint_state": joint_state[13:14],
        }
        # Unrelated optional client metadata is not a new rejection condition.
        observation["robot_type"] = "arm"
        model.update_obs(observation)
        actions = model.get_action()
        np.testing.assert_array_equal(pipeline.last_state, joint_state)
        self.assertEqual(len(actions), 30)
        self.assertIn("left_ee_pose", actions[0])

    def test_every_piperx_task_defaults_to_joint_without_fk_or_z(self):
        for output_type in (None, "auto", "joint"):
            model, pipeline = self.make_model(output_action_type=output_type)
            self.assertEqual(model.output_action_type, "joint")
            prompts = [(rule.prompt, EXPECTED_HORIZONS["piper_x"][rule.task])
                       for rule in model._gripper_thresholds.values()]
            prompts.append(("An unconfigured task prompt.", 20))
            with patch("XPolicyLab.policy.XBrain_v1.model.joints14_to_endpose16",
                       side_effect=AssertionError("Default PiperX output must not use FK")), \
                 patch("XPolicyLab.policy.XBrain_v1.model.adjust_piperx_endpose_z",
                       side_effect=AssertionError("Default PiperX output must not adjust Z")):
                for prompt, horizon in prompts:
                    with self.subTest(output_type=output_type, prompt=prompt):
                        self.observe(model, prompt)
                        actions = model.get_action()
                        self.assertEqual(len(actions), horizon)
                        for index, action in enumerate(actions):
                            self.assertEqual(set(action), {
                                "left_arm_joint_state", "left_ee_joint_state",
                                "right_arm_joint_state", "right_ee_joint_state"})
                            np.testing.assert_array_equal(action["left_arm_joint_state"], pipeline.prediction[index, :6].numpy())

    def test_task_scales_are_resolved_for_each_new_prompt(self):
        model, _ = self.make_model("piper")
        for prompt, left, right in (
            ("Pick up the pen holder and place all the pens into it.", 0.0, 0.0),
            ("Place all the objects on the table into the basket.", 0.0, 0.0),
            ("Insert the charger plug into the power strip, then connect the charging cable to the plug.", 0.385, 0.0),
            ("Stack the bowls on the table.", 0.455, 0.39),
        ):
            with self.subTest(prompt=prompt):
                self.observe(model, prompt)
                rows = self.rows(model.get_action())
                np.testing.assert_allclose(rows[:, 6], left, rtol=1e-6)
                np.testing.assert_allclose(rows[:, 13], right, rtol=1e-6)

    def test_standard_protocol_roundtrip_preserves_output_types(self):
        from client_server.ws.model_server import PolicyServer
        from client_server.ws.protocol.codec import decode_envelope, encode_frame
        from client_server.ws.protocol.messages import MessageType
        from client_server.ws.protocol.schemas import Frame

        async def check():
            cases = (
                ("piper_x", "Hang the mugs on the mug rack.", 0.4025, 0.345, 30),
                ("piper_x", "Place all the objects on the table into the backpack.", 0.385, 0.33, 20),
                ("piper_x", "Pick up the broom, hand it over to the right hand, then use the dustpan to sweep the blocks.", 0.455, 0.39, 30),
                ("piper", "Place all the objects on the table into the basket.", 0.0, 0.0, 20),
                ("piper", "Pick up the pen holder and place all the pens into it.", 0.0, 0.0, 20),
                ("piper", "Stack the blocks on the table, then cover them with the cup.", 0.0, 0.0, 40),
                ("arx_x5", "Pick up the two slices of bread from the bowl and place them into the toaster, then place the two small bowls on the plate.", 0.455, 0.39, 30),
                ("arx_x5", "Place all the fruits into the blue bowl, then pour the fruits from the blue bowl into the large white bowl.", 0.385, 0.33, 20),
            )
            for robot, prompt, left, right, horizon in cases:
                model, _ = self.make_model(robot)
                server = PolicyServer(model)
                self.observe(model, prompt)
                observation = model._obs
                reset = Frame(message_type=MessageType.RESET, request_id="reset",
                              evaluation_id="offline-check", payload={})
                reset_reply = await server.process_frame(decode_envelope(encode_frame(reset)))
                self.assertEqual(reset_reply.message_type, MessageType.RESET_RESULT)
                infer = Frame(message_type=MessageType.INFER, request_id="infer",
                              evaluation_id="offline-check", payload={"observation": observation})
                reply = await server.process_frame(decode_envelope(encode_frame(infer)))
                response = decode_envelope(encode_frame(reply))
                self.assertEqual(response.message_type, MessageType.INFER_RESULT, response.payload)
                actions = response.payload["actions"]
                self.assertEqual(len(actions), horizon)
                self.assertIn("left_arm_joint_state", actions[0])
                for action in actions:
                    self.assertNotIn("left_ee_pose", action)
                    np.testing.assert_allclose(action["left_ee_joint_state"], [left], rtol=1e-6)
                    np.testing.assert_allclose(action["right_ee_joint_state"], [right], rtol=1e-6)
                    self.assertTrue(all(np.isfinite(value).all() for value in action.values()))

        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
