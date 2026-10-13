"""Check real-state triggers, ordered chunk exits and per-trial/arm isolation."""

import unittest

import numpy as np
import torch

from XPolicyLab.policy.XBrain_v1.tests import test_action_output


BOTTLE_PROMPT = "Stand the bottle upright."


class BottleJoint5CorrectionTests(unittest.TestCase):
    make_model = test_action_output.ActionOutputTests.make_model
    rows = staticmethod(test_action_output.ActionOutputTests.rows)

    @staticmethod
    def observation(left_gripper=.5, left_joint5=-1.2, right_gripper=.7,
                    right_joint5=-1.2, *, structured=False, prompt=BOTTLE_PROMPT):
        state = np.zeros(14, dtype=np.float32)
        state[4], state[6] = left_joint5, left_gripper
        state[11], state[13] = right_joint5, right_gripper
        if structured:
            state = {"left_arm_joint_state": state[:6], "left_ee_joint_state": state[6:7],
                     "right_arm_joint_state": state[7:13], "right_ee_joint_state": state[13:14]}
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        return dict(state=state, instruction=prompt,
                    vision={k: {"color": image} for k in
                            ("cam_head", "cam_left_wrist", "cam_right_wrist")})

    def make_bottle_model(self):
        model, pipe = self.make_model("piper")
        pipe.prediction[:, 4] = pipe.prediction[:, 11] = -1.15
        pipe.prediction[:, 6] = pipe.prediction[:, 13] = .75
        return model, pipe

    def test_each_arm_uses_its_own_real_gripper_and_joint5_state(self):
        for selected_left in (True, False):
            model, pipe = self.make_bottle_model()
            model.update_obs(self.observation(.5 if selected_left else .7, -1.2,
                                              .7 if selected_left else .5, -1.2,
                                              structured=True))
            original = pipe.prediction.numpy().copy()
            expected = original[:30].copy()
            expected[:, 4 if selected_left else 11] -= np.float32(.05)
            expected[:, [6, 13]] *= np.float32(1.2)
            actual = self.rows(model.get_action())
            with self.subTest(selected_left=selected_left):
                np.testing.assert_allclose(actual, expected, rtol=1e-6)
                np.testing.assert_array_equal(pipe.prediction.numpy(), original)
                self.assertEqual(actual.shape, (30, 14))

    def test_only_joint5_changes_and_floor_is_applied_without_accumulation(self):
        model, pipe = self.make_bottle_model()
        pipe.prediction[:5, 4] = torch.tensor([-1.1, -1.15, -1.20, -1.24, -1.30])
        model.update_obs(self.observation(right_gripper=.5))
        first = self.rows(model.get_action())
        second = self.rows(model.get_action())
        np.testing.assert_allclose(first[:5, 4], [-1.15, -1.20, -1.25, -1.25, -1.25], rtol=1e-6)
        self.assertTrue(np.all(first[:, [4, 11]] >= np.float32(-1.25)))
        np.testing.assert_array_equal(first, second)
        untouched = [0, 1, 2, 3, 5, 7, 8, 9, 10, 12]
        np.testing.assert_array_equal(first[:, untouched], pipe.prediction.numpy()[:30, untouched])

    def test_strict_observation_cutoffs_and_active_equality(self):
        for gripper, joint5 in ((.6, -1.2), (.5, -1.1), (.7, -1.2), (.5, -1.0)):
            model, pipe = self.make_bottle_model()
            model.update_obs(self.observation(gripper, joint5))
            with self.subTest(gripper=gripper, joint5=joint5):
                np.testing.assert_array_equal(self.rows(model.get_action())[:, 4],
                                              pipe.prediction.numpy()[:30, 4])
        model, _ = self.make_bottle_model()
        model.update_obs(self.observation())
        model.update_obs(self.observation(.6, -1.1))
        np.testing.assert_allclose(self.rows(model.get_action())[:, 4], -1.20, rtol=1e-6)

    def test_prediction_exit_keeps_that_row_and_tail_raw_until_a_new_observation(self):
        model, pipe = self.make_bottle_model()
        pipe.prediction[:5, 4] = torch.tensor([-1.15, -1.1, -1.09, -1.2, -1.24])
        model.update_obs(self.observation(right_gripper=.5))
        first = self.rows(model.get_action())
        np.testing.assert_allclose(first[:2, 4], [-1.20, -1.15], rtol=1e-6)
        np.testing.assert_array_equal(first[2:, 4], pipe.prediction.numpy()[2:30, 4])
        # The right arm stays active independently of the left arm's exit.
        np.testing.assert_allclose(first[:, 11], -1.20, rtol=1e-6)
        second = self.rows(model.get_action())
        np.testing.assert_array_equal(second[:, 4], pipe.prediction.numpy()[:30, 4])
        # Another received low-gripper/low-joint5 observation rearms left.
        model.update_obs(self.observation(right_gripper=.5))
        np.testing.assert_array_equal(self.rows(model.get_action()), first)

    def test_discarded_prediction_tail_cannot_disable_returned_prefix(self):
        model, pipe = self.make_bottle_model()
        pipe.prediction[30, 4] = -1.0
        model.update_obs(self.observation())
        for _ in range(2):
            np.testing.assert_allclose(self.rows(model.get_action())[:, 4], -1.20, rtol=1e-6)
        self.assertTrue(model._bottle_joint5_corrections["single"].active[0])

    def test_real_gripper_exit_is_processed_between_inferences_and_can_rearm(self):
        model, pipe = self.make_bottle_model()
        model.update_obs(self.observation())
        model.get_action()
        model.update_obs(self.observation(.61, -1.2))
        model.update_obs(self.observation(.6, -1.2))
        np.testing.assert_array_equal(self.rows(model.get_action())[:, 4],
                                      pipe.prediction.numpy()[:30, 4])
        model.update_obs(self.observation(.5, -1.0))
        np.testing.assert_array_equal(self.rows(model.get_action())[:, 4],
                                      pipe.prediction.numpy()[:30, 4])
        model.update_obs(self.observation(.5, -1.2))
        np.testing.assert_allclose(self.rows(model.get_action())[:, 4], -1.20, rtol=1e-6)

    def test_observed_joint5_can_activate_between_action_requests(self):
        model, _ = self.make_bottle_model()
        model.update_obs(self.observation(.5, -1.0))
        model.get_action()
        model.update_obs(self.observation(.5, -1.2))
        # Only predicted joint5 or real gripper state can end an active rule.
        model.update_obs(self.observation(.5, -1.0))
        np.testing.assert_allclose(self.rows(model.get_action())[:, 4], -1.20, rtol=1e-6)

    def test_trial_boundaries_and_prompt_changes_clear_corrections(self):
        model, pipe = self.make_bottle_model()
        for boundary in (model.reset, model.prepare_case, model.on_trial_end):
            model.update_obs(self.observation())
            self.assertTrue(model._bottle_joint5_corrections["single"].active[0])
            boundary()
            self.assertFalse(model._bottle_joint5_corrections)
            model.update_obs(self.observation(.6, -1.2))
            np.testing.assert_array_equal(self.rows(model.get_action())[:, 4],
                                          pipe.prediction.numpy()[:30, 4])
        model.update_obs(self.observation())
        model.update_obs(self.observation(prompt="Stack the bowls on the table."))
        self.assertFalse(model._bottle_joint5_corrections)
        np.testing.assert_array_equal(self.rows(model.get_action())[:, 4],
                                      pipe.prediction.numpy()[:30, 4])
        model.update_obs(self.observation(.6, -1.2))
        np.testing.assert_array_equal(self.rows(model.get_action())[:, 4],
                                      pipe.prediction.numpy()[:30, 4])

    def test_other_robots_do_not_enable_this_piper_task_correction(self):
        for robot in ("piper_x", "arx_x5"):
            model, pipe = self.make_model(robot)
            pipe.prediction[:, 4] = pipe.prediction[:, 11] = -1.2
            model.update_obs(self.observation(right_gripper=.5))
            self.assertFalse(model._bottle_joint5_corrections)
            np.testing.assert_array_equal(self.rows(model.get_action()), pipe.prediction.numpy()[:20])

    def test_batch_observation_history_is_ordered_and_environments_do_not_share_gates(self):
        model, pipe = self.make_bottle_model()
        model.update_obs_batch([self.observation(), self.observation(.7, -1.2, .5, -1.2)])
        # Low observation followed by high must leave env 2 disarmed, even if
        # its final gripper state equals the cutoff before indices arrive.
        model.update_obs_batch([self.observation(.7), self.observation(.7, -1.2, .5, -1.2)])
        model.update_obs_batch([self.observation(.6), self.observation(.7, -1.2, .5, -1.2)])
        actions = model.get_action_batch([2, 9])
        np.testing.assert_array_equal(self.rows(actions[0])[:, 4], pipe.prediction.numpy()[:30, 4])
        np.testing.assert_allclose(self.rows(actions[1])[:, 11], -1.20, rtol=1e-6)
        model.update_obs_batch([self.observation(), self.observation(.7, -1.2, .7, -1.2)])
        pipe.prediction[2, 4] = -1.0
        first = model.get_action_batch([2, 9])
        np.testing.assert_allclose(self.rows(first[0])[:2, 4], -1.20, rtol=1e-6)
        # Repeating get_action_batch without update_obs_batch must not rearm.
        repeated = model.get_action_batch([2, 9])
        np.testing.assert_array_equal(self.rows(repeated[0])[:, 4], pipe.prediction.numpy()[:30, 4])
        np.testing.assert_array_equal(self.rows(repeated[1])[:, 11], pipe.prediction.numpy()[:30, 11])

    def test_invalid_observations_fail_before_inference(self):
        model, pipe = self.make_bottle_model()
        for state in (np.zeros(13, dtype=np.float32), np.full(14, np.nan, dtype=np.float32)):
            obs = self.observation()
            obs["state"] = state
            with self.assertRaisesRegex(ValueError, "finite state"):
                model.update_obs(obs)
        self.assertEqual(pipe.calls, 0)


if __name__ == "__main__":
    unittest.main()
