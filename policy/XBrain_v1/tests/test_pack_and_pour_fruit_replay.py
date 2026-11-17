"""Synthetic tests for pack-and-pour fruit visual trajectory replay."""

import unittest

import numpy as np

from XPolicyLab.policy.XBrain_v1.pack_and_pour_fruit import detect_blue_side
from XPolicyLab.policy.XBrain_v1.tests import test_action_output


PROMPT = "Place all the fruits into the blue bowl, then pour the fruits from the blue bowl into the large white bowl."


class PackAndPourFruitReplayTests(unittest.TestCase):
    make_model = test_action_output.ActionOutputTests.make_model
    rows = staticmethod(test_action_output.ActionOutputTests.rows)

    @staticmethod
    def observation(side):
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        if side == "left":
            image[80:150, 35:95] = [40, 80, 220]
        elif side == "right":
            image[80:150, 130:190] = [40, 80, 220]
        return dict(
            vision={name: {"color": image} for name in
                    ("cam_head", "cam_left_wrist", "cam_right_wrist")},
            state=np.zeros(14, dtype=np.float32), instruction=PROMPT,
        )

    def test_blue_detector_handles_stitched_reference_layout_upper_half(self):
        image = np.zeros((360, 320, 3), dtype=np.uint8)
        image[60:130, 30:90] = [40, 80, 220]
        image[260:330, 240:300] = [40, 80, 220]
        self.assertEqual(detect_blue_side(image)[0], "left")

    def test_left_side_replays_trajectory_one_then_resumes_vla(self):
        model, pipeline = self.make_model("arx_x5")
        source = model._fruit_trajectories["left"].copy()
        observation = self.observation("left")
        model.update_obs(observation)
        first = self.rows(model.get_action())
        self.assertEqual(first.shape, (50, 14))
        np.testing.assert_array_equal(first, source[:50])
        self.assertEqual(pipeline.calls, 1)

        model.update_obs(observation)
        second = self.rows(model.get_action())
        self.assertEqual(second.shape, (40, 14))
        np.testing.assert_array_equal(second, source[50:90])
        self.assertEqual(pipeline.calls, 1)

        # A repeated action request without a new observation replays the same
        # chunk and does not consume the trajectory a second time.
        repeated = self.rows(model.get_action())
        np.testing.assert_array_equal(repeated, second)
        self.assertEqual(pipeline.calls, 1)

        model.update_obs(observation)
        resumed = self.rows(model.get_action())
        self.assertEqual(resumed.shape, (20, 14))
        self.assertEqual(pipeline.calls, 2)
        np.testing.assert_allclose(resumed[:, 6], .385, rtol=1e-6)
        np.testing.assert_allclose(resumed[:, 13], .33, rtol=1e-6)

    def test_right_side_selects_trajectory_two_and_first_prediction_is_discarded(self):
        model, pipeline = self.make_model("arx_x5")
        source = model._fruit_trajectories["right"].copy()
        model.update_obs(self.observation("right"))
        first = self.rows(model.get_action())
        self.assertEqual(first.shape, (50, 14))
        np.testing.assert_array_equal(first, source[:50])
        self.assertEqual(pipeline.calls, 1)

    def test_replay_resets_for_a_new_episode_and_ambiguous_blue_uses_vla(self):
        model, pipeline = self.make_model("arx_x5")
        model.update_obs(self.observation("left"))
        model.get_action()
        model.reset()
        ambiguous = self.observation("none")
        model.update_obs(ambiguous)
        action = self.rows(model.get_action())
        self.assertEqual(action.shape, (20, 14))
        self.assertEqual(pipeline.calls, 2)


if __name__ == "__main__":
    unittest.main()
