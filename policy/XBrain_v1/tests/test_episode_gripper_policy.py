"""First observed joint2 threshold hit locks gripper roles until the next trial."""

import asyncio
import unittest

import numpy as np
import torch

from XPolicyLab.policy.XBrain_v1.tests import test_action_output


FILL_PROMPT = "Pick up the pen holder and place all the pens into it."


class EpisodeGripperPolicyTests(unittest.TestCase):
    make_model = test_action_output.ActionOutputTests.make_model
    rows = staticmethod(test_action_output.ActionOutputTests.rows)

    @staticmethod
    def observation(left_joint2, right_joint2, *, structured=False, prompt=FILL_PROMPT):
        state = np.zeros(14, dtype=np.float32)
        state[1], state[8] = left_joint2, right_joint2
        if structured:
            state = {"left_arm_joint_state": state[:6], "left_ee_joint_state": state[6:7],
                     "right_arm_joint_state": state[7:13], "right_ee_joint_state": state[13:14]}
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        return dict(state=state, instruction=prompt,
                    vision={k: {"color": image} for k in
                            ("cam_head", "cam_left_wrist", "cam_right_wrist")})

    def test_first_threshold_hit_is_fixed_until_reset(self):
        model, pipe = self.make_model("piper")
        pipe.prediction[:, 6] = .9
        pipe.prediction[:, 13] = .3
        model.update_obs(self.observation(.5, .5))
        pending = self.rows(model.get_action())
        self.assertEqual(pending.shape, (40, 14))
        np.testing.assert_allclose(pending[:, 6], .9, rtol=1e-6)
        np.testing.assert_allclose(pending[:, 13], .3, rtol=1e-6)
        self.assertFalse(model._episode_gripper_rules)
        model.update_obs(self.observation(1.2, .5))
        first = self.rows(model.get_action())
        self.assertEqual(first.shape, (40, 14))
        np.testing.assert_allclose(first[:, 6], .945, rtol=1e-6)
        np.testing.assert_allclose(first[:, 13], .39, rtol=1e-6)
        # Both qualify now; the episode's first observed hit is already fixed.
        model.update_obs(self.observation(1.5, 1.6))
        np.testing.assert_array_equal(self.rows(model.get_action()), first)
        # Cross the cutoff and exchange arm poses: still keep the initial role.
        model.update_obs(self.observation(.5, 1.2))
        np.testing.assert_array_equal(self.rows(model.get_action()), first)
        model.reset()
        model.update_obs(self.observation(.5, 1.2))
        second = self.rows(model.get_action())
        self.assertEqual(second.shape, (40, 14))
        np.testing.assert_allclose(second[:, 6], 1.17, rtol=1e-6)
        np.testing.assert_array_equal(second[:, 13], 0)
        self.assertEqual(model.action_horizon, 20)

    def test_updates_between_inferences_select_the_first_arm_not_the_latest_state(self):
        for selected_left in (True, False):
            with self.subTest(selected_left=selected_left):
                model, pipe = self.make_model("piper")
                model.update_obs(self.observation(.5, .5))
                model.get_action()
                # No get_action between these two observation updates.
                model.update_obs(self.observation(1.2 if selected_left else .5,
                                                  .5 if selected_left else 1.2))
                model.update_obs(self.observation(.5 if selected_left else 1.2,
                                                  1.2 if selected_left else .5))
                actual = self.rows(model.get_action())
                expected_left, expected_right = (0, .39) if selected_left else (.455, 0)
                np.testing.assert_allclose(actual[:, 6], expected_left, rtol=1e-6)
                np.testing.assert_allclose(actual[:, 13], expected_right, rtol=1e-6)
                self.assertEqual(pipe.calls, 2)

    def test_pending_gripper_threshold_is_applied_without_amplification(self):
        model, pipe = self.make_model("piper")
        pipe.prediction[:3, 6] = torch.tensor([.199, .2, .201])
        pipe.prediction[:3, 13] = torch.tensor([.201, .2, .199])
        original = pipe.prediction.numpy().copy()
        model.update_obs(self.observation(.5, .5))
        result = self.rows(model.get_action())
        np.testing.assert_allclose(result[:3, 6], [0, .2, .201], rtol=1e-6)
        np.testing.assert_allclose(result[:3, 13], [.201, .2, 0], rtol=1e-6)
        np.testing.assert_array_equal(pipe.prediction.numpy(), original)
        self.assertFalse(model._episode_gripper_rules)

    def test_both_arm_assignments_threshold_boundaries_and_joint_targets(self):
        for selected_left in (True, False):
            with self.subTest(selected_left=selected_left):
                model, pipe = self.make_model("piper")
                left_threshold, right_threshold = (.75, .2) if selected_left else (.2, .75)
                pipe.prediction[:3, 6] = torch.tensor([
                    left_threshold - .01, left_threshold, left_threshold + .01])
                pipe.prediction[:3, 13] = torch.tensor([
                    right_threshold + .01, right_threshold, right_threshold - .01])
                original = pipe.prediction.numpy().copy()
                model.update_obs(self.observation(1.2 if selected_left else .5,
                                                  .5 if selected_left else 1.2, structured=True))
                actions = self.rows(model.get_action())
                left_scale, right_scale = (1.05, 1.3) if selected_left else (1.3, 1.05)
                np.testing.assert_allclose(actions[:3, 6],
                    [0, left_threshold * left_scale, (left_threshold + .01) * left_scale], rtol=1e-6)
                np.testing.assert_allclose(actions[:3, 13],
                    [(right_threshold + .01) * right_scale, right_threshold * right_scale, 0], rtol=1e-6)
                arm_columns = list(range(6)) + list(range(7, 13))
                np.testing.assert_array_equal(actions[:, arm_columns], original[:40, arm_columns])
                np.testing.assert_array_equal(pipe.prediction.numpy(), original)

    def test_exact_cutoff_and_simultaneous_hits_stay_pending_and_invalid_states_fail(self):
        for left, right in ((.5, .5), (1.2, 1.3), (1.0, 1.0)):
            model, pipe = self.make_model("piper")
            with self.subTest(left=left, right=right):
                model.update_obs(self.observation(left, right))
                result = self.rows(model.get_action())
                np.testing.assert_allclose(result[:, 6], .35, rtol=1e-6)
                np.testing.assert_allclose(result[:, 13], .30, rtol=1e-6)
            self.assertEqual(pipe.calls, 1)
            self.assertFalse(model._episode_gripper_rules)
        model, pipe = self.make_model("piper")
        for state in (np.zeros(13, dtype=np.float32), np.full(14, np.nan, dtype=np.float32)):
            obs = self.observation(1.2, .5)
            obs["state"] = state
            with self.assertRaisesRegex(ValueError, "finite state"):
                model.update_obs(obs)
        self.assertEqual(pipe.calls, 0)
        self.assertFalse(model._episode_gripper_rules)
        # Exactly 1.0 is not greater, so the right arm is the unique selected arm.
        model.update_obs(self.observation(1.0, 1.01))
        rule = model._episode_gripper_rules["single"]
        self.assertEqual((rule.left, rule.right, rule.left_scale, rule.right_scale), (.2, .75, 1.3, 1.05))

    def test_prepare_case_trial_end_and_prompt_changes_reselect_without_restart(self):
        model, _ = self.make_model("piper")
        for boundary in (model.prepare_case, model.on_trial_end, model.reset):
            model.update_obs(self.observation(1.2, .5))
            self.assertEqual(model._episode_gripper_rules["single"].left, .75)
            boundary()
            self.assertFalse(model._episode_gripper_rules)
            self.assertIsNone(model._obs)
            model.update_obs(self.observation(.5, 1.2))
            self.assertEqual(model._episode_gripper_rules["single"].right, .75)
            model.reset()
        model.update_obs(self.observation(1.2, .5))
        model.update_obs(self.observation(.5, 1.2, prompt="Stack the bowls on the table."))
        self.assertFalse(model._episode_gripper_rules)
        model.update_obs(self.observation(.5, 1.2))
        self.assertEqual(model._episode_gripper_rules["single"].right, .75)

    def test_batch_arms_are_selected_separately_and_stay_with_environment_ids(self):
        model, pipe = self.make_model("piper")
        pipe.prediction[:, 6] = .9
        pipe.prediction[:, 13] = .3
        model.update_obs_batch([self.observation(1.2, .5), self.observation(.5, 1.2)])
        first = model.get_action_batch([2, 9])
        np.testing.assert_allclose(self.rows(first[0])[:, 6], .945, rtol=1e-6)
        np.testing.assert_allclose(self.rows(first[1])[:, 6], 1.17, rtol=1e-6)
        # The surviving environment moves to batch slot 0, but keeps its role.
        model.update_obs_batch([self.observation(1.2, .5)])
        remaining = self.rows(model.get_action_batch([9])[0])
        np.testing.assert_array_equal(remaining, self.rows(first[1]))
        model.reset()
        model.update_obs_batch([self.observation(1.2, .5)])
        restarted = self.rows(model.get_action_batch([9])[0])
        np.testing.assert_array_equal(restarted, self.rows(first[0]))

    def test_batch_observations_between_inferences_keep_first_hits(self):
        model, pipe = self.make_model("piper")
        pipe.prediction[:, 6] = .9
        pipe.prediction[:, 13] = .3
        model.update_obs_batch([self.observation(.5, .5), self.observation(.5, .5)])
        pending = model.get_action_batch([2, 9])
        np.testing.assert_allclose(self.rows(pending[0])[:, 6], .9, rtol=1e-6)
        model.update_obs_batch([self.observation(1.2, .5), self.observation(.5, 1.2)])
        model.update_obs_batch([self.observation(.5, 1.2), self.observation(1.2, .5)])
        selected = model.get_action_batch([2, 9])
        np.testing.assert_allclose(self.rows(selected[0])[:, 6], .945, rtol=1e-6)
        np.testing.assert_allclose(self.rows(selected[1])[:, 6], 1.17, rtol=1e-6)
        self.assertEqual(pipe.calls, 4)

    def test_batch_first_hits_before_indices_arrive_are_replayed_in_observation_order(self):
        model, _ = self.make_model("piper")
        model.update_obs_batch([self.observation(.5, .5), self.observation(.5, .5)])
        model.update_obs_batch([self.observation(1.2, .5), self.observation(.5, 1.2)])
        model.update_obs_batch([self.observation(.5, 1.2), self.observation(1.2, .5)])
        model.get_action_batch([2, 9])
        self.assertEqual(model._episode_gripper_rules[("batch", 2)].left, .75)
        self.assertEqual(model._episode_gripper_rules[("batch", 9)].right, .75)

    def test_new_static_task_parameters_return_exact_prefixes(self):
        model, pipe = self.make_model("piper")
        for prompt, gripper, expected in (
            ("Stack the bowls on the table.", .30, .45),
            ("Place all the objects on the table into the basket.", .30, .30),
            ("Stand the bottle upright.", .75, .90),
        ):
            with self.subTest(prompt=prompt):
                pipe.prediction[:, 6] = pipe.prediction[:, 13] = gripper
                model.update_obs(self.observation(0, 0, prompt=prompt))
                actions = self.rows(model.get_action())
                self.assertEqual(actions.shape, (30, 14))
                np.testing.assert_allclose(actions[:, [6, 13]], expected, rtol=1e-6)
                np.testing.assert_array_equal(actions[:, :6], pipe.prediction.numpy()[:30, :6])

    def test_standard_protocol_repeated_trials_select_new_roles_and_retries_do_not_reset(self):
        from client_server.ws.model_server import PolicyServer
        from client_server.ws.protocol.messages import MessageType
        from client_server.ws.protocol.schemas import Frame
        from client_server.ws.protocol.codec import decode_envelope, encode_frame
        async def check():
            model, pipe = self.make_model("piper")
            pipe.prediction[:, 6] = .9
            pipe.prediction[:, 13] = .3
            server = PolicyServer(model)
            async def rpc(kind, rid, payload=None):
                frame = Frame(message_type=kind, request_id=rid, evaluation_id="eval",
                              trial_id="same-task-trial", payload=payload or {})
                reply = await server.process_frame(decode_envelope(encode_frame(frame)))
                self.assertNotEqual(reply.message_type, MessageType.ERROR, reply.payload)
                return reply
            await rpc(MessageType.RESET, "reset-1")
            first = await rpc(MessageType.INFER, "infer-1", {"observation": self.observation(1.2, .5)})
            np.testing.assert_allclose(self.rows(first.payload["actions"])[:, 6], .945, rtol=1e-6)
            # A retry of that exact inference never reselects roles or reinfers.
            await rpc(MessageType.INFER, "infer-1", {"observation": self.observation(.5, 1.2)})
            self.assertEqual(pipe.calls, 1)
            self.assertEqual(model._episode_gripper_rules["single"].left, .75)
            await rpc(MessageType.TRIAL_END, "end-1", {"success": True})
            self.assertFalse(model._episode_gripper_rules)
            await rpc(MessageType.RESET, "reset-2")
            pending = await rpc(MessageType.INFER, "infer-2", {"observation": self.observation(.5, .5)})
            np.testing.assert_allclose(self.rows(pending.payload["actions"])[:, 6], .9, rtol=1e-6)
            # These CALL observation updates come from the unchanged client flow.
            await rpc(MessageType.CALL, "obs-right", {
                "func_name": "update_obs", "obs": self.observation(.5, 1.2)})
            await rpc(MessageType.CALL, "obs-left", {
                "func_name": "update_obs", "obs": self.observation(1.2, .5)})
            second = await rpc(MessageType.CALL, "actions-2", {"func_name": "get_action"})
            result = self.rows(second.payload["result"])
            self.assertEqual(result.shape, (40, 14))
            np.testing.assert_allclose(result[:, 6], 1.17, rtol=1e-6)
            np.testing.assert_array_equal(result[:, 13], 0)
            self.assertEqual(pipe.calls, 3)
        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
