"""Regression tests for the boundary between observation caching and inference."""
from pathlib import Path
import runpy
import unittest
import sys
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from XPolicyLab.utils.action_chunks import validate_action_batch
from XPolicyLab.utils.batch_indices import observation_rows
from XPolicyLab.utils.process_data import decode_obs_images, encode_image_bit


POLICIES = Path(__file__).resolve().parents[1] / "policy"


class PolicyBatchTests(unittest.TestCase):
    def test_sparse_environment_selection(self):
        self.assertEqual(observation_rows([7, 3], None), [0, 1])
        self.assertEqual(observation_rows([7, 3], [3, 7]), [1, 0])
        self.assertEqual(observation_rows([7, 3], []), [])
        for observed, requested in (([7, 7], [7]), ([7], [7, 7]), ([7], [3])):
            with self.assertRaises(ValueError):
                observation_rows(observed, requested)

    def test_go3_predicts_for_requested_environment(self):
        with patch.object(sys, "path", list(sys.path)):
            model_type = runpy.run_path(str(POLICIES / "GO_3_Nano_preview/model.py"))["Model"]
        model = model_type.__new__(model_type)
        model.reset()
        model._prepare = lambda obs: obs["label"]
        model._infer = Mock(side_effect=lambda label: [{"label": label}])
        model.update_obs_batch([{"env_idx": 7, "label": "left"}, {"env_idx": 3, "label": "right"}])
        for ids, labels in (([3], ["right"]), ([3, 7], ["right", "left"]), (None, ["left", "right"])):
            self.assertEqual([chunk[0]["label"] for chunk in model.get_action_batch(ids)], labels)
        model._infer.reset_mock()
        with self.assertRaises(ValueError):
            model.get_action_batch([7, 99])
        model._infer.assert_not_called()

    def test_g05_reorders_history_keys_with_the_observations(self):
        with patch.dict(sys.modules, {"torch": Mock()}):
            model_type = runpy.run_path(str(POLICIES / "G05/model.py"))["Model"]
        model = model_type.__new__(model_type)
        model._predict_chunks = Mock(return_value=[])
        batch = [{"env_idx": 7, "label": "left"}, {"env_idx": 3, "label": "right"}]
        model.update_obs_batch(batch)
        model.get_action_batch([3, 7])
        model._predict_chunks.assert_called_once_with(batch[::-1], env_keys=[3, 7])
        model.update_obs(batch[0])
        with self.assertRaises(ValueError):
            model.get_action_batch([3, 7])

    def test_pi05_reorders_stacked_observations_and_returns_packed_actions(self):
        modules = {f"openpi.{name}": Mock() for name in ("policies", "shared", "training")}
        with patch.dict(sys.modules, modules):
            model_type = runpy.run_path(str(POLICIES / "Pi_05/model.py"))["Model"]
        model = model_type.__new__(model_type)
        model.robot_action_dim_info = None
        model._latest_env_idx_list = [7, 3]
        states = np.array([[70], [30]])
        model.observation_window = {
            "state": states, "prompt": ["left", "right"],
            "images": dict.fromkeys(("cam_high", "cam_left_wrist", "cam_right_wrist"), states),
        }
        model.policy = Mock()
        model.policy.infer.side_effect = lambda obs: {"actions": obs["state"][None, :]}
        chunks = model.get_action_batch([3, 7])
        self.assertEqual(validate_action_batch(chunks, 2), 1)
        np.testing.assert_array_equal(chunks, [[[30]], [[70]]])

    def test_rdt_infers_with_each_environments_current_instruction(self):
        name = "XPolicyLab.policy.RDT_1B"
        torch = MagicMock()
        torch.inference_mode.return_value = lambda f: f
        modules = {
            "torch": torch,
            f"{name}.rdt.scripts.robodojo_model": Mock(),
            f"{name}.rdt.models.multimodal_encoder.t5_encoder": Mock(),
        }
        with patch.dict(sys.modules, modules), patch.object(sys, "path", list(sys.path)):
            module = runpy.run_path(str(POLICIES / "RDT_1B/model.py"), run_name=f"{name}.model")
        model = module["Model"].__new__(module["Model"])
        model.reset()
        model.device = "cpu"
        model.default_prompt = ""
        model.config = {"camera_names": ["head", "left", "right"]}
        model._encode_instruction = Mock(side_effect=lambda prompt: prompt)
        model._jpeg_mapping = model._resize_img = lambda image: image
        model.policy = Mock()
        frame = np.zeros((4, 4, 3), np.uint8)
        cameras = {key: frame for key in ("cam_high", "cam_right_wrist", "cam_left_wrist")}
        with patch.dict(model.update_obs_batch.__globals__, {
            "encode_obs": lambda obs, _: dict(obs, images=cameras, state=[0]),
        }):
            model.update_obs_batch([{"env_idx": 7, "prompt": "lift"}, {"env_idx": 3, "prompt": "place"}])
            model.infer(model._observation_windows[3])
            model.infer(model._observation_windows[7])
            model.update_obs_batch([{"env_idx": 7, "prompt": "place"}])
            model.infer()
        self.assertEqual([call.kwargs["text_embeds"] for call in model.policy.step.call_args_list],
                         ["place", "lift", "place"])
        self.assertEqual(model._encode_instruction.call_count, 2)

    def test_numeric_chunks_and_invalid_late_steps(self):
        packed = np.zeros((2, 7), dtype=np.float32)
        self.assertEqual(validate_action_batch([packed, packed.copy()], 2), 2)
        for invalid in (None, {}, "wait", np.empty(0)):
            with self.assertRaises(ValueError):
                validate_action_batch([[{"x": 0}, {"x": 1}], [{"x": 0}, invalid]], 2)

    def test_server_decoder_visits_temporal_observations(self):
        rgb = np.full((8, 8, 3), [220, 30, 10], np.uint8)
        history = {"vision": {"head": {"color": encode_image_bit(rgb)}}}
        current = {"__g05_history_observations__": [history]}
        decode_obs_images({"__xpolicylab_batch_observations__": [current]})
        np.testing.assert_allclose(history["vision"]["head"]["color"], rgb, atol=3)


    def test_eventvla_failed_batch_preserves_runtime_and_retry_step(self):
        from XPolicyLab.policy.EventVLA.model import Model

        model = Model.__new__(Model)
        model.runtime_by_env = {}
        model.temporal_history_capacity = 5
        model.first_chunk_random_replan_seed = 123
        model.action_dim = 14
        model.action_type = "joint"
        model.robot_action_dim_info = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
        model._convert_obs = lambda obs: obs
        model.update_obs_batch([{"env_idx": 7}, {"env_idx": 2}])

        def step(runtime, obs, tick):
            runtime.image_history.append(tick)
            runtime.keyframe_image_memory.append(tick)
            runtime.first_chunk_random_replan_rng.random()
            return np.full(14, tick, dtype=np.float32)

        model._step_runtime = Mock(side_effect=step)
        for ids in ([7, 99], [7, 7]):
            with self.assertRaises(ValueError):
                model.get_action_batch(ids)
        model._step_runtime.assert_not_called()
        model.get_action_batch()
        old = model.runtime_by_env[7]
        rng_state = old.first_chunk_random_replan_rng.getstate()
        model._step_runtime.side_effect = lambda rt, obs, t: (
            step(rt, obs, t) if obs["env_idx"] == 7 else np.zeros(1))
        with self.assertRaisesRegex(ValueError, "raw action dim"):
            model.get_action_batch()
        self.assertEqual(old.step, 1)
        self.assertEqual(list(old.image_history), [0])
        self.assertEqual(old.keyframe_image_memory, [0])
        self.assertEqual(old.first_chunk_random_replan_rng.getstate(), rng_state)
        self.assertEqual(model.get_action_batch([7])[0][0]["left_arm_joint_state"][0], 1)
        self.assertEqual(model.runtime_by_env[7].step, 2)
