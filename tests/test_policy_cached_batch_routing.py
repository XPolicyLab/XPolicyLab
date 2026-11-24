"""Exercise cached adapters with real observation encoders and CPU inference doubles.

Only heavyweight module imports, constructors and model inference are replaced.
The adapter methods, RGB/state preprocessing and DreamZero history updates run
from their source so failed updates can be checked without model weights.
"""
from __future__ import annotations

import ast
from contextlib import nullcontext, redirect_stdout
import io
from pathlib import Path
import time
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from XPolicyLab.utils.batch_indices import observation_env_ids, selected_env_ids, validate_observed_ids
from XPolicyLab.utils.process_data import pack_robot_state, unpack_robot_state


POLICIES = Path(__file__).resolve().parents[1] / "policy"
NAMES = ("GO1", "A1", "DreamZero", "EWAM")
ROBOT = {"arm_dim": [2, 2], "ee_dim": [1, 1]}
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def observation(env_id, marker):
    obs = {
        "instruction": str(marker),
        "state": unpack_robot_state(np.full(6, marker, dtype=np.float32), "joint", ROBOT),
        "vision": {camera: {"color": np.full((4, 5, 3), marker, np.uint8)} for camera in CAMERAS},
    }
    if env_id is not None:
        obs["env_idx"] = env_id
    return obs


def markers(chunks):
    return [[int(action["left_arm_joint_state"][0]) for action in chunk] for chunk in chunks]


def history_markers(model):
    return {
        env_id: {key: [int(frame[0, 0, 0]) for frame in frames] for key, frames in buffers.items()}
        for env_id, buffers in model._frame_buffers.items()
    }


def make_model(name):
    path = POLICIES / name / "model.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model")
    method_names = {"update_obs", "update_obs_batch", "get_action", "get_action_batch", "reset"}
    if name == "DreamZero":
        method_names.add("_encode_obs")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in method_names]
    helper_names = {
        "_encode_obs", "_extract_and_prepare_image", "_extract_rgb_image",
        "_extract_image", "_stack_recent_frames", "_xpolicylab_packed_to_native_dojo_fields",
    }
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in helper_names]
    constants = [node for node in tree.body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in {"DEFAULT_CAMERA_GROUPS", "DEFAULT_IMAGE_SIZE"}
        for target in node.targets
    )]
    namespace = dict(
        np=np, cv2=cv2, time=time, Batch=SimpleNamespace,
        torch=SimpleNamespace(inference_mode=nullcontext, tensor=np.array, float32=np.float32),
        observation_env_ids=observation_env_ids, selected_env_ids=selected_env_ids,
        validate_observed_ids=validate_observed_ids, pack_robot_state=pack_robot_state,
        unpack_robot_state=unpack_robot_state,
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module([future] + constants + helpers + methods, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    model = type("CachedRoutingModel", (), {node.name: namespace[node.name] for node in methods})()
    model.action_type, model.robot_action_dim_info = "joint", ROBOT
    model.default_prompt, model.ctrl_freq, model.image_resize = "fallback", 25, None
    model.action_chunk_size, model.action_dim = 2, 6
    model.calls = []
    model._obs_batch = {}
    with redirect_stdout(io.StringIO()):
        model.reset()

    def predict(state, images, prompt):
        state = np.asarray(state).reshape(-1)
        marker = int(state[0])
        if str(marker) != prompt or any(int(image.reshape(-1, 3)[-1, 0]) != marker for image in images):
            raise AssertionError("Camera, prompt and robot state came from different observations")
        model.calls.append(marker)
        return np.stack([state, state + 1])

    if name == "GO1":
        model.model = SimpleNamespace(inference=lambda payload: predict(
            payload["state"], [payload[key] for key in ("top", "left", "right")], payload["instruction"]
        ))
    elif name == "A1":
        model.model, model.norm_stats, model.normalization_type = None, None, None
        model.sequence_length, model.use_wrist_image, model.no_norm, model.delta = 2, True, True, False
        model._save_request_log = lambda *args: None
        namespace["run_inference"] = lambda _model, payload, *args, **kwargs: {
            "predicted_actions": predict(payload["proprio"], payload["images"], payload["instruction"])
        }
    elif name == "DreamZero":
        model.video_history, model.native_dojo_action = 3, True
        model.inference_method = "lazy_joint_forward_causal"

        def infer(batch):
            payload = batch.obs
            state = np.concatenate([payload[f"state.{side}_{field}"] for side in ("left", "right")
                                    for field in ("arm_joint_position", "effector_position")], axis=-1)
            actions = predict(state, [payload[key] for key in ("video.top_head", "video.hand_left", "video.hand_right")],
                              payload["annotation.language.action_text"])
            return SimpleNamespace(act=actions), None

        model.policy = SimpleNamespace(lazy_joint_forward_causal=infer)
        model._decode_actions = lambda actions: unpack_robot_state(actions, "joint", ROBOT)
    else:
        def infer_batch(observations):
            return [unpack_robot_state(predict(pack_robot_state(obs, "joint", ROBOT),
                        [obs["vision"][camera]["color"] for camera in CAMERAS], obs["instruction"]), "joint", ROBOT)
                    for obs in observations]
        model._infer_batch = infer_batch
        model._infer = lambda obs: infer_batch([obs])[0]
    return model


class CachedPolicyBatchRoutingTests(unittest.TestCase):
    def test_normal_updates_reordering_and_subsets_preserve_each_observation(self):
        for name in NAMES:
            with self.subTest(policy=name):
                model = make_model(name)
                for step in (1, 2):
                    model.update_obs_batch([observation(2, 20 + step), observation(7, 70 + step)])
                    self.assertEqual(markers(model.get_action_batch()), [[20 + step, 21 + step], [70 + step, 71 + step]])
                    self.assertEqual(markers(model.get_action_batch([7, 2])), [[70 + step, 71 + step], [20 + step, 21 + step]])
                    self.assertEqual(markers(model.get_action_batch([7])), [[70 + step, 71 + step]])

    def test_missing_ids_use_row_positions_instead_of_overwriting_env_zero(self):
        for name in NAMES:
            with self.subTest(policy=name):
                model = make_model(name)
                model.update_obs_batch([observation(1, 10)])
                model.update_obs_batch([observation(None, 20), observation(None, 30)])
                self.assertEqual(markers(model.get_action_batch()), [[20, 21], [30, 31]])
                self.assertEqual(markers(model.get_action_batch([1, 0])), [[30, 31], [20, 21]])

    def test_only_latest_ids_are_available_and_invalid_requests_do_not_infer(self):
        for name in NAMES:
            with self.subTest(policy=name):
                model = make_model(name)
                model.update_obs_batch([observation(2, 20), observation(7, 70)])
                model.update_obs_batch([observation(7, 71)])
                self.assertEqual(markers(model.get_action_batch()), [[71, 72]])
                for requested in ([7, 2], [7, 99], [7, 7]):
                    before = list(model.calls)
                    with self.assertRaises(ValueError):
                        model.get_action_batch(requested)
                    self.assertEqual(model.calls, before)
                self.assertEqual(model.get_action_batch([]), [])

    def test_duplicate_updates_leave_previous_batch_and_history_unchanged(self):
        for name in NAMES:
            with self.subTest(policy=name):
                model = make_model(name)
                model.update_obs_batch([observation(2, 20), observation(7, 70)])
                history = history_markers(model) if name == "DreamZero" else None
                with self.assertRaises(ValueError):
                    model.update_obs_batch([observation(2, 21), observation(2, 22)])
                self.assertEqual(markers(model.get_action_batch()), [[20, 21], [70, 71]])
                if history is not None:
                    self.assertEqual(history_markers(model), history)

    def test_encoding_failure_is_atomic_including_dreamzero_history_and_retry(self):
        for name in NAMES[:-1]:
            with self.subTest(policy=name):
                model = make_model(name)
                model.update_obs_batch([observation(2, 20), observation(7, 70)])
                history = history_markers(model) if name == "DreamZero" else None
                malformed = observation(7, 71)
                malformed["state"]["right_arm_joint_state"] = np.zeros(3)
                with self.assertRaises(ValueError):
                    model.update_obs_batch([observation(2, 21), malformed])
                self.assertEqual(markers(model.get_action_batch()), [[20, 21], [70, 71]])
                if history is not None:
                    self.assertEqual(history_markers(model), history)
                model.update_obs_batch([observation(2, 21), observation(7, 71)])
                self.assertEqual(markers(model.get_action_batch()), [[21, 22], [71, 72]])
                if name == "DreamZero":
                    for key in model._frame_buffers[2]:
                        self.assertEqual(history_markers(model)[2][key], [20, 21])
                        self.assertEqual(history_markers(model)[7][key], [70, 71])

    def test_failed_update_cannot_relabel_old_observations_as_new_ids(self):
        for name in NAMES:
            with self.subTest(policy=name):
                model = make_model(name)
                model.update_obs_batch([observation(2, 20), observation(7, 70)])
                malformed = observation(4, 40)
                if name == "EWAM":
                    malformed["env_idx"] = "invalid"
                else:
                    malformed["state"]["right_arm_joint_state"] = np.zeros(3)
                with self.assertRaises(ValueError):
                    model.update_obs_batch([observation(3, 30), malformed])
                self.assertEqual(markers(model.get_action_batch()), [[20, 21], [70, 71]])
                with self.assertRaises(ValueError):
                    model.get_action_batch([3])

    def test_empty_update_reset_and_single_observation_follow_the_current_batch(self):
        for name in NAMES:
            with self.subTest(policy=name):
                model = make_model(name)
                model.update_obs(observation(7, 70))
                self.assertEqual(markers([model.get_action()]), [[70, 71]])
                model.update_obs_batch([])
                self.assertEqual(model.get_action_batch(), [])
                with self.assertRaises(RuntimeError):
                    model.get_action()
                with self.assertRaises(ValueError):
                    model.get_action_batch([7])
                model.update_obs(observation(None, 20))
                self.assertEqual(markers(model.get_action_batch()), [[20, 21]])
                with redirect_stdout(io.StringIO()):
                    model.reset()
                self.assertEqual(model.get_action_batch(), [])
                with self.assertRaises(ValueError):
                    model.get_action_batch([0])
                if name == "DreamZero":
                    self.assertEqual(model._frame_buffers, {})

    def test_dreamzero_retains_isolated_history_when_an_environment_reappears(self):
        model = make_model("DreamZero")
        model.update_obs_batch([observation(2, 20), observation(7, 70)])
        model.update_obs_batch([observation(7, 71)])
        model.update_obs_batch([observation(2, 21)])
        self.assertEqual(markers(model.get_action_batch()), [[21, 22]])
        for key in model._frame_buffers[2]:
            self.assertEqual(history_markers(model)[2][key], [20, 21])
            self.assertEqual(history_markers(model)[7][key], [70, 71])

    def test_ewam_normalizes_ids_before_duplicate_validation(self):
        model = make_model("EWAM")
        model.update_obs_batch([observation("2", 20), observation("7", 70)])
        self.assertEqual(markers(model.get_action_batch(["7", "2"])), [[70, 71], [20, 21]])
        with self.assertRaises(ValueError):
            model.update_obs_batch([observation(2, 21), observation("2", 22)])
        with self.assertRaises(ValueError):
            model.get_action_batch([2, "2"])
        self.assertEqual(markers(model.get_action_batch()), [[20, 21], [70, 71]])


if __name__ == "__main__":
    unittest.main()
