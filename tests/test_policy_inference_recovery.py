"""Exercise real preprocessing and history commits without model weights or CUDA."""
from __future__ import annotations

import ast
import collections
from contextlib import nullcontext
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from XPolicyLab.utils.action_chunks import validate_action_batch
from XPolicyLab.utils.batch_indices import observation_env_ids, observation_rows, validate_observed_ids
from XPolicyLab.utils.process_data import pack_robot_state, unpack_robot_state

POLICIES = Path(__file__).resolve().parents[1] / "policy"
DIMS = {"arm_dim": [1, 1], "ee_dim": [1, 1]}
FIELDS = ("left_arm_joint_state", "left_ee_joint_state", "right_arm_joint_state", "right_ee_joint_state")


class Tensor(np.ndarray):
    """CPU array double for tensor transfers; numerical preprocessing stays real."""
    def __new__(cls, value):
        return np.asarray(value).view(cls)

    def cpu(self):
        return self

    def detach(self):
        return self

    def float(self):
        return self.astype(np.float32)

    def to(self, dtype=None, **kwargs):
        return self.astype(dtype) if dtype in (np.float32, np.uint8) else self

    def numpy(self):
        return np.asarray(self)

    def clone(self):
        return self.copy()


def namespace():
    return dict(np=np, Image=Image, collections=collections,
                torch=SimpleNamespace(from_numpy=Tensor, inference_mode=nullcontext,
                                      float32=np.float32, uint8=np.uint8, bfloat16="bf16"),
                pack_robot_state=pack_robot_state, unpack_robot_state=unpack_robot_state,
                observation_env_ids=observation_env_ids, observation_rows=observation_rows,
                validate_observed_ids=validate_observed_ids, validate_action_batch=validate_action_batch)


def load_methods(path, names, ns, helpers=(), class_name="Model"):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    standalone = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in helpers]
    for method in methods:
        method.decorator_list = []
    code = ast.Module([ast.ImportFrom("__future__", [ast.alias("annotations")], 0)] + standalone + methods, [])
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), ns)
    return type(class_name, (), {n.name: ns[n.name] for n in methods})()


def observation(env_id, value):
    return dict(env_idx=env_id, instruction="pick", state={k: np.array([value], np.float32) for k in FIELDS},
                vision={camera: {"color": np.full((5, 6, 3), [40, 90, 160], np.uint8)}
                        for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist")})


def packed(chunks):
    return [[np.concatenate([step[k] for k in FIELDS]).tolist() for step in chunk] for chunk in chunks]


def lingbot_model():
    ns = namespace()
    normalizer = load_methods(POLICIES / "LingBot_VLA/lingbot_vla/lingbotvla/data/vla_data/transform.py",
                              {"normalize"}, ns, class_name="Normalizer")
    normalizer.norm_stats = {"observation.state": {"mean": np.zeros(4), "std": np.ones(4)}}
    normalizer.norm_type = {"observation.state": "meanstd"}
    ns.update(prepare_state=lambda config, obs: obs["state"],
              prepare_language=lambda *args: (None, None),
              prepare_images=lambda config, processor, obs: (obs["image"], None, None))
    model = load_methods(POLICIES / "LingBot_VLA/model.py",
                         {"update_obs", "update_obs_batch", "get_action", "get_action_batch", "infer", "resize_image", "reset"},
                         ns, helpers={"encode_obs", "extract_image", "extract_prompt"})
    model.data_config, model.config = SimpleNamespace(img_size=8), SimpleNamespace()
    model.action_type, model.robot_action_dim_info, model.default_prompt = "joint", DIMS, "pick"
    model.joint_max_dim = model.action_dim = 4
    model.chunk_size, model.use_bf16 = 2, False
    model.language_tokenizer = model.image_processor = None
    model.inputs, model.fail_on = [], None

    def select_action(obs, *args):
        model.inputs.append(copy.deepcopy(obs))
        if len(model.inputs) == model.fail_on:
            raise RuntimeError("temporary forward failure")
        return {"action": Tensor(np.tile(obs["state"], (2, 1)))}

    model.vla = SimpleNamespace(normalizer=normalizer, select_action=select_action)
    model.reset()
    return model


def openvla_model():
    ns = namespace()
    helper_path = POLICIES / "OpenVLA_OFT/openvla_oft/experiments/robot/openvla_utils.py"
    tree = ast.parse(helper_path.read_text())
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
               {"get_vla_action", "normalize_proprio", "prepare_images_for_vla", "check_image_format"}]
    ns.update(DEVICE="cpu", OPENVLA_IMAGE_SIZE=4,
              NormalizationType=SimpleNamespace(BOUNDS="bounds", BOUNDS_Q99="q99"),
              ACTION_PROPRIO_NORMALIZATION_TYPE="bounds")
    code = ast.Module([ast.ImportFrom("__future__", [ast.alias("annotations")], 0)] + helpers, [])
    exec(compile(ast.fix_missing_locations(code), str(helper_path), "exec"), ns)
    model = load_methods(POLICIES / "OpenVLA_OFT/model.py",
                         {"get_action", "get_action_batch", "infer", "reset"}, ns)
    model.cfg = SimpleNamespace(num_images_in_input=1, use_proprio=True, use_film=False,
                                unnorm_key="test", center_crop=False)
    model.action_type, model.robot_action_dim_info = "joint", DIMS
    model.action_head, model.proprio_projector = object(), None
    model.inputs, model.fail_on = [], None

    class Inputs(dict):
        def to(self, *args, **kwargs):
            return self

    model.processor = lambda *args: Inputs()

    def predict_action(**kwargs):
        state = kwargs["proprio"].copy()
        model.inputs.append(state)
        if len(model.inputs) == model.fail_on:
            raise RuntimeError("temporary forward failure")
        return np.tile(state, (2, 1)), None

    model.vla = SimpleNamespace(predict_action=predict_action, norm_stats={"test": {
        "proprio": {"min": np.zeros(4), "max": np.full(4, 10)}}})
    model._latest_env_idx_list = [7, 3]
    model.observation_window = [dict(full_image=np.full((4, 4, 3), 50, np.uint8),
                                     state=np.full(4, value, np.float32), instruction="pick") for value in (8, 2)]
    return model


class InferenceRecoveryTests(unittest.TestCase):
    def test_lingbot_preprocessing_keeps_cached_observations_on_success_and_failure(self):
        for fail_on in (None, 2):
            with self.subTest(fail_on=fail_on):
                model = lingbot_model()
                model.update_obs_batch([observation(7, 8), observation(3, 2)])
                original = copy.deepcopy(model.observation_window)
                model.fail_on = fail_on
                if fail_on:
                    with self.assertRaisesRegex(RuntimeError, "temporary forward"):
                        model.get_action_batch([7, 3])
                else:
                    first = packed(model.get_action_batch([7, 3]))
                model.fail_on = None
                result = packed(model.get_action_batch([7, 3]))
                np.testing.assert_allclose(result, np.array([[[8] * 4] * 2, [[2] * 4] * 2]) / (1 + 1e-6))
                if not fail_on:
                    self.assertEqual(first, result)
                for before, after in zip(original, model.observation_window):
                    for key in before:
                        np.testing.assert_equal(before[key], after[key])
                image = model.inputs[-1]["images"]["base_0_rgb"]
                self.assertEqual(image.shape, (3, 8, 8))
                np.testing.assert_array_equal(image[:, 0, 0], [40, 90, 160])
                model.reset()
                with self.assertRaises(AssertionError):
                    model.get_action_batch([7])
                model.update_obs(observation(9, 4))
                self.assertEqual(len(model.get_action()), 2)

    def test_openvla_normalizes_raw_state_once_even_after_partial_batch_failure(self):
        for fail_on in (None, 2):
            with self.subTest(fail_on=fail_on):
                model = openvla_model()
                model.fail_on = fail_on
                if fail_on:
                    with self.assertRaisesRegex(RuntimeError, "temporary forward"):
                        model.get_action_batch([7, 3])
                else:
                    first = packed(model.get_action_batch([7, 3]))
                model.fail_on = None
                result = packed(model.get_action_batch([7, 3]))
                np.testing.assert_allclose(result, [[[0.6] * 4] * 2, [[-0.6] * 4] * 2], atol=1e-6)
                if not fail_on:
                    self.assertEqual(first, result)
                for obs, value in zip(model.observation_window, (8, 2)):
                    np.testing.assert_array_equal(obs["state"], np.full(4, value))
                for index, state in enumerate(model.inputs):
                    np.testing.assert_allclose(state, 0.6 if index % 2 == 0 else -0.6, atol=1e-6)
                model.reset()
                with self.assertRaises(AssertionError):
                    model.get_action_batch([7])

    def test_g05_rejects_invalid_complete_batch_before_committing_any_history(self):
        for invalid in ("empty", "nan", "width", "missing_key", "unequal_horizons"):
            with self.subTest(invalid=invalid):
                ns = namespace()
                model = load_methods(POLICIES / "G05/model.py",
                                     {"update_obs_batch", "get_action_batch", "_predict_chunks", "_env_key_from_obs",
                                      "_update_history_buffers", "_format_action_chunk"}, ns, helpers={"_to_horizon_array"})
                model._history_buffers = {}
                model.processor = SimpleNamespace(num_obs_steps=2, shape_meta={
                    "images": [{"key": "head"}], "state": [{"key": "arm"}]})
                model._build_obs_accepts_buffers = True
                model.inference_batch_size = 1
                model.action_steps, model.action_type, model.robot_action_dim_info = 2, "joint", DIMS
                model._to_g05_raw_obs = lambda obs: {"images": {"head": obs["step"]}, "state": {"arm": obs["step"]}}
                model._explicit_history_buffers_from_obs = lambda *args: None
                model._build_obs_dict = lambda obs, processor, frame_buffers, state_buffers: list(frame_buffers["head"])
                calls, failure = [], [None]

                def infer(inputs):
                    calls.extend([[int(t) for t in history] for history in inputs])
                    result = {k: np.ones((2, 1), np.float32) for k in
                              ("left_arm", "left_gripper", "right_arm", "right_gripper")}
                    if failure[0] and len(calls) % 2 == 0:
                        if failure[0] in ("empty", "unequal_horizons"):
                            size = 0 if failure[0] == "empty" else 1
                            result = {k: v[:size] for k, v in result.items()}
                        elif failure[0] == "nan":
                            result["left_arm"][0, 0] = np.nan
                        elif failure[0] == "width":
                            result["left_arm"] = np.ones((2, 2), np.float32)
                        else:
                            del result["right_arm"]
                    return [result]

                model.inferencer = SimpleNamespace(infer=infer)
                model.update_obs_batch([dict(env_idx=i, step=1) for i in (7, 3)])
                baseline = packed(model.get_action_batch())
                model.update_obs_batch([dict(env_idx=i, step=2) for i in (7, 3)])
                failure[0] = invalid
                with self.assertRaises((ValueError, KeyError)):
                    model.get_action_batch()
                for entry in model._history_buffers.values():
                    self.assertEqual([int(t) for t in entry["frames"]["head"]], [1, 1])
                failure[0] = None
                self.assertEqual(packed(model.get_action_batch()), baseline)
                self.assertEqual(calls[-2:], [[1, 2], [1, 2]])


if __name__ == "__main__":
    unittest.main()
