"""CPU contract test for the WUJI 58-D hybrid path of the InternW0_delta adapter.

Runs without torch or any weights: it exercises ``StatefulWAMAdapter`` (pure numpy)
with ``allow_dummy_policy`` and patches the robot registry lookup, which needs the
parent workspace's ``env_cfg/`` tree.

    .venv/bin/python -m pytest policy/InternW0_delta/test_wuji58_adapter.py -q
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from XPolicyLab.policy.InternW0_delta import _adapter_base as base
from XPolicyLab.utils import wuji58

WUJI = {"arm_dim": [7, 7], "ee_dim": [20, 20]}
X5 = {"arm_dim": [6, 6], "ee_dim": [1, 1]}


class _Adapter(base.StatefulWAMAdapter):
    def _load_real_policy(self) -> None:  # never reached with allow_dummy_policy
        raise AssertionError("real policy must not load in this test")


def _rand_pose(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return np.concatenate([rng.uniform(-1, 1, 3), q if q[0] >= 0 else -q]).astype(np.float32)


def _obs(rng, env_idx=None):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    obs = {
        "vision": {
            "cam_head": {"color": image},
            "cam_left_wrist": {"color": image},
            "cam_right_wrist": {"color": image},
        },
        "state": {
            "left_ee_pose": _rand_pose(rng),
            "left_ee_joint_state": rng.uniform(-0.4, 1.5, 20).astype(np.float32),
            "right_ee_pose": _rand_pose(rng),
            "right_ee_joint_state": rng.uniform(-0.4, 1.5, 20).astype(np.float32),
        },
        "instruction": "pour the water",
    }
    if env_idx is not None:
        obs["env_idx"] = env_idx
    return obs


def _make(dims, action_type="ee", **extra):
    cfg = {
        "action_type": action_type,
        "env_cfg_type": "tianji_wuji",
        "allow_dummy_policy": True,
        "replan_steps": 4,
        "action_horizon": 8,
        **extra,
    }
    with patch.object(base, "get_robot_action_dim_info", return_value=dims):
        return _Adapter(cfg)


class Wuji58AdapterTest(unittest.TestCase):
    def test_contract_detection_and_rejection(self):
        model = _make(WUJI)
        self.assertTrue(model.wuji58)
        self.assertEqual(model.action_dim, 58)
        self.assertEqual(model.gripper_slices, [])
        x5 = _make(X5, action_type="joint")
        self.assertFalse(x5.wuji58)
        self.assertEqual(x5.action_dim, 14)
        with self.assertRaisesRegex(ValueError, "Supported contracts"):
            _make(WUJI, action_type="joint")  # 54-D joint hands are not a trained contract
        with self.assertRaisesRegex(ValueError, "Supported contracts"):
            _make(X5, action_type="ee")
        with self.assertRaises(ValueError):
            _make(WUJI, wrist_frame="base")

    def test_state_packs_to_58_in_wuji_layout(self):
        rng = np.random.default_rng(0)
        model = _make(WUJI)
        obs = _obs(rng)
        adapted = model._adapt_obs(obs)
        vec = adapted["joint_action"]["vector"]
        self.assertEqual(vec.shape, (58,))
        self.assertEqual(vec.dtype, np.float32)
        np.testing.assert_allclose(vec, wuji58.state58_from_obs(obs["state"], model.wrist_frame), rtol=1e-6)
        np.testing.assert_array_equal(vec[wuji58.LEFT_HAND], obs["state"]["left_ee_joint_state"])
        self.assertEqual(set(adapted), {"observation", "joint_action"})

    def test_dummy_hold_position_single_env(self):
        rng = np.random.default_rng(1)
        model = _make(WUJI)
        obs = _obs(rng)
        model.update_obs(obs)
        actions = model.get_action()
        self.assertEqual(len(actions), 4)
        for action in actions:
            self.assertEqual(set(action), set(wuji58.STATE_KEYS))
            for key in wuji58.STATE_KEYS:
                self.assertEqual(action[key].dtype, np.float32)
                np.testing.assert_allclose(action[key], obs["state"][key], atol=1e-5)

    def test_dummy_hold_position_batched(self):
        rng = np.random.default_rng(2)
        model = _make(WUJI)
        obs_list = [_obs(rng, env_idx=3), _obs(rng, env_idx=1)]
        model.update_obs_batch(obs_list)
        chunks = model.get_action_batch([3, 1])
        self.assertEqual([len(c) for c in chunks], [4, 4])
        for chunk, obs in zip(chunks, obs_list):
            for key in wuji58.STATE_KEYS:
                np.testing.assert_allclose(chunk[0][key], obs["state"][key], atol=1e-5)
        model.reset()
        self.assertEqual(model.env_obs, {})
        with self.assertRaises(ValueError):
            model.get_action()

    def test_predicted_chunk_decodes_against_its_observation(self):
        rng = np.random.default_rng(3)
        model = _make(WUJI)
        obs = _obs(rng)
        target = _obs(rng)["state"]
        chunk = np.stack([wuji58.action58_from_target(target, obs["state"], model.wrist_frame)] * 4)
        # Simulate the real path: dummy off, session returning our chunk.
        model.allow_dummy_policy = False

        class _Session:
            pending_model_actions = []

            def get_action(self, payload):
                assert payload["observation"]["joint_action"]["vector"].shape == (58,)
                return chunk

        decoded = model._predict(_Session(), model._adapt_obs(obs), "x", model._raw_state(obs))
        self.assertEqual(len(decoded), 4)
        for key in wuji58.STATE_KEYS:
            np.testing.assert_allclose(decoded[0][key], target[key], atol=1e-5)

    def test_wrong_hand_width_fails(self):
        rng = np.random.default_rng(4)
        model = _make(WUJI)
        obs = _obs(rng)
        obs["state"]["left_ee_joint_state"] = np.zeros(1, dtype=np.float32)
        with self.assertRaises(ValueError):
            model.update_obs(obs)
        del obs["state"]["left_ee_joint_state"]
        with self.assertRaises(KeyError):
            model.update_obs(obs)


class StatsWidthTest(unittest.TestCase):
    """The load-time guard in model.py, tested without importing torch."""

    def _check(self, width, expected):
        import importlib
        import sys
        import types

        # model.py imports torch/torchvision/omegaconf at module level; stub them so the
        # pure-python helpers can be exercised in this venv.
        stubs = {}
        for name in ("torch", "torchvision", "torchvision.transforms", "torchvision.transforms.functional", "omegaconf"):
            if name not in sys.modules:
                stubs[name] = sys.modules[name] = types.ModuleType(name)
        stubs.get("torchvision", sys.modules["torchvision"]).transforms = sys.modules["torchvision.transforms"]
        sys.modules["torchvision.transforms"].functional = sys.modules["torchvision.transforms.functional"]
        sys.modules["omegaconf"].OmegaConf = object
        try:
            module_name = "XPolicyLab.policy.InternW0_delta.model"
            sys.modules.pop(module_name, None)
            module = importlib.import_module(module_name)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "dataset_stats.json"
                path.write_text(json.dumps({"action": {"default": {"global_min": [0.0] * width}}}))
                module._require_stats_width(path, expected)
        finally:
            sys.modules.pop("XPolicyLab.policy.InternW0_delta.model", None)
            for name in stubs:
                sys.modules.pop(name, None)

    def test_matching_width_passes(self):
        self._check(58, 58)
        self._check(14, 14)

    def test_mismatch_names_expected_width(self):
        with self.assertRaisesRegex(ValueError, "14-D action stats.*expects 58-D"):
            self._check(14, 58)


if __name__ == "__main__":
    unittest.main()
