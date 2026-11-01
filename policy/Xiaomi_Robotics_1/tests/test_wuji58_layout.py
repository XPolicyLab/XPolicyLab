"""CPU contracts for the XR-1 WUJI 58-D slot layout (no torch, no checkpoint)."""

import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from XPolicyLab.policy.Xiaomi_Robotics_1 import wuji58_layout as L
from XPolicyLab.utils import wuji58 as W

POLICY_DIR = Path(__file__).resolve().parents[1]
XR1_DIR = POLICY_DIR / "xiaomi_robotics_1" / "xr1"


def _rand_quat(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def _rand_pose(rng):
    return np.concatenate([rng.uniform(-1, 1, 3), _rand_quat(rng)])


def _obs_state(rng):
    return {
        "left_ee_pose": _rand_pose(rng),
        "left_ee_joint_state": rng.uniform(-0.4, 1.5, 20),
        "right_ee_pose": _rand_pose(rng),
        "right_ee_joint_state": rng.uniform(-0.4, 1.5, 20),
    }


def _plural(rows):
    return {k + "s": np.stack([r[k] for r in rows]) for k in W.STATE_KEYS}


class RobotResolutionTest(unittest.TestCase):
    def test_registry_fallback_detects_wuji58(self):
        with patch("XPolicyLab.utils.process_data.get_robot_action_dim_info", side_effect=FileNotFoundError):
            self.assertTrue(L.is_wuji58_run("tianji_wuji", "ee"))
            self.assertFalse(L.is_wuji58_run("tianji_wuji", "joint"))
            self.assertFalse(L.is_wuji58_run("arx_x5", "ee"))
            with self.assertRaises(KeyError):
                L.is_wuji58_run("no_such_robot", "ee")


class InferenceLayoutTest(unittest.TestCase):
    def test_compose_state_and_mask_shapes(self):
        rng = np.random.default_rng(0)
        frame = W.WristFrame.te2()
        obs = _obs_state(rng)
        state = L.compose_state58(obs, frame)
        self.assertEqual(state.shape, (1, 60))
        self.assertEqual(state.dtype, np.float32)
        np.testing.assert_allclose(state[0, :58], W.state58_from_obs(obs, frame), rtol=1e-6)
        np.testing.assert_array_equal(state[0, 58:], 0)
        mask = L.action_mask58(30)
        self.assertEqual(mask.shape, (30, 60))
        self.assertEqual(int(mask.sum()), 30 * 58)
        self.assertTrue((mask[:, 58:] == 0).all())
        temporal = L.action_mask58(5, [1, 1, 0, 0, 0])
        self.assertEqual(int(temporal.sum()), 2 * 58)

    def test_decode_zero_delta_holds_pose(self):
        rng = np.random.default_rng(1)
        frame = W.WristFrame.te2()
        obs = _obs_state(rng)
        hold = np.zeros((3, 60))
        hold[:, L.WUJI_SLOTS] = W.pack58(
            obs["left_ee_joint_state"],
            np.concatenate([np.zeros(3), np.eye(3)[:2].reshape(6)]),
            obs["right_ee_joint_state"],
            np.concatenate([np.zeros(3), np.eye(3)[:2].reshape(6)]),
        )
        for action in L.decode_actions58(hold, obs, frame):
            for key in W.STATE_KEYS:
                np.testing.assert_allclose(action[key], obs[key], atol=1e-6)

    def test_decode_rejects_wrong_width(self):
        with self.assertRaises(ValueError):
            L.decode_actions58(np.zeros((2, 58)), _obs_state(np.random.default_rng(2)), W.WristFrame.te2())


class DatasetEncodingTest(unittest.TestCase):
    def test_encode_and_chunk_match_shared_codec(self):
        rng = np.random.default_rng(3)
        frame = W.WristFrame.te2()
        states = [_obs_state(rng) for _ in range(4)]
        targets = [_obs_state(rng) for _ in range(4)]
        state58, target58 = L.encode_episode(_plural(states), _plural(targets), frame)
        self.assertEqual(state58.shape, (4, 58))
        for t in range(4):
            np.testing.assert_allclose(state58[t], W.state58_from_obs(states[t], frame))
        # chunk starting at frame 1: every wrist delta is relative to the frame-1 wrist
        chunk = L.chunk_actions(state58[1], target58[1:4])
        for k, t in enumerate(range(1, 4)):
            np.testing.assert_allclose(chunk[k], W.action58_from_target(targets[t], states[1], frame), atol=1e-9)
        # and the shared chunk decoder inverts it against the frame-1 observation
        decoded = W.decode_action_chunk58(chunk, states[1], frame, reference="chunk_start")
        for k, t in enumerate(range(1, 4)):
            for key in W.STATE_KEYS:
                np.testing.assert_allclose(decoded[k][key], targets[t][key], atol=1e-5)

    def test_training_sample_pads_and_masks(self):
        rng = np.random.default_rng(4)
        frame = W.WristFrame.te2()
        states = [_obs_state(rng) for _ in range(3)]
        targets = [_obs_state(rng) for _ in range(3)]
        state58, target58 = L.encode_episode(_plural(states), _plural(targets), frame)
        traj = {"proprios": {"wuji58_state": state58.tolist()}, "actions": {"wuji58_target": target58.tolist()}}
        state, action, mask = L.training_sample(traj, frame=1, steps=2, action_length=5)
        self.assertEqual(state.shape, (1, 60))
        self.assertEqual(action.shape, (5, 60))
        self.assertEqual(mask.shape, (5, 60))
        np.testing.assert_array_equal(mask[:2, :58], 1)
        np.testing.assert_array_equal(mask[2:], 0)
        np.testing.assert_array_equal(action[2], action[1])  # tail repeats the last row
        np.testing.assert_array_equal(action[:, 58:], 0)
        with self.assertRaises(IndexError):
            L.training_sample(traj, frame=2, steps=2, action_length=5)

    def test_stats_builder(self):
        rng = np.random.default_rng(5)
        frame = W.WristFrame.world()
        T, L_ = 40, 6
        states = [_obs_state(rng) for _ in range(T)]
        targets = [_obs_state(rng) for _ in range(T)]
        state58, target58 = L.encode_episode(_plural(states), _plural(targets), frame)
        actions = L.episode_action_samples(state58, target58, L_)
        samples = L.episode_state_samples(state58, L_)
        self.assertEqual(actions.shape, (T - L_ + 1, L_, 60))
        self.assertEqual(samples.shape, (T - L_ + 1, 60))
        mean, std, q01, q99 = L.compute_stats58(actions, samples)
        self.assertEqual(mean.shape, (L_, 60))
        self.assertEqual(std.shape, (L_, 60))
        self.assertEqual(q01.shape, (1, 60))
        self.assertTrue((std > 0).all())
        # padding slots: identity gaussian and 0/0 quantiles
        np.testing.assert_array_equal(mean[:, 58:], 0)
        np.testing.assert_array_equal(std[:, 58:], 1)
        np.testing.assert_array_equal(q01[:, 58:], 0)
        np.testing.assert_array_equal(q99[:, 58:], 0)
        self.assertTrue((q99[0, :58] > q01[0, :58]).all())
        # the loader's validators accept them (mibot.utils.io needs torch; skip if absent)
        if importlib.util.find_spec("torch") is not None and importlib.util.find_spec("PIL") is not None:
            import sys

            sys.path.insert(0, str(XR1_DIR))
            from mibot.utils.io import validate_quantiles, validate_stats

            validate_stats(mean, std, L_)
            validate_quantiles(q01, q99)


class RegistryTest(unittest.TestCase):
    def test_tianji_wuji_registered(self):
        registry = json.loads((Path(W.__file__).resolve().parent / "robot" / "_robot_info.json").read_text())
        self.assertEqual(registry["tianji_wuji"], {"arm_dim": [7, 7], "ee_dim": [20, 20]})


if __name__ == "__main__":
    unittest.main()
