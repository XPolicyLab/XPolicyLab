"""CPU contracts for the G05 WUJI 58-D hybrid path (no torch, no checkpoint).

Run from the repo root: ``.venv/bin/python -m pytest policy/G05/tests -q``.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import yaml

from XPolicyLab.policy.G05 import wuji58_layout as L
from XPolicyLab.utils import wuji58 as W

POLICY_DIR = Path(__file__).resolve().parents[1]
CONFIGS = POLICY_DIR / "G05" / "configs"


def _rand_quat(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def _obs_state(rng):
    return {
        "left_ee_pose": np.concatenate([rng.uniform(-1, 1, 3), _rand_quat(rng)]),
        "left_ee_joint_state": rng.uniform(-0.4, 1.5, 20),
        "right_ee_pose": np.concatenate([rng.uniform(-1, 1, 3), _rand_quat(rng)]),
        "right_ee_joint_state": rng.uniform(-0.4, 1.5, 20),
    }


class RawStateTest(unittest.TestCase):
    def test_keys_widths_and_layout(self):
        rng = np.random.default_rng(0)
        obs = _obs_state(rng)
        frame = W.WristFrame.te2()
        raw = L.build_raw_state(obs, frame)
        self.assertEqual(tuple(raw), L.WUJI58_KEYS)
        self.assertEqual({k: v.shape for k, v in raw.items()}, {"left_hand": (20,), "left_wrist": (9,), "right_hand": (20,), "right_wrist": (9,)})
        self.assertTrue(all(v.dtype == np.float32 for v in raw.values()))
        np.testing.assert_allclose(raw["left_hand"], obs["left_ee_joint_state"], rtol=1e-6)
        np.testing.assert_allclose(raw["right_hand"], obs["right_ee_joint_state"], rtol=1e-6)
        np.testing.assert_allclose(raw["left_wrist"], W.encode_wrist_state(obs["left_ee_pose"], frame), rtol=1e-5)
        np.testing.assert_allclose(np.concatenate(list(raw.values())), W.state58_from_obs(obs, frame), rtol=1e-5)

    def test_gripper_robot_is_rejected(self):
        rng = np.random.default_rng(1)
        obs = _obs_state(rng)
        obs["left_ee_joint_state"] = np.zeros(1)
        with self.assertRaises(ValueError):
            L.build_raw_state(obs, W.WristFrame.te2())


class ChunkTest(unittest.TestCase):
    def test_zero_delta_holds_pose_and_hands_pass_through(self):
        rng = np.random.default_rng(2)
        obs = _obs_state(rng)
        frame = W.WristFrame.te2()
        H = 5
        hold = np.concatenate([np.zeros(3), np.eye(3)[:2].reshape(6)])
        parts = {
            "left_hand": np.tile(obs["left_ee_joint_state"], (H, 1)),
            "left_wrist": np.tile(hold, (H, 1)),
            "right_hand": np.tile(obs["right_ee_joint_state"], (H, 1)),
            "right_wrist": np.tile(hold, (H, 1)),
        }
        out = L.decode_chunk(parts, obs, frame, max_steps=16)
        self.assertEqual(len(out), H)
        for step in out:
            self.assertEqual(set(step), set(W.STATE_KEYS))
            for key in W.STATE_KEYS:
                np.testing.assert_allclose(step[key], obs[key], atol=1e-5)

    def test_round_trip_against_targets_and_horizon_cap(self):
        rng = np.random.default_rng(3)
        obs = _obs_state(rng)
        frame = W.WristFrame.te2()
        targets = [_obs_state(rng) for _ in range(8)]
        # per-frame deltas (the converter convention): each target vs. the previous target's wrist
        refs = [obs] + targets[:-1]
        chunk = np.stack([W.action58_from_target(t, r, frame) for t, r in zip(targets, refs)])
        parts = {key: chunk[:, sl] for key, sl in L.WUJI58_SLICES.items()}
        np.testing.assert_allclose(L.chunk_from_parts(parts, 8), chunk, rtol=1e-6)
        out = L.decode_chunk(parts, obs, frame, max_steps=3)
        self.assertEqual(len(out), 3)
        for step, want in zip(out, targets):
            for key in W.STATE_KEYS:
                np.testing.assert_allclose(step[key], want[key], atol=1e-4)

    def test_missing_or_misshaped_parts_fail(self):
        with self.assertRaises(KeyError):
            L.chunk_from_parts({"left_hand": np.zeros((2, 20))}, 2)
        parts = {k: np.zeros((2, sl.stop - sl.start)) for k, sl in L.WUJI58_SLICES.items()}
        parts["left_wrist"] = np.zeros((2, 7))
        with self.assertRaises(ValueError):
            L.chunk_from_parts(parts, 2)


class ConfigTest(unittest.TestCase):
    def _load(self, rel):
        return yaml.safe_load((CONFIGS / rel).read_text())

    def test_parts_meta_matches_codec(self):
        cfg = self._load("data/parts_meta/wuji58.yaml")
        self.assertEqual(cfg["parts_meta"], {k: sl.stop - sl.start for k, sl in L.WUJI58_SLICES.items()})
        self.assertEqual(sum(cfg["parts_meta"].values()), W.WUJI58_DIM)
        self.assertEqual(list(cfg["merge_spec"]), list(L.WUJI58_KEYS))
        for group, alts in cfg["merge_spec"].items():
            self.assertEqual(alts, [group])

    def test_data_config_slices_cover_58(self):
        cfg = self._load("data/robodojo_wuji58.yaml")
        ds = cfg["embodiment_datasets"]["robodojo_wuji58"]
        self.assertEqual(ds["embodiment_type"], L.WUJI58_EMBODIMENT)
        for group, lerobot_key in (("action", "action"), ("state", "observation.state")):
            entries = ds["shape_meta"][group]
            self.assertEqual([e["key"] for e in entries], list(L.WUJI58_KEYS))
            cursor = 0
            for e in entries:
                self.assertEqual(e["lerobot_key"], lerobot_key)
                self.assertEqual(e["start_index"], cursor)
                self.assertEqual(e["raw_shape"], e["shape"])
                self.assertEqual((e["start_index"], e["start_index"] + e["raw_shape"]), (L.WUJI58_SLICES[e["key"]].start, L.WUJI58_SLICES[e["key"]].stop))
                cursor += e["raw_shape"]
            self.assertEqual(cursor, W.WUJI58_DIM)
        self.assertEqual([e["key"] for e in ds["shape_meta"]["images"]], ["cam_high", "cam_left_wrist", "cam_right_wrist"])
        proc = cfg["processors"]["robodojo_wuji58"]["action_state_merger"]
        self.assertEqual((proc["max_action_shape_meta"], proc["max_state_shape_meta"]), (58, 58))

    def test_tokenizer_matches_parts(self):
        tok = self._load("tokenizer/actioncodec_wuji58.yaml")["vq_config"]
        parts = self._load("data/parts_meta/wuji58.yaml")["parts_meta"]
        self.assertEqual(tok["parts_meta"], parts)
        self.assertGreaterEqual(tok["model_arch"]["max_component_dim"], max(parts.values()))
        self.assertEqual(tok["model_arch"]["max_component_dim"], 20)

    def test_task_config_dims_and_references(self):
        task = self._load("task/robodojo_tianji_wuji_ee.yaml")
        arch = task["model"]["model_arch"]
        self.assertEqual((arch["action_dim"], arch["proprio_dim"]), (58, 58))
        defaults = {k: v for d in task["defaults"] if isinstance(d, dict) for k, v in d.items()}
        self.assertEqual(defaults["override /data"], "robodojo_wuji58")
        self.assertEqual(defaults["override /tokenizer"], "actioncodec_wuji58")
        merger = task["model"]["processor"]["action_state_merger"]
        for key in ("max_action_shape_meta", "max_state_shape_meta", "merge_spec"):
            self.assertIn("parts_meta/wuji58.yaml", merger[key])
        self.assertEqual(task["model"]["processor"]["norm_default_mode"], "q01/q99")
        self.assertIsNone(task["model"]["processor"]["norm_exception_mode"])
        self.assertEqual(task["tokenizer"]["vq_config"]["rule_based_key_patterns"], [])
        include = self._load("task/robodojo/robodojo_tianji_wuji_ee.yaml")
        self.assertIn("/task/robodojo_tianji_wuji_ee", include["defaults"])


if __name__ == "__main__":
    unittest.main()
