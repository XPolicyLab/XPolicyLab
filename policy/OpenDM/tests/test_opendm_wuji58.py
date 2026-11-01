"""CPU contracts for the OpenDM wuji58 (tianji_wuji / ee) path; no model is loaded."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from XPolicyLab.policy.OpenDM import model as adapter
from XPolicyLab.policy.OpenDM.scripts.schema import (
    JOINT14_NAMES,
    contract_name,
    validate_source_info,
)
from XPolicyLab.utils import wuji58

WUJI = {"arm_dim": [7, 7], "ee_dim": [20, 20]}
ARX = {"arm_dim": [6, 6], "ee_dim": [1, 1]}


def _pose(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return np.concatenate([rng.uniform(-1, 1, 3), q if q[0] >= 0 else -q])


def _wuji_obs(rng):
    return {
        "state": {
            "left_ee_pose": _pose(rng),
            "left_ee_joint_state": rng.uniform(-0.4, 1.5, 20),
            "right_ee_pose": _pose(rng),
            "right_ee_joint_state": rng.uniform(-0.4, 1.5, 20),
        }
    }


class EmbodimentTest(unittest.TestCase):
    def test_wuji58_contract(self):
        emb = adapter.resolve_embodiment("tianji_wuji", "ee", WUJI)
        self.assertEqual(emb["name"], "wuji58")
        self.assertEqual(emb["action_dim"], 58)

    def test_joint14_contract_unchanged(self):
        emb = adapter.resolve_embodiment("arx_x5", "joint", ARX)
        self.assertEqual((emb["name"], emb["action_dim"]), ("joint14", 14))

    def test_rejected_combinations(self):
        with self.assertRaises(ValueError):
            adapter.resolve_embodiment("tianji_wuji", "joint", WUJI)  # joint space on the hand robot
        with self.assertRaises(ValueError):
            adapter.resolve_embodiment("arx_x5", "ee", ARX)  # gripper is not a 20-joint hand
        with self.assertRaises(ValueError):
            adapter.resolve_embodiment("piper", "joint", {"arm_dim": [7, 7], "ee_dim": [1, 1]})

    def test_state_desc(self):
        desc = adapter.build_state_desc(adapter.resolve_embodiment("tianji_wuji", "ee", WUJI))
        self.assertEqual(len(desc), 58)
        self.assertEqual(desc[:20], ["joint"] * 20)
        self.assertEqual(desc[20:29], ["eef"] * 9)
        self.assertEqual(desc[29:49], ["joint"] * 20)
        self.assertEqual(desc[49:], ["eef"] * 9)
        joint = adapter.build_state_desc(adapter.resolve_embodiment("arx_x5", "joint", ARX))
        self.assertEqual(joint, (["joint"] * 6 + ["gripper"]) * 2)


class StateAndActionTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.emb = adapter.resolve_embodiment("tianji_wuji", "ee", WUJI)
        self.frame = wuji58.WristFrame.from_name("te2")

    def test_state_is_58d_wuji_layout(self):
        obs = _wuji_obs(self.rng)
        state = adapter.build_state(obs, self.emb, "ee", self.frame)
        self.assertEqual(state.shape, (58,))
        self.assertEqual(state.dtype, np.float32)
        np.testing.assert_allclose(state, wuji58.state58_from_obs(obs["state"], self.frame), rtol=1e-6)
        np.testing.assert_allclose(state[:20], obs["state"]["left_ee_joint_state"], rtol=1e-6)

    def test_chunk_decodes_to_four_keys_against_the_observation(self):
        obs = _wuji_obs(self.rng)
        targets = [_wuji_obs(self.rng)["state"] for _ in range(50)]
        # per-frame deltas: each target vs. the wrist the previous target moved to
        refs = [obs["state"]] + targets[:-1]
        chunk = np.stack([wuji58.action58_from_target(t, r, self.frame) for t, r in zip(targets, refs)])
        actions = adapter.decode_action_chunk(chunk, obs, self.emb, "ee", self.frame, steps=25)
        self.assertEqual(len(actions), 25)
        for got, want in zip(actions, targets):
            self.assertEqual(set(got), set(wuji58.STATE_KEYS))
            for key in wuji58.STATE_KEYS:
                np.testing.assert_allclose(got[key], want[key], atol=1e-5)

    def test_wrong_width_chunk_rejected(self):
        with self.assertRaises(ValueError):
            adapter.decode_action_chunk(np.zeros((50, 14)), _wuji_obs(self.rng), self.emb, "ee", self.frame, 25)

    def test_joint14_path_still_unpacks(self):
        emb = adapter.resolve_embodiment("arx_x5", "joint", ARX)
        obs = {
            "state": {
                "left_arm_joint_state": np.zeros(6),
                "left_ee_joint_state": np.zeros(1),
                "right_arm_joint_state": np.zeros(6),
                "right_ee_joint_state": np.zeros(1),
            }
        }
        self.assertEqual(adapter.build_state(obs, emb, "joint", self.frame).shape, (14,))
        actions = adapter.decode_action_chunk(np.ones((50, 14)), obs, emb, "joint", self.frame, 25)
        self.assertEqual(len(actions), 25)
        self.assertEqual(actions[0]["left_arm_joint_state"].shape, (6,))


class WidthChecksTest(unittest.TestCase):
    def test_latent_width_must_hold_robot(self):
        adapter.check_latent_action_width(14, 32)
        adapter.check_latent_action_width(58, 64)
        with self.assertRaises(ValueError):
            adapter.check_latent_action_width(58, 32)

    def test_norm_stats_width(self):
        def stats(width):
            return {"mean": [0.0] * width, "std": [1.0] * width, "q01": [0.0] * width, "q99": [1.0] * width}

        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "legacy.json"
            legacy.write_text(json.dumps({"norm_stats": {"state": stats(14), "action": stats(14)}}))
            self.assertEqual(adapter.norm_stats_action_width(legacy), 14)
            multi = Path(tmp) / "multi.json"
            multi.write_text(
                json.dumps(
                    {
                        "default_robot_type": "Tianji Wuji",
                        "norm_stats": {"state": stats(58), "action": stats(58)},
                        "norm_stats_by_robot": {"Tianji Wuji": {"state": stats(58), "action": stats(58)}},
                    }
                )
            )
            self.assertEqual(adapter.norm_stats_action_width(multi), 58)
            mixed = Path(tmp) / "mixed.json"
            mixed.write_text(
                json.dumps(
                    {
                        "default_robot_type": "a",
                        "norm_stats": {"state": stats(14), "action": stats(14)},
                        "norm_stats_by_robot": {"a": {"action": stats(14)}, "b": {"action": stats(58)}},
                    }
                )
            )
            with self.assertRaises(ValueError):
                adapter.norm_stats_action_width(mixed)


class SourceSchemaTest(unittest.TestCase):
    def _info(self, names, width, **extra):
        feature = {"names": [names], "shape": [width]}
        return {"fps": 25, "features": {"observation.state": feature, "action": feature}, **extra}

    def test_contracts(self):
        self.assertEqual(contract_name("RoboDojo", "arx_x5", "joint"), "joint14")
        self.assertEqual(contract_name("RoboDojo", "tianji_wuji", "ee"), "wuji58")
        with self.assertRaises(ValueError):
            contract_name("RoboDojo", "tianji_wuji", "joint")

    def test_wuji58_export_validation(self):
        names = list(wuji58.MOTOR_NAMES)
        validate_source_info(self._info(names, 58, repr_mode=wuji58.REPR_MODE, robot="tianji_wuji"), "wuji58", 58)
        with self.assertRaises(ValueError):  # no repr_mode stamp
            validate_source_info(self._info(names, 58), "wuji58", 58)
        with self.assertRaises(ValueError):  # joint export offered for wuji58
            validate_source_info(self._info(JOINT14_NAMES, 14, repr_mode=wuji58.REPR_MODE), "wuji58", 58)
        with self.assertRaises(ValueError):  # wuji export offered for joint14
            validate_source_info(self._info(names, 58, repr_mode=wuji58.REPR_MODE), "joint14", 14)
        validate_source_info(self._info(JOINT14_NAMES, 14), "joint14", 14)
        with self.assertRaises(ValueError):
            validate_source_info({**self._info(JOINT14_NAMES, 14), "fps": 30}, "joint14", 14)


if __name__ == "__main__":
    unittest.main()
