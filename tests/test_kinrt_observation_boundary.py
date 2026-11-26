"""KinRT routing through real RGB encoding, stacking and action unpacking."""
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

# Load native dependencies before patch.dict restores the module registry.
from XPolicyLab.utils import process_data


ROOT = Path(__file__).resolve().parents[1]


def observation(env_id, marker):
    images = {}
    for index, camera in enumerate(("cam_head", "cam_left_wrist", "cam_right_wrist")):
        image = np.full((4, 5, 3), [marker, marker + index, 1], dtype=np.uint8)
        image.flags.writeable = False
        images[camera] = {"color": image}
    return {
        "env_idx": env_id,
        "state": {"arm_joint_state": np.array([marker, marker + 1], dtype=np.float32),
                  "ee_joint_state": np.array([marker + 2], dtype=np.float32)},
        "vision": images, "instruction": str(marker),
    }


class KinRTObservationBoundaryTests(unittest.TestCase):
    def setUp(self):
        modules = {f"openpi.{name}": Mock() for name in ("policies", "shared", "training")}
        with patch.dict(sys.modules, modules):
            cls = runpy.run_path(str(ROOT / "policy/KinRT/model.py"))["Model"]
        self.model = cls.__new__(cls)
        self.model.action_type = "joint"
        self.model.robot_action_dim_info = {"arm_dim": [2], "ee_dim": [1]}
        self.model.action_chunk_size = 2
        self.calls = []

        def predict(obs):
            self.calls.append(obs)
            marker = obs["state"][0]
            self.assertEqual(obs["prompt"], str(int(marker)))
            for index, camera in enumerate(("cam_high", "cam_left_wrist", "cam_right_wrist")):
                self.assertEqual(obs["images"][camera].shape, (3, 4, 5))
                np.testing.assert_array_equal(obs["images"][camera][:, 0, 0], [marker, marker + index, 1])
            return {"actions": np.stack([obs["state"] + offset for offset in range(3)])}

        self.model.policy = SimpleNamespace(infer=predict)
        self.model.reset()

    def assert_markers(self, chunks, expected):
        self.assertEqual(len(chunks), len(expected))
        for chunk, marker in zip(chunks, expected):
            self.assertEqual(len(chunk), 2)
            for offset, action in enumerate(chunk):
                np.testing.assert_array_equal(action["arm_joint_state"], [marker + offset, marker + offset + 1])
                np.testing.assert_array_equal(action["ee_joint_state"], [marker + offset + 2])

    def test_normal_updates_and_reordered_subset_requests_keep_modalities_aligned(self):
        for offset in [0, 1]:
            self.model.update_obs_batch([observation(2, 20 + offset), observation(7, 70 + offset)])
            self.assert_markers(self.model.get_action_batch(), [20 + offset, 70 + offset])
            self.assert_markers(self.model.get_action_batch([7, 2]), [70 + offset, 20 + offset])
            self.assert_markers(self.model.get_action_batch([7]), [70 + offset])
            self.assert_markers([self.model.get_action()], [20 + offset])

    def test_failed_camera_encoding_and_stacking_keep_previous_observations_and_ids(self):
        self.model.update_obs_batch([observation(2, 20), observation(7, 70)])
        original = self.model.observation_window
        for kind in ["missing_camera", "incompatible_shapes"]:
            bad = observation(4, 40)
            if kind == "missing_camera":
                del bad["vision"]["cam_head"]
            else:
                bad["vision"]["cam_head"]["color"] = np.zeros((6, 5, 3), dtype=np.uint8)
            with self.subTest(kind=kind), self.assertRaises((KeyError, ValueError)):
                self.model.update_obs_batch([observation(3, 30), bad])
            self.assertIs(self.model.observation_window, original)
            self.assertEqual(self.model._latest_env_idx_list, [2, 7])
            self.assert_markers(self.model.get_action_batch(), [20, 70])

    def test_invalid_ids_do_not_infer_or_replace_the_batch(self):
        self.model.update_obs_batch([observation(2, 20), observation(7, 70)])
        for ids in [[98, 99], [2, 2]]:
            with self.assertRaises(ValueError):
                self.model.get_action_batch(ids)
        self.assertEqual(self.calls, [])
        with self.assertRaises(ValueError):
            self.model.update_obs_batch([observation(2, 30), observation(2, 40)])
        self.assert_markers(self.model.get_action_batch(), [20, 70])

    def test_empty_batch_and_reset_do_not_expose_previous_observations(self):
        self.model.update_obs(observation(7, 70))
        self.model.update_obs_batch([])
        self.assertEqual(self.model.get_action_batch(), [])
        with self.assertRaises(RuntimeError):
            self.model.get_action()
        with self.assertRaises(ValueError):
            self.model.get_action_batch([7])
        self.model.reset()
        with self.assertRaises(RuntimeError):
            self.model.get_action_batch()
        self.model.update_obs(observation(3, 30))
        self.assert_markers([self.model.get_action()], [30])


if __name__ == "__main__":
    unittest.main()
