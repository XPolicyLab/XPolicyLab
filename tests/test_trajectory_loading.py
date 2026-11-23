"""Exercise single-frame trajectories through the public loading and conversion helpers."""
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from XPolicyLab.utils.load_file import load_hdf5
from XPolicyLab.utils.process_data import decode_image_bit, encode_image_bit, pack_robot_state


class TrajectoryLoadingTests(unittest.TestCase):
    def test_single_frame_retains_time_and_field_axes(self):
        rgb = np.full((8, 12, 3), [220, 30, 10], np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.hdf5"
            with h5py.File(path, "w") as file:
                file["state/arm_joint_states"] = np.arange(6).reshape(1, 6)
                file["state/ee_joint_states"] = [[0.5]]
                file["vision/head/colors"] = np.array([encode_image_bit(rgb)])
                file["instruction"] = "pick up the cup"
                file["seed"] = 42
            episode = load_hdf5(path)

        self.assertEqual(episode["instruction"], "pick up the cup")
        self.assertIsInstance(episode["seed"], int)
        self.assertEqual(episode["state"]["ee_joint_states"].shape, (1, 1))
        packed = pack_robot_state(episode, "joint", {"arm_dim": [6], "ee_dim": [1]}, "dataset")
        np.testing.assert_array_equal(packed, [[0, 1, 2, 3, 4, 5, 0.5]])
        colors = episode["vision"]["head"]["colors"]
        self.assertEqual(colors.shape, (1,))
        np.testing.assert_allclose(decode_image_bit(colors[0]), rgb, atol=3)

    def test_scalar_jpeg_roundtrips_alongside_utf8_instruction(self):
        rgb = np.full((8, 12, 3), [220, 30, 10], np.uint8)
        jpeg = encode_image_bit(rgb)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.hdf5"
            with h5py.File(path, "w") as file:
                file["vision/head/colors"] = np.bytes_(jpeg)
                file["instruction"] = "拿杯子"
            episode = load_hdf5(path)
        self.assertEqual(episode["instruction"], "拿杯子")
        stored = episode["vision"]["head"]["colors"]
        self.assertEqual(stored, jpeg)
        np.testing.assert_allclose(decode_image_bit(stored), rgb, atol=3)
