"""PiperX FK boundary tests; they do not load a model checkpoint."""

import unittest
from pathlib import Path

import numpy as np

from XPolicyLab.policy.XBrain_v1.piperx_fk_adapter import (
    adjust_piperx_endpose_z,
    fk_source_path,
    joints14_to_endpose16,
)


class PiperXFKAdapterTests(unittest.TestCase):
    def test_fk_source_is_vendored_with_policy(self):
        self.assertEqual(fk_source_path(), str(Path(__file__).resolve().parents[1] / "piperx_fk.py"))

    def test_batched_joint_chunk_converts_to_pose_chunk(self):
        joints = np.zeros((20, 14), dtype=np.float32)
        joints[:, 6] = 0.35
        joints[:, 13] = 0.30
        poses = joints14_to_endpose16(joints)
        self.assertEqual(poses.shape, (20, 16))
        self.assertEqual(poses.dtype, np.float32)
        self.assertTrue(np.isfinite(poses).all())
        np.testing.assert_allclose(poses[:, 7], joints[:, 6])
        np.testing.assert_allclose(poses[:, 15], joints[:, 13])
        np.testing.assert_allclose(np.linalg.norm(poses[:, 3:7], axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(poses[:, 11:15], axis=1), 1.0, atol=1e-5)

    def test_invalid_joint_width_is_rejected(self):
        with self.assertRaises(ValueError):
            joints14_to_endpose16(np.zeros((20, 13), dtype=np.float32))

    def test_z_adjustment_is_row_and_arm_local_with_floor(self):
        values = np.zeros((5, 16), dtype=np.float32)
        values[:, 2] = [0.12, 0.13, 0.15, 0.17, 0.171]
        values[:, 10] = [0.13, 0.14, 0.16, 0.17, 0.20]
        adjusted = adjust_piperx_endpose_z(values)
        np.testing.assert_allclose(adjusted[:, 2], [0.12, 0.13, 0.13, 0.15, 0.171])
        np.testing.assert_allclose(adjusted[:, 10], [0.13, 0.13, 0.14, 0.15, 0.20])
        np.testing.assert_array_equal(adjusted[:, [0, 1, 3, 11, 15]], values[:, [0, 1, 3, 11, 15]])


if __name__ == "__main__":
    unittest.main()
