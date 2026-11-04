"""Focused regressions for the X-VLA/GeoRefiner action-space bridge."""

from __future__ import annotations

from pathlib import Path
import os
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

from XPolicyLab.policy.GeoRefiner.georefiner_postprocessor import (
    GeoRefinerPostProcessor,
    GeoRefinerPostProcessorConfig,
    XVLAActionBridge,
)


class XVLAActionBridgeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(20260906)
        self.bridge = XVLAActionBridge()
        self.batch_size = 3
        self.horizon = 30
        self.current_poses = np.zeros((self.batch_size, 2, 7), dtype=np.float64)
        self.current_poses[..., :3] = self.rng.normal(
            size=(self.batch_size, 2, 3)
        )
        quaternion_xyzw = Rotation.random(
            self.batch_size * 2, random_state=self.rng
        ).as_quat().reshape(self.batch_size, 2, 4)
        self.current_poses[..., 3:] = np.concatenate(
            (quaternion_xyzw[..., 3:], quaternion_xyzw[..., :3]), axis=-1
        )
        self.current_grippers = self.rng.uniform(
            0.0, 1.0, size=(self.batch_size, 2)
        )
        self.chunk = np.empty(
            (self.batch_size, self.horizon, 20), dtype=np.float32
        )
        for arm_index, offset in enumerate((0, 10)):
            position_delta = self.rng.normal(
                scale=0.01, size=(self.batch_size, self.horizon, 3)
            )
            self.chunk[..., offset : offset + 3] = (
                self.current_poses[:, arm_index, None, :3]
                + np.cumsum(position_delta, axis=1)
            )
            matrices = Rotation.random(
                self.batch_size * self.horizon, random_state=self.rng
            ).as_matrix().reshape(self.batch_size, self.horizon, 3, 3)
            self.chunk[..., offset + 3 : offset + 9] = (
                self.bridge.matrix_to_rotation6d(matrices)
            )
            self.chunk[..., offset + 9] = self.rng.uniform(
                0.0, 1.0, size=(self.batch_size, self.horizon)
            )

    def test_absolute_canonical_round_trip(self) -> None:
        left, right = self.bridge.absolute_to_canonical(
            self.chunk, self.current_poses
        )
        self.assertEqual(left.shape, (self.batch_size, self.horizon, 7))
        self.assertEqual(right.shape, (self.batch_size, self.horizon, 7))
        restored = self.bridge.canonical_to_absolute(
            left, right, self.current_poses, self.chunk
        )
        np.testing.assert_allclose(restored, self.chunk, rtol=0.0, atol=3e-6)

    def test_state_self_other_order_and_identity(self) -> None:
        left, right = self.bridge.build_canonical_state(
            self.current_poses, self.current_grippers
        )
        self.assertEqual(left.shape, (self.batch_size, 16))
        self.assertEqual(right.shape, (self.batch_size, 16))
        np.testing.assert_allclose(left[:, :7], right[:, 7:14], atol=1e-6)
        np.testing.assert_allclose(left[:, 7:14], right[:, :7], atol=1e-6)
        np.testing.assert_array_equal(left[:, 14:], [[1.0, 0.0]] * self.batch_size)
        np.testing.assert_array_equal(right[:, 14:], [[0.0, 1.0]] * self.batch_size)

    def test_disabled_mode_is_exact(self) -> None:
        policy_dir = Path(__file__).resolve().parents[1]
        processor = GeoRefinerPostProcessor(
            {"mode": "disabled"},
            action_horizon=self.horizon,
            policy_dir=policy_dir,
        )
        float64_chunks = [value.astype(np.float64) for value in self.chunk]
        result = processor.process(float64_chunks)
        for actual, expected in zip(result, float64_chunks):
            self.assertEqual(actual.dtype, expected.dtype)
            np.testing.assert_array_equal(actual, expected)

    def test_environment_mode_overrides_deploy_config(self) -> None:
        with mock.patch.dict(os.environ, {"GEOREFINER_MODE": "disabled"}):
            config = GeoRefinerPostProcessorConfig.from_mapping({"mode": "refine"})
        self.assertEqual(config.mode, "disabled")


if __name__ == "__main__":
    unittest.main()
