"""Check temporal sampling, episode boundaries, and the main-view input contract."""

import unittest

import numpy as np

from focus_vlwa.data import head_history as h


class HeadHistoryTest(unittest.TestCase):
    def test_sampling_uses_completed_episode_grid_slots(self):
        self.assertEqual(h.frame_offsets(25), list(range(-500, 0, 25)))
        indices, valid = h.history_indices(53)
        self.assertEqual(indices[valid].tolist(), [0, 25])
        for fps in (0, -1, 10, 30, float("nan")):
            with self.assertRaises(ValueError):
                h.frame_offsets(fps)
        for frame in (0, 24, 25, 53, 499, 500, 513, 800):
            indices, valid = h.history_indices(frame)
            self.assertTrue(np.all(indices % 25 == 0))
            self.assertTrue(np.all(frame - indices[valid] >= 25))
            self.assertEqual(len(indices), 20)

    def test_short_history_left_pads_and_preserves_order(self):
        frames = [np.full((3, 12, 20), v, np.uint8) for v in (30, 90)]
        packed = h.pack_history(frames)
        self.assertEqual(packed["hist_images"].shape, (20, 224, 224, 3))
        self.assertEqual(packed["hist_mask"].sum(), 2)
        np.testing.assert_array_equal(packed["hist_mask"][-2:], [1, 1])
        self.assertEqual(int(packed["hist_images"][-2, 112, 112, 0]), 30)
        self.assertEqual(int(packed["hist_images"][-1, 112, 112, 0]), 90)

    def test_invalid_slots_are_zero_and_legacy_crops_are_rejected(self):
        packed = h.pack_history([None, np.full((224, 224, 3), 7, np.uint8)], [0, 1])
        self.assertEqual(packed["hist_images"][:-1].sum(), 0)
        with self.assertRaises(ValueError):
            h.pack_history(np.zeros((21, 3, 84, 84, 3), np.uint8))
        with self.assertRaises(ValueError):
            h.pack_history([None], [0.5])
        with self.assertRaises(ValueError):
            h.pack_history([None] * 33)

    def test_lerobot_split_masks_episode_start_and_keeps_wrist_unchanged(self):
        key = "observation.images.cam_high"
        images = np.ones((21, 3, 8, 8), np.float32)
        images[-1] = 0.25
        padding = np.ones(21, dtype=bool)
        padding[-3:] = False
        wrist = object()
        data = {key: images, key + "_is_pad": padding, "wrist": wrist}
        out = h.SplitLeRobotHeadHistory(key)(data)
        self.assertEqual(out["hist_mask"].sum(), 2)
        np.testing.assert_array_equal(out[key], images[-1])
        self.assertEqual(out["hist_images"][-1].max(), 255)
        self.assertIs(out["wrist"], wrist)
        self.assertEqual(data[key].shape[0], 21)

    def test_float_pixels_and_validation(self):
        self.assertEqual(h.full_frame(np.ones((4, 5, 3), np.float32)).max(), 255)
        for image in (np.full((4, 5, 3), -1.0), np.full((4, 5, 3), np.nan)):
            with self.assertRaises(ValueError):
                h.full_frame(image)

    def test_online_history_matches_training_offsets_and_resets(self):
        buffer = h.HeadHistoryBuffer(25)
        for index in range(54):
            buffer.observe(index, np.full((4, 4, 3), index, np.uint8))
        packed = buffer.pack(53)
        self.assertEqual(packed["hist_mask"].sum(), 2)
        self.assertEqual(packed["hist_images"][-2, 0, 0, 0], 0)
        self.assertEqual(packed["hist_images"][-1, 0, 0, 0], 25)
        with self.assertRaises(ValueError):
            buffer.observe(0, np.zeros((4, 4, 3), np.uint8))
        buffer.reset()
        buffer.observe(0, np.zeros((224, 224, 3), np.uint8))
        self.assertEqual(buffer.pack(0)["hist_mask"].sum(), 0)


if __name__ == "__main__":
    unittest.main()
