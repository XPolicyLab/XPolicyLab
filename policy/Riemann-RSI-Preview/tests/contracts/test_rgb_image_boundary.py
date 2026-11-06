"""Known RGB colours and feature values across real policy/file boundaries."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from rsi.image_files import copy_rgb, load_public_rgb, write_rgb_preview


class RGBImageBoundary(unittest.TestCase):
    def test_rgb_copy_preserves_channel_order_and_has_independent_storage(self):
        image = np.array([[[231, 7, 19, 100]]], dtype=np.uint8)
        copied = copy_rgb(image)
        np.testing.assert_array_equal(copied, [[[231, 7, 19]]])
        copied[0, 0, 0] = 0
        self.assertEqual(image[0, 0, 0], 231)

    def test_static_public_png_is_loaded_as_rgb(self):
        image = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'public-asset.png'
            Image.fromarray(image).save(path)
            np.testing.assert_array_equal(load_public_rgb(path), image)
            with self.assertRaises(TypeError):
                load_public_rgb(path.read_bytes())
            with self.assertRaises(TypeError):
                load_public_rgb(image)

    def test_rgb_preview_has_standard_file_colours_and_preserves_gray_masks(self):
        image = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'preview.png'
            self.assertTrue(write_rgb_preview(path, image))
            self.assertTrue(path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
            with Image.open(path) as saved:
                np.testing.assert_array_equal(np.asarray(saved), image)
            mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
            self.assertTrue(write_rgb_preview(path, mask))
            with Image.open(path) as saved:
                np.testing.assert_array_equal(np.asarray(saved), mask)

    def test_preview_failure_does_not_turn_diagnostic_io_into_a_policy_error(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(write_rgb_preview(Path(directory)/'absent'/'preview.png',
                                               np.zeros((2, 2, 3), dtype=np.uint8)))

    def test_imitation_checkpoint_receives_the_same_rgb_signed_features(self):
        from tasks.imitate_sorting_sequence.skills.rgb_perception.implementation import input_image
        image = np.array([[[255, 128, 0]]], dtype=np.uint8)
        background = np.array([[[127, 0, 255]]], dtype=np.uint8)
        expected = np.array([1., 128./255., 0., 1., 1., -255./128., 255./128.], dtype=np.float32)
        actual = input_image(image, background)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual[0, 0], expected)

    def test_imitation_colour_histogram_treats_red_as_red(self):
        from tasks.imitate_sorting_sequence.skills.rgb_perception.implementation import foreground_hsv_histogram
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        image[..., 0] = 255
        actual = foreground_hsv_histogram(image, np.zeros_like(image), [40, 40])
        expected = np.zeros(56, dtype=np.float32)
        expected[[0, 24+15, 24+16+15]] = 1.
        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
