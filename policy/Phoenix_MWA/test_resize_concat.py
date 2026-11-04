"""Geometry checks for direct resize and RobotWin-compatible view boundaries."""

import unittest

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, to_tensor

from .phoenix_resize_concat import CAMERAS, current_rgb, resize_concat_rgb


class ResizeConcatTests(unittest.TestCase):
    def test_pixels_match_direct_resize_at_multiple_aspect_ratios(self):
        rng = np.random.default_rng(17)
        images = [
            rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            for height, width in ((480, 640), (640, 480), (477, 639))
        ]
        observation = {'observation': {
            name: {'rgb': image} for name, image in zip(CAMERAS, images)
        }}
        actual = current_rgb(observation)
        for view, image in enumerate(images):
            expected = resize(Image.fromarray(image), [256, 256],
                              InterpolationMode.BICUBIC, antialias=True)
            expected = to_tensor(expected).mul(2).sub(1)
            torch.testing.assert_close(actual[0, view], expected, rtol=0, atol=0)

    def test_horizontal_camera_order_and_roundtrip(self):
        colors = np.eye(3, dtype=np.uint8) * 255
        observation = {'observation': {
            name: {'rgb': np.broadcast_to(color, (480, 640, 3)).copy()}
            for name, color in zip(CAMERAS, colors)
        }}
        panorama = resize_concat_rgb(observation)
        self.assertEqual(tuple(panorama.shape), (1, 3, 256, 768))
        views = current_rgb(observation)
        self.assertEqual(tuple(views.shape), (1, 3, 3, 256, 256))
        for view in range(3):
            self.assertTrue(torch.all(views[0, view, view] == 1))
            torch.testing.assert_close(
                views[:, view], panorama[..., view * 256:(view + 1) * 256],
                rtol=0, atol=0,
            )
        # This is the exact transport layout used inside JointVisualEncoder.
        video = views.unsqueeze(3).repeat(1, 1, 1, 9, 1, 1)
        transport = video.permute(0, 2, 3, 4, 1, 5).reshape(1, 3, 9, 256, 768)
        torch.testing.assert_close(transport[:, :, 0], panorama, rtol=0, atol=0)

    def test_keeps_edges_instead_of_center_crop(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:, :48, 0] = 255
        image[:, -48:, 2] = 255
        observation = {'observation': {name: {'rgb': image} for name in CAMERAS}}
        actual = current_rgb(observation)
        self.assertTrue(torch.all(actual[:, :, 0, :, :8] > 0.99))
        self.assertTrue(torch.all(actual[:, :, 2, :, -8:] > 0.99))


if __name__ == '__main__':
    unittest.main()
