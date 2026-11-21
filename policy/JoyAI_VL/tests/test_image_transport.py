"""Image boundaries: official JPEG formats, internal PNGs and SAM3 masks."""
import base64
import importlib.util
import io
import itertools
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[2]))
from XPolicyLab.utils.process_data import decode_obs_images, encode_image_bit
from roboshell.image_transport import decode_png, encode_png
from roboshell.server.bridge import BridgeExecutor, Model
from roboshell.server.core import CAMERA_NAMES, pixel_array
from roboshell.server.robot15 import RobotTools
from vendor.robot15.hexashell.vision.sam3_client import Sam3Client, Sam3Error


def stripes():
    rgb = np.zeros((48, 96, 3), np.uint8)
    rgb[:, :32] = (230, 20, 10)
    rgb[:, 32:64] = (10, 230, 20)
    rgb[:, 64:] = (20, 10, 230)
    return rgb


class Images(unittest.TestCase):
    def test_official_formats_reach_internal_png_and_arrays_as_rgb(self):
        rgb = stripes()
        # A legacy fixture is consumed only here, never stored or published.
        ok, legacy = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 100])
        self.assertTrue(ok)
        buffers = (rgb, encode_image_bit(rgb, quality=100), legacy.tobytes())
        obs = decode_obs_images({'vision': {
            name: {'color': value} for name, value in zip(CAMERA_NAMES, buffers)
        }})
        executor = BridgeExecutor.__new__(BridgeExecutor)
        executor.camera_model = {name: {'intrinsics': np.eye(3).tolist(),
                                       'extrinsics_world': np.eye(4).tolist()} for name in CAMERA_NAMES}
        model = Model.__new__(Model)
        model.executor = executor
        model.update_obs(obs)
        pictures = executor.observe()['png']
        for name, picture in pictures.items():
            decoded = decode_png(picture)
            np.testing.assert_array_equal(decoded, obs['vision'][name]['color'])
            for col, channel in ((16, 0), (48, 1), (80, 2)):
                self.assertEqual(int(np.argmax(decoded[20, col])), channel)
                self.assertGreater(int(decoded[20, col, channel]), 220)
            array = np.load(io.BytesIO(base64.b64decode(pixel_array(picture))), allow_pickle=False)
            np.testing.assert_array_equal(array, decoded)

    def test_tool_pictures_preserve_rgb_pixels(self):
        tool = RobotTools.__new__(RobotTools)
        tool.files, tool.captions, tool.picture_index = {}, {}, itertools.count()
        rgb = stripes()
        tool.attach('image', rgb, 'synthetic stripes')
        decoded = decode_png(base64.b64decode(tool.files['robot_000000.png']))
        np.testing.assert_array_equal(decoded, rgb)
        self.assertEqual(tool.captions['robot_000000.png'], 'synthetic stripes')

    def test_internal_decoder_does_not_accept_official_jpeg_buffers(self):
        with self.assertRaisesRegex(ValueError, 'internal PNG'):
            decode_png(encode_image_bit(stripes()))

    def test_scene_colour_names_and_overlay_preserve_rgb(self):
        spec = importlib.util.spec_from_file_location('transport_scene', ROOT / 'tools/scene/tool.py')
        scene = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scene)
        rgb = stripes()
        camera = scene.Camera('head', rgb, np.eye(3), np.eye(4))
        labels = scene.colour_names(rgb[20, [16, 48, 80]])
        self.assertEqual([scene.COLOUR_NAMES[int(i)] for i in labels], ['red', 'green', 'blue'])
        state = types.SimpleNamespace(cams={'head': camera})
        seg = {'camera': 'head', 'inside': np.ones(rgb.shape[:2], bool),
               'robot': np.zeros(rgb.shape[:2], bool), 'objects': [], 'masks': []}
        np.testing.assert_array_equal(decode_png(scene.overlay(state, seg)), rgb)

    def test_sam3_mask_keeps_small_components_and_rejects_wrong_shape(self):
        mask = np.zeros((48, 96), np.uint8)
        mask[1, 2] = 255
        mask[10:12, 20:23] = 255
        item = {'png_b64': base64.b64encode(encode_png(mask)).decode(),
                'score': .9, 'box': [0, 0, 95, 47]}
        client = Sam3Client('http://unused.invalid')
        try:
            with patch.object(client, '_request', side_effect=[{'image_id': 'test'}, {'masks': [item]}]):
                result = client.segment(stripes(), text='synthetic')
            np.testing.assert_array_equal(result[0].mask, mask > 0)
            self.assertEqual(result[0].area, 7)
            item['png_b64'] = base64.b64encode(encode_png(mask[:10])).decode()
            with patch.object(client, '_request', return_value={'masks': [item]}):
                with self.assertRaises(Sam3Error):
                    client.segment(stripes(), text='synthetic')
        finally:
            client.close()


if __name__ == '__main__':
    unittest.main()
