"""Deployment regression checks against the actual H5 checkpoint metadata."""
import copy
import os
import json
from pathlib import Path
import unittest

import numpy as np
from omegaconf import OmegaConf
import torch

from .joint96_assets import POLICY_DIR, read_source
from .model import Model, _activate_runtime


class H5DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, assets = read_source()
        _activate_runtime(assets['runtime_root'])
        from . import phoenix_backend, phoenix_contract

        cls.backend = phoenix_backend
        cls.contract = phoenix_contract
        cls.package = POLICY_DIR / 'phoenix_source'
        cls.release = json.loads((POLICY_DIR / 'phoenix_release.json').read_text())
        cls.weights = Path(os.environ['PHOENIX_H5_TEST_WEIGHTS']) if os.environ.get('PHOENIX_H5_TEST_WEIGHTS') else None
        torch.set_num_threads(1)

    def load_pair(self, family):
        if self.weights is None:
            self.skipTest('Set PHOENIX_H5_TEST_WEIGHTS to the downloaded H5 weights directory')
        spec = self.release['checkpoints'][family]
        payload = torch.load(self.weights / spec['path'], map_location='cpu', mmap=True, weights_only=True)
        return OmegaConf.load(self.package / f'{family}.json'), payload

    def test_all_current_checkpoint_contracts(self):
        for family, slot in [('arx_x5', 2), ('piper', 4), ('piper_x', 15)]:
            with self.subTest(family=family):
                cfg, payload = self.load_pair(family)
                result = self.contract.validate_training(cfg, payload, self.package, family, 35000, packaged=True)
                self.assertEqual(result['support']['embodiment_id'], slot)
                self.assertFalse(result['cached_visuals'])
                self.assertEqual(len(payload['mot']), 1650)
                self.assertEqual(len(payload['proprio_encoder']), 3)

    def test_reject_legacy_cached_checkpoint(self):
        cfg, payload = self.load_pair('piper')
        bad = dict(payload)
        bad['joint96_metadata'] = copy.deepcopy(payload['joint96_metadata'])
        bad['joint96_metadata']['soft_prompt']['training_contract']['dataloader'] = 'openwam_joint14_with_legacy_clean_visual_cache'
        with self.assertRaisesRegex(ValueError, 'direct-resize'):
            self.contract.validate_training(cfg, bad, self.package, 'piper', 35000, packaged=True)

    def test_reject_wrong_family_step_and_normalization(self):
        cfg, payload = self.load_pair('arx_x5')
        for family, step in [('piper', 35000), ('arx_x5', 40000)]:
            with self.assertRaises(ValueError):
                self.contract.validate_training(cfg, payload, self.package, family, step, packaged=True)
        for key, value in [('data.train.processor.norm_default_mode', 'z-score'),
                           ('data.train.edge_manifest_sha256', '0' * 64),
                           ('data.train.image_color_jitter.enabled', False),
                           ('data.train.visual_cache', {'root': 'wrong-cache'}),
                           ('data.train.processor.action_state_merger.state_target_slices.0.source_slice', [7, 13])]:
            bad_cfg = copy.deepcopy(cfg)
            OmegaConf.update(bad_cfg, key, value)
            with self.assertRaises(ValueError):
                self.contract.validate_training(bad_cfg, payload, self.package, 'arx_x5', 35000, packaged=True)

    def test_reject_changed_edge25_geometry_and_trim_contracts(self):
        cfg, payload = self.load_pair('arx_x5')
        for key, value in (
            ('boundary_trim', {}),
            ('rgb_crop', [256, 256]),
            ('image_augmentation', {'inference_enabled': True}),
            ('boundary_trim_source_sha256', {}),
            ('rgb_camera_order', ['cam_left_wrist', 'cam_head', 'cam_right_wrist']),
            ('temporal_trim_source_sha256', {}),
            ('action_target_validity', 'terminal_action_repeated'),
        ):
            bad = dict(payload)
            bad['joint96_metadata'] = copy.deepcopy(payload['joint96_metadata'])
            bad['joint96_metadata']['soft_prompt']['training_contract'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.contract.validate_training(cfg, bad, self.package, 'arx_x5', 35000, packaged=True)

    def test_gripper_and_invalid_state(self):
        state = np.zeros(14, dtype=np.float32)
        state[[6, 13]] = [-0.03, 1.03]
        for family in ('piper', 'piper_x'):
            actual = self.contract.preprocess_state(state, family)
            np.testing.assert_array_equal(actual[[6, 13]], [0, 1])
        np.testing.assert_array_equal(self.contract.preprocess_state(state, 'arx_x5'), state)
        state[6] = -0.06
        with self.assertRaises(ValueError):
            self.contract.preprocess_state(state, 'piper')
        state[0] = np.nan
        with self.assertRaises(ValueError):
            self.contract.preprocess_state(state, 'arx_x5')

    def test_geometry_and_rgb_channels(self):
        from PIL import Image
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import resize, to_tensor

        rng = np.random.default_rng(7)
        for h, w in [(480, 640), (640, 480), (256, 256), (477, 639)]:
            image = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            obs = {'observation': {name: {'rgb': image} for name in ('head_camera', 'left_camera', 'right_camera')}}
            actual = self.contract.current_rgb(obs)
            expected = resize(Image.fromarray(image), [256, 256], InterpolationMode.BICUBIC, antialias=True)
            expected = to_tensor(expected).mul(2).sub(1)
            torch.testing.assert_close(actual[0, 0], expected, rtol=0, atol=0)
        red = np.zeros((256, 256, 3), dtype=np.uint8)
        red[:, :, 0] = 255
        obs = {'observation': {name: {'rgb': red} for name in ('head_camera', 'left_camera', 'right_camera')}}
        actual = self.contract.current_rgb(obs)
        self.assertTrue(torch.all(actual[:, :, 0] == 1))
        self.assertTrue(torch.all(actual[:, :, 2] == -1))

    def test_text_cache_and_unknown_instruction(self):
        index = json.loads((self.package / 'index.json').read_text())
        text = self.contract.CachedText(self.package / 'text.pt', index['package_sha256']['text.pt'])
        original = torch.load(self.package / 'text.pt', map_location='cpu', weights_only=True)
        instructions = json.loads((self.package / 'instructions.json').read_text())
        self.assertEqual(set(original['rows']), {self.contract.PROMPT.format(task=task)
            for tasks in instructions.values() for task in tasks})
        for prompt, row in original['rows'].items():
            actual, mask = text.rows[prompt]
            expected = row['context'].masked_fill(~row['mask'].bool()[:, None], 0)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(mask.all())
        with self.assertRaises(ValueError):
            text.get('unregistered test task', device='cpu', dtype=torch.float32)

    def test_disabled_dummy_and_wrong_options(self):
        with self.assertRaises(ValueError):
            Model({'allow_dummy_policy': True})
        for value in [dict(current_visual_mode='repeat9_full_stvae'), dict(replan_steps=33),
                      dict(action_horizon=16), dict(action_type='ee')]:
            with self.assertRaises(ValueError):
                self.backend.validate_options(dict(env_cfg_type='piper', **value))


if __name__ == '__main__':
    unittest.main()
