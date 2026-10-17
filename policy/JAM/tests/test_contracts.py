"""Portable adapter contract checks; these are not policy-quality or robot tests."""
from pathlib import Path
import json
import hashlib
import io
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import numpy as np

POLICY_DIR = Path(__file__).resolve().parents[1]
ROOT = POLICY_DIR.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'XPolicyLab'), str(POLICY_DIR)]
from XPolicyLab.policy.JAM.delivery import TASKS, load_deployment, validate_overrides
from XPolicyLab.policy.JAM.model import Model, observation_for_policy
from XPolicyLab.utils.process_data import get_robot_action_dim_info


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.dims = get_robot_action_dim_info('piper_x')
        self.observation = {
            'instruction': 'cap the pen',
            'vision': {name: {'color': np.full((480, 640, 3), i, np.uint8)} for i, name in
                       enumerate(('cam_head', 'cam_left_wrist', 'cam_right_wrist'))},
            'state': {'left_arm_joint_state': np.arange(6, dtype=np.float32),
                      'left_ee_joint_state': np.array([6], dtype=np.float32),
                      'right_arm_joint_state': np.arange(7, 13, dtype=np.float32),
                      'right_ee_joint_state': np.array([13], dtype=np.float32)},
        }

    def test_camera_and_joint_order(self):
        obs, instruction = observation_for_policy(self.observation, self.dims)
        np.testing.assert_array_equal(obs['qpos'], np.arange(14))
        self.assertIs(obs['cam_high'], self.observation['vision']['cam_head']['color'])
        self.assertIs(obs['cam_left_wrist'], self.observation['vision']['cam_left_wrist']['color'])
        self.assertEqual(instruction, 'cap the pen')

    def test_encoded_images_are_rejected_in_model(self):
        self.observation['vision']['cam_head']['color'] = np.zeros(50, np.uint8)
        with self.assertRaises(ValueError):
            observation_for_policy(self.observation, self.dims)

    def test_fixed_manifests_and_override_rejection(self):
        for method in ('jam', 'fastwam'):
            manifest = load_deployment(ROOT / 'models' / method, verify_checkpoint=False)
            for task in TASKS:
                validate_overrides({'task_name': task, 'seed': 42}, manifest)
            for key, value in [('seed', 0), ('replan_steps', 32), ('num_inference_steps', 5), ('task_name', 'stack_bowls')]:
                cfg = {'task_name': TASKS[0], key: value}
                with self.assertRaises(ValueError):
                    validate_overrides(cfg, manifest)

    def test_missing_instruction(self):
        self.observation['instruction'] = ''
        with self.assertRaises(ValueError):
            observation_for_policy(self.observation, self.dims)

    def test_public_download_only_fetches_selected_policy(self):
        sys.path.insert(0, str(ROOT / 'XPolicyLab/policy/JAM'))
        try:
            from XPolicyLab.policy.JAM import assets
        finally:
            sys.path.pop(0)
        payload = b'fixed-asset-fixture'
        items = [dict(kind=kind, method=method, path=path,
                      url='https://example.invalid/' + path, bytes=len(payload),
                      sha256=hashlib.sha256(payload).hexdigest())
                 for kind, method, path in [('base', None, 'assets/base/init.pt'),
                                            ('checkpoint', 'jam', 'assets/weights/jam/step.pt'),
                                            ('checkpoint', 'fastwam', 'assets/weights/fastwam/step.pt')]]
        for method in ('jam', 'fastwam'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temporary:
                with patch.object(assets, 'asset_entries', return_value=items), \
                     patch.object(assets.urllib.request, 'urlopen', side_effect=lambda *a, **k: io.BytesIO(payload)) as fetch, \
                     patch.object(sys, 'argv', ['assets.py', 'download-public', '--method', method, '--root', temporary]), \
                     redirect_stdout(io.StringIO()):
                    assets.main()
                self.assertEqual(fetch.call_count, 2)
                self.assertEqual((Path(temporary) / 'assets/base/init.pt').read_bytes(), payload)
                self.assertEqual((Path(temporary) / f'assets/weights/{method}/step.pt').read_bytes(), payload)
                other = 'fastwam' if method == 'jam' else 'jam'
                self.assertFalse((Path(temporary) / f'assets/weights/{other}/step.pt').exists())

    def test_hub_download_uses_pinned_revision_and_final_layout(self):
        from XPolicyLab.policy.JAM import assets
        payload = b'fixed-hub-fixture'
        revision = 'a' * 40
        pairs = [
            ('assets/base/Wan-AI/model/vae.pth', 'vae.pth'),
            ('assets/base/Wan-AI/tokenizer/google/umt5/tokenizer.json', 'google/umt5/tokenizer.json'),
            ('assets/weights/jam/step_028630.pt', 'assets/weights/jam/step_028630.pt'),
        ]
        items = [dict(kind='base', path=dest, source_file=src,
                      source_repository='unit/fixture', source_revision=revision,
                      url=f'https://huggingface.co/unit/fixture/resolve/{revision}/{src}',
                      bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
                 for dest, src in pairs]
        with tempfile.TemporaryDirectory() as temporary:
            def download(**kwargs):
                self.assertEqual(kwargs['revision'], revision)
                self.assertEqual(kwargs['repo_id'], 'unit/fixture')
                target = Path(kwargs['local_dir']) / kwargs['filename']
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                return str(target)
            with patch.object(assets, 'asset_entries', return_value=items), \
                 patch('huggingface_hub.hf_hub_download', side_effect=download) as hub, \
                 patch.object(sys, 'argv', ['assets.py', 'download-public', '--root', temporary]), \
                 redirect_stdout(io.StringIO()):
                assets.main()
            self.assertEqual(hub.call_count, 3)
            for destination, _ in pairs:
                self.assertEqual((Path(temporary) / destination).read_bytes(), payload)

    def test_download_dry_run_never_downloads_or_creates_paths(self):
        from XPolicyLab.policy.JAM import assets
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'not-created'
            output = io.StringIO()
            with patch('huggingface_hub.hf_hub_download', side_effect=AssertionError('network')) as hub, \
                 patch.object(assets.urllib.request, 'urlopen', side_effect=AssertionError('network')) as fetch, \
                 patch.object(sys, 'argv', ['assets.py', 'download-public', '--method', 'jam', '--root', str(root), '--dry-run']), \
                 redirect_stdout(output):
                assets.main()
            rows = [json.loads(line) for line in output.getvalue().splitlines() if line.startswith('{')]
            self.assertEqual(len(rows), 11)
            self.assertFalse(any('/weights/fastwam/' in row['path'] for row in rows))
            self.assertTrue(any('/weights/jam/' in row['path'] for row in rows))
            for row in rows:
                self.assertRegex(row['url'], r'/resolve/[0-9a-f]{40}/')
                self.assertRegex(row['sha256'], r'^[0-9a-f]{64}$')
            hub.assert_not_called()
            fetch.assert_not_called()
            self.assertFalse(root.exists())

    def test_chunk_dimension_and_reset(self):
        class StubPolicy:
            def predict(self, *args):
                return np.tile(np.arange(14, dtype=np.float32), (32, 1)), 0.0
        model = Model.__new__(Model)
        model.policy, model.replan_steps, model.dimensions = StubPolicy(), 24, self.dims
        model.reset()
        with self.assertRaises(RuntimeError):
            model.get_action()
        model.update_obs(self.observation)
        actions = model.get_action()
        self.assertEqual(len(actions), 24)
        self.assertEqual(set(actions[0]), set(self.observation['state']))
        np.testing.assert_array_equal(actions[0]['right_arm_joint_state'], np.arange(7, 13))
        self.assertEqual(actions[0]['left_ee_joint_state'].shape, (1,))
        first, second = dict(self.observation, env_idx=9), dict(self.observation, env_idx=3)
        model.update_obs_batch([first, second])
        self.assertEqual(len(model.get_action_batch([3, 9])), 2)
        model.reset()
        self.assertIsNone(model.current)
        self.assertEqual(model.batch, {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
