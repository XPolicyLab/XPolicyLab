"""CPU adapter checks and optional historical z-score saved-weight tests."""
import copy
import json
import os
from pathlib import Path
from functools import partial
from types import SimpleNamespace
import tempfile
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .current_visual_adapter import configure_current_visual, install_current_repeat9, restore_current_encoder
from .model import Model, _activate_runtime
from .joint96_assets import relative_path, load_assets

class Encoder:
    def __init__(self):
        self.calls = []

    def encode_raw_video_latents(self, rgb, *, num_views=1):
        self.calls.append((rgb.clone(), num_views))
        # The first result depends on ALL input frames, so cropping early would
        # produce a different value. Other temporal positions carry sentinels.
        value = rgb[:, :1].mean(dim=(3, 4), keepdim=True).sum(dim=2, keepdim=True)
        return torch.cat((value, value + 100, value + 200), dim=2)


class Repeat9Tests(unittest.TestCase):
    def test_single_frame_and_batch_isolation(self):
        encoder = Encoder()
        install_current_repeat9(encoder)
        rgb = torch.stack((torch.ones(3, 1, 4, 12), torch.ones(3, 1, 4, 12) * 2))
        result = encoder.encode_raw_video_latents(rgb, num_views=3)
        self.assertEqual(tuple(result.shape), (2, 1, 1, 1, 1))
        self.assertEqual(result.flatten().tolist(), [9., 18.])
        self.assertTrue(result.is_contiguous())
        self.assertEqual(len(encoder.calls), 2)
        for original, (repeated, views) in zip(rgb.split(1), encoder.calls):
            self.assertEqual(views, 3)
            torch.testing.assert_close(repeated, original.repeat(1, 1, 9, 1, 1))

    def test_multiframe_passthrough_and_restore(self):
        encoder = Encoder()
        original = install_current_repeat9(encoder)
        rgb = torch.randn(2, 3, 5, 4, 12)
        result = encoder.encode_raw_video_latents(rgb, num_views=3)
        torch.testing.assert_close(result, original(rgb, num_views=3))
        torch.testing.assert_close(encoder.calls[0][0], rgb)
        restore_current_encoder(encoder)
        self.assertEqual(encoder.encode_raw_video_latents, original)

    def test_double_install_and_bad_inputs(self):
        encoder = Encoder()
        install_current_repeat9(encoder)
        with self.assertRaises(RuntimeError):
            install_current_repeat9(encoder)
        for value in (torch.zeros(1, 3, 4, 4), torch.zeros(0, 3, 1, 4, 4)):
            with self.assertRaises(ValueError):
                encoder.encode_raw_video_latents(value)

    def test_requires_full_stvae_temporal_output(self):
        encoder = SimpleNamespace(encode_raw_video_latents=lambda rgb, **_: rgb[:, :1, :2])
        install_current_repeat9(encoder)
        with self.assertRaises(RuntimeError):
            encoder.encode_raw_video_latents(torch.ones(1, 3, 1, 4, 4))

    def test_identity_and_wan_unchanged(self):
        wan = object()
        visual = SimpleNamespace(current_conditioning='legacy_clip_prefix', stvae_encoder=Encoder())
        model = SimpleNamespace(joint_visual_encoder=visual, vae=wan)
        record = configure_current_visual(model, 'repeat9_full_stvae')
        self.assertIs(model.vae, wan)
        self.assertEqual(record['repeat_frames'], 9)
        self.assertEqual(record['slice_after'], 'full_stvae')
        self.assertFalse(record['wan_branch_changed'])
        visual.current_conditioning = 'independent_single_frame_v1'
        with self.assertRaises(ValueError):
            configure_current_visual(model, 'repeat9_full_stvae')



class AdapterTests(unittest.TestCase):
    def test_manifest_paths_are_relative(self):
        for path in ('/server/secret', '../escape', 'a/../../escape', ''):
            with self.assertRaises(ValueError):
                relative_path('/bundle', path)
        self.assertEqual(relative_path('/bundle', 'runtime/src'), Path('/bundle/runtime/src'))

    def test_shared_checkpoint_resolution(self):
        from . import checkpoint_paths
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / 'checkpoints/RoboDojo-release-arx_x5-joint-0'
            bundle.mkdir(parents=True)
            (bundle / 'phoenix-wam-arx-x5-step_010000.pt').write_bytes(b'fixture')
            config = dict(runtime_mode='joint96_real', bench_name='RoboDojo', ckpt_name='release',
                          env_cfg_type='arx_x5', action_type='joint', seed=0)
            with patch.object(checkpoint_paths, 'POLICY_DIR', root):
                self.assertEqual(checkpoint_paths.resolve_policy_checkpoint(config), bundle)
                config['checkpoint_path'] = str(root / 'missing')
                config['ckpt_name'] = str(root / 'also-missing')
                with self.assertRaises(FileNotFoundError):
                    checkpoint_paths.resolve_policy_checkpoint(config)

    def test_phoenix_default_checkpoint_directory(self):
        from . import checkpoint_paths
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / 'checkpoints/phoenix-wam-step10000'
            weights.mkdir(parents=True)
            checkpoint = weights / 'phoenix-wam-arx-x5-step_010000.pt'
            checkpoint.write_bytes(b'fixture')
            with patch.object(checkpoint_paths, 'POLICY_DIR', root):
                self.assertEqual(checkpoint_paths.resolve_policy_checkpoint(
                    {'ckpt_name': 'phoenix-wam-step10000'}), weights)
                with self.assertRaises(FileNotFoundError):
                    checkpoint_paths.resolve_policy_checkpoint({'checkpoint_path': str(checkpoint)})
        model = SimpleNamespace(model_cfg={}, _load_phoenix_policy=lambda: 'loaded')
        with patch.object(model, '_load_phoenix_policy') as load:
            Model._load_real_policy(model)
            load.assert_called_once_with()
        model.model_cfg['runtime_mode'] = 'official'
        with self.assertRaisesRegex(ValueError, 'H5 Phoenix deployment'):
            Model._load_real_policy(model)

    def test_standard_model_batch_and_rgb(self):
        dims = dict(arm_dim=[6, 6], ee_dim=[1, 1])
        with patch('XPolicyLab.policy.Phoenix_MWA._adapter_base.get_robot_action_dim_info', return_value=dims):
            # Mock only neural inference; exercise the real adapter batch/observation logic.
            with patch.object(Model, '_load_real_policy'):
                model = Model(dict(runtime_mode='joint96_real_phoenix', env_cfg_type='arx_x5', action_type='joint'))
            model.session_factory = lambda: SimpleNamespace(pending_model_actions=[], reset_model=lambda: None)
            model._predict = lambda *_: [dict(left_arm_joint_state=np.zeros(6), left_ee_joint_state=np.zeros(1), right_arm_joint_state=np.zeros(6), right_ee_joint_state=np.zeros(1)) for _ in range(32)]
        obs = observation(7)
        model.update_obs_batch([obs, observation(2)])
        self.assertEqual(model.env_order, [7, 2])
        chunks = model.get_action_batch([2, 7])
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 32)
        self.assertEqual(chunks[0][0]['left_arm_joint_state'].shape, (6,))
        self.assertEqual(chunks[0][0]['right_ee_joint_state'].shape, (1,))
        np.testing.assert_array_equal(model.env_obs[7][0]['observation']['head_camera']['rgb'], obs['vision']['cam_head']['color'])
        with self.assertRaises(ValueError):
            model.update_obs_batch([obs, obs])
        with self.assertRaises(ValueError):
            model.get_action_batch([99])
        model.reset()
        with self.assertRaises(ValueError):
            model.get_action_batch()


def observation(index=0):
    # Distinct RGB channels catch any accidental channel swap. Read-only arrays
    # exercise the actual server's decoded-array contract.
    color = np.full((256, 256, 3), [23, 81, 190], dtype=np.uint8)
    color.flags.writeable = False
    return dict(env_idx=index, instruction='cover the blocks',
        vision={name: {'color': color} for name in ('cam_head', 'cam_left_wrist', 'cam_right_wrist')},
        state=dict(left_arm_joint_state=np.zeros(6), left_ee_joint_state=np.array([.5]),
                   right_arm_joint_state=np.zeros(6), right_ee_joint_state=np.array([.5])))


@unittest.skipUnless(os.environ.get('JOINT96_LEGACY_TEST_WEIGHTS'), 'Legacy z-score release tests; current release tests are in test_phoenix')
class RealContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from hydra.utils import instantiate
        from .joint96_assets import read_release
        if read_release().get('runtime_mode') == 'joint96_real_phoenix':
            raise unittest.SkipTest('Requires a legacy z-score release manifest')
        cls.releases = {robot: load_assets(os.environ['JOINT96_LEGACY_TEST_WEIGHTS'], robot_type=robot)
                        for robot in ('arx_x5', 'piper', 'piper_x')}
        cls.manifest, cls.assets = cls.releases['arx_x5']
        _activate_runtime(cls.assets['runtime_root'])
        from . import joint96_backend as backend
        cls.backend = backend
        from wam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        cls.cfg = backend.load_inference_config(cls.assets['inference_config'], cls.assets)
        cls.stats = cls.assets['dataset_stats']
        cls.payloads = {robot: torch.load(assets['checkpoint'], map_location='cpu', mmap=True, weights_only=True)
                        for robot, (_, assets) in cls.releases.items()}
        payload = cls.payloads['arx_x5']
        cls.payload = {key: payload[key] for key in ('step', 'joint96_metadata')}
        cls.full_payload = payload
        cls.processor = instantiate(cls.cfg.observation.processor).eval()
        cls.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(cls.stats)))

    def check_contract(self, payload=None, stats=None):
        return self.backend.validate_contract(self.cfg, payload or self.payload,
                                             stats or self.stats, self.manifest['step'])

    def test_three_robot_profiles_statistics_and_weights(self):
        from .robot_profiles import select_robot_assets, checkpoint_robot_support
        from hydra.utils import instantiate
        from wam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        for robot, slot in [('arx_x5', 2), ('piper', 4), ('piper_x', 15)]:
            with self.subTest(robot=robot):
                manifest, assets = self.releases[robot]
                cfg = self.backend.load_inference_config(assets['inference_config'], assets)
                payload = self.payloads[robot]
                self.backend.validate_contract(cfg, payload, assets['dataset_stats'], manifest['step'])
                self.assertIn(robot.replace('_', '-'), assets['checkpoint'].name)
                self.assertEqual(cfg.checkpoint_contract.expected_pretrain_step,
                                 {'arx_x5': 230000, 'piper': 235000, 'piper_x': 245000}[robot])
                support = checkpoint_robot_support(payload, assets['robot_profile'])
                self.assertEqual(support['embodiment_id'], slot)
                self.assertTrue(support['posttrained'])
                self.assertTrue(support['structurally_supported'])
                processor = instantiate(cfg.observation.processor).eval()
                stats = json.loads(assets['dataset_stats'].read_text())
                processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(assets['dataset_stats'])))
                runtime = SimpleNamespace(model=object(), processor=processor, device='cpu',
                    model_dtype=torch.float32, embodiment_id=slot, robot_type=robot,
                    action_dim_is_pad=self.backend._static_dimension_is_pad(processor, 'action'))
                config = SimpleNamespace(num_inference_steps=1, replan_steps=10, seed=0, rand_device='cpu')
                session = self.backend.RoboDojoRealJoint96Session(runtime, config)
                mean = np.asarray(stats['state']['default']['global_mean'], dtype=np.float32)
                normalized = session._normalize_state(mean)
                torch.testing.assert_close(normalized, torch.zeros_like(normalized), atol=1e-6, rtol=0)
                actions = session._denormalize_action(torch.zeros(32, 80))[0]
                expected = np.asarray(stats['action']['default']['global_mean'], dtype=np.float32)
                np.testing.assert_allclose(actions, np.broadcast_to(expected, (32, 14)), atol=1e-6)
                if slot != 2:
                    with self.assertRaisesRegex(ValueError, 'normalization'):
                        self.backend.validate_contract(cfg, payload, self.stats, manifest['step'])

    def test_cross_robot_checkpoints_are_rejected(self):
        for robot, (manifest, assets) in self.releases.items():
            cfg = self.backend.load_inference_config(assets['inference_config'], assets)
            for other, payload in self.payloads.items():
                if other == robot:
                    continue
                with self.subTest(robot=robot, checkpoint=other):
                    with self.assertRaisesRegex(ValueError, 'embodiment_ids'):
                        self.backend.validate_contract(cfg, payload, assets['dataset_stats'], manifest['step'])

    def test_wrong_loss_recipe_is_rejected(self):
        manifest, assets = self.releases['piper_x']
        cfg = self.backend.load_inference_config(assets['inference_config'], assets)
        cfg.checkpoint_contract.loss_recipe = 'robodojo_real_half_video_v1'
        with self.assertRaisesRegex(ValueError, 'loss recipe'):
            self.backend.validate_contract(cfg, self.payloads['piper_x'], assets['dataset_stats'], manifest['step'])
        cfg.checkpoint_contract.loss_weights = {'vae': .25, 'stvae': .25, 'action': 1.0}
        with self.assertRaisesRegex(ValueError, 'loss_weights'):
            self.backend.validate_contract(cfg, self.payloads['piper_x'], assets['dataset_stats'], manifest['step'])

    def test_prompt_id_reaches_dit_for_each_robot(self):
        from .robot_profiles import select_robot_assets
        from hydra.utils import instantiate
        from wam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        for robot, slot in [('arx_x5', 2), ('piper', 4), ('piper_x', 15)]:
            manifest, assets = self.releases[robot]
            cfg = self.backend.load_inference_config(assets['inference_config'], assets)
            processor = instantiate(cfg.observation.processor).eval()
            processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(assets['dataset_stats'])))
            model = SimpleNamespace(encode_prompt=lambda prompt: (torch.zeros(1,128,4096), torch.ones(1,128,dtype=torch.bool)))
            runtime = SimpleNamespace(model=model, processor=processor, device='cpu', first_action_recorded=True,
                model_dtype=torch.float32, embodiment_id=slot, robot_type=robot,
                action_dim_is_pad=self.backend._static_dimension_is_pad(processor, 'action'))
            config = SimpleNamespace(num_inference_steps=1, replan_steps=10, seed=0, rand_device='cpu')
            session = self.backend.RoboDojoRealJoint96Session(runtime, config)
            obs = dict(observation={key: {'rgb': np.zeros((256,256,3),np.uint8)} for key in self.backend.CAMERAS},
                       joint_action={'vector': np.zeros(14, np.float32)})
            with patch.object(self.backend, 'infer_frame_joint_window', return_value={'action':torch.zeros(32,80)}) as infer:
                result, _ = session._infer_action_chunk(obs, 'pick up the block')
                self.assertEqual(infer.call_args.kwargs['embodiment_ids'].tolist(), [slot])
                self.assertEqual(result.shape, (32,14))
                self.assertTrue(np.isfinite(result).all())

    def test_missing_or_corrupt_robot_prompt_rejected(self):
        from .robot_profiles import select_robot_assets, checkpoint_robot_support, PROMPT_KEY
        profile = select_robot_assets(self.assets, 'piper')['robot_profile']
        for value in [None, torch.zeros(1,8192), torch.zeros(28,8192), torch.full((28,8192), float('nan'))]:
            payload = dict(self.payload, mot={PROMPT_KEY:value})
            with self.assertRaises(ValueError):
                checkpoint_robot_support(payload, profile)
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            select_robot_assets(self.assets, '../other')

    def test_complete_restore_keeps_training_initialization_guard(self):
        from wam.model import factory
        kwargs = dict(video_dit_config={'in_dim': 96, 'out_dim': 96, 'text_dim': 4096},
            action_dit_config={'action_dim': 80}, proprio_dim=80,
            joint96_three_view=True, joint_visual_codec={'source_root': 'fixture'},
            action_dit_use_interpolated_init=False, skip_dit_load_from_pretrain=True)
        with self.assertRaisesRegex(ValueError, 'requires official Wan'):
            factory.build_wam_from_wan22(**kwargs)
        with patch.object(factory, 'load_wan22_ti2v_5b_components',
                          side_effect=RuntimeError('reached component construction')) as loader:
            with self.assertRaisesRegex(RuntimeError, 'reached component construction'):
                factory.build_wam_from_wan22(**kwargs, joint96_checkpoint_restore=True)
            self.assertTrue(loader.call_args.kwargs['skip_dit_load_from_pretrain'])

    def test_pruned_encoder_source_keeps_checkpoint_identity(self):
        import hashlib
        from wam.model.modules.codecs.joint_visual_encoder import encoder_source_sha256
        self.assertEqual(encoder_source_sha256(self.assets['encoder_source_root']),
                         self.check_contract()['source_sha256'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'src/fastwam/models/wan22'
            source.mkdir(parents=True)
            encoder = source / 'online_dinov3_stvae_encoder.py'
            encoder.write_text('# unchanged inference encoder\n')
            records = {'models/wan22/online_dinov3_stvae_encoder.py': hashlib.sha256(encoder.read_bytes()).hexdigest(),
                       'removed_trainer.py': hashlib.sha256(b'not shipped').hexdigest()}
            digest = hashlib.sha256()
            for name, sha in sorted(records.items()):
                digest.update(name.encode() + b'\0')
                digest.update(bytes.fromhex(sha))
            (root / 'UPSTREAM_SOURCE.json').write_text(json.dumps(dict(
                schema='phoenix.encoder.inference-subset.v1', upstream_files=records,
                upstream_source_sha256=digest.hexdigest())))
            self.assertEqual(encoder_source_sha256(root), digest.hexdigest())
            encoder.write_text('# modified encoder\n')
            with self.assertRaisesRegex(ValueError, 'differs from upstream'):
                encoder_source_sha256(root)

    def test_frozen_models_load_only_manifest_paths(self):
        assets = self.assets
        model = SimpleNamespace(model_paths=dict(
            vae=str(assets['wan_vae']), text_encoder=str(assets['t5_encoder']),
            tokenizer=str(assets['tokenizer_root']), joint_visual_encoder={'stvae':dict(
                teacher_checkpoint_path=str(assets['dino_checkpoint']),
                stvae_checkpoint_path=str(assets['stvae_checkpoint']))}))
        self.assertEqual(len(self.backend.verify_loaded_weight_paths(model, assets)), 5)
        model.model_paths['text_encoder'] = str(assets['wan_vae'])
        with self.assertRaisesRegex(ValueError, 'different release resource: t5_encoder'):
            self.backend.verify_loaded_weight_paths(model, assets)

    def test_saved_posttraining_contract(self):
        self.assertEqual(self.check_contract()['action_dim'], 80)

    def test_reject_domain_family_cache_and_lineage(self):
        for key, value in [('domain', 'robotwin'), ('embodiment_ids', {'piper': 4}),
                           ('cache_contract_sha256', 'invalid')]:
            payload = copy.deepcopy(self.payload)
            payload['joint96_metadata']['soft_prompt']['training_contract'][key] = value
            with self.assertRaises(ValueError):
                self.check_contract(payload)
        payload = copy.deepcopy(self.payload)
        payload['joint96_metadata']['model_paths']['posttrain_initialization']['step'] = 150000
        with self.assertRaises(ValueError):
            self.check_contract(payload)

    def test_reject_wrong_step_and_stats(self):
        payload = copy.deepcopy(self.payload)
        payload['step'] += 1
        with self.assertRaises(ValueError):
            self.check_contract(payload)
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / 'stats.json'
            bad.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'normalization'):
                self.check_contract(stats=bad)

    def test_training_camera_order_and_canonical_mapping(self):
        obs = {'observation': {name: {'rgb': np.full((256, 256, 3), value, np.uint8)}
               for name, value in zip(self.backend.CAMERAS, (0, 127, 255))}}
        rgb = self.backend.current_rgb(self.processor, obs)
        self.assertEqual(tuple(rgb.shape), (1, 3, 3, 256, 256))
        torch.testing.assert_close(rgb.mean(dim=(2, 3, 4)), torch.tensor([[-1., 127/127.5-1, 1.]]))
        for field in ('action', 'state'):
            actual = (~self.backend._static_dimension_is_pad(self.processor, field)).nonzero().flatten().tolist()
            self.assertEqual(actual, self.backend.VALID_DIMS)

    def test_rectangular_rgb_matches_training_square_crop(self):
        # The center is already 256x256, so crop correctness is independent
        # of the resize implementation. Out-of-view border sentinels must vanish.
        for height, width in ((256, 384), (384, 256), (256, 385), (385, 256)):
            with self.subTest(shape=(height, width)):
                square_obs, raw_obs = {'observation': {}}, {'observation': {}}
                for index, name in enumerate(self.backend.CAMERAS):
                    y, x = np.indices((256, 256))
                    square = np.stack((x, y, np.full_like(x, 40 + index * 50)), -1).astype(np.uint8)
                    image = np.full((height, width, 3), [255, 0, 255], np.uint8)
                    top, left = (height - 256) // 2, (width - 256) // 2
                    image[top:top + 256, left:left + 256] = square
                    square_obs['observation'][name] = {'rgb': square}
                    raw_obs['observation'][name] = {'rgb': image}
                original = raw_obs['observation']['head_camera']['rgb'].copy()
                actual = self.backend.current_rgb(self.processor, raw_obs)
                expected = self.backend.current_rgb(self.processor, square_obs)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                np.testing.assert_array_equal(raw_obs['observation']['head_camera']['rgb'], original)
                for index, view in enumerate(actual[0]):
                    torch.testing.assert_close(view[:, 100, 150],
                        torch.tensor([150, 100, 40 + index * 50]) / 127.5 - 1)

    def test_camera_640x480_equals_explicit_480_square(self):
        rng = np.random.default_rng(42)
        raw = {'observation': {name: {'rgb': rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)}
                              for name in self.backend.CAMERAS}}
        square = {'observation': {name: {'rgb': raw['observation'][name]['rgb'][:, 80:560].copy()}
                                 for name in self.backend.CAMERAS}}
        torch.testing.assert_close(self.backend.current_rgb(self.processor, raw),
                                   self.backend.current_rgb(self.processor, square), rtol=0, atol=0)

    def test_real_session_batch_isolation_acknowledgments_and_reset(self):
        runtime = SimpleNamespace(model=object(), processor=self.processor, device='cpu',
            model_dtype=torch.float32, action_dim_is_pad=self.backend._static_dimension_is_pad(self.processor, 'action'))
        config = SimpleNamespace(num_inference_steps=10, replan_steps=32, seed=0, rand_device='cpu')
        dims = dict(arm_dim=[6, 6], ee_dim=[1, 1])
        with patch('XPolicyLab.policy.Phoenix_MWA._adapter_base.get_robot_action_dim_info', return_value=dims), patch.object(Model, '_load_real_policy'):
            model = Model(dict(runtime_mode='joint96_real'))
        model.session_factory = partial(self.backend.RoboDojoRealJoint96Session, runtime, config)
        model.session = model.session_factory()
        with patch.object(self.backend.RoboDojoRealJoint96Session, '_infer_action_chunk',
                          return_value=(np.zeros((32, 14), np.float32), torch.zeros(32, 80))):
            model.update_obs(observation())
            self.assertEqual(len(model.get_action()), 32)
            with self.assertRaises(RuntimeError):
                model.get_action()
            for _ in range(32):
                model.update_obs(observation())
            self.assertEqual(len(model.get_action()), 32)
            model.update_obs_batch([observation(7), observation(2)])
            self.assertEqual(len(model.get_action_batch([2, 7])), 2)
            self.assertIsNot(model.env_sessions[7], model.env_sessions[2])
            for _ in range(32):
                model.update_obs_batch([observation(7)])
            self.assertFalse(model.env_sessions[7].pending_model_actions)
            self.assertEqual(len(model.env_sessions[2].pending_model_actions), 32)
            model.reset()
            self.assertFalse(model.session.pending_model_actions)
            self.assertTrue(all(not x.pending_model_actions for x in model.env_sessions.values()))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
