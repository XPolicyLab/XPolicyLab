"""Independent camera-head feature flow and checkpoint compatibility."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from lingbot_parity import make_model

ROOT = Path(__file__).resolve().parents[1]


def independent_config(layer=10):
    return dict(feature_layer_index=layer, feature_dim=8,
                pool_grid=[1, 2], num_queries=1, num_heads=2,
                spatial_hidden_dim=12, branch_output_dim=8, head_hidden_dim=12)


def sample():
    torch.manual_seed(19)
    return dict(video=torch.randn(1, 5, 3, 3, 32, 48), action=torch.randn(1, 8, 3),
                proprio=torch.randn(1, 8, 3), context=torch.randn(1, 4, 16),
                context_mask=torch.tensor([[True, True, True, False]]),
                action_is_pad=torch.tensor([[False] * 7 + [True]]),
                image_is_pad=torch.zeros(1, 5, dtype=torch.bool))


class CameraHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_distinct_features_reach_both_heads_and_cf_interventions(self):
        for intervention in ('pixel', 'router_zero'):
            with self.subTest(intervention=intervention):
                model = make_model('cpu', torch.float32, independent_config(),
                                   router_cf_config={'intervention': intervention})
                self.assertEqual(model.router_cf_intervention, intervention)
                model.set_train_progress(0, stage=4, stage4_progress=0.5)
                forward = model._forward_video_branch
                snapshots = []

                def record(**kwargs):
                    result = forward(**kwargs)
                    snapshots.append((kwargs, result))
                    return result

                with patch.object(model, '_forward_video_branch', side_effect=record):
                    loss, _ = model.training_loss(sample())
                self.assertEqual(len(snapshots), 1)
                kwargs, output = snapshots[0]
                self.assertEqual(kwargs['snapshot_layer_indices'], (10, 18))
                ctx = model._deferred_pixel_cf_ctx
                self.assertIsNotNone(ctx)
                for role, layer in [('router', 18), ('critic', 10)]:
                    features = ctx[role + '_features']
                    expected = model._extract_first_frame_view_tokens(
                        output['layer_tokens'][layer], kwargs['video_pre']['meta'])
                    torch.testing.assert_close(features, expected, rtol=0, atol=0)
                    self.assertFalse(features.requires_grad)
                self.assertFalse(torch.equal(ctx['router_features'], ctx['critic_features']))
                self.assertEqual(model.action_camera_critic.feature_dim, 8)
                loss.backward()
                cf_loss, metrics = model.run_deferred_pixel_critic_cf()
                self.assertEqual(metrics['router_cf_ran'], 1.0)
                self.assertTrue(torch.isfinite(cf_loss))
                cf_loss.backward()
                for head in (model.action_camera_router, model.action_camera_critic):
                    grad = head.head[-1].weight.grad
                    self.assertIsNotNone(grad)
                    self.assertGreater(grad.abs().sum().item(), 0)

    def test_intervention_config_compatibility_and_validation(self):
        for config, expected in [
            ({}, 'pixel'),
            ({'probe': 'router_zero'}, 'router_zero'),
            ({'probe': 'router_zero', 'intervention': 'pixel'}, 'pixel'),
        ]:
            with self.subTest(config=config):
                model = make_model('cpu', torch.float32, router_cf_config=config)
                self.assertEqual(model.router_cf_intervention, expected)
        with self.assertRaisesRegex(ValueError, 'router_cf.intervention'):
            make_model('cpu', torch.float32, router_cf_config={'intervention': 'invalid'})
        self.assertEqual(make_model('cpu', torch.float32).router_cf_teacher, 'critic')
        with self.assertRaisesRegex(ValueError, 'router_cf.teacher'):
            make_model('cpu', torch.float32, router_cf_config={'teacher': 'invalid'})

    def test_direct_cf_supervises_router_without_critic(self):
        model = make_model(
            'cpu', torch.float32,
            router_cf_config={'teacher': 'cf', 'intervention': 'router_zero'},
        )
        self.assertTrue(model._router_cf_direct())
        batch = sample()

        model.set_train_progress(0, stage=2, stage4_progress=0.0)
        self.assertTrue(model.action_camera_router_train_enabled)
        self.assertFalse(model.action_camera_critic_train_enabled)
        with patch.object(model.action_camera_critic, 'forward', side_effect=AssertionError('critic executed')):
            loss, metrics = model.training_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNone(model._deferred_pixel_cf_ctx)
        self.assertNotIn('loss_router_cf', metrics)

        model.zero_grad(set_to_none=True)
        model.set_train_progress(0, stage=4, stage4_progress=0.25)
        with patch.object(model.action_camera_critic, 'forward', side_effect=AssertionError('critic executed')):
            loss, metrics = model.training_loss(batch)
        ctx = model._deferred_pixel_cf_ctx
        self.assertIsNotNone(ctx)
        self.assertTrue(ctx['direct_router'])
        self.assertIsNone(ctx['critic_features'])
        loss.backward()
        model.action_camera_router.zero_grad(set_to_none=True)
        with patch.object(model.action_camera_critic, 'forward', side_effect=AssertionError('critic executed')):
            cf_loss, cf_metrics = model.run_deferred_pixel_critic_cf()
        self.assertEqual(cf_metrics['router_cf_ran'], 1.0)
        self.assertEqual(cf_metrics['router_cf_direct_teacher'], 1.0)
        self.assertEqual(cf_metrics['router_cf_intervention_router_zero'], 1.0)
        self.assertNotIn('loss_router_cf_critic', cf_metrics)
        self.assertIn('loss_router_cf_kl', cf_metrics)
        self.assertGreater(cf_metrics['router_cf_kl_lambda'], 0.0)
        self.assertTrue(torch.isfinite(cf_loss))
        cf_loss.backward()
        cf_grad = model.action_camera_router.head[-1].weight.grad
        self.assertIsNotNone(cf_grad)
        self.assertGreater(cf_grad.abs().sum().item(), 0)
        self.assertIsNone(model.action_camera_critic.head[-1].weight.grad)

    def test_same_layer_reuses_features(self):
        model = make_model('cpu', torch.float32)
        model.set_train_progress(0, stage=3)
        with patch.object(model.mot, 'forward_video', wraps=model.mot.forward_video) as forward:
            model.training_loss(sample())
        self.assertEqual(forward.call_count, 1)
        self.assertEqual(forward.call_args.kwargs['snapshot_layer_indices'], (18,))
        ctx = model._deferred_pixel_cf_ctx
        self.assertEqual(ctx['router_features'].data_ptr(), ctx['critic_features'].data_ptr())

    def test_inference_only_snapshots_router_and_never_executes_critic(self):
        model = make_model('cpu', torch.float32, independent_config())
        batch = sample()
        with patch.object(model.action_camera_critic, 'forward', side_effect=AssertionError('critic executed')), \
             patch.object(model.mot, 'forward_video', wraps=model.mot.forward_video) as forward:
            model.infer_action(prompt=None, input_image=batch['video'][:, 0], action_horizon=8,
                               proprio=batch['proprio'][0, 0], context=batch['context'],
                               context_mask=batch['context_mask'], num_inference_steps=2, seed=71)
        self.assertTrue(forward.called)
        for call in forward.call_args_list:
            self.assertEqual(call.kwargs['snapshot_layer_indices'], (18,))

    def test_direct_factory_checks_layer_bounds(self):
        with self.assertRaisesRegex(ValueError, 'critic.feature_layer_index'):
            make_model('cpu', torch.float32, independent_config(layer=20))

    def test_checkpoint_metadata_legacy_load_and_explicit_reinitialization(self):
        cache = ROOT / '.cache' / 'head-checkpoints'
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            path = Path(directory) / 'checkpoint.pt'
            model = make_model('cpu', torch.float32)
            model.save_checkpoint(path, optimizer=torch.optim.AdamW(model.parameters()))
            payload = torch.load(path, weights_only=False)
            self.assertEqual(payload['action_router_arch'], 'spatial_query_v1')
            self.assertEqual(payload['action_critic_arch'], 'spatial_query_v1')
            self.assertNotIn('same_router', payload['action_critic_config'])
            self.assertNotIn('action_router_feature_source', payload)

            legacy = {k: v for k, v in payload.items()
                      if k not in {'action_router_config', 'action_critic_config',
                                   'action_critic_arch', 'action_critic_feature_layer_index'}}
            legacy['action_router_arch'] = 'l18_spatial_query_v1'
            legacy['action_router_feature_source'] = 'l18'
            torch.save(legacy, path)
            clone = make_model('cpu', torch.float32)
            clone.load_checkpoint(path)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(clone.state_dict()[key], value, rtol=0, atol=0)

            independent = make_model('cpu', torch.float32, independent_config(layer=18))
            with self.assertRaises((ValueError, RuntimeError)):
                independent.load_checkpoint(path)
            independent = make_model('cpu', torch.float32, independent_config(layer=18))
            initial = {k: v.clone() for k, v in independent.action_camera_critic.state_dict().items()}
            independent.load_checkpoint(path, reinit_incompatible_heads=True)
            for key, value in initial.items():
                torch.testing.assert_close(independent.action_camera_critic.state_dict()[key], value, rtol=0, atol=0)
            optimizer = torch.optim.AdamW(independent.parameters())
            with self.assertRaisesRegex(ValueError, 'optimizer'):
                independent.load_checkpoint(path, optimizer=optimizer, reinit_incompatible_heads=True)

            independent.save_checkpoint(path)
            restored = make_model('cpu', torch.float32, independent_config(layer=18))
            restored.load_checkpoint(path)
            for key, value in independent.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
