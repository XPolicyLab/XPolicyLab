"""Optional state paths: causal masks, gradients, CF replay, cache and migration."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lingbot_parity import make_model
from test_camera_heads import sample
from flexwam.training.trainer import FlexWAMTrainer
from flexwam.models.architectures.builder import create_lingbot_flexwam


def enabled_model(**kwargs):
    return make_model('cpu', torch.float32,
                      video_state_config={'enabled': True},
                      head_state_config={'enabled': True, 'hidden_dim': 8}, **kwargs)


def video_pre(model, latents, state, keep=None):
    return model._video_expert_pre_dit(
        x=latents, timestep=torch.tensor([0.7]), context=torch.ones(1, 4, 16),
        proprio=state, state_keep_mask=keep,
    )


def run_video(model, pre):
    mask = model._build_mot_attention_mask(
        pre['tokens'].shape[1], 9, pre['meta']['tokens_per_frame'], torch.device('cpu'),
        action_prefix_len=1, num_views=3, video_meta=pre['meta'],
    )
    out = model._forward_video_branch(pre, mask, None, snapshot_layer_indices=(18,))
    return out, mask


class StateConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_defaults_preserve_parameters_rng_and_inputs(self):
        old = make_model('cpu', torch.float32)
        explicit = make_model('cpu', torch.float32, video_state_config={'enabled': False},
                              head_state_config={'enabled': False}, state_conditioning={'drop_prob': 0.0})
        self.assertEqual(old.state_dict().keys(), explicit.state_dict().keys())
        for k, value in old.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[k], rtol=0, atol=0)
        rng = torch.random.get_rng_state().clone()
        self.assertIsNone(old._sample_state_keep_mask(torch.ones(2, 3)))
        torch.testing.assert_close(rng, torch.random.get_rng_state(), rtol=0, atol=0)
        latent = torch.randn(1, 3, 2, 2, 4, 6)
        first = video_pre(old, latent, torch.zeros(1, 3))
        other = video_pre(old, latent, torch.ones(1, 3))
        self.assertEqual(first['meta']['state_len'], 0)
        torch.testing.assert_close(first['tokens'], other['tokens'], rtol=0, atol=0)

    def test_pretrained_factory_passes_independent_state_options(self):
        base = make_model('cpu', torch.float32)
        components = SimpleNamespace(transformer=base.video_expert, vae=base.vae,
                                     transformer_path='fixture', vae_path='fixture')
        head = {**base.action_router_config, 'state_conditioning': {'enabled': True, 'hidden_dim': 8}}
        with patch('flexwam.models.backbone.lingbot_video.load_lingbot_components', return_value=components), \
             patch('flexwam.models.backbone.lingbot_video.action_dit.LingBotActionDiT.from_pretrained',
                   return_value=base.action_expert):
            model = create_lingbot_flexwam(
                model_path='fixture', proprio_dim=3, device='cpu', model_dtype=torch.float32,
                video_dit_config={'num_views': 3, 'text_dim': 16, 'state_conditioning': {'enabled': True}},
                action_dit_config={}, action_router=head, action_critic=base.action_critic_config,
                action_scheduler={'train_shift': 5, 'infer_shift': 5, 'num_train_timesteps': 1000},
                state_conditioning={'drop_prob': 0.2},
            )
        self.assertIsNotNone(model.video_expert.state_proj)
        self.assertIsNotNone(model.action_camera_router.state_encoder)
        self.assertIsNone(model.action_camera_critic.state_encoder)
        self.assertEqual(model.state_drop_prob, 0.2)

    def test_full_training_resume_rejects_state_changes_before_loading(self):
        model = enabled_model()
        accelerator = SimpleNamespace(unwrap_model=lambda _: model)
        trainer = SimpleNamespace(model=model, accelerator=accelerator)
        with tempfile.TemporaryDirectory() as directory:
            # Without metadata this is a legacy training state, not a migration source.
            with self.assertRaisesRegex(ValueError, 'Legacy full training state'):
                FlexWAMTrainer.load_training_state(trainer, directory)

    def test_prefix_masks_and_first_frame_do_not_read_future(self):
        model = enabled_model().eval()
        latent, s = torch.randn(1, 3, 2, 2, 4, 6), torch.randn(1, 3)
        with torch.no_grad():
            pre = video_pre(model, latent, s)
            out, mask = run_video(model, pre)
            length, prefix = pre['tokens'].shape[1], pre['meta']['prefix_len']
            self.assertEqual(prefix, 5)
            self.assertEqual(pre['meta']['state_len'], 1)
            language_len = int(pre['meta']['context_len'])
            self.assertEqual(language_len, 4)
            self.assertFalse(mask[:prefix, prefix:].any())
            self.assertTrue(mask[:prefix, :prefix].all())
            self.assertTrue(mask[length:, :language_len].all())
            self.assertFalse(mask[length:, language_len:prefix].any())
            self.assertFalse(mask[length, length + 1:].any())
            changed = latent.clone()
            changed[:, :, :, 1:] += 100
            future_out, _ = run_video(model, video_pre(model, changed, s))
            torch.testing.assert_close(out['tokens'][:, :prefix], future_out['tokens'][:, :prefix], rtol=0, atol=0)
            features = model._extract_first_frame_view_tokens(out['tokens'], pre['meta'])
            other = model._extract_first_frame_view_tokens(future_out['tokens'], pre['meta'])
            torch.testing.assert_close(features, other, rtol=0, atol=0)
            state_out, _ = run_video(model, video_pre(model, latent, s + 1))
            self.assertFalse(torch.equal(features, model._extract_first_frame_view_tokens(state_out['tokens'], pre['meta'])))

    def test_dropout_masks_embeddings_but_keeps_action_state(self):
        model = enabled_model(state_conditioning={'drop_prob': 1.0})
        s = torch.randn(1, 3)
        keep = model._sample_state_keep_mask(s)
        self.assertFalse(keep.any())
        latent = torch.randn(1, 3, 2, 1, 4, 6)
        a, b = video_pre(model, latent, s, keep), video_pre(model, latent, s + 1, keep)
        torch.testing.assert_close(a['tokens'], b['tokens'], rtol=0, atol=0)
        self.assertFalse(torch.equal(model._action_state_tokens(s), model._action_state_tokens(s + 1)))
        features = torch.randn(1, 3, 6, 48)
        for head in (model.action_camera_router, model.action_camera_critic):
            a = head(features, (2, 3), True, s, keep)
            b = head(features, (2, 3), True, s + 1, keep)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertFalse(torch.equal(head(features, (2, 3), True, s), head(features, (2, 3), True, s + 1)))
        model.eval()
        self.assertIsNone(model._sample_state_keep_mask(s))

    def test_gradients_and_cf_reuse_one_mask(self):
        for intervention in ('pixel', 'router_zero'):
            with self.subTest(intervention=intervention):
                model = enabled_model(state_conditioning={'drop_prob': 0.2},
                                      router_cf_config={'intervention': intervention})
                FlexWAMTrainer._apply_dit_only_train_mode(model)
                model.set_train_progress(0, stage=4)
                keep = torch.tensor([[True], [False]])
                batch = {key: value.repeat(2, *([1] * (value.ndim - 1))) for key, value in sample().items()}
                seen = []
                original = model.video_expert.pre_dit

                def record(**kwargs):
                    seen.append(kwargs['state_keep_mask'].clone())
                    return original(**kwargs)

                with patch.object(model, '_sample_state_keep_mask', return_value=keep) as sampler, \
                     patch.object(model.video_expert, 'pre_dit', side_effect=record):
                    loss, _ = model.training_loss(batch)
                    loss.backward()
                    self.assertGreater(model.video_expert.state_proj.weight.grad.abs().sum().item(), 0)
                    self.assertGreater(model.action_camera_router.state_encoder[0].weight.grad.abs().sum().item(), 0)
                    self.assertEqual(sampler.call_count, 1)
                    self.assertFalse(model._deferred_pixel_cf_ctx['critic_features'].requires_grad)
                    model.zero_grad(set_to_none=True)
                    cf_loss, _ = model.run_deferred_pixel_critic_cf()
                    cf_loss.backward()
                    self.assertGreater(model.action_camera_critic.state_encoder[0].weight.grad.abs().sum().item(), 0)
                    self.assertGreater(model.action_camera_router.state_encoder[0].weight.grad.abs().sum().item(), 0)
                    self.assertIsNone(model.video_expert.state_proj.weight.grad)
                self.assertGreater(len(seen), 1)
                for mask in seen:
                    torch.testing.assert_close(mask, keep, rtol=0, atol=0)

    def test_cached_first_frame_matches_full_video_action(self):
        model = enabled_model().eval()
        latent, s = torch.randn(1, 3, 2, 2, 4, 6), torch.randn(1, 3)
        action_pre = model._action_expert_pre_dit(
            action_tokens=torch.randn(1, 8, 3), timestep=torch.ones(1),
            action_context=torch.empty(1, 0, 16), action_context_mask=torch.empty(1, 0, dtype=torch.bool),
            proprio=s,
        )
        actions = []
        with torch.no_grad():
            for x in (latent, latent[:, :, :, :1]):
                pre = video_pre(model, x, s)
                _, mask = run_video(model, pre)
                out, _, _ = model._run_mot_video_action(
                    video_pre=pre, action_pre=action_pre, proprio=s,
                    attention_mask=mask, include_critic=False,
                )
                actions.append(model.action_expert.post_dit(out['action'], action_pre))
        torch.testing.assert_close(actions[0], actions[1], rtol=1e-5, atol=1e-5)

    def test_inference_prefill_receives_new_state_every_call(self):
        model, batch = enabled_model().eval(), sample()
        seen = []
        original = model.video_expert.pre_dit

        def record(**kwargs):
            seen.append(kwargs['proprio'].clone())
            return original(**kwargs)

        with patch.object(model.video_expert, 'pre_dit', side_effect=record), \
             patch.object(model.action_camera_critic, 'forward', side_effect=AssertionError('critic in inference')):
            for s in (torch.zeros(3), torch.ones(3)):
                result = model.infer_action(prompt=None, input_image=batch['video'][:, 0], action_horizon=8,
                                            proprio=s, context=batch['context'], context_mask=batch['context_mask'],
                                            num_inference_steps=2, seed=71)
                self.assertIn('action', result)
        self.assertEqual(len(seen), 2)
        self.assertFalse(torch.equal(seen[0], seen[1]))

    def test_checkpoint_explicit_migration_and_round_trip(self):
        old, model = make_model('cpu', torch.float32), enabled_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            old.save_checkpoint(path)
            with self.assertRaisesRegex(ValueError, 'migrate_state_conditioning'):
                model.load_checkpoint(path)
            with self.assertRaisesRegex(ValueError, 'optimizer'):
                model.load_checkpoint(path, optimizer=torch.optim.AdamW(model.parameters()), migrate_state_conditioning=True)
            model.load_checkpoint(path, migrate_state_conditioning=True)
            features, s = torch.randn(1, 3, 6, 48), torch.randn(1, 3)
            for name in ('action_camera_router', 'action_camera_critic'):
                expected = getattr(old, name)(features, (2, 3), True)
                actual = getattr(model, name)(features, (2, 3), True, s)
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
            model.save_checkpoint(path, optimizer=torch.optim.AdamW(model.parameters()))
            clone = enabled_model()
            clone.load_checkpoint(path, optimizer=torch.optim.AdamW(clone.parameters()))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, clone.state_dict()[key], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'enable'):
                old.load_checkpoint(path)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for graph replay')
    def test_cuda_graph_replays_updated_state_and_video_cache(self):
        if torch.cuda.mem_get_info()[0] < 2 * 1024**3:
            self.skipTest('CUDA graph test requires at least 2 GiB free GPU memory')
        model = make_model('cuda', torch.bfloat16,
                           video_state_config={'enabled': True},
                           head_state_config={'enabled': True, 'hidden_dim': 8}).eval()
        batch = sample()
        kwargs = dict(prompt=None, input_image=batch['video'][:, 0], action_horizon=8,
                      context=batch['context'], context_mask=batch['context_mask'],
                      num_inference_steps=3, seed=71)
        for s in (torch.zeros(3), torch.ones(3)):
            model.use_cuda_graph = False
            eager = model.infer_action(proprio=s, **kwargs)['action']
            model.use_cuda_graph = True
            captured = model.infer_action(proprio=s, **kwargs)['action']
            graph = model._action_denoise_cuda_graph
            self.assertIsNotNone(graph)
            self.assertFalse(graph.failed)
            self.assertIsNotNone(graph.graph)
            torch.testing.assert_close(captured, eager, rtol=0.01, atol=0.01)


if __name__ == '__main__':
    unittest.main()
