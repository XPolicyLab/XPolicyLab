"""Shared state visibility, cached inference, and legacy separate-state loading."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from flexwam.config import compose_task
from flexwam.training.trainer import FlexWAMTrainer
from lingbot_parity import make_model
from test_wan_alignment import make_wan, sample


def model_for(backbone, mode='shared', video=False, drop_prob=0.0):
    if backbone == 'wan':
        return make_wan(state_mode=mode, video_state=video, drop_prob=drop_prob)
    return make_model('cpu', torch.float32, video_state_config={'enabled': video},
                      state_conditioning={'mode': mode, 'drop_prob': drop_prob})


def inputs(model, state, frames=2):
    torch.manual_seed(37)
    context = model._build_video_action_contexts(
        torch.randn(1, 4, 16), torch.tensor([[True, True, True, False]]), state)
    latent = torch.randn(1, 3, 2, frames, 4, 4)
    action = torch.randn(1, 8, 3)
    video_pre = model._video_expert_pre_dit(
        proprio=state, x=latent, timestep=torch.tensor([0.5]),
        context=context['video_context'], context_mask=context['video_context_mask'])
    action_pre = model._action_expert_pre_dit(
        action_tokens=action, timestep=torch.tensor([0.7]), proprio=state,
        action_context=context['action_context'], action_context_mask=context['action_context_mask'])
    length = video_pre['tokens'].shape[1]
    mask = model._build_mot_attention_mask(
        length, action_pre['tokens'].shape[1], video_pre['meta']['tokens_per_frame'],
        torch.device('cpu'), action_prefix_len=model._action_prefix_len(action_pre),
        num_views=3, video_meta=video_pre['meta'])
    return video_pre, action_pre, mask, latent, action, context


class StateModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_config_default_and_both_backbones(self):
        for backbone in ('wan22', 'lingbot_video'):
            for mode in ('separate', 'shared'):
                cfg = compose_task('penbox_v2', overrides=[f'model={backbone}',
                    f'model.state_conditioning.mode={mode}', 'model.video.state_conditioning.enabled=true'])
                self.assertEqual(cfg.model.state_conditioning.mode, mode)
                self.assertTrue(cfg.model.video_dit_config.state_conditioning.enabled)
        self.assertEqual(compose_task('penbox_v2').model.state_conditioning.mode, 'separate')
        with self.assertRaisesRegex(ValueError, 'mode'):
            compose_task('penbox_v2', overrides=['model.state_conditioning.mode=typo'])

    def test_one_shared_prefix_and_video_switch(self):
        for backbone in ('wan', 'lingbot'):
            for enabled in (False, True):
                with self.subTest(backbone=backbone, video=enabled):
                    model = model_for(backbone, video=enabled)
                    state = torch.randn(1, 3)
                    pre, act, mask, *_ = inputs(model, state)
                    length, prefix = pre['tokens'].shape[1], pre['meta']['prefix_len']
                    self.assertIsNone(model.video_expert.state_proj)
                    self.assertEqual(model._action_prefix_len(act), 0)
                    self.assertEqual(act['tokens'].shape[1], 8)
                    torch.testing.assert_close(pre['tokens'][:, :1], model._project_state(state))
                    self.assertTrue(mask[length:, 0].all())
                    self.assertEqual(bool(mask[1:length, 0].all()), enabled)
                    self.assertFalse(mask[:prefix, prefix:].any())
                    if backbone == 'lingbot':
                        self.assertEqual(pre['meta']['context_start'], 1)
                        self.assertTrue(mask[length:, 1:prefix].all())

    def test_state_perturbation_and_no_future_leakage(self):
        for backbone in ('wan', 'lingbot'):
            for enabled in (False, True):
                with self.subTest(backbone=backbone, video=enabled), torch.no_grad():
                    model = model_for(backbone, video=enabled).eval()
                    outputs = []
                    for state in (torch.zeros(1, 3), torch.ones(1, 3)):
                        pre, act, mask, *_ = inputs(model, state)
                        out, _, _ = model._run_mot_video_action(
                            video_pre=pre, action_pre=act, proprio=state, attention_mask=mask)
                        outputs.append(out)
                    self.assertFalse(torch.equal(outputs[0]['action'], outputs[1]['action']))
                    if enabled:
                        self.assertFalse(torch.equal(outputs[0]['video'][:, 1:], outputs[1]['video'][:, 1:]))
                    else:
                        # Includes LingBot language tokens: video cannot recover state through text.
                        torch.testing.assert_close(outputs[0]['video'][:, 1:], outputs[1]['video'][:, 1:], rtol=0, atol=0)
                    state = torch.zeros(1, 3)
                    pre, act, mask, latent, _, context = inputs(model, state)
                    latent[:, :, :, 1:] += 9
                    changed = model._video_expert_pre_dit(proprio=state, x=latent,
                        timestep=torch.tensor([0.5]), context=context['video_context'],
                        context_mask=context['video_context_mask'])
                    out, _, _ = model._run_mot_video_action(
                        video_pre=changed, action_pre=act, proprio=state, attention_mask=mask)
                    torch.testing.assert_close(outputs[0]['action'], out['action'], rtol=0, atol=0)

    def test_cached_joint_inference_and_shared_state_gradients(self):
        for backbone in ('wan', 'lingbot'):
            for enabled in (False, True):
                with self.subTest(backbone=backbone, video=enabled):
                    model = model_for(backbone, video=enabled, drop_prob=1.0)
                    state = torch.ones(1, 3)
                    pre, act, mask, latent, action, context = inputs(model, state)
                    out, _, _ = model._run_mot_video_action(
                        video_pre=pre, action_pre=act, proprio=state, attention_mask=mask)
                    model.action_expert.post_dit(out['action'], act).square().mean().backward()
                    self.assertGreater(model.state_proj.bias.grad.abs().sum().item(), 0)
                    self.assertGreater(model.state_proj.weight.grad.abs().sum().item(), 0)
                    model.eval()
                    with torch.no_grad():
                        tv, ta = torch.tensor([0.5]), torch.tensor([0.7])
                        _, joint = model._predict_joint_noise(latent, action, tv, ta, **context, proprio=state)
                        pre = model._video_expert_pre_dit(proprio=state, x=latent[:, :, :, :1], timestep=tv,
                            context=context['video_context'], context_mask=context['video_context_mask'])
                        length = pre['tokens'].shape[1]
                        mask = model._build_mot_attention_mask(length, 8, pre['meta']['tokens_per_frame'],
                            torch.device('cpu'), num_views=3, video_meta=pre['meta'])
                        video = model._forward_video_branch(pre, mask, state,
                            snapshot_layer_indices=(model.action_router_feature_layer_index,))
                        features = model._extract_first_frame_view_tokens(
                            video['layer_tokens'][model.action_router_feature_layer_index], pre['meta'])
                        router = model._build_action_router_output(pre, features, proprio=state)
                        cached = model._predict_action_noise_with_cache_eager(action, ta,
                            context['action_context'], context['action_context_mask'], video['kv_cache'],
                            mask, length, pre['meta'], proprio=state, **model._action_router_mot_kwargs(router))
                        torch.testing.assert_close(cached, joint, rtol=1e-5, atol=1e-6)
                        image = torch.randn(1, 3, 3, 32, 32)
                        kwargs = dict(prompt=None, input_image=image, action_horizon=8, num_inference_steps=2,
                                      context=context['video_context'], context_mask=context['video_context_mask'], seed=17)
                        first = model.infer_action(proprio=state, **kwargs)['action']
                        second = model.infer_action(proprio=state + 1, **kwargs)['action']
                        self.assertFalse(torch.equal(first, second))

    def test_shared_and_legacy_checkpoint_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            for backbone in ('wan', 'lingbot'):
                for mode in ('separate', 'shared'):
                    for enabled in (False, True):
                        with self.subTest(backbone=backbone, mode=mode, video=enabled):
                            original = model_for(backbone, mode, enabled)
                            original.save_checkpoint(path)
                            if mode == 'separate':
                                payload = torch.load(path, weights_only=False)
                                if 'state_conditioning' in payload:
                                    payload['state_conditioning'].pop('mode', None)
                                    payload['state_conditioning']['version'] = 1
                                torch.save(payload, path)
                            clone = model_for(backbone, mode, enabled)
                            clone.load_checkpoint(path)
                            for key, value in original.state_dict().items():
                                torch.testing.assert_close(value, clone.state_dict()[key], rtol=0, atol=0)
                            other_mode = model_for(backbone, 'shared' if mode == 'separate' else 'separate', enabled)
                            with self.assertRaisesRegex(ValueError, 'mode mismatch'):
                                other_mode.load_checkpoint(path)

    def test_legacy_full_training_signature_defaults_to_separate(self):
        model = model_for('lingbot', 'separate', True)
        signature = model.state_conditioning_signature()
        signature.pop('mode')
        accelerator = Mock()
        accelerator.num_processes = 1
        accelerator.unwrap_model.return_value = model
        trainer = SimpleNamespace(model=model, accelerator=accelerator, train_sampler=Mock(), batch_size=1)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'trainer_state.json').write_text(json.dumps(dict(
                state_conditioning=signature, global_step=2, epoch=0, batch_in_epoch=1)))
            FlexWAMTrainer.load_training_state(trainer, directory)
            accelerator.load_state.assert_called_once_with(input_dir=directory)
            accelerator.unwrap_model.return_value = model_for('lingbot', 'shared', True)
            with self.assertRaisesRegex(ValueError, 'State conditioning config differs'):
                FlexWAMTrainer.load_training_state(trainer, directory)

    def test_shared_training_stages_and_counterfactuals(self):
        for backbone in ('wan', 'lingbot'):
            for enabled in (False, True):
                with self.subTest(backbone=backbone, video=enabled):
                    model = model_for(backbone, video=enabled)
                    for stage in (1, 2, 3, 4):
                        model.zero_grad(set_to_none=True)
                        model.set_train_progress(0, stage=stage, stage4_progress=0.5)
                        loss, _ = model.training_loss(sample())
                        self.assertTrue(torch.isfinite(loss))
                        loss.backward()
                        if stage in (1, 2):
                            self.assertIsNotNone(model.state_proj.weight.grad)
                        if model._deferred_pixel_cf_ctx is not None:
                            cf_loss, metrics = model.run_deferred_pixel_critic_cf()
                            self.assertEqual(metrics['router_cf_ran'], 1.0)
                            self.assertTrue(torch.isfinite(cf_loss))
                            cf_loss.backward()


if __name__ == '__main__':
    unittest.main()
