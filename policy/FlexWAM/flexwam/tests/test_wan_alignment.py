"""WAN adapter contracts, using real small transformers and a deterministic VAE."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from flexwam.config import compose_task
from flexwam.models.architectures.builder import create_flexwam
from flexwam.models.architectures.flexwam import FlexWAM
from flexwam.models.architectures.mot import MoT
from flexwam.models.backbone.wan22.action_dit import ActionDiT
from flexwam.models.backbone.wan22.wan_video import WanVideoDiT
from flexwam.training.trainer import FlexWAMTrainer


class FixtureWanVAE(torch.nn.Module):
    upsampling_factor = 16
    temporal_downsample_factor = 4
    model = SimpleNamespace(z_dim=2)

    def encode(self, video, device=None, **kwargs):
        if not torch.is_tensor(video):
            video = torch.stack(list(video))
        video = video.to(device=device)[:, :2, ::4]
        b, c, t, h, w = video.shape
        return F.avg_pool2d(video.transpose(1, 2).reshape(b*t, c, h, w), 16).reshape(
            b, t, c, h//16, w//16).transpose(1, 2)


def video_config():
    return dict(hidden_dim=48, ffn_dim=80, num_layers=4, num_heads=2, attn_head_dim=24,
                in_dim=2, out_dim=2, text_dim=16, freq_dim=16, eps=1e-6,
                patch_size=(1, 2, 2), has_image_input=False, seperated_timestep=True,
                num_views=3, use_camera_embedding=True, cross_view_attention_interval=2,
                video_attention_mask_mode='first_frame_causal')


def action_config():
    return dict(hidden_dim=24, ffn_dim=48, num_layers=4, num_heads=2, attn_head_dim=24,
                text_dim=16, freq_dim=16, eps=1e-6, action_dim=3)


def head_config():
    return dict(num_views=3, feature_layer_index=2, feature_dim=16, pool_grid=[2, 2],
                num_queries=2, num_heads=2, spatial_hidden_dim=24,
                branch_output_dim=16, head_hidden_dim=24)


def make_wan(*, state=False, drop_prob=0.0, intervention='pixel', context_input_dim=16,
             checkpoint=False, dtype=torch.float32, state_mode='separate', video_state=None):
    torch.manual_seed(118)
    video = WanVideoDiT(**video_config(), use_gradient_checkpointing=checkpoint)
    action = ActionDiT(**action_config(), use_gradient_checkpointing=checkpoint)
    head = {**head_config(), 'state_conditioning': {'enabled': state, 'hidden_dim': 8}}
    model = FlexWAM(
        video_expert=video, action_expert=action,
        mot=MoT({'video': video, 'action': action}, mot_checkpoint_mixed_attn=checkpoint),
        vae=FixtureWanVAE(), text_dim=16, context_input_dim=context_input_dim,
        proprio_dim=3, torch_dtype=dtype, action_router_config=head,
        action_critic_config={**head, 'feature_layer_index': 1},
        video_state_config={'enabled': state if video_state is None else video_state},
        state_conditioning={'mode': state_mode, 'drop_prob': drop_prob},
        router_cf_config={'interval': 1, 'intervention': intervention, 'uniform_probe': True,
                          'action_sigma_min': 0.6, 'action_sigma_max': 1.0},
    ).to(dtype=dtype)
    with torch.no_grad():
        model.action_camera_router.head[-1].weight.normal_(std=0.05)
        model.action_camera_critic.head[-1].weight.normal_(std=0.05)
    return model


def sample(context_dim=16):
    torch.manual_seed(19)
    return dict(video=torch.randn(1, 5, 3, 3, 32, 64), action=torch.randn(1, 8, 3),
                proprio=torch.randn(1, 8, 3), context=torch.randn(1, 4, context_dim),
                context_mask=torch.tensor([[True, True, True, False]]),
                action_is_pad=torch.tensor([[False]*7 + [True]]),
                image_is_pad=torch.zeros(1, 5, dtype=torch.bool))


def prepare_video(model, latents, state, context, keep=None):
    return model._video_expert_pre_dit(
        x=latents, timestep=torch.tensor([0.5]), context=context,
        proprio=state, state_keep_mask=keep,
    )


def run_video(model, pre):
    length = pre['tokens'].shape[1]
    mask = model._build_mot_attention_mask(length, 9, pre['meta']['tokens_per_frame'],
        torch.device('cpu'), action_prefix_len=1, num_views=3, video_meta=pre['meta'])
    return model._forward_video_branch(pre, mask, None, snapshot_layer_indices=(2,)), mask


class WanAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_native_config_views_heads_text_and_deployment(self):
        from deploy.policy import _model_config_for_deploy
        wan = compose_task('penbox_v2', overrides=['model=wan22', 'data.roots=[/tmp/wan-test]'])
        ling = compose_task('penbox_v2')
        self.assertEqual(wan.model.video_dit_config.num_views, 3)
        self.assertTrue(wan.model.video_dit_config.use_camera_embedding)
        self.assertEqual(wan.model.video_dit_config.cross_view_attention_interval, 2)
        self.assertEqual(wan.model.action_attention.mode, 'logit_bias')
        self.assertEqual(wan.model.action_attention.bias_strength, 1.0)
        self.assertEqual(wan.model.action_scheduler.train_shift, 1.0)
        self.assertEqual(wan.model.action_scheduler.infer_shift, 1.0)
        self.assertIsNone(wan.model.video_dit_config.cross_view_attention_layers)
        self.assertEqual(ling.model.video_dit_config.cross_view_attention_interval, 4)
        for role in ('action_router', 'action_critic'):
            wan_head, ling_head = dict(wan.model[role]), dict(ling.model[role])
            self.assertEqual(wan_head.pop('feature_layer_index'), 19)
            self.assertEqual(ling_head.pop('feature_layer_index'), 18)
            self.assertEqual(wan_head, ling_head)
        self.assertFalse(wan.model.video_dit_config.action_conditioned)
        for split in ('train', 'val'):
            self.assertEqual(wan.data[split].text_embedding_cache_encoder_id, 'wan22ti2v5b')
        runtime = _model_config_for_deploy(wan)
        self.assertTrue(runtime.load_text_encoder)
        self.assertTrue(runtime.skip_dit_load_from_pretrain)
        self.assertEqual(runtime.tokenizer_max_len, wan.data.train.context_len)
        self.assertFalse(wan.model.load_text_encoder)
        self.assertFalse(_model_config_for_deploy(ling).load_text_encoder)
        adapted = compose_task('penbox_v2', overrides=['model=wan22', 'model.context_input_dim=2560'])
        self.assertEqual(adapted.data.train.text_embedding_cache_encoder_id, 'lingbotqwen3vl')
        with self.assertRaisesRegex(ValueError, 'native UMT5'):
            _model_config_for_deploy(adapted)

    def test_spatial_constraints_in_config_training_and_inference(self):
        for image_size in ('[192,240]', '[208,224]'):
            with self.subTest(image_size=image_size):
                override = f'data.observation.image_size={image_size}'
                with self.assertRaisesRegex(ValueError, 'multiples of 32'):
                    compose_task('penbox_v2', overrides=['model=wan22', override])
                compose_task('penbox_v2', overrides=[override])
        model = make_wan()
        model._validate_video_spatial_size(192, 224)
        for height, width in ((48, 64), (32, 48)):
            with self.subTest(height=height, width=width):
                data = sample()
                data['video'] = torch.randn(1, 5, 3, 3, height, width)
                with patch.object(model.vae, 'encode', side_effect=AssertionError('VAE must not run')) as encode:
                    with self.assertRaisesRegex(ValueError, 'multiples of 32'):
                        model.training_loss(data)
                    kwargs = dict(prompt=None, input_image=data['video'][:, 0], action_horizon=8,
                        proprio=data['proprio'][:, 0], context=data['context'],
                        context_mask=data['context_mask'], num_inference_steps=1)
                    with self.assertRaisesRegex(ValueError, 'multiples of 32'):
                        model.infer_action(**kwargs)
                    with self.assertRaisesRegex(ValueError, 'multiples of 32'):
                        model.infer_joint(**kwargs, num_video_frames=5)
                    encode.assert_not_called()

    def test_gt_action_position_encoding_and_causal_isolation(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(21)
                video = WanVideoDiT(**video_config(), action_conditioned=True, action_dim=3,
                                   action_group_causal_mask_mode='group_diagonal').to(dtype=dtype)
                video.configure_state_conditioning(3, {'enabled': True})
                latent = torch.randn(1, 3, 2, 3, 2, 4, dtype=dtype)
                context = torch.randn(1, 4, 16, dtype=dtype)
                action = torch.zeros(1, 8, 3, dtype=dtype)
                state = torch.randn(1, 3, dtype=dtype)
                timestep = torch.tensor([0.5], dtype=dtype)
                args = dict(x=latent, timestep=timestep, context=context, action=action, proprio=state)
                pre = video.pre_dit(**args)
                # Equal actions must still have distinct, non-integer positional encodings.
                positions = torch.arange(8, dtype=torch.float64)[:, None]
                frequencies = 10000 ** (-torch.arange(24, dtype=torch.float64)/24)
                phase = positions * frequencies
                expected = torch.cat([phase.cos(), phase.sin()], dim=-1).to(dtype)
                expected = video.action_embedding(action) + expected.unsqueeze(0)
                torch.testing.assert_close(pre['context'][:, 4:], expected)
                self.assertFalse(torch.equal(pre['context'][:, 5], pre['context'][:, 6]))
                mask = pre['context_mask'][0]
                self.assertFalse(mask[0, 4:].any())  # Shared state cannot read actions.
                for view in range(3):
                    start = 1 + view * 6
                    self.assertFalse(mask[start:start+2, 4:].any())  # Clean first frame.
                    self.assertTrue(mask[start+2:start+4, 4:8].all())
                    self.assertFalse(mask[start+2:start+4, 8:].any())
                    self.assertTrue(mask[start+4:start+6, 8:].all())
                first = video(**args)
                changed = video(**{**args, 'action': action+1})
                torch.testing.assert_close(first[:, :, :, :1], changed[:, :, :, :1], rtol=0, atol=0)
                self.assertFalse(torch.equal(first[:, :, :, 1:], changed[:, :, :, 1:]))
                first[:, :, :, 1:].float().square().mean().backward()
                self.assertTrue(video.action_embedding.weight.grad.isfinite().all())
                self.assertGreater(video.action_embedding.bias.grad.abs().sum().item(), 0)

    def test_wan_cache_never_falls_back_to_qwen(self):
        from flexwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
        dataset = object.__new__(RobotVideoDataset)
        dataset.context_len = 4
        dataset.text_embedding_cache_encoder_id = 'wan22ti2v5b'
        digest = hashlib.sha256(b'test').hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            dataset.text_embedding_cache_dir = directory
            torch.save({'context': torch.ones(4, 2560), 'mask': torch.ones(4, dtype=torch.bool)},
                       Path(directory)/f'{digest}.text_len4.lingbotqwen3vl.pt')
            with self.assertRaisesRegex(FileNotFoundError, 'precompute_text_embeds.py'):
                dataset._get_cached_text_context('test')
            torch.save({'context': torch.ones(4, 4096), 'mask': torch.ones(4, dtype=torch.bool)},
                       Path(directory)/f'{digest}.t5_len4.wan22ti2v5b.pt')
            self.assertEqual(dataset._get_cached_text_context('test')[0].shape, (4, 4096))

    def test_preprocessing_accepts_dataset_dependent_wan_preset(self):
        from scripts.preprocess.preprocess_action_dit_backbone import _load_model_config
        from flexwam.models.backbone.wan22.loader import _validate_dit_config
        root = Path(__file__).resolve().parents[1]
        video, action, _ = _load_model_config(root/'configs/model/wan22.yaml')
        _validate_dit_config(video)
        self.assertEqual(video['num_views'], 1)
        self.assertFalse(video['use_camera_embedding'])
        self.assertNotIn('state_conditioning', video)
        self.assertFalse(action['use_gradient_checkpointing'])

    def test_factory_wires_state_and_default_cross_view_layers(self):
        captured = {}
        def load(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(dit=WanVideoDiT(**kwargs['dit_config']), vae=FixtureWanVAE(),
                text_encoder=None, tokenizer=None, dit_path='fixture', vae_path='fixture',
                text_encoder_path=None, tokenizer_path=None)
        config = video_config()
        del config['cross_view_attention_interval']
        config['state_conditioning'] = {'enabled': True}
        with patch('flexwam.models.architectures.builder.load_wan22_ti2v_5b_components', side_effect=load):
            model = create_flexwam(model_id='fixture', tokenizer_model_id='fixture',
                video_dit_config=config, action_dit_config=action_config(),
                proprio_dim=3, device='cpu', model_dtype=torch.float32,
                context_input_dim=12, state_conditioning={'drop_prob': 0.2}, use_cuda_graph=True,
                action_router=head_config(), action_scheduler={
                    'train_shift': 5, 'infer_shift': 5, 'num_train_timesteps': 1000})
        self.assertNotIn('state_conditioning', captured['dit_config'])
        self.assertEqual(model.video_expert.cross_view_layer_indices, (3,))
        self.assertIsNotNone(model.video_expert.state_proj)
        self.assertEqual(model.state_drop_prob, 0.2)
        self.assertTrue(model.use_cuda_graph)
        self.assertIsNotNone(model.context_adapter)
        self.assertIn('state_conditioning', config)

    def test_action_state_prefix_and_direct_language_gradients(self):
        model = make_wan()
        data = sample()
        context = model._build_video_action_contexts(data['context'], data['context_mask'], None)
        self.assertIs(context['action_context'], context['video_context'])
        state = data['proprio'][:, 0]
        def pre(t):
            return model._action_expert_pre_dit(action_tokens=data['action'], timestep=torch.tensor([t]),
                action_context=context['action_context'], action_context_mask=context['action_context_mask'], proprio=state)
        first, other = pre(0.2), pre(0.8)
        self.assertEqual(first['tokens'].shape, (1, 9, 24))
        torch.testing.assert_close(first['t_mod'][:, 0], torch.zeros_like(first['t_mod'][:, 0]))
        torch.testing.assert_close(first['tokens'][:, 0], other['tokens'][:, 0])
        torch.testing.assert_close(first['freqs'][1:, 0], model.action_expert.freqs[:8])
        self.assertEqual(first['context'].shape, (1, 4, 24))
        self.assertEqual(first['context_mask'].shape, (1, 9, 4))
        out = model.action_expert(data['action'], torch.tensor([0.5]), data['context'],
                                 data['context_mask'], model._action_state_tokens(state))
        self.assertEqual(out.shape, (1, 8, 3))
        out.square().mean().backward()
        for param in (model.state_proj.weight, model.action_expert.text_embedding[0].weight,
                      model.action_expert.blocks[0].cross_attn.v.weight):
            self.assertGreater(param.grad.abs().sum().item(), 0)

    def test_video_state_causality_dropout_and_direct_forward(self):
        model = make_wan(state=True).eval()
        latent, state, context = torch.randn(1, 3, 2, 2, 2, 4), torch.randn(1, 3), torch.randn(1, 4, 16)
        with torch.no_grad():
            pre = prepare_video(model, latent, state, context)
            out, mask = run_video(model, pre)
            length = pre['tokens'].shape[1]
            self.assertFalse(mask[:1, 1:length].any())
            self.assertFalse(mask[length:, :1].any())
            self.assertFalse(mask[length, length+1:].any())
            changed = latent.clone()
            changed[:, :, :, 1:] += 10
            future, _ = run_video(model, prepare_video(model, changed, state, context))
            torch.testing.assert_close(out['tokens'][:, :1], future['tokens'][:, :1], rtol=0, atol=0)
            features = model._extract_first_frame_view_tokens(out['tokens'], pre['meta'])
            torch.testing.assert_close(features, model._extract_first_frame_view_tokens(future['tokens'], pre['meta']))
            different, _ = run_video(model, prepare_video(model, latent, state+1, context))
            self.assertFalse(torch.equal(features, model._extract_first_frame_view_tokens(different['tokens'], pre['meta'])))
            keep = torch.zeros(1, 1, dtype=torch.bool)
            a, b = prepare_video(model, latent, state, context, keep), prepare_video(model, latent, state+1, context, keep)
            torch.testing.assert_close(a['tokens'], b['tokens'], rtol=0, atol=0)
            direct = model.video_expert(latent, torch.tensor([0.5]), context, proprio=state)
            torch.testing.assert_close(direct, model.video_expert.post_dit(out['tokens'], pre))
            self.assertEqual(direct.shape, latent.shape)

    def test_training_stages_and_resampled_counterfactuals(self):
        for intervention in ('pixel', 'router_zero'):
            for stage in (1, 2, 3, 4, 5):
                with self.subTest(intervention=intervention, stage=stage):
                    model = make_wan(state=True, intervention=intervention, checkpoint=True)
                    model.set_train_progress(0, stage=stage, stage4_progress=0.5)
                    self.assertEqual(model.action_camera_router_train_enabled, stage >= 2)
                    self.assertEqual(model.action_camera_critic_train_enabled, stage in (3, 4))
                    loss, _ = model.training_loss(sample())
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    ctx = model._deferred_pixel_cf_ctx
                    if stage == 5:
                        self.assertIsNone(ctx)
                    if ctx is not None:
                        cf_loss, metrics = model.run_deferred_pixel_critic_cf()
                        self.assertEqual(metrics['router_cf_ran'], 1.0)
                        self.assertTrue(torch.isfinite(cf_loss))
                        cf_loss.backward()
                        self.assertEqual(ctx['action_pre']['input_context'].shape, (1, 4, 16))
                        self.assertEqual(ctx['action_pre']['input_context_mask'].shape, (1, 4))
                        if stage == 4:
                            self.assertGreater(metrics['router_cf_kl_lambda'], 0.0)
                            self.assertIn('loss_router_cf_kl', metrics)
                        else:
                            self.assertEqual(metrics['router_cf_kl_lambda'], 0.0)
                            self.assertNotIn('loss_router_cf_kl', metrics)
                    self.assertGreater(model.action_expert.text_embedding[0].weight.grad.abs().sum().item(), 0)

    def test_disabled_heads_stay_off_across_stages(self):
        model = make_wan()
        model.camera_router_enabled = False
        model.camera_critic_enabled = False
        model.action_camera_router.requires_grad_(False)
        model.action_camera_critic.requires_grad_(False)
        for stage in (1, 2, 3, 4, None):
            with self.subTest(stage=stage):
                model.zero_grad(set_to_none=True)
                model.set_train_progress(10, stage=stage, stage4_progress=0.5)
                self.assertFalse(model.action_camera_router_train_enabled)
                self.assertFalse(model.action_camera_critic_train_enabled)
                loss, metrics = model.training_loss(sample())
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertIsNone(model._deferred_pixel_cf_ctx)
                self.assertNotIn('loss_router_cf', metrics)
                self.assertIsNone(model.action_camera_router.head[-1].weight.grad)
                self.assertIsNone(model.action_camera_critic.head[-1].weight.grad)
                router_weights = [value for key, value in metrics.items() if key.startswith('router_')]
                self.assertEqual(len(router_weights), 3)
                for weight in router_weights:
                    self.assertAlmostEqual(weight, 1.0 / 3.0, places=5)

    def test_disabled_critic_keeps_router_on_action_loss(self):
        model = make_wan()
        model.camera_critic_enabled = False
        model.action_camera_critic.requires_grad_(False)
        model.action_camera_critic.eval()
        for stage in (1, 2, 3, 4, None):
            with self.subTest(stage=stage):
                model.zero_grad(set_to_none=True)
                model.set_train_progress(10, stage=stage, stage4_progress=0.5)
                router_on = stage is None or stage >= 2
                self.assertEqual(model.action_camera_router_train_enabled, router_on)
                self.assertFalse(model.action_camera_critic_train_enabled)
                loss, metrics = model.training_loss(sample())
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertIsNone(model._deferred_pixel_cf_ctx)
                self.assertNotIn('loss_router_cf', metrics)
                self.assertIsNone(model.action_camera_critic.head[-1].weight.grad)
                if router_on:
                    self.assertIsNotNone(model.action_camera_router.head[-1].weight.grad)
                else:
                    self.assertIsNone(model.action_camera_router.head[-1].weight.grad)

    def test_cache_matches_joint_and_inference_responds_to_text_and_state(self):
        for enabled in (False, True):
            with self.subTest(video_state=enabled), torch.no_grad():
                model = make_wan(state=enabled).eval()
                data = sample()
                state = data['proprio'][:, 0]
                context = model._build_video_action_contexts(data['context'], data['context_mask'], state)
                latent = model._encode_video_latents(data['video'].permute(0, 2, 3, 1, 4, 5))
                tv, ta = torch.tensor([0.5]), torch.tensor([0.7])
                _, joint = model._predict_joint_noise(latent, data['action'], tv, ta, **context, proprio=state)
                pre = model._video_expert_pre_dit(x=latent[:, :, :, :1], timestep=tv,
                    context=context['video_context'], context_mask=context['video_context_mask'], proprio=state)
                video, mask = run_video(model, pre)
                features = model._extract_first_frame_view_tokens(video['layer_tokens'][2], pre['meta'])
                router = model._build_action_router_output(pre, features, proprio=state)
                eager = model._predict_action_noise_with_cache_eager(data['action'], ta,
                    context['action_context'], context['action_context_mask'], video['kv_cache'], mask,
                    pre['tokens'].shape[1], video_meta=pre['meta'],
                    **model._action_router_mot_kwargs(router), proprio=state)
                torch.testing.assert_close(joint, eager, rtol=1e-5, atol=1e-5)
                kwargs = dict(prompt=None, input_image=data['video'][:, 0], action_horizon=8,
                    proprio=state, context=data['context'], context_mask=data['context_mask'],
                    num_inference_steps=2, seed=71)
                action = model.infer_action(**kwargs)['action']
                self.assertTrue(torch.isfinite(action).all())
                for change in ({'context': data['context']+1}, {'proprio': state+1}):
                    self.assertFalse(torch.equal(action, model.infer_action(**{**kwargs, **change})['action']))

    def test_optional_state_dropout_and_adapter_checkpoint(self):
        model = make_wan(state=True, drop_prob=1.0, context_input_dim=12)
        self.assertFalse(model._sample_state_keep_mask(torch.randn(1, 3)).any())
        FlexWAMTrainer._apply_dit_only_train_mode(model)
        self.assertTrue(model.context_adapter.weight.requires_grad)
        self.assertFalse(model.training)
        self.assertFalse(model._sample_state_keep_mask(torch.randn(1, 3)).any())
        loss, _ = model.training_loss(sample(12))
        loss.backward()
        self.assertGreater(model.context_adapter.weight.grad.abs().sum().item(), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pt'
            model.save_checkpoint(path)
            clone = make_wan(state=True, drop_prob=1.0, context_input_dim=12)
            clone.load_checkpoint(path)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, clone.state_dict()[key], rtol=0, atol=0)

    def test_bfloat16_training_and_inference(self):
        model = make_wan(state=True, dtype=torch.bfloat16)
        data = sample()
        loss, _ = model.training_loss(data)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        result = model.infer_action(prompt=None, input_image=data['video'][:, 0], action_horizon=8,
            proprio=data['proprio'][:, 0], context=data['context'], context_mask=data['context_mask'],
            num_inference_steps=2, seed=71)
        self.assertTrue(torch.isfinite(result['action']).all())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for WAN graph replay')
    def test_action_cuda_graph_matches_eager_same_seed(self):
        if torch.cuda.mem_get_info()[0] < 2 * 1024**3:
            self.skipTest('CUDA graph test requires at least 2 GiB free GPU memory')
        model = make_wan(dtype=torch.bfloat16).to(device='cuda').eval()
        self.assertEqual(model.device.type, 'cuda')
        data = sample()
        kwargs = dict(
            prompt=None,
            input_image=data['video'][:, 0].to(device='cuda'),
            action_horizon=8,
            proprio=data['proprio'][:, 0].to(device='cuda'),
            context=data['context'].to(device='cuda'),
            context_mask=data['context_mask'].to(device='cuda'),
            num_inference_steps=3,
            seed=71,
        )
        model.use_cuda_graph = False
        eager = model.infer_action(**kwargs)['action']
        model.use_cuda_graph = True
        model._action_denoise_cuda_graph = None
        captured = model.infer_action(**kwargs)['action']
        graph = model._action_denoise_cuda_graph
        self.assertIsNotNone(graph)
        self.assertFalse(graph.failed)
        self.assertIsNotNone(graph.graph)
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
