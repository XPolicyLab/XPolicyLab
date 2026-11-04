"""Joint router attention: numerical stability, CF removal and cached inference."""
import math
import unittest

import torch
import torch.nn.functional as F

from flexwam.inference.cuda_graph import _ActionDenoiseCudaGraph
from flexwam.models.architectures.mot import _router_logit_bias_mask
from test_state_modes import inputs, model_for
from test_wan_alignment import make_wan, sample


class ActionAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_joint_attention_matches_multiplicative_prior(self):
        torch.manual_seed(51)
        q, k, v = torch.randn(2, 2, 2, 4), torch.randn(2, 2, 8, 4), torch.randn(2, 2, 8, 4)
        allowed = torch.ones(2, 8, dtype=torch.bool)
        allowed[1, 2] = False
        slices = [(1, 3), (3, 5), (5, 7)]  # Shared prefix at 0, action key at 7.
        for scales in (torch.ones(2, 3), torch.tensor([[1.8, .9, .3], [.3, .9, 1.8]])):
            bias = _router_logit_bias_mask(allowed, slices, scales, 1., q.dtype)
            actual = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
            logits = q @ k.transpose(-2, -1) / math.sqrt(4)
            prior = torch.ones(2, 8)
            for camera, (start, end) in enumerate(slices):
                prior[:, start:end] = scales[:, camera, None]
            mass = (logits - logits.amax(-1, keepdim=True)).exp() * prior[:, None, None, :]
            mass = mass * allowed[None, None, :, :]
            expected = (mass / mass.sum(-1, keepdim=True)) @ v
            torch.testing.assert_close(actual, expected)
            if torch.equal(scales, torch.ones_like(scales)):
                torch.testing.assert_close(actual, F.scaled_dot_product_attention(q, k, v, attn_mask=allowed))

    def test_camera_removal_is_independent_of_bias_strength(self):
        torch.manual_seed(52)
        allowed = torch.ones(2, 8, dtype=torch.bool)
        allowed[1, 4] = False
        slices = [(1, 3), (3, 5), (5, 7)]
        scales = torch.tensor([[0., 1.5, 1.5], [1.5, 0., 1.5]])
        for dtype in (torch.float32, torch.bfloat16):
            q, k, v = (torch.randn(*shape).to(dtype) for shape in ((2, 2, 2, 4), (2, 2, 8, 4), (2, 2, 8, 4)))
            for strength in (0., 1.):
                for use_logs in (False, True):
                    with self.subTest(dtype=dtype, strength=strength, logs=use_logs):
                        logs = scales.clamp_min(1).log().masked_fill(scales == 0, -torch.inf) if use_logs else None
                        bias = _router_logit_bias_mask(allowed, slices, scales, strength, dtype, logs)
                        self.assertTrue(torch.isneginf(bias[0, :, :, 1:3]).all())
                        self.assertTrue(torch.isneginf(bias[1, :, :, 3:5]).all())
                        self.assertTrue((bias[..., [0, 7]] == 0).all())
                        self.assertTrue(torch.isneginf(bias[:, :, 1, 4]).all())
                        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
                        changed = v.clone()
                        changed[0, :, 1:3] += 100
                        changed[1, :, 3:5] += 100
                        torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, changed, attn_mask=bias), rtol=0, atol=0)
                        if strength == 0:
                            reference_mask = allowed[None, None].expand(2, 1, -1, -1).clone()
                            reference_mask[0, :, :, 1:3] = False
                            reference_mask[1, :, :, 3:5] = False
                            torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v, attn_mask=reference_mask))

    def test_scale_only_zero_backward_is_finite(self):
        for strength in (0., 1.):
            scales = torch.tensor([[0., 1.5, 1.5]], requires_grad=True)
            bias = _router_logit_bias_mask(torch.ones(1, 4, dtype=torch.bool), [(0, 1), (1, 2), (2, 3)], scales, strength, torch.float32)
            q, k, v = torch.ones(1, 1, 1, 4), torch.ones(1, 1, 4, 4), torch.arange(16.).reshape(1, 1, 4, 4)
            F.scaled_dot_product_attention(q, k, v, attn_mask=bias).sum().backward()
            self.assertTrue(torch.isfinite(scales.grad).all())
            self.assertEqual(scales.grad[0, 0].item(), 0.)

    def test_saturated_bf16_router_uses_finite_log_prior(self):
        model = make_wan(dtype=torch.bfloat16)
        model.action_attention_mode = 'logit_bias'
        model.set_train_progress(0, stage=2)
        head = model.action_camera_router.head[-1]
        with torch.no_grad():
            head.weight.zero_()
            head.bias.copy_(torch.tensor([-100., 0., 0.]))
        features = torch.randn(1, 3, 4, 48, dtype=torch.bfloat16)
        pre = {'tokens': features[:, 0], 'meta': {'num_views': 3, 'grid_size': (1, 2, 2)}}
        router = model._build_action_router_output(pre, features)
        self.assertEqual(router['weights'][0, 0].item(), 0.)
        self.assertEqual(router['log_scales'].dtype, torch.float32)
        self.assertTrue(torch.isfinite(router['log_scales']).all())
        allowed, slices = torch.ones(1, 4, dtype=torch.bool), [(0, 1), (1, 2), (2, 3)]
        # Rounding a probability to zero must not be interpreted as a CF drop.
        unbiased = _router_logit_bias_mask(allowed, slices, router['scales'], 0., torch.bfloat16, router['log_scales'])
        self.assertTrue((unbiased == 0).all())
        bias = _router_logit_bias_mask(allowed, slices, router['scales'], 1., torch.bfloat16, router['log_scales'])
        q, k = torch.ones(1, 1, 1, 4, dtype=torch.bfloat16), torch.ones(1, 1, 4, 4, dtype=torch.bfloat16)
        v = torch.arange(16, dtype=torch.bfloat16).reshape(1, 1, 4, 4)
        F.scaled_dot_product_attention(q, k, v, attn_mask=bias).float().sum().backward()
        self.assertTrue(torch.isfinite(head.bias.grad).all())
        self.assertGreater(head.bias.grad.float().abs().sum().item(), 0.)
        dropped = model._router_output_zero_view(router, 1)
        self.assertTrue(torch.isneginf(dropped['log_scales'][0, 1]))
        self.assertTrue(torch.isfinite(router['log_scales']).all())
        dropped_bias = _router_logit_bias_mask(allowed, slices, dropped['scales'], 0., torch.bfloat16, dropped['log_scales'])
        self.assertTrue(torch.isneginf(dropped_bias[..., 1]).all())
        self.assertTrue((dropped_bias[..., 0] == 0).all())

    @staticmethod
    def cached_inputs(model, state):
        pre, act, mask, latent, action, context = inputs(model, state, frames=1)
        video = model._forward_video_branch(pre, mask, state, snapshot_layer_indices=(model.action_router_feature_layer_index,))
        features = model._extract_first_frame_view_tokens(video['layer_tokens'][model.action_router_feature_layer_index], pre['meta'])
        router = model._build_action_router_output(pre, features, proprio=state)
        kwargs = dict(latents_action=action, timestep_action=torch.tensor([.7]),
                      context=context['action_context'], context_mask=context['action_context_mask'],
                      video_kv_cache=video['kv_cache'], attention_mask=mask,
                      video_seq_len=pre['tokens'].shape[1], video_meta=pre['meta'], proprio=state,
                      **model._action_router_mot_kwargs(router))
        return kwargs, latent, context

    def test_cached_inference_matches_joint_for_both_backbones_and_modes(self):
        for backbone in ('wan', 'lingbot'):
            for state_mode in ('shared', 'separate'):
                for attention_mode in ('additive', 'logit_bias'):
                    with self.subTest(backbone=backbone, state=state_mode, attention=attention_mode), torch.no_grad():
                        model = model_for(backbone, mode=state_mode, video=True).eval()
                        model.action_attention_mode = attention_mode
                        model.action_attention_bias_strength = .35
                        state = torch.ones(1, 3)
                        kwargs, latent, context = self.cached_inputs(model, state)
                        _, expected = model._predict_joint_noise(latent, kwargs['latents_action'], torch.tensor([.5]), torch.tensor([.7]), **context, proprio=state)
                        actual = model._predict_action_noise_with_cache(**kwargs)
                        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_cuda_graph_static_buffers_copy_log_priors(self):
        # Exercise graph input staging on CPU as well; actual CUDA capture has a
        # separate conditional test below.
        with torch.no_grad():
            model = make_wan(state_mode='shared', video_state=True).eval()
            model.action_attention_mode = 'logit_bias'
            kwargs, _, _ = self.cached_inputs(model, torch.ones(1, 3))
            graph = _ActionDenoiseCudaGraph(model._predict_action_noise_with_cache_eager, torch.device('cpu'))
            graph._allocate(kwargs)
            first = graph._call_eager_static()
            torch.testing.assert_close(first, model._predict_action_noise_with_cache_eager(**kwargs))
            kwargs['action_video_router_log_scales'] = torch.tensor([[-torch.inf, math.log(1.5), math.log(1.5)]])
            kwargs['action_video_router_weights'] = torch.tensor([[0., 1.5, 1.5]])
            graph._copy_inputs(kwargs)
            changed = graph._call_eager_static()
            torch.testing.assert_close(changed, model._predict_action_noise_with_cache_eager(**kwargs))
            self.assertFalse(torch.allclose(first, changed))

    def test_training_backward_and_cf_with_zero_or_nonzero_bias(self):
        for dtype in (torch.float32, torch.bfloat16):
            for strength in (0., 1.):
                with self.subTest(dtype=dtype, strength=strength):
                    model = make_wan(state_mode='shared', video_state=True, intervention='router_zero', checkpoint=True, dtype=dtype).train()
                    model.action_attention_mode = 'logit_bias'
                    model.action_attention_bias_strength = strength
                    model.set_train_progress(0, stage=4, stage4_progress=.5)
                    loss, _ = model.training_loss(sample())
                    loss.backward()
                    cf, metrics = model.run_deferred_pixel_critic_cf()
                    cf.backward()
                    self.assertTrue(torch.isfinite(loss) and torch.isfinite(cf))
                    self.assertGreater(metrics['router_cf_delta_std'], 0.)
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                    self.assertGreater(model.action_camera_router.head[-1].weight.grad.float().norm().item(), 0.)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for graph capture')
    def test_logit_bias_cuda_graph_matches_eager(self):
        if torch.cuda.mem_get_info()[0] < 2 * 1024**3:
            self.skipTest('Requires at least 2 GiB free GPU memory')
        model = make_wan(state_mode='shared', video_state=True, dtype=torch.bfloat16).to('cuda').eval()
        model.action_attention_mode = 'logit_bias'
        model.action_attention_bias_strength = .35
        data = sample()
        kwargs = dict(prompt=None, input_image=data['video'][:, 0].cuda(), action_horizon=8,
                      proprio=data['proprio'][:, 0].cuda(), context=data['context'].cuda(),
                      context_mask=data['context_mask'].cuda(), num_inference_steps=2, seed=71)
        eager = model.infer_action(**kwargs)['action']
        model.use_cuda_graph = True
        captured = model.infer_action(**kwargs)['action']
        self.assertFalse(model._action_denoise_cuda_graph.failed)
        self.assertIsNotNone(model._action_denoise_cuda_graph.graph)
        torch.testing.assert_close(captured, eager, rtol=1e-5, atol=1e-5)
        kwargs['input_image'] = kwargs['input_image'] * .5
        replayed = model.infer_action(**kwargs)['action']
        model.use_cuda_graph = False
        torch.testing.assert_close(replayed, model.infer_action(**kwargs)['action'], rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
