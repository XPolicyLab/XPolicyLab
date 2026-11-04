"""VGGT-style cross-view layers: selection, view exchange, and first-frame causality."""
import unittest

import torch

from lingbot_parity import make_model
from test_wan_alignment import make_wan
from flexwam.models.architectures.structures import _video_visual_view_slices
from flexwam.models.backbone.common.cross_view_attention import resolve_cross_view_layers


def video_tokens(model, latents, context):
    pre = model._video_expert_pre_dit(
        x=latents, timestep=torch.tensor([0.5]), context=context, proprio=torch.zeros(1, 3),
    )
    length = pre['tokens'].shape[1]
    mask = model._build_mot_attention_mask(
        length, 9, pre['meta']['tokens_per_frame'], torch.device('cpu'),
        action_prefix_len=1, num_views=3, video_meta=pre['meta'],
    )
    tokens = model._forward_video_branch(pre, mask, None)['tokens']
    frame = int(pre['meta'].get('video_tokens_per_frame', pre['meta']['tokens_per_frame']))
    views = [tokens[:, start:end] for start, end in _video_visual_view_slices(length, pre['meta'])]
    return views, frame


class CrossViewAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_layer_selection(self):
        self.assertEqual(resolve_cross_view_layers(24, 4), (3, 7, 11, 15, 19, 23))
        self.assertEqual(resolve_cross_view_layers(30, 4), (3, 7, 11, 15, 19, 23, 27))
        self.assertEqual(resolve_cross_view_layers(24, 4, layers=[11, 5, 5]), (5, 11))
        self.assertEqual(resolve_cross_view_layers(24, 0), ())
        with self.assertRaisesRegex(ValueError, 'outside depth'):
            resolve_cross_view_layers(24, 4, layers=[24])
        with self.assertRaisesRegex(ValueError, '>= 0'):
            resolve_cross_view_layers(24, -1)

    def test_flexwam_requires_cross_view_layers(self):
        from flexwam.models.architectures.flexwam import FlexWAM
        model = make_wan()
        model.video_expert.configure_cross_view_attention(0)
        with self.assertRaisesRegex(ValueError, 'cross-view attention layer'):
            FlexWAM(video_expert=model.video_expert, action_expert=model.action_expert,
                    mot=model.mot, vae=model.vae, text_dim=16, proprio_dim=3)

    def test_views_exchange_only_at_matching_times(self):
        backbones = {
            'lingbot': (lambda: make_model('cpu', torch.float32), (1, 3, 2, 2, 4, 6)),
            'wan': (make_wan, (1, 3, 2, 2, 2, 4)),
        }
        for name, (build, shape) in backbones.items():
            with self.subTest(backbone=name), torch.no_grad():
                model = build().eval()
                torch.manual_seed(5)
                latent, context = torch.randn(shape), torch.randn(1, 4, 16)
                base, frame = video_tokens(model, latent, context)

                later = latent.clone()
                later[:, 1, :, 1:] += 10
                changed, _ = video_tokens(model, later, context)
                for view in range(3):
                    torch.testing.assert_close(changed[view][:, :frame], base[view][:, :frame], rtol=0, atol=0)
                self.assertFalse(torch.equal(changed[0][:, frame:], base[0][:, frame:]))

                first = latent.clone()
                first[:, 1, :, :1] += 10
                changed, _ = video_tokens(model, first, context)
                self.assertFalse(torch.equal(changed[0][:, :frame], base[0][:, :frame]))

                model.video_expert.configure_cross_view_attention(layers=[])
                isolated, _ = video_tokens(model, latent, context)
                perturbed, _ = video_tokens(model, first, context)
                torch.testing.assert_close(perturbed[0], isolated[0], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
