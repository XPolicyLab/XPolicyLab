"""Deterministic LingBot regression snapshot; compare against a pre-refactor run.

Uses the real transformer/router/critic code with small widths and a deterministic
VAE stand-in. No downloaded weights or robot hardware are needed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import tempfile

import torch
import torch.nn.functional as F

from flexwam.models.architectures.flexwam import FlexWAM
from flexwam.models.architectures.mot import MoT
from flexwam.models.backbone.lingbot_video.lingbot_video import LingBotVideoTransformer3DModel
from flexwam.models.backbone.lingbot_video.action_dit import LingBotActionDiT


class FixtureVAE(torch.nn.Module):
    upsampling_factor = 8
    temporal_downsample_factor = 4
    model = SimpleNamespace(z_dim=2)

    def encode(self, video, device=None, **kwargs):
        if not torch.is_tensor(video):
            video = torch.stack(list(video))
        video = video.to(device=device)[:, :2, ::4]
        b, c, t, h, w = video.shape
        return F.avg_pool2d(video.transpose(1, 2).reshape(b*t, c, h, w), 8).reshape(b, t, c, h//8, w//8).transpose(1, 2)

    def decode(self, latents, device=None, **kwargs):
        latents = latents.to(device=device)
        b, c, t, h, w = latents.shape
        pixels = F.interpolate(latents.transpose(1, 2).reshape(b*t, c, h, w).float(), scale_factor=8, mode='nearest')
        pixels = torch.cat([pixels, pixels[:, :1]], dim=1)
        return pixels.reshape(b, t, 3, h*8, w*8).transpose(1, 2).to(latents.dtype)


def make_model(device, dtype, critic_config=None, router_cf_config=None, *,
               video_state_config=None, head_state_config=None, state_conditioning=None):
    torch.manual_seed(117)
    video = LingBotVideoTransformer3DModel(
        hidden_size=48, intermediate_size=80, depth=20,
        num_attention_heads=2, in_channels=2, out_channels=2,
        text_dim=16, freq_dim=16, axes_dims=(8,8,8), axes_lens=(64,32,32),
        num_views=3, use_camera_embedding=True, cross_view_attention_interval=4,
    )
    video.video_attention_mask_mode = 'first_frame_causal'
    video.use_gradient_checkpointing = False
    action = LingBotActionDiT(action_dim=3, hidden_size=48, intermediate_size=80,
        depth=20, num_attention_heads=2, freq_dim=16, axes_dims=(8,8,8), axes_lens=(64,32,32))
    model = FlexWAM(video_expert=video, action_expert=action,
        mot=MoT({'video': video, 'action': action}, mot_checkpoint_mixed_attn=False),
        vae=FixtureVAE(), text_dim=16, proprio_dim=3, device=device,
        torch_dtype=dtype, action_router_config={'num_views':3, 'feature_layer_index':18,
        'feature_dim':16, 'pool_grid':[2,2], 'num_queries':2, 'num_heads':2,
        'spatial_hidden_dim':24, 'branch_output_dim':16, 'head_hidden_dim':24,
        **({'state_conditioning': head_state_config} if head_state_config is not None else {})},
        action_critic_config=critic_config,
        video_state_config=video_state_config,
        state_conditioning=state_conditioning,
        router_cf_config={'interval':1, 'uniform_probe':True,
        'action_sigma_min':0.6, 'action_sigma_max':1.0, **(router_cf_config or {})})
    model = model.to(device=device, dtype=dtype)
    # Exercise non-uniform routing instead of a zero-initialized head.
    with torch.no_grad():
        model.action_camera_router.head[-1].weight.normal_(std=0.05)
        model.action_camera_critic.head[-1].weight.normal_(std=0.05)
    return model


def snapshot(device='cpu', dtype=torch.float32):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    sample = {'video':torch.randn(1,5,3,3,32,48), 'action':torch.randn(1,8,3),
              'proprio':torch.randn(1,8,3), 'context':torch.randn(1,4,16),
              'context_mask':torch.tensor([[True,True,True,False]]),
              'action_is_pad':torch.tensor([[False]*7+[True]]),
              'image_is_pad':torch.zeros(1,5,dtype=torch.bool)}
    model = make_model(device, dtype)
    values = {}
    def capture(prefix, obj):
        if torch.is_tensor(obj):
            values[prefix] = obj.detach().cpu().clone()
        elif isinstance(obj, dict):
            for k,v in obj.items():
                capture(f'{prefix}/{k}', v)
        elif isinstance(obj, (list,tuple)):
            for i,v in enumerate(obj): capture(f'{prefix}/{i}',v)
        elif isinstance(obj, (float,int,bool,str)):
            values[prefix] = obj
    capture('initial_state', model.state_dict())
    for intervention in ['pixel','router_zero']:
        model.router_cf_intervention = intervention
        for stage in [1,2,3,4]:
            model.train()
            model.set_train_progress(0, stage=stage, stage4_progress=0.5)
            model.zero_grad(set_to_none=True)
            torch.manual_seed(42)
            loss, metrics = model.training_loss(sample)
            loss.backward()
            cf_loss, cf_metrics = model.run_deferred_pixel_critic_cf()
            if cf_loss is not None: cf_loss.backward()
            tag=f'{intervention}/stage{stage}'
            capture(tag+'/loss', loss)
            capture(tag+'/metrics', metrics)
            capture(tag+'/cf_loss', cf_loss)
            capture(tag+'/cf_metrics', cf_metrics)
            for name,param in model.named_parameters():
                capture(tag+'/grad/'+name, param.grad)
    # Actual update and checkpoint round-trip.
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.step()
    capture('updated_state', model.state_dict())
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'model.pt'
        model.save_checkpoint(path, optimizer=optimizer, step=7)
        capture('checkpoint', torch.load(path, weights_only=False))
        clone=make_model(device, dtype)
        clone.load_checkpoint(path)
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value, clone.state_dict()[key], rtol=0, atol=0)
    model.eval()
    out=model.infer_action(prompt=None, input_image=sample['video'][:,0],
        action_horizon=8, proprio=sample['proprio'][0,0],
        context=sample['context'], context_mask=sample['context_mask'],
        num_inference_steps=3, seed=71, return_action_attention_stats=True)
    for key,value in out.items():
        if 'timing' not in key: capture('inference/'+key,value)
    # Ensure multiple tensor-valued outputs were captured, including actions.
    assert any(k.startswith('inference/') and torch.is_tensor(v) for k,v in values.items())
    return values


def compare(reference, actual):
    # Head metadata intentionally changed; numerical and optimizer payloads must
    # still match the independently recorded original snapshots exactly.
    metadata = ('checkpoint/action_router_arch', 'checkpoint/action_router_feature_source',
                'checkpoint/action_router_config', 'checkpoint/action_critic_arch',
                'checkpoint/action_critic_feature_layer_index', 'checkpoint/action_critic_config')
    def numerical_payload(values):
        return {k: v for k, v in values.items()
                if not any(k == prefix or k.startswith(prefix + '/') for prefix in metadata)}
    reference, actual = numerical_payload(reference), numerical_payload(actual)
    assert reference.keys()==actual.keys(), f'Snapshot keys changed: {reference.keys() ^ actual.keys()}'
    for key, expected in reference.items():
        value=actual[key]
        if torch.is_tensor(expected):
            torch.testing.assert_close(value,expected,rtol=0,atol=0,msg=lambda m:f'{key}: {m}')
        else:
            assert value==expected, f'{key}: {value!r} != {expected!r}'
    print(f'PASS: {len(actual)} values exactly match reference (rtol=0, atol=0).')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save',type=Path)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--dtype',choices=['float32','bfloat16'],default='float32')
    args=parser.parse_args()
    result=snapshot(args.device,getattr(torch,args.dtype))
    if args.save:
        torch.save(result,args.save)
        print(f'Saved {len(result)} values to {args.save}')
    if args.reference: compare(torch.load(args.reference,weights_only=False), result)
