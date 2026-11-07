"""Native heterogeneous channel architecture; no training/mask dependencies."""
from torch import nn
from ..wan_video_dit import WanVideoDiT,I2VCrossAttention,precompute_freqs_cis_3d


def build(spec,device='cpu'):
    if spec.get('schema')!='vpp2-compact-channels-v1':raise ValueError('Unknown channel architecture')
    g=spec['geometry'];selection=spec['selection'];h=len(selection['hidden_indices'])
    def valid(ids,maximum):
        return ids==sorted(set(ids)) and len(ids)>0 and all(type(i) is int and 0<=i<maximum for i in ids)
    if not valid(selection['hidden_indices'],g['hidden']):raise ValueError('Invalid hidden coordinate identities')
    for key,total in (('ffn_indices',g['ffn']),('cross_head_indices',g['heads'])):
        if len(selection[key])!=g['layers'] or any(not valid(x,total) for x in selection[key]):
            raise ValueError('Invalid per-layer identities')
        if selection[key][-1]!=list(range(total)):raise ValueError('Terminal internals must remain complete')
    args=dict(spec['model_kwargs']);args.update(dim=h,hidden_dim=h,ffn_dim=g['ffn'],num_layers=g['layers'],
                                               num_heads=g['heads'],attn_head_dim=g['head_dim'])
    import torch
    with torch.device(device):
        model=WanVideoDiT(**args)
        for i,b in enumerate(model.blocks):
            f=len(selection['ffn_indices'][i]);heads=len(selection['cross_head_indices'][i])
            b.ffn[0]=nn.Linear(h,f);b.ffn[2]=nn.Linear(f,h);b.ffn_dim=f
            b.cross_attn=I2VCrossAttention(h,g['head_dim'],heads,b.cross_attn.norm_q.eps,True)
    model.freqs=precompute_freqs_cis_3d(g['head_dim'])
    if sum(p.numel() for p in model.parameters())!=spec['parameters']:raise ValueError('Physical channel parameter count mismatch')
    return model
