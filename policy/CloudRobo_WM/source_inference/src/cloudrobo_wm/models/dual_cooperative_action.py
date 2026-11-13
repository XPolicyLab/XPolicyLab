from __future__ import annotations
import copy
import torch
from torch import nn
from .mot import MoT
ACTION_TOPOLOGY = 'dual_cooperative_robotwin_v1'
JG_INDICES = (74, 75, 76, 77, 78, 79, 6, 82, 83, 84, 85, 86, 87, 37)
EEF_INDICES = (0, 1, 2, 3, 4, 5, 31, 32, 33, 34, 35, 36)
ACTIVE_INDICES = JG_INDICES + EEF_INDICES
INACTIVE_INDICES = tuple((i for i in range(90) if i not in ACTIVE_INDICES))

def split_action(action):
    if action.shape[-1] != 90:
        raise ValueError('Cooperative RobotWin action input must have A90 fields')
    return (action[..., list(JG_INDICES)], action[..., list(EEF_INDICES)])

def merge_action(joint, eef):
    if joint.shape[:-1] != eef.shape[:-1] or joint.shape[-1] != 14 or eef.shape[-1] != 12:
        raise ValueError('Expected aligned JG14 and EEF12 predictions')
    if joint.device != eef.device or joint.dtype != eef.dtype:
        raise ValueError('Dual action heads must share dtype and device')
    result = joint.new_zeros((*joint.shape[:-1], 90))
    result[..., list(JG_INDICES)] = joint
    result[..., list(EEF_INDICES)] = eef
    return result

def validate_robotwin_action_valid(action_valid):
    if action_valid.shape[-1] != 90 or action_valid.dtype != torch.bool:
        raise ValueError('RobotWin action_valid must be a boolean A90 mask')
    if action_valid[..., list(INACTIVE_INDICES)].any().item():
        raise ValueError('Valid action labels outside RobotWin JG14/EEF12 are unsupported')

def _compact_copy(source, indices):
    if source.action_dim != 90:
        raise ValueError('Dual initialization requires a single A90 ActionDiT')
    tower = copy.deepcopy(source)
    tower.action_encoder._forward_hooks.clear()
    tower.action_encoder._forward_pre_hooks.clear()
    tower.action_dim = len(indices)
    tower.action_encoder.in_features = len(indices)
    tower.action_encoder.weight = nn.Parameter(source.action_encoder.weight[:, list(indices)].detach().clone(), requires_grad=source.action_encoder.weight.requires_grad)
    tower.head.out_features = len(indices)
    tower.head.weight = nn.Parameter(source.head.weight[list(indices)].detach().clone(), requires_grad=source.head.weight.requires_grad)
    if source.head.bias is not None:
        tower.head.bias = nn.Parameter(source.head.bias[list(indices)].detach().clone(), requires_grad=source.head.bias.requires_grad)
    return tower

class DualCooperativeActionExpert(nn.Module):

    def __init__(self, joint, eef):
        super().__init__()
        if joint.action_dim != 14 or eef.action_dim != 12:
            raise ValueError('Dual RobotWin experts must have dimensions 14 and 12')
        if joint.hidden_dim != eef.hidden_dim:
            raise ValueError('Packed dual action tokens require equal hidden widths')
        object.__setattr__(self, 'joint', joint)
        object.__setattr__(self, 'eef', eef)
        self.action_dim = 90
        self.hidden_dim = joint.hidden_dim
        self.action_topology = ACTION_TOPOLOGY

    @classmethod
    def from_single(cls, action):
        return cls(_compact_copy(action, JG_INDICES), _compact_copy(action, EEF_INDICES))

    @property
    def freqs(self):
        return self.joint.freqs

    @freqs.setter
    def freqs(self, value):
        self.joint.freqs = value
        self.eef.freqs = value

    def pre_dit(self, action_tokens, timestep, context, context_mask=None):
        joint, eef = split_action(action_tokens)
        pres = [tower.pre_dit(value, timestep, context, context_mask) for tower, value in ((self.joint, joint), (self.eef, eef))]
        return dict(tokens=torch.cat([p['tokens'] for p in pres], dim=1), freqs=torch.cat([p['freqs'] for p in pres], dim=0), t_mod=tuple((p['t_mod'] for p in pres)), context=tuple((p['context'] for p in pres)), context_mask=tuple((p['context_mask'] for p in pres)), branches=pres)

    def post_dit(self, tokens, pre_state):
        if tokens.shape[1] % 2:
            raise ValueError('Packed dual action token length must equal 2H')
        joint, eef = tokens.chunk(2, dim=1)
        return merge_action(self.joint.post_dit(joint, pre_state['branches'][0]), self.eef.post_dit(eef, pre_state['branches'][1]))

def _contexts(payload):
    if payload is None:
        return (None, None)
    return tuple(({'context': payload['context'][i], 'mask': payload['mask'][i]} for i in range(2)))

def _split_hbridge(lens, masks, horizon):
    if lens is None and masks is None:
        return (None, None)
    if lens is None or masks is None or len(lens) != len(masks):
        raise ValueError('Dual HBridge needs matching sub-stream lengths and masks')
    if len(lens) >= 2 and tuple(lens[-2:]) == (horizon, horizon):
        return (lens, masks)
    if lens[-1] != 2 * horizon:
        raise ValueError('Dual HBridge final action sub-stream must contain 2H tokens')
    action_mask = masks[-1]
    return (list(lens[:-1]) + [horizon, horizon], list(masks[:-1]) + [action_mask[..., :horizon, :horizon], action_mask[..., horizon:, horizon:]])

def _visual_cannot_read_action(mask, visual_length):
    if mask is None or mask.dtype != torch.bool:
        raise ValueError('Cooperative dual action requires an explicit boolean attention mask')
    positions = torch.arange(mask.shape[-1], device=mask.device)
    allowed = ~((positions[:, None] < visual_length) & (positions[None, :] >= visual_length))
    return mask & allowed

class DualCooperativeMoT(MoT):

    def __init__(self, mixtures, **kwargs):
        facade = mixtures['action']
        if kwargs.get('attn_mode', 'sdpa') != 'sdpa':
            raise ValueError('Dual action requires explicit masked SDPA')
        super().__init__({'video': mixtures['video'], 'action': facade.joint, 'action_eef': facade.eef}, **kwargs)

    @classmethod
    def from_single(cls, mot, facade):
        upgraded = cls({'video': mot.mixtures['video'], 'action': facade}, mot_checkpoint_mixed_attn=mot.mot_checkpoint_mixed_attn, attn_mode=mot.attn_mode, hbridge_enabled=mot.hbridge_enabled, hbridge_bottom_ratio=mot.hbridge_bottom_ratio, hbridge_top_ratio=mot.hbridge_top_ratio)
        upgraded.checkpoint_full_layers = mot.checkpoint_full_layers
        upgraded.ffn_chunk_size = mot.ffn_chunk_size
        upgraded.train(mot.training)
        return upgraded

    def set_attn_mode(self, mode):
        if mode != 'sdpa':
            raise ValueError('Dual action requires explicit masked SDPA')
        super().set_attn_mode(mode)

    def forward(self, embeds_all, attention_mask, freqs_all, context_all, t_mod_all, sub_stream_lens=None, sub_stream_self_masks=None):
        if getattr(self, 'history_payload', None) is not None:
            video, action = self._forward_joint_inner(embeds_all['video'], embeds_all['action'],
                freqs_all['video'], freqs_all['action'], t_mod_all['video'], t_mod_all['action'],
                context_all['video'], context_all['action'], attention_mask, sub_stream_lens, sub_stream_self_masks)
            return dict(video=video, action=action)
        if embeds_all['action'].shape[1] % 2:
            raise ValueError('Packed dual action length must equal 2H')
        joint, eef = embeds_all['action'].chunk(2, dim=1)
        fj, fe = freqs_all['action'].chunk(2, dim=0)
        cj, ce = _contexts(context_all['action'])
        lens, masks = _split_hbridge(sub_stream_lens, sub_stream_self_masks, joint.shape[1])
        if self.hbridge_enabled and lens is None:
            raise ValueError('Dual HBridge must receive sub-stream masks')
        attention_mask = _visual_cannot_read_action(attention_mask, embeds_all['video'].shape[1])
        result = super().forward({'video': embeds_all['video'], 'action': joint, 'action_eef': eef}, attention_mask, {'video': freqs_all['video'], 'action': fj, 'action_eef': fe}, {'video': context_all['video'], 'action': cj, 'action_eef': ce}, {'video': t_mod_all['video'], 'action': t_mod_all['action'][0], 'action_eef': t_mod_all['action'][1]}, lens, masks)
        return {'video': result['video'], 'action': torch.cat([result['action'], result['action_eef']], dim=1)}

    def _forward_joint_inner(self, video_tokens, action_tokens, video_freqs, action_freqs, video_t_mod, action_t_mod, video_context_payload, action_context_payload, attention_mask, sub_stream_lens=None, sub_stream_self_masks=None):
        if getattr(self, 'history_payload', None) is not None:
            from .history_attention import forward_history
            return forward_history(self, video_tokens, action_tokens, video_freqs, action_freqs,
                video_t_mod, action_t_mod, video_context_payload, action_context_payload,
                attention_mask, sub_stream_lens, sub_stream_self_masks, self.history_payload)
        if action_tokens.shape[1] % 2:
            raise ValueError('Packed dual action length must equal 2H')
        horizon = action_tokens.shape[1] // 2
        lens, masks = _split_hbridge(sub_stream_lens, sub_stream_self_masks, horizon)
        if self.hbridge_enabled and lens is None:
            raise ValueError('Dual HBridge must receive sub-stream masks')
        attention_mask = _visual_cannot_read_action(attention_mask, video_tokens.shape[1])
        cj, ce = _contexts(action_context_payload)
        experts = (self.mixtures['video'], self.mixtures['action'], self.mixtures['action_eef'])
        xs = (video_tokens, *action_tokens.chunk(2, dim=1))
        freqs = (video_freqs, *action_freqs.chunk(2, dim=0))
        mods = (video_t_mod, action_t_mod[0], action_t_mod[1])
        contexts = (video_context_payload, cj, ce)
        lengths = (video_tokens.shape[1], horizon, horizon)
        for layer in range(self.num_layers):
            ios = [self._build_expert_attention_io(expert=expert, block=expert.blocks[layer], x=xs[i], freqs=freqs[i], t_mod=mods[i]) for i, expert in enumerate(experts)]
            q = torch.cat([io[0] for io in ios], dim=1)
            k = torch.cat([io[1] for io in ios], dim=1)
            v = torch.cat([io[2] for io in ios], dim=1)
            if self.hbridge_enabled and self._is_outer_layer(layer):
                pieces = []
                offset = 0
                for length, mask in zip(lens, masks):
                    pieces.append(self._mixed_attention(q[:, offset:offset + length], k[:, offset:offset + length], v[:, offset:offset + length], attention_mask=mask))
                    offset += length
                mixed = torch.cat(pieces, dim=1)
            else:
                mixed = self._mixed_attention(q, k, v, attention_mask=attention_mask)
            mixed_streams = mixed.split(lengths, dim=1)
            xs = tuple((self._apply_expert_post_block(block=expert.blocks[layer], residual_x=ios[i][3], mixed_attn_out=mixed_streams[i], gate_msa=ios[i][4], shift_mlp=ios[i][5], scale_mlp=ios[i][6], gate_mlp=ios[i][7], context_payload=contexts[i]) for i, expert in enumerate(experts)))
        return (xs[0], torch.cat(xs[1:], dim=1))

    @staticmethod
    def _unsupported(*args, **kwargs):
        raise NotImplementedError('Dual action supports full-joint denoising only; KV-cache/fused path is unsupported')
    prefill_video_cache = _unsupported
    forward_action_with_video_cache = _unsupported
    _prefill_video_cache_inner = _unsupported
    _forward_action_with_video_cache_inner = _unsupported
    prefill_joint_anchor_kv = _unsupported
    decode_joint_noisy = _unsupported
