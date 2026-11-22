"""VAPv2-only joint block: no legacy text cross-attention or extra self-attn."""

import torch
from torch import nn
from wall_x.model.vap.block import VAPBlock, _modulate
from .joint import JointTextIVAVAAttention


class VAPv2Block(nn.Module):
    def __init__(self, base_block):
        super().__init__()
        if base_block.first_frame_memory_reader:
            raise ValueError(
                "VAPv2 accepts prefix Memory, not cross-attention Anchor Banks"
            )
        self.video_dim, self.action_dim = base_block.video_dim, base_block.action_dim
        # Transfer ownership; never register a second alias of the same module.
        self.attention = JointTextIVAVAAttention.from_existing(base_block.attention)
        for name in (
            "video_norm1",
            "video_norm2",
            "action_norm1",
            "action_norm2",
            "video_ffn",
            "action_ffn",
            "video_modulation",
            "action_modulation",
        ):
            setattr(self, name, getattr(base_block, name))
        self.first_frame_memory_reader = False

    def forward(
        self,
        video,
        action,
        context,
        *,
        video_time_mod,
        action_time_mod,
        text_time_mod,
        video_position_ids,
        action_position_ids,
        text_position_ids,
        joint_layout,
        capture_prefix_rank=None,
        rope_cache=None,
    ):
        resolve = VAPBlock._resolve_modulation_parameters
        vs, vc, vg, vfs, vfc, vfg = resolve(
            self.video_modulation, video_time_mod, video.shape[1]
        )
        ass, ac, ag, afs, afc, afg = resolve(
            self.action_modulation, action_time_mod, action.shape[1]
        )
        ts, tc, tg, tfs, tfc, tfg = resolve(
            self.video_modulation, text_time_mod, context.shape[1]
        )
        attention_result = self.attention(
            _modulate(self.video_norm1(context), ts, tc),
            _modulate(self.video_norm1(video), vs, vc),
            _modulate(self.action_norm1(action), ass, ac),
            text_position_ids=text_position_ids,
            video_position_ids=video_position_ids,
            action_position_ids=action_position_ids,
            joint_layout=joint_layout,
            capture_prefix_rank=capture_prefix_rank,
            rope_cache=rope_cache,
        )
        if capture_prefix_rank is not None:
            attention_result, cache = attention_result
        ta, va, aa = attention_result
        context, video, action = context + tg * ta, video + vg * va, action + ag * aa
        # Text shares video-width FFN, with fixed-clean AdaLN. One invocation
        # maintains an unambiguous DMuon/FSDP boundary; no duplicate text FFN.
        inputs = torch.cat(
            (
                _modulate(self.video_norm2(context), tfs, tfc),
                _modulate(self.video_norm2(video), vfs, vfc),
            ),
            dim=1,
        )
        tf, vf = self.video_ffn(inputs).split((context.shape[1], video.shape[1]), dim=1)
        context, video = context + tfg * tf, video + vfg * vf
        action = action + afg * self.action_ffn(
            _modulate(self.action_norm2(action), afs, afc)
        )
        context = context.masked_fill(~joint_layout.text_valid[..., None], 0)
        if joint_layout.memory_valid is not None:
            m = joint_layout.memory_valid.shape[1]
            video = torch.cat(
                (
                    video[:, :m].masked_fill(~joint_layout.memory_valid[..., None], 0),
                    video[:, m:],
                ),
                dim=1,
            )
        if capture_prefix_rank is not None:
            return video, action, context, cache
        return video, action, context

    def forward_cached_action(self, action, *, time_mod, positions, cache, key_valid,
                              reference_layout, active_slice, action_rope=None):
        ass, ac, ag, afs, afc, afg = VAPBlock._resolve_modulation_parameters(
            self.action_modulation, time_mod, action.shape[1]
        )
        attended = self.attention.forward_cached_action(
            _modulate(self.action_norm1(action), ass, ac),
            positions=positions,
            cache=cache,
            key_valid=key_valid,
            reference_layout=reference_layout,
            active_slice=active_slice,
            action_rope=action_rope,
        )
        action = action + ag * attended
        return action + afg * self.action_ffn(
            _modulate(self.action_norm2(action), afs, afc)
        )
