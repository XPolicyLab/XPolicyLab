"""One joint softmax over Text + the original logical IVAVA sequence."""

from copy import deepcopy
import torch
from torch.nn import functional as F
from wall_x.model.vap.attention import (
    VAPJointAttention,
    _compiled_flex_attention,
    _flex_kernel_options,
)
from wall_x.model.vap.rope import apply_rope_4d


class JointTextIVAVAAttention(VAPJointAttention):
    """Text[B,L,Dv], Video[B,Sv,Dv], Action[B,Sa,Da]; one packed Q/K/V sequence."""

    @classmethod
    def from_existing(cls, base):
        base.__class__ = cls
        for suffix in ("q", "k", "v", "o", "norm_q", "norm_k"):
            setattr(base, f"text_{suffix}", deepcopy(getattr(base, f"video_{suffix}")))
        return base

    def forward(
        self,
        text,
        video,
        action,
        *,
        text_position_ids,
        video_position_ids,
        action_position_ids,
        joint_layout,
        capture_prefix_rank=None,
        rope_cache=None,
    ):
        layout = joint_layout.base
        self._check_inputs(
            video, action, video_position_ids, action_position_ids, layout
        )
        if text.shape != (video.shape[0], joint_layout.text_len, self.video_dim):
            raise ValueError("Text must be [B,L,Dvideo] matching joint layout")
        qs, ks, values = [], [], []
        for stream, tensor, positions in (
            ("text", text, text_position_ids),
            ("video", video, video_position_ids),
            ("action", action, action_position_ids),
        ):
            q = self._heads(
                getattr(self, f"{stream}_norm_q")(getattr(self, f"{stream}_q")(tensor))
            )
            k = self._heads(
                getattr(self, f"{stream}_norm_k")(getattr(self, f"{stream}_k")(tensor))
            )
            v = self._heads(getattr(self, f"{stream}_v")(tensor))
            phase_cache = (
                None if rope_cache is None
                else rope_cache.for_stream(stream, q, k, positions)
            )
            q, k = apply_rope_4d(
                q, k, positions, axes_dim=self.rope_axes_dim, phase_cache=phase_cache
            )
            qs.append(q.transpose(1, 2))
            ks.append(k.transpose(1, 2))
            values.append(v.transpose(1, 2))

        def pack(parts):
            return torch.cat((parts[0], layout.pack(parts[1], parts[2])), dim=-2)

        q, k, v = pack(qs), pack(ks), pack(values)
        if self.attention_backend == "flex_attention":
            if q.device.type != "cuda":
                raise RuntimeError("VAPv2 FlexAttention requires CUDA")
            output = _compiled_flex_attention()(
                q,
                k,
                v,
                block_mask=joint_layout.flex_block_mask(q.device),
                score_mod=joint_layout.flex_score_mod,
                kernel_options=_flex_kernel_options(),
            )
        else:
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=joint_layout.attention_mask(q.device)[:, None],
                dropout_p=self.dropout if self.training else 0.0,
            )
        t = output[:, :, : joint_layout.text_len]
        vo, ao = layout.unpack(output[:, :, joint_layout.text_len :])
        outputs = tuple(
            getattr(self, f"{stream}_o")(tensor.transpose(1, 2).flatten(2))
            for stream, tensor in (("text", t), ("video", vo), ("action", ao))
        )
        if capture_prefix_rank is None:
            return outputs
        from wall_x.model.vap.kv_cache import VAPAttentionKVCache

        active = layout.action_tokens_per_chunk
        chunk = (capture_prefix_rank - 1) // 2
        prefix_end = (joint_layout.text_len + layout.condition_video_tokens
                      + chunk * (layout.video_tokens_per_chunk + active))
        # Preserve full-forward shapes at capture, retaining ONLY clean causal
        # prefix K/V. Neither active nor future tokens may enter the cache.
        cache = VAPAttentionKVCache(
            k[..., :prefix_end, :].detach().clone(),
            v[..., :prefix_end, :].detach().clone(), active
        )
        return outputs, cache

    def forward_cached_action(self, action, *, positions, cache, key_valid,
                              reference_layout, active_slice, action_rope=None):
        q = self._heads(self.action_norm_q(self.action_q(action)))
        k = self._heads(self.action_norm_k(self.action_k(action)))
        v = self._heads(self.action_v(action))
        if action_rope is None:
            q, k = apply_rope_4d(q, k, positions, axes_dim=self.rope_axes_dim)
        else:
            from .inference_rope import rotate

            q, k = rotate(q, k, action_rope.cos, action_rope.sin)
        start, end = active_slice
        k, v = cache.update_active(k[:, start:end].transpose(1, 2),
                                   v[:, start:end].transpose(1, 2))
        prefix_end = k.shape[-2] - (end - start)
        total = reference_layout.sequence_length
        # Ordinary full-forward sparse attention, identical sequence shapes and
        # query offsets. Inactive Q and causally invisible future K/V are zero
        # scratch; only the active rows are used. This avoids BF16 drift from
        # compact-query/split-KV kernels without recomputing prefix experts.
        full_q = q.new_zeros(q.shape[0], q.shape[2], total, q.shape[3])
        full_q[:, :, prefix_end:prefix_end + end - start] = q[:, start:end].transpose(1, 2)
        k = F.pad(k, (0, 0, 0, total - k.shape[-2]))
        v = F.pad(v, (0, 0, 0, total - v.shape[-2]))
        if self.attention_backend == "flex_attention":
            output = _compiled_flex_attention()(
                full_q, k, v,
                block_mask=reference_layout.flex_block_mask(q.device),
                score_mod=reference_layout.flex_score_mod,
                kernel_options=_flex_kernel_options(),
            )
        else:
            output = F.scaled_dot_product_attention(
                full_q, k, v,
                attn_mask=reference_layout.attention_mask(q.device)[:, None],
                dropout_p=0.0,
            )
        logical = output[:, :, reference_layout.text_len:]
        indices = getattr(reference_layout, "graph_unpack_indices", None)
        if indices is None:
            _, action_output = reference_layout.base.unpack(logical)
        else:
            # Same full index_select as reference; only rank construction moves
            # out of capture (the reference constructs CPU indices).
            physical = logical.index_select(-2, indices)
            action_output = physical[..., reference_layout.base.num_video_tokens:, :]
        return self.action_o(action_output.transpose(1, 2).flatten(2))
