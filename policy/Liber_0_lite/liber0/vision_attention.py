"""Batch equal-length images without allowing attention across image boundaries."""
from types import MethodType

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward


def grouped_vision_attention(self, hidden_states, cu_seqlens, rotary_pos_emb=None,
                             position_embeddings=None, **kwargs):
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    return _grouped_attention(self, hidden_states, position_embeddings, lengths, **kwargs)


def cached_grouped_vision_attention(self, hidden_states, cu_seqlens, rotary_pos_emb=None,
                                    position_embeddings=None, *, o1_image_lengths, **kwargs):
    return _grouped_attention(self, hidden_states, position_embeddings, o1_image_lengths, **kwargs)


def _grouped_attention(self, hidden_states, position_embeddings, lengths, **kwargs):
    from .vendor.qwen3_vl_transformers import apply_rotary_pos_emb_vision

    length = hidden_states.shape[0]
    query, key, value = self.qkv(hidden_states).reshape(
        length, 3, self.num_heads, -1
    ).permute(1, 0, 2, 3).unbind(0)
    query, key = apply_rotary_pos_emb_vision(query, key, *position_embeddings)
    parts = [tensor.split(lengths, dim=0) for tensor in (query, key, value)]
    outputs = [None] * len(lengths)
    for image_length in sorted(set(lengths)):
        indices = [i for i, size in enumerate(lengths) if size == image_length]
        query, key, value = [
            torch.stack([part[i] for i in indices]).transpose(1, 2) for part in parts
        ]
        result = sdpa_attention_forward(
            self, query, key, value, attention_mask=None, scaling=self.scaling,
            dropout=self.attention_dropout if self.training else 0.0,
            is_causal=False, **kwargs,
        )[0]
        for index, output in zip(indices, result.unbind(0)):
            outputs[index] = output
    return self.proj(torch.cat(outputs, dim=0).reshape(length, -1).contiguous())


def install_grouped_vision_attention(visual, *, cache_geometry=False):
    for block in visual.blocks:
        if block.attn.config._attn_implementation != 'sdpa':
            raise ValueError('Grouped O1 vision attention requires the SDPA backend.')
        forward = cached_grouped_vision_attention if cache_geometry else grouped_vision_attention
        block.attn.forward = MethodType(forward, block.attn)
    if cache_geometry:
        from .vision_geometry import install_cached_vision_geometry
        install_cached_vision_geometry(visual)
