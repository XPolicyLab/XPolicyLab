"""Fixed-image geometry and attention batching, without caching learned values.

The policy supplies independent 224px images. Native video/variable-grid calls
retain their original attention implementation unless explicitly marked here.
"""

from functools import lru_cache
from types import MethodType

import torch


@lru_cache(maxsize=16)
def _one_image_geometry(device, height, width, merge, side, default_dtype):
    from transformers.vision_utils import (
        get_vision_bilinear_indices_and_weights, get_vision_position_ids,
    )

    # Cache only integer positions and interpolation coefficients. Normal
    # tensors are required even when the first caller uses inference_mode:
    # an embedding's backward can later save these indices in a training call.
    with torch.inference_mode(False), torch.no_grad():
        grid = torch.tensor([[1, height, width]], device=device, dtype=torch.long)
        indices, weights = get_vision_bilinear_indices_and_weights(grid, side, merge)
        positions = get_vision_position_ids(grid, merge)
    return indices, weights, positions


def fixed_image_geometry(vision, count, *, device, height, width):
    if count < 1:
        raise ValueError("Fixed image geometry requires at least one image")
    indices, weights, positions = _one_image_geometry(
        torch.device(device), height, width, int(vision.spatial_merge_size),
        int(vision.num_grid_per_side), torch.get_default_dtype(),
    )
    return dict(
        bilinear_indices=indices.repeat(1, count),
        bilinear_weights=weights.repeat(1, count),
        position_ids=positions.repeat(count, 1),
        cu_seqlens=torch.arange(count + 1, device=device, dtype=torch.int32) * (height * width),
    )


def _fixed_image_attention(self, hidden_states, cu_seqlens,
                           position_embeddings=None, **kwargs):
    tokens = kwargs.pop("xvla_fixed_image_tokens", None)
    implementation = self.config._attn_implementation
    if (tokens is None or implementation not in {"sdpa", "flash_attention_2"}
            or (self.training and self.attention_dropout != 0)):
        return self._xvla_original_forward(
            hidden_states, cu_seqlens, position_embeddings=position_embeddings, **kwargs)
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb_vision, eager_attention_forward,
    )

    length = hidden_states.shape[0]
    if length % tokens or cu_seqlens.numel() != length // tokens + 1:
        raise ValueError("Fixed image attention received inconsistent image boundaries")
    images = length // tokens
    query, key, value = self.qkv(hidden_states).reshape(
        length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    query, key = apply_rotary_pos_emb_vision(query, key, *position_embeddings)
    query, key, value = [tensor.reshape(images, tokens, self.num_heads, self.head_dim)
                         .transpose(1, 2) for tensor in (query, key, value)]
    interface = ALL_ATTENTION_FUNCTIONS.get_interface(implementation, eager_attention_forward)
    # Independent, equal-sized images are the batch dimension. This preserves
    # within-image attention and avoids reading max(cu_seqlens.diff()) from
    # the GPU in every vision layer. Language attention is untouched.
    output, _ = interface(self, query, key, value, attention_mask=None,
                          scaling=self.scaling, dropout=0.0, is_causal=False, **kwargs)
    return self.proj(output.reshape(length, self.dim).contiguous())


def install_fixed_image_vision(vision):
    # Import before altering any module, so unsupported Transformers versions
    # fail before a partial installation rather than silently changing layout.
    from transformers.vision_utils import get_vision_bilinear_indices_and_weights
    import inspect

    if "kwargs" not in inspect.signature(get_vision_bilinear_indices_and_weights).parameters:
        raise RuntimeError("Fixed vision geometry needs Transformers precomputed vision kwargs")
    count = 0
    for block in vision.blocks:
        attention = block.attn
        if not hasattr(attention, "_xvla_original_forward"):
            attention._xvla_original_forward = attention.forward
            attention.forward = MethodType(_fixed_image_attention, attention)
            count += 1
    vision._xvla_fixed_image_geometry = True
    return count
