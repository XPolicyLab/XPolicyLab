from __future__ import annotations

import inspect

import torch
from torch import nn

from policy_core.model.WAM_policy.ltx_models.transformer_ltx_multiview_vlm_imf_v2 import (
    LTXVideoRotaryPosEmbed,
    LTXVideoTransformer3DModel as BaseLTXVideoTransformer3DModel,
)


class MemoryOrderRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        base_rope: LTXVideoRotaryPosEmbed,
        *,
        num_mem_frames: int,
        theta: float = 10000.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        if theta <= 0:
            raise ValueError(f"memory_order_rope_theta must be positive, got {theta}.")
        if scale < 0:
            raise ValueError(f"memory_order_rope_scale must be non-negative, got {scale}.")
        if base_rope.dim % 2:
            raise ValueError(f"Memory order RoPE requires an even dimension, got {base_rope.dim}.")
        if num_mem_frames < 1:
            raise ValueError("memory_order_rope_num_mem_frames must be positive.")
        self.base_rope = base_rope
        self.num_mem_frames = int(num_mem_frames)
        self.theta = float(theta)
        self.scale = float(scale)

    def _order_embedding(self, history_frames, tokens_per_frame, *, device):
        positions = torch.arange(
            -history_frames, 0, device=device, dtype=torch.float32
        ).repeat_interleave(tokens_per_frame)
        pair_count = self.base_rope.dim // 2
        inv_freq = self.theta ** (
            -torch.arange(pair_count, device=device, dtype=torch.float32)
            / pair_count
        )
        angles = positions[:, None] * inv_freq[None, :] * self.scale
        return (
            angles.cos().repeat_interleave(2, dim=-1).unsqueeze(0),
            angles.sin().repeat_interleave(2, dim=-1).unsqueeze(0),
        )

    def forward(
        self,
        hidden_states,
        rope_interpolation_scale=None,
        num_frames=None,
        height=None,
        width=None,
        video_rope_coords=None,
        **kwargs,
    ):
        base_cos, base_sin = self.base_rope(
            hidden_states,
            rope_interpolation_scale,
            num_frames,
            height,
            width,
            video_rope_coords=video_rope_coords,
            **kwargs,
        )
        history_frames = self.num_mem_frames - 1
        if history_frames == 0 or self.scale == 0:
            return base_cos, base_sin
        if height is None or width is None:
            raise ValueError("Memory order RoPE requires height and width.")

        memory_tokens = history_frames * int(height) * int(width)
        if memory_tokens > base_cos.shape[1]:
            raise ValueError(
                f"History memory tokens {memory_tokens} exceed sequence length "
                f"{base_cos.shape[1]}."
            )
        order_cos, order_sin = self._order_embedding(
            history_frames,
            int(height) * int(width),
            device=base_cos.device,
        )
        memory_cos = base_cos[:, :memory_tokens]
        memory_sin = base_sin[:, :memory_tokens]
        combined_cos = base_cos.clone()
        combined_sin = base_sin.clone()
        combined_cos[:, :memory_tokens] = (
            memory_cos * order_cos - memory_sin * order_sin
        )
        combined_sin[:, :memory_tokens] = (
            memory_sin * order_cos + memory_cos * order_sin
        )
        return combined_cos, combined_sin


class LTXVideoTransformer3DModel(BaseLTXVideoTransformer3DModel):
    def __init__(
        self,
        *args,
        memory_order_rope_num_mem_frames: int = 8,
        memory_order_rope_theta: float = 10000.0,
        memory_order_rope_scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.rope = MemoryOrderRotaryPosEmbed(
            self.rope,
            num_mem_frames=memory_order_rope_num_mem_frames,
            theta=memory_order_rope_theta,
            scale=memory_order_rope_scale,
        )
        self.register_to_config(
            memory_order_rope_num_mem_frames=int(memory_order_rope_num_mem_frames),
            memory_order_rope_theta=float(memory_order_rope_theta),
            memory_order_rope_scale=float(memory_order_rope_scale),
        )

    def forward(self, *args, **kwargs):
        runtime_frames = int(kwargs.get("num_mem_frames", self.rope.num_mem_frames))
        if runtime_frames != self.rope.num_mem_frames:
            raise ValueError(
                "Runtime num_mem_frames must match memory RoPE config: "
                f"{runtime_frames} != {self.rope.num_mem_frames}."
            )
        kwargs.setdefault("num_mem_frames", self.rope.num_mem_frames)
        return super().forward(*args, **kwargs)


def _extended_init_signature():
    base_signature = inspect.signature(BaseLTXVideoTransformer3DModel.__init__)
    parameters = []
    extra = (
        ("memory_order_rope_num_mem_frames", 8, int),
        ("memory_order_rope_theta", 10000.0, float),
        ("memory_order_rope_scale", 1.0, float),
    )
    for parameter in base_signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            parameters.extend(
                inspect.Parameter(
                    name,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=annotation,
                )
                for name, default, annotation in extra
            )
        parameters.append(parameter)
    return base_signature.replace(parameters=parameters)


LTXVideoTransformer3DModel.__init__.__signature__ = _extended_init_signature()

__all__ = ["LTXVideoTransformer3DModel", "MemoryOrderRotaryPosEmbed"]
