"""One native VAP transformer block with independent expert modulation."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import VAPJointAttention, VAPTextCrossAttention
from .initialization import copy_linear, copy_norm, resize_vector
from .kv_cache import VAPAttentionKVCache, VAPLayerKVCache
from .layout import AttentionLayout
from .rope import RoPE4DPhaseCache


def _modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x * (1 + scale) + shift


class VAPBlock(nn.Module):
    def __init__(
        self,
        *,
        video_dim: int,
        action_dim: int,
        video_ffn_dim: int,
        action_ffn_dim: int,
        num_heads: int,
        head_dim: int,
        rope_axes_dim: tuple[int, int, int, int],
        attention_backend: str = "sdpa",
        first_frame_memory_reader: bool = False,
        eps: float = 1e-6,
        wan_block: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.video_dim = video_dim
        self.action_dim = action_dim
        self.attention = VAPJointAttention(
            video_dim=video_dim,
            action_dim=action_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            rope_axes_dim=rope_axes_dim,
            attention_backend=attention_backend,
            eps=eps,
            wan_attention=None if wan_block is None else wan_block.self_attn,
        )
        self.text_attention = VAPTextCrossAttention(
            video_dim=video_dim,
            action_dim=action_dim,
            context_dim=video_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            rope_axes_dim=rope_axes_dim,
            eps=eps,
            wan_attention=None if wan_block is None else wan_block.cross_attn,
        )
        self.first_frame_memory_reader = bool(first_frame_memory_reader)
        if self.first_frame_memory_reader:
            # Per-head zero gates preserve the inherited Wan/Text behavior at
            # initialization while allowing selected depths to learn retrieval.
            self.first_frame_memory_video_gate = nn.Parameter(torch.zeros(num_heads))
            self.first_frame_memory_action_gate = nn.Parameter(torch.zeros(num_heads))
        else:
            self.register_parameter("first_frame_memory_video_gate", None)
            self.register_parameter("first_frame_memory_action_gate", None)
        self.video_norm1 = (
            nn.LayerNorm(video_dim, eps=eps, elementwise_affine=False)
            if wan_block is None
            else wan_block.norm1
        )
        self.video_norm2 = (
            nn.LayerNorm(video_dim, eps=eps, elementwise_affine=False)
            if wan_block is None
            else wan_block.norm2
        )
        self.action_norm1 = nn.LayerNorm(action_dim, eps=eps, elementwise_affine=False)
        self.action_norm2 = nn.LayerNorm(action_dim, eps=eps, elementwise_affine=False)
        self.video_norm3 = (
            nn.LayerNorm(video_dim, eps=eps) if wan_block is None else wan_block.norm3
        )
        self.action_norm3 = nn.LayerNorm(action_dim, eps=eps)
        self.video_ffn = (
            nn.Sequential(
                nn.Linear(video_dim, video_ffn_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(video_ffn_dim, video_dim),
            )
            if wan_block is None
            else wan_block.ffn
        )
        self.action_ffn = nn.Sequential(
            nn.Linear(action_dim, action_ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_ffn_dim, action_dim),
        )
        self.video_modulation = (
            nn.Parameter(torch.randn(1, 6, video_dim) / video_dim**0.5)
            if wan_block is None
            else wan_block.modulation
        )
        self.action_modulation = nn.Parameter(
            torch.randn(1, 6, action_dim) / action_dim**0.5
        )
        if wan_block is not None:
            self.initialize_action_from_wan_block(wan_block)

    @staticmethod
    def _resolve_modulation_parameters(
        learned: torch.Tensor, time_mod: torch.Tensor, sequence_length: int
    ) -> tuple[torch.Tensor, ...]:
        if time_mod.ndim == 3 and time_mod.shape[1:] == learned.shape[1:]:
            values = learned.to(dtype=time_mod.dtype, device=time_mod.device) + time_mod
            return tuple(value[:, None, :] for value in values.unbind(dim=1))
        if (
            time_mod.ndim == 4
            and time_mod.shape[1] == sequence_length
            and time_mod.shape[2:] == learned.shape[1:]
        ):
            values = (
                learned[:, None].to(dtype=time_mod.dtype, device=time_mod.device)
                + time_mod
            )
            return tuple(values[:, :, index, :] for index in range(6))
        else:
            raise ValueError(
                f"time modulation must be [B,6,{learned.shape[-1]}] or "
                f"[B,S,6,{learned.shape[-1]}], "
                f"got {tuple(time_mod.shape)}"
            )

    def forward(
        self,
        video: torch.Tensor,
        action: torch.Tensor,
        *,
        video_time_mod: torch.Tensor,
        action_time_mod: torch.Tensor,
        video_position_ids: torch.Tensor,
        action_position_ids: torch.Tensor,
        layout: AttentionLayout,
        context: torch.Tensor | None = None,
        text_position_ids: torch.Tensor | None = None,
        capture_prefix_rank: int | None = None,
        video_rope_cache: RoPE4DPhaseCache | None = None,
        action_rope_cache: RoPE4DPhaseCache | None = None,
        text_rope_cache: RoPE4DPhaseCache | None = None,
        video_camera_ids: torch.Tensor | None = None,
        first_frame_memory: VAPAttentionKVCache | None = None,
        first_frame_memory_camera_ids: torch.Tensor | None = None,
        condition_video_tokens: int | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, VAPLayerKVCache]
    ):
        vs_msa, vc_msa, vg_msa, vs_ffn, vc_ffn, vg_ffn = (
            self._resolve_modulation_parameters(
                self.video_modulation, video_time_mod, video.shape[1]
            )
        )
        as_msa, ac_msa, ag_msa, as_ffn, ac_ffn, ag_ffn = (
            self._resolve_modulation_parameters(
                self.action_modulation, action_time_mod, action.shape[1]
            )
        )
        video_input = _modulate(self.video_norm1(video), vs_msa, vc_msa)
        action_input = _modulate(self.action_norm1(action), as_msa, ac_msa)
        attention_result = self.attention(
            video_input,
            action_input,
            video_position_ids=video_position_ids,
            action_position_ids=action_position_ids,
            layout=layout,
            capture_prefix_rank=capture_prefix_rank,
            video_rope_cache=video_rope_cache,
            action_rope_cache=action_rope_cache,
        )
        if capture_prefix_rank is None:
            video_attn, action_attn = attention_result
            self_cache = None
        else:
            video_attn, action_attn, self_cache = attention_result
        video = video + vg_msa * video_attn
        action = action + ag_msa * action_attn
        text_cache = None
        if first_frame_memory is not None:
            if not self.first_frame_memory_reader:
                raise ValueError("Anchor Bank was passed to a non-reader VAP block")
            if (
                first_frame_memory_camera_ids is None
                or video_camera_ids is None
                or condition_video_tokens is None
                or self.first_frame_memory_video_gate is None
                or self.first_frame_memory_action_gate is None
            ):
                raise ValueError("first-frame memory reader inputs are incomplete")
            text_result = self.text_attention.forward_with_anchor(
                self.video_norm3(video),
                self.action_norm3(action),
                context,
                video_position_ids=video_position_ids,
                action_position_ids=action_position_ids,
                text_position_ids=text_position_ids,
                anchor_context=first_frame_memory,
                anchor_camera_ids=first_frame_memory_camera_ids,
                video_camera_ids=video_camera_ids,
                condition_video_tokens=condition_video_tokens,
                anchor_video_gate=self.first_frame_memory_video_gate,
                anchor_action_gate=self.first_frame_memory_action_gate,
                return_context_kv=capture_prefix_rank is not None,
                video_rope_cache=video_rope_cache,
                action_rope_cache=action_rope_cache,
                text_rope_cache=text_rope_cache,
            )
            if capture_prefix_rank is None:
                video_text, action_text = text_result
            else:
                video_text, action_text, text_cache = text_result
            video = video + video_text
            action = action + action_text
        elif context is not None:
            if text_position_ids is None:
                raise ValueError(
                    "text_position_ids are required when context is present"
                )
            text_result = self.text_attention(
                self.video_norm3(video),
                self.action_norm3(action),
                context,
                video_position_ids=video_position_ids,
                action_position_ids=action_position_ids,
                text_position_ids=text_position_ids,
                return_context_kv=capture_prefix_rank is not None,
                video_rope_cache=video_rope_cache,
                action_rope_cache=action_rope_cache,
                text_rope_cache=text_rope_cache,
            )
            if capture_prefix_rank is None:
                video_text, action_text = text_result
            else:
                video_text, action_text, text_cache = text_result
            video = video + video_text
            action = action + action_text
        video = video + vg_ffn * self.video_ffn(
            _modulate(self.video_norm2(video), vs_ffn, vc_ffn)
        )
        action = action + ag_ffn * self.action_ffn(
            _modulate(self.action_norm2(action), as_ffn, ac_ffn)
        )
        if capture_prefix_rank is not None:
            if self_cache is None:
                raise AssertionError(
                    "prefix cache capture produced no self-attention KV"
                )
            return (
                video,
                action,
                VAPLayerKVCache(
                    self_attention=self_cache,
                    text_attention=text_cache,
                    first_frame_memory=first_frame_memory,
                ),
            )
        return video, action

    def forward_active_action(
        self,
        action: torch.Tensor,
        *,
        action_time_mod: torch.Tensor,
        action_rope_cache: RoPE4DPhaseCache,
        cache: VAPLayerKVCache,
    ) -> torch.Tensor:
        """Run only one noisy Action stage against an immutable prefix cache."""

        as_msa, ac_msa, ag_msa, as_ffn, ac_ffn, ag_ffn = (
            self._resolve_modulation_parameters(
                self.action_modulation, action_time_mod, action.shape[1]
            )
        )
        action_input = _modulate(self.action_norm1(action), as_msa, ac_msa)
        action_attn = self.attention.forward_active_action(
            action_input,
            action_rope_cache=action_rope_cache,
            prefix=cache.self_attention,
        )
        action = action + ag_msa * action_attn
        if cache.first_frame_memory is not None:
            if (
                not self.first_frame_memory_reader
                or self.first_frame_memory_action_gate is None
            ):
                raise ValueError("cached Anchor Bank reached a non-reader block")
            action = action + self.text_attention.forward_active_action_with_anchor(
                self.action_norm3(action),
                action_rope_cache=action_rope_cache,
                text_context=cache.text_attention,
                anchor_context=cache.first_frame_memory,
                anchor_action_gate=self.first_frame_memory_action_gate,
            )
        elif cache.text_attention is not None:
            action = action + self.text_attention.forward_active_action(
                self.action_norm3(action),
                action_rope_cache=action_rope_cache,
                context=cache.text_attention,
            )
        action = action + ag_ffn * self.action_ffn(
            _modulate(self.action_norm2(action), as_ffn, ac_ffn)
        )
        return action

    @torch.no_grad()
    def initialize_from_wan_block(self, wan_block: nn.Module) -> None:
        self.attention.initialize_from_wan_attention(wan_block.self_attn)
        self.text_attention.initialize_from_wan_cross_attention(wan_block.cross_attn)
        copy_norm(self.video_norm3, wan_block.norm3, allow_resize=False)
        copy_norm(self.action_norm3, wan_block.norm3, allow_resize=True)
        copy_linear(self.video_ffn[0], wan_block.ffn[0], allow_resize=False)
        copy_linear(self.video_ffn[2], wan_block.ffn[2], allow_resize=False)
        copy_linear(self.action_ffn[0], wan_block.ffn[0], allow_resize=True)
        copy_linear(self.action_ffn[2], wan_block.ffn[2], allow_resize=True)
        self.video_modulation.copy_(wan_block.modulation)
        resized = torch.stack(
            [
                resize_vector(row, self.action_dim)
                for row in wan_block.modulation.squeeze(0)
            ],
            dim=0,
        ).unsqueeze(0)
        self.action_modulation.copy_(
            resized.to(
                device=self.action_modulation.device,
                dtype=self.action_modulation.dtype,
            )
        )

    @torch.no_grad()
    def initialize_action_from_wan_block(self, wan_block: nn.Module) -> None:
        copy_linear(self.action_ffn[0], wan_block.ffn[0], allow_resize=True)
        copy_linear(self.action_ffn[2], wan_block.ffn[2], allow_resize=True)
        copy_norm(self.action_norm3, wan_block.norm3, allow_resize=True)
        resized = torch.stack(
            [
                resize_vector(row, self.action_dim)
                for row in wan_block.modulation.squeeze(0)
            ],
            dim=0,
        ).unsqueeze(0)
        self.action_modulation.copy_(
            resized.to(
                device=self.action_modulation.device,
                dtype=self.action_modulation.dtype,
            )
        )


__all__ = ["VAPBlock"]
