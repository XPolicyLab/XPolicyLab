"""Native joint attention for physically separate Video and Action experts."""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .initialization import copy_linear, copy_norm
from .kv_cache import VAPAttentionKVCache
from .layout import ActionStageLayout, AttentionLayout, InterleaveLayout
from .rope import RoPE4DPhaseCache, apply_rope_4d


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(dtype=dtype) * self.weight


@lru_cache(maxsize=1)
def _compiled_flex_attention():
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(
        flex_attention,
        dynamic=False,
        mode="max-autotune-no-cudagraphs",
    )


def _flex_kernel_options() -> dict[str, int] | None:
    """Use a lower-resource forward kernel only for explicit serving runs.

    Prefix Memory slightly lengthens the joint sequence.  PyTorch 2.12's
    default B300 autotune candidate can exceed the per-block hardware resource
    limit even though device memory is ample.  Smaller tiles and one pipeline
    stage preserve the same attention equation while avoiding that compile-time
    resource failure.  Training and ordinary inference remain unchanged.
    """

    enabled = os.environ.get("VAP_FLEX_ATTENTION_LOW_RESOURCE", "").strip().lower()
    if enabled in {"", "0", "false", "no", "off"}:
        return None
    if enabled not in {"1", "true", "yes", "on"}:
        raise ValueError(
            "VAP_FLEX_ATTENTION_LOW_RESOURCE must be a boolean, "
            f"got {enabled!r}"
        )
    return {"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 1}


class VAPJointAttention(nn.Module):
    """Joint SDPA with expert-specific projections and an I-A-V-A-V mask.

    Inputs use physical expert storage: video ``[B,Sv,Dv]`` and action
    ``[B,Sa,Da]``. Q/K/V are packed into logical ``I,A0,V0,A1,V1`` order only
    for attention, then unpacked before the expert-specific output projections.
    """

    def __init__(
        self,
        *,
        video_dim: int,
        action_dim: int,
        num_heads: int,
        head_dim: int,
        rope_axes_dim: Sequence[int],
        dropout: float = 0.0,
        eps: float = 1e-6,
        attention_backend: str = "sdpa",
        wan_attention: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.video_dim = video_dim
        self.action_dim = action_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.joint_dim = num_heads * head_dim
        self.rope_axes_dim = tuple(rope_axes_dim)
        self.dropout = dropout
        if attention_backend not in {"sdpa", "flex_attention"}:
            raise ValueError(f"unsupported attention_backend={attention_backend!r}")
        if attention_backend == "flex_attention" and dropout:
            raise ValueError("flex_attention VAP path requires dropout=0")
        self.attention_backend = attention_backend

        if wan_attention is None:
            self.video_q = nn.Linear(video_dim, self.joint_dim)
            self.video_k = nn.Linear(video_dim, self.joint_dim)
            self.video_v = nn.Linear(video_dim, self.joint_dim)
            self.video_o = nn.Linear(self.joint_dim, video_dim)
            self.video_norm_q = RMSNorm(self.joint_dim, eps=eps)
            self.video_norm_k = RMSNorm(self.joint_dim, eps=eps)
        else:
            self.video_q = wan_attention.q
            self.video_k = wan_attention.k
            self.video_v = wan_attention.v
            self.video_o = wan_attention.o
            self.video_norm_q = wan_attention.norm_q
            self.video_norm_k = wan_attention.norm_k
        self.action_q = nn.Linear(action_dim, self.joint_dim)
        self.action_k = nn.Linear(action_dim, self.joint_dim)
        self.action_v = nn.Linear(action_dim, self.joint_dim)
        self.action_o = nn.Linear(self.joint_dim, action_dim)

        self.action_norm_q = RMSNorm(self.joint_dim, eps=eps)
        self.action_norm_k = RMSNorm(self.joint_dim, eps=eps)
        if wan_attention is not None:
            self.initialize_action_from_wan_attention(wan_attention)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.unflatten(-1, (self.num_heads, self.head_dim))

    def _check_inputs(
        self,
        video: torch.Tensor,
        action: torch.Tensor,
        video_position_ids: torch.Tensor,
        action_position_ids: torch.Tensor,
        layout: AttentionLayout,
    ) -> None:
        if video.ndim != 3 or video.shape[-1] != self.video_dim:
            raise ValueError(
                f"video must be [B,Sv,{self.video_dim}], got {video.shape}"
            )
        if action.ndim != 3 or action.shape[-1] != self.action_dim:
            raise ValueError(
                f"action must be [B,Sa,{self.action_dim}], got {action.shape}"
            )
        if video.shape[0] != action.shape[0]:
            raise ValueError("video and action batch sizes must match")
        if video.shape[1] != layout.num_video_tokens:
            raise ValueError("video sequence does not match InterleaveLayout")
        if action.shape[1] != layout.num_action_tokens:
            raise ValueError("action sequence does not match InterleaveLayout")
        expected_video_pos = (video.shape[0], video.shape[1], 4)
        expected_action_pos = (action.shape[0], action.shape[1], 4)
        if tuple(video_position_ids.shape) != expected_video_pos:
            raise ValueError(
                f"video_position_ids must be {expected_video_pos}, "
                f"got {tuple(video_position_ids.shape)}"
            )
        if tuple(action_position_ids.shape) != expected_action_pos:
            raise ValueError(
                f"action_position_ids must be {expected_action_pos}, "
                f"got {tuple(action_position_ids.shape)}"
            )

    def forward(
        self,
        video: torch.Tensor,
        action: torch.Tensor,
        *,
        video_position_ids: torch.Tensor,
        action_position_ids: torch.Tensor,
        layout: AttentionLayout,
        capture_prefix_rank: int | None = None,
        video_rope_cache: RoPE4DPhaseCache | None = None,
        action_rope_cache: RoPE4DPhaseCache | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, VAPAttentionKVCache]
    ):
        self._check_inputs(
            video, action, video_position_ids, action_position_ids, layout
        )

        vq = self._heads(self.video_norm_q(self.video_q(video)))
        vk = self._heads(self.video_norm_k(self.video_k(video)))
        vv = self._heads(self.video_v(video))
        aq = self._heads(self.action_norm_q(self.action_q(action)))
        ak = self._heads(self.action_norm_k(self.action_k(action)))
        av = self._heads(self.action_v(action))

        vq, vk = apply_rope_4d(
            vq,
            vk,
            video_position_ids,
            axes_dim=self.rope_axes_dim,
            phase_cache=video_rope_cache,
        )
        aq, ak = apply_rope_4d(
            aq,
            ak,
            action_position_ids,
            axes_dim=self.rope_axes_dim,
            phase_cache=action_rope_cache,
        )

        # SDPA consumes [B,H,S,D]; on this representation sequence is -2, the
        # exact axis used by InterleaveLayout.pack/unpack.
        q = layout.pack(vq.transpose(1, 2), aq.transpose(1, 2))
        k = layout.pack(vk.transpose(1, 2), ak.transpose(1, 2))
        value = layout.pack(vv.transpose(1, 2), av.transpose(1, 2))
        prefix_cache = None
        if capture_prefix_rank is not None:
            if not isinstance(layout, (ActionStageLayout, InterleaveLayout)):
                raise TypeError("prefix KV capture requires an IVAVA causal layout")
            ranks = layout.logical_ranks(device=k.device)
            prefix_indices = torch.nonzero(
                ranks < int(capture_prefix_rank), as_tuple=False
            ).flatten()
            if prefix_indices.numel() == 0:
                raise ValueError(
                    f"active rank {capture_prefix_rank} has no causal prefix"
                )
            prefix_cache = VAPAttentionKVCache(
                key=k.index_select(-2, prefix_indices).detach(),
                value=value.index_select(-2, prefix_indices).detach(),
                active_capacity=layout.action_tokens_per_chunk,
            )
        if self.attention_backend == "flex_attention":
            if q.device.type != "cuda":
                raise RuntimeError("FlexAttention training requires CUDA")
            output = _compiled_flex_attention()(
                q,
                k,
                value,
                block_mask=layout.flex_block_mask(q.device),
                score_mod=getattr(layout, "flex_score_mod", None),
                kernel_options=_flex_kernel_options(),
            )
        else:
            mask = layout.attention_mask(device=q.device)
            mask = mask[None, None] if mask.ndim == 2 else mask[:, None]
            output = F.scaled_dot_product_attention(
                q,
                k,
                value,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        video_output, action_output = layout.unpack(output)
        video_output = video_output.transpose(1, 2).flatten(2)
        action_output = action_output.transpose(1, 2).flatten(2)
        result = (self.video_o(video_output), self.action_o(action_output))
        if prefix_cache is not None:
            return (*result, prefix_cache)
        return result

    def forward_active_action(
        self,
        action: torch.Tensor,
        *,
        action_rope_cache: RoPE4DPhaseCache,
        prefix: VAPAttentionKVCache,
    ) -> torch.Tensor:
        """Attend one noisy Action stage to exact cached clean-prefix K/V."""

        if action.ndim != 3 or action.shape[-1] != self.action_dim:
            raise ValueError(
                f"active action must be [B,S,{self.action_dim}], got {action.shape}"
            )
        aq = self._heads(self.action_norm_q(self.action_q(action)))
        ak = self._heads(self.action_norm_k(self.action_k(action)))
        av = self._heads(self.action_v(action))
        aq, ak = apply_rope_4d(
            aq,
            ak,
            None,
            axes_dim=self.rope_axes_dim,
            phase_cache=action_rope_cache,
        )
        query = aq.transpose(1, 2)
        active_key = ak.transpose(1, 2)
        active_value = av.transpose(1, 2)
        expected_prefix = (
            action.shape[0],
            self.num_heads,
            prefix.key.shape[-2],
            self.head_dim,
        )
        if tuple(prefix.key.shape) != expected_prefix:
            raise ValueError(
                f"prefix KV shape {tuple(prefix.key.shape)} != {expected_prefix}"
            )
        if prefix.key.device != query.device or prefix.key.dtype != query.dtype:
            raise ValueError("prefix KV must share active query device and dtype")
        key, value = prefix.update_active(active_key, active_value)
        # All tokens inside one Action stage share the same causal rank, hence
        # they are mutually visible. Only the immutable earlier-stage prefix is
        # prepended; no triangular token-level mask is correct here.
        output = F.scaled_dot_product_attention(query, key, value)
        return self.action_o(output.transpose(1, 2).flatten(2))

    @torch.no_grad()
    def initialize_from_wan_attention(self, wan_attention: nn.Module) -> None:
        """Copy video weights exactly and resize them into the action expert."""

        mapping = (("q", "q"), ("k", "k"), ("v", "v"), ("o", "o"))
        for ours, source in mapping:
            src = getattr(wan_attention, source)
            copy_linear(getattr(self, f"video_{ours}"), src, allow_resize=False)
            copy_linear(getattr(self, f"action_{ours}"), src, allow_resize=True)
        copy_norm(self.video_norm_q, wan_attention.norm_q, allow_resize=False)
        copy_norm(self.video_norm_k, wan_attention.norm_k, allow_resize=False)
        copy_norm(self.action_norm_q, wan_attention.norm_q, allow_resize=True)
        copy_norm(self.action_norm_k, wan_attention.norm_k, allow_resize=True)

    @torch.no_grad()
    def initialize_action_from_wan_attention(self, wan_attention: nn.Module) -> None:
        for ours, source in (("q", "q"), ("k", "k"), ("v", "v"), ("o", "o")):
            copy_linear(
                getattr(self, f"action_{ours}"),
                getattr(wan_attention, source),
                allow_resize=True,
            )
        copy_norm(self.action_norm_q, wan_attention.norm_q, allow_resize=True)
        copy_norm(self.action_norm_k, wan_attention.norm_k, allow_resize=True)


class VAPTextCrossAttention(nn.Module):
    """Minimal T5-conditioning path shared by both native experts.

    This is not a VLM planner. It preserves Wan's existing text context as an
    always-visible condition, with separate query/output projections for Video
    and Action and shared context K/V projections.
    """

    def __init__(
        self,
        *,
        video_dim: int,
        action_dim: int,
        context_dim: int,
        num_heads: int,
        head_dim: int,
        rope_axes_dim: Sequence[int],
        eps: float = 1e-6,
        wan_attention: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.video_dim = video_dim
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.joint_dim = num_heads * head_dim
        self.rope_axes_dim = tuple(rope_axes_dim)
        self.video_q = (
            nn.Linear(video_dim, self.joint_dim)
            if wan_attention is None
            else wan_attention.q
        )
        self.action_q = nn.Linear(action_dim, self.joint_dim)
        self.context_k = (
            nn.Linear(context_dim, self.joint_dim)
            if wan_attention is None
            else wan_attention.k
        )
        self.context_v = (
            nn.Linear(context_dim, self.joint_dim)
            if wan_attention is None
            else wan_attention.v
        )
        self.video_o = (
            nn.Linear(self.joint_dim, video_dim)
            if wan_attention is None
            else wan_attention.o
        )
        self.action_o = nn.Linear(self.joint_dim, action_dim)
        self.video_norm_q = (
            RMSNorm(self.joint_dim, eps=eps)
            if wan_attention is None
            else wan_attention.norm_q
        )
        self.action_norm_q = RMSNorm(self.joint_dim, eps=eps)
        self.context_norm_k = (
            RMSNorm(self.joint_dim, eps=eps)
            if wan_attention is None
            else wan_attention.norm_k
        )
        if wan_attention is not None:
            self.initialize_action_from_wan_cross_attention(wan_attention)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.unflatten(-1, (self.num_heads, self.head_dim))

    def project_context_kv(
        self,
        context: torch.Tensor,
        *,
        text_position_ids: torch.Tensor,
        text_rope_cache: RoPE4DPhaseCache | None = None,
    ) -> VAPAttentionKVCache:
        if context.ndim != 3 or context.shape[-1] != self.context_dim:
            raise ValueError(
                f"context must be [B,L,{self.context_dim}], got {context.shape}"
            )
        if tuple(text_position_ids.shape) != (*context.shape[:2], 4):
            raise ValueError("text_position_ids must be [B,L,4]")
        key = self._heads(self.context_norm_k(self.context_k(context)))
        value = self._heads(self.context_v(context))
        key = apply_rope_4d(
            key,
            key,
            text_position_ids,
            axes_dim=self.rope_axes_dim,
            phase_cache=text_rope_cache,
        )[0]
        return VAPAttentionKVCache(
            key=key.transpose(1, 2),
            value=value.transpose(1, 2),
        )

    def forward(
        self,
        video: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
        *,
        video_position_ids: torch.Tensor,
        action_position_ids: torch.Tensor,
        text_position_ids: torch.Tensor,
        return_context_kv: bool = False,
        video_rope_cache: RoPE4DPhaseCache | None = None,
        action_rope_cache: RoPE4DPhaseCache | None = None,
        text_rope_cache: RoPE4DPhaseCache | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, VAPAttentionKVCache]
    ):
        if context.ndim != 3 or context.shape[-1] != self.context_dim:
            raise ValueError(
                f"context must be [B,L,{self.context_dim}], got {context.shape}"
            )
        if context.shape[0] != video.shape[0] or context.shape[0] != action.shape[0]:
            raise ValueError("context/video/action batch sizes must match")
        expected = {
            "video": (video.shape[0], video.shape[1], 4),
            "action": (action.shape[0], action.shape[1], 4),
            "text": (context.shape[0], context.shape[1], 4),
        }
        actual = {
            "video": tuple(video_position_ids.shape),
            "action": tuple(action_position_ids.shape),
            "text": tuple(text_position_ids.shape),
        }
        if actual != expected:
            raise ValueError(
                f"hierarchical text-attention positions {actual} != {expected}"
            )
        context_kv = self.project_context_kv(
            context,
            text_position_ids=text_position_ids,
            text_rope_cache=text_rope_cache,
        )
        video_query = self._heads(self.video_norm_q(self.video_q(video)))
        action_query = self._heads(self.action_norm_q(self.action_q(action)))
        video_query = apply_rope_4d(
            video_query,
            video_query,
            video_position_ids,
            axes_dim=self.rope_axes_dim,
            phase_cache=video_rope_cache,
        )[0]
        action_query = apply_rope_4d(
            action_query,
            action_query,
            action_position_ids,
            axes_dim=self.rope_axes_dim,
            phase_cache=action_rope_cache,
        )[0]
        video_query = video_query.transpose(1, 2)
        action_query = action_query.transpose(1, 2)
        video_output = F.scaled_dot_product_attention(
            video_query, context_kv.key, context_kv.value
        )
        action_output = F.scaled_dot_product_attention(
            action_query, context_kv.key, context_kv.value
        )
        video_output = video_output.transpose(1, 2).flatten(2)
        action_output = action_output.transpose(1, 2).flatten(2)
        result = (self.video_o(video_output), self.action_o(action_output))
        if return_context_kv:
            return (*result, context_kv)
        return result

    def forward_with_anchor(
        self,
        video: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor | None,
        *,
        video_position_ids: torch.Tensor,
        action_position_ids: torch.Tensor,
        text_position_ids: torch.Tensor | None,
        anchor_context: VAPAttentionKVCache,
        anchor_camera_ids: torch.Tensor,
        video_camera_ids: torch.Tensor,
        condition_video_tokens: int,
        anchor_video_gate: torch.Tensor,
        anchor_action_gate: torch.Tensor,
        return_context_kv: bool = False,
        video_rope_cache: RoPE4DPhaseCache | None = None,
        action_rope_cache: RoPE4DPhaseCache | None = None,
        text_rope_cache: RoPE4DPhaseCache | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, VAPAttentionKVCache | None]
    ):
        """Apply Text and an independent first-frame bank through shared Q/O.

        Text and Anchor use separate attention softmaxes. Their head-space
        results are combined before the existing output projection so its bias
        is applied exactly once.
        """

        batch = video.shape[0]
        if action.shape[0] != batch:
            raise ValueError("video/action batch sizes must match")
        if tuple(video_position_ids.shape) != (batch, video.shape[1], 4):
            raise ValueError("video_position_ids must be [B,Sv,4]")
        if tuple(action_position_ids.shape) != (batch, action.shape[1], 4):
            raise ValueError("action_position_ids must be [B,Sa,4]")
        if not 0 < int(condition_video_tokens) <= video.shape[1]:
            raise ValueError(
                "condition_video_tokens must cover I and may equal the Video "
                "length for an Action-only stage"
            )
        if tuple(video_camera_ids.shape) != tuple(video.shape[:2]):
            raise ValueError("video_camera_ids must be [B,Sv]")
        if tuple(anchor_camera_ids.shape) != (
            batch,
            anchor_context.key.shape[-2],
        ):
            raise ValueError("anchor_camera_ids must be [B,M]")
        expected_anchor_prefix = (batch, self.num_heads)
        if anchor_context.key.shape[:2] != expected_anchor_prefix:
            raise ValueError("first-frame Anchor K/V head shape is incompatible")

        context_kv = None
        if context is not None:
            if context.ndim != 3 or context.shape[0] != batch:
                raise ValueError("context must be [B,L,C]")
            if text_position_ids is None:
                raise ValueError("text_position_ids are required with context")
            context_kv = self.project_context_kv(
                context,
                text_position_ids=text_position_ids,
                text_rope_cache=text_rope_cache,
            )
        elif text_position_ids is not None:
            raise ValueError("text_position_ids were provided without context")

        video_query = self._heads(self.video_norm_q(self.video_q(video)))
        action_query = self._heads(self.action_norm_q(self.action_q(action)))
        video_query = apply_rope_4d(
            video_query,
            video_query,
            video_position_ids,
            axes_dim=self.rope_axes_dim,
            phase_cache=video_rope_cache,
        )[0].transpose(1, 2)
        action_query = apply_rope_4d(
            action_query,
            action_query,
            action_position_ids,
            axes_dim=self.rope_axes_dim,
            phase_cache=action_rope_cache,
        )[0].transpose(1, 2)

        if context_kv is None:
            video_output = torch.zeros_like(video_query)
            action_output = torch.zeros_like(action_query)
        else:
            video_output = F.scaled_dot_product_attention(
                video_query, context_kv.key, context_kv.value
            )
            action_output = F.scaled_dot_product_attention(
                action_query, context_kv.key, context_kv.value
            )

        if int(condition_video_tokens) == video.shape[1]:
            anchor_video = torch.zeros_like(video_query)
        else:
            future_query = video_query[:, :, condition_video_tokens:, :]
            same_view = (
                video_camera_ids[:, condition_video_tokens:, None]
                == anchor_camera_ids[:, None, :]
            )
            has_same_view = same_view.any(dim=-1)
            # Avoid relying on backend-specific all-masked-row behavior.  A
            # query without a same-camera Anchor temporarily sees all keys,
            # then its complete Memory contribution is explicitly zeroed.
            safe_same_view = same_view | (~has_same_view[..., None])
            anchor_video = F.scaled_dot_product_attention(
                future_query,
                anchor_context.key,
                anchor_context.value,
                attn_mask=safe_same_view[:, None, :, :],
            )
            anchor_video = anchor_video * has_same_view[:, None, :, None]
            anchor_video = F.pad(
                anchor_video,
                (0, 0, int(condition_video_tokens), 0),
            )
        anchor_action = F.scaled_dot_product_attention(
            action_query, anchor_context.key, anchor_context.value
        )
        video_gate = (
            torch.tanh(anchor_video_gate)
            .to(video_output)
            .reshape(1, self.num_heads, 1, 1)
        )
        action_gate = (
            torch.tanh(anchor_action_gate)
            .to(action_output)
            .reshape(1, self.num_heads, 1, 1)
        )
        video_output = video_output + video_gate * anchor_video
        action_output = action_output + action_gate * anchor_action
        video_result = self.video_o(video_output.transpose(1, 2).flatten(2))
        action_result = self.action_o(action_output.transpose(1, 2).flatten(2))
        if context_kv is None:
            # With no Text path, zero gates must still be an exact no-op even
            # when the inherited Wan output projections contain a bias.
            if self.video_o.bias is not None:
                video_result = video_result - self.video_o.bias.to(video_result)
            if self.action_o.bias is not None:
                action_result = action_result - self.action_o.bias.to(action_result)
        result = (video_result, action_result)
        if return_context_kv:
            return (*result, context_kv)
        return result

    def forward_active_action(
        self,
        action: torch.Tensor,
        *,
        action_rope_cache: RoPE4DPhaseCache,
        context: VAPAttentionKVCache,
    ) -> torch.Tensor:
        """Cross-attend one Action stage to cached static instruction K/V."""

        if action.ndim != 3 or action.shape[-1] != self.action_dim:
            raise ValueError(
                f"active action must be [B,S,{self.action_dim}], got {action.shape}"
            )
        query = self._heads(self.action_norm_q(self.action_q(action)))
        query = apply_rope_4d(
            query,
            query,
            None,
            axes_dim=self.rope_axes_dim,
            phase_cache=action_rope_cache,
        )[0].transpose(1, 2)
        if (
            context.key.shape[:2] != query.shape[:2]
            or context.key.shape[-1] != query.shape[-1]
        ):
            raise ValueError("cached text KV is incompatible with active action query")
        if context.key.device != query.device or context.key.dtype != query.dtype:
            raise ValueError("cached text KV must share active query device and dtype")
        output = F.scaled_dot_product_attention(query, context.key, context.value)
        return self.action_o(output.transpose(1, 2).flatten(2))

    def forward_active_action_with_anchor(
        self,
        action: torch.Tensor,
        *,
        action_rope_cache: RoPE4DPhaseCache,
        text_context: VAPAttentionKVCache | None,
        anchor_context: VAPAttentionKVCache,
        anchor_action_gate: torch.Tensor,
    ) -> torch.Tensor:
        """Active Action decode with the same static Anchor Bank as full forward."""

        if action.ndim != 3 or action.shape[-1] != self.action_dim:
            raise ValueError(
                f"active action must be [B,S,{self.action_dim}], got {action.shape}"
            )
        query = self._heads(self.action_norm_q(self.action_q(action)))
        query = apply_rope_4d(
            query,
            query,
            None,
            axes_dim=self.rope_axes_dim,
            phase_cache=action_rope_cache,
        )[0].transpose(1, 2)
        if text_context is None:
            output = torch.zeros_like(query)
        else:
            output = F.scaled_dot_product_attention(
                query, text_context.key, text_context.value
            )
        anchor_output = F.scaled_dot_product_attention(
            query, anchor_context.key, anchor_context.value
        )
        gate = (
            torch.tanh(anchor_action_gate).to(output).reshape(1, self.num_heads, 1, 1)
        )
        result = self.action_o(
            (output + gate * anchor_output).transpose(1, 2).flatten(2)
        )
        if text_context is None and self.action_o.bias is not None:
            result = result - self.action_o.bias.to(result)
        return result

    @torch.no_grad()
    def initialize_from_wan_cross_attention(self, wan_attention: nn.Module) -> None:
        copy_linear(self.video_q, wan_attention.q, allow_resize=False)
        copy_linear(self.action_q, wan_attention.q, allow_resize=True)
        copy_linear(self.context_k, wan_attention.k, allow_resize=False)
        copy_linear(self.context_v, wan_attention.v, allow_resize=False)
        copy_linear(self.video_o, wan_attention.o, allow_resize=False)
        copy_linear(self.action_o, wan_attention.o, allow_resize=True)
        copy_norm(self.video_norm_q, wan_attention.norm_q, allow_resize=False)
        copy_norm(self.action_norm_q, wan_attention.norm_q, allow_resize=True)
        copy_norm(self.context_norm_k, wan_attention.norm_k, allow_resize=False)

    @torch.no_grad()
    def initialize_action_from_wan_cross_attention(
        self, wan_attention: nn.Module
    ) -> None:
        copy_linear(self.action_q, wan_attention.q, allow_resize=True)
        copy_linear(self.action_o, wan_attention.o, allow_resize=True)
        copy_norm(self.action_norm_q, wan_attention.norm_q, allow_resize=True)


__all__ = ["RMSNorm", "VAPJointAttention", "VAPTextCrossAttention"]
