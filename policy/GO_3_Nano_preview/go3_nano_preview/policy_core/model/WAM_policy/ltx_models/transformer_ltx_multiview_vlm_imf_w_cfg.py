# Copyright 2024 The Genmo team and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.forward_ad as fwAD

from einops import rearrange

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm
import torch.utils.checkpoint

from policy_core.model.WAM_policy.ltx_models.ltx_attention_processor import Attention
from policy_core.model.WAM_policy.vlm_patches.vlm_token_select import slice_vlm_text_part
from policy_core.utils.WAM_policy.video_sparse_utils import centered_spatial_rope_axis


logger = logging.get_logger(__name__)

from policy_core.utils.WAM_policy import import_custom_class


class LTXVideoAttentionProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "LTXVideoAttentionProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        n_view: int = 1,
        cross_view_attn: bool = False,
        is_causal: bool = False,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )


        if cross_view_attn:
            assert n_view is not None
            if attention_mask is not None:
                batch_size = batch_size // n_view
                attention_mask = attention_mask.repeat(batch_size, n_view, n_view)
                sequence_length = attention_mask.shape[-1]

        elif image_rotary_emb is not None and attention_mask is not None:
            attention_mask = attention_mask.repeat(batch_size, 1, 1)
            sequence_length = attention_mask.shape[-1]


        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states


        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
            if cross_view_attn:
                query = rearrange(query, '(b v) l c -> b (v l) c', v=n_view)
                key = rearrange(key, '(b v) l c -> b (v l) c', v=n_view)
                value = rearrange(value, '(b v) l c -> b (v l) c', v=n_view)

        else:
            query = rearrange(query, '(b v) l c -> b (v l) c', v=n_view)


        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)


        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=is_causal
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if cross_view_attn or image_rotary_emb is None:
            hidden_states = rearrange(hidden_states, 'b (v l) c -> (b v) l c', v=n_view)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


### modified from cosmos
class PatchEmbed(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, patch_size: Tuple[int, int, int], bias: bool = True
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size[0] * patch_size[1] * patch_size[2], out_channels, bias=bias)

    def forward(self, hidden_states, num_frames, height, width) -> torch.Tensor:
        batch_size, seq_length, num_channels = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        hidden_states = hidden_states.reshape(
            batch_size, num_channels, num_frames // p_t, p_t, height // p_h, p_h, width // p_w, p_w
        )
        hidden_states = hidden_states.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1,3)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class LTXVideoRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        base_num_frames: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        patch_size: int = 1,
        patch_size_t: int = 1,
        theta: float = 10000.0,
        spatial_rope_grid_mode: str = "legacy",
    ) -> None:
        super().__init__()

        self.dim = dim
        self.base_num_frames = base_num_frames
        self.base_height = base_height
        self.base_width = base_width
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.theta = theta
        spatial_rope_grid_mode = str(spatial_rope_grid_mode).strip().lower()
        if spatial_rope_grid_mode not in {"legacy", "canonical_center"}:
            raise ValueError(
                "spatial_rope_grid_mode must be 'legacy' or 'canonical_center'. "
                f"Got {spatial_rope_grid_mode!r}."
            )
        self.spatial_rope_grid_mode = spatial_rope_grid_mode

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_interpolation_scale: Optional[Tuple[torch.Tensor, float, float]] = None,
        num_frames=None, height=None, width=None,
        video_rope_coords: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size = hidden_states.shape[0]

        if video_rope_coords is None:
            if self.spatial_rope_grid_mode == "canonical_center":
                grid_h = centered_spatial_rope_axis(
                    height, self.base_height, device=hidden_states.device
                )
                grid_w = centered_spatial_rope_axis(
                    width, self.base_width, device=hidden_states.device
                )
            else:
                grid_h = torch.arange(height, dtype=torch.float32, device=hidden_states.device)
                grid_w = torch.arange(width, dtype=torch.float32, device=hidden_states.device)
            grid_f = torch.arange(num_frames, dtype=torch.float32, device=hidden_states.device)
            grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
            grid = torch.stack(grid, dim=0)
            grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
            grid = grid.flatten(2, 4).transpose(1, 2)
        else:
            if video_rope_coords.ndim != 3 or video_rope_coords.shape[-1] != 3:
                raise ValueError(
                    f"Expected video_rope_coords of shape [batch, tokens, 3], got {tuple(video_rope_coords.shape)}."
                )
            if video_rope_coords.shape[0] != batch_size:
                raise ValueError(
                    "video_rope_coords batch dimension must match hidden_states. "
                    f"Got {video_rope_coords.shape[0]} and {batch_size}."
                )
            grid = video_rope_coords.to(device=hidden_states.device, dtype=torch.float32)

        if rope_interpolation_scale is not None:
            if video_rope_coords is None:
                grid[:, :, 0] = grid[:, :, 0] * rope_interpolation_scale[0] * self.patch_size_t / self.base_num_frames
                grid[:, :, 1] = grid[:, :, 1] * rope_interpolation_scale[1] * self.patch_size / self.base_height
                grid[:, :, 2] = grid[:, :, 2] * rope_interpolation_scale[2] * self.patch_size / self.base_width
            else:
                grid = grid.clone()
                grid[:, :, 0] = grid[:, :, 0] * rope_interpolation_scale[0] * self.patch_size_t / self.base_num_frames
                grid[:, :, 1] = grid[:, :, 1] * rope_interpolation_scale[1] * self.patch_size / self.base_height
                grid[:, :, 2] = grid[:, :, 2] * rope_interpolation_scale[2] * self.patch_size / self.base_width


        start = 1.0
        end = self.theta
        freqs = self.theta ** torch.linspace(
            math.log(start, self.theta),
            math.log(end, self.theta),
            self.dim // 6,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        freqs = freqs * math.pi / 2.0
        freqs = freqs * (grid.unsqueeze(-1) * 2 - 1)
        freqs = freqs.transpose(-1, -2).flatten(2)

        cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

        if self.dim % 6 != 0:
            cos_padding = torch.ones_like(cos_freqs[:, :, : self.dim % 6])
            sin_padding = torch.zeros_like(cos_freqs[:, :, : self.dim % 6])
            cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
            sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        return cos_freqs, sin_freqs

# Transformer block used in [LTX](https://huggingface.co/Lightricks/LTX-Video).
@maybe_allow_in_graph
class LTXVideoTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            processor=LTXVideoAttentionProcessor2_0(),
        )

        self.norm2 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            processor=LTXVideoAttentionProcessor2_0(),
        )

        self.ff = FeedForward(dim, activation_fn=activation_fn)

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        n_view: int = None,
        cross_view_attn: bool = False,
        self_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        norm_hidden_states = self.norm1(hidden_states)

        num_ada_params = self.scale_shift_table.shape[0]
        ada_values = self.scale_shift_table[None, None] + temb.reshape(batch_size, temb.size(1), num_ada_params, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa


        attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            image_rotary_emb=image_rotary_emb,
            n_view=n_view,
            cross_view_attn=cross_view_attn,
            attention_mask=self_attention_mask,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa

        attn_hidden_states = self.attn2(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=None,
            attention_mask=encoder_attention_mask,
            n_view=n_view,
        )
        hidden_states = hidden_states + attn_hidden_states
        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states


class SimpleGateFusion(nn.Module):
    def __init__(self, hidden_size, num_layers, normalize_gate=False, gate_bias=True):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1, bias=gate_bias),
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.normalize_gate = normalize_gate

    def forward(self, hidden_states_list, attention_mask=None):
        if attention_mask is None:
            pooled = torch.stack([hs.mean(dim=1) for hs in hidden_states_list], dim=1)
        else:
            m = attention_mask.to(hidden_states_list[0].dtype).unsqueeze(-1)
            denom = m.sum(dim=1).clamp(min=1.0)
            pooled = torch.stack(
                [(hs * m).sum(dim=1) / denom for hs in hidden_states_list], dim=1
            )
        if self.normalize_gate:
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
        weights = self.gate(pooled).softmax(dim=1)

        output = torch.zeros_like(hidden_states_list[0])
        for i, hs in enumerate(hidden_states_list):
            output = output + weights[:, i:i+1] * hs

        return self.norm(output)


@maybe_allow_in_graph
class IMFMultiConditionAdaLayerNormSingle(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        *,
        condition_dim: Optional[int] = None,
        fourier_dim: int = 256,
        use_cfg_conditioning: bool = False,
    ) -> None:
        super().__init__()
        condition_dim = condition_dim or embedding_dim // 4
        self.embedding_dim = embedding_dim
        self.condition_dim = condition_dim
        self.use_cfg_conditioning = bool(use_cfg_conditioning)

        def _make_scalar_encoder():
            projection = Timesteps(
                num_channels=fourier_dim,
                flip_sin_to_cos=True,
                downscale_freq_shift=0,
            )
            embedder = TimestepEmbedding(fourier_dim, condition_dim)
            return projection, embedder

        self.t_projector, self.t_embedder = _make_scalar_encoder()
        self.h_projector, self.h_embedder = _make_scalar_encoder()
        if self.use_cfg_conditioning:
            self.g_projector, self.g_embedder = _make_scalar_encoder()

        num_conditions = 3 if self.use_cfg_conditioning else 2
        self.fuse = nn.Sequential(
            nn.Linear(num_conditions * condition_dim, embedding_dim),
            nn.SiLU(),
            RMSNorm(embedding_dim, eps=1e-6, elementwise_affine=False),
            nn.Linear(embedding_dim, embedding_dim, bias=False),
        )
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=False)
        self.zero_init_conditioning_output()

    @staticmethod
    def _as_batch_tokens(
        value: torch.Tensor,
        batch_size: int,
        name: str,
    ) -> torch.Tensor:
        if value.ndim == 0:
            return value.reshape(1, 1).expand(batch_size, 1)
        if value.ndim == 1:
            if value.shape[0] == batch_size:
                return value[:, None]
            if value.shape[0] == 1:
                return value.reshape(1, 1).expand(batch_size, 1)
            raise ValueError(
                f"{name} has shape {tuple(value.shape)}; expected scalar or "
                f"one value per model row ({batch_size})."
            )
        if value.ndim != 2:
            raise ValueError(
                f"{name} must be scalar, [B], or [B,K], got {tuple(value.shape)}."
            )
        if value.shape[0] == batch_size:
            return value
        if value.shape[0] == 1:
            return value.expand(batch_size, -1)
        raise ValueError(
            f"{name} batch dimension {value.shape[0]} does not match {batch_size}."
        )

    @staticmethod
    def _broadcast_token_dims(named_values) -> list[torch.Tensor]:
        num_tokens = max(value.shape[1] for _, value in named_values)
        expanded = []
        for name, value in named_values:
            if value.shape[1] not in (1, num_tokens):
                raise ValueError(
                    f"{name} has K={value.shape[1]}; expected 1 or {num_tokens}."
                )
            expanded.append(value.expand(-1, num_tokens))
        return expanded

    def _embed_scalar(
        self,
        value: torch.Tensor,
        projector: nn.Module,
        embedder: nn.Module,
    ) -> torch.Tensor:
        projected = projector(value.float().reshape(-1))
        projected = projected.to(dtype=embedder.linear_1.weight.dtype)
        embedded = embedder(projected)
        return embedded.view(value.shape[0], value.shape[1], -1)

    def _zero_centered_embed(
        self,
        value: torch.Tensor,
        projector: nn.Module,
        embedder: nn.Module,
    ) -> torch.Tensor:
        embedded = self._embed_scalar(value, projector, embedder)
        embedded_zero = self._embed_scalar(
            torch.zeros_like(value), projector, embedder
        )
        return embedded - embedded_zero

    def zero_init_conditioning_output(self) -> None:
        nn.init.zeros_(self.fuse[-1].weight)

    def forward(
        self,
        timestep: torch.Tensor,
        *,
        timestep_r: torch.Tensor,
        omega: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        hidden_dtype: Optional[torch.dtype] = None,
    ):
        if batch_size is None:
            batch_size = 1 if timestep.ndim == 0 else timestep.shape[0]

        t = self._as_batch_tokens(timestep, batch_size, "timestep")
        r = self._as_batch_tokens(timestep_r, batch_size, "timestep_r")
        named_values = [("timestep", t), ("timestep_r", r)]

        if self.use_cfg_conditioning:
            if omega is None:
                raise ValueError(
                    "omega is required when use_cfg_conditioning=True; "
                    "use omega=1 for the unguided conditional point."
                )
            omega = self._as_batch_tokens(omega, batch_size, "omega")
            named_values.append(("omega", omega))
        elif omega is not None:
            raise ValueError(
                "omega was provided but use_cfg_conditioning=False."
            )

        values = self._broadcast_token_dims(named_values)
        t_full = values[0].float()
        r_full = values[1].float()
        h_full = t_full - r_full
        conditions = [
            self._embed_scalar(t_full, self.t_projector, self.t_embedder),
            self._zero_centered_embed(
                h_full, self.h_projector, self.h_embedder
            ),
        ]
        if self.use_cfg_conditioning:
            g_full = 1.0 - values[2].float().reciprocal()
            conditions.append(
                self._zero_centered_embed(
                    g_full, self.g_projector, self.g_embedder
                )
            )

        embedded_timestep = self.fuse(torch.cat(conditions, dim=-1))
        temb = self.linear(self.silu(embedded_timestep))
        if hidden_dtype is not None:
            temb = temb.to(dtype=hidden_dtype)
            embedded_timestep = embedded_timestep.to(dtype=hidden_dtype)
        return (
            temb.reshape(-1, 6 * self.embedding_dim),
            embedded_timestep.reshape(-1, self.embedding_dim),
        )


# A Transformer model for video-like data used in [LTX](https://huggingface.co/Lightricks/LTX-Video).
class LTXVideoTransformer3DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        patch_size: int = 1,
        patch_size_t: int = 1,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        cross_attention_dim: int = 2048,
        num_layers: int = 28,
        activation_fn: str = "gelu-approximate",
        qk_norm: str = "rms_norm_across_heads",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = 4096,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        use_view_embed: bool = True,
        max_view: int = 3,
        num_view_identities: int = None,
        action_expert: bool = False,
        vlm_interface_class_name = None,
        vlm_interface_class_path = None,
        vlm_model_id = None,
        vlm_layer_ids = None,
        vlm_fixed_sequence_length: Optional[int] = None,
        vlm_text_tokens_only: bool = False,
        vlm_head_view_only: bool = False,
        vlm_head_view_index: int = 0,
        v_head_depth: int = 4,
        use_t_and_r_embedding: bool = False,
        use_cfg_conditioning: bool = False,
        time_embed_zero_init: bool = False,
        frame_concat_conditioning: bool = False,
        frame_concat_mode_dim: int = 32,
        t2v_mode: bool = False,
        goal_mode_null_text_tokens: int = 0,
        spatial_rope_grid_mode: str = "legacy",
        **kwargs,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        inner_dim = num_attention_heads * attention_head_dim

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.t2v_mode = bool(t2v_mode)
        if self.t2v_mode and (self.patch_size != 1 or self.patch_size_t != 1):
            raise ValueError(
                "t2v_mode currently requires patch_size == patch_size_t == 1, "
                f"got patch_size={self.patch_size!r}, patch_size_t={self.patch_size_t!r}."
            )
        if self.t2v_mode and frame_concat_conditioning:
            raise ValueError(
                "t2v_mode is incompatible with frame_concat_conditioning."
            )

        self.frame_concat_conditioning = frame_concat_conditioning
        self.frame_concat_mode_dim = frame_concat_mode_dim
        proj_in_channels = in_channels
        if frame_concat_conditioning:
            proj_in_channels = in_channels + in_channels + frame_concat_mode_dim
            self.mode_embed = nn.Embedding(2, frame_concat_mode_dim)
        if self.patch_size > 1 or self.patch_size_t > 1:
            self.proj_in = PatchEmbed(
                proj_in_channels, inner_dim, patch_size, bias=False
            )
        else:
            self.proj_in = nn.Linear(proj_in_channels, inner_dim)

        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.use_t_and_r_embedding = bool(use_t_and_r_embedding)
        self.use_cfg_conditioning = bool(use_cfg_conditioning)
        if self.use_cfg_conditioning and not self.use_t_and_r_embedding:
            raise ValueError(
                "use_cfg_conditioning=True requires use_t_and_r_embedding=True."
            )
        if self.use_t_and_r_embedding:
            self.time_embed = IMFMultiConditionAdaLayerNormSingle(
                inner_dim,
                use_cfg_conditioning=self.use_cfg_conditioning,
            )
        else:
            self.time_embed = AdaLayerNormSingle(
                inner_dim, use_additional_conditions=False
            )

        self.time_embed_zero_init = time_embed_zero_init
        if time_embed_zero_init:
            self._zero_init_time_embedder(self.time_embed)

        self.use_view_embed = use_view_embed
        self.max_view = max_view
        self.num_view_identities = num_view_identities if num_view_identities is not None else max_view
        self.inner_dim = inner_dim
        if self.use_view_embed:
            self.make_view_emb(self.num_view_identities, inner_dim)
            self.view_ada = nn.Sequential(
                                            nn.SiLU(),
                                            nn.Linear(inner_dim, 6 * inner_dim, bias=True)
                                        )

        self.vlm_text_tokens_only = vlm_text_tokens_only
        self.vlm_head_view_only = vlm_head_view_only
        self.vlm_head_view_index = vlm_head_view_index

        if vlm_interface_class_name is not None:
            vlm_interface_class = import_custom_class(
                vlm_interface_class_name, vlm_interface_class_path,
            )
            self.vlm_interface = vlm_interface_class(
                vlm_model_id,
                fixed_sequence_length=vlm_fixed_sequence_length,
            )
            if vlm_layer_ids is None:
                self.vlm_layer_ids = list(range(num_layers))
            else:
                self.vlm_layer_ids = vlm_layer_ids

            self.fuser = SimpleGateFusion(
                hidden_size=2048,
                num_layers=29,
                normalize_gate=kwargs.get("fuser_normalize_gate", False),
                gate_bias=kwargs.get("fuser_gate_bias", True),
            )
        else:
            self.vlm_interface = None
            self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)

        self.goal_mode_null_text_tokens = int(goal_mode_null_text_tokens or 0)
        if self.goal_mode_null_text_tokens > 0:
            if self.vlm_interface is None:
                raise ValueError(
                    "goal_mode_null_text_tokens requires the VLM/fuser "
                    "conditioning path (set vlm_interface_class_name)."
                )
            self.null_text_embedding = nn.Parameter(
                torch.randn(1, self.goal_mode_null_text_tokens, 2048) * 0.02
            )

        self.rope = LTXVideoRotaryPosEmbed(
            dim=inner_dim,
            base_num_frames=9,
            base_height=4,
            base_width=6,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            theta=10000.0,
            spatial_rope_grid_mode=spatial_rope_grid_mode,
        )

        block_kwargs = dict(
            dim=inner_dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            qk_norm=qk_norm,
            activation_fn=activation_fn,
            attention_bias=attention_bias,
            attention_out_bias=attention_out_bias,
            eps=norm_eps,
            elementwise_affine=norm_elementwise_affine,
        )

        self.transformer_blocks = nn.ModuleList(
            [LTXVideoTransformerBlock(**block_kwargs) for _ in range(num_layers)]
        )

        self.v_head_depth = v_head_depth
        self.v_head_blocks = nn.ModuleList(
            [LTXVideoTransformerBlock(**block_kwargs) for _ in range(v_head_depth)]
        )

        self.norm_out = nn.LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)

        if self.patch_size > 1 or self.patch_size_t > 1:
            self.proj_out = nn.Linear(
                inner_dim, patch_size[0] * patch_size[1] * patch_size[2] * out_channels, bias=False
            )
        else:
            self.proj_out = nn.Linear(inner_dim, out_channels)

        self.norm_out_v = nn.LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)
        self.scale_shift_table_v = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)

        if self.patch_size > 1 or self.patch_size_t > 1:
            self.proj_out_v = nn.Linear(
                inner_dim, patch_size[0] * patch_size[1] * patch_size[2] * out_channels, bias=False
            )
        else:
            self.proj_out_v = nn.Linear(inner_dim, out_channels)

        self.gradient_checkpointing = False


    def init_missing_v_head_from_u_head(self, missing_keys) -> None:
        if self.v_head_depth == 0:
            return
        v_keys_missing = [k for k in missing_keys if k.startswith("v_head_blocks.")]
        if not v_keys_missing:
            return
        n_v_keys = sum(1 for k in self.state_dict() if k.startswith("v_head_blocks."))
        if len(v_keys_missing) != n_v_keys:
            logger.warning(
                "Checkpoint has a PARTIAL v-head (%d/%d keys missing); leaving "
                "loaded values untouched and NOT copying from the u-head.",
                len(v_keys_missing), n_v_keys,
            )
            return
        shared_depth = len(self.transformer_blocks) - self.v_head_depth
        with torch.no_grad():
            for i, v_block in enumerate(self.v_head_blocks):
                v_block.load_state_dict(
                    self.transformer_blocks[shared_depth + i].state_dict()
                )
            self.proj_out_v.load_state_dict(self.proj_out.state_dict())
            self.scale_shift_table_v.data.copy_(self.scale_shift_table.data)
        logger.warning(
            "Checkpoint has no v-head: initialized %d v_head_blocks from u-head "
            "blocks %d-%d and proj_out_v/scale_shift_table_v from the u output "
            "head (the checkpoint's v output keys are untrained leftovers).",
            self.v_head_depth, shared_depth, len(self.transformer_blocks) - 1,
        )

    @staticmethod
    def _zero_init_time_embedder(ada_norm: nn.Module) -> None:
        zero_init = getattr(ada_norm, "zero_init_conditioning_output", None)
        if callable(zero_init):
            zero_init()
            return
        emb = getattr(ada_norm, "emb", None)
        timestep_embedder = getattr(emb, "timestep_embedder", None)
        linear_2 = getattr(timestep_embedder, "linear_2", None)
        if linear_2 is None:
            raise AttributeError(
                "time_embed_zero_init expects AdaLayerNormSingle to expose "
                "emb.timestep_embedder.linear_2; the installed diffusers "
                "version has a different structure."
            )
        nn.init.zeros_(linear_2.weight)
        if linear_2.bias is not None:
            nn.init.zeros_(linear_2.bias)

    def _apply_goal_mode_null_text(self, fused, vlm_attention_mask, goal_mode_mask):
        if getattr(self, "goal_mode_null_text_tokens", 0) <= 0:
            raise ValueError(
                "goal_mode_mask passed but the model was built with "
                "goal_mode_null_text_tokens=0; set video_model.config."
                "goal_mode_null_text_tokens (trainer injects it from "
                "goal_mode.null_text_tokens)."
            )
        b, l_text, _ = fused.shape
        mask = goal_mode_mask.to(device=fused.device)
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if mask.shape != (b,):
            raise ValueError(
                f"goal_mode_mask shape {tuple(mask.shape)} != encoder batch ({b},)"
            )
        null_tokens = self.null_text_embedding.to(dtype=fused.dtype)
        n_null = null_tokens.shape[1]
        l_out = max(l_text, n_null)
        text_states = F.pad(fused, (0, 0, 0, l_out - l_text))
        null_states = F.pad(null_tokens, (0, 0, 0, l_out - n_null)).expand(b, -1, -1)
        states = torch.where(mask.view(b, 1, 1), null_states, text_states)
        if vlm_attention_mask is not None:
            text_attn = F.pad(vlm_attention_mask.to(fused.dtype), (0, l_out - l_text))
        else:
            text_attn = F.pad(fused.new_ones(b, l_text), (0, l_out - l_text))
        null_attn = F.pad(fused.new_ones(1, n_null), (0, l_out - n_null)).expand(b, -1)
        attn = torch.where(mask.view(b, 1), null_attn, text_attn)
        return states, attn

    @staticmethod
    def align_time_condition_tokens(
        value: Optional[torch.Tensor],
        name: str,
        *,
        batch_size: int,
        hidden_sequence_length: int,
        num_frames: Optional[int],
        height: Optional[int],
        width: Optional[int],
        expand_singletons: bool = False,
    ) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if value.ndim == 0:
            value = value.reshape(1, 1).expand(batch_size, 1)
        elif value.ndim == 1:
            if value.shape[0] == batch_size:
                value = value[:, None]
            elif value.shape[0] == 1:
                value = value.reshape(1, 1).expand(batch_size, 1)
            else:
                raise ValueError(
                    f"{name} has shape {tuple(value.shape)}; expected [B] with "
                    f"B={batch_size}."
                )
        elif value.ndim == 2:
            if value.shape[0] == 1 and batch_size > 1:
                value = value.expand(batch_size, -1)
            elif value.shape[0] != batch_size:
                raise ValueError(
                    f"{name} batch dimension {value.shape[0]} != {batch_size}."
                )
        else:
            raise ValueError(
                f"{name} must be scalar, [B], or [B,K], got {tuple(value.shape)}."
            )

        condition_tokens = value.shape[1]
        if condition_tokens == 1:
            if expand_singletons:
                return value.expand(-1, hidden_sequence_length)
            return value
        if condition_tokens == hidden_sequence_length:
            return value

        if num_frames is not None and height is not None and width is not None:
            spatial_tokens = int(height) * int(width)
            if (
                condition_tokens == int(num_frames)
                and hidden_sequence_length == int(num_frames) * spatial_tokens
            ):
                return value.repeat_interleave(spatial_tokens, dim=1)

        raise ValueError(
            f"Cannot align {name} shape {tuple(value.shape)} with sequence "
            f"length {hidden_sequence_length}. Expected K=1, K=sequence length, "
            "or K=num_frames for a dense F*H*W layout."
        )

    def compute_time_conditioning(
        self,
        timestep: torch.Tensor,
        *,
        timestep_r: Optional[torch.Tensor] = None,
        omega: Optional[torch.Tensor] = None,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ):
        if self.use_t_and_r_embedding:
            if timestep_r is None:
                raise ValueError(
                    "use_t_and_r_embedding=True requires timestep_r."
                )
            if self.use_cfg_conditioning and omega is None:
                raise ValueError(
                    "use_cfg_conditioning=True requires omega; use omega=1 "
                    "for the unguided conditional point."
                )
            return self.time_embed(
                timestep,
                timestep_r=timestep_r,
                omega=omega,
                batch_size=batch_size,
                hidden_dtype=hidden_dtype,
            )

        if omega is not None:
            raise ValueError(
                "omega was provided to a model without joint CFG conditioning."
            )
        return self.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_dtype,
        )


    def make_view_emb(self, max_view, inner_dim, ):
        self.view_embed = nn.Parameter(torch.randn(max_view, inner_dim))
        if hasattr(self, "view_embed"):
            dtype = self.view_embed.dtype
            device = self.view_embed.device
            self.view_embed = self.view_embed.to(device=device, dtype=dtype)


    def forward(
        self,
        hidden_states: torch.Tensor = None,
        encoder_hidden_states = None,
        timestep: torch.LongTensor = None,
        timestep_r: torch.Tensor = None,
        omega: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
        n_view: int = 1,
        view_ids: Optional[torch.Tensor] = None,
        rope_interpolation_scale: Optional[Tuple[float, float, float]] = None,
        return_dict: bool = True,
        action_states: torch.Tensor = None,
        action_timestep: torch.LongTensor = None,
        return_video: bool = True,
        return_action: bool = False,
        store_buffer=False,
        video_states_buffer=None,
        video_attention_mask: torch.Tensor = None,
        history_action_state: torch.Tensor = None,
        num_frames: int = None,
        height: int = None,
        width: int = None,
        video_rope_coords: Optional[torch.FloatTensor] = None,
        vlm_inputs = None,
        token_dropout_ratio: float = 0.0,
        num_frames_wo_dropout: int = 0,
        precomputed_vlm_hidden_states: list = None,
        vlm_attention_mask: torch.Tensor = None,
        goal_mode_mask: torch.Tensor = None,
        cfg_unconditional_mask: torch.Tensor = None,
        compute_v_head: bool = True,
        cond_latents: torch.Tensor = None,
        cond_mode_ids: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:

        def _to_hidden_device(value):
            if value is None:
                return None
            if not torch.is_tensor(value):
                return torch.as_tensor(
                    value,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
            return value.to(device=hidden_states.device)

        timestep = _to_hidden_device(timestep)
        timestep_r = _to_hidden_device(timestep_r)
        omega = _to_hidden_device(omega)

        t2v_prefix_tokens = 0
        t2v_image_rotary_emb = None
        if self.t2v_mode:
            if num_frames is None or height is None or width is None:
                raise ValueError("t2v_mode requires num_frames, height and width.")
            if int(num_frames) < 2:
                raise ValueError(
                    f"t2v_mode requires memory + future frames, got num_frames={num_frames}."
                )
            if float(token_dropout_ratio) != 0.0:
                raise NotImplementedError(
                    "t2v_mode is incompatible with token dropout."
                )
            if (
                bool(getattr(self.config, "frame_concat_conditioning", False))
                or cond_latents is not None
                or cond_mode_ids is not None
            ):
                raise NotImplementedError(
                    "t2v_mode is incompatible with frame-concat conditioning."
                )
            if video_attention_mask is not None:
                raise NotImplementedError(
                    "t2v_mode currently supports the dense unmasked video path only."
                )

            input_batch = hidden_states.shape[0]
            input_num_frames = int(num_frames)
            t2v_prefix_tokens = int(height) * int(width)
            expected_tokens = input_num_frames * t2v_prefix_tokens
            if hidden_states.ndim != 3 or hidden_states.shape[1] != expected_tokens:
                raise ValueError(
                    "t2v_mode expects a dense frame-major token sequence: "
                    f"got hidden_states={tuple(hidden_states.shape)}, expected token length "
                    f"{input_num_frames}*{height}*{width}={expected_tokens}."
                )

            def _drop_first_timestep_frame(value, name):
                if value is None or value.ndim == 0:
                    return value
                if value.ndim == 1:
                    if value.shape[0] != input_batch:
                        raise ValueError(
                            f"{name} has shape {tuple(value.shape)}; expected one value "
                            f"per batch item ({input_batch})."
                        )
                    return value
                if value.shape[-1] == expected_tokens:
                    return value[..., t2v_prefix_tokens:]
                if value.shape[-1] == input_num_frames:
                    return value[..., 1:]
                if value.shape[-1] == 1:
                    return value
                raise ValueError(
                    f"t2v_mode cannot align {name} shape {tuple(value.shape)} with "
                    f"{input_num_frames} frames / {expected_tokens} tokens."
                )

            if video_rope_coords is not None and (
                video_rope_coords.ndim != 3
                or video_rope_coords.shape[1] != expected_tokens
                or video_rope_coords.shape[2] != 3
            ):
                raise ValueError(
                    "t2v_mode expects video_rope_coords [batch, dense_tokens, 3], "
                    f"got {tuple(video_rope_coords.shape)}."
                )

            full_image_rotary_emb = self.rope(
                hidden_states,
                rope_interpolation_scale,
                input_num_frames,
                height,
                width,
                video_rope_coords=video_rope_coords,
            )
            t2v_image_rotary_emb = tuple(
                value[:, t2v_prefix_tokens:] for value in full_image_rotary_emb
            )
            hidden_states = hidden_states[:, t2v_prefix_tokens:]
            timestep = _drop_first_timestep_frame(timestep, "timestep")
            timestep_r = _drop_first_timestep_frame(timestep_r, "timestep_r")
            omega = _drop_first_timestep_frame(omega, "omega")
            num_frames = input_num_frames - 1

        batch_size = hidden_states.size(0)
        condition_sequence_length = hidden_states.shape[1]
        align_kwargs = dict(
            batch_size=batch_size,
            hidden_sequence_length=condition_sequence_length,
            num_frames=num_frames,
            height=height,
            width=width,
            expand_singletons=token_dropout_ratio > 0,
        )
        timestep = self.align_time_condition_tokens(
            timestep, "timestep", **align_kwargs
        )
        timestep_r = self.align_time_condition_tokens(
            timestep_r, "timestep_r", **align_kwargs
        )
        omega = self.align_time_condition_tokens(
            omega, "omega", **align_kwargs
        )

        for condition_name, condition_value in (
            ("timestep", timestep),
            ("timestep_r", timestep_r),
            ("omega", omega),
        ):
            if condition_value is not None and condition_value.shape[1] not in (
                1, condition_sequence_length
            ):
                raise ValueError(f"{condition_name} was not aligned to the DiT sequence.")

        image_rotary_emb = (
            t2v_image_rotary_emb
            if t2v_image_rotary_emb is not None
            else self.rope(
                hidden_states,
                rope_interpolation_scale,
                num_frames,
                height,
                width,
                video_rope_coords=video_rope_coords,
            )
        )

        token_reserve_index = None

        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
        if video_attention_mask is not None and video_attention_mask.ndim == 2:
            video_attention_mask = (1 - video_attention_mask.to(hidden_states.dtype)) * -10000.0
            video_attention_mask = video_attention_mask.unsqueeze(0)

        batch_size = hidden_states.size(0)

        if getattr(self, "frame_concat_conditioning", False):
            assert cond_latents is not None and cond_mode_ids is not None, \
                "frame_concat_conditioning requires cond_latents and cond_mode_ids"
            assert token_dropout_ratio == 0, \
                "frame_concat_conditioning + token_dropout not supported yet"
            assert not (self.patch_size > 1 or self.patch_size_t > 1), \
                "frame_concat_conditioning assumes patch_size == 1"
            mode_emb = self.mode_embed(cond_mode_ids).to(hidden_states.dtype)
            hidden_states = torch.cat(
                [hidden_states, cond_latents.to(hidden_states.dtype), mode_emb], dim=-1
            )

        if self.patch_size > 1 or self.patch_size_t > 1:
            assert(token_dropout_ratio == 0)
            hidden_states = self.proj_in(hidden_states, num_frames, height, width)
        else:
            if token_dropout_ratio > 0:
                if num_frames_wo_dropout > 0:
                    hidden_states = rearrange(hidden_states, "bv (t hw) c -> bv t hw c", t=num_frames)
                    hidden_states_wo_dropout = hidden_states[:,:num_frames_wo_dropout].flatten(1,2)
                    hidden_states = hidden_states[:,num_frames_wo_dropout:].flatten(1,2)

                token_reserve_num = int(hidden_states.shape[1] * (1.0-token_dropout_ratio))
                token_reserve_index = torch.stack([torch.randperm(hidden_states.shape[1])[:token_reserve_num] for _ in range(hidden_states.shape[0])], dim=0).unsqueeze(dim=2).to(device=hidden_states.device)
                hidden_states = hidden_states.gather(dim=1, index=token_reserve_index.repeat(1,1,hidden_states.shape[-1]))

                if num_frames_wo_dropout > 0:
                    hidden_states = torch.cat((hidden_states_wo_dropout, hidden_states), dim=1)

                    with torch.no_grad():
                        image_rotary_emb_cos = rearrange(image_rotary_emb[0], "bv (t hw) c -> bv t hw c", t=num_frames)
                        image_rotary_emb_cos_wo_dropout = image_rotary_emb_cos[:,:num_frames_wo_dropout].flatten(1,2)
                        image_rotary_emb_cos = image_rotary_emb_cos[:,num_frames_wo_dropout:].flatten(1,2)
                        image_rotary_emb_cos = torch.cat((
                            image_rotary_emb_cos_wo_dropout,
                            image_rotary_emb_cos.gather(dim=1, index=token_reserve_index.repeat(1,1,image_rotary_emb_cos.shape[-1])),
                        ), dim=1)
                        image_rotary_emb_sin = rearrange(image_rotary_emb[1], "bv (t hw) c -> bv t hw c", t=num_frames)
                        image_rotary_emb_sin_wo_dropout = image_rotary_emb_sin[:,:num_frames_wo_dropout].flatten(1,2)
                        image_rotary_emb_sin = image_rotary_emb_sin[:,num_frames_wo_dropout:].flatten(1,2)
                        image_rotary_emb_sin = torch.cat((
                            image_rotary_emb_sin_wo_dropout,
                            image_rotary_emb_sin.gather(dim=1, index=token_reserve_index.repeat(1,1,image_rotary_emb_sin.shape[-1])),
                        ), dim=1)
                        image_rotary_emb = (image_rotary_emb_cos, image_rotary_emb_sin)

                        timestep = rearrange(timestep, "bv (t hw) -> bv t hw", t=num_frames)
                        timestep_wo_dropout = timestep[:,:num_frames_wo_dropout].flatten(1,2)
                        timestep = timestep[:,num_frames_wo_dropout:].flatten(1,2)
                        timestep = torch.cat((
                            timestep_wo_dropout,
                            timestep.gather(dim=1, index=token_reserve_index[:,:,0]),
                        ), dim=1)

                        if self.use_t_and_r_embedding and timestep_r is not None:
                            timestep_r = rearrange(timestep_r, "bv (t hw) -> bv t hw", t=num_frames)
                            timestep_r_wo_dropout = timestep_r[:,:num_frames_wo_dropout].flatten(1,2)
                            timestep_r = timestep_r[:,num_frames_wo_dropout:].flatten(1,2)
                            timestep_r = torch.cat((
                                timestep_r_wo_dropout,
                                timestep_r.gather(dim=1, index=token_reserve_index[:,:,0]),
                            ), dim=1)
                        if omega is not None:
                            omega = rearrange(omega, "bv (t hw) -> bv t hw", t=num_frames)
                            omega_wo_dropout = omega[:,:num_frames_wo_dropout].flatten(1,2)
                            omega = omega[:,num_frames_wo_dropout:].flatten(1,2)
                            omega = torch.cat((
                                omega_wo_dropout,
                                omega.gather(dim=1, index=token_reserve_index[:,:,0]),
                            ), dim=1)


                        if video_attention_mask is not None:
                            video_attention_mask = rearrange(video_attention_mask, "b (t hw) l_k -> b t hw l_k", t=num_frames)
                            video_attention_mask = torch.cat((
                                video_attention_mask[:, :num_frames_wo_dropout].flatten(1,2),
                                video_attention_mask[:, num_frames_wo_dropout:].flatten(1,2).gather(dim=1, index=token_reserve_index.repeat(1,1,video_attention_mask.shape[-1])),
                            ), dim=1).permute(0,2,1)
                            video_attention_mask = rearrange(video_attention_mask, "b (t hw) l_q -> b t hw l_q", t=num_frames)
                            video_attention_mask = torch.cat((
                                video_attention_mask[:, :num_frames_wo_dropout].flatten(1,2),
                                video_attention_mask[:, num_frames_wo_dropout:].flatten(1,2).gather(dim=1, index=token_reserve_index.repeat(1,1,video_attention_mask.shape[-1])),
                            ), dim=1).permute(0,2,1)
                else:
                    with torch.no_grad():
                        image_rotary_emb_cos = image_rotary_emb[0].gather(dim=1, index=token_reserve_index.repeat(1,1,target.shape[-1]))
                        image_rotary_emb_sin = image_rotary_emb[1].gather(dim=1, index=token_reserve_index.repeat(1,1,target.shape[-1]))
                        image_rotary_emb = (image_rotary_emb_cos, image_rotary_emb_sin)

                        timestep = timestep.gather(dim=1, index=token_reserve_index[:,:,0])
                        if self.use_t_and_r_embedding and timestep_r is not None:
                            timestep_r = timestep_r.gather(dim=1, index=token_reserve_index[:,:,0])
                        if omega is not None:
                            omega = omega.gather(dim=1, index=token_reserve_index[:,:,0])
                        if video_attention_mask is not None:
                            video_attention_mask = video_attention_mask.gather(dim=1, index=token_reserve_index.repeat(1,1,video_attention_mask.shape[-1])).permute(0,2,1)
                            video_attention_mask = video_attention_mask.gather(dim=1, index=token_reserve_index.repeat(1,1,video_attention_mask.shape[-1])).permute(0,2,1)

            hidden_states = self.proj_in(hidden_states)

        temb, embedded_timestep = self.compute_time_conditioning(
            timestep,
            timestep_r=timestep_r,
            omega=omega,
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )


        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        if self.use_view_embed:
            if view_ids is None:
                view_ids = torch.arange(n_view, device=hidden_states.device, dtype=torch.long).unsqueeze(0)
            else:
                view_ids = view_ids.to(device=hidden_states.device, dtype=torch.long)
            assert view_ids.ndim == 2 and view_ids.shape[1] == n_view, f"Expected view_ids shape [B, {n_view}], got {tuple(view_ids.shape)}"
            if view_ids.shape[0] == 1 and (batch_size // n_view) > 1:
                view_ids = view_ids.expand(batch_size // n_view, -1)
            embedded_view = self.view_embed[view_ids]
            embedded_view = rearrange(embedded_view, 'b v c -> (b v) c').unsqueeze(1)
            vemb = self.view_ada(embedded_view)
            temb = temb + vemb
            embedded_timestep = embedded_timestep + embedded_view

        if self.vlm_interface is not None:
            if precomputed_vlm_hidden_states is not None:
                _fuser_mask = vlm_attention_mask if getattr(self, "vlm_text_tokens_only", False) else None
                encoder_hidden_states = self.fuser(precomputed_vlm_hidden_states, attention_mask=_fuser_mask)
                null_text_mask = goal_mode_mask
                if cfg_unconditional_mask is not None:
                    cfg_unconditional_mask = cfg_unconditional_mask.to(
                        device=encoder_hidden_states.device,
                        dtype=torch.bool,
                    )
                    null_text_mask = (
                        cfg_unconditional_mask
                        if null_text_mask is None
                        else null_text_mask.to(
                            device=encoder_hidden_states.device,
                            dtype=torch.bool,
                        ) | cfg_unconditional_mask
                    )
                if null_text_mask is not None:
                    encoder_hidden_states, vlm_attention_mask = self._apply_goal_mode_null_text(
                        encoder_hidden_states, vlm_attention_mask, null_text_mask
                    )
            else:
                if goal_mode_mask is not None or cfg_unconditional_mask is not None:
                    raise NotImplementedError(
                        "null-text masks require precomputed_vlm_hidden_states; "
                        "the live vlm_inputs path is not wired for row replacement."
                    )
                encoder_hidden_states = self.vlm_interface(
                    **vlm_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                ).hidden_states

                _text_mask = None
                if getattr(self, "vlm_text_tokens_only", False):
                    encoder_hidden_states, _text_mask = slice_vlm_text_part(
                        encoder_hidden_states,
                        vlm_inputs.get("attention_mask", None),
                        vlm_inputs["input_ids"],
                        self.vlm_interface.model.config.vision_end_token_id,
                    )

                if batch_size // encoder_hidden_states[0].shape[0] != 1:
                    assert(batch_size // encoder_hidden_states[0].shape[0] == n_view)
                    encoder_hidden_states = [_encoder_hidden_states for _encoder_hidden_states in encoder_hidden_states]
                else:
                    encoder_hidden_states = [_encoder_hidden_states.view(batch_size // n_view, -1, _encoder_hidden_states.size(-1)) for _encoder_hidden_states in encoder_hidden_states]
                    if _text_mask is not None:
                        _text_mask = _text_mask.reshape(batch_size // n_view, -1)
                if _text_mask is not None:
                    vlm_attention_mask = _text_mask

                _fuser_mask = vlm_attention_mask if getattr(self, "vlm_text_tokens_only", False) else None
                encoder_hidden_states = self.fuser(encoder_hidden_states, attention_mask=_fuser_mask)

            if vlm_attention_mask is not None:
                if vlm_attention_mask.shape[-1] != encoder_hidden_states.shape[1]:
                    raise ValueError(
                        f"vlm_attention_mask token dim {tuple(vlm_attention_mask.shape)} != fused VLM "
                        f"tokens {encoder_hidden_states.shape[1]}; the mask reshape must mirror the "
                        f"hidden-state reshape in compute_vlm_hidden_states."
                    )
                _vm = (1 - vlm_attention_mask.to(hidden_states.dtype)) * -10000.0
                encoder_attention_mask = _vm.unsqueeze(1)
            else:
                encoder_attention_mask = None
        else:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(hidden_states.size(0) // n_view, -1, hidden_states.size(-1))

        shared_depth = len(self.transformer_blocks) - self.v_head_depth

        def _run_block(block, block_idx, hs):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

                hs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hs,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    encoder_attention_mask,
                    n_view,
                    block_idx%3==0,
                    video_attention_mask,
                    **ckpt_kwargs,
                )
            else:
                hs = block(
                    hidden_states=hs,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                    n_view=n_view,
                    cross_view_attn=block_idx % 3 == 0,
                    self_attention_mask=video_attention_mask,
                )
            return hs

        for block_idx in range(shared_depth):
            hidden_states = _run_block(self.transformer_blocks[block_idx], block_idx, hidden_states)

        hidden_states_shared = hidden_states if compute_v_head else None

        for i, block in enumerate(self.transformer_blocks[shared_depth:]):
            block_idx = shared_depth + i
            hidden_states = _run_block(block, block_idx, hidden_states)

        hidden_states_v = None
        if compute_v_head:
            hidden_states_v = hidden_states_shared
            for i, block in enumerate(self.v_head_blocks):
                block_idx = shared_depth + i
                hidden_states_v = _run_block(block, block_idx, hidden_states_v)

        final_output = {}

        def _head_proj(hs, norm, sst, proj):
            ssv = sst[None, None] + embedded_timestep[:, :, None]
            shift, scale = ssv[:, :, 0], ssv[:, :, 1]
            hs = norm(hs) * (1 + scale) + shift
            _d = fwAD.unpack_dual(hs)
            if _d.tangent is not None:
                hs = fwAD.make_dual(_d.primal, _d.tangent.to(_d.primal.dtype))
            out = proj(hs)
            if self.patch_size > 1 or self.patch_size_t > 1:
                out = out.unflatten(2, (self.patch_size, self.patch_size, self.patch_size_t, -1))
                out = out.unflatten(1, (num_frames//self.patch_size_t, height//self.patch_size, width//self.patch_size))
                out = out.permute(0, 7, 1, 6, 2, 4, 3, 5).flatten(2, 6)
            return out

        def _restore_t2v_prefix(value):
            if not self.t2v_mode:
                return value
            zero_prefix = value[:, :t2v_prefix_tokens] * 0.0
            return torch.cat((zero_prefix, value), dim=1)

        output = _restore_t2v_prefix(
            _head_proj(hidden_states, self.norm_out, self.scale_shift_table, self.proj_out)
        )
        final_output['video'] = output

        if compute_v_head:
            final_output['v_video'] = _restore_t2v_prefix(
                _head_proj(
                    hidden_states_v, self.norm_out_v, self.scale_shift_table_v, self.proj_out_v
                )
            )
        final_output['token_reserve_index'] = token_reserve_index
        final_output['num_frames_wo_dropout'] = num_frames_wo_dropout

        if not return_dict:
            return (final_output,)

        return Transformer2DModelOutput(sample=final_output)


def apply_rotary_emb(x, freqs):
    cos, sin = freqs
    batch_size = x.shape[0]
    if cos.shape[0] == 1 and batch_size > 1:
        cos = cos.repeat(batch_size, 1, 1)
    if sin.shape[0] == 1 and batch_size > 1:
        sin = sin.repeat(batch_size, 1, 1)

    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out
