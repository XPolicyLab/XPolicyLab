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
from diffusers.models.embeddings import PixArtAlphaTextProjection
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


# A Transformer model for video-like data used in [LTX](https://huggingface.co/Lightricks/LTX-Video).
@maybe_allow_in_graph
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
        use_bbox_condition: bool = False,
        bbox_add_image_description: bool = True,
        vlm_text_tokens_only: bool = False,
        vlm_head_view_only: bool = False,
        vlm_head_view_index: int = 0,
        v_head_depth: int = 4,
        use_t_and_r_embedding: bool = False,
        time_embed_zero_init: bool = False,
        frame_concat_conditioning: bool = False,
        frame_concat_mode_dim: int = 32,
        t2v_mode: bool = False,
        block_memory_to_future_attention: bool = False,
        goal_mode_null_text_tokens: int = 0,
        spatial_rope_grid_mode: str = "legacy",
        base_num_frames: int = 9,
        **kwargs,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        inner_dim = num_attention_heads * attention_head_dim

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.t2v_mode = bool(t2v_mode)
        self.block_memory_to_future_attention = bool(
            block_memory_to_future_attention
        )
        if self.t2v_mode and (self.patch_size != 1 or self.patch_size_t != 1):
            raise ValueError(
                "t2v_mode currently requires patch_size == patch_size_t == 1, "
                f"got patch_size={self.patch_size!r}, patch_size_t={self.patch_size_t!r}."
            )
        if self.block_memory_to_future_attention and self.t2v_mode:
            raise ValueError(
                "block_memory_to_future_attention is incompatible with t2v_mode."
            )
        if self.block_memory_to_future_attention and (
            self.patch_size != 1 or self.patch_size_t != 1
        ):
            raise ValueError(
                "block_memory_to_future_attention requires patch_size == "
                "patch_size_t == 1."
            )

        proj_in_channels = in_channels

        if self.patch_size > 1 or self.patch_size_t > 1:
            self.proj_in = PatchEmbed(
                proj_in_channels, inner_dim, patch_size, bias=False
            )
        else:
            self.proj_in = nn.Linear(proj_in_channels, inner_dim)

        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.time_embed = AdaLayerNormSingle(inner_dim, use_additional_conditions=False)

        self.use_t_and_r_embedding = use_t_and_r_embedding
        if use_t_and_r_embedding:
            self.time_embed_r = AdaLayerNormSingle(inner_dim, use_additional_conditions=False)

        self.time_embed_zero_init = time_embed_zero_init
        if time_embed_zero_init:
            self._zero_init_time_embedder(self.time_embed)
            if use_t_and_r_embedding:
                self._zero_init_time_embedder(self.time_embed_r)

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

        self.use_bbox_condition = bool(use_bbox_condition)
        self.bbox_add_image_description = bool(bbox_add_image_description)
        if self.use_bbox_condition and vlm_fixed_sequence_length != 384:
            raise ValueError("bbox evaluation requires vlm_fixed_sequence_length=384")
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

            vlm_hidden_size = int(self.vlm_interface.model.config.hidden_size)
            if vlm_hidden_size != cross_attention_dim:
                raise ValueError(
                    f"VLM text hidden size {vlm_hidden_size} != cross_attention_dim "
                    f"{cross_attention_dim}: set model.video.cross_attention_dim to the "
                    f"VLM width of {vlm_model_id!r}."
                )
            self.fuser = SimpleGateFusion(
                hidden_size=vlm_hidden_size,
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
                torch.randn(1, self.goal_mode_null_text_tokens, cross_attention_dim) * 0.02
            )

        self.rope = LTXVideoRotaryPosEmbed(
            dim=inner_dim,
            base_num_frames=base_num_frames,
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

    def make_view_emb(self, max_view, inner_dim, ):
        self.view_embed = nn.Parameter(torch.randn(max_view, inner_dim))
        if hasattr(self, "view_embed"):
            dtype = self.view_embed.dtype
            device = self.view_embed.device
            self.view_embed = self.view_embed.to(device=device, dtype=dtype)

    def _build_memory_attention_mask(
        self,
        sequence_length: int,
        num_mem_frames: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.block_memory_to_future_attention:
            return None

        memory_tokens = int(num_mem_frames) * int(height) * int(width)
        sequence_length = int(sequence_length)
        if memory_tokens <= 0 or memory_tokens >= sequence_length:
            raise ValueError(
                "block_memory_to_future_attention requires both memory and "
                f"future tokens, got {memory_tokens=} and {sequence_length=}."
            )
        mask = torch.ones(
            (sequence_length, sequence_length), device=device, dtype=dtype
        )
        mask[:memory_tokens, memory_tokens:] = 0
        return mask

    def _stage1_run_stack(self, blocks, first_index, hidden_states, context):
        def run_block(block, block_idx, hidden_states):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    return lambda *inputs: module(*inputs)

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                return torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    context["encoder_hidden_states"],
                    context["temb"],
                    context["image_rotary_emb"],
                    context["encoder_attention_mask"],
                    context["n_view"],
                    block_idx % 3 == 0,
                    context["video_attention_mask"],
                    **ckpt_kwargs,
                )
            return block(
                hidden_states=hidden_states,
                encoder_hidden_states=context["encoder_hidden_states"],
                temb=context["temb"],
                image_rotary_emb=context["image_rotary_emb"],
                encoder_attention_mask=context["encoder_attention_mask"],
                n_view=context["n_view"],
                cross_view_attn=block_idx % 3 == 0,
                self_attention_mask=context["video_attention_mask"],
            )

        for offset, block in enumerate(blocks):
            hidden_states = run_block(block, first_index + offset, hidden_states)
        return hidden_states

    def _stage1_finish_heads(self, context, hidden_states_u, hidden_states_v):
        def head_proj(hidden_states, norm, scale_shift_table, proj):
            embedded_timestep = context["embedded_timestep"]
            if hidden_states.shape[0] != embedded_timestep.shape[0]:
                embedded_timestep = embedded_timestep[:hidden_states.shape[0]]
            scale_shift = scale_shift_table[None, None] + embedded_timestep[:, :, None]
            shift, scale = scale_shift[:, :, 0], scale_shift[:, :, 1]
            hidden_states = norm(hidden_states) * (1 + scale) + shift
            dual = fwAD.unpack_dual(hidden_states)
            if dual.tangent is not None:
                hidden_states = fwAD.make_dual(
                    dual.primal, dual.tangent.to(dual.primal.dtype)
                )
            output = proj(hidden_states)
            if self.patch_size > 1 or self.patch_size_t > 1:
                output = output.unflatten(
                    2, (self.patch_size, self.patch_size, self.patch_size_t, -1)
                )
                output = output.unflatten(
                    1,
                    (
                        context["num_frames"] // self.patch_size_t,
                        context["height"] // self.patch_size,
                        context["width"] // self.patch_size,
                    ),
                )
                output = output.permute(0, 7, 1, 6, 2, 4, 3, 5).flatten(2, 6)
            return output

        def restore_t2v_prefix(value):
            if not self.t2v_mode:
                return value
            zero_prefix = value[:, :context["t2v_prefix_tokens"]] * 0.0
            return torch.cat((zero_prefix, value), dim=1)

        output = {
            "video": restore_t2v_prefix(
                head_proj(
                    hidden_states_u,
                    self.norm_out,
                    self.scale_shift_table,
                    self.proj_out,
                )
            ),
            "token_reserve_index": context["token_reserve_index"],
            "num_frames_wo_dropout": context["num_frames_wo_dropout"],
        }
        if context["compute_v_head"]:
            output["v_video"] = restore_t2v_prefix(
                head_proj(
                    hidden_states_v,
                    self.norm_out_v,
                    self.scale_shift_table_v,
                    self.proj_out_v,
                )
            )
        if not context["return_dict"]:
            return (output,)
        return Transformer2DModelOutput(sample=output)


    def _stage1_finish_context(self, context, *, run_shared=True, shared_only=False):
        shared_depth = len(self.transformer_blocks) - self.v_head_depth
        if run_shared:
            context["hidden_states"] = self._stage1_run_stack(
                self.transformer_blocks[:shared_depth],
                0,
                context["hidden_states"],
                context,
            )
        if shared_only:
            return context

        hidden_states = context["hidden_states"]
        batch_size = hidden_states.shape[0]
        u_context = context
        if context["u_tail_rows"] != batch_size:
            u_context = dict(context)
            u_context["encoder_hidden_states"] = context["encoder_hidden_states"][:context["u_tail_batch_size"]]
            u_context["temb"] = context["temb"][:context["u_tail_rows"]]
            u_context["image_rotary_emb"] = tuple(
                value[:context["u_tail_rows"]] for value in context["image_rotary_emb"]
            )
            if context["encoder_attention_mask"] is not None:
                u_context["encoder_attention_mask"] = context["encoder_attention_mask"][:context["u_tail_batch_size"]]

        hidden_states_u = hidden_states
        if context["u_tail_rows"] != batch_size:
            hidden_states_u = hidden_states[:context["u_tail_rows"]]
        if context["u_tail_rows"] > 0:
            hidden_states_u = self._stage1_run_stack(
                self.transformer_blocks[shared_depth:],
                shared_depth,
                hidden_states_u,
                u_context,
            )

        hidden_states_v = None
        if context["compute_v_head"]:
            hidden_states_v = self._stage1_run_stack(
                self.v_head_blocks,
                shared_depth,
                hidden_states,
                context,
            )

        return self._stage1_finish_heads(context, hidden_states_u, hidden_states_v)


    @staticmethod
    def _stage1_merge_contexts(h0_context, mf_context):
        n_view = h0_context["n_view"]
        h0_rows, mf_rows = h0_context["hidden_states"].shape[0], mf_context["hidden_states"].shape[0]
        if (
            n_view not in (1, 2, 3)
            or n_view != mf_context["n_view"]
            or h0_rows <= 0
            or mf_rows <= 0
            or h0_rows % n_view
            or mf_rows % n_view
        ):
            raise ValueError(
                f"Stage-1 ordinary pair requires complete 1/2/3-view groups: H={h0_rows}, M={mf_rows}, V={n_view}."
            )
        for key in ("num_frames", "height", "width", "t2v_prefix_tokens"):
            if h0_context[key] != mf_context[key]:
                raise ValueError(f"Stage-1 ordinary pair has incompatible {key}.")
        if h0_context["video_attention_mask"] is not None or mf_context["video_attention_mask"] is not None:
            raise ValueError("Stage-1 ordinary pair currently requires the dense unmasked video path.")

        video_pairs = (
            (h0_context["hidden_states"], mf_context["hidden_states"]),
            (h0_context["temb"], mf_context["temb"]),
            *zip(h0_context["image_rotary_emb"], mf_context["image_rotary_emb"]),
        )
        if (
            len(h0_context["image_rotary_emb"]) != len(mf_context["image_rotary_emb"])
            or any(
                left.shape[0] != h0_rows
                or right.shape[0] != mf_rows
                or left.shape[1:] != right.shape[1:]
                for left, right in video_pairs
            )
        ):
            raise ValueError("Stage-1 ordinary pair has incompatible video-shaped inputs.")
        h0_tasks, mf_tasks = h0_rows // n_view, mf_rows // n_view
        h0_encoder, mf_encoder = h0_context["encoder_hidden_states"], mf_context["encoder_hidden_states"]
        if (
            h0_encoder.shape[0] != h0_tasks
            or mf_encoder.shape[0] != mf_tasks
            or h0_encoder.shape[1:] != mf_encoder.shape[1:]
            or (h0_context["encoder_attention_mask"] is None) != (mf_context["encoder_attention_mask"] is None)
        ):
            raise ValueError("Stage-1 ordinary pair has incompatible conditioning inputs.")
        if h0_context["encoder_attention_mask"] is None:
            encoder_attention_mask = None
        else:
            h0_mask, mf_mask = h0_context["encoder_attention_mask"], mf_context["encoder_attention_mask"]
            if (
                h0_mask.shape[0] != h0_tasks
                or mf_mask.shape[0] != mf_tasks
                or h0_mask.shape[1:] != mf_mask.shape[1:]
            ):
                raise ValueError("Stage-1 ordinary pair has incompatible encoder masks.")
            encoder_attention_mask = torch.cat((h0_mask, mf_mask), dim=0)

        merged = dict(h0_context)
        merged.update(
            hidden_states=torch.cat((h0_context["hidden_states"], mf_context["hidden_states"]), dim=0),
            encoder_hidden_states=torch.cat((h0_encoder, mf_encoder), dim=0),
            temb=torch.cat((h0_context["temb"], mf_context["temb"]), dim=0),
            image_rotary_emb=tuple(
                torch.cat((left, right), dim=0)
                for left, right in zip(h0_context["image_rotary_emb"], mf_context["image_rotary_emb"])
            ),
            encoder_attention_mask=encoder_attention_mask,
            compute_v_head=False,
            return_dict=False,
            u_tail_batch_size=h0_tasks + mf_tasks,
            u_tail_rows=h0_rows + mf_rows,
        )
        return merged


    def forward(
        self,
        hidden_states: torch.Tensor = None,
        encoder_hidden_states = None,
        timestep: torch.LongTensor = None,
        timestep_r: torch.Tensor = None,
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
        num_frames_wo_dropout: int = 0,
        num_mem_frames: int = 0,
        precomputed_vlm_hidden_states: list = None,
        vlm_attention_mask: torch.Tensor = None,
        goal_mode_mask: torch.Tensor = None,
        compute_v_head: bool = True,
        u_tail_batch_size: int | None = None,
        cond_latents: torch.Tensor = None,
        cond_mode_ids: torch.Tensor = None,
        _stage1_prepare_only: bool = False,
        _stage1_ordinary_pair: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        t2v_prefix_tokens = 0
        t2v_image_rotary_emb = None
        if self.t2v_mode:
            if num_frames is None or height is None or width is None:
                raise ValueError("t2v_mode requires num_frames, height and width.")
            if int(num_frames) < 2:
                raise ValueError(
                    f"t2v_mode requires memory + future frames, got num_frames={num_frames}."
                )
            if (
                cond_latents is not None
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
            num_frames = input_num_frames - 1

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

        if self.block_memory_to_future_attention:
            if video_attention_mask is not None:
                raise ValueError(
                    "An explicit video_attention_mask cannot be combined with "
                    "block_memory_to_future_attention."
                )
            video_attention_mask = self._build_memory_attention_mask(
                hidden_states.shape[1],
                num_mem_frames,
                height,
                width,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        token_reserve_index = None

        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
        if video_attention_mask is not None and video_attention_mask.ndim == 2:
            video_attention_mask = (1 - video_attention_mask.to(hidden_states.dtype)) * -10000.0
            video_attention_mask = video_attention_mask.unsqueeze(0)

        batch_size = hidden_states.size(0)
        if u_tail_batch_size is None:
            u_tail_batch_size = batch_size // n_view
        u_tail_rows = u_tail_batch_size * n_view

        if self.patch_size > 1 or self.patch_size_t > 1:
            hidden_states = self.proj_in(hidden_states, num_frames, height, width)
        else:
            hidden_states = self.proj_in(hidden_states)

        temb, embedded_timestep = self.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        if self.use_t_and_r_embedding:
            assert timestep_r is not None, (
                "use_t_and_r_embedding=True requires `timestep_r` in the forward call."
            )
            temb_r, embedded_timestep_r = self.time_embed_r(
                timestep_r.flatten(),
                batch_size=batch_size,
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb + temb_r
            embedded_timestep = embedded_timestep + embedded_timestep_r

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
                if goal_mode_mask is not None:
                    encoder_hidden_states, vlm_attention_mask = self._apply_goal_mode_null_text(
                        encoder_hidden_states, vlm_attention_mask, goal_mode_mask
                    )
            else:
                if goal_mode_mask is not None:
                    raise NotImplementedError(
                        "goal_mode_mask requires precomputed_vlm_hidden_states; "
                        "the live vlm_inputs path is not wired for goal mode."
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

        context = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "temb": temb,
            "embedded_timestep": embedded_timestep,
            "image_rotary_emb": image_rotary_emb,
            "encoder_attention_mask": encoder_attention_mask,
            "video_attention_mask": video_attention_mask,
            "n_view": n_view,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "compute_v_head": compute_v_head,
            "return_dict": return_dict,
            "num_frames_wo_dropout": num_frames_wo_dropout,
            "token_reserve_index": token_reserve_index,
            "t2v_prefix_tokens": t2v_prefix_tokens,
            "u_tail_batch_size": u_tail_batch_size,
            "u_tail_rows": u_tail_rows,
        }
        if _stage1_prepare_only:
            if _stage1_ordinary_pair is not None:
                raise ValueError("A prepared Stage-1 context cannot contain another ordinary pair.")
            return (context,)
        if _stage1_ordinary_pair is None:
            return self._stage1_finish_context(context)

        mf_context = self.forward(
            **_stage1_ordinary_pair,
            _stage1_prepare_only=True,
        )[0]
        merged_context = self._stage1_finish_context(
            self._stage1_merge_contexts(context, mf_context), shared_only=True
        )
        h0_rows = context["hidden_states"].shape[0]
        context["hidden_states"] = merged_context["hidden_states"][:h0_rows]
        mf_context["hidden_states"] = merged_context["hidden_states"][h0_rows:]
        h0_output = self._stage1_finish_context(context, run_shared=False)
        mf_output = self._stage1_finish_context(mf_context, run_shared=False)
        h0_sample = h0_output.sample if context["return_dict"] else h0_output[0]
        mf_sample = mf_output.sample if mf_context["return_dict"] else mf_output[0]
        h0_sample["_stage1_ordinary_pair_video"] = mf_sample["video"]
        return h0_output


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
