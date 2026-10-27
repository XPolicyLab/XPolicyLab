# Parts of this file are copied from HuggingFace diffusers (Apache-2.0), marked "Copied from" below.

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers, is_torch_version
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    SanaLinearAttnProcessor2_0,
)
from diffusers.models.embeddings import PatchEmbed, PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm
import math
from einops import rearrange
from diffusers.models.attention import FeedForward
from diffusers import AutoencoderDC
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.autoencoders.vae import DecoderOutput, EncoderOutput
from policy_core.utils.WAM_policy.profile_utils import (
    StageProfiler,
    inference_profile_active,
)


import torch
import os
import matplotlib.pyplot as plt


def save_attn_heatmaps(query, key, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    query = query.float()
    key = key.float()

    B, H, L1, C = query.shape
    L2 = key.shape[2]

    attn_scores = torch.matmul(query, key.transpose(-1, -2)) / (C ** 0.5)
    attn_probs = torch.softmax(attn_scores, dim=-1)

    for b in range(1):
        for h in range(H):
            attn_map = attn_probs[b, h].detach().cpu().numpy()

            figsize = (max(6, L2 / 20), max(4, L1 / 20))
            plt.figure(figsize=figsize)
            ax = sns.heatmap(attn_map,
                             cmap="viridis",
                             xticklabels=10,
                             yticklabels=10,
                             cbar=True)

            ax.set_title(f"Attention Map (Batch {b}, Head {h})")
            ax.set_xlabel("Key Position (L2)")
            ax.set_ylabel("Query Position (L1)")

            for frac in [1/3, 2/3]:
                xpos = int(L2 * frac)
                ax.axvline(x=xpos, color='red', linestyle='--', linewidth=1)

            plt.tight_layout()

            save_path = os.path.join(save_dir, f"attn_b{b}_h{h}.png")
            plt.savefig(save_path)
            plt.close()

class AutoencoderDCos(AutoencoderDC):
    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True, n_chunk=4) -> Union[EncoderOutput, Tuple[torch.Tensor]]:
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.chunk(n_chunk)]
            encoded = torch.cat(encoded_slices)
        else:
            encoded = self._encode(x)

        if not return_dict:
            return (encoded,)
        return EncoderOutput(latent=encoded)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True, n_chunk=4) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        if self.use_slicing and z.size(0) > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.chunk(n_chunk)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z)

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

logger = logging.get_logger(__name__)

def _rank0_print(message: str) -> None:
    if (
        (not torch.distributed.is_available())
        or (not torch.distributed.is_initialized())
        or torch.distributed.get_rank() == 0
    ):
        print(message, flush=True)

def apply_rotary_emb(x, freqs):
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out


class LTXVideoAttentionProcessor2_0:
    def __init__(
        self,
        is_causal: bool = False,
        enable_vis: bool = False,
        vis_save_dir: str = "outputs/attn_vis",
        layer_idx: int = -1,
    ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Requires PyTorch 2.0")

        self.is_causal = is_causal
        self.enable_vis = enable_vis
        self.vis_save_dir = vis_save_dir
        self.layer_idx = layer_idx

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        chunkwise_causal: bool = False,
        visible_chunk: int = 32,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if image_rotary_emb is not None and attention_mask is not None:
            if attention_mask.shape[0] != batch_size:
                attention_mask = attention_mask.repeat(batch_size, 1, 1)
            sequence_length = attention_mask.shape[-1]

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.norm_q(attn.to_q(hidden_states))
        key = attn.norm_k(attn.to_k(encoder_hidden_states))
        value = attn.to_v(encoder_hidden_states)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if chunkwise_causal:
            attention_mask = torch.zeros(1, 1, query.shape[-2], key.shape[-2], dtype=torch.bool, device=query.device)
            n_chunk = key.shape[-2] // visible_chunk
            visible_chunk_q = query.shape[-2] // n_chunk
            for i_chunk in range(n_chunk):
                attention_mask[:, :, i_chunk*visible_chunk_q:(i_chunk+1)*visible_chunk_q, i_chunk*visible_chunk:(i_chunk+1)*visible_chunk] = True

        if self.enable_vis and not torch.is_grad_enabled():
            is_cross = encoder_hidden_states is not hidden_states
            cur_save_dir = os.path.join(
                self.vis_save_dir,
                f"layer_{self.layer_idx}",
                "cross" if is_cross else "self",
            )
            save_attn_heatmaps(query, key, cur_save_dir)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=self.is_causal
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class CustomTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class HighCapacityAttentionStack(nn.Module):
    def __init__(self, dim_in=32, dim_hidden=1024, num_heads=8, depth=4):
        super().__init__()
        self.input_proj = nn.Linear(dim_in, dim_hidden)
        self.blocks = nn.Sequential(*[
            CustomTransformerBlock(dim=dim_hidden, num_heads=num_heads) for _ in range(depth)
        ])
        self.output_norm = nn.LayerNorm(dim_hidden)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.output_norm(x)
        return x


# Transformer block used in [LTX](https://huggingface.co/Lightricks/LTX-Video).
@maybe_allow_in_graph
class ActionTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        num_attention_heads: int = 16,
        attention_head_dim: int = 32,
        cross_attention_dim: int = 2048,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        chunkwise_causal: bool = False,
        visible_chunk: int = 32,
        is_causal: bool = False,
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
            processor=LTXVideoAttentionProcessor2_0(is_causal=is_causal),
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

        self.chunkwise_causal = chunkwise_causal
        self.visible_chunk = visible_chunk

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: torch.Tensor = None,
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
            image_rotary_emb=rotary_emb,
            chunkwise_causal=self.chunkwise_causal,
            visible_chunk=self.visible_chunk,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa


        attn_hidden_states = self.attn2(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=None,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = hidden_states + attn_hidden_states

        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states


class ActionRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        base_seq_length: int = 57,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.base_seq_length = base_seq_length
        self.theta = theta

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_length: int,
        interpolation_scale: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.size(0)

        grid = torch.arange(seq_length, dtype=torch.float32, device=hidden_states.device).unsqueeze(0).repeat(batch_size, 1)

        grid = grid * interpolation_scale / self.base_seq_length

        grid = grid.unsqueeze(-1)

        start = 1.0
        end = self.theta
        freqs = self.theta ** torch.linspace(
            math.log(start, self.theta),
            math.log(end, self.theta),
            self.dim // 2,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        freqs = freqs * math.pi / 2.0
        freqs = freqs * (grid * 2 - 1)

        cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

        if self.dim % 2 != 0:
            cos_padding = torch.ones_like(cos_freqs[:, :, : self.dim % 2])
            sin_padding = torch.zeros_like(sin_freqs[:, :, : self.dim % 2])
            cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
            sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        return cos_freqs, sin_freqs

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
    ) -> None:
        super().__init__()

        self.dim = dim
        self.base_num_frames = base_num_frames
        self.base_height = base_height
        self.base_width = base_width
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.theta = theta

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        is_sparse: bool,
        factor: int,
        rope_interpolation_scale: Optional[Tuple[torch.Tensor, float, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rope_interpolation_scale is None:
            rope_interpolation_scale = (
                1,
                1,
                1,
            )
        batch_size = hidden_states.size(0)
        factor = torch.tensor(factor,device=hidden_states.device,dtype=torch.float32)
        if is_sparse:
            grid_h = torch.arange(height, dtype=torch.float32, device=hidden_states.device)*factor+((factor-1)/2)
            grid_w = torch.arange(width, dtype=torch.float32, device=hidden_states.device)*factor+((factor-1)/2)
        else:
            grid_h = torch.arange(height, dtype=torch.float32, device=hidden_states.device)
            grid_w = torch.arange(width, dtype=torch.float32, device=hidden_states.device)
        grid_f = torch.arange(num_frames, dtype=torch.float32, device=hidden_states.device)
        grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
        grid = torch.stack(grid, dim=0)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

        if rope_interpolation_scale is not None:
            grid[:, 0:1] = grid[:, 0:1] * rope_interpolation_scale[0] * self.patch_size_t / self.base_num_frames
            grid[:, 1:2] = grid[:, 1:2] * rope_interpolation_scale[1] * self.patch_size / self.base_height
            grid[:, 2:3] = grid[:, 2:3] * rope_interpolation_scale[2] * self.patch_size / self.base_width

        grid = grid.flatten(2, 4).transpose(1, 2)

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

class ActionExpert(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):

    _supports_gradient_checkpointing = True
    _no_split_modules = ["SanaTransformerBlock", "PatchEmbed", "SanaModulatedNorm"]
    _skip_layerwise_casting_patterns = ["patch_embed", "norm"]

    @register_to_config
    def __init__(
        self,
        num_layers: int = 28,
        attention_bias: bool = False,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        qk_norm: Optional[str] = None,
        activation_fn: str = "gelu-approximate",
        attention_out_bias: bool = False,
        vision_in_dim: int = 32,
        max_action_length: int = 64,
        action_in_channels: int = 14,
        action_num_attention_heads: int = 16,
        action_attention_head_dim: int = 32,
        residual_scale: float = 0.0,
        use_state_direct: bool = True,
        state_dim: int = 30,
        chunkwise_causal: bool = False,
        visible_chunk: int = 32,
        is_causal: bool = False,
        n_view: int = 3,
        num_view_identities: int = 3,
        body_embedding_enabled: bool = False,
        num_body_embeddings: int = 14,
        body_embedding_dim: Optional[int] = None,
        body_embedding_zero_init: bool = True,
        use_complex_adapter: bool = False,
        adapter_depth: int = 4,
        latent_base_f: int = 9,
        latent_base_h: int = 4,
        latent_base_w: int = 6,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers

        self.gradient_checkpointing = False

        self.vision_rotary_pos_emb = LTXVideoRotaryPosEmbed(
            dim=vision_in_dim,
            base_num_frames=latent_base_f,
            base_height=latent_base_h,
            base_width=latent_base_w,
            patch_size=1,
            patch_size_t=1,
            theta=10000.0,

        )
        self.view_identities = nn.Embedding(num_embeddings=num_view_identities, embedding_dim=vision_in_dim)
        nn.init.trunc_normal_(self.view_identities.weight, std=0.02)
        self.action_inner_dim = action_num_attention_heads * action_attention_head_dim
        self.action_in_channels = action_in_channels

        self.body_embedding_enabled = bool(body_embedding_enabled)
        self.body_embedding_dim = int(body_embedding_dim or (6 * self.action_inner_dim))
        if self.body_embedding_enabled:
            if int(num_body_embeddings) <= 0:
                raise ValueError(f"num_body_embeddings must be positive, got {num_body_embeddings}")
            self.body_embedding = nn.Embedding(
                num_embeddings=int(num_body_embeddings),
                embedding_dim=self.body_embedding_dim,
            )
            if body_embedding_zero_init:
                nn.init.zeros_(self.body_embedding.weight)
            else:
                nn.init.trunc_normal_(self.body_embedding.weight, std=0.02)
            self.body_embedding_to_adaln = None
            if self.body_embedding_dim != 6 * self.action_inner_dim:
                self.body_embedding_to_adaln = nn.Linear(
                    self.body_embedding_dim, 6 * self.action_inner_dim, bias=False
                )
                nn.init.xavier_uniform_(self.body_embedding_to_adaln.weight)
            self._body_embedding_grad_logged = False

            def _log_first_body_embedding_grad(grad):
                if not self._body_embedding_grad_logged:
                    self._body_embedding_grad_logged = True
                    _rank0_print(
                        "### [body_embedding] first backward: "
                        f"grad_norm={grad.detach().float().norm().item():.6g} "
                        f"nonzero={int(torch.count_nonzero(grad).item())}/{grad.numel()}"
                    )
                return grad

            self.body_embedding.weight.register_hook(_log_first_body_embedding_grad)
        else:
            self.body_embedding = None
            self.body_embedding_to_adaln = None


        self.action_proj_in = nn.Linear(action_in_channels, self.action_inner_dim)
        self.action_scale_shift_table = nn.Parameter(torch.randn(2, self.action_inner_dim) / self.action_inner_dim**0.5)
        self.action_time_embed = AdaLayerNormSingle(self.action_inner_dim, use_additional_conditions=False)


        self.action_rope = ActionRotaryPosEmbed(
            dim=self.action_inner_dim,
            base_seq_length=max_action_length,
            theta=10000.0,
        )

        self.use_complex_adapter = use_complex_adapter
        if self.use_complex_adapter:
            self.cross_attention_dim = self.action_inner_dim
            self.vision_feature_adapter = HighCapacityAttentionStack(vision_in_dim, self.action_inner_dim, action_num_attention_heads, adapter_depth)
        else:
            self.cross_attention_dim = vision_in_dim


        self.action_blocks = nn.ModuleList(
            [
                ActionTransformerBlock(
                    dim=self.action_inner_dim,
                    num_attention_heads=action_num_attention_heads,
                    attention_head_dim=action_attention_head_dim,
                    cross_attention_dim=self.cross_attention_dim,
                    qk_norm=qk_norm,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    attention_out_bias=attention_out_bias,
                    eps=norm_eps,
                    elementwise_affine=norm_elementwise_affine,
                    chunkwise_causal = chunkwise_causal,
                    visible_chunk = visible_chunk,
                    is_causal=is_causal,
                )
                for _ in range(num_layers)
            ]
        )
        for layer_idx, block in enumerate(self.action_blocks):
            block.attn1.processor.layer_idx = layer_idx
            block.attn2.processor.layer_idx = layer_idx

        self.action_norm_out = nn.LayerNorm(self.action_inner_dim, eps=1e-6, elementwise_affine=False)

        self.action_proj_out = nn.Linear(self.action_inner_dim, action_in_channels)


        self.use_state_direct = use_state_direct

        if self.use_state_direct:
            self.action_state_adaptor = nn.Sequential(
                nn.Linear(
                    (state_dim),
                    2*self.action_in_channels,
                ),
                nn.GELU(approximate="tanh"),
                nn.Linear(2*self.action_in_channels, 2*self.action_in_channels),
                nn.GELU(approximate="tanh"),
                nn.Linear(2*self.action_in_channels, self.action_in_channels),
            )

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors
    def get_vision_rope(self, vision_latents, n_frame, l_h, l_w, is_sparse, factor):
        return self.vision_rotary_pos_emb(
            vision_latents,
            num_frames=n_frame,
            height=l_h,
            width=l_w,
            is_sparse=is_sparse,
            factor=factor
        )
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)


    def forward(
        self,
        vision_velocity: Optional[torch.Tensor] = None,
        view_ids: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        action_states: torch.Tensor = None,
        action_timestep: torch.LongTensor = None,
        state_cond: torch.Tensor = None,
        mask_to_action=None,
        body_emb_id: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, ...], Transformer2DModelOutput]:

        assert action_states is not None and action_timestep is not None
        profiler = StageProfiler(
            "AE_FORWARD_PROFILE",
            action_states,
            enabled=inference_profile_active() and not self.training,
        )
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        profiler.mark("setup_and_lora")

        n_view = vision_velocity.shape[1]
        if view_ids is None:
            view_ids = torch.arange(
                n_view, device=vision_velocity.device, dtype=torch.long
            ).unsqueeze(0)
        else:
            view_ids = view_ids.to(device=vision_velocity.device, dtype=torch.long)
        assert view_ids.ndim == 2 and view_ids.shape[1] == n_view, f"Expected view_ids shape [B, {n_view}], got {tuple(view_ids.shape)}"
        v_emb = self.view_identities(view_ids)
        vision_velocity = vision_velocity + v_emb.unsqueeze(2)
        vision_velocity = rearrange(vision_velocity, 'b v l c -> b (v l) c')
        if self.use_complex_adapter:
            vision_velocity = self.vision_feature_adapter(vision_velocity)

        profiler.mark("vision_conditioning")

        if self.use_state_direct:
            state_cond = self.action_state_adaptor(state_cond)

            state_token_count = state_cond.shape[1]
            action_states = torch.cat((state_cond, action_states), dim=1)
            state_timesteps = torch.zeros_like(action_timestep[:, :1]).expand(
                -1, state_token_count
            )
            action_timestep = torch.cat((state_timesteps, action_timestep), dim=1)
        batch_size = action_states.shape[0]

        profiler.mark("state_conditioning")

        action_seq_length = action_states.shape[1]
        action_rotary_emb = self.action_rope(action_states, action_seq_length)
        action_hidden_states = self.action_proj_in(action_states)

        profiler.mark("action_input_projection")

        action_temb, action_embedded_timestep = self.action_time_embed(
            action_timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=action_hidden_states.dtype,
        )

        action_temb = action_temb.view(batch_size, -1, action_temb.size(-1))
        action_embedded_timestep = action_embedded_timestep.view(batch_size, -1, action_embedded_timestep.size(-1))

        profiler.mark("timestep_embedding")

        if self.body_embedding_enabled:
            if body_emb_id is None:
                if not getattr(self, "_body_embedding_missing_id_warned", False):
                    self._body_embedding_missing_id_warned = True
                    logger.warning(
                        "[body_embedding] body_emb_id was not provided; defaulting to ID 0. "
                        "Check dataset/trainer wiring."
                    )
                body_emb_id = torch.zeros(
                    batch_size, device=action_hidden_states.device, dtype=torch.long
                )
            else:
                body_emb_id = body_emb_id.to(device=action_hidden_states.device, dtype=torch.long)
                if body_emb_id.ndim == 2 and body_emb_id.shape[-1] == 1:
                    body_emb_id = body_emb_id[:, 0]
                if body_emb_id.ndim != 1 or body_emb_id.shape[0] != batch_size:
                    raise ValueError(
                        "body_emb_id must have shape [B] (or [B, 1]), "
                        f"got {tuple(body_emb_id.shape)} for B={batch_size}"
                    )
            if body_emb_id.numel() and (
                body_emb_id.min() < 0 or body_emb_id.max() >= self.body_embedding.num_embeddings
            ):
                raise ValueError(
                    f"body_emb_id values must be in [0, {self.body_embedding.num_embeddings - 1}], "
                    f"got min={body_emb_id.min().item()} max={body_emb_id.max().item()}"
                )
            body_ada = self.body_embedding(body_emb_id).to(dtype=action_temb.dtype)
            if self.body_embedding_to_adaln is not None:
                body_ada = self.body_embedding_to_adaln(body_ada)
            if body_ada.shape[-1] != action_temb.shape[-1]:
                raise RuntimeError(
                    "body embedding output must match AdaLN timestep width: "
                    f"{body_ada.shape[-1]} != {action_temb.shape[-1]}"
                )
            action_temb = action_temb + body_ada[:, None, :]


        profiler.mark("body_conditioning")

        results = {}

        for block in self.action_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

                action_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    action_hidden_states,
                    vision_velocity,
                    action_temb,
                    action_rotary_emb,
                    None,
                    mask_to_action,
                    **ckpt_kwargs,
                )

            else:
                action_hidden_states = block(
                    hidden_states=action_hidden_states,
                    encoder_hidden_states=vision_velocity,
                    temb=action_temb,
                    rotary_emb=action_rotary_emb,
                    attention_mask=mask_to_action,
                )

        profiler.mark("transformer_blocks")


        action_scale_shift_values = self.action_scale_shift_table[None, None] + action_embedded_timestep[:, :, None]
        action_shift, action_scale = action_scale_shift_values[:,:,0], action_scale_shift_values[:,:,1]

        action_hidden_states = self.action_norm_out(action_hidden_states)
        action_hidden_states = action_hidden_states * (1 + action_scale) + action_shift

        if self.use_state_direct:
            action_hidden_states = action_hidden_states[:, state_token_count:]

        action_output = self.action_proj_out(action_hidden_states)

        results['action'] = action_output

        profiler.mark("output_head")
        profiler.report(
            metadata={
                "blocks": len(self.action_blocks),
            }
        )

        return results


    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func = torch.utils.checkpoint):
        is_gradient_checkpointing_set = False

        if hasattr(self, "gradient_checkpointing"):
            self._gradient_checkpointing_func = gradient_checkpointing_func
            self.gradient_checkpointing = enable
            is_gradient_checkpointing_set = True

        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
                is_gradient_checkpointing_set = True

        if not is_gradient_checkpointing_set:
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with gradient checkpointing. Make sure all the architecture support it by setting a boolean attribute"
                " `gradient_checkpointing` to modules of the model that uses checkpointing."
            )

    def enable_gradient_checkpointing(self, gradient_checkpointing_func = None) -> None:
        if not self._supports_gradient_checkpointing:
            raise ValueError(
                f"{self.__class__.__name__} does not support gradient checkpointing. Please make sure to set the boolean attribute "
                f"`_supports_gradient_checkpointing` to `True` in the class definition."
            )

        if gradient_checkpointing_func is None:

            def _gradient_checkpointing_func(module, *args):
                ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                return torch.utils.checkpoint.checkpoint(
                    module.__call__,
                    *args,
                    **ckpt_kwargs,
                )

            gradient_checkpointing_func = _gradient_checkpointing_func

        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)
