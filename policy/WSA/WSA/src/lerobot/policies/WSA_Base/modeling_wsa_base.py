#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

import copy
import logging
import math
import os
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
import torch._dynamo as dynamo
from einops import rearrange

from transformers.models.auto import CONFIG_MAPPING
from transformers.models.qwen3_vl import modeling_qwen3_vl
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLTextModel

from lerobot.policies.WSA_Base.cosmos_tokenizer.image_lib import ImageTokenizer
from lerobot.policies.WSA_Base.configuration_wsa_base import WSABaseConfig
from lerobot.policies.WSA_Base.da3_teacher import DA3BackboneTeacher
from lerobot.policies.WSA_Base.lora import (
    LoRALinear,
    apply_lora_to_linear_modules,
    freeze_module_except_lora,
    is_lora_parameter_name,
    merge_and_unload_lora_modules_,
    resolve_lora_target_linear_names,
)
from lerobot.policies.names import WSA_BASE
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.utils import format_big_number
from lerobot.utils.constants import (
    HF_HOME,
    ACTION,
    SAMPLE_ACTION_LOSS_MASK,
    OBS_STATE,
    OBS_PREFIX,
    OBS_IMAGES,
    OPENPI_ATTENTION_MASK_VALUE,
)

_MISSING_ATTN_IMPL = object()


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for tensors shaped as [B, C, H, W]."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class Future3DRefineBlock(nn.Module):
    """A lightweight shared spatial refiner used after per-view 2D upsampling."""

    def __init__(self, dim: int):
        super().__init__()
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.depthwise_norm = LayerNorm2d(dim)
        # A per-location linear projection is equivalent to a 1x1 conv here, but
        # keeps DDP gradient buckets in a standard contiguous layout more reliably.
        self.pointwise = nn.Linear(dim, dim)
        self.pointwise_norm = LayerNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.depthwise_norm(x)
        x = F.silu(x)
        x = rearrange(x, "b c h w -> b h w c")
        x = self.pointwise(x)
        x = rearrange(x, "b h w c -> b c h w").contiguous()
        x = self.pointwise_norm(x)
        x = F.silu(x)
        return x + residual


class Future3DPerceiverFeedForward(nn.Module):
    """LayerNorm-MLP residual branch used inside the DA3 query resampler."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim, bias=False),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Future3DPerceiverAttention(nn.Module):
    """Perceiver-style cross-attention where queries read messenger tokens only."""

    def __init__(self, dim: int, dim_head: int, heads: int):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.memory_norm = nn.LayerNorm(dim)
        self.query_norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, width = x.shape
        x = x.view(batch_size, seq_len, self.heads, width // self.heads)
        return x.transpose(1, 2)

    def forward(self, memory: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        memory = self.memory_norm(memory)
        queries = self.query_norm(queries)

        q = self.to_q(queries)
        k, v = self.to_kv(memory).chunk(2, dim=-1)

        q = self._reshape_heads(q)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(queries.shape[0], queries.shape[1], -1)
        return self.to_out(attended)


class Future3DPerceiverResampler(nn.Module):
    """Shared Perceiver-style resampler that expands messenger tokens into DA3-aligned query latents."""

    def __init__(self, dim: int, num_heads: int, output_dim: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        dim_head = dim // num_heads
        self.attn = Future3DPerceiverAttention(dim=dim, dim_head=dim_head, heads=num_heads)
        self.ff = Future3DPerceiverFeedForward(dim=dim)
        self.output_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, output_dim)

    def forward(self, output_queries: torch.Tensor, messenger_tokens: torch.Tensor) -> torch.Tensor:
        latents = output_queries
        latents = latents + self.attn(messenger_tokens, latents)
        latents = latents + self.ff(latents)
        return self.output_proj(self.output_norm(latents))


def infer_token_grid(num_tokens: int) -> tuple[int, int]:
    """Factorizes token count into a near-square 2D grid."""

    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")

    best_h, best_w = 1, num_tokens
    for height in range(1, math.isqrt(num_tokens) + 1):
        if num_tokens % height != 0:
            continue
        width = num_tokens // height
        if abs(width - height) < abs(best_w - best_h):
            best_h, best_w = height, width
    return best_h, best_w


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))

# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(
    layer_idx, inputs_embeds, attention_mask, position_ids, und_expert, gen_expert, act_expert
):
    models = [und_expert.language_model, gen_expert, act_expert]
    query_states = []
    key_states = []
    value_states = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states = layer.input_layernorm(hidden_states)  # noqa: PLW2901
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        if layer.self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype=torch.bfloat16)
        query_state = layer.self_attn.q_norm(layer.self_attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_state = layer.self_attn.k_norm(layer.self_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = und_expert.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_qwen3_vl.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    scaling = und_expert.language_model.layers[layer_idx].self_attn.scaling
    # Attention computation
    att_output, _ = modeling_qwen3_vl.eager_attention_forward(
        und_expert.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = und_expert.language_model.layers[layer_idx].self_attn.head_dim
    num_attention_heads = und_expert.language_model.layers[layer_idx].self_attn.config.num_attention_heads
    att_output = att_output.reshape(batch_size, -1, 1 * num_attention_heads * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = out_emb + hidden_states
        after_first_residual = out_emb.clone()
        out_emb = layer.post_attention_layernorm(out_emb)
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = out_emb + after_first_residual
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class QwenConfig:
    """Configuration for Qwen model variants."""

    def __init__(self, head_dim, hidden_size, intermediate_size, num_attention_heads, num_hidden_layers, num_key_value_heads):
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads


def get_qwen_config(variant: str) -> QwenConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    num_hidden_layers = int(variant.split('_')[-1][:-1])  # pattern: qwen3_vl_28l or qwen3_xxl
    if variant.startswith("qwen3_vl"):
        return QwenConfig(
            head_dim=128,
            hidden_size=2048,
            intermediate_size=6144,
            num_attention_heads=16,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=8,
        )
    elif variant.startswith("qwen3"):
        return QwenConfig(
            head_dim=128,
            hidden_size=1024,
            intermediate_size=3072,
            num_attention_heads=16,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=8,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class Qwen3VLWithExpertModel(
    nn.Module
):
    """Qwen3-VL backbone with expert heads for WSABase."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        qwen3_vl_pretrained_path: str,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["qwen3_vl"]()
        vlm_config_hf.text_config.hidden_size = vlm_config.hidden_size
        vlm_config_hf.text_config.intermediate_size = vlm_config.intermediate_size
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_attention_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.num_hidden_layers
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_key_value_heads
        vlm_config_hf.text_config.max_position_embeddings = 262144
        vlm_config_hf.text_config.rope_scaling = {
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_type": "default"
        }
        vlm_config_hf.text_config.tie_word_embeddings = True
        vlm_config_hf.tie_word_embeddings = True
        vlm_config_hf.vision_config.deepstack_visual_indexes=[5, 11, 17]
        vlm_config_hf.vision_config.depth=24
        vlm_config_hf.vision_config.hidden_size=1024
        vlm_config_hf.vision_config.intermediate_size=4096
        vlm_config_hf.vision_config.out_hidden_size=2048

        # self.und_expert = Qwen3VLForConditionalGeneration(config=vlm_config_hf)
        self.und_expert = Qwen3VLForConditionalGeneration.from_pretrained(
            qwen3_vl_pretrained_path,
            config=vlm_config_hf,
            ignore_mismatched_sizes=False
        )

        gen_expert_config_hf = CONFIG_MAPPING["qwen3_vl_text"]()
        gen_expert_config_hf.head_dim=action_expert_config.head_dim
        gen_expert_config_hf.hidden_size=action_expert_config.hidden_size
        gen_expert_config_hf.intermediate_size=action_expert_config.intermediate_size
        gen_expert_config_hf.num_attention_heads=action_expert_config.num_attention_heads
        gen_expert_config_hf.num_hidden_layers=action_expert_config.num_hidden_layers
        gen_expert_config_hf.num_key_value_heads=action_expert_config.num_key_value_heads
        gen_expert_config_hf.max_position_embeddings = self.und_expert.config.text_config.max_position_embeddings
        gen_expert_config_hf.rope_scaling = self.und_expert.config.text_config.rope_scaling
        self.gen_expert = Qwen3VLTextModel(config=gen_expert_config_hf)
        self.gen_expert.embed_tokens = None
        self.gen_expert.lm_head = None

        action_expert_config_hf = CONFIG_MAPPING["qwen3_vl_text"]()
        action_expert_config_hf.head_dim=action_expert_config.head_dim
        action_expert_config_hf.hidden_size=action_expert_config.hidden_size
        action_expert_config_hf.intermediate_size=action_expert_config.intermediate_size
        action_expert_config_hf.num_attention_heads=action_expert_config.num_attention_heads
        action_expert_config_hf.num_hidden_layers=action_expert_config.num_hidden_layers
        action_expert_config_hf.num_key_value_heads=action_expert_config.num_key_value_heads
        action_expert_config_hf.max_position_embeddings = self.und_expert.config.text_config.max_position_embeddings
        action_expert_config_hf.rope_scaling = self.und_expert.config.text_config.rope_scaling
        self.act_expert = Qwen3VLTextModel(config=action_expert_config_hf)
        self.act_expert.embed_tokens = None
        self.act_expert.lm_head = None

        assert self.und_expert.config.text_config.num_hidden_layers == self.act_expert.config.num_hidden_layers

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            # "visual.patch_embed.proj.weight",
            # "visual.patch_embed.proj.bias",
            # "visual.pos_embed.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        collect_middle_layers: tuple[int, ...] | None = None,
    ):
        collected_middle_states = None
        if inputs_embeds[1] is None and inputs_embeds[2] is None:
            prefix_output = self.und_expert.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            middle_output = None
            suffix_output = None

        elif inputs_embeds[0] is None and inputs_embeds[2] is None:
            middle_output = self.gen_expert.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            past_key_values = middle_output.past_key_values
            prefix_output = None
            middle_output = middle_output.last_hidden_state
            suffix_output = None

        elif inputs_embeds[0] is None and inputs_embeds[1] is None:
            suffix_output = self.act_expert.forward(
                inputs_embeds=inputs_embeds[2],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            past_key_values = None
            prefix_output = None
            middle_output = None
            suffix_output = suffix_output.last_hidden_state
        else:
            models = [self.und_expert.language_model, self.gen_expert, self.act_expert]
            num_layers = self.und_expert.config.text_config.num_hidden_layers
            collect_middle_map = None
            if collect_middle_layers:
                collect_middle_map = {}

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.act_expert, "gradient_checkpointing")
                and self.gen_expert.gradient_checkpointing
                and self.act_expert.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        und_expert=self.und_expert,
                        gen_expert=self.gen_expert,
                        act_expert=self.act_expert,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        und_expert=self.und_expert,
                        gen_expert=self.gen_expert,
                        act_expert=self.act_expert,
                    )

                if collect_middle_map is not None and layer_idx in collect_middle_layers:
                    collect_middle_map[layer_idx] = inputs_embeds[1]

            # final norm
            def compute_final_norms(inputs_embeds):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb = models[i].norm(hidden_states)
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds)

            past_key_values = None
            prefix_output = outputs_embeds[0]
            middle_output = outputs_embeds[1]
            suffix_output = outputs_embeds[2]
            if collect_middle_map is not None:
                collected_middle_states = tuple(collect_middle_map[layer_idx] for layer_idx in collect_middle_layers)

        if collect_middle_layers is None:
            return [prefix_output, middle_output, suffix_output], past_key_values
        return [prefix_output, middle_output, suffix_output], past_key_values, collected_middle_states


class WSABaseModel(nn.Module):

    @staticmethod
    def _resolve_cosmos_tokenizer_dir(path_or_name: str) -> str:
        candidate_dir = os.path.expanduser(path_or_name)
        if os.path.isdir(candidate_dir):
            enc_path = os.path.join(candidate_dir, "encoder.jit")
            dec_path = os.path.join(candidate_dir, "decoder.jit")
            if not os.path.exists(enc_path) or not os.path.exists(dec_path):
                raise FileNotFoundError(
                    f"Cosmos tokenizer directory '{candidate_dir}' must contain encoder.jit and decoder.jit."
                )
            return candidate_dir

        from huggingface_hub import snapshot_download

        logging.warning("Cosmos tokenizer '%s' not found as a local directory, resolving via Hugging Face Hub.", path_or_name)
        downloaded_dir = snapshot_download(repo_id=path_or_name)
        enc_path = os.path.join(downloaded_dir, "encoder.jit")
        dec_path = os.path.join(downloaded_dir, "decoder.jit")
        if not os.path.exists(enc_path) or not os.path.exists(dec_path):
            raise FileNotFoundError(
                f"Resolved cosmos tokenizer '{path_or_name}' to '{downloaded_dir}', but encoder.jit/decoder.jit were not found."
            )
        return downloaded_dir

    def _get_lora_injection_roots(self) -> dict[str, nn.Module]:
        return {
            "und": self.qwen3_vl_with_expert.und_expert.language_model,
            "gen": self.qwen3_vl_with_expert.gen_expert,
            "act": self.qwen3_vl_with_expert.act_expert,
        }

    def _get_lora_freeze_roots(self) -> dict[str, nn.Module]:
        return {
            "und": self.qwen3_vl_with_expert.und_expert.language_model,
            "gen": self.qwen3_vl_with_expert.gen_expert,
            "act": self.qwen3_vl_with_expert.act_expert,
        }

    def _register_frozen_eval_module(self, module: nn.Module):
        if any(existing_module is module for existing_module in self._frozen_eval_modules):
            return
        self._frozen_eval_modules.append(module)

    def _apply_lora_to_selected_experts(self):
        self.lora_module_counts = {}
        if not self.config.lora_enabled:
            return

        target_names = resolve_lora_target_linear_names(self.config.lora_targets)
        for module_name, root_module in self._get_lora_injection_roots().items():
            if module_name not in self.config.lora_modules:
                continue
            module_rank, module_alpha = self.config.get_lora_hparams_for(module_name)
            num_replaced = apply_lora_to_linear_modules(
                root_module,
                target_names=target_names,
                rank=module_rank,
                alpha=module_alpha,
                dropout=self.config.lora_dropout,
            )
            if num_replaced == 0:
                raise ValueError(
                    f"LoRA module selection '{module_name}' did not match any Linear layers for "
                    f"targets {self.config.lora_targets}."
                )
            self.lora_module_counts[module_name] = num_replaced

        logging.info(
            "Applied WSABase LoRA to experts: %s",
            ", ".join(
                f"{name}(count={count}, rank={self.config.get_lora_rank_for(name)}, "
                f"alpha={self.config.get_lora_alpha_for(name)})"
                for name, count in self.lora_module_counts.items()
            ),
        )

    def __init__(self, config: WSABaseConfig):
        super().__init__()
        self.config = config
        self.omit_visual_tokens_in_causal_inference = True
        self._frozen_eval_modules: list[nn.Module] = []
        self.lora_module_counts: dict[str, int] = {}

        vlm_config = get_qwen_config(config.qwen3_vl_variant)
        action_expert_config = get_qwen_config(config.action_expert_variant)

        self.qwen3_vl_with_expert = Qwen3VLWithExpertModel(
            vlm_config,
            action_expert_config,
            qwen3_vl_pretrained_path=config.qwen3_vl_pretrained_path,
            precision=config.dtype,
        )
        self._apply_lora_to_selected_experts()

        cosmos_tokenizer_dir = self._resolve_cosmos_tokenizer_dir(config.cosmos_tokenizer_path_or_name)
        cosmos_device = getattr(config, "cosmos_device", None) or config.device or "cuda"

        self.cosmos = ImageTokenizer(
            checkpoint_enc=os.path.join(cosmos_tokenizer_dir, "encoder.jit"),
            checkpoint_dec=os.path.join(cosmos_tokenizer_dir, "decoder.jit"),
            device=cosmos_device,
        )

        vae_dim = 16
        gen_proj_dim = action_expert_config.hidden_size
        ds = self.config.scale_factor
        # self.downsample_conv = nn.Conv2d(in_channels=vae_dim, out_channels=gen_proj_dim, kernel_size=ds, stride=ds, padding=0) # junhao
        # self.cosmos_in_proj = nn.Conv2d(in_channels=gen_proj_dim, out_channels=vlm_config.hidden_size, kernel_size=1, stride=1, padding=0) # junhao
        # self.cosmos_out_proj = nn.Conv2d(in_channels=vlm_config.hidden_size, out_channels=gen_proj_dim, kernel_size=1, stride=1, padding=0) # junhao
        # self.upsample_conv = nn.ConvTranspose2d(in_channels=gen_proj_dim, out_channels=vae_dim, kernel_size=ds, stride=ds, padding=0, output_padding=0) # junhao

        self.cosmos_in_proj = nn.Conv2d(in_channels=vae_dim, out_channels=gen_proj_dim, kernel_size=1, stride=1, padding=0) # jia
        self.downsample_conv = nn.Conv2d(in_channels=gen_proj_dim, out_channels=gen_proj_dim, kernel_size=ds, stride=ds, padding=0) # jia
        self.upsample_conv = nn.ConvTranspose2d(in_channels=gen_proj_dim, out_channels=gen_proj_dim, kernel_size=ds, stride=ds, padding=0, output_padding=0) # jia
        # self.cosmos_out_proj = nn.Conv2d(in_channels=gen_proj_dim, out_channels=vae_dim, kernel_size=1, stride=1, padding=0) # jia
        self.cosmos_out_proj = nn.Linear(gen_proj_dim, vae_dim)
        self.cosmos_out_layer_norm = nn.LayerNorm(gen_proj_dim) # jia

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.hidden_size)
        self.action_out_proj = nn.Linear(action_expert_config.hidden_size, config.max_action_dim)

        self.state_proj = nn.Linear(config.max_state_dim, action_expert_config.hidden_size)
        self.action_time_mlp_in = nn.Linear(2 * action_expert_config.hidden_size, action_expert_config.hidden_size)
        self.action_time_mlp_out = nn.Linear(action_expert_config.hidden_size, action_expert_config.hidden_size)

        self.query_layer_indices = tuple(config.query_layer_indices)
        self.da3_teacher_layers = tuple(config.da3_teacher_layers)
        self.middle_visual_token_count = 0
        self.middle_visual_tokens_per_view = 0
        self.middle_query_token_count = 0
        self.middle_total_token_count = 0
        self.future_3d_output_token_count = config.da3_num_views * config.da3_tokens_per_view
        self.future_3d_output_queries_per_view = config.da3_tokens_per_view
        self.future_3d_tokens_per_view = config.num_3d_query_tokens // config.da3_num_views
        self.future_3d_query_grid = infer_token_grid(self.future_3d_tokens_per_view)
        self.da3_target_grid = infer_token_grid(config.da3_tokens_per_view)

        if config.enable_3d_queries:
            self.future_3d_queries = nn.Parameter(
                torch.randn(1, config.num_3d_query_tokens, action_expert_config.hidden_size)
                * config.future_query_init_std
            )
            if config.da3_alignment_mode == "query_decoder":
                self.future_3d_shared_output_queries = nn.Parameter(
                    torch.randn(1, self.future_3d_output_queries_per_view, action_expert_config.hidden_size)
                    * config.future_query_init_std
                )
                self.future_3d_messenger_norms = nn.ModuleList(
                    [nn.LayerNorm(action_expert_config.hidden_size) for _ in self.query_layer_indices]
                )
                self.future_3d_output_decoder = Future3DPerceiverResampler(
                    dim=action_expert_config.hidden_size,
                    num_heads=action_expert_config.num_attention_heads,
                    output_dim=config.da3_query_dim,
                )
                self.future_3d_layer_input_norms = nn.ModuleList()
                self.future_3d_shared_refine_trunk = None
                self.da3_query_projectors = nn.ModuleList()
            else:
                self.register_parameter("future_3d_shared_output_queries", None)
                self.future_3d_messenger_norms = nn.ModuleList()
                self.future_3d_output_decoder = None
                self.future_3d_layer_input_norms = nn.ModuleList(
                    [nn.LayerNorm(action_expert_config.hidden_size) for _ in self.query_layer_indices]
                )
                self.future_3d_shared_refine_trunk = Future3DRefineBlock(action_expert_config.hidden_size)
                self.da3_query_projectors = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.LayerNorm(action_expert_config.hidden_size),
                            nn.Linear(action_expert_config.hidden_size, config.da3_query_dim),
                        )
                        for _ in self.query_layer_indices
                    ]
                )
        else:
            self.register_parameter("future_3d_queries", None)
            self.register_parameter("future_3d_shared_output_queries", None)
            self.future_3d_messenger_norms = nn.ModuleList()
            self.future_3d_output_decoder = None
            self.future_3d_layer_input_norms = nn.ModuleList()
            self.future_3d_shared_refine_trunk = None
            self.da3_query_projectors = nn.ModuleList()

        if config.lambda_3d > 0 and config.enable_3d_queries:
            teacher_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
            self.da3_teacher = DA3BackboneTeacher(
                model_path_or_name=config.da3_model_path_or_name,
                code_root=config.da3_code_root,
                process_res=config.da3_teacher_process_res,
                dtype=teacher_dtype,
                teacher_layers=self.da3_teacher_layers,
            )
            if self.da3_teacher.feature_dim != config.da3_query_dim:
                raise ValueError(
                    f"DA3 query dim ({config.da3_query_dim}) does not match the loaded teacher feature dim "
                    f"({self.da3_teacher.feature_dim}). Check policy.da3_variant / policy.da3_model_path_or_name."
                )
        else:
            self.da3_teacher = None

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            # Also compile the main forward pass used during training
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

        self.set_requires_grad()

    def set_requires_grad(self):
        self._frozen_eval_modules = []

        if self.config.freeze_vision_encoder:
            self.qwen3_vl_with_expert.und_expert.visual.eval()
            for params in self.qwen3_vl_with_expert.und_expert.visual.parameters():
                params.requires_grad = False
            self._register_frozen_eval_module(self.qwen3_vl_with_expert.und_expert.visual)

        if self.config.lora_enabled:
            freeze_roots = self._get_lora_freeze_roots()
            selected_modules = set(self.config.lora_modules)

            for module_name in selected_modules:
                freeze_root = freeze_roots[module_name]
                freeze_module_except_lora(freeze_root)

            if self.config.lora_unselected_mode == "freeze":
                for module_name, freeze_root in freeze_roots.items():
                    if module_name in selected_modules:
                        continue
                    freeze_root.eval()
                    for params in freeze_root.parameters():
                        params.requires_grad = False
                    self._register_frozen_eval_module(freeze_root)

        if self.config.train_expert_only:
            self.qwen3_vl_with_expert.und_expert.eval()
            for params in self.qwen3_vl_with_expert.und_expert.parameters():
                params.requires_grad = False
            self._register_frozen_eval_module(self.qwen3_vl_with_expert.und_expert)

        if self.config.train_vlm_only:
            self.qwen3_vl_with_expert.gen_expert.eval()
            for params in self.qwen3_vl_with_expert.gen_expert.parameters():
                params.requires_grad = False
            self.qwen3_vl_with_expert.act_expert.eval()
            for params in self.qwen3_vl_with_expert.act_expert.parameters():
                params.requires_grad = False
            self._register_frozen_eval_module(self.qwen3_vl_with_expert.gen_expert)
            self._register_frozen_eval_module(self.qwen3_vl_with_expert.act_expert)

        self.cosmos.eval()
        for params in self.cosmos.parameters():
            params.requires_grad = False
        self._register_frozen_eval_module(self.cosmos)

        if self.da3_teacher is not None:
            self.da3_teacher.eval()
            for params in self.da3_teacher.parameters():
                params.requires_grad = False
            self._register_frozen_eval_module(self.da3_teacher)

    def train(self, mode: bool = True):
        super().train(mode)
        for frozen_eval_module in self._frozen_eval_modules:
            frozen_eval_module.eval()
        return self

    @contextmanager
    def _temporary_attention_implementations(
        self,
        *,
        und_expert_impl: str | None = None,
        gen_expert_impl: str | None = None,
        act_expert_impl: str | None = None,
    ):
        previous_impls = []

        def set_impl(config: object, new_impl: str | None):
            if new_impl is None:
                return
            previous_impls.append((config, getattr(config, "_attn_implementation", _MISSING_ATTN_IMPL)))
            config._attn_implementation = new_impl

        set_impl(self.qwen3_vl_with_expert.und_expert.language_model.config, und_expert_impl)
        set_impl(self.qwen3_vl_with_expert.gen_expert.config, gen_expert_impl)
        set_impl(self.qwen3_vl_with_expert.act_expert.config, act_expert_impl)

        try:
            yield
        finally:
            for config, previous_impl in reversed(previous_impls):
                if previous_impl is _MISSING_ATTN_IMPL:
                    delattr(config, "_attn_implementation")
                else:
                    config._attn_implementation = previous_impl

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.qwen3_vl_with_expert.und_expert.language_model.gradient_checkpointing = True
        self.qwen3_vl_with_expert.und_expert.visual.gradient_checkpointing = True
        self.qwen3_vl_with_expert.gen_expert.gradient_checkpointing = True
        self.qwen3_vl_with_expert.act_expert.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for WSABase model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.qwen3_vl_with_expert.und_expert.language_model.gradient_checkpointing = False
        self.qwen3_vl_with_expert.und_expert.visual.gradient_checkpointing = False
        self.qwen3_vl_with_expert.gen_expert.gradient_checkpointing = False
        self.qwen3_vl_with_expert.act_expert.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for WSABase model")

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def _prepare_suffix_denoise_cache(
        self,
        prefix_pad_masks: torch.Tensor,
        max_prefix_position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build suffix attention/position tensors once for the denoising loop.

        During inference, the suffix layout is fixed: one state token followed by
        `chunk_size` action tokens. Only the suffix embeddings change per denoise
        step, so the attention mask and position ids can be reused.
        """
        suffix_len = 1 + self.config.chunk_size
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_pad_masks = torch.ones(
            batch_size,
            suffix_len,
            dtype=torch.bool,
            device=prefix_pad_masks.device,
        )
        suffix_att_masks = torch.zeros(
            batch_size,
            suffix_len,
            dtype=torch.float32,
            device=prefix_pad_masks.device,
        )
        suffix_att_masks[:, 0] = True
        suffix_att_masks[:, 1] = True
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        position_ids = (
            torch.arange(1, suffix_len + 1).repeat(3, 1, 1).to(max_prefix_position_ids)
            + max_prefix_position_ids
        )
        return full_att_2d_masks_4d, position_ids

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    @dynamo.disable
    def embed_prefix(
        self, pixel_values, image_grid_thw, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_token_id = self.qwen3_vl_with_expert.und_expert.config.image_token_id
        D1 = pixel_values.shape[-1]
        pixel_values = pixel_values.view(-1, D1)
        image_grid_thw = image_grid_thw.view(-1, 3)
        image_embs, _ = self.qwen3_vl_with_expert.und_expert.visual(pixel_values, image_grid_thw)

        embs = self.qwen3_vl_with_expert.und_expert.get_input_embeddings()(lang_tokens)
        B, L, D2 = embs.shape
        embs = embs.view(-1, D2)
        lang_tokens = lang_tokens.view(-1)
        embs[lang_tokens == image_token_id] = image_embs  # replace dummy embeds with image_embeds
        embs = embs.view(B, L, D2)

        pad_masks = lang_masks.to(torch.bool)
        att_masks = torch.zeros_like(pad_masks, dtype=torch.bool, device=pad_masks.device)

        return embs, pad_masks, att_masks

    def get_cosmos_features(self, images):
        shape = images.shape[:-3]
        c, h, w = images.shape[-3:]
        images = images.reshape(-1, c, h, w)
        images = F.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)
        images = images * 2 - 1  # [-1, 1]
        features = self.cosmos.encode(images)
        c, h, w = features.shape[-3:]
        features = features.view(*shape, c, h, w)
        return features

    def append_future_query_tokens(self, embs, pad_masks, att_masks):
        if self.future_3d_queries is None:
            self.middle_query_token_count = 0
            self.middle_total_token_count = embs.shape[1]
            return embs, pad_masks, att_masks

        bsize = embs.shape[0]
        query_tokens = self.future_3d_queries.expand(bsize, -1, -1).to(dtype=embs.dtype, device=embs.device)
        query_pad_masks = torch.ones(
            bsize,
            query_tokens.shape[1],
            dtype=pad_masks.dtype,
            device=pad_masks.device,
        )
        query_att_masks = torch.zeros(
            bsize,
            query_tokens.shape[1],
            dtype=att_masks.dtype,
            device=att_masks.device,
        )
        # Keep cosmos visual tokens and 3D queries inside the same middle block, but
        # split them into two sub-blocks:
        # - visual/cosmos tokens attend bidirectionally within the visual sub-block only
        # - 3D query tokens attend bidirectionally within the query sub-block and can
        #   also attend to the preceding visual/cosmos sub-block
        # This is encoded by starting a new attention segment at the first query token.
        if query_tokens.shape[1] > 0:
            query_att_masks[:, 0] = True

        self.middle_query_token_count = query_tokens.shape[1]
        self.middle_total_token_count = embs.shape[1] + query_tokens.shape[1]
        embs = torch.cat([embs, query_tokens], dim=1)
        pad_masks = torch.cat([pad_masks, query_pad_masks], dim=1)
        att_masks = torch.cat([att_masks, query_att_masks], dim=1)
        return embs, pad_masks, att_masks

    def apply_view_aware_query_attention(
        self,
        att_2d_masks: torch.Tensor,
        prefix_len: int = 0,
    ) -> torch.Tensor:
        """Restrict query-to-visual access per view while preserving full query-group communication."""
        if self.middle_query_token_count == 0:
            return att_2d_masks

        if self.middle_query_token_count % self.config.da3_num_views != 0:
            raise ValueError(
                f"middle_query_token_count ({self.middle_query_token_count}) must be divisible by "
                f"da3_num_views ({self.config.da3_num_views})"
            )
        if self.middle_visual_token_count % self.config.da3_num_views != 0:
            raise ValueError(
                f"middle_visual_token_count ({self.middle_visual_token_count}) must be divisible by "
                f"da3_num_views ({self.config.da3_num_views})"
            )

        visual_start = prefix_len
        visual_end = visual_start + self.middle_visual_token_count
        query_start = visual_end
        query_end = query_start + self.middle_query_token_count

        original_query_rows = att_2d_masks[:, query_start:query_end, :].clone()
        att_2d_masks = att_2d_masks.clone()
        att_2d_masks[:, query_start:query_end, prefix_len:] = False

        if prefix_len > 0:
            att_2d_masks[:, query_start:query_end, :prefix_len] = original_query_rows[:, :, :prefix_len]

        # Keep the full 3D messenger block bidirectional so view groups can exchange information,
        # while still constraining each group to read visual evidence only from its matching view.
        att_2d_masks[:, query_start:query_end, query_start:query_end] = original_query_rows[:, :, query_start:query_end]

        visual_tokens_per_view = self.middle_visual_token_count // self.config.da3_num_views
        query_tokens_per_view = self.middle_query_token_count // self.config.da3_num_views

        for view_idx in range(self.config.da3_num_views):
            rel_query_start = view_idx * query_tokens_per_view
            rel_query_end = rel_query_start + query_tokens_per_view
            abs_query_start = query_start + rel_query_start
            abs_query_end = abs_query_start + query_tokens_per_view
            abs_visual_start = visual_start + view_idx * visual_tokens_per_view
            abs_visual_end = abs_visual_start + visual_tokens_per_view

            att_2d_masks[:, abs_query_start:abs_query_end, abs_visual_start:abs_visual_end] = original_query_rows[
                :, rel_query_start:rel_query_end, abs_visual_start:abs_visual_end
            ]

        return att_2d_masks

    def uses_causal_attention(self) -> bool:
        return getattr(self.config, "attention_mask_mode", "default") == "causal"

    @staticmethod
    def _allow_attention_block(mask: torch.Tensor, query_slice: slice, key_slice: slice) -> None:
        if query_slice.start == query_slice.stop or key_slice.start == key_slice.stop:
            return
        mask[:, query_slice, key_slice] = True

    def apply_causal_attention(
        self,
        att_2d_masks: torch.Tensor,
        pad_masks: torch.Tensor,
        prefix_len: int,
    ) -> torch.Tensor:
        """Apply the optional causal ablation mask.

        The default WSABase path never calls this method. In causal mode,
        Middle visual tokens, 3D scene/query tokens, and action tokens follow
        the block relation used for the ablation:

            visual       -> current, visual, scene
            scene        -> current, scene, action tokens
            action tokens -> current, scene, state, action tokens

        The state token keeps the default suffix behavior: it is a separate
        token before action tokens, so action tokens can read it, while middle
        visual/scene rows do not receive state as an extra condition. In this
        ablation, state/action suffix rows also do not read middle visual tokens.
        """
        if self.middle_query_token_count <= 0:
            raise ValueError("causal attention requires enabled 3D query tokens.")

        total_len = pad_masks.shape[1]
        visual_start = prefix_len
        visual_end = visual_start + self.middle_visual_token_count
        scene_start = visual_end
        scene_end = scene_start + self.middle_query_token_count
        suffix_start = scene_end
        state_start = suffix_start
        state_end = state_start + 1
        action_start = state_end
        action_end = total_len

        if action_start > total_len:
            raise ValueError(
                "causal attention expected suffix tokens after middle tokens, "
                f"got total_len={total_len}, action_start={action_start}."
            )

        prefix = slice(0, prefix_len)
        visual = slice(visual_start, visual_end)
        scene = slice(scene_start, scene_end)
        state = slice(state_start, state_end)
        action = slice(action_start, action_end)

        allowed = torch.zeros_like(att_2d_masks, dtype=torch.bool)

        # Prefix/current-condition rows.
        self._allow_attention_block(allowed, prefix, prefix)

        # Subgoal / middle 2D image rows.
        for key_block in (prefix, visual, scene):
            self._allow_attention_block(allowed, visual, key_block)

        # Goal 3D scene rows.
        for key_block in (prefix, scene, action):
            self._allow_attention_block(allowed, scene, key_block)

        # State row follows the default suffix position, but does not read
        # middle visual tokens in the causal ablation.
        for key_block in (prefix, scene, state):
            self._allow_attention_block(allowed, state, key_block)

        # Action token rows follow the target mask: current, scene, state, and
        # action tokens, but not middle visual tokens.
        for key_block in (prefix, scene, state, action):
            self._allow_attention_block(allowed, action, key_block)

        pad_2d_masks = pad_masks[:, None, :].to(torch.bool) & pad_masks[:, :, None].to(torch.bool)
        return allowed & pad_2d_masks

    def build_training_attention_mask(
        self,
        pad_masks: torch.Tensor,
        att_masks: torch.Tensor,
        prefix_len: int,
    ) -> torch.Tensor:
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        if self.uses_causal_attention():
            return self.apply_causal_attention(att_2d_masks, pad_masks, prefix_len=prefix_len)
        return self.apply_view_aware_query_attention(att_2d_masks, prefix_len=prefix_len)

    def split_middle_tokens(self, middle_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        visual_tokens = middle_tokens[:, : self.middle_visual_token_count]
        if self.middle_query_token_count == 0:
            return visual_tokens, None
        query_tokens = middle_tokens[:, self.middle_visual_token_count :]
        return visual_tokens, query_tokens

    def reshape_future_queries_to_view_grid(self, query_tokens: torch.Tensor) -> torch.Tensor:
        expected_tokens = self.config.da3_num_views * self.future_3d_tokens_per_view
        if query_tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} future 3D query tokens, got {query_tokens.shape[1]}"
            )
        grid_h, grid_w = self.future_3d_query_grid
        return rearrange(
            query_tokens,
            "b (v h w) c -> b v h w c",
            v=self.config.da3_num_views,
            h=grid_h,
            w=grid_w,
        )

    def refine_projected_query_grid(self, query_tokens: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if self.future_3d_shared_refine_trunk is None:
            raise RuntimeError("Future 3D refine trunk is unavailable in the current alignment mode")

        query_tokens = self.future_3d_layer_input_norms[layer_idx](query_tokens.to(dtype=torch.float32))
        query_grid = self.reshape_future_queries_to_view_grid(query_tokens)
        query_grid = rearrange(query_grid, "b v h w c -> (b v) c h w")
        query_grid = F.interpolate(
            query_grid,
            size=self.da3_target_grid,
            mode="bilinear",
            align_corners=False,
        )
        query_grid = self.future_3d_shared_refine_trunk(query_grid)
        return rearrange(
            query_grid,
            "(b v) c h w -> b (v h w) c",
            b=query_tokens.shape[0],
            v=self.config.da3_num_views,
        )

    def get_3d_token_mask(self, img_masks: torch.Tensor, target_len: int) -> torch.Tensor:
        token_mask = img_masks.unsqueeze(-1).expand(-1, -1, self.config.da3_tokens_per_view).reshape(img_masks.shape[0], -1)
        if token_mask.shape[1] == target_len:
            return token_mask
        token_mask = token_mask[:, None, :].to(dtype=torch.float32)
        token_mask = F.interpolate(token_mask, size=target_len, mode="nearest")
        return token_mask[:, 0, :].to(dtype=torch.bool)

    def prepare_da3_teacher_inputs(
        self,
        future_images: torch.Tensor,
        img_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace placeholder views before DA3 forward while keeping the original loss mask.

        This keeps the main model's masking semantics unchanged while ensuring the
        teacher never sees placeholder views. The returned mask still reflects the
        originally valid views, so 3D loss is only applied where the sample had
        real visual evidence.
        """
        valid_view_masks = img_masks.to(dtype=torch.bool)
        valid_view_counts = valid_view_masks.sum(dim=1)
        incomplete_samples = (valid_view_counts > 0) & (valid_view_counts < future_images.shape[1])
        if not incomplete_samples.any():
            return future_images, valid_view_masks

        num_views = future_images.shape[1]
        batch_indices = torch.arange(future_images.shape[0], device=future_images.device)
        primary_view_indices = valid_view_masks.to(dtype=torch.int64).argmax(dim=1)
        primary_images = future_images[batch_indices, primary_view_indices]

        teacher_images = future_images.clone()
        invalid_view_masks = ~valid_view_masks
        if invalid_view_masks.any():
            teacher_images[invalid_view_masks] = primary_images.unsqueeze(1).expand(-1, num_views, -1, -1, -1)[
                invalid_view_masks
            ]

        teacher_img_masks = valid_view_masks.clone()
        return teacher_images, teacher_img_masks

    def decode_3d_queries_from_messenger_tokens(
        self,
        messenger_tokens: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        if self.future_3d_shared_output_queries is None or self.future_3d_output_decoder is None:
            raise RuntimeError("Future 3D output decoder is unavailable when 3D queries are disabled")

        decoder_dtype = next(self.future_3d_output_decoder.parameters()).dtype
        decoder_device = next(self.future_3d_output_decoder.parameters()).device
        messenger_tokens = messenger_tokens.to(device=decoder_device, dtype=decoder_dtype)
        messenger_tokens = self.future_3d_messenger_norms[layer_idx](messenger_tokens)
        expected_messenger_tokens = self.config.da3_num_views * self.future_3d_tokens_per_view
        if messenger_tokens.shape[1] != expected_messenger_tokens:
            raise ValueError(
                f"Expected {expected_messenger_tokens} messenger tokens, got {messenger_tokens.shape[1]}"
            )

        batch_size = messenger_tokens.shape[0]
        num_views = self.config.da3_num_views
        output_queries = self.future_3d_shared_output_queries.expand(batch_size, -1, -1)
        output_queries = output_queries.to(device=decoder_device, dtype=decoder_dtype)
        messenger_tokens = rearrange(
            messenger_tokens,
            "b (v q) c -> (b v) q c",
            v=num_views,
            q=self.future_3d_tokens_per_view,
        )
        output_queries = output_queries[:, None, :, :].expand(batch_size, num_views, -1, -1)
        output_queries = rearrange(output_queries, "b v q c -> (b v) q c")

        decoded_queries = self.future_3d_output_decoder(output_queries, messenger_tokens)
        return rearrange(decoded_queries, "(b v) q c -> b (v q) c", b=batch_size, v=num_views)

    def upsample_3d_queries_from_messenger_tokens(
        self,
        messenger_tokens: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        if not self.da3_query_projectors:
            raise RuntimeError("DA3 upsample projector stack is unavailable in the current alignment mode")

        refined_queries = self.refine_projected_query_grid(messenger_tokens, layer_idx)
        return self.da3_query_projectors[layer_idx](refined_queries)

    def project_query_layers(self, middle_layer_outputs: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        projected_queries = []
        target_len = self.future_3d_output_token_count
        for layer_idx, middle_layer in enumerate(middle_layer_outputs):
            _, query_tokens = self.split_middle_tokens(middle_layer)
            if query_tokens is None:
                raise RuntimeError("3D query tokens are disabled but DA3 projection was requested")
            if self.config.da3_alignment_mode == "query_decoder":
                projected_query = self.decode_3d_queries_from_messenger_tokens(query_tokens, layer_idx)
            else:
                projected_query = self.upsample_3d_queries_from_messenger_tokens(query_tokens, layer_idx)
            projected_queries.append(projected_query)

        for projected_query in projected_queries:
            if projected_query.shape[1] != target_len:
                raise ValueError(
                    f"Projected query length ({projected_query.shape[1]}) does not match target length ({target_len})"
                )
        return projected_queries

    def compute_3d_query_loss(self, middle_layer_outputs, future_images, img_masks):
        if self.da3_teacher is None or middle_layer_outputs is None:
            zero = future_images.new_zeros((), dtype=torch.float32)
            return zero, {}

        if len(middle_layer_outputs) != len(self.query_layer_indices):
            raise ValueError(
                f"Expected {len(self.query_layer_indices)} query-token layers, got {len(middle_layer_outputs)}"
            )

        teacher_device = next(self.da3_teacher.parameters()).device
        teacher_forward_s = future_images.new_zeros((), dtype=torch.float32)
        start_event = None
        end_event = None
        teacher_forward_start = None
        should_log_teacher_timing = self.config.log_da3_teacher_timing
        if should_log_teacher_timing and torch.distributed.is_available() and torch.distributed.is_initialized():
            should_log_teacher_timing = torch.distributed.get_rank() == 0

        if should_log_teacher_timing:
            if teacher_device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                teacher_forward_start = time.perf_counter()
        teacher_images, teacher_img_masks = self.prepare_da3_teacher_inputs(future_images, img_masks)
        with torch.no_grad():
            teacher_layers = self.da3_teacher(teacher_images)
        if should_log_teacher_timing:
            if teacher_device.type == "cuda":
                assert start_event is not None and end_event is not None
                end_event.record()
                end_event.synchronize()
                teacher_forward_s = torch.tensor(
                    start_event.elapsed_time(end_event) / 1000.0,
                    device=teacher_images.device,
                    dtype=torch.float32,
                )
            else:
                assert teacher_forward_start is not None
                teacher_forward_s = torch.tensor(
                    time.perf_counter() - teacher_forward_start,
                    device=teacher_images.device,
                    dtype=torch.float32,
                )

        if len(teacher_layers) != len(self.da3_teacher_layers):
            raise ValueError(
                f"Expected {len(self.da3_teacher_layers)} DA3 teacher layers, got {len(teacher_layers)}"
            )

        predicted_queries = self.project_query_layers(middle_layer_outputs)
        token_mask = self.get_3d_token_mask(teacher_img_masks, teacher_layers[0].shape[1])

        total_loss = teacher_images.new_zeros((), dtype=torch.float32)
        loss_logs = {}
        if should_log_teacher_timing:
            loss_logs["time_3d_teacher_forward_s"] = teacher_forward_s.detach()
        for idx, (pred, target, weight, teacher_layer_idx, query_layer_idx) in enumerate(
            zip(
                predicted_queries,
                teacher_layers,
                self.config.da3_layer_weights,
                self.da3_teacher_layers,
                self.query_layer_indices,
                strict=False,
            )
        ):
            target = target.to(device=pred.device, dtype=pred.dtype)
            if pred.shape[1] != target.shape[1]:
                raise ValueError(
                    f"Projected query length ({pred.shape[1]}) does not match DA3 teacher token length "
                    f"({target.shape[1]}). Check da3_tokens_per_view, da3_num_views, and teacher process resolution."
                )
            if pred.shape[-1] != target.shape[-1]:
                raise ValueError(
                    f"Projected query dim ({pred.shape[-1]}) does not match DA3 teacher dim ({target.shape[-1]}). "
                    "Check da3_query_dim and the selected teacher model."
                )
            pred_valid = pred[token_mask]
            target_valid = target[token_mask]
            if pred_valid.numel() == 0:
                continue

            pred_norm = F.normalize(pred_valid, p=2, dim=-1)
            target_norm = F.normalize(target_valid.detach(), p=2, dim=-1)
            cos_loss = (1.0 - (pred_norm * target_norm).sum(dim=-1)).mean()

            pred_ln = F.layer_norm(pred_valid, normalized_shape=(pred_valid.shape[-1],))
            target_ln = F.layer_norm(target_valid.detach(), normalized_shape=(target_valid.shape[-1],))
            mse_loss = F.mse_loss(pred_ln, target_ln)

            layer_loss = (cos_loss + mse_loss) * weight
            total_loss = total_loss + layer_loss
            loss_logs[f"loss_3d_q{query_layer_idx}_t{teacher_layer_idx}"] = layer_loss.detach()

        if predicted_queries:
            total_loss = total_loss / len(predicted_queries)
        return total_loss, loss_logs

    def reshape_projected_queries(self, projected_queries: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        reshaped_queries = {}
        for query_layer_idx, projected_query in zip(self.query_layer_indices, projected_queries, strict=False):
            reshaped_queries[f"layer_{query_layer_idx}"] = rearrange(
                projected_query,
                "b (v q) c -> b v q c",
                v=self.config.da3_num_views,
                q=self.config.da3_tokens_per_view,
            )
        return reshaped_queries

    @torch.no_grad()
    def predict_future_3d_queries(
        self,
        images,
        img_masks,
        pixel_values,
        image_grid_thw,
        lang_tokens,
        lang_masks,
        state,
    ) -> dict[str, torch.Tensor]:
        if self.future_3d_queries is None:
            raise RuntimeError("3D query tokens are disabled in the current WSABase configuration")

        bsize = state.shape[0]
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pixel_values, image_grid_thw, lang_tokens, lang_masks
        )
        middle_embs, middle_pad_masks, middle_att_masks = self.embed_middle(images[:, :, :2], img_masks)

        zero_actions = torch.zeros(
            bsize,
            self.config.chunk_size,
            self.config.max_action_dim,
            device=state.device,
            dtype=state.dtype,
        )
        zero_time = torch.zeros(bsize, device=state.device, dtype=torch.float32)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, zero_actions, zero_time)

        if (
            self.qwen3_vl_with_expert.und_expert.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            middle_embs = middle_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, middle_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, middle_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = self.build_training_attention_mask(
            pad_masks,
            att_masks,
            prefix_len=prefix_pad_masks.shape[1],
        )
        position_ids, rope_deltas = self.get_position_ids(lang_tokens, image_grid_thw, pad_masks)
        del rope_deltas

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
        outputs = self.qwen3_vl_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, middle_embs, suffix_embs],
            use_cache=False,
            collect_middle_layers=self.query_layer_indices,
        )
        (_, middle_out, _), _, middle_layer_outputs = outputs
        _, final_query_tokens = self.split_middle_tokens(middle_out)

        projected_queries = self.project_query_layers(tuple(middle_layer_outputs))
        result = self.reshape_projected_queries(projected_queries)

        if final_query_tokens is not None:
            result["final_query_tokens"] = final_query_tokens

        return result

    def embed_middle(self, images, img_masks):
        device = images[0].device
        B, N_view, T = images.shape[:3]
        features = self.get_cosmos_features(images)

        B, N_view, T = features.shape[:3]
        features = rearrange(features, 'b n t c h w -> (b n t) c h w')
        features = self.cosmos_in_proj(features)
        features = self.downsample_conv(features)
        features = rearrange(features, '(b n t) c h w -> b n t c h w', b=B, n=N_view, t=T)
        self.cosmos_feat_shape = features.shape

        B, N_view, T, _, H, W = features.shape
        embs = rearrange(features, 'b n t c h w -> b (n t h w) c', b=B, n=N_view, t=T)
        self.middle_visual_token_count = embs.shape[1]
        self.middle_visual_tokens_per_view = T * H * W
        # pad_masks = torch.ones((B, embs.shape[1]), dtype=torch.bool, device=device)
        pad_masks = torch.zeros((B, N_view, T, H, W), dtype=torch.bool, device=device)
        pad_masks[img_masks] = True
        pad_masks = rearrange(pad_masks, 'b n t h w -> b (n t h w)', b=B, n=N_view, t=T)

        att_masks = [1] + [0] * (embs.shape[1] - 1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(B, len(att_masks))
        embs, pad_masks, att_masks = self.append_future_query_tokens(embs, pad_masks, att_masks)
        return embs, pad_masks, att_masks

    def infer_middle_visual_token_count(self, images: torch.Tensor) -> int:
        """Infer how many visual middle tokens would be produced without running Cosmos."""
        _, num_views, timesteps = images.shape[:3]
        # embed_middle always resizes to 256 before the Cosmos CI8x8 tokenizer,
        # then applies the stride=scale_factor downsample projection.
        cosmos_latent_h = 256 // 8
        cosmos_latent_w = 256 // 8
        grid_h = cosmos_latent_h // int(self.config.scale_factor)
        grid_w = cosmos_latent_w // int(self.config.scale_factor)
        return int(num_views * timesteps * grid_h * grid_w)

    def embed_middle_queries_only(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.future_3d_queries is None:
            raise RuntimeError("causal query-only inference requires enabled 3D query tokens")

        query_tokens = self.future_3d_queries.expand(batch_size, -1, -1).to(device=device)
        query_pad_masks = torch.ones(
            batch_size,
            query_tokens.shape[1],
            dtype=torch.bool,
            device=device,
        )
        query_att_masks = torch.zeros(
            batch_size,
            query_tokens.shape[1],
            dtype=torch.bool,
            device=device,
        )
        if query_tokens.shape[1] > 0:
            query_att_masks[:, 0] = True

        self.middle_visual_token_count = 0
        self.middle_visual_tokens_per_view = 0
        self.middle_query_token_count = query_tokens.shape[1]
        self.middle_total_token_count = query_tokens.shape[1]
        return query_tokens, query_pad_masks, query_att_masks

    def get_position_ids_with_omitted_middle_visual_tokens(
        self,
        lang_tokens: torch.Tensor,
        image_grid_thw: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        middle_pad_masks: torch.Tensor,
        suffix_pad_masks: torch.Tensor,
        virtual_visual_token_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if virtual_visual_token_count <= 0:
            pad_masks = torch.cat([prefix_pad_masks, middle_pad_masks, suffix_pad_masks], dim=1)
            return self.get_position_ids(lang_tokens, image_grid_thw, pad_masks)

        batch_size = prefix_pad_masks.shape[0]
        virtual_visual_pad_masks = torch.ones(
            batch_size,
            virtual_visual_token_count,
            dtype=prefix_pad_masks.dtype,
            device=prefix_pad_masks.device,
        )
        position_pad_masks = torch.cat(
            [
                prefix_pad_masks,
                virtual_visual_pad_masks,
                middle_pad_masks,
                suffix_pad_masks,
            ],
            dim=1,
        )
        full_position_ids, rope_deltas = self.get_position_ids(lang_tokens, image_grid_thw, position_pad_masks)

        prefix_len = prefix_pad_masks.shape[1]
        middle_len = middle_pad_masks.shape[1]
        suffix_len = suffix_pad_masks.shape[1]
        keep_indices = torch.cat(
            [
                torch.arange(0, prefix_len, device=prefix_pad_masks.device),
                torch.arange(
                    prefix_len + virtual_visual_token_count,
                    prefix_len + virtual_visual_token_count + middle_len,
                    device=prefix_pad_masks.device,
                ),
                torch.arange(
                    prefix_len + virtual_visual_token_count + middle_len,
                    prefix_len + virtual_visual_token_count + middle_len + suffix_len,
                    device=prefix_pad_masks.device,
                ),
            ],
            dim=0,
        )
        return full_position_ids.index_select(-1, keep_indices), rope_deltas

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []
        suffix_dtype = self.qwen3_vl_with_expert.act_expert.layers[0].self_attn.q_proj.weight.dtype

        state = state.to(dtype=self.state_proj.weight.dtype)

        def state_proj_func(state):
            return self.state_proj(state)

        state_emb = self._apply_checkpoint(state_proj_func, state)
        embs.append(state_emb.to(dtype=suffix_dtype)[:, None, :])
        bsize = state_emb.shape[0]
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1]

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(
            action_proj_func,
            noisy_actions.to(dtype=self.action_in_proj.weight.dtype),
        )

        time_emb = time_emb.to(dtype=action_emb.dtype)[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        def mlp_func(action_time_emb):
            x = self.action_time_mlp_in(
                action_time_emb.to(dtype=self.action_time_mlp_in.weight.dtype)
            )
            x = F.silu(x)
            return self.action_time_mlp_out(x.to(dtype=self.action_time_mlp_out.weight.dtype))

        action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)

        embs.append(action_time_emb.to(dtype=suffix_dtype))
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def get_position_ids(self, lang_tokens, image_grid_thw, pad_masks):
        L = lang_tokens.shape[1]
        pseudo_avail_token_id = 777
        padded_lang_tokens = torch.ones_like(pad_masks).to(lang_tokens) * pseudo_avail_token_id
        padded_lang_tokens[:, :L] = lang_tokens
        attention_mask = pad_masks.to(lang_tokens)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.view(-1, 3)
        position_ids, rope_deltas = self.qwen3_vl_with_expert.und_expert.model.get_rope_index(
            padded_lang_tokens,
            image_grid_thw,
            attention_mask=attention_mask,
        )
        return position_ids, rope_deltas

    def decode_cosmos(self, features):
        b, n, t, c, h, w =self.cosmos_feat_shape
        features = rearrange(features, 'b (n t h w) c -> b n t c h w', b=b, n=n, t=t, h=h, w=w)
        features = features.mean(2)  # b n c h w
        features = rearrange(features, 'b n c h w -> (b n) c h w')

        features = self.upsample_conv(features) # dtype: torch.float32
        h_upsampled, w_upsampled = features.shape[-2:]
        features = features.permute(0, 2, 3, 1)
        features = features.reshape(b * n, -1, c)
        features = self.cosmos_out_proj(self.cosmos_out_layer_norm(features)) # dtype: torch.float32
        features = features.view(b, n, h_upsampled, w_upsampled, features.shape[-1])
        features = features.permute(0, 1, 4, 2, 3)
        return features

    def forward(
        self, images, img_masks, pixel_values, image_grid_thw, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Do a full training forward pass and compute the loss."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pixel_values, image_grid_thw, lang_tokens, lang_masks
        )
        middle_embs, middle_pad_masks, middle_att_masks = self.embed_middle(
            images[:, :, :2], img_masks,  # remove the future observation
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, time)

        if (
            self.qwen3_vl_with_expert.und_expert.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            middle_embs = middle_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, middle_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, middle_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = self.build_training_attention_mask(
            pad_masks,
            att_masks,
            prefix_len=prefix_pad_masks.shape[1],
        )
        position_ids, rope_deltas = self.get_position_ids(lang_tokens, image_grid_thw, pad_masks)

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
        collect_middle_layers = self.query_layer_indices if self.da3_teacher is not None else None

        def forward_func(prefix_embs, middle_embs, suffix_embs, att_2d_masks_4d, position_ids):
            outputs = self.qwen3_vl_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, middle_embs, suffix_embs],
                use_cache=False,
                collect_middle_layers=collect_middle_layers,
            )
            if collect_middle_layers is None:
                (_, middle_out, suffix_out), _ = outputs
                return middle_out, suffix_out
            (_, middle_out, suffix_out), _, middle_layer_outputs = outputs
            return (middle_out, suffix_out, *middle_layer_outputs)

        forward_outputs = self._apply_checkpoint(
            forward_func, prefix_embs, middle_embs, suffix_embs, att_2d_masks_4d, position_ids
        )
        middle_out, suffix_out, *middle_layer_outputs = forward_outputs

        if float(self.config.lambda_gen) > 0.0:
            middle_visual_out, _ = self.split_middle_tokens(middle_out)

            def cosmos_out_func(middle_out):
                return self.decode_cosmos(middle_out)

            pred_cosmos_features = self._apply_checkpoint(cosmos_out_func, middle_visual_out.to(dtype=torch.float32))
            future_embs = self.get_cosmos_features(images[:, :, 2])
            loss_gen = F.mse_loss(pred_cosmos_features[img_masks], future_embs.to(dtype=torch.float32)[img_masks])
        else:
            loss_gen = middle_out.new_zeros((), dtype=torch.float32)
        loss_3d, loss_3d_logs = self.compute_3d_query_loss(tuple(middle_layer_outputs), images[:, :, 2], img_masks)

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        loss_action = F.mse_loss(u_t, v_t, reduction="none")

        aux_losses = {
            "loss_gen": loss_gen,
            "loss_3d": loss_3d,
        }
        aux_losses.update(loss_3d_logs)
        return loss_action, aux_losses

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self, images, img_masks, pixel_values, image_grid_thw, lang_tokens, lang_masks, state, noise=None, num_steps=None, decode_image=False
    ) -> tuple[Tensor, Tensor | None]:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)

        if self.uses_causal_attention():
            return self.sample_actions_causal(
                images,
                img_masks,
                pixel_values,
                image_grid_thw,
                lang_tokens,
                lang_masks,
                state,
                noise=noise,
                num_steps=num_steps,
                decode_image=decode_image,
            )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pixel_values, image_grid_thw, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids, rope_deltas = self.get_position_ids(lang_tokens, image_grid_thw, prefix_pad_masks)

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        with self._temporary_attention_implementations(
            und_expert_impl="eager",
            gen_expert_impl="eager",
            act_expert_impl="eager",
        ):
            _, past_key_values = self.qwen3_vl_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None, None],
                use_cache=True,
            )
            max_prefix_position_ids = prefix_position_ids.max(dim=-1, keepdim=True).values

            middle_embs, middle_pad_masks, middle_att_masks = self.embed_middle(
                images[:, :, :2], img_masks,
            )

            middle_len = middle_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, middle_len, prefix_len)
            middle_att_2d_masks = make_att_2d_masks(middle_pad_masks, middle_att_masks)
            middle_att_2d_masks = self.apply_view_aware_query_attention(middle_att_2d_masks, prefix_len=0)
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, middle_att_2d_masks], dim=2)

            middle_position_ids = torch.arange(1, middle_len + 1).repeat(3, 1, 1).to(max_prefix_position_ids) + max_prefix_position_ids

            full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

            (_, middle_out, _), past_key_values = self.qwen3_vl_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=middle_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, middle_embs, None],
                use_cache=True,
            )

            max_position_ids = middle_position_ids.max(dim=-1, keepdim=True).values
            curr_pad_masks = torch.cat([prefix_pad_masks, middle_pad_masks], dim=1)
            suffix_att_2d_masks_4d, suffix_position_ids = self._prepare_suffix_denoise_cache(
                curr_pad_masks,
                max_position_ids,
            )

            dt = -1.0 / num_steps
            dt = torch.tensor(dt, dtype=torch.float32, device=device)

            x_t = noise
            time = torch.tensor(1.0, dtype=torch.float32, device=device)
            while time >= -dt / 2:
                expanded_time = time.expand(bsize)
                v_t = self.denoise_step(
                    state,
                    past_key_values,
                    suffix_att_2d_masks_4d,
                    suffix_position_ids,
                    x_t.to(dtype),
                    expanded_time.to(dtype),
                )
                x_t = x_t + dt * v_t
                time += dt

            if decode_image:
                def cosmos_out_func(middle_out):
                    return self.decode_cosmos(middle_out)
                middle_visual_out, _ = self.split_middle_tokens(middle_out)
                decode_dtype = torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float32
                pred_cosmos_features = self._apply_checkpoint(
                    cosmos_out_func,
                    middle_visual_out.to(dtype=decode_dtype),
                )
                pred_cosmos_features = pred_cosmos_features.squeeze(0)
                recon_images = self.cosmos.decode(pred_cosmos_features.squeeze(0))
            else:
                recon_images = None

            return x_t, recon_images

    @torch.no_grad()
    def sample_actions_causal(
        self,
        images,
        img_masks,
        pixel_values,
        image_grid_thw,
        lang_tokens,
        lang_masks,
        state,
        noise,
        num_steps,
        decode_image=False,
    ) -> tuple[Tensor, Tensor | None]:
        """Exact inference path for the causal attention ablation.

        The 3D scene/query block is allowed to read the current denoising action
        block, so middle tokens cannot be cached independently of suffix tokens.
        This path recomputes the full prefix/middle/suffix transformer pass at each
        denoising step. The default path remains cache based.
        """
        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pixel_values, image_grid_thw, lang_tokens, lang_masks
        )
        causal_action_only_fast_path = (
            not decode_image
            and bool(getattr(self, "omit_visual_tokens_in_causal_inference", True))
        )
        middle_images = images[:, :, :2]
        virtual_visual_token_count = 0
        if causal_action_only_fast_path:
            virtual_visual_token_count = self.infer_middle_visual_token_count(middle_images)
            middle_embs, middle_pad_masks, middle_att_masks = self.embed_middle_queries_only(
                batch_size=bsize,
                device=device,
            )
        else:
            middle_embs, middle_pad_masks, middle_att_masks = self.embed_middle(
                middle_images, img_masks,
            )

        zero_actions = torch.zeros(
            bsize,
            self.config.chunk_size,
            self.config.max_action_dim,
            device=device,
            dtype=dtype,
        )
        zero_time = torch.zeros(bsize, device=device, dtype=torch.float32)
        _, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, zero_actions, zero_time)

        if (
            self.qwen3_vl_with_expert.und_expert.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            middle_embs = middle_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, middle_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, middle_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = self.build_training_attention_mask(
            pad_masks,
            att_masks,
            prefix_len=prefix_pad_masks.shape[1],
        )
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
        if causal_action_only_fast_path:
            position_ids, rope_deltas = self.get_position_ids_with_omitted_middle_visual_tokens(
                lang_tokens,
                image_grid_thw,
                prefix_pad_masks,
                middle_pad_masks,
                suffix_pad_masks,
                virtual_visual_token_count,
            )
        else:
            position_ids, rope_deltas = self.get_position_ids(lang_tokens, image_grid_thw, pad_masks)
        del rope_deltas

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        last_middle_out = None
        with self._temporary_attention_implementations(
            und_expert_impl="eager",
            gen_expert_impl="eager",
            act_expert_impl="eager",
        ):
            while time >= -dt / 2:
                expanded_time = time.expand(bsize)
                suffix_embs, _, _ = self.embed_suffix(state, x_t.to(dtype), expanded_time.to(dtype))
                if (
                    self.qwen3_vl_with_expert.und_expert.language_model.layers[0].self_attn.q_proj.weight.dtype
                    == torch.bfloat16
                ):
                    suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

                outputs_embeds, _ = self.qwen3_vl_with_expert.forward(
                    attention_mask=att_2d_masks_4d,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, middle_embs, suffix_embs],
                    use_cache=False,
                )
                _, middle_out, suffix_out = outputs_embeds
                last_middle_out = middle_out
                suffix_out = suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)
                v_t = self.action_out_proj(suffix_out)
                x_t = x_t + dt * v_t
                time += dt

        if decode_image:
            if last_middle_out is None:
                raise RuntimeError("causal inference produced no middle tokens to decode.")

            def cosmos_out_func(middle_out):
                return self.decode_cosmos(middle_out)

            middle_visual_out, _ = self.split_middle_tokens(last_middle_out)
            decode_dtype = torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float32
            pred_cosmos_features = self._apply_checkpoint(
                cosmos_out_func,
                middle_visual_out.to(dtype=decode_dtype),
            )
            pred_cosmos_features = pred_cosmos_features.squeeze(0)
            recon_images = self.cosmos.decode(pred_cosmos_features.squeeze(0))
        else:
            recon_images = None

        return x_t, recon_images

    def denoise_step(
        self,
        state,
        past_key_values,
        suffix_att_2d_masks_4d,
        suffix_position_ids,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, _, _ = self.embed_suffix(state, x_t, timestep)
        outputs_embeds, _ = self.qwen3_vl_with_expert.forward(
            attention_mask=suffix_att_2d_masks_4d,
            position_ids=suffix_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, None, suffix_embs],
            use_cache=False,
        )

        suffix_out = outputs_embeds[2]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class WSABasePolicy(PreTrainedPolicy):
    """WSABase policy with Qwen3-VL perception, expert action heads, and optional 3D alignment."""

    config_class = WSABaseConfig
    name = WSA_BASE
    _save_excluded_prefixes = (
        "model.cosmos.",
        "model.da3_teacher.",
    )

    def __init__(
        self,
        config: WSABaseConfig,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = WSABaseModel(config)

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self.reset()

    def __str__(self) -> str:
        lines = []

        # ---- basic info ----
        lines.append("=" * 60)
        lines.append(f"Policy: {self.__class__.__name__}")
        lines.append(f"Attention mask mode: {getattr(self.config, 'attention_mask_mode', 'default')}")
        lines.append("")

        # ---- parameter counts ----
        num_total_params = sum(p.numel() for p in self.parameters())
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        num_und = sum(p.numel() for p in self.model.qwen3_vl_with_expert.und_expert.parameters())
        num_gen = sum(p.numel() for p in self.model.qwen3_vl_with_expert.gen_expert.parameters())
        num_act = sum(p.numel() for p in self.model.qwen3_vl_with_expert.act_expert.parameters())

        lines.append("Parameter statistics:")
        lines.append(f"  - Total params        : {num_total_params} ({format_big_number(num_total_params)})")
        lines.append(f"  - Trainable params    : {num_trainable_params} ({format_big_number(num_trainable_params)})")
        lines.append(f"  - Und params          : {num_und} ({format_big_number(num_und)})")
        lines.append(f"  - Gen params          : {num_gen} ({format_big_number(num_gen)})")
        lines.append(f"  - Act params          : {num_act} ({format_big_number(num_act)})")
        if self.config.lora_enabled:
            num_lora_params = sum(
                p.numel()
                for name, p in self.named_parameters()
                if is_lora_parameter_name(name.rsplit(".", 1)[-1])
            )
            lines.append(f"  - LoRA params         : {num_lora_params} ({format_big_number(num_lora_params)})")
            lines.append(f"  - LoRA experts        : {', '.join(self.config.lora_modules)}")
            lines.append(
                f"  - LoRA targets        : {', '.join(self.config.lora_targets)} | drop={self.config.lora_dropout}"
            )
            lines.append(
                "  - LoRA rank/alpha     : "
                + ", ".join(
                    f"{module}(r={self.config.get_lora_rank_for(module)}, a={self.config.get_lora_alpha_for(module)})"
                    for module in self.config.lora_modules
                )
            )

        if self.model.da3_teacher is not None:
            num_da3 = sum(p.numel() for p in self.model.da3_teacher.backbone.parameters())
            num_da3_trainable = sum(
                p.numel() for p in self.model.da3_teacher.backbone.parameters() if p.requires_grad
            )
            lines.append("")
            lines.append("DA3 teacher:")
            lines.append(f"  - Source              : {self.config.da3_model_path_or_name}")
            lines.append(f"  - Variant             : {self.model.da3_teacher.variant}")
            lines.append(f"  - Teacher layers      : {self.model.da3_teacher.teacher_layers}")
            lines.append(f"  - Feature dim         : {self.model.da3_teacher.feature_dim}")
            lines.append(f"  - Params              : {num_da3} ({format_big_number(num_da3)})")
            lines.append(
                f"  - Trainable params    : {num_da3_trainable} ({format_big_number(num_da3_trainable)})"
            )
        elif self.config.enable_3d_queries and self.config.lambda_3d > 0:
            lines.append("")
            lines.append("DA3 teacher:")
            lines.append("  - Status              : requested but not initialized")
        else:
            lines.append("")
            lines.append("DA3 teacher:")
            lines.append("  - Status              : disabled")

        if self.config.enable_3d_queries:
            query_grid_h, query_grid_w = self.model.future_3d_query_grid
            target_grid_h, target_grid_w = self.model.da3_target_grid
            lines.append("")
            lines.append("Future 3D alignment:")
            lines.append(f"  - Alignment mode      : {self.config.da3_alignment_mode}")
            lines.append(f"  - Messenger tokens    : {self.config.num_3d_query_tokens}")
            lines.append(f"  - Views               : {self.config.da3_num_views}")
            lines.append(
                f"  - Messenger tokens/view : {self.model.future_3d_tokens_per_view} (~{query_grid_h} x {query_grid_w} layout)"
            )
            if self.config.da3_alignment_mode == "query_decoder":
                lines.append(
                    f"  - Shared queries/view : {self.model.future_3d_output_queries_per_view}"
                )
                lines.append(f"  - Total decoded tokens: {self.model.future_3d_output_token_count}")
                lines.append("  - Resampler block     : cross-attn + SiLU FFN (fixed)")
                lines.append(
                    f"  - Output tokens/view  : {self.config.da3_tokens_per_view} "
                    f"(~{target_grid_h} x {target_grid_w} equivalent)"
                )
            else:
                lines.append(
                    f"  - Target grid / view  : {target_grid_h} x {target_grid_w}"
                )

        lines.append("=" * 60)

        return "\n".join(lines)

    def to(self, *args, **kwargs):
        cosmos_device = getattr(self.config, "cosmos_device", None)
        if cosmos_device is None:
            super().to(*args, **kwargs)
        else:
            cosmos_module = self.model.cosmos
            self.model.cosmos = nn.Identity()
            try:
                super().to(*args, **kwargs)
            finally:
                self.model.cosmos = cosmos_module
        target_cosmos_device = cosmos_device or next(self.model.cosmos.parameters()).device
        if str(target_cosmos_device).startswith("cpu"):
            cosmos_dtype = torch.float32
        else:
            cosmos_dtype = torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float32
        if cosmos_device is None:
            self.model.cosmos.to(cosmos_dtype)
        else:
            self.model.cosmos.to(device=cosmos_device, dtype=cosmos_dtype)
        # Keep the regression head in fp32 for the existing bf16 training/inference path.
        self.model.action_out_proj.to(torch.float32)
        return self

    def get_optim_params(self) -> dict:
        return (param for param in self.parameters() if param.requires_grad)

    def merge_lora_weights_(self):
        if not self.config.lora_enabled:
            return self

        num_merged = merge_and_unload_lora_modules_(self.model)
        logging.info("Merged and unloaded %d WSABase LoRA linear modules.", num_merged)
        self.config.lora_modules = ()
        self.model.lora_module_counts = {}
        self.model.set_requires_grad()
        return self

    def get_merged_state_dict_to_save(self) -> dict[str, Tensor]:
        merged_state_dict = dict(self.get_state_dict_to_save())
        if not self.config.lora_enabled:
            return merged_state_dict

        for module_name, module in self.named_modules():
            if not isinstance(module, LoRALinear):
                continue

            weight_key = f"{module_name}.weight"
            lora_a_key = f"{module_name}.lora_A"
            lora_b_key = f"{module_name}.lora_B"
            if weight_key not in merged_state_dict:
                continue
            if lora_a_key not in merged_state_dict or lora_b_key not in merged_state_dict:
                raise KeyError(
                    f"Missing LoRA tensors for merged export of module '{module_name}'. "
                    f"Expected keys '{lora_a_key}' and '{lora_b_key}'."
                )

            delta = torch.matmul(merged_state_dict[lora_b_key], merged_state_dict[lora_a_key]) * module.scaling
            merged_state_dict[weight_key] = merged_state_dict[weight_key] + delta.to(
                dtype=merged_state_dict[weight_key].dtype,
                device=merged_state_dict[weight_key].device,
            )
            del merged_state_dict[lora_a_key]
            del merged_state_dict[lora_b_key]

        return merged_state_dict

    def save_merged_pretrained(self, save_directory: str | Path) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        merged_config = copy.deepcopy(self.config)
        merged_config.lora_modules = ()
        merged_config.pretrained_path = None

        self._save_pretrained_artifacts(
            save_directory,
            config=merged_config,
            state_dict=self.get_merged_state_dict_to_save(),
        )

    def _get_loaded_pretrained_source_config(self) -> WSABaseConfig | None:
        source_config = getattr(self, "_loaded_pretrained_source_config", None)
        return source_config if isinstance(source_config, WSABaseConfig) else None

    def _validate_lora_loading_compatibility(self) -> bool:
        source_config = self._get_loaded_pretrained_source_config()
        if source_config is None:
            return False

        if source_config.lora_enabled and not self.config.lora_enabled:
            raise ValueError(
                "The source checkpoint was saved with LoRA adapters enabled, but the current WSABase config "
                "does not enable LoRA. Load it with matching LoRA settings or export a merged checkpoint first."
            )

        if source_config.lora_enabled and self.config.lora_enabled:
            mismatches: list[str] = []
            training_only_mismatches: list[str] = []
            if source_config.lora_modules != self.config.lora_modules:
                mismatches.append(
                    f"lora_modules source={source_config.lora_modules} current={self.config.lora_modules}"
                )
            if source_config.lora_targets != self.config.lora_targets:
                mismatches.append(
                    f"lora_targets source={source_config.lora_targets} current={self.config.lora_targets}"
                )
            for module_name in self.config.lora_modules:
                source_rank, source_alpha = source_config.get_lora_hparams_for(module_name)
                current_rank, current_alpha = self.config.get_lora_hparams_for(module_name)
                if source_rank != current_rank:
                    mismatches.append(
                        f"{module_name}.lora_rank source={source_rank} current={current_rank}"
                    )
                if not math.isclose(source_alpha, current_alpha, rel_tol=0.0, abs_tol=1e-12):
                    mismatches.append(
                        f"{module_name}.lora_alpha source={source_alpha} current={current_alpha}"
                    )
            if mismatches:
                raise ValueError(
                    "The source LoRA checkpoint is incompatible with the current WSABase LoRA config: "
                    + "; ".join(mismatches)
                )

            if source_config.lora_unselected_mode != self.config.lora_unselected_mode:
                training_only_mismatches.append(
                    "lora_unselected_mode "
                    f"source={source_config.lora_unselected_mode} current={self.config.lora_unselected_mode}"
                )
            if not math.isclose(source_config.lora_dropout, self.config.lora_dropout, rel_tol=0.0, abs_tol=1e-12):
                training_only_mismatches.append(
                    f"lora_dropout source={source_config.lora_dropout} current={self.config.lora_dropout}"
                )
            if training_only_mismatches:
                logging.warning(
                    "Loading WSABase LoRA checkpoint with changed training-only settings: %s",
                    "; ".join(training_only_mismatches),
                )

        return (not source_config.lora_enabled) and self.config.lora_enabled

    def get_state_dict_to_save(self) -> dict[str, Tensor]:
        state_dict = self.state_dict()
        filtered_state_dict = {
            key: value
            for key, value in state_dict.items()
            if not any(key.startswith(prefix) for prefix in self._save_excluded_prefixes)
        }
        dropped_params = sum(
            value.numel()
            for key, value in state_dict.items()
            if any(key.startswith(prefix) for prefix in self._save_excluded_prefixes)
        )
        logging.info(
            "Saving WSABase checkpoint without frozen external modules: "
            f"dropped {format_big_number(dropped_params)} parameters from "
            f"{', '.join(self._save_excluded_prefixes)}"
        )
        return filtered_state_dict

    def classify_model_loading_keys(
        self, missing_keys: list[str], unexpected_keys: list[str]
    ) -> tuple[list[str], list[str], list[str]]:
        ignored_missing_prefixes = (
            "model.cosmos.",
            "model.da3_teacher.",
            "model.da3_query_projectors.",
            "model.future_3d_layer_input_norms.",
            "model.future_3d_shared_refine_trunk.",
            "model.future_3d_messenger_norms.",
            "model.future_3d_output_decoder.",
        )
        ignored_missing_exact = {
            "model.future_3d_queries",
            "model.future_3d_shared_output_queries",
        }
        ignored_unexpected_prefixes = ignored_missing_prefixes
        ignored_unexpected_exact = {
            *ignored_missing_exact,
            "model.future_3d_output_queries",
        }
        expected_lora_missing = set()
        if self._validate_lora_loading_compatibility():
            expected_lora_missing = {
                key
                for key in missing_keys
                if key.endswith(".lora_A") or key.endswith(".lora_B")
            }

        expected_missing = [
            key
            for key in missing_keys
            if key in ignored_missing_exact
            or any(key.startswith(prefix) for prefix in ignored_missing_prefixes)
            or key in expected_lora_missing
        ]
        filtered_missing = [key for key in missing_keys if key not in expected_missing]
        filtered_unexpected = [
            key
            for key in unexpected_keys
            if key not in ignored_unexpected_exact
            and not any(key.startswith(prefix) for prefix in ignored_unexpected_prefixes)
        ]
        return filtered_missing, filtered_unexpected, expected_missing

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.
        """
        images = []
        img_masks = []

        for img_idx in range(3):
            img = batch[f"{OBS_IMAGES}.image{img_idx}"]
            mask = batch[f"{OBS_IMAGES}.image{img_idx}_mask"]

            images.append(img)
            img_masks.append(mask)

        images = torch.stack(images, dim=1)  # B, N_view, T, C, H, W
        img_masks = torch.stack(img_masks, dim=1)

        return images, img_masks

    def prepare_state(self, batch):
        """Pad state"""
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def prepare_gen_features(self, batch):
        images = torch.stack([batch[f"{OBS_IMAGES}.image{i}"] for i in range(3)], dim=1)  # B, N_view, T, C, H, W
        B, N_view, T = images.shape[:3]
        images = rearrange(images, 'b n t c h w -> (b n t) c h w')
        images = F.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)
        images = images * 2 - 1  # [-1, 1]
        features = self.model.cosmos.encode(images)
        features = rearrange(features, '(b n t) c h w -> b n t c h w', b=B, n=N_view, t=T)
        return features

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions, _ = self.predict_action_chunk(batch)
            actions = actions[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], decode_image=False) -> tuple[Tensor, Tensor | None]:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        pixel_values = batch[f"{OBS_PREFIX}pixel_values"]
        image_grid_thw = batch[f"{OBS_PREFIX}image_grid_thw"]
        lang_tokens = batch[f"{OBS_PREFIX}input_ids"]
        lang_masks = batch[f"{OBS_PREFIX}attention_mask"]
        state = self.prepare_state(batch)

        images, img_masks = self._preprocess_images(batch)

        # Sample actions using the model
        actions, recon_images = self.model.sample_actions(images, img_masks, pixel_values, image_grid_thw, lang_tokens, lang_masks, state, decode_image = decode_image)

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions, recon_images

    @torch.no_grad()
    def predict_future_3d_queries(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Export projected future 3D query tokens for the selected generation-expert layers."""
        self.eval()

        pixel_values = batch[f"{OBS_PREFIX}pixel_values"]
        image_grid_thw = batch[f"{OBS_PREFIX}image_grid_thw"]
        lang_tokens = batch[f"{OBS_PREFIX}input_ids"]
        lang_masks = batch[f"{OBS_PREFIX}attention_mask"]
        state = self.prepare_state(batch)
        images, img_masks = self._preprocess_images(batch)

        return self.model.predict_future_3d_queries(
            images,
            img_masks,
            pixel_values,
            image_grid_thw,
            lang_tokens,
            lang_masks,
            state,
        )

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training."""

        # Prepare inputs
        pixel_values = batch[f"{OBS_PREFIX}pixel_values"]
        image_grid_thw = batch[f"{OBS_PREFIX}image_grid_thw"]
        lang_tokens = batch[f"{OBS_PREFIX}input_ids"]
        lang_masks = batch[f"{OBS_PREFIX}attention_mask"]

        images, img_masks = self._preprocess_images(batch)

        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)

        # Compute loss
        losses_action, aux_losses = self.model.forward(
            images,
            img_masks,
            pixel_values,
            image_grid_thw,
            lang_tokens,
            lang_masks,
            state,
            actions,
        )

        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses_action = losses_action[:, :, :original_action_dim]
        action_loss_mask = batch.get(SAMPLE_ACTION_LOSS_MASK)
        if action_loss_mask is None:
            action_loss_mask = torch.ones(
                losses_action.shape[0],
                dtype=torch.bool,
                device=losses_action.device,
            )
        else:
            action_loss_mask = action_loss_mask.to(losses_action.device)
            if action_loss_mask.ndim > 1:
                action_loss_mask = action_loss_mask.squeeze(-1)
            action_loss_mask = action_loss_mask > 0.5

        if action_loss_mask.any():
            sample_losses_action = losses_action[action_loss_mask]
            if self.config.mask_action_dim_padding_loss:
                valid_action_dim = min(int(self.config.action_loss_valid_dim), original_action_dim)
                valid_losses_action = sample_losses_action[:, :, :valid_action_dim]
                loss_action = valid_losses_action.mean()
                loss_action_by_dim_tensor = sample_losses_action.new_zeros(original_action_dim)
                loss_action_by_dim_tensor[:valid_action_dim] = valid_losses_action.mean(dim=[0, 1])
                loss_action_by_dim = loss_action_by_dim_tensor.detach().cpu().numpy().tolist()
            else:
                loss_action = sample_losses_action.mean()
                loss_action_by_dim = sample_losses_action.mean(dim=[0, 1]).detach().cpu().numpy().tolist()
        else:
            loss_action = losses_action.new_zeros(())
            loss_action_by_dim = [0.0] * original_action_dim
        loss_gen = aux_losses["loss_gen"]
        loss_3d = aux_losses["loss_3d"]

        loss = loss_action + self.config.lambda_gen * loss_gen + self.config.lambda_3d * loss_3d

        loss_dict = {
            "loss": loss.item(),
            "loss_action": loss_action.item(),
            "loss_gen": loss_gen.item(),
            "loss_3d": loss_3d.item(),
        }
        for key, value in aux_losses.items():
            if key in {"loss_gen", "loss_3d"}:
                continue
            loss_dict[key] = float(value.item())

        loss_dict.update({
            f"loss_action_dim{i}": loss_action_by_dim[i] for i in range(original_action_dim)
        })

        return loss, loss_dict
if __name__ == "__main__":
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE, ACTION
    from lerobot.policies.WSA_Base.transform_wsa_base import Qwen3_VLProcessorTransformFn
    from pprint import pp
    torch.manual_seed(0)
    device = torch.device("cuda")

    processor = Qwen3_VLProcessorTransformFn()

    cfg = WSABaseConfig()
    cfg.qwen3_vl_variant="qwen3_vl_28l"
    cfg.action_expert_variant="qwen3_28l"
    cfg.freeze_vision_encoder=True
    dtype = torch.float32 if cfg.dtype == 'float32' else torch.bfloat16

    model = WSABasePolicy(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}  ({total_params / 1e9:.2f}B)")
    print(f"Trainable parameters: {trainable_params:,}  ({trainable_params / 1e9:.2f}B)")
    print(f"Und params: {sum(p.numel() for p in model.model.qwen3_vl_with_expert.und_expert.parameters()) / 1e9:.2f}B")
    print(f"Gen params: {sum(p.numel() for p in model.model.qwen3_vl_with_expert.gen_expert.parameters()) / 1e9:.2f}B")
    print(f"Act params: {sum(p.numel() for p in model.model.qwen3_vl_with_expert.act_expert.parameters()) / 1e9:.2f}B")

    B = 2
    samples = [{
        f"{OBS_IMAGES}.image0": torch.rand((3, 3, 224, 224)),
        f"{OBS_IMAGES}.image1": torch.rand((3, 3, 224, 224)),
        f"{OBS_IMAGES}.image2": torch.rand((3, 3, 224, 224)),
        f"{OBS_IMAGES}.image0_mask": torch.tensor(True).cuda(),
        f"{OBS_IMAGES}.image1_mask": torch.tensor(True).cuda(),
        f"{OBS_IMAGES}.image2_mask": torch.tensor(True).cuda(),
        "task": f"This is test sample {i}.",
        OBS_STATE: torch.rand((14, )),
        ACTION: torch.rand((50, 14)),
    } for i in range(B)]
    samples = [processor(sample) for sample in samples]
    inputs = {}
    for key in samples[0].keys():
        if key != "task":
            inputs[key] = torch.stack([sample[key] for sample in samples], dim=0).to(device=device)
    loss, loss_dict = model.forward(inputs)
    pp(loss)
    pp(loss_dict)
