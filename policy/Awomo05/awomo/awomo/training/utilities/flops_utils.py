# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any, Callable, Optional


def calculate_mfu(tflops, world_size, time_seconds, reference_mfu=1979.0):
    mfu = tflops / (world_size * time_seconds)
    mfu = mfu / reference_mfu
    return mfu * 100


def gpt3_flops(config, gbs=1, seq_len=None):

    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    hs = config.hidden_size
    layers = config.num_hidden_layers
    vocab_size = config.vocab_size
    causal_self_attn = True

    return (24 * gbs * seq_len * hs * hs + 4 * gbs * seq_len * seq_len * hs * (0.5 if causal_self_attn else 1)) * (
        3 * layers
    ) + (6 * gbs * seq_len * hs * vocab_size)


def llama2_flops(config, gbs=1, seq_len=None):

    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    layers = config.num_hidden_layers
    hs = config.hidden_size
    attention_heads = config.num_attention_heads
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else attention_heads
    ffn_hs = config.intermediate_size
    vocab_size = config.vocab_size
    causal_self_attn = True

    return (
        gbs
        * seq_len
        * layers
        * hs
        * hs
        * (
            12
            + (12 * query_groups / attention_heads)
            + (18 * ffn_hs / hs)
            + (12 * seq_len / hs) * (0.5 if causal_self_attn else 1)
            + (6 * vocab_size / (layers * hs))
        )
    )


def llama3_flops(config, gbs=1, seq_len=None):

    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    layers = config.num_hidden_layers
    hs = config.hidden_size
    attention_heads = config.num_attention_heads
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else attention_heads
    ffn_hs = config.intermediate_size
    vocab_size = config.vocab_size
    causal_self_attn = True

    return (
        gbs
        * seq_len
        * layers
        * hs
        * hs
        * (
            12
            + (12 * query_groups / attention_heads)
            + (18 * ffn_hs / hs)
            + (12 * seq_len / hs) * (0.5 if causal_self_attn else 1)
            + (6 * vocab_size / (layers * hs))
        )
    )


def nemotron_flops(config, gbs=1, seq_len=None):

    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    layers = config.num_hidden_layers
    hs = config.hidden_size
    attention_heads = config.num_attention_heads
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else attention_heads
    ffn_hs = config.intermediate_size
    vocab_size = config.vocab_size
    causal_self_attn = True

    return (
        gbs
        * seq_len
        * layers
        * hs
        * hs
        * (
            12
            + (12 * query_groups / attention_heads)
            + (12 * ffn_hs / hs)
            + (12 * seq_len / hs) * (0.5 if causal_self_attn else 1)
            + (6 * vocab_size / (layers * hs))
        )
    )


def mixtral_flops(config, gbs=1, seq_len=None):

    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    layers = config.num_hidden_layers
    hs = config.hidden_size
    attention_heads = config.num_attention_heads
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else attention_heads
    ffn_hs = config.intermediate_size
    vocab_size = config.vocab_size
    moe_router_topk = config.num_experts_per_tok if hasattr(config, "num_experts_per_tok") else 2
    causal_self_attn = True

    return (
        gbs
        * seq_len
        * layers
        * hs
        * hs
        * (
            12
            + (12 * query_groups / attention_heads)
            + (18 * moe_router_topk * ffn_hs / hs)
            + (12 * seq_len / hs) * (0.5 if causal_self_attn else 1)
            + (6 * vocab_size / (layers * hs))
        )
    )


def qwen3_flops(config, gbs=1, seq_len=None):

    if hasattr(config, "text_config") and not hasattr(config, "num_hidden_layers"):
        config = config.text_config

    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    layers = config.num_hidden_layers
    hs = config.hidden_size
    attention_heads = config.num_attention_heads
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else attention_heads
    vocab_size = config.vocab_size
    head_dim = config.head_dim if hasattr(config, "head_dim") else (hs // attention_heads)
    query_projection_to_hidden_size_ratio = (head_dim * attention_heads) / hs

    moe_router_topk = config.num_experts_per_tok if hasattr(config, "num_experts_per_tok") else 1
    moe_ffn_hidden_size = (
        config.moe_intermediate_size if hasattr(config, "moe_intermediate_size") else config.intermediate_size
    )

    causal_self_attn = True
    hidden_size = hs
    gated_linear_multiplier = 2

    attention_flops = (
        3
        * 2
        * gbs
        * layers
        * seq_len
        * hidden_size
        * hidden_size
        * query_projection_to_hidden_size_ratio
        * (
            (query_groups / attention_heads * 2 + 1)
            + (seq_len / hidden_size * 2 * (0.5 if causal_self_attn else 1))
            + 1
        )
    )

    mlp_flops = (
        3
        * 2
        * gbs
        * layers
        * seq_len
        * hidden_size
        * (1 + gated_linear_multiplier)
        * (moe_ffn_hidden_size * moe_router_topk)
    )

    vocab_flops = 3 * 2 * gbs * seq_len * hidden_size * vocab_size

    return attention_flops + mlp_flops + vocab_flops


def bert_flops(config, gbs=1, seq_len=None):

    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 512

    layers = config.num_hidden_layers
    hs = config.hidden_size
    vocab_size = config.vocab_size

    return 72 * gbs * layers * seq_len * hs * hs * (1 + (seq_len / (6 * hs)) + (vocab_size / (12 * hs * layers)))


def transformer_flops(config, gbs=1, seq_len=None):
    batch_size = gbs
    if seq_len is None:
        seq_length = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048
    else:
        seq_length = seq_len

    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    ffn_hidden_size = config.intermediate_size
    vocab_size = config.vocab_size

    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else num_attention_heads
    causal_self_attn = True
    moe_router_topk = config.num_experts_per_tok if hasattr(config, "num_experts_per_tok") else 0
    kv_channels = hidden_size // num_attention_heads

    query_projection_size = kv_channels * num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / hidden_size

    if moe_router_topk == 0:
        num_dense_layers = num_layers
        num_moe_layers = 0
        num_experts_routed_to = 0
    else:
        # Simplified MoE handling - assuming uniform distribution of MoE layers
        # This can be expanded based on NeMo's actual MoE implementation
        num_moe_layers = num_layers // 2  # Simplified assumption
        num_dense_layers = num_layers - num_moe_layers
        num_experts_routed_to = moe_router_topk

    gated_linear_multiplier = 1

    expansion_factor = 3 * 2 * 2
    if not causal_self_attn:
        attention_component = (
            1
            + (query_groups / num_attention_heads)
            + (seq_length / hidden_size)
        ) * query_projection_to_hidden_size_ratio
    else:
        attention_component = (
            1
            + (query_groups / num_attention_heads)
            + (seq_length / hidden_size / 2)
        ) * query_projection_to_hidden_size_ratio

    total_flops = (
        expansion_factor
        * batch_size
        * seq_length
        * num_layers
        * hidden_size
        * hidden_size
        * (
            attention_component
            + (
                (
                    (ffn_hidden_size * num_dense_layers)
                    +
                    (
                        (
                            ffn_hidden_size * num_experts_routed_to
                        )
                        * num_moe_layers
                    )
                )
                * gated_linear_multiplier
                / (num_layers * hidden_size)
            )
            + (vocab_size / (2 * num_layers * hidden_size))
        )
    )

    return total_flops


def clip_vit_l_flops(config):

    if config.img_seq_len is None:
        config.img_seq_len = (config.img_h * config.img_w) / (
            config.patch_dim * config.patch_dim
        ) + config.class_token_len
    return config.gbs * config.layers * config.hs * config.hs * config.img_seq_len * (
        24 + (4 * config.img_seq_len / config.hs)
    ) + (2 * config.gbs * config.hs * config.in_channels * config.img_h * config.img_w)


def neva_projection_flops(config):

    if "mlp" in config.projector_type:
        return 6 * config.gbs * config.img_seq_len * config.ffn_hs * (config.inp_s + config.hs)
    elif config.projector_type == "affine":
        return 6 * config.gbs * config.img_seq_len * config.inp_s * config.hs
    else:
        raise ValueError(
            f"NeVA Projections FLOPs calculator only supports 'mlp', 'mcore_mlp'"
            f" or 'affine' projector_type but found {config.projector_type}"
        )


def flux_flops(config):

    hs = config.hs
    seq_len = config.model_channels + config.inp_s
    base_factor = 6 * config.gbs

    joint_layer_flops = (
        base_factor
        * config.layers[0]
        * (
            10 * hs * hs
            + 2 * hs * (config.model_channels + config.inp_s) * (1 + hs * 7)
            + 2 * (config.model_channels + config.inp_s) * hs
        )
    )

    single_layer_flops = (
        base_factor
        * config.layers[1]
        * seq_len
        * hs
        * (
            3
            + 1
            + 4 * hs
            + (3 * hs + 2 * seq_len)
            + 5 * hs
            + 1
        )
    )

    other_flops = base_factor * (
        config.inp_s * config.in_channels * hs
        + config.inp_s * hs * config.model_channels
        + config.vec_in_dim * hs
        + hs * hs
        + 2 * (config.model_channels * hs + hs * hs)
        + (config.inp_s * config.in_channels * hs) / config.gbs
    )

    return joint_layer_flops + single_layer_flops + other_flops


def deepseekv3_flops(config, gbs=1, seq_len=None):

    hs = config.hidden_size
    layers = config.num_hidden_layers
    attention_heads = config.num_attention_heads
    ffn_hs = config.intermediate_size
    vocab_size = config.vocab_size

    q_lora_rank = config.q_lora_rank if hasattr(config, "q_lora_rank") else None
    kv_lora_rank = config.kv_lora_rank
    qk_rope_head_dim = config.qk_rope_head_dim
    qk_nope_head_dim = config.qk_nope_head_dim if hasattr(config, "qk_nope_head_dim") else None

    v_head_dim = config.v_head_dim

    moe_intermediate_size = config.moe_intermediate_size
    moe_shared_expert_intermediate_size = moe_intermediate_size
    moe_ffn_hidden_size = moe_intermediate_size
    moe_router_topk = config.num_experts_per_tok

    first_k_dense_replace = config.first_k_dense_replace if hasattr(config, "first_k_dense_replace") else 0
    if hasattr(config, "moe_layer_freq"):
        moe_layer_freq = config.moe_layer_freq
    else:
        moe_layer_freq = [0] * first_k_dense_replace + [1] * (layers - first_k_dense_replace)

    mtp_num_layers = config.mtp_num_layers if hasattr(config, "mtp_num_layers") else None

    index_topk = getattr(config, "index_topk", None)
    index_n_heads = getattr(config, "index_n_heads", 0)
    index_head_dim = getattr(config, "index_head_dim", 0)

    if index_topk is not None and index_topk > 0:
        bmm1_flops = (qk_nope_head_dim + qk_rope_head_dim) * attention_heads * seq_len * index_topk
        bmm2_flops = v_head_dim * attention_heads * seq_len * index_topk
    else:
        bmm1_flops = 0.5 * (qk_nope_head_dim + qk_rope_head_dim) * attention_heads * (seq_len**2)
        bmm2_flops = 0.5 * v_head_dim * attention_heads * (seq_len**2)
    per_input_attention_flops = 6 * (bmm1_flops + bmm2_flops) * layers
    if mtp_num_layers is not None:
        per_input_attention_flops += 6 * (bmm1_flops + bmm2_flops) * mtp_num_layers

    if index_topk is not None and index_topk > 0 and index_n_heads > 0:
        idx_proj_params = (q_lora_rank or 0) * index_n_heads * index_head_dim + hs * index_head_dim + hs * index_n_heads
        idx_bmm = index_n_heads * index_head_dim * seq_len * seq_len
        per_layer_indexer = 6 * (idx_proj_params * seq_len + idx_bmm)
        total_indexer_layers = layers + (mtp_num_layers or 0)
        per_input_attention_flops += per_layer_indexer * total_indexer_layers

    if q_lora_rank is not None:
        per_layer_mla_params = hs * q_lora_rank + q_lora_rank * (
            (qk_nope_head_dim + qk_rope_head_dim) * attention_heads
        )
    else:
        per_layer_mla_params = hs * ((qk_nope_head_dim + qk_rope_head_dim) * attention_heads)

    per_layer_mla_params += hs * qk_rope_head_dim
    per_layer_mla_params += hs * kv_lora_rank + kv_lora_rank * (
        (qk_nope_head_dim + v_head_dim) * attention_heads
    )
    per_layer_mla_params += v_head_dim * attention_heads * hs
    mla_params = per_layer_mla_params * layers
    if mtp_num_layers is not None:
        mla_params += per_layer_mla_params * mtp_num_layers

    dense_layer_ffn_params = hs * ffn_hs * 3
    per_shared_expert_params = hs * moe_shared_expert_intermediate_size * 3
    per_selected_expert_params = hs * moe_ffn_hidden_size * 3
    ffn_params = 0

    if isinstance(moe_layer_freq, int):
        moe_layer_pattern = [1 if (i % moe_layer_freq == 0) else 0 for i in range(layers)]
    else:
        moe_layer_pattern = moe_layer_freq
    for i in moe_layer_pattern:
        if i == 0:
            ffn_params += dense_layer_ffn_params
        else:
            ffn_params += per_shared_expert_params + (per_selected_expert_params * moe_router_topk)
    if mtp_num_layers is not None:
        for i in range(mtp_num_layers):
            ffn_params += per_shared_expert_params + (per_selected_expert_params * moe_router_topk)
    per_input_params = mla_params + ffn_params
    per_input_linear_flops = 6 * per_input_params * seq_len

    per_input_vocab_flops = 6 * vocab_size * hs * seq_len
    if mtp_num_layers is not None:
        for i in range(mtp_num_layers):
            per_input_vocab_flops += 6 * vocab_size * hs * seq_len
            per_input_vocab_flops += 6 * hs * 2 * hs * seq_len

    return (per_input_attention_flops + per_input_linear_flops + per_input_vocab_flops) * gbs


def _nemotronh_mlp_layer_flops(config, gbs, seq_len):
    return 6 * gbs * seq_len * config.hidden_size * config.intermediate_size * 3


def _nemotronh_moe_layer_flops(config, gbs, seq_len):
    hs = config.hidden_size
    num_tokens = gbs * seq_len

    moe_latent_size = getattr(config, "moe_latent_size", None)

    if moe_latent_size is not None:
        expert_dim = moe_latent_size
        latent_proj_flops = 6 * num_tokens * hs * moe_latent_size * 2
    else:
        expert_dim = hs
        latent_proj_flops = 0

    routed_expert_flops = 6 * num_tokens * config.num_experts_per_tok * expert_dim * config.moe_intermediate_size * 2

    shared_expert_flops = 6 * num_tokens * hs * config.moe_shared_expert_intermediate_size * 2

    gate_flops = 6 * num_tokens * hs * config.n_routed_experts

    return routed_expert_flops + shared_expert_flops + gate_flops + latent_proj_flops


def _non_mla_attn_layer_flops(config, gbs, seq_len):
    hs = config.hidden_size
    attention_heads = config.num_attention_heads
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else attention_heads

    return (
        6
        * gbs
        * seq_len
        * hs
        * (
            hs
            + query_groups / attention_heads * hs * 2
            + seq_len / 2 * 2
            + hs
        )
    )


def _mamba_layer_flops(config, gbs, seq_len):
    hs = config.hidden_size
    if hasattr(config, "mamba_state_dim"):
        mamba_state_dim = config.mamba_state_dim
    elif hasattr(config, "ssm_state_size"):
        mamba_state_dim = config.ssm_state_size
    else:
        raise ValueError("Expected config to have 'mamba_state_dim' or 'ssm_state_size'")
    mamba_head_dim = config.mamba_head_dim
    if hasattr(config, "mamba_num_groups"):
        mamba_num_groups = config.mamba_num_groups
    elif hasattr(config, "n_groups"):
        mamba_num_groups = config.n_groups
    else:
        raise ValueError("Expected config to have 'mamba_num_groups' or 'n_groups'")

    if hasattr(config, "mamba_num_heads") and config.mamba_num_heads:
        nheads = config.mamba_num_heads
    else:
        nheads = 2 * hs // mamba_head_dim
    d_in = nheads * mamba_head_dim

    in_proj = 6 * gbs * seq_len * hs * (2 * d_in + 2 * mamba_num_groups * mamba_state_dim + nheads)
    scan = 7 * 3 * gbs * seq_len * d_in * mamba_state_dim
    out_proj = 6 * gbs * seq_len * d_in * hs
    return in_proj + scan + out_proj


def _hybrid_model_flops(config, gbs, seq_len):
    if hasattr(config, "is_hybrid_model"):
        if not config.is_hybrid_model:
            raise ValueError("Config must have is_hybrid_model=True")
    elif not hasattr(config, "hybrid_override_pattern"):
        raise ValueError("Expected config to have `is_hybrid_model` or `hybrid_override_pattern`")

    hybrid_override_pattern = config.hybrid_override_pattern
    hs = config.hidden_size
    vocab_size = config.vocab_size

    num_attn_layers, num_mamba_layers, num_mlp_layers, num_moe_layers = 0, 0, 0, 0
    for c in hybrid_override_pattern:
        if c == "M":
            num_mamba_layers += 1
        elif c == "-":
            num_mlp_layers += 1
        elif c == "*":
            num_attn_layers += 1
        elif c == "E":
            num_moe_layers += 1

    total = 6 * gbs * seq_len * hs * vocab_size
    if num_attn_layers:
        total += num_attn_layers * _non_mla_attn_layer_flops(config, gbs, seq_len)
    if num_mamba_layers:
        total += num_mamba_layers * _mamba_layer_flops(config, gbs, seq_len)
    if num_mlp_layers:
        total += num_mlp_layers * _nemotronh_mlp_layer_flops(config, gbs, seq_len)
    if num_moe_layers:
        total += num_moe_layers * _nemotronh_moe_layer_flops(config, gbs, seq_len)
    return total


def nemotronh_flops(config, gbs=1, seq_len=None):
    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    return _hybrid_model_flops(config, gbs, seq_len)


def attention_flops_calculator(
    seqlen,
    hidden_size,
    num_attention_heads,
    num_query_groups,
    kv_channels: Optional[int] = None,
    is_swa: bool = False,
    swa_window_size: int = 128,
):
    kv_channels = kv_channels or (hidden_size // num_attention_heads)

    linear_qkv = seqlen * hidden_size * (kv_channels * (num_attention_heads + num_query_groups * 2))

    linear_proj = seqlen * hidden_size * (kv_channels * num_attention_heads)

    if is_swa:
        attention_mask_nz_elem = (
            swa_window_size * (swa_window_size + 1) / 2 + (seqlen - swa_window_size) * swa_window_size
        )
        attention = num_attention_heads * (attention_mask_nz_elem * kv_channels) * 2
    else:
        bmm_k = kv_channels
        bmm_b = num_attention_heads
        attention_mask_nz_elem = seqlen * (seqlen + 1) / 2
        attention = bmm_b * attention_mask_nz_elem * bmm_k * 2

    return (linear_qkv + linear_proj + attention) * 6


def moe_mlp_flops_calculator(
    seqlen,
    hidden_size,
    moe_ffn_hidden_size,
    moe_router_topk,
    gated_linear_unit: bool = True,
):
    total_num_tokens = seqlen * moe_router_topk
    linear_fc1 = total_num_tokens * hidden_size * moe_ffn_hidden_size * (2 if gated_linear_unit else 1)
    linear_fc2 = total_num_tokens * moe_ffn_hidden_size * hidden_size
    return (linear_fc1 + linear_fc2) * 6


def loss_flops_calculator(
    seqlen,
    hidden_size,
    vocab_size,
):
    return (seqlen * hidden_size * vocab_size) * 6


def gpt_oss_flops_calculator(
    gbs,
    num_layers,
    seqlen,
    hidden_size,
    num_attention_heads,
    num_query_groups,
    moe_ffn_hidden_size,
    moe_router_topk,
    vocab_size,
    kv_channels: Optional[int] = None,
    swa_window_size: int = 128,
    window_attn_skip_freq: Optional[int] = 2,
):
    flops = 0
    for i in range(num_layers):
        if i % window_attn_skip_freq == 0:
            flops += attention_flops_calculator(
                seqlen,
                hidden_size,
                num_attention_heads,
                num_query_groups,
                kv_channels,
                is_swa=False,
            )
        else:
            flops += attention_flops_calculator(
                seqlen,
                hidden_size,
                num_attention_heads,
                num_query_groups,
                kv_channels,
                is_swa=True,
                swa_window_size=swa_window_size,
            )
        flops += moe_mlp_flops_calculator(
            seqlen,
            hidden_size,
            moe_ffn_hidden_size,
            moe_router_topk,
        )
    flops += loss_flops_calculator(seqlen, hidden_size, vocab_size)
    flops *= gbs
    return flops


def gpt_oss_flops(config, gbs=1, seq_len=None):
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else num_attention_heads
    vocab_size = config.vocab_size

    moe_ffn_hidden_size = (
        config.moe_ffn_hidden_size if hasattr(config, "moe_ffn_hidden_size") else config.intermediate_size
    )
    moe_router_topk = config.num_experts_per_tok
    kv_channels = config.kv_channels if hasattr(config, "kv_channels") else (hidden_size // num_attention_heads)
    swa_window_size = config.window_size[0] if hasattr(config, "window_size") and config.window_size else 128
    window_attn_skip_freq = config.window_attn_skip_freq if hasattr(config, "window_attn_skip_freq") else 2

    return gpt_oss_flops_calculator(
        gbs=gbs,
        num_layers=num_layers,
        seqlen=seq_len,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        moe_ffn_hidden_size=moe_ffn_hidden_size,
        moe_router_topk=moe_router_topk,
        vocab_size=vocab_size,
        kv_channels=kv_channels,
        swa_window_size=swa_window_size,
        window_attn_skip_freq=window_attn_skip_freq,
    )


def glm4_moe_flops(config, gbs=1, seq_len=None):
    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    layers = config.num_hidden_layers
    hs = config.hidden_size
    attention_heads = config.num_attention_heads
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else attention_heads
    vocab_size = config.vocab_size

    head_dim = getattr(config, "head_dim", hs // attention_heads)
    query_projection_to_hidden_size_ratio = (head_dim * attention_heads) / hs

    ffn_hs = config.intermediate_size
    moe_intermediate_size = config.moe_intermediate_size if hasattr(config, "moe_intermediate_size") else ffn_hs
    moe_router_topk = config.num_experts_per_tok if hasattr(config, "num_experts_per_tok") else 1
    n_shared_experts = config.n_shared_experts if hasattr(config, "n_shared_experts") else 0
    first_k_dense_replace = config.first_k_dense_replace if hasattr(config, "first_k_dense_replace") else 0

    causal_self_attn = True
    hidden_size = hs
    gated_linear_multiplier = 2

    attention_flops = (
        3
        * 2
        * gbs
        * layers
        * seq_len
        * hidden_size
        * hidden_size
        * query_projection_to_hidden_size_ratio
        * (
            (query_groups / attention_heads * 2 + 1)
            + (seq_len / hidden_size * 2 * (0.5 if causal_self_attn else 1))
            + 1
        )
    )

    dense_mlp_flops = (
        3 * 2 * gbs * first_k_dense_replace * seq_len * hidden_size * (1 + gated_linear_multiplier) * ffn_hs
    )

    num_moe_layers = layers - first_k_dense_replace

    shared_expert_flops = (
        3
        * 2
        * gbs
        * num_moe_layers
        * seq_len
        * hidden_size
        * (1 + gated_linear_multiplier)
        * (moe_intermediate_size * n_shared_experts)
    )

    routed_expert_flops = (
        3
        * 2
        * gbs
        * num_moe_layers
        * seq_len
        * hidden_size
        * (1 + gated_linear_multiplier)
        * (moe_intermediate_size * moe_router_topk)
    )

    mlp_flops = dense_mlp_flops + shared_expert_flops + routed_expert_flops

    vocab_flops = 3 * 2 * gbs * seq_len * hidden_size * vocab_size

    return attention_flops + mlp_flops + vocab_flops


def minimax_m2_flops(config, gbs=1, seq_len=None):

    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    layers = config.num_hidden_layers
    hs = config.hidden_size
    attention_heads = config.num_attention_heads
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else attention_heads
    vocab_size = config.vocab_size
    head_dim = getattr(config, "head_dim", hs // attention_heads)
    query_projection_to_hidden_size_ratio = (head_dim * attention_heads) / hs

    ffn_hs = config.intermediate_size
    moe_router_topk = config.num_experts_per_tok if hasattr(config, "num_experts_per_tok") else 8
    shared_intermediate_size = getattr(config, "shared_intermediate_size", 0)

    use_mtp = getattr(config, "use_mtp", False)
    num_mtp_modules = getattr(config, "num_mtp_modules", 0) if use_mtp else 0
    mtp_transformer_layers = getattr(config, "mtp_transformer_layers", 1)

    causal_self_attn = True
    gated_linear_multiplier = 2

    def _attention_flops_per_layer():
        return (
            6
            * gbs
            * seq_len
            * hs
            * hs
            * query_projection_to_hidden_size_ratio
            * (
                (query_groups / attention_heads * 2 + 1)
                + (seq_len / hs * 2 * (0.5 if causal_self_attn else 1))
                + 1
            )
        )

    attention_flops = _attention_flops_per_layer() * layers

    def _moe_mlp_flops_per_layer():
        routed = 6 * gbs * seq_len * hs * (1 + gated_linear_multiplier) * (ffn_hs * moe_router_topk)
        shared = (
            6 * gbs * seq_len * hs * (1 + gated_linear_multiplier) * shared_intermediate_size
            if shared_intermediate_size > 0
            else 0
        )
        return routed + shared

    mlp_flops = _moe_mlp_flops_per_layer() * layers

    vocab_flops = 6 * gbs * seq_len * hs * vocab_size

    mtp_flops = 0
    if num_mtp_modules > 0:
        total_mtp_layers = num_mtp_modules * mtp_transformer_layers
        mtp_flops += 6 * gbs * seq_len * hs * 2 * hs * num_mtp_modules
        mtp_flops += _attention_flops_per_layer() * total_mtp_layers
        mtp_flops += _moe_mlp_flops_per_layer() * total_mtp_layers
        mtp_flops += 6 * gbs * seq_len * hs * vocab_size * num_mtp_modules

    return attention_flops + mlp_flops + vocab_flops + mtp_flops


def _gdn_attention_per_layer_flops(
    gbs,
    seq_len,
    hidden_size,
    linear_key_head_dim,
    linear_value_head_dim,
    linear_num_key_heads,
    linear_num_value_heads,
    linear_conv_kernel_dim,
):
    """FLOPs for a single Gated DeltaNet (GDN / linear attention) layer.

    Based on the GDN FLOPs calculator from Megatron-Bridge PR #2925.
    """
    qk_dim = linear_key_head_dim * linear_num_key_heads
    v_dim = linear_value_head_dim * linear_num_value_heads

    return (
        3
        * 2
        * gbs
        * seq_len
        * (
            hidden_size * (2 * qk_dim + 2 * v_dim + 2 * linear_num_value_heads)
            + linear_conv_kernel_dim * (2 * qk_dim + v_dim)
            + linear_num_value_heads * (linear_value_head_dim**2) * 4
            + hidden_size * v_dim
        )
    )


def qwen3_5_flops(config, gbs=1, seq_len=None):
    if hasattr(config, "text_config") and not hasattr(config, "num_hidden_layers"):
        config = config.text_config

    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    layers = config.num_hidden_layers
    hs = config.hidden_size
    attention_heads = config.num_attention_heads
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else attention_heads
    vocab_size = config.vocab_size
    head_dim = getattr(config, "head_dim", hs // attention_heads)

    linear_key_head_dim = config.linear_key_head_dim
    linear_value_head_dim = config.linear_value_head_dim
    linear_num_key_heads = config.linear_num_key_heads
    linear_num_value_heads = config.linear_num_value_heads
    linear_conv_kernel_dim = getattr(config, "linear_conv_kernel_dim", 4)

    if hasattr(config, "layer_types") and config.layer_types:
        layer_types = config.layer_types
        num_full_attn_layers = sum(1 for lt in layer_types if lt == "full_attention")
        num_gdn_layers = layers - num_full_attn_layers
    else:
        full_attention_interval = getattr(config, "full_attention_interval", 4)
        num_full_attn_layers = layers // full_attention_interval
        num_gdn_layers = layers - num_full_attn_layers

    is_moe = hasattr(config, "num_experts") and config.num_experts is not None and config.num_experts > 1
    moe_router_topk = getattr(config, "num_experts_per_tok", 1) if is_moe else 1
    moe_intermediate_size = getattr(config, "moe_intermediate_size", 0) if is_moe else 0
    shared_expert_intermediate_size = getattr(config, "shared_expert_intermediate_size", 0) if is_moe else 0
    ffn_hs = getattr(config, "intermediate_size", 0) if not is_moe else 0

    mtp_num_layers = getattr(config, "mtp_num_hidden_layers", 0) or 0

    causal_self_attn = True
    gated_linear_multiplier = 2

    query_projection_to_hidden_size_ratio = (head_dim * attention_heads) / hs

    attn_output_gate = getattr(config, "attn_output_gate", True)
    q_gate_multiplier = 2 if attn_output_gate else 1

    full_attn_per_layer = (
        6
        * gbs
        * seq_len
        * hs
        * hs
        * query_projection_to_hidden_size_ratio
        * (
            (query_groups / attention_heads * 2 + q_gate_multiplier)
            + (seq_len / hs * 2 * (0.5 if causal_self_attn else 1))
            + 1
        )
    )

    gdn_attn_per_layer = _gdn_attention_per_layer_flops(
        gbs,
        seq_len,
        hs,
        linear_key_head_dim,
        linear_value_head_dim,
        linear_num_key_heads,
        linear_num_value_heads,
        linear_conv_kernel_dim,
    )

    attention_flops = full_attn_per_layer * num_full_attn_layers + gdn_attn_per_layer * num_gdn_layers

    if is_moe:
        routed_expert_flops = (
            6 * gbs * layers * seq_len * hs * (1 + gated_linear_multiplier) * (moe_intermediate_size * moe_router_topk)
        )
        shared_expert_flops = (
            6 * gbs * layers * seq_len * hs * (1 + gated_linear_multiplier) * shared_expert_intermediate_size
        )
        mlp_flops = routed_expert_flops + shared_expert_flops
    else:
        mlp_flops = 6 * gbs * layers * seq_len * hs * (1 + gated_linear_multiplier) * ffn_hs

    vocab_flops = 6 * gbs * seq_len * hs * vocab_size

    mtp_flops = 0
    if mtp_num_layers > 0:
        mtp_flops += 6 * gbs * seq_len * hs * 2 * hs * mtp_num_layers
        mtp_flops += full_attn_per_layer * mtp_num_layers
        if is_moe:
            mtp_mlp_per_layer = (
                6
                * gbs
                * seq_len
                * hs
                * (1 + gated_linear_multiplier)
                * (moe_intermediate_size * moe_router_topk + shared_expert_intermediate_size)
            )
        else:
            mtp_mlp_per_layer = 6 * gbs * seq_len * hs * (1 + gated_linear_multiplier) * ffn_hs
        mtp_flops += mtp_mlp_per_layer * mtp_num_layers
        mtp_flops += 6 * gbs * seq_len * hs * vocab_size * mtp_num_layers

    return attention_flops + mlp_flops + vocab_flops + mtp_flops


def _mla_attention_per_layer_flops(
    gbs,
    seq_len,
    hs,
    attention_heads,
    q_lora_rank,
    kv_lora_rank,
    qk_rope_head_dim,
    qk_nope_head_dim,
    v_head_dim,
    index_topk=None,
    index_n_heads=0,
    index_head_dim=0,
):
    if index_topk is not None and index_topk > 0:
        bmm1 = (qk_nope_head_dim + qk_rope_head_dim) * attention_heads * seq_len * index_topk
        bmm2 = v_head_dim * attention_heads * seq_len * index_topk
    else:
        bmm1 = 0.5 * (qk_nope_head_dim + qk_rope_head_dim) * attention_heads * (seq_len**2)
        bmm2 = 0.5 * v_head_dim * attention_heads * (seq_len**2)
    bmm_flops = 6 * gbs * (bmm1 + bmm2)

    if q_lora_rank is not None:
        q_params = hs * q_lora_rank + q_lora_rank * ((qk_nope_head_dim + qk_rope_head_dim) * attention_heads)
    else:
        q_params = hs * ((qk_nope_head_dim + qk_rope_head_dim) * attention_heads)

    kr_params = hs * qk_rope_head_dim
    kv_params = hs * kv_lora_rank + kv_lora_rank * ((qk_nope_head_dim + v_head_dim) * attention_heads)
    o_params = v_head_dim * attention_heads * hs

    linear_flops = 6 * gbs * seq_len * (q_params + kr_params + kv_params + o_params)

    indexer_flops = 0
    if index_topk is not None and index_topk > 0 and index_n_heads > 0:
        idx_proj_params = (
            (q_lora_rank or 0) * index_n_heads * index_head_dim
            + hs * index_head_dim
            + hs * index_n_heads
        )
        idx_bmm = index_n_heads * index_head_dim * seq_len * seq_len
        indexer_flops = 6 * gbs * (idx_proj_params * seq_len + idx_bmm)

    return bmm_flops + linear_flops + indexer_flops


def _mla_moe_model_flops(
    gbs,
    seq_len,
    hs,
    layers,
    attention_heads,
    vocab_size,
    q_lora_rank,
    kv_lora_rank,
    qk_rope_head_dim,
    qk_nope_head_dim,
    v_head_dim,
    dense_ffn_hs,
    moe_ffn_hs,
    moe_router_topk,
    moe_shared_expert_hs,
    moe_layer_pattern,
    mtp_num_layers=0,
    index_topk=None,
    index_n_heads=0,
    index_head_dim=0,
):
    mla_per_layer = _mla_attention_per_layer_flops(
        gbs,
        seq_len,
        hs,
        attention_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_rope_head_dim,
        qk_nope_head_dim,
        v_head_dim,
        index_topk=index_topk,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
    )
    attention_flops = mla_per_layer * layers

    dense_layer_ffn_params = hs * dense_ffn_hs * 3
    per_shared_expert_params = hs * moe_shared_expert_hs * 3
    per_selected_expert_params = hs * moe_ffn_hs * 3

    ffn_params = 0
    for is_moe in moe_layer_pattern:
        if is_moe == 0:
            ffn_params += dense_layer_ffn_params
        else:
            ffn_params += per_shared_expert_params + (per_selected_expert_params * moe_router_topk)
    ffn_flops = 6 * gbs * seq_len * ffn_params

    vocab_flops = 6 * gbs * seq_len * hs * vocab_size

    mtp_flops = 0
    if mtp_num_layers > 0:
        mtp_flops += mla_per_layer * mtp_num_layers
        last_is_moe = moe_layer_pattern[-1] if moe_layer_pattern else 0
        if last_is_moe:
            mtp_ffn_params = per_shared_expert_params + (per_selected_expert_params * moe_router_topk)
        else:
            mtp_ffn_params = dense_layer_ffn_params
        mtp_flops += 6 * gbs * seq_len * mtp_ffn_params * mtp_num_layers
        mtp_flops += 6 * gbs * seq_len * hs * vocab_size * mtp_num_layers
        mtp_flops += 6 * gbs * seq_len * hs * 2 * hs * mtp_num_layers

    return attention_flops + ffn_flops + vocab_flops + mtp_flops


def _build_moe_layer_pattern(config, layers):
    mlp_layer_types = getattr(config, "mlp_layer_types", None)
    if mlp_layer_types is not None:
        return [0 if lt == "dense" else 1 for lt in mlp_layer_types]

    first_k_dense = getattr(config, "first_k_dense_replace", 0)
    moe_layer_freq = getattr(config, "moe_layer_freq", 1)
    if isinstance(moe_layer_freq, list):
        return moe_layer_freq
    return [0] * first_k_dense + [
        1 if ((i - first_k_dense) % moe_layer_freq == 0) else 0 for i in range(first_k_dense, layers)
    ]


def mla_moe_flops(config, gbs=1, seq_len=None):
    cfg = config
    if hasattr(config, "text_config") and not hasattr(config, "num_hidden_layers"):
        cfg = config.text_config

    if seq_len is None:
        seq_len = getattr(cfg, "max_position_embeddings", 2048)

    layers = cfg.num_hidden_layers
    hs = cfg.hidden_size
    n_shared = getattr(cfg, "n_shared_experts", 0)

    moe_int_size = getattr(cfg, "moe_intermediate_size", None)
    if moe_int_size is None:
        moe_int_size = getattr(cfg, "expert_ffn_hidden_size", cfg.intermediate_size)

    dense_ffn_hs = getattr(cfg, "intermediate_size", None)
    if dense_ffn_hs is None:
        dense_ffn_hs = getattr(cfg, "ffn_hidden_size", moe_int_size)

    moe_topk = getattr(cfg, "num_experts_per_tok", None)
    if moe_topk is None:
        moe_topk = getattr(cfg, "moe_topk", 1)

    moe_layer_pattern = _build_moe_layer_pattern(cfg, layers)

    mtp = getattr(cfg, "num_nextn_predict_layers", None)
    if mtp is None:
        mtp = getattr(cfg, "mtp_num_layers", 0)
    mtp = mtp or 0

    idx_topk = getattr(cfg, "index_topk", None)
    idx_n_heads = getattr(cfg, "index_n_heads", 0)
    idx_head_dim = getattr(cfg, "index_head_dim", 0)

    return _mla_moe_model_flops(
        gbs=gbs,
        seq_len=seq_len,
        hs=hs,
        layers=layers,
        attention_heads=cfg.num_attention_heads,
        vocab_size=cfg.vocab_size,
        q_lora_rank=getattr(cfg, "q_lora_rank", None),
        kv_lora_rank=cfg.kv_lora_rank,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        qk_nope_head_dim=cfg.qk_nope_head_dim,
        v_head_dim=cfg.v_head_dim,
        dense_ffn_hs=dense_ffn_hs,
        moe_ffn_hs=moe_int_size,
        moe_router_topk=moe_topk,
        moe_shared_expert_hs=moe_int_size * n_shared,
        moe_layer_pattern=moe_layer_pattern,
        mtp_num_layers=mtp,
        index_topk=idx_topk,
        index_n_heads=idx_n_heads,
        index_head_dim=idx_head_dim,
    )


def step3_5_flash_flops(config, gbs=1, seq_len=None):
    if seq_len is None:
        seq_len = getattr(config, "max_position_embeddings", 2048)

    layers = config.num_hidden_layers
    hs = config.hidden_size
    vocab_size = config.vocab_size

    full_attn_heads = config.num_attention_heads
    attn_other = getattr(config, "attention_other_setting", None)
    if attn_other is not None and isinstance(attn_other, dict):
        sliding_attn_heads = attn_other.get("num_attention_heads", full_attn_heads)
    else:
        sliding_attn_heads = full_attn_heads
    num_query_groups = getattr(config, "num_attention_groups", full_attn_heads)
    head_dim = getattr(config, "head_dim", hs // full_attn_heads)
    sliding_window = getattr(config, "sliding_window", 512)

    moe_top_k = getattr(config, "moe_top_k", 8)
    moe_ffn_hs = getattr(config, "moe_intermediate_size", 1280)
    share_expert_dim = getattr(config, "share_expert_dim", moe_ffn_hs)
    dense_ffn_hs = config.intermediate_size

    moe_layers_raw = getattr(config, "moe_layers_enum", None)
    if moe_layers_raw is not None:
        if isinstance(moe_layers_raw, str):
            moe_layers_set = set(int(x.strip()) for x in moe_layers_raw.split(",") if x.strip())
        else:
            moe_layers_set = set(int(x) for x in moe_layers_raw)
    else:
        moe_layers_set = set(range(3, layers))

    layer_types = getattr(config, "layer_types", None)

    mtp_num_layers = getattr(config, "num_nextn_predict_layers", 0) or 0

    total_attn = 0
    total_mlp = 0

    for i in range(layers):
        if layer_types and i < len(layer_types):
            is_full = layer_types[i] == "full_attention"
        else:
            is_full = i % 4 == 0

        if is_full:
            total_attn += attention_flops_calculator(
                seq_len,
                hs,
                full_attn_heads,
                num_query_groups,
                head_dim,
                is_swa=False,
            )
        else:
            total_attn += attention_flops_calculator(
                seq_len,
                hs,
                sliding_attn_heads,
                num_query_groups,
                head_dim,
                is_swa=True,
                swa_window_size=sliding_window,
            )

        if i in moe_layers_set:
            total_mlp += moe_mlp_flops_calculator(
                seq_len,
                hs,
                moe_ffn_hs,
                moe_top_k,
                gated_linear_unit=True,
            )
            total_mlp += 6 * seq_len * hs * share_expert_dim * 3
        else:
            total_mlp += 6 * seq_len * hs * dense_ffn_hs * 3

    total_vocab = loss_flops_calculator(seq_len, hs, vocab_size)

    mtp_total = 0
    if mtp_num_layers > 0:
        mtp_total += 6 * seq_len * hs * 2 * hs * mtp_num_layers
        mtp_total += (
            attention_flops_calculator(
                seq_len,
                hs,
                full_attn_heads,
                num_query_groups,
                head_dim,
                is_swa=False,
            )
            * mtp_num_layers
        )
        mtp_total += (
            moe_mlp_flops_calculator(
                seq_len,
                hs,
                moe_ffn_hs,
                moe_top_k,
                gated_linear_unit=True,
            )
            * mtp_num_layers
        )
        mtp_total += 6 * seq_len * hs * share_expert_dim * 3 * mtp_num_layers
        mtp_total += loss_flops_calculator(seq_len, hs, vocab_size) * mtp_num_layers

    return gbs * (total_attn + total_mlp + total_vocab + mtp_total)


def get_flops_formula_for_hf_config(config: Any) -> Optional[Callable]:
    config_class_name = config.__class__.__name__

    class_name_to_formula = {
        "GPT2Config": gpt3_flops,
        "GPTNeoConfig": gpt3_flops,
        "GPTNeoXConfig": gpt3_flops,
        "GPTJConfig": gpt3_flops,
        "LlamaConfig": llama2_flops,
        "MixtralConfig": mixtral_flops,
        "Qwen2Config": qwen3_flops,
        "Qwen3Config": qwen3_flops,
        "Qwen3MoeConfig": qwen3_flops,
        "Qwen3_5Config": qwen3_5_flops,
        "Qwen3_5MoeConfig": qwen3_5_flops,
        "Qwen3NextConfig": qwen3_5_flops,
        "Qwen3VLMoeConfig": qwen3_flops,
        "Qwen3VLMoeTextConfig": qwen3_flops,
        "Qwen3VLConfig": qwen3_flops,
        "Qwen3VLTextConfig": qwen3_flops,
        "BertConfig": bert_flops,
        "RobertaConfig": bert_flops,
        "AlbertConfig": bert_flops,
        "ElectraConfig": bert_flops,
        "DeepseekV3Config": deepseekv3_flops,
        "GptOssConfig": gpt_oss_flops,
        "Glm4Config": qwen3_flops,
        "Glm4MoeConfig": glm4_moe_flops,
        "Glm4MoeLiteConfig": mla_moe_flops,
        "GlmMoeDsaConfig": mla_moe_flops,
        "MiniMaxM2Config": minimax_m2_flops,
        "Mistral3Config": mla_moe_flops,
        "KimiK2Config": mla_moe_flops,
        "KimiK25Config": mla_moe_flops,
        "Step3p5Config": step3_5_flash_flops,
        "LongcatFlashConfig": mla_moe_flops,
        "T5Config": transformer_flops,
        "MT5Config": transformer_flops,
        "NemotronConfig": nemotron_flops,
        "NemotronHConfig": nemotronh_flops,
        "OPTConfig": transformer_flops,
        "BloomConfig": transformer_flops,
        "FalconConfig": transformer_flops,
    }

    formula = class_name_to_formula.get(config_class_name)

    if formula is None:
        formula = transformer_flops

    return formula
