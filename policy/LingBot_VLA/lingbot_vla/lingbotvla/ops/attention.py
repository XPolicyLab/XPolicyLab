from typing import Optional, Tuple, Literal

import torch
from transformers.modeling_flash_attention_utils import _flash_attention_forward
import torch.nn.functional as F  # noqa: N812
from packaging.version import Version
import einops
from ..distributed.parallel_state import get_parallel_state
from ..distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
)
from ..utils import logging
from ..utils.import_utils import is_seed_kernels_available

if is_seed_kernels_available():
    from seed_kernels.transformers.functional import seed_flash_attention_forward

logger = logging.get_logger(__name__)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    implementation: Optional[Literal["fa2", "lego", "fa3"]] = None,
    skip_ulysses: bool = False,  # Skip ulysses for some ViT cases like internvl3.5
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask", None) is not None:
        logger.warning_once(
            "`flash_attention_2` does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )

    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # FA2 always relies on the value set in the module, so remove it if present in kwargs to avoid passing it twice
    kwargs.pop("is_causal", None)

    # This is for Qwen2VL's mrope
    position_ids = kwargs.pop("position_ids", None)
    if position_ids is not None and position_ids.dim() == 3:
        position_ids = position_ids[0]

    # Ulysses patch
    ulysses_enabled = get_parallel_state().ulysses_enabled
    if ulysses_enabled and not skip_ulysses:
        ulysses_group = get_parallel_state().ulysses_group
        # Sanity Check & Repeat Key & Value
        ulysses_size = get_parallel_state().ulysses_size
        q_head_num = query.shape[2]
        kv_head_num = key.shape[2]
        unpadded_seq_len = None

        assert q_head_num % ulysses_size == 0, (
            f"num_query_heads ({q_head_num}) must be divisible by ulysses_size ({ulysses_size})"
        )
        if ulysses_size > kv_head_num:
            assert ulysses_size % kv_head_num == 0, (
                f"ulysses_size ({ulysses_size}) must be divisible by num_key_value_heads ({kv_head_num})"
            )
            n_repeat = ulysses_size // kv_head_num
            key = repeat_kv(key, n_repeat)
            value = repeat_kv(value, n_repeat)

        if query.ndim == 4 and query.size(0) == 1:
            query, key, value = query.squeeze(0), key.squeeze(0), value.squeeze(0)
            query = gather_seq_scatter_heads(
                query, seq_dim=0, head_dim=1, group=ulysses_group, unpadded_dim_size=unpadded_seq_len
            )
            key = gather_seq_scatter_heads(
                key, seq_dim=0, head_dim=1, group=ulysses_group, unpadded_dim_size=unpadded_seq_len
            )
            value = gather_seq_scatter_heads(
                value, seq_dim=0, head_dim=1, group=ulysses_group, unpadded_dim_size=unpadded_seq_len
            )
            query, key, value = query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0)
        else:
            query = gather_seq_scatter_heads(
                query, seq_dim=1, head_dim=2, group=ulysses_group, unpadded_dim_size=unpadded_seq_len
            )
            key = gather_seq_scatter_heads(
                key, seq_dim=1, head_dim=2, group=ulysses_group, unpadded_dim_size=unpadded_seq_len
            )
            value = gather_seq_scatter_heads(
                value, seq_dim=1, head_dim=2, group=ulysses_group, unpadded_dim_size=unpadded_seq_len
            )

    # Only after all_to_all we got the full seq_len
    seq_len = query.shape[1]

    if is_seed_kernels_available() and implementation is not None:
        attn_output: torch.Tensor = seed_flash_attention_forward(
            query,
            key,
            value,
            attention_mask,
            query_length=seq_len,
            is_causal=module.is_causal,
            dropout=dropout,
            position_ids=position_ids,
            softmax_scale=scaling,
            sliding_window=sliding_window,
            softcap=softcap,
            use_top_left_mask=False,
            implementation=implementation,
            cu_seqlens=kwargs.get("cu_seq_lens_q", None),
            max_seqlen=kwargs.get("max_length_q", None),
            **kwargs,
        )
    else:
        assert implementation is None, (
            f"You set {implementation=} but seed_kernels is not installed. Check --model.attn_implementation."
        )
        attn_output: torch.Tensor = _flash_attention_forward(
            query,
            key,
            value,
            attention_mask,
            query_length=seq_len,
            is_causal=module.is_causal,
            dropout=dropout,
            position_ids=position_ids,
            softmax_scale=scaling,
            sliding_window=sliding_window,
            softcap=softcap,
            use_top_left_mask=False,
            implementation="flash_attention_2",
            **kwargs,
        )

    # Ulysses patch
    if ulysses_enabled and not skip_ulysses:
        ulysses_group = get_parallel_state().ulysses_group
        if attn_output.ndim == 4 and attn_output.size(0) == 1:
            attn_output = attn_output.squeeze(0)
            attn_output = gather_heads_scatter_seq(attn_output, seq_dim=0, head_dim=1, group=ulysses_group)
            attn_output = attn_output.unsqueeze(0)
        else:
            attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2, group=ulysses_group)

    return attn_output, None

if Version(torch.__version__) > Version("2.5.0"):
    # Ffex attention is only available from torch 2.5 onwards
    from torch.nn.attention.flex_attention import (
        _mask_mod_signature,
        _round_up_to_multiple,
        create_block_mask,
        create_mask,
        flex_attention,
    )

# @torch.compile(dynamic=False)
def flex_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling=None,
):
    """
    This is defined out of classes to make compile happy.
    """
    batch_size, seq_len, num_att_heads, head_dim = query_states.shape # head_dim=256
    original_dtype = query_states.dtype
    num_key_value_heads = key_states.shape[2] # 1
    num_key_value_groups = num_att_heads // num_key_value_heads # 8 // 1

    key_states = einops.repeat(
        key_states, "b l h d -> b l (h g) d", g=num_key_value_groups
    )
    value_states = einops.repeat(
        value_states, "b l h d -> b l (h g) d", g=num_key_value_groups
    )

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    query_states = query_states.to(torch.float32)
    key_states = key_states.to(torch.float32)
    value_states = value_states.to(torch.float32)

    causal_mask = attention_mask
    if causal_mask is not None:
        causal_mask = causal_mask[:, None, :, : key_states.shape[2]]

        if causal_mask.shape[1] == 1 and query_states.shape[1] > 1:
            causal_mask = causal_mask.expand(-1, query_states.shape[1], -1, -1)

    def precomputed_mask_factory(precomputed_mask: torch.Tensor) -> _mask_mod_signature:
        def mask_mod(b, h, q_idx, kv_idx):
            # Danger zone: if b,h,q_idx,kv_idx exceed the shape, device-side assert occurs.
            return precomputed_mask[b][h][q_idx][kv_idx]

        return mask_mod

    b_mask, h_mask, q_len, kv_len = causal_mask.shape  # The shape of your mask
    # ipdb.set_trace()
    block_size = 128
    q_len_rounded = _round_up_to_multiple(q_len, block_size)
    kv_len_rounded = _round_up_to_multiple(kv_len, block_size)

    # *CRITICAL* we do need to expand here, else we get a CUDA index error

    pad_q = q_len_rounded - q_len
    pad_k = kv_len_rounded - kv_len

    if pad_q > 0:
        query_states = F.pad(query_states, (0, 0, 0, pad_q), value=0.0)  # [B, H, q_len_rounded, D]
    if pad_k > 0:
        key_states = F.pad(key_states, (0, 0, 0, pad_k), value=0.0)
        value_states = F.pad(value_states, (0, 0, 0, pad_k), value=0.0)
    padded_causal_mask = F.pad(causal_mask, (0, pad_k, 0, pad_q), value=0.0)
    mask_mod_fn_orig = precomputed_mask_factory(padded_causal_mask)

    mask_4d = create_mask(
        mod_fn=mask_mod_fn_orig,
        B=b_mask,
        H=h_mask,
        Q_LEN=q_len_rounded,
        KV_LEN=kv_len_rounded,
        device=causal_mask.device,
    )

    mask_mod_fn_padded = precomputed_mask_factory(mask_4d)
    block_mask = create_block_mask(
        mask_mod=mask_mod_fn_padded,
        B=b_mask,
        H=h_mask,
        Q_LEN=q_len_rounded,
        KV_LEN=kv_len_rounded,
        BLOCK_SIZE=block_size,
        device=causal_mask.device,
        _compile=False,
    )

    #  mask is applied inside the kernel, ideally more efficiently than score_mod.
    attn_output, attention_weights = flex_attention(
        query_states,
        key_states,
        value_states,
        block_mask=block_mask,
        enable_gqa=True,  # because we shaped query/key states for GQA
        scale=head_dim**-0.5 if scaling is None else scaling,
        return_lse=True,
    )
    attn_output = attn_output[:, :, :seq_len, :].to(dtype=original_dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()  # [B, Q_LEN, H, head_dim]
    attn_output = attn_output.reshape(
        batch_size,
        -1,
        attn_output.shape[2] * attn_output.shape[3],  # merges [H, head_dim]
    )
    return attn_output