"""Parameter-free temporal V mixing with an unchanged current-only LM sequence.

Past head images traverse shared decoder blocks as independent 49-token rows.
Only full-attention blocks mix V across time (at matching spatial locations).
All state is explicit so non-reentrant checkpoint recomputation is safe.

Time is encoded as a RELATIVE rotation on the temporal branch's Q/K, over the
host model's own rotary width and in history-slot units. See
`docs/history_probe_separability_2026-10-20.md` section 10 for the measurement
that chose this over the additive frame code of
`models/qwen_spacetime.py` / `hy_vla/space_time_attention.py` and over the
native mRoPE temporal section.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


MODE = "vlm_history_value_v1"

# Rotary base for the temporal branch. 10000 keeps roughly half of the host's
# frequency pairs inside a usable angle for 8 slots while leaving the oldest
# frame's weight at the no-code level; 100 buys more order information but
# halves the oldest-frame weight, which is the wrong direction for memory
# tasks. Overridable via `qwen_history_value_rope_theta`.
DEFAULT_ROPE_THETA = 10000.0


def temporal_rotary_dim(attn) -> int:
    """Rotary width of the host attention, so content dims stay unrotated.

    Qwen 3.5 rotates only `head_dim * partial_rotary_factor` dims (64 of 256 on
    the 2B); the rest is pure content matching. The temporal branch mirrors
    that split instead of rotating the whole head, which is what the first
    implementation did and what destroyed most of the cross-frame content dot
    product.
    """
    params = getattr(attn.config, "rope_parameters", None) or {}
    factor = float(params.get("partial_rotary_factor", 1.0))
    dim = int(attn.head_dim * factor) // 2 * 2
    if dim < 2 or dim > attn.head_dim:
        raise ValueError(
            f"Unusable temporal rotary width {dim} for head_dim {attn.head_dim}.")
    return dim


def apply_time_rope(x, offsets, rotary_dim: int, theta: float):
    """Rotate-half RoPE over `offsets` on the leading `rotary_dim` dims.

    `x` is `[B,T,N,H,d]` and `offsets` is `[B,T]` in history-slot units, signed
    with 0 at the current frame. Because the rotation is orthogonal and applied
    to both Q and K, the resulting time-attention logits depend only on
    `offsets[tau] - offsets[t]`, so the encoding is relative and invariant to a
    constant shift of all offsets. At `offsets == 0` the rotation is the
    identity, which is what keeps a no-history sample bit-exact.
    """
    rotated, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    freq = theta ** (
        -torch.arange(0, rotary_dim, 2, device=x.device).float() / rotary_dim)
    angle = offsets.float()[..., None] * freq
    angle = torch.cat((angle, angle), dim=-1)[:, :, None, None, :]
    left, right = rotated.chunk(2, dim=-1)
    swapped = torch.cat((-right, left), dim=-1)
    rotated = rotated * angle.cos().to(x.dtype) + swapped * angle.sin().to(x.dtype)
    return torch.cat((rotated, passthrough), dim=-1)


def mix_temporal_values(attn, normalized_frames, frame_mask, offsets,
                        theta: float = DEFAULT_ROPE_THETA):
    """[B,T,N,D] -> [B,T,N,Hkv,d], causal time attention per patch.

    `normalized_frames` is the layer's post-RMSNorm input, used unchanged for
    Q, K and V; time enters only as the relative rotation on Q/K. V therefore
    stays uncoded for free, so the mixed V is a convex combination of the
    frames' ORIGINAL V and no new direction enters the warm-started main
    attention.

    Every query head keeps its own softmax and the grouped probabilities are
    averaged afterwards. A GQA group shares one V, so this is exactly
    "average the attention weights", and it avoids the magnitude loss (and the
    resulting flatter softmax) of averaging the query vectors themselves.
    """
    b, t, n, _ = normalized_frames.shape
    d = attn.head_dim
    groups = attn.num_key_value_groups
    rotary_dim = temporal_rotary_dim(attn)
    q, _ = attn.q_proj(normalized_frames).reshape(b, t, n, -1, 2 * d).chunk(2, -1)
    q = apply_time_rope(attn.q_norm(q), offsets, rotary_dim, theta)
    k = attn.k_norm(attn.k_proj(normalized_frames).reshape(b, t, n, -1, d))
    k = apply_time_rope(k, offsets, rotary_dim, theta)
    v = attn.v_proj(normalized_frames).reshape(b, t, n, -1, d).permute(0, 2, 3, 1, 4)
    q = q.permute(0, 2, 3, 1, 4)
    k = k.permute(0, 2, 3, 1, 4)
    if groups > 1:
        k = k.repeat_interleave(groups, dim=2)
        v = v.repeat_interleave(groups, dim=2)
    causal = offsets[:, None, :] <= offsets[:, :, None]
    allowed = causal & frame_mask[:, None, :]
    # Padded history rows are still computed; they are never read as keys or
    # values, so their output is discarded. Forcing their own slot open is a
    # no-op for valid queries (whose diagonal is already open) and guarantees
    # no query row is fully masked, which would make the softmax emit NaN and
    # poison the checkpointed backward.
    allowed = allowed | torch.eye(t, device=offsets.device, dtype=torch.bool)
    mixed = F.scaled_dot_product_attention(
        q, k, v, attn_mask=allowed[:, None, None], dropout_p=0.0,
        scale=attn.scaling,
    )
    mixed = mixed.reshape(b, n, -1, groups, t, d).mean(3)
    return mixed.permute(0, 3, 1, 2, 4)


def _run_layer(layer, states, *, values=None, positions=None, **kwargs):
    """Replace only the original V projection output during this layer call."""
    handle = None
    if values is not None:
        def replace_value(_module, _inputs, output):
            replacement = values.flatten(-2)
            if positions is None:
                return replacement
            result = output.clone()
            rows = torch.arange(output.shape[0], device=output.device)[:, None]
            result[rows, positions] = replacement
            return result

        handle = layer.self_attn.v_proj.register_forward_hook(replace_value)
    try:
        # Bypass the block's automatic checkpoint wrapper: checkpoint the whole
        # two-stream step instead, including creation/removal of the V hook.
        return layer.forward(states, **kwargs)
    finally:
        if handle is not None:
            handle.remove()


def forward_history_value(language_model, *, inputs_embeds, attention_mask,
                          position_ids, history, history_mask, offsets,
                          head_positions, selected_layers,
                          rope_theta: float = DEFAULT_ROPE_THETA,
                          return_prefix_kv: bool = False):
    """Return selected main-stream states; history never reaches action directly."""
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

    b, h, n, d = history.shape
    if h == 0:
        raise ValueError("Use the ordinary language forward for single-frame input.")
    # Mirror Qwen3.5TextModel's public position layout exactly.
    if position_ids.ndim != 3 or position_ids.shape[0] not in (3, 4):
        raise ValueError("History V fusion requires Qwen 3D position IDs.")
    text_positions = position_ids[0] if position_ids.shape[0] == 4 else None
    rope_positions = position_ids[-3:]
    main_masks = dict(
        config=language_model.config, inputs_embeds=inputs_embeds,
        attention_mask=attention_mask, past_key_values=None, position_ids=text_positions,
    )
    masks = {
        "full_attention": create_causal_mask(**main_masks),
        "linear_attention": create_recurrent_attention_mask(**main_masks),
    }
    main_rope = language_model.rotary_emb(inputs_embeds, rope_positions)
    gather = head_positions[None].expand(3, -1, -1)
    side_positions = torch.gather(rope_positions, 2, gather)
    side_positions = side_positions[:, :, None].expand(-1, -1, h, -1).reshape(3, b * h, n)
    side_states = history.reshape(b * h, n, d)
    side_rope = language_model.rotary_emb(side_states, side_positions)
    side_mask = create_causal_mask(
        config=language_model.config, inputs_embeds=side_states,
        attention_mask=torch.ones((b * h, n), device=history.device, dtype=torch.long),
        past_key_values=None, position_ids=None,
    )
    rows = torch.arange(b, device=history.device)[:, None]
    frame_mask = torch.cat((history_mask, history_mask.new_ones(b, 1)), dim=1)
    states = inputs_embeds
    selected = {}
    prefix_kv = [None] * len(language_model.layers)
    for index, layer in enumerate(language_model.layers):
        def step(main, past, block=layer):
            main_values = past_values = None
            key = value = None
            if block.block_type == "full_attention":
                head = main[rows, head_positions]
                frames = torch.cat((past.reshape(b, h, n, d), head[:, None]), dim=1)
                mixed = mix_temporal_values(
                    block.self_attn, block.input_layernorm(frames), frame_mask,
                    offsets, rope_theta,
                )
                main_values = mixed[:, -1]
                past_values = mixed[:, :-1].reshape(b * h, n, *mixed.shape[-2:])
                if return_prefix_kv:
                    from XPolicyLab.policy.Era0_Preview.runtime.models.vlm_backends.qwen3_5 import Qwen35Backend

                    # Publish only the main stream's actual K/V, including
                    # the mixed head V. Side-stream hooks would overwrite it;
                    # reprojecting after the forward would lose the fusion.
                    key, value = Qwen35Backend._project_qwen_full_attn_kv(
                        block.self_attn, block.input_layernorm(main),
                        {"position_embeddings": main_rope},
                    )
                    value = value.transpose(1, 2).clone()
                    value[rows, head_positions] = main_values
                    value = value.transpose(1, 2)
            main = _run_layer(
                block, main, values=main_values, positions=head_positions,
                position_embeddings=main_rope, attention_mask=masks[block.block_type],
                position_ids=text_positions, past_key_values=None, use_cache=False,
            )
            past = _run_layer(
                block, past, values=past_values,
                position_embeddings=side_rope,
                attention_mask=side_mask if block.block_type == "full_attention" else None,
                position_ids=None, past_key_values=None, use_cache=False,
            )
            return main, past, key, value

        if language_model.training and getattr(layer, "gradient_checkpointing", False):
            states, side_states, key, value = checkpoint(step, states, side_states, use_reentrant=False)
        else:
            states, side_states, key, value = step(states, side_states)
        if key is not None:
            prefix_kv[index] = (key, value)
        if index in selected_layers[:-1]:
            selected[index] = states
    result = [selected[i] for i in selected_layers[:-1]] + [language_model.norm(states)]
    return (result, prefix_kv) if return_prefix_kv else result
