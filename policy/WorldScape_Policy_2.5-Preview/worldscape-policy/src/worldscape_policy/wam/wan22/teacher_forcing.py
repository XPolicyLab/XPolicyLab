"""Optional Wan teacher-forcing execution paths, sharing the original backend.

Ported from unified_vla 8379d078 (wsp-2-robodojo), preserving local FPA hooks
and the independently encoded 14B history prefix at the model call sites.

These helpers preserve token visibility and own no parameters or KV cache.
The original attention path remains in model.py and is selected by default.
"""
from __future__ import annotations

import torch


def validate_attention_options(merge: bool, gather: bool) -> None:
    if not isinstance(merge, bool) or not isinstance(gather, bool):
        raise TypeError("teacher-forcing attention options must be booleans")
    if gather and not merge:
        raise ValueError("gather_teacher_forcing_qkv requires merge_teacher_forcing_attention=True")


class _GatherAttentionSpans(torch.autograd.Function):
    """Collect independent attention inputs with one gradient buffer per source.

    ``groups`` contains ordered (start, end) spans along the token dimension.
    Ordinary slice/cat autograd allocates a full source-sized gradient for each
    span. Here all outputs share one backward node; overlapping history spans
    add directly into the same buffer. Only metadata is retained, not activations.
    Attention boundaries and the order of tokens within each group are unchanged.
    """

    @staticmethod
    def forward(ctx, source, groups):
        ctx.groups = groups
        ctx.source_shape = source.shape
        ctx.set_materialize_grads(False)
        return tuple(torch.cat([source[:, start:end] for start, end in spans], dim=1)
                     for spans in groups)

    @staticmethod
    def backward(ctx, *grad_outputs):
        reference = next((grad for grad in grad_outputs if grad is not None), None)
        if reference is None:
            return None, None
        grad_source = reference.new_zeros(ctx.source_shape)
        # A fixed reverse order also avoids atomic scatter accumulation.
        for spans, grad in reversed(list(zip(ctx.groups, grad_outputs))):
            if grad is None:
                continue
            offset = 0
            for start, end in spans:
                length = end - start
                grad_source[:, start:end].add_(grad[:, offset:offset + length])
                offset += length
        return grad_source, None


def merged_image_action_attention(
    self, noisy_image_q, noisy_image_k, noisy_image_v,
    noisy_action_q, noisy_action_k, noisy_action_v,
    clean_image_k, clean_image_v, noisy_state_k, noisy_state_v,
    num_blocks, clean_ctx_k=None, clean_ctx_v=None,
):
    """Share K/V construction and attention for each image/action query pair.

    Both query groups read the same ordered K/V with no causal mask inside
    this call. Keep chunks separate to preserve teacher-forcing visibility.
    The conditioning frame retains its original self-attention path.
    """
    first_k = noisy_image_k[:, :self.frame_seqlen]
    first_v = noisy_image_v[:, :self.frame_seqlen]
    has_context = clean_ctx_k is not None and clean_ctx_v is not None
    if has_context:
        first_k = torch.cat([clean_ctx_k, first_k], dim=1)
        first_v = torch.cat([clean_ctx_v, first_v], dim=1)
    images = [self.attn(noisy_image_q[:, :self.frame_seqlen], first_k, first_v)]
    actions = []
    block_size = self.frame_seqlen * self.num_frame_per_block
    for index in range(num_blocks):
        image_start = self.frame_seqlen + index * block_size
        image_slice = slice(image_start, image_start + block_size)
        action_slice = slice(index * self.num_action_per_block,
                             (index + 1) * self.num_action_per_block)
        state_slice = slice(index * self.num_state_per_block,
                            (index + 1) * self.num_state_per_block)
        keys = [clean_image_k[:, :image_start], noisy_image_k[:, image_slice],
                noisy_action_k[:, action_slice], noisy_state_k[:, state_slice]]
        values = [clean_image_v[:, :image_start], noisy_image_v[:, image_slice],
                  noisy_action_v[:, action_slice], noisy_state_v[:, state_slice]]
        if has_context:
            keys.insert(0, clean_ctx_k)
            values.insert(0, clean_ctx_v)
        query = torch.cat([noisy_image_q[:, image_slice],
                           noisy_action_q[:, action_slice]], dim=1)
        output = self.attn(query, torch.cat(keys, dim=1), torch.cat(values, dim=1))
        images.append(output[:, :block_size])
        actions.append(output[:, block_size:])
    return (torch.cat(images, dim=1),
            torch.cat(actions, dim=1) if actions else noisy_action_q[:, :0])

def gathered_teacher_forcing_attention(self, q, k, v, image_len, num_blocks, context_len=0):
    """Prepare Q/K/V together; keep the existing independent attention calls.

    Input/output layout is [clean context][clean image][noisy image][action]
    [state]. Noisy queries read clean history and their own image/action/state
    block. Clean queries retain the existing local/global attention semantics.
    """
    frame = self.frame_seqlen
    block = frame * self.num_frame_per_block
    clean = context_len
    noisy = clean + image_len
    action = noisy + image_len
    state = action + num_blocks * self.num_action_per_block
    q_groups, kv_groups, causal = [], [], []

    def append(q_spans, kv_spans, is_causal=False):
        q_groups.append(tuple(q_spans))
        kv_groups.append(tuple(kv_spans))
        causal.append(is_causal)

    prefix = [(0, context_len)] if context_len else []
    if context_len:
        append(prefix, prefix)
    append([(clean, clean + frame)], prefix + [(clean, clean + frame)])
    global_clean = self.local_attn_size == -1 and not context_len
    if num_blocks and global_clean:
        append([(clean + frame, noisy)], [(clean, noisy)], True)
    else:
        for index in range(num_blocks):
            start = clean + frame + index * block
            end = min(start + block, noisy)
            history = (clean + frame if self.local_attn_size == -1 else
                       max(clean + frame, end - self.local_attn_size * frame))
            append([(start, end)], prefix + [(clean, clean + frame), (history, end)])
    clean_calls = 1 + (1 if global_clean and num_blocks else num_blocks)

    append([(noisy, noisy + frame)], prefix + [(noisy, noisy + frame)])
    for index in range(num_blocks):
        image_start = noisy + frame + index * block
        action_start = action + index * self.num_action_per_block
        state_start = state + index * self.num_state_per_block
        current = [(image_start, image_start + block),
                   (action_start, action_start + self.num_action_per_block)]
        append(current, prefix + [(clean, clean + frame + index * block)] +
               current + [(state_start, state_start + self.num_state_per_block)])
    if self.num_state_per_block != 1:
        for index in range(num_blocks):
            start = state + index * self.num_state_per_block
            spans = [(start, start + self.num_state_per_block)]
            append(spans, spans)

    queries = _GatherAttentionSpans.apply(q, tuple(q_groups))
    keys = _GatherAttentionSpans.apply(k, tuple(kv_groups))
    values = _GatherAttentionSpans.apply(v, tuple(kv_groups))
    outputs = [
        (self.causal_attn if use_causal else self.attn)(query, key, value)
        for query, key, value, use_causal in zip(queries, keys, values, causal)
    ]
    cursor = int(bool(context_len))
    parts = outputs[:cursor] + outputs[cursor:cursor + clean_calls]
    cursor += clean_calls
    images, actions = [outputs[cursor]], []
    for output in outputs[cursor + 1:cursor + 1 + num_blocks]:
        image_output, action_output = output.split([block, self.num_action_per_block], dim=1)
        images.append(image_output)
        actions.append(action_output)
    parts.extend(images)
    parts.extend(actions)
    if self.num_state_per_block == 1:
        # Singleton state attention is exactly V, as in the merged path.
        parts.append(v[:, state:])
    else:
        parts.extend(outputs[cursor + 1 + num_blocks:])
    return torch.cat(parts, dim=1)
