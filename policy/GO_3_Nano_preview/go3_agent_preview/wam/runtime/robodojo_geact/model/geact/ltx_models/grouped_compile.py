from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class DiTBlockGroup(nn.Module):
    def __init__(self, blocks, start_index, checkpoint_enabled=False):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.start_mod3 = start_index % 3
        self.checkpoint_enabled = checkpoint_enabled

    def _forward_blocks(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        image_rotary_emb,
        encoder_attention_mask,
        n_view,
        video_attention_mask,
    ):
        for index, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
                n_view=n_view,
                cross_view_attn=(self.start_mod3 + index) % 3 == 0,
                self_attention_mask=video_attention_mask,
            )
        return hidden_states


    def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb,
                encoder_attention_mask, n_view, video_attention_mask):
        arguments = (hidden_states, encoder_hidden_states, temb, image_rotary_emb,
                     encoder_attention_mask, n_view, video_attention_mask)
        if self.checkpoint_enabled and torch.is_grad_enabled():
            return checkpoint(self._forward_blocks, *arguments,
                              use_reentrant=False, preserve_rng_state=True)
        return self._forward_blocks(*arguments)


@torch.compiler.disable
def run_grouped_blocks(
    model,
    role,
    hidden_states,
    encoder_hidden_states,
    temb,
    image_rotary_emb,
    encoder_attention_mask,
    n_view,
    video_attention_mask,
):
    for index, group in enumerate(model.__dict__["_vgen_compiled_groups"][role]):
        arguments = (
            hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb,
            encoder_attention_mask,
            n_view,
            video_attention_mask,
        )
        hidden_states = group(*arguments)
    return hidden_states


def enable_grouped_forward(model, group_size, compile_kwargs, checkpoint_groups=None,
                           share_attention_processor=False, merge_u_batches=False):
    from .transformer_ltx_multiview_vlm_imf_v2 import (
        LTXVideoAttentionProcessor2_0,
        LTXVideoTransformer3DModel,
    )

    if not isinstance(model, LTXVideoTransformer3DModel):
        raise TypeError(
            "Grouped VGen compilation requires the LTX multiview VLM v2 model"
        )
    if type(group_size) is not int or group_size <= 0:
        raise ValueError("vgen_compile_group_size must be a positive integer")
    if model.gradient_checkpointing:
        raise ValueError(
            "Grouped VGen compilation requires gradient_checkpointing=False"
        )
    if compile_kwargs.get("fullgraph", False):
        raise ValueError("Grouped VGen compilation requires fullgraph=False")
    if compile_kwargs.get("dynamic", False):
        raise ValueError("Grouped VGen compilation requires static shapes")
    if "_vgen_compiled_groups" in model.__dict__:
        raise RuntimeError("Grouped VGen compilation is already enabled")

    checkpoint_groups = {} if checkpoint_groups is None else checkpoint_groups
    if not isinstance(checkpoint_groups, dict) or set(checkpoint_groups) - {
        "shared",
        "u",
        "v",
    }:
        raise ValueError("checkpoint_groups must map shared/u/v to group indices")
    shared_count = len(model.transformer_blocks) - model.v_head_depth
    role_sizes = {
        "shared": shared_count,
        "u": model.v_head_depth,
        "v": len(model.v_head_blocks),
    }
    selected = {}
    for role, size in role_sizes.items():
        indices = checkpoint_groups.get(role, [])
        if not isinstance(indices, list | tuple) or any(
            type(i) is not int for i in indices
        ):
            raise ValueError("Checkpoint group indices must be integer lists")
        if len(set(indices)) != len(indices) or any(
            i < 0 or i >= (size + group_size - 1) // group_size for i in indices
        ):
            raise ValueError(f"Invalid checkpoint group indices for {role}: {indices}")
        selected[role] = frozenset(indices)

    before = [
        (name, id(param), tuple(param.shape))
        for name, param in model.named_parameters()
    ]
    state_keys = tuple(model.state_dict())
    attention_modules = [
        attention
        for block in [*model.transformer_blocks, *model.v_head_blocks]
        for attention in (block.attn1, block.attn2)
    ]
    processors = [attention.processor for attention in attention_modules]
    if any(
        type(proc) is not LTXVideoAttentionProcessor2_0 or vars(proc)
        for proc in processors
    ):
        raise ValueError(
            "Grouped compilation requires the stateless LTXVideoAttentionProcessor2_0"
        )
    if share_attention_processor:
        for attention in attention_modules:
            attention.set_processor(processors[0])

    shared = len(model.transformer_blocks) - model.v_head_depth
    stacks = {
        "shared": (list(model.transformer_blocks[:shared]), 0),
        "u": (list(model.transformer_blocks[shared:]), shared),
        "v": (list(model.v_head_blocks), shared),
    }
    groups = {}
    inventory = {}
    group_kwargs = dict(compile_kwargs, dynamic=False, fullgraph=False)
    for role, (blocks, first) in stacks.items():
        groups[role] = []
        inventory[role] = []
        for start in range(0, len(blocks), group_size):
            stop = min(start + group_size, len(blocks))
            group = DiTBlockGroup(
                blocks[start:stop], first + start,
                checkpoint_enabled=start // group_size in selected[role],
            )
            groups[role].append(torch.compile(group, **group_kwargs))
            inventory[role].append(list(range(first + start, first + stop)))

    model.__dict__["_vgen_merged_u_enabled"] = bool(merge_u_batches)
    model.__dict__["_vgen_compiled_groups"] = groups
    model.__dict__["_vgen_checkpoint_groups"] = selected
    eager_forward = model.forward
    model._vgen_eager_forward = eager_forward
    model.forward = torch.compile(
        partial(eager_forward, _use_compiled_block_groups=True),
        **compile_kwargs,
    )
    after = [
        (name, id(param), tuple(param.shape))
        for name, param in model.named_parameters()
    ]
    if before != after or state_keys != tuple(model.state_dict()):
        raise RuntimeError(
            "Grouped compilation changed parameter ownership or checkpoint keys"
        )
    model._vgen_compile_group_audit = {
        "group_size": group_size,
        "merge_u_batches": bool(merge_u_batches),
        "groups": inventory,
        "checkpoint_groups": {
            role: sorted(indices) for role, indices in selected.items()
        },
        "checkpoint_blocks": {
            role: [inventory[role][i] for i in sorted(indices)]
            for role, indices in selected.items()
        },
        "parameters_identity_and_order_preserved": True,
        "checkpoint_keys_preserved": True,
        "jvp_eager_forward_preserved": model._vgen_eager_forward == eager_forward,
        "processor_instances_before": len({id(proc) for proc in processors}),
        "processor_instances_after": len(
            {id(attention.processor) for attention in attention_modules}
        ),
    }
    return model._vgen_compile_group_audit
