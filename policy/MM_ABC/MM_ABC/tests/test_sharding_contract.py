"""Guards on the arrangement FSDP requires, which is easy to break by accident.

FSDP makes a unit's parameters whole in that unit's pre-forward hook. A
parameter reached from anywhere else is still sharded, and the failure is
silent: the master weight reads correctly while the matmul consumes an unfilled
bf16 buffer, so the loss goes NaN with nothing in the weights or the gradients
pointing at a cause. It cost a long hunt once; these tests make the same
mistake fail immediately and locally instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mmabc.models.joint_expert import ContextEncoder, MMABCJointExpert
from mmabc.train.fsdp import _context_encoder, _expert_blocks

CONTEXT_DIM = 32
HIDDEN = 16
BLOCKS = 3
INJECT = (0, 1)


def _expert() -> MMABCJointExpert:
    return MMABCJointExpert(
        head_slices={"manip": (0, 58), "aux": (58, 80)},
        chunk_size=4,
        near_steps=2,
        hidden=HIDDEN,
        num_blocks=BLOCKS,
        num_heads=2,
        ffn=32,
        dropout=0.0,
        context_dim=CONTEXT_DIM,
        injection_blocks=INJECT,
        future_tokens=2,
        future_dim=8,
        decoder_hidden=8,
    )


def test_context_work_happens_inside_a_forward():
    """The pre-block context must be built by a module call, not a bare method.

    ``build_context_cache`` runs before ``expert.forward``, so any parameter it
    touches directly is sharded at that moment. Routing it through a module
    means FSDP's hook fires and hands it whole parameters instead.
    """
    expert = _expert()
    memory = [torch.randn(2, 5, CONTEXT_DIM) for _ in range(len(INJECT) + 1)]

    seen = []
    expert.context.register_forward_hook(lambda *_: seen.append(True))
    cache = expert.build_context_cache(memory)

    assert seen, "build_context_cache bypassed ContextEncoder.__call__"
    assert len(cache) == BLOCKS


def test_every_pre_block_parameter_belongs_to_the_context_encoder():
    """Nothing used before the block loop may sit outside that unit.

    A parameter left behind here is exactly the silent-corruption case: it is
    owned by an enclosing unit whose forward has not run when it is read.
    """
    expert = _expert()
    owned = {id(p) for p in expert.context.parameters()}

    memory = [torch.randn(2, 5, CONTEXT_DIM) for _ in range(len(INJECT) + 1)]
    touched: list[str] = []

    def watch(name):
        def hook(module, _args, _out):
            for pname, p in module.named_parameters(recurse=False):
                if id(p) not in owned:
                    touched.append(f"{name}.{pname}")

        return hook

    handles = [m.register_forward_hook(watch(n)) for n, m in expert.named_modules()]
    try:
        expert.build_context_cache(memory)
    finally:
        for h in handles:
            h.remove()

    assert not touched, f"parameters used before the block loop but outside it: {touched}"


def test_sharding_plan_covers_the_context_encoder():
    """The plan must give that module a unit; otherwise the hook never fires."""
    model = nn.Module()
    model.expert = _expert()

    context = _context_encoder(model)
    assert isinstance(context, ContextEncoder)

    # Blocks and the context encoder must be disjoint, or a parameter would be
    # claimed by two units.
    block_params = {id(p) for b in _expert_blocks(model) for p in b.parameters()}
    context_params = {id(p) for p in context.parameters()}
    assert not (block_params & context_params)
