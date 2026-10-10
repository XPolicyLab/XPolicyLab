"""FSDP2 wrapping, distributed gradient norms and training diagnostics."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard


@dataclass
class DistInfo:
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    device: torch.device

    @property
    def is_master(self) -> bool:
        return self.rank == 0

    @property
    def num_nodes(self) -> int:
        return max(self.world_size // max(self.local_world_size, 1), 1)


def init_distributed() -> DistInfo:
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count() or 1))

    torch.cuda.set_device(local_rank)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return DistInfo(
        rank=rank,
        local_rank=local_rank,
        world_size=world,
        local_world_size=local_world,
        device=torch.device("cuda", local_rank),
    )


def build_mesh(
    info: DistInfo, *, strategy: str = "hsdp", shard_size: int | None = None
) -> object | None:
    """Device mesh for FSDP2. `hsdp` shards in-node and replicates across.

    `shard_size` overrides the shard-group width, which defaults to one node so
    that all-gathers stay on NVLink. Setting it smaller is useful for a model
    that fits in fewer ranks, and for exercising the two-dimensional mesh on a
    single node.
    """
    if info.world_size <= 1:
        return None
    if strategy == "fsdp":
        return init_device_mesh("cuda", (info.world_size,), mesh_dim_names=("shard",))
    if strategy != "hsdp":
        raise ValueError(f"unknown sharding strategy {strategy!r}")

    shard = int(shard_size) if shard_size else min(info.local_world_size, info.world_size)
    shard = min(max(shard, 1), info.world_size)
    if info.world_size % shard:
        raise ValueError(
            f"world size {info.world_size} is not a multiple of shard size {shard}"
        )
    replicate = info.world_size // shard
    return init_device_mesh(
        "cuda", (replicate, shard), mesh_dim_names=("replicate", "shard")
    )


def _backbone_blocks(model: nn.Module, *, include_visual: bool = True) -> list[nn.Module]:
    """Trunk and vision blocks: repeated units of comparable size.

    That is the granularity FSDP wants -- big enough that the all-gather
    amortises, small enough that only one block's parameters are resident at a
    time. Each runs exactly once per step.

    When the visual tower is sharded separately (higher precision), pass
    ``include_visual=False`` so its blocks are not also wrapped here.
    """
    blocks: list[nn.Module] = []
    trunk = getattr(model.backbone.model, "language_model", None)
    if trunk is not None and hasattr(trunk, "layers"):
        blocks += list(trunk.layers)
    if include_visual:
        visual = getattr(model.backbone.model, "visual", None)
        if visual is not None and hasattr(visual, "blocks"):
            blocks += list(visual.blocks)
    return blocks


def _visual_module(model: nn.Module) -> nn.Module | None:
    """The Qwen3-VL vision tower, wrapped as one unit when kept in fp32.

    Wrapping the whole module -- not just ``visual.blocks`` -- is deliberate: the
    overflow that blows the backward up to 1e20+/NaN on certain images lives in
    the non-block leaves too (patch_embed.proj, pos_embed, deepstack_merger),
    which would otherwise fall under the bf16 root unit.
    """
    return getattr(model.backbone.model, "visual", None)


def _expert_blocks(model: nn.Module) -> list[nn.Module]:
    """Expert blocks, which differ from the backbone in one way that matters.

    They are re-entered once per noise draw, so several forwards accumulate into
    the same parameters before any backward runs. Resharding between those
    forwards is where FSDP and repeated entry interact badly, so these are
    sharded separately from the backbone and left gathered.
    """
    return list(model.expert.blocks)


def _context_encoder(model: nn.Module) -> nn.Module | None:
    """The module that builds the context, which runs before the block loop.

    It needs a unit of its own because FSDP gathers a unit's parameters in that
    unit's pre-forward hook, and this work happens outside every other unit's
    forward. Without one, its projections are multiplied while still sharded:
    the master weight reads a pristine 0.0198 and the matmul uses an unfilled
    bf16 buffer of 3.4e38, which surfaces as a NaN loss with nothing in the
    weights or the gradients to explain it.
    """
    return getattr(model.expert, "context", None)


def shard_model(
    model: nn.Module,
    mesh,
    *,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    reshard_after_forward: bool = True,
    visual_param_dtype: torch.dtype | None = None,
) -> nn.Module:
    """Apply FSDP2 block-wise, then to the root.

    ``visual_param_dtype`` (e.g. ``torch.float32``) shards the vision tower as
    its own unit at that precision, leaving the rest of the backbone at
    ``param_dtype``. This is the fix for the deterministic bf16 overflow in the
    vision tower's backward (grad norm 1e20+/NaN on specific images), which FA2
    alone does not cure because it is not confined to attention.
    """
    if mesh is None:
        return model

    policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    kwargs = {"mesh": mesh, "mp_policy": policy}

    include_visual = visual_param_dtype is None
    for block in _backbone_blocks(model, include_visual=include_visual):
        fully_shard(block, reshard_after_forward=reshard_after_forward, **kwargs)

    if visual_param_dtype is not None:
        visual = _visual_module(model)
        if visual is not None:
            vpolicy = MixedPrecisionPolicy(
                param_dtype=visual_param_dtype, reduce_dtype=reduce_dtype
            )
            fully_shard(
                visual,
                reshard_after_forward=reshard_after_forward,
                mesh=mesh,
                mp_policy=vpolicy,
            )
    # Retain expert parameters across repeated noise draws to avoid repeated all-gathers.
    for block in _expert_blocks(model):
        fully_shard(block, reshard_after_forward=False, **kwargs)
    # Built once per step and read by every block afterwards, so it is left
    # gathered like the blocks rather than re-fetched.
    context = _context_encoder(model)
    if context is not None:
        fully_shard(context, reshard_after_forward=False, **kwargs)
    # Shard expert blocks independently; leave shared encoders/decoders outside a parent shard unit.
    fully_shard(model, reshard_after_forward=False, **kwargs)
    return model


def clip_grad_norm(parameters, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
    """Gradient clipping that is correct for sharded (DTensor) gradients."""
    from torch.distributed.tensor import DTensor

    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.zeros(())

    total_norm = torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite=False)
    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()
    torch.nn.utils.clip_grads_with_norm_(
        [p for p in parameters if p.grad is not None], max_norm, total_norm
    )
    return total_norm


def grad_diagnostics(model: nn.Module, mesh) -> dict:
    """Describe the gradient without DTensor, and say where any damage sits."""
    device = next(model.parameters()).device
    finite_sq = torch.zeros((), dtype=torch.float64, device=device)
    finite_max = torch.zeros_like(finite_sq)
    bad = torch.zeros_like(finite_sq)
    n_inf = torch.zeros_like(finite_sq)
    n_nan = torch.zeros_like(finite_sq)
    worst: list[str] = []

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        g = g.to_local() if hasattr(g, "to_local") else g
        if g.numel() == 0:  # a shard group larger than dim 0 leaves ranks empty
            continue
        g = g.double()
        ok = torch.isfinite(g)
        n_bad = int((~ok).sum())
        if n_bad:
            bad += n_bad
            # Report infinity and NaN separately to distinguish overflow from invalid arithmetic.
            n_inf += int(torch.isinf(g).sum())
            n_nan += int(torch.isnan(g).sum())
            worst.append(name)
        finite_sq += torch.where(ok, g, torch.zeros_like(g)).pow(2).sum()
        finite_max = torch.maximum(finite_max, g.abs().nan_to_num(0.0).max())

    if mesh is not None and dist.is_initialized():
        names = getattr(mesh, "mesh_dim_names", None) or ()
        group = mesh["shard"].get_group() if "shard" in names else mesh.get_group()
        dist.all_reduce(finite_sq, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(finite_max, op=dist.ReduceOp.MAX)
    return {
        "norm": float(finite_sq.sqrt()),
        "max": float(finite_max),
        "n_bad": int(bad),
        "n_inf": int(n_inf),
        "n_nan": int(n_nan),
        "worst": worst,
    }


def grad_group_norms(model: nn.Module, mesh) -> dict[str, float]:
    """Gradient norm per architectural group, to localise an explosion."""
    groups = {
        "visual": "backbone.model.visual.",
        "language": "backbone.model.language_model.",
        "context": "expert.context.",
        "expert_blocks": "expert.blocks.",
        "decoders": "expert.decoders.",
        "future": "expert.future",
    }
    device = next(model.parameters()).device
    keys = list(groups) + ["other"]
    sq = {k: torch.zeros((), dtype=torch.float64, device=device) for k in keys}
    bad = {k: torch.zeros((), dtype=torch.float64, device=device) for k in keys}

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        g = g.to_local() if hasattr(g, "to_local") else g
        if g.numel() == 0:
            continue
        key = next((k for k, prefix in groups.items() if name.startswith(prefix)), "other")
        g = g.double()
        ok = torch.isfinite(g)
        sq[key] += torch.where(ok, g, torch.zeros_like(g)).pow(2).sum()
        bad[key] += (~ok).sum()

    if mesh is not None and dist.is_initialized():
        names = getattr(mesh, "mesh_dim_names", None) or ()
        group = mesh["shard"].get_group() if "shard" in names else mesh.get_group()
        stacked = torch.stack([sq[k] for k in keys] + [bad[k] for k in keys])
        dist.all_reduce(stacked, op=dist.ReduceOp.SUM, group=group)
        n = len(keys)
        for i, k in enumerate(keys):
            sq[k], bad[k] = stacked[i], stacked[n + i]
    # The count matters as much as the norm: a group whose gradient overflowed
    # contributes nothing to the norm, so without it the group that broke looks
    # like the quietest one.
    return {k: (float(sq[k].sqrt()), int(bad[k])) for k in keys}


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def all_reduce_mean(value: float, device: torch.device) -> float:
    if not dist.is_initialized():
        return value
    t = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return float(t.item())


def all_reduce_min(value: float, device: torch.device) -> float:
    """Collective AND-like gate: 0 on any rank becomes 0 on every rank."""
    if not dist.is_initialized():
        return value
    t = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return float(t.item())
