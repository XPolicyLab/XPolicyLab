"""
Sequence Parallelism Communication Operations

SP communication operations: sharding, gathering, all-to-all, etc.
"""

import torch
import torch.distributed as dist
from typing import Tuple
from .parallel_state import get_sp_group, get_sp_world_size, get_sp_rank


class _Shard(torch.autograd.Function):
    """
    Differentiable shard operation.

    Forward: Take only this rank's portion of the input tensor.
    Backward: All-gather gradients from all ranks.
    """

    @staticmethod
    def forward(
        ctx, input_: torch.Tensor, dim: int, padded_seq_len: int
    ) -> torch.Tensor:
        sp_world_size = get_sp_world_size()
        sp_rank = get_sp_rank()
        sp_group = get_sp_group()

        ctx.dim = dim
        ctx.sp_world_size = sp_world_size
        ctx.sp_group = sp_group
        ctx.padded_seq_len = padded_seq_len

        # Shard: take only this rank's portion
        elements_per_rank = padded_seq_len // sp_world_size
        start_idx = sp_rank * elements_per_rank
        end_idx = (sp_rank + 1) * elements_per_rank

        slices = [slice(None)] * input_.dim()
        slices[dim] = slice(start_idx, end_idx)
        return input_[tuple(slices)].contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        # Backward of shard is all-gather
        grad_input = _gather_impl(grad_output, ctx.dim, ctx.sp_world_size, ctx.sp_group)
        return grad_input, None, None


def sp_shard(
    input_: torch.Tensor,
    dim: int = 1,
) -> Tuple[torch.Tensor, int]:
    """
    Differentiable shard of the input tensor along the specified dimension across SP ranks.

    Args:
        input_: Input tensor, shape like [B, S, D]
        dim: Dimension to shard, default 1 (sequence dimension)

    Returns:
        tuple: (sharded tensor, original sequence length)

    Example:
        # SP=4, input [B, 1000, D]
        # output [B, 256, D] (padded to 1024, then 256 per rank)
    """
    sp_world_size = get_sp_world_size()

    if sp_world_size == 1:
        return input_, input_.shape[dim]

    original_seq_len = input_.shape[dim]

    # Calculate padding (ensure divisible by sp_size)
    remainder = original_seq_len % sp_world_size
    if remainder != 0:
        padding_amount = sp_world_size - remainder
        padded_seq_len = original_seq_len + padding_amount

        # Create padding (this is differentiable - just concat with zeros)
        pad_shape = list(input_.shape)
        pad_shape[dim] = padding_amount
        padding = torch.zeros(pad_shape, dtype=input_.dtype, device=input_.device)
        input_ = torch.cat([input_, padding], dim=dim)
    else:
        padded_seq_len = original_seq_len

    # Differentiable shard
    output = _Shard.apply(input_, dim, padded_seq_len)

    return output, original_seq_len


def _gather_impl(
    input_: torch.Tensor,
    dim: int,
    sp_world_size: int,
    sp_group,
) -> torch.Tensor:
    """
    Internal implementation of all-gather (non-differentiable).
    """
    gathered_list = [torch.empty_like(input_) for _ in range(sp_world_size)]
    dist.all_gather(gathered_list, input_, group=sp_group)
    return torch.cat(gathered_list, dim=dim)


def _split_impl(
    input_: torch.Tensor,
    dim: int,
    sp_world_size: int,
    sp_rank: int,
) -> torch.Tensor:
    """
    Internal implementation of split/scatter (non-differentiable).
    Takes only the slice corresponding to the current rank.
    """
    dim_size = input_.shape[dim]
    assert (
        dim_size % sp_world_size == 0
    ), f"dim_size ({dim_size}) must be divisible by sp_size ({sp_world_size})"

    chunk_size = dim_size // sp_world_size
    start_idx = sp_rank * chunk_size
    end_idx = (sp_rank + 1) * chunk_size

    slices = [slice(None)] * input_.dim()
    slices[dim] = slice(start_idx, end_idx)
    return input_[tuple(slices)].contiguous()


class _AllGather(torch.autograd.Function):
    """
    Differentiable all-gather operation.

    Forward: Gathers shards from all ranks along specified dimension.
    Backward: Splits gradient and returns only this rank's portion.
    """

    @staticmethod
    def forward(ctx, input_: torch.Tensor, dim: int) -> torch.Tensor:
        sp_world_size = get_sp_world_size()
        sp_group = get_sp_group()
        sp_rank = get_sp_rank()
        ctx.dim = dim
        ctx.sp_world_size = sp_world_size
        ctx.sp_rank = sp_rank
        return _gather_impl(input_, dim, sp_world_size, sp_group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # Backward of all-gather is split: take only this rank's portion
        grad_input = _split_impl(grad_output, ctx.dim, ctx.sp_world_size, ctx.sp_rank)
        return grad_input, None


def sp_gather(
    input_: torch.Tensor,
    original_seq_len: int,
    dim: int = 1,
) -> torch.Tensor:
    """
    Differentiable gather of tensors from all SP ranks and remove padding.

    Args:
        input_: Sharded tensor
        original_seq_len: Original sequence length
        dim: Gather dimension

    Returns:
        Gathered tensor
    """
    sp_world_size = get_sp_world_size()

    if sp_world_size == 1:
        return input_

    # Differentiable all-gather
    gathered = _AllGather.apply(input_, dim)

    # Remove padding (this is differentiable - just slicing)
    if gathered.shape[dim] > original_seq_len:
        slices = [slice(None)] * gathered.dim()
        slices[dim] = slice(0, original_seq_len)
        gathered = gathered[tuple(slices)].contiguous()

    return gathered


def reduce_sp_partial_grads(named_parameters, is_sharded, sp_group=None) -> int:
    """Sum partial grads of SP-sharded params across the sequence-parallel group.

    Under sequence parallelism (sp_size>1) the params selected by ``is_sharded``
    run on this rank's sequence shard, so their ``.grad`` is partial and must be
    summed over the SP group. Params trained on the gathered full sequence hold
    the identical full grad on every SP rank and must be excluded (summing would
    over-count them by sp_size). The orthogonal DP reduction stays with FSDP's
    reduce-scatter. The many small block-grad shards are coalesced into one flat
    all-reduce per dtype (~1 collective) instead of one NCCL launch per param.

    Args:
        named_parameters: iterable of (name, param), e.g. model.named_parameters().
        is_sharded: predicate on the param name; True => SP-sharded (partial grad).
        sp_group: SP process group; defaults to get_sp_group().

    Returns:
        Number of params whose grad was summed (0 when SP is off or none matched).
    """
    if get_sp_world_size() <= 1:
        return 0
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

    if sp_group is None:
        sp_group = get_sp_group()
    by_dtype = {}
    for name, p in named_parameters:
        if p.grad is None or not is_sharded(name):
            continue
        local = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
        by_dtype.setdefault(local.dtype, []).append(local)
    for grads in by_dtype.values():
        flat = _flatten_dense_tensors(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=sp_group)
        for t, synced in zip(grads, _unflatten_dense_tensors(flat, grads)):
            t.copy_(synced)
    return sum(len(v) for v in by_dtype.values())


def _all_to_all_4d_impl(
    input_: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    sp_world_size: int,
    sp_group,
) -> torch.Tensor:
    """
    Internal implementation of 4D all-to-all (non-differentiable).

    For seq->head (scatter_dim=2, gather_dim=1):
    - Input: [B, S/sp, H, D] -> Output: [B, S, H/sp, D]

    For head->seq (scatter_dim=1, gather_dim=2):
    - Input: [B, S, H/sp, D] -> Output: [B, S/sp, H, D]
    """
    if sp_world_size == 1:
        return input_

    if scatter_dim == 2 and gather_dim == 1:
        bs, shard_seqlen, h, d = input_.shape
        seq_len = shard_seqlen * sp_world_size
        shard_h = h // sp_world_size
        sp = sp_world_size
        input_ = input_.view(bs, shard_seqlen, sp, shard_h, d)
        input_ = input_.permute(
            2, 0, 1, 3, 4
        ).contiguous()  # sp, bs, shard_seqlen, shard_h, d
        output = torch.empty_like(input_)
        dist.all_to_all_single(
            output, input_, group=sp_group
        )  # sp, bs, shard_seqlen, shard_h, d

        output = output.permute(
            1, 0, 2, 3, 4
        ).contiguous()  # bs, sp, shard_seqlen, shard_h, d
        output = output.view(bs, seq_len, shard_h, d)  # bs, seq_len, h, d

    elif scatter_dim == 1 and gather_dim == 2:
        bs, seq_len, shard_h, d = input_.shape
        h = shard_h * sp_world_size
        shard_seqlen = seq_len // sp_world_size
        sp = sp_world_size
        input_ = input_.view(bs, sp, shard_seqlen, shard_h, d)
        input_ = input_.permute(
            1, 0, 2, 3, 4
        ).contiguous()  # sp, bs, shard_seqlen, shard_h, d
        output = torch.empty_like(input_)
        dist.all_to_all_single(
            output, input_, group=sp_group
        )  # sp, bs, shard_seqlen, shard_h, d
        output = output.permute(
            1, 2, 0, 3, 4
        ).contiguous()  # bs, shard_seqlen, sp, shard_h, d
        output = output.view(bs, shard_seqlen, h, d)  # bs, shard_seqlen, h, d

    return output


class _SeqToHeadAllToAll(torch.autograd.Function):
    """
    Differentiable all-to-all: sequence sharding -> head sharding.

    Forward: [B, S/sp, H, D] -> [B, S, H/sp, D]
    Backward: [B, S, H/sp, D] -> [B, S/sp, H, D]
    """

    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        sp_world_size = get_sp_world_size()
        sp_group = get_sp_group()
        ctx.sp_world_size = sp_world_size
        ctx.sp_group = sp_group
        return _all_to_all_4d_impl(
            input_,
            scatter_dim=2,
            gather_dim=1,
            sp_world_size=sp_world_size,
            sp_group=sp_group,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # Backward of seq->head is head->seq
        return _all_to_all_4d_impl(
            grad_output,
            scatter_dim=1,
            gather_dim=2,
            sp_world_size=ctx.sp_world_size,
            sp_group=ctx.sp_group,
        )


class _HeadToSeqAllToAll(torch.autograd.Function):
    """
    Differentiable all-to-all: head sharding -> sequence sharding.

    Forward: [B, S, H/sp, D] -> [B, S/sp, H, D]
    Backward: [B, S/sp, H, D] -> [B, S, H/sp, D]
    """

    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        sp_world_size = get_sp_world_size()
        sp_group = get_sp_group()
        ctx.sp_world_size = sp_world_size
        ctx.sp_group = sp_group
        return _all_to_all_4d_impl(
            input_,
            scatter_dim=1,
            gather_dim=2,
            sp_world_size=sp_world_size,
            sp_group=sp_group,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # Backward of head->seq is seq->head
        return _all_to_all_4d_impl(
            grad_output,
            scatter_dim=2,
            gather_dim=1,
            sp_world_size=ctx.sp_world_size,
            sp_group=ctx.sp_group,
        )


def sp_all_to_all_4d(
    input_: torch.Tensor,
    scatter_dim: int = 2,
    gather_dim: int = 1,
) -> torch.Tensor:
    """
    Differentiable all-to-all communication for 4D tensors.

    In attention computation:
    - Input: [B, S/sp, H, D] (sequence sharded)
    - Output: [B, S, H/sp, D] (head sharded)

    Args:
        input_: Input tensor [B, S_local, H, D]
        scatter_dim: Scatter dimension (usually the head dimension)
        gather_dim: Gather dimension (usually the sequence dimension)

    Returns:
        Rearranged tensor
    """
    sp_world_size = get_sp_world_size()
    if sp_world_size == 1:
        return input_

    return _SeqToHeadAllToAll.apply(input_)


def sp_all_gather_4d(
    input_: torch.Tensor,
    scatter_dim: int = 1,
    gather_dim: int = 2,
) -> torch.Tensor:
    """
    Differentiable reverse all-to-all for sp_all_to_all_4d.

    After attention computation:
    - Input: [B, S, H/sp, D] (head sharded)
    - Output: [B, S/sp, H, D] (sequence sharded)
    """
    sp_world_size = get_sp_world_size()
    if sp_world_size == 1:
        return input_

    return _HeadToSeqAllToAll.apply(input_)
