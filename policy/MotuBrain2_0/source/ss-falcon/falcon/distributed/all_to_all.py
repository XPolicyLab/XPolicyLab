# Copyright (c) Microsoft Corporation and Jiarui Fang
# SPDX-License-Identifier: Apache-2.0


from typing import Any

import torch
import torch.distributed as dist


def all_to_all_3D(
    input: torch.Tensor,
    scatter_idx: int = 1,
    gather_idx: int = 0,
    group: dist.ProcessGroup | None = None,
    use_sync: bool = False,
) -> torch.Tensor:
    assert input.dim() == 3, f"input must be 3D tensor, got {input.dim()} and shape {input.shape}"
    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 1 and gather_idx == 0:
        # input (torch.tensor): a tensor sharded along dim 1 (seqlen/P, hc, hs) output: (seqlen, hc/P, hs)
        shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (seqlen/P, hc, hs) -reshape-> (seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, hc/P, hs)
        input_t = input.reshape(shard_seqlen, seq_world_size, shard_hc, hs).transpose(0, 1).contiguous()

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, hc/P, hs) scatter head

        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, shard_hc, hs)

        # (seq_len, hc/P, hs) -reshape-> (seq_len, hc/P, hs)
        output = output.contiguous().reshape(seqlen, shard_hc, hs)

        return output

    elif scatter_idx == 0 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (seqlen, hc/P, hs) output: (seqlen/P, hc, hs)
        seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        # (seqlen, hc/P, hs) -> P, seqlen/P, hc/P, hs -> P, hc/P, seqlen/P, hs
        input_t = input.reshape(seq_world_size, shard_seqlen, shard_hc, hs).permute(0, 2, 1, 3).contiguous()
        # input_t = einops.rearrange(
        #     input,
        #     "seqlen (hc/P P) hs -> P hc/P seqlen/P hs",
        #     P=seq_world_size,
        # )
        # (
        #     input.reshape(seq_world_size, shard_seqlen, shard_hc, hs)
        #     .transpose(0, 2)
        #     .transpose(0, 1)
        #     .contiguous()
        #     .reshape(seq_world_size, shard_seqlen, shard_hc, hs)
        # )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, hs)

        # (hc, seqlen/N, hs) -tranpose(0,2)-> (seqlen/N, hc, hs)
        output = output.transpose(0, 1).contiguous().reshape(shard_seqlen, hc, hs)

        return output
    else:
        raise RuntimeError("scatter_idx must be 0 or 1 and gather_idx must be 0 or 1")


class SeqAllToAll3D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: torch.Tensor,
        scatter_idx: int,
        gather_idx: int,
        use_sync: bool = False,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.use_sync = use_sync
        return all_to_all_3D(input, scatter_idx, gather_idx, group=group, use_sync=use_sync)

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> tuple[None, torch.Tensor, None, None]:
        return (
            None,
            SeqAllToAll3D.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx, ctx.use_sync),
            None,
            None,
            None,
        )


def all_to_all_2_or_3D(
    input: torch.Tensor,
    scatter_idx: int = 1,
    gather_idx: int = 0,
    group: dist.ProcessGroup | None = None,
    use_sync: bool = False,
) -> torch.Tensor:
    if group is None:
        return input

    if input.dim() == 2:
        return SeqAllToAll3D.apply(group, input.unsqueeze(2), scatter_idx, gather_idx, use_sync).squeeze(2)
    elif input.dim() == 3:
        return SeqAllToAll3D.apply(group, input, scatter_idx, gather_idx, use_sync)
    else:
        raise RuntimeError("input must be 2D or 3D tensor")
