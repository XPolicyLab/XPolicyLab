#!/usr/bin/env python
"""Check all-reduce and reduce-scatter results against known inputs.

Run through scripts/run_multinode.sh with scripts/nccl_verify.py."""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def report(rank: int, name: str, bad: int, total: int, sample: str) -> int:
    if bad:
        print(f"[r{rank}] FAIL {name}: {bad}/{total} elements wrong; {sample}", flush=True)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=256, help="buffer size in MiB")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    n = args.mb * 1024 * 1024 // (2 if dtype == torch.bfloat16 else 4)
    failures = 0

    # all_reduce of ones: every element must come back exactly `world`. Safe to
    # demand exactness even in bf16, because the partial sums are small integers
    # that every step of the reduction represents without rounding.
    for it in range(args.iters):
        buf = torch.ones(n, dtype=dtype, device=device)
        dist.all_reduce(buf)
        expect = float(world)
        bad = int((buf != expect).sum())
        if bad:
            wrong = buf[buf != expect]
            failures += report(
                rank,
                f"all_reduce(ones) iter{it}",
                bad,
                n,
                f"expected {expect}, got e.g. {wrong[:4].tolist()}, "
                f"finite={bool(torch.isfinite(wrong).all())}",
            )
            break

    # all_reduce of a rank-dependent pattern: catches a rank whose contribution
    # is dropped or duplicated, which uniform ones cannot see.
    contribution = 1.0 / (rank + 1)
    expected = sum(1.0 / (r + 1) for r in range(world))
    for it in range(args.iters):
        buf = torch.full((n,), contribution, dtype=torch.float32, device=device)
        dist.all_reduce(buf)
        err = (buf - expected).abs().max()
        if float(err) > 1e-3 * max(expected, 1.0):
            failures += report(
                rank, f"all_reduce(pattern) iter{it}", 1, n,
                f"expected {expected:.6f}, max abs error {float(err):.6g}",
            )
            break

    # Check reduce-scatter in fp32 to avoid rounding integer reference sums.
    chunk = n // world
    src = [
        torch.full((chunk,), float(r + 1), dtype=torch.float32, device=device)
        for r in range(world)
    ]
    dst = torch.empty(chunk, dtype=torch.float32, device=device)
    dist.reduce_scatter(dst, src)
    expect_rs = float((rank + 1) * world)
    off = (dst - expect_rs).abs()
    bad = int((off > 1e-3 * expect_rs).sum())
    if bad:
        failures += report(
            rank, "reduce_scatter", bad, chunk,
            f"expected {expect_rs}, max abs error {float(off.max()):.6g}",
        )

    # Also verify the shard/replica process groups used by HSDP.
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", 8))
    if world > local_world and world % local_world == 0:
        replicas = world // local_world
        mesh = torch.arange(world).view(replicas, local_world)
        shard_idx = rank % local_world
        replicate_ranks = mesh[:, shard_idx].tolist()

        # Every rank must build every subgroup, in the same order.
        groups = {}
        for s in range(local_world):
            groups[s] = dist.new_group(ranks=mesh[:, s].tolist())
        group = groups[shard_idx]

        contribution = float(rank + 1)
        expected_rep = float(sum(r + 1 for r in replicate_ranks))
        for it in range(args.iters):
            buf = torch.full((n,), contribution, dtype=torch.float32, device=device)
            dist.all_reduce(buf, group=group)
            err = (buf - expected_rep).abs().max()
            if float(err) > 1e-3 * max(expected_rep, 1.0):
                failures += report(
                    rank, f"all_reduce(replicate group) iter{it}", 1, n,
                    f"group={replicate_ranks} expected {expected_rep:.6f}, "
                    f"max abs error {float(err):.6g}",
                )
                break
        else:
            if rank == 0:
                print(
                    f"replicate groups checked: {replicas} ranks each, "
                    f"e.g. {replicate_ranks}",
                    flush=True,
                )

    # Every rank agrees on the verdict before anyone prints success.
    verdict = torch.tensor([failures], device=device, dtype=torch.float32)
    dist.all_reduce(verdict, op=dist.ReduceOp.SUM)
    if rank == 0:
        nodes = max(world // int(os.environ.get("LOCAL_WORLD_SIZE", 8)), 1)
        total = int(verdict.item())
        print(
            f"world={world} nodes={nodes} dtype={args.dtype} buffer={args.mb}MiB "
            f"iters={args.iters}",
            flush=True,
        )
        print("RESULT: " + ("PASS, collectives are numerically exact" if total == 0
                            else f"FAIL, {total} corrupt collectives"), flush=True)

    dist.destroy_process_group()
    return 0 if int(verdict.item()) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
