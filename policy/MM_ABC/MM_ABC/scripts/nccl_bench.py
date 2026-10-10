#!/usr/bin/env python
"""Measure distributed collective throughput.

Run through scripts/run_multinode.sh with scripts/nccl_bench.py."""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist

# Parameters exchanged per step under HSDP: gradients are reduce-scattered
# within the shard group, then all-reduced across replicas over 1/shard of the
# parameters. This is that per-replica share for the 4B configuration.
MODEL_PARAMS = 5.27e9


def bench(nbytes: int, iters: int, warmup: int, device: torch.device) -> float:
    """Return achieved algorithm bandwidth in GB/s for an all-reduce."""
    n = nbytes // 2  # bf16
    buf = torch.ones(n, dtype=torch.bfloat16, device=device)
    for _ in range(warmup):
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return nbytes * iters / elapsed / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    nodes = max(world // max(local_world, 1), 1)
    device = torch.device("cuda", local_rank)

    if rank == 0:
        print(f"world={world}  nodes={nodes}  gpus/node={local_world}")
        print(f"NCCL_IB_HCA={os.environ.get('NCCL_IB_HCA')}")
        print(f"NCCL_IB_GID_INDEX={os.environ.get('NCCL_IB_GID_INDEX')}")
        print(f"NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME')}")
        print(f"\n{'size':>10s} {'GB/s':>9s}")

    results = {}
    for mb in (4, 32, 128, 512):
        bw = bench(mb * 1024 * 1024, args.iters, args.warmup, device)
        results[mb] = bw
        if rank == 0:
            print(f"{mb:8d}MB {bw:9.2f}")

    if rank == 0:
        peak = max(results.values())
        # Under HSDP the cross-node exchange is one all-reduce of the replica's
        # parameter shard per step.
        grad_bytes = MODEL_PARAMS * 2 / max(local_world, 1)
        print(f"\npeak all-reduce bandwidth: {peak:.2f} GB/s")
        if nodes > 1:
            print(
                f"HSDP cross-node volume/step: {grad_bytes / 1e9:.2f} GB "
                f"-> ~{grad_bytes / 1e9 / peak * 1000:.0f} ms of exchange"
            )
            if peak < 2.0:
                print(
                    "\nWARNING: under 2 GB/s across nodes means NCCL is very likely on TCP,\n"
                    "not RDMA. Re-run with NCCL_DEBUG=INFO and check that it selects an\n"
                    "IB/RoCE transport, and that NCCL_IB_GID_INDEX matches the RoCE v2\n"
                    "entry in /sys/class/infiniband/<hca>/ports/1/gid_attrs/types/."
                )
        else:
            print("single node: this is NVLink bandwidth; rerun across nodes to test RoCE")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
