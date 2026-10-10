#!/usr/bin/env python
"""Compare seeded GPU workloads across ranks and report available ECC counters."""

from __future__ import annotations

import argparse
import os
import subprocess

import torch
import torch.distributed as dist
import torch.nn.functional as F


def checksum(device: torch.device, iters: int) -> torch.Tensor:
    """A seeded mix of the kernels training actually leans on."""
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    total = torch.zeros((), dtype=torch.float64, device=device)

    for _ in range(iters):
        a = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
        b = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
        total += (a @ b).float().double().sum()

        x = torch.randn(4, 8, 256, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
        y = F.scaled_dot_product_attention(x, x, x)
        y.float().pow(2).mean().backward()
        total += x.grad.float().double().sum()

        z = torch.randn(512, 1024, device=device, dtype=torch.float32, requires_grad=True)
        n = F.layer_norm(z, (1024,))
        n.pow(2).mean().backward()
        total += z.grad.double().sum()

    return total


def ecc_report(local_rank: int) -> str:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={local_rank}",
                "--query-gpu=uuid,ecc.errors.uncorrected.volatile.total,"
                "retired_pages.pending,temperature.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return out.stdout.strip() or out.stderr.strip()
    except Exception as exc:  # a diagnostic must not fail the sweep
        return f"nvidia-smi unavailable: {exc!r}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", local_rank)

    value = checksum(device, args.iters)
    finite = torch.isfinite(value)
    info = f"r{rank} {torch.cuda.get_device_name(local_rank)} | {ecc_report(local_rank)}"

    gathered = [torch.zeros_like(value) for _ in range(world)]
    dist.all_gather(gathered, value)
    lines = [None] * world
    dist.all_gather_object(lines, info)

    if rank == 0:
        vals = [float(g) for g in gathered]
        # The majority answer is the reference: a single faulty card cannot
        # outvote 63 healthy ones.
        counts: dict[float, int] = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        reference = max(counts, key=lambda k: counts[k])
        print(f"world={world}  reference checksum={reference!r} "
              f"agreeing ranks={counts[reference]}/{world}")
        bad = [(i, v) for i, v in enumerate(vals) if v != reference]
        for i, v in bad:
            print(f"  MISMATCH r{i}: {v!r}   {lines[i]}")
        if not bad:
            print("RESULT: PASS, every GPU computes identical results")
        else:
            print(f"RESULT: FAIL, {len(bad)} GPU(s) disagree; see above")
        # Surface ECC even on a pass, since a degrading card often still agrees.
        for line in lines:
            if line and ("0" != line.split(",")[-3].strip() if line.count(",") >= 3 else False):
                print(f"  ECC note: {line}")

    ok = int(finite)
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
