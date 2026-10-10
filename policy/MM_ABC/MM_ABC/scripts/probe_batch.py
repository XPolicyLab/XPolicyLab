#!/usr/bin/env python
"""Probe training memory use at a selected micro-batch size."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--micro-batch", type=int, required=True)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--out", help="append one JSON line of results here")
    args = ap.parse_args()

    from torch.utils.data import DataLoader

    from mmabc.data import MixtureDataset, collate, to_device
    from mmabc.models.mmabc import MMABCPolicy, load_model_config
    from mmabc.train.fsdp import build_mesh, init_distributed, shard_model

    cfg = OmegaConf.load(str(REPO / args.config))
    info = init_distributed()
    model_cfg = load_model_config(str(cfg.model), repo_root=str(REPO))

    model = MMABCPolicy(model_cfg, repo_root=str(REPO)).to(info.device)
    mesh = build_mesh(info, strategy=str(cfg.get("shard_strategy", "hsdp")))
    model = shard_model(model, mesh)
    opt = torch.optim.AdamW(model.param_groups(cfg.optim))

    ds = MixtureDataset(
        REPO / str(cfg.data.mixture),
        chunk_size=int(model_cfg.flow.chunk_size),
        image_size=int(model_cfg.obs.image_size),
        future_offsets=model.future_offsets,
        prompt_dropout=float(cfg.data.prompt_dropout),
        norm_stats_dir=REPO / str(cfg.data.norm_stats_dir),
        seed=info.rank,
        repo_root=str(REPO),
    )
    dl = DataLoader(
        ds,
        batch_size=args.micro_batch,
        num_workers=int(cfg.data.num_workers),
        collate_fn=collate,
        pin_memory=True,
    )
    it = iter(dl)

    model.train()
    torch.cuda.reset_peak_memory_stats()
    durations: list[float] = []
    status = "ok"
    try:
        for i in range(args.steps):
            batch = to_device(next(it), info.device)
            t0 = time.time()
            out = model(batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if i >= 2:  # skip teacher load and allocator warmup
                durations.append(time.time() - t0)
    except torch.cuda.OutOfMemoryError:
        status = "OOM"

    peak = torch.cuda.max_memory_allocated() / 2**30
    sec = sum(durations) / len(durations) if durations else float("nan")
    result = {
        "micro_batch": args.micro_batch,
        "status": status,
        "peak_gib": round(peak, 2),
        "s_per_step": round(sec, 3),
        "samples_per_s_per_gpu": round(args.micro_batch / sec, 2) if durations else 0.0,
        "world_size": info.world_size,
        "per_node": args.micro_batch * info.local_world_size,
    }
    if info.is_master:
        print(json.dumps(result), flush=True)
        if args.out:
            with open(args.out, "a") as fh:
                fh.write(json.dumps(result) + "\n")
    # An OOM is a valid probe result, not a launch failure; exiting non-zero
    # would make torchrun print a scary ChildFailedError for a normal outcome.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
