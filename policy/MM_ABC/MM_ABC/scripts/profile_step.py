#!/usr/bin/env python
"""Profile a training step with a selected model and dataset mixture."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


class Timer:
    def __init__(self) -> None:
        self.marks: dict[str, list[float]] = {}

    def __call__(self, name: str):
        return _Scope(self, name)

    def add(self, name: str, dt: float) -> None:
        self.marks.setdefault(name, []).append(dt)

    def report(self, total_key: str | None = None) -> None:
        import statistics as st

        base = st.mean(self.marks[total_key]) if total_key in self.marks else None
        print(f"\n{'component':28s} {'mean ms':>10s} {'share':>8s} {'calls':>6s}")
        for name, vals in self.marks.items():
            m = st.mean(vals) * 1000
            share = f"{m / (base * 1000) * 100:7.1f}%" if base else "      -"
            print(f"{name:28s} {m:10.1f} {share:>8s} {len(vals):6d}")


class _Scope:
    def __init__(self, timer: Timer, name: str) -> None:
        self.timer, self.name = timer, name

    def __enter__(self):
        torch.cuda.synchronize()
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        self.timer.add(self.name, time.time() - self.t0)
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="configs/model/mmabc_4b_mobile.yaml")
    ap.add_argument("--mixture", default="configs/mixture/mobile_all.yaml")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()

    import warnings

    warnings.filterwarnings("ignore")
    from torch.utils.data import DataLoader

    from mmabc.data import MixtureDataset, collate, to_device
    from mmabc.models.heads import future_alignment_loss, masked_action_loss
    from mmabc.models.mmabc import MMABCPolicy, load_model_config

    cfg = load_model_config(args.model, repo_root=str(REPO))
    dev = torch.device("cuda")
    model = MMABCPolicy(cfg, repo_root=str(REPO)).to(dev)
    model.train()

    ds = MixtureDataset(
        REPO / args.mixture,
        chunk_size=int(cfg.flow.chunk_size),
        image_size=int(cfg.obs.image_size),
        future_offsets=model.future_offsets,
        prompt_dropout=0.15,
        norm_stats_dir=REPO / "configs/norm_stats",
        seed=0,
    )
    dl = DataLoader(ds, batch_size=args.micro_batch, num_workers=4, collate_fn=collate)
    it = iter(dl)
    draws = int(cfg.flow.repeated_noise_draws)
    T = Timer()

    for step in range(args.steps):
        batch = to_device(next(it), dev)
        measure = step > 0  # first step warms up allocator, teacher load, cudnn

        def rec(name):
            return T(name) if measure else _Null()

        with rec("00 step total"):
            with rec("01 backbone (1x)"):
                memory, ctx_valid = model.encode_context(batch)
            with rec("02 teacher (2 horizons)"):
                with torch.no_grad():
                    fut_target = model._extract_future_targets(batch["images"], dev)
            with rec("02b context cache (1x)"):
                context_cache = model.expert.build_context_cache(memory)

            target, mask = batch["target"], batch["target_mask"]
            head_active = {
                "manip": (mask[..., 0:58].sum((1, 2)) > 0).float(),
                "aux": batch["aux_active"],
            }
            loss = torch.zeros((), device=dev)
            for d in range(draws):
                with rec(f"03 expert fwd (x{draws})"):
                    noise = torch.randn_like(target)
                    t = model.flow.sample_time(target.shape[0], dev)
                    out = model.expert(
                        noisy_actions=model.flow.interpolate(target, noise, t),
                        flow_time=t,
                        state=batch["state"],
                        state_mask=batch["state_mask"],
                        memory=memory,
                        context_valid=ctx_valid,
                        aux_active=batch["aux_active"],
                        want_future=(d == 0),
                        context_cache=context_cache,
                    )
                with rec(f"04 loss (x{draws})"):
                    al, _, _ = masked_action_loss(
                        out.predictions, target, mask, model.head_slices,
                        weight=model.flow.loss_weight(t), head_active=head_active,
                    )
                    loss = loss + al / draws
                    if out.future_near is not None:
                        valid = batch.get("future_valid")
                        near_v = None if valid is None else valid[:, 0]
                        far_v = None if valid is None else valid[:, 1]
                        loss = loss + model.future_weight * 0.5 * (
                            future_alignment_loss(out.future_near, fut_target[0], near_v)
                            + future_alignment_loss(out.future_far, fut_target[1], far_v)
                        )
            with rec("05 backward"):
                loss.backward()
            with rec("06 zero_grad"):
                model.zero_grad(set_to_none=True)

    ctx_len = memory[-1].shape[1]
    print(f"\nmicro_batch={args.micro_batch}  draws={draws}  chunk={cfg.flow.chunk_size}")
    print(f"context tokens={ctx_len}  future tokens={cfg.future.num_tokens}")
    print(f"peak mem={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    T.report("00 step total")
    return 0


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
