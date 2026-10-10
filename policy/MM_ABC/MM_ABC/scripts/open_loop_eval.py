#!/usr/bin/env python
"""Evaluate a checkpoint against a converted mixture without a simulator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="milestone or resume directory")
    ap.add_argument("--mixture", default=None, help="defaults to the run's training mixture")
    ap.add_argument("--samples", type=int, default=128, help="per profile")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-inference-steps", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from omegaconf import OmegaConf

    from mmabc.canonical.layout import load_layout
    from mmabc.canonical.normalize import NormStats
    from mmabc.data.dataset import ProfileConfig
    from mmabc.eval.open_loop import evaluate_profile, report
    from mmabc.eval.policy import MMABCInferencePolicy, find_run_meta

    meta = find_run_meta(args.checkpoint)
    train_cfg = OmegaConf.load(meta / "train_config.yaml")
    policy = MMABCInferencePolicy.from_run(
        args.checkpoint, repo_root=str(REPO), num_inference_steps=args.num_inference_steps
    )
    model, cfg = policy.model, OmegaConf.load(meta / "model_config.yaml")
    layout = load_layout(str(meta / "canonical.yaml"))
    mixture = OmegaConf.load(str(REPO / (args.mixture or train_cfg.data.mixture)))

    results = []
    for entry in mixture.datasets:
        pc = ProfileConfig.load(REPO / str(entry.config))
        stats = NormStats.load(meta / f"norm_stats_{pc.embodiment_tag}.json")
        offsets = tuple(int(o) for o in (cfg.future.get("offsets") or (cfg.flow.near_steps, cfg.flow.chunk_size)))
        results.append(
            evaluate_profile(
                model, pc, layout,
                chunk_size=int(cfg.flow.chunk_size),
                image_size=int(cfg.obs.image_size),
                future_offsets=offsets,
                norm_stats=stats,
                num_samples=args.samples,
                batch_size=args.batch_size,
                device=torch.device("cuda"),
                prompt_header=bool(cfg.get("prompt", {}).get("header", False)),
            )
        )
        print(f"  done {pc.name}: overall_norm_mae={results[-1]['overall_normalized_mae']:.4f}", flush=True)

    out = args.out or str(Path(args.checkpoint) / "open_loop.json")
    print()
    print(report(results, out=out))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
