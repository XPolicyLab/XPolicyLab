#!/usr/bin/env python
"""Compute model-space normalisation statistics for a dataset mixture."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mmabc.canonical import NormStats, RunningStats, load_layout  # noqa: E402
from mmabc.canonical.transforms import EmbodimentSpec, TargetBuilder  # noqa: E402
from mmabc.data.dataset import ProfileConfig  # noqa: E402
from mmabc.data.lerobot_v3 import LeRobotV3Profile  # noqa: E402


def _resolve(rel: str) -> Path:
    p = Path(str(rel))
    return p if p.is_absolute() else REPO / p


def scan_profile(args: tuple) -> tuple[str, dict, dict, int]:
    """Accumulate statistics for one profile; returns partial sums to merge.

    Runs in a subprocess: parquet handles and episode indices are per-process,
    and the scan is embarrassingly parallel across profiles.
    """
    cfg_path, canonical_path, chunk_size, n_samples, seed = args
    cfg = ProfileConfig.load(cfg_path)
    layout = load_layout(canonical_path)
    profile = LeRobotV3Profile(cfg.path, cfg.views)
    embodiment = json.loads((Path(cfg.path) / "meta" / "embodiment.json").read_text())
    spec = EmbodimentSpec.from_meta(embodiment, fps=cfg.fps)
    builder = TargetBuilder(layout, spec, reference_frame=cfg.reference_frame)

    action_stats = RunningStats(layout.total_dim)
    state_stats = RunningStats(layout.state_dim)
    lengths = np.asarray(profile.length)
    cumsum = np.cumsum(lengths)
    total = int(cumsum[-1]) if len(cumsum) else 0
    if total == 0:
        return cfg.embodiment_tag, action_stats.result(), state_stats.result(), 0

    rng = random.Random(seed)
    state_valid = spec.state_valid[None, :]
    done = 0
    types = [a for a in cfg.available_action_types if a in builder.available_action_types]

    for i in range(n_samples):
        g = rng.randrange(total)
        ep = int(np.searchsorted(cumsum, g, side="right"))
        frame = g - (int(cumsum[ep - 1]) if ep > 0 else 0)
        # Statistics must cover every convention the profile can be trained
        # under, since each produces a different target distribution.
        action_type = types[i % len(types)]
        try:
            state, actions, valid = profile.sample_window(ep, frame, chunk_size)
            target, mask = builder.build(
                state, actions, action_type=action_type, valid_steps=valid
            )
        except Exception:
            continue
        action_stats.update(target, mask)
        state_stats.update(state[None, :], state_valid)
        done += 1

    return cfg.embodiment_tag, action_stats.result(), state_stats.result(), done


def _merge(dst: RunningStats, part: dict) -> None:
    """Fold a subprocess's partial result into an accumulator."""
    count = np.asarray(part["count"], dtype=np.int64)
    mean = np.asarray(part["mean"], dtype=np.float64)
    std = np.asarray(part["std"], dtype=np.float64)
    m2 = np.where(count > 1, std**2 * (count - 1), 0.0)

    total = dst.count + count
    delta = np.where(count > 0, mean - dst.mean, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        dst.m2 = dst.m2 + m2 + delta**2 * dst.count * count / np.maximum(total, 1)
        dst.mean = np.where(total > 0, (dst.mean * dst.count + mean * count) / np.maximum(total, 1), 0.0)
    dst.count = total
    dst.min = np.minimum(dst.min, np.asarray(part["min"], dtype=np.float64))
    dst.max = np.maximum(dst.max, np.asarray(part["max"], dtype=np.float64))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixture", default="configs/mixture/mobile_all.yaml")
    ap.add_argument("--out", default="configs/norm_stats")
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument("--samples-per-profile", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mixture = OmegaConf.load(str(_resolve(args.mixture)))
    canonical = str(_resolve(mixture.canonical))
    layout = load_layout(canonical)
    out_dir = _resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for i, entry in enumerate(mixture.datasets):
        jobs.append(
            (
                str(_resolve(entry.config)),
                canonical,
                args.chunk_size,
                args.samples_per_profile,
                args.seed + i,
            )
        )

    action_acc: dict[str, RunningStats] = defaultdict(lambda: RunningStats(layout.total_dim))
    state_acc: dict[str, RunningStats] = defaultdict(lambda: RunningStats(layout.state_dim))
    counts: dict[str, int] = defaultdict(int)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for tag, a_part, s_part, n in pool.map(scan_profile, jobs):
            _merge(action_acc[tag], a_part)
            _merge(state_acc[tag], s_part)
            counts[tag] += n
            print(f"  scanned {n:6d} samples  tag={tag}", flush=True)

    for tag in sorted(action_acc):
        a = action_acc[tag].result()
        s = state_acc[tag].result()
        NormStats(embodiment_tag=tag, action=a, state=s).save(out_dir / f"{tag}.json")
        std = np.asarray(a["std"])
        active = np.asarray(a["count"]) > 0
        print(
            f"{tag:44s} n={counts[tag]:6d} active_dims={int(active.sum()):3d} "
            f"std[min={std[active].min():.4f} med={np.median(std[active]):.4f} "
            f"max={std[active].max():.4f}]"
        )
    print(f"\nwrote {len(action_acc)} stat files to {out_dir} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
