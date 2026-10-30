#!/usr/bin/env python3
"""Compute full-frame absolute RoboDojo state/action z-score statistics (ddof=0)."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from worldscape_policy.robodojo_joint import JOINT_DIM, JOINT_LAYOUT
from tools.data.robodojo_common import discover_episodes, map_jobs, output_path, read_joints, write_statistics


def episode_moments(path: Path) -> dict:
    result = {}
    for group, value in zip(("state", "action"), read_joints(path), strict=True):
        # Match the loader's float32 input; accumulate population moments in float64.
        array = value.astype(np.float64)
        mean = array.mean(axis=0)
        result[group] = (len(array), mean, ((array - mean) ** 2).sum(axis=0))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.data_root = args.data_root.expanduser().resolve()
    output = output_path(args.output, overwrite=args.overwrite)
    paths = discover_episodes(args.data_root)
    totals = {group: (0, np.zeros(JOINT_DIM), np.zeros(JOINT_DIM)) for group in ("state", "action")}
    for index, result in enumerate(map_jobs(episode_moments, paths, args.workers), 1):
        for group, (n, mean, m2) in result.items():
            count, old_mean, old_m2 = totals[group]
            delta = mean - old_mean
            total = count + n
            totals[group] = (
                total, old_mean + delta * (n / total),
                old_m2 + m2 + delta ** 2 * (count * n / total),
            )
        if index % 500 == 0 or index == len(paths):
            print(f"Read {index}/{len(paths)}", flush=True)
    artifact = {
        "schema_version": "1", "layout": list(JOINT_LAYOUT),
        "normalization": "zscore", "clip_range": [-5.0, 5.0],
        "relative_joint_action": False,
        "provenance": {
            "data_root": str(args.data_root.resolve()), "episodes": len(paths),
            "dataset_format": "official_robodojo_hdf5",
            "weighting": "all raw frames; equal weight per frame; no sampling",
            "ddof": 0, "input_dtype": "float32", "accumulator_dtype": "float64",
            "excluded_episodes": [], "excluded_frames": 0,
            "source_paths_sha256": hashlib.sha256(
                "\n".join(str(p) for p in paths).encode()
            ).hexdigest(),
        },
    }
    for group, (count, mean, m2) in totals.items():
        std = np.sqrt(m2 / count)
        if not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError(f"Nonpositive/nonfinite {group} standard deviation")
        artifact[group] = {"mean": mean.tolist(), "std": std.tolist(), "samples": count}
    write_statistics(output, artifact, overwrite=args.overwrite)
    print(f"Validated {output}", flush=True)


if __name__ == "__main__":
    main()
