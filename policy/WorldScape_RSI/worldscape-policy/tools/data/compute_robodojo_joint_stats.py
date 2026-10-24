#!/usr/bin/env python3
"""Compute absolute-joint q01/q99 statistics for official RoboDojo HDF5 data."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools.data.robodojo_common import (
    discover_episodes, map_jobs, output_path, read_joints, write_statistics,
)
from worldscape_policy.robodojo_joint import JOINT_LAYOUT


def _sample_rows(value: np.ndarray, count: int | None, rng: np.random.Generator) -> np.ndarray:
    if count is None or len(value) <= count:
        return value
    return value[rng.choice(len(value), size=count, replace=False)]


def _process_file(item) -> tuple[np.ndarray, np.ndarray]:
    path, per_file, seed = item
    state, action = read_joints(path)
    rng = np.random.default_rng(seed)
    return _sample_rows(state, per_file, rng), _sample_rows(action, per_file, rng)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--max-samples", type=int, default=2_000_000,
                        help="Approximate state/action sample budget, divided equally per episode; 0 uses all frames.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.max_samples < 0:
        raise ValueError("max-samples must be nonnegative")
    root = args.data_root.expanduser().resolve()
    output = output_path(args.output, overwrite=args.overwrite)
    paths = discover_episodes(root)
    per_file = max(1, args.max_samples // len(paths)) if args.max_samples else None
    work = [(path, per_file, args.seed + i) for i, path in enumerate(paths)]
    states, actions = [], []
    for index, (state, action) in enumerate(map_jobs(_process_file, work, args.workers), 1):
        states.append(state)
        actions.append(action)
        if index % 250 == 0 or index == len(paths):
            print(f"Read {index}/{len(paths)} episodes", flush=True)

    def summary(values):
        array = np.concatenate(values, axis=0)
        return {
            "q01": np.quantile(array, 0.01, axis=0).tolist(),
            "q99": np.quantile(array, 0.99, axis=0).tolist(),
            "min": array.min(axis=0).tolist(), "max": array.max(axis=0).tolist(),
            "mean": array.mean(axis=0, dtype=np.float64).tolist(),
            "std": array.std(axis=0, dtype=np.float64).tolist(),
            "samples": len(array),
        }

    artifact = {
        "schema_version": "1", "layout": list(JOINT_LAYOUT),
        "normalization": "q99", "relative_joint_action": False,
        "source_root": str(root), "source_files": len(paths), "seed": args.seed,
        "provenance": {"dataset_format": "official_robodojo_hdf5", "input_dtype": "float32"},
        "sampling": {"max_samples": args.max_samples, "per_file": per_file,
                     "weighting": "all frames" if per_file is None else "equal per-episode sample budget"},
        "state": summary(states), "action": summary(actions),
    }
    write_statistics(output, artifact, overwrite=args.overwrite)
    print(f"Validated {output}", flush=True)


if __name__ == "__main__":
    main()
