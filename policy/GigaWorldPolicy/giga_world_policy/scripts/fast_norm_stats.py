"""Fast norm stats computation — reads parquet files directly, no video decoding.

Usage:
    python scripts/fast_norm_stats.py \
        --data_root /path/to/lerobot_dataset \
        --output /path/to/lerobot_dataset/norm_stats_delta.json \
        --action_dim 12 --state_dim 16
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


class RunningStats:
    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None

    def update(self, batch: np.ndarray):
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        n, d = batch.shape
        batch_mean = np.mean(batch, axis=0)
        batch_ms = np.mean(batch ** 2, axis=0)
        batch_min = np.min(batch, axis=0)
        batch_max = np.max(batch, axis=0)

        if self._count == 0:
            self._mean = batch_mean
            self._mean_of_squares = batch_ms
            self._min = batch_min
            self._max = batch_max
        else:
            self._min = np.minimum(self._min, batch_min)
            self._max = np.maximum(self._max, batch_max)
            self._mean += (batch_mean - self._mean) * (n / (self._count + n))
            self._mean_of_squares += (batch_ms - self._mean_of_squares) * (n / (self._count + n))
        self._count += n

    def get_stats(self):
        var = self._mean_of_squares - self._mean ** 2
        std = np.sqrt(np.maximum(0, var))
        return {
            "mean": self._mean.tolist(),
            "std": std.tolist(),
            "min": self._min.tolist(),
            "max": self._max.tolist(),
        }


def pad_or_truncate(arr: np.ndarray, target_dim: int) -> np.ndarray:
    d = arr.shape[-1]
    if d == target_dim:
        return arr
    if d > target_dim:
        return arr[..., :target_dim]
    pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, target_dim - d)]
    return np.pad(arr, pad_width, constant_values=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=None, help="Root dir (will glob for parquets)")
    parser.add_argument("--parquet_list", default=None, help="Text file with one parquet path per line (faster)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--action_dim", type=int, default=12)
    parser.add_argument("--state_dim", type=int, default=16)
    args = parser.parse_args()

    if args.parquet_list:
        with open(args.parquet_list) as f:
            parquet_files = [l.strip() for l in f if l.strip()]
    elif args.data_root:
        parquet_files = sorted(glob.glob(
            os.path.join(args.data_root, "**", "data", "**", "*.parquet"),
            recursive=True,
        ))
    else:
        raise ValueError("Provide --parquet_list or --data_root")

    print(f"Found {len(parquet_files)} parquet files")

    action_stats = RunningStats()
    state_stats = RunningStats()

    for pq in tqdm(parquet_files, desc="Processing"):
        df = pd.read_parquet(pq, columns=["action", "observation.state"])
        actions = np.stack(df["action"].values).astype(np.float64)
        states = np.stack(df["observation.state"].values).astype(np.float64)

        actions = pad_or_truncate(actions, args.action_dim)
        states = pad_or_truncate(states, args.state_dim)

        action_stats.update(actions)
        state_stats.update(states)

    result = {
        "norm_stats": {
            "action": action_stats.get_stats(),
            "observation.state": state_stats.get_stats(),
        }
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Written to {args.output}")
    print(f"action: mean={result['norm_stats']['action']['mean'][:5]}...")
    print(f"state:  mean={result['norm_stats']['observation.state']['mean'][:5]}...")


if __name__ == "__main__":
    main()
