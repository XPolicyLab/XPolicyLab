#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(description="Compute 30-D action quantile stats.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def iter_actions(dataset_root: Path):
    for parquet_path in sorted((dataset_root / "data").glob("chunk-*/*.parquet")):
        table = pq.read_table(parquet_path, columns=["action"])
        for value in table.column("action").to_pylist():
            arr = np.asarray(value, dtype=np.float32)
            if arr.shape != (30,):
                raise ValueError(f"Unexpected action shape {arr.shape} in {parquet_path}")
            yield arr


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    actions = list(iter_actions(dataset_root))
    if not actions:
        raise RuntimeError("No actions found")

    actions = np.stack(actions, axis=0)

    stats = {
        "raw_action_dim": 30,
        "target_action_dim": 30,
        "used_action_channel_ids": list(range(30)),
        "inverse_used_action_channel_ids": list(range(30)),
        "q01": np.quantile(actions, 0.01, axis=0).astype(float).tolist(),
        "q99": np.quantile(actions, 0.99, axis=0).astype(float).tolist(),
        "num_samples": int(actions.shape[0]),
    }

    output_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved: {output_path}")
    print(f"Samples: {stats['num_samples']}")


if __name__ == "__main__":
    main()
