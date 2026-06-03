#!/usr/bin/env python3
"""Generate a Hydra data config from XR-0 action_stats.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--json_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    stats = json.loads(args.stats_path.read_text(encoding="utf-8"))
    mean = stats["mean"]
    std = stats["std"]
    action_length = stats.get("action_length", 30)

    json_dir = str(args.json_dir.resolve())
    lines = [
        "# @package _global_",
        "",
        "data:",
        "  type: BaseDataModule",
        "  params:",
        "    type: json",
        "    max_steps: ${trainer.max_steps}",
        "    train_datasets:",
        f"      batch_size: {args.batch_size}",
        f"      action_length: {action_length}",
        "      train_path:",
        f"      - {json_dir}",
        "      mean:",
    ]
    for row in mean:
        lines.append("      - " + json.dumps(row))
    lines.append("      std:")
    for row in std:
        lines.append("      - " + json.dumps(row))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote data config to {args.output_path}")


if __name__ == "__main__":
    main()
