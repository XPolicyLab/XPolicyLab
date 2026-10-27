#!/usr/bin/env python3
"""Prepare indexes, normalization statistics and text caches for a recipe.

Outputs are always new files; existing indexes and statistics are never overwritten.
Commands that take --config also accept key=value overrides, e.g. paths.data_root=/data.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prismwam import data_tools


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser(
        "robodojo-index", help="scan RoboDojo HDF5 files into an episode index"
    )
    index.add_argument("--root", required=True)
    index.add_argument("--output", required=True)
    relocate = commands.add_parser(
        "relocate-index", help="point an existing index at a new data root"
    )
    relocate.add_argument("--index", required=True)
    relocate.add_argument("--root", required=True)
    relocate.add_argument("--output", required=True)
    split = commands.add_parser(
        "robodojo-holdout", help="90/10 per-task split used for RoboDojo statistics"
    )
    split.add_argument("--index", required=True)
    split.add_argument("--output-dir", required=True)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--expected-tasks", type=int, default=35)
    stats = commands.add_parser(
        "stats", help="fit action/state statistics of a recipe's training data"
    )
    stats.add_argument("--config", required=True)
    stats.add_argument(
        "--node",
        default="data.train",
        help="e.g. data.train.sources.0 for one robot task",
    )
    stats.add_argument("--output", required=True)
    stats.add_argument("overrides", nargs="*", help="key=value config overrides")
    joint14 = commands.add_parser(
        "joint14-stats", help="joint-action statistics from an explicit episode index"
    )
    joint14.add_argument("--index", required=True)
    joint14.add_argument("--output", required=True)
    text = commands.add_parser(
        "text", help="encode every instruction of a recipe into its text cache"
    )
    text.add_argument("--config", required=True)
    text.add_argument("--checkpoint", required=True, help="text encoder weights")
    text.add_argument("--tokenizer", required=True, help="tokenizer directory")
    text.add_argument("--device", default="cuda")
    text.add_argument("--batch-size", type=int, default=8)
    text.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()
    if args.command == "robodojo-index":
        result = data_tools.build_robodojo_index(args.root, args.output)
    elif args.command == "relocate-index":
        result = data_tools.relocate_index(args.index, args.root, args.output)
    elif args.command == "robodojo-holdout":
        result = data_tools.build_holdout(
            args.index, args.output_dir, args.seed, args.expected_tasks
        )
    elif args.command == "stats":
        result = data_tools.dataset_statistics(
            args.config, args.output, args.node, args.overrides
        )
    elif args.command == "joint14-stats":
        result = data_tools.joint14_statistics(args.index, args.output)
    else:
        result = data_tools.text_embeddings(
            args.config,
            checkpoint=args.checkpoint,
            tokenizer=args.tokenizer,
            device=args.device,
            batch_size=args.batch_size,
            overrides=args.overrides,
        )
    if isinstance(result, dict) and "episodes" in result:
        result = {k: v for k, v in result.items() if k != "episodes"}
    print(json.dumps(result, indent=2, default=lambda value: value.tolist()))


if __name__ == "__main__":
    main()
