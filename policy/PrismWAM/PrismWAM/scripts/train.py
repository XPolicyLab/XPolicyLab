#!/usr/bin/env python3
"""Train one PrismWAM stage from a recipe. Usually started by scripts/train.sh.

python scripts/train.py configs/libero/stage1_depth.yaml --dry-run
python scripts/train.py configs/libero/stage1_depth.yaml --check-assets
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prismwam.config import check_recipe, gradient_accumulation_steps, load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path, help="recipe YAML under configs/")
    parser.add_argument("overrides", nargs="*", help="key=value overrides")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="check the recipe and print a summary"
    )
    mode.add_argument(
        "--check-assets",
        action="store_true",
        help="like --dry-run, and list referenced files that do not exist",
    )
    mode.add_argument(
        "--print-launch-options", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_intermixed_args()

    cfg = load_config(args.config, args.overrides)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    cfg.gradient_accumulation_steps = gradient_accumulation_steps(cfg, world_size)
    if not math.isfinite(float(cfg.max_grad_norm)) or float(cfg.max_grad_norm) < 0:
        raise ValueError("max_grad_norm must be finite and non-negative.")

    if args.print_launch_options:
        # Read by scripts/train.sh: accelerate needs these before the processes start.
        print(cfg.gradient_accumulation_steps)
        print(float(cfg.max_grad_norm))
        print(cfg.distributed_config)
        return

    report = check_recipe(cfg, world_size=world_size, assets=args.check_assets)
    if args.dry_run or args.check_assets:
        print(json.dumps(report, indent=2))
        if report.get("missing_assets"):
            raise SystemExit(1)
        return

    from prismwam.runtime import run_training
    from prismwam.training import ReleaseTrainer

    run_training(cfg, source_config_path=args.config, trainer_cls=ReleaseTrainer)


if __name__ == "__main__":
    main()
