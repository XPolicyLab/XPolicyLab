#!/usr/bin/env python3
"""Package a trained Stage-2 checkpoint as a model folder (config, weights, VAEs, stats).

    python scripts/export_model.py \
        --config outputs/robotwin/stage2_joint/config.yaml \
        --checkpoint outputs/robotwin/stage2_joint/checkpoints/weights/step_015000.pt \
        --output exports/robotwin --sanitize-checkpoint
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prismwam.deployment.bundle import export_bundle


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", required=True, help="resolved Stage-2 config of the run"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Stage-2 weights (step_*.pt)"
    )
    parser.add_argument("--output", required=True, help="new model folder")
    parser.add_argument(
        "--evaluation",
        help="inference settings (YAML/JSON); default: the config's `deployment`",
    )
    parser.add_argument(
        "--statistics", help="action statistics; default: from the config"
    )
    parser.add_argument("--task-name", help="one task of the mixed real-robot recipe")
    parser.add_argument("--text-cache", help="instruction features to include")
    parser.add_argument(
        "--asset",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help="relocated asset, e.g. model.rgb_vae_checkpoint_path=/path/vae.pt",
    )
    parser.add_argument(
        "--hardlink", action="store_true", help="hard-link files instead of copying"
    )
    parser.add_argument(
        "--sanitize-checkpoint",
        action="store_true",
        help="rewrite the weights without local paths in their metadata",
    )
    args = parser.parse_args()
    overrides = {}
    for value in args.asset:
        if "=" not in value:
            parser.error("--asset requires KEY=PATH")
        key, path = value.split("=", 1)
        overrides[key] = path
    directory = export_bundle(
        args.config,
        args.checkpoint,
        args.output,
        evaluation=args.evaluation,
        statistics=args.statistics,
        text_cache=args.text_cache,
        task_name=args.task_name,
        hardlink=args.hardlink,
        sanitize_checkpoint=args.sanitize_checkpoint,
        overrides=overrides,
    )
    for path in sorted(directory.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
