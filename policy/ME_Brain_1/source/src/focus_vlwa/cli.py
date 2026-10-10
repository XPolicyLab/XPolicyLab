"""Unified Focus-VLWA command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    from focus_vlwa.scripts.post_train import build_parser

    parser = argparse.ArgumentParser(prog="focus-vlwa")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-checkpoint", help="validate checkpoint names and shapes")
    check.add_argument("checkpoint", type=Path)
    commands.add_parser("install-transformers-patch", help="install the pinned transformers compatibility patch")
    commands.add_parser("post-train", help="run Focus-VLWA post-training", parents=[build_parser(add_help=False)])
    args = parser.parse_args()

    if args.command == "check-checkpoint":
        from focus_vlwa.scripts.check_checkpoint import validate_checkpoint

        missing, unexpected, mismatches = validate_checkpoint(args.checkpoint)
        if missing or unexpected or mismatches:
            raise SystemExit(
                f"Checkpoint validation failed: missing={missing[:20]}, unexpected={unexpected[:20]}, "
                f"shape_mismatches={mismatches[:20]}"
            )
        print(f"Checkpoint is compatible: {args.checkpoint}")
    elif args.command == "install-transformers-patch":
        from focus_vlwa.scripts.install_transformers_patch import install_transformers_patch

        install_transformers_patch()
    elif args.command == "post-train":
        from focus_vlwa.scripts.post_train import run_from_args

        run_from_args(args)


if __name__ == "__main__":
    main()
