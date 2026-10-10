"""Focus-VLWA post-training command."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from focus_vlwa.configs.training import PostTrainingConfig, stage_defaults


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    """Expose identical training arguments for both public entry points."""
    parser = argparse.ArgumentParser(description=__doc__, add_help=add_help)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--init-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--norm-stats-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--stage", choices=("joint", "frozen"), default="joint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--history-mode", choices=("head_history",))
    parser.add_argument("--max-token-len", type=int)
    parser.add_argument("--world-model-loss-weight", type=float)
    parser.add_argument("--world-model-event-loss-weight", type=float)
    parser.add_argument("--world-model-state-loss-weight", type=float)
    parser.add_argument("--freeze-world-model-expert", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--decay-steps", type=int)
    parser.add_argument("--peak-learning-rate", type=float)
    parser.add_argument("--final-learning-rate", type=float)
    parser.add_argument("--init-world-model-from-action", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--parameter-precision", choices=("float32",))
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-batch-change-step", type=int)
    parser.add_argument("--resume-old-batch-size", type=int)
    return parser


def run_from_args(namespace: argparse.Namespace) -> None:
    """Resolve stage defaults before loading the training dependencies."""
    from focus_vlwa.post_training.trainer import run_post_training

    args = vars(namespace).copy()
    args.pop("command", None)
    defaults = stage_defaults(args.pop("stage"))
    defaults.update({key: value for key, value in args.items() if value is not None})
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_post_training(PostTrainingConfig(**defaults))


def main() -> None:
    run_from_args(build_parser().parse_args())


if __name__ == "__main__":
    main()
