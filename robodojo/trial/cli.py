"""Shared CLI flags for trial environment configuration."""

from __future__ import annotations

import argparse


def add_trial_env_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--eval-env",
        choices=("debug", "sim"),
        help="Environment backend override (defaults to dispatch.eval_env or debug)",
    )
    parser.add_argument(
        "--root-dir",
        help="Repository root for sim env discovery (defaults to dispatch.root_dir)",
    )
    parser.add_argument(
        "--sim-env-factory",
        help="Import path for sim env factory, e.g. my_sim:create_trial_env",
    )
    parser.add_argument(
        "--episode-step-limit",
        type=int,
        default=5,
        help="Debug env episode length when eval_env=debug",
    )
