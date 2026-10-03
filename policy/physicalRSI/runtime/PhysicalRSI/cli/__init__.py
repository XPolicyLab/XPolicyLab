"""Minimal physicalRSI command line entry point."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="physicalrsi", description="physicalRSI · composable embodied intelligence"
    )
    parser.add_argument("--workspace", type=Path, default=Path(".physicalrsi"))
    parser.add_argument("--plain", action="store_true", help="Plain terminal text")
    parser.add_argument(
        "--command",
        action="append",
        help="Run a slash command; repeat for multiple commands",
    )
    args = parser.parse_args(argv)
    from PhysicalRSI.application import Application

    from .terminal import run_console

    return run_console(Application(args.workspace), args.command, plain=args.plain)
