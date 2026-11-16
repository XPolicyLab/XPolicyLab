"""Launch the bundled source without installing generic packages globally."""

import argparse
import os
from pathlib import Path
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        parser = argparse.ArgumentParser(
            prog="playground.sh",
            description="Start Playground, freeze Skills/Lessons, and export them for XPolicyLab Test.",
        )
        parser.add_argument("command", choices=("explore", "export", "test", "run"),
                            help="Use COMMAND --help for phase-specific options")
        parser.parse_args(args)
    caller = Path.cwd()
    for index, value in enumerate(args):
        key, separator, argument = value.partition("=")
        if key not in {"--run-dir", "--output", "--task"}:
            continue
        if not separator:
            if index + 1 >= len(args):
                continue  # Let argparse report the missing argument.
            argument = args[index + 1]
        path = Path(argument).expanduser()
        if key == "--task" and not (caller / path / "task.toml").is_file():
            continue  # Benchmark names resolve inside the bundled tasks/ tree.
        absolute = str((caller / path).resolve())
        if separator:
            args[index] = f"{key}={absolute}"
        else:
            args[index + 1] = absolute
    root = Path(__file__).resolve().parent
    os.chdir(root)
    # Subprocesses must also import this snapshot, including on spawn.
    os.environ["PYTHONPATH"] = str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    if args and args[0] == "export":
        from export_workspace import main as export

        return export(args[1:])
    from orchestrator.__main__ import main as orchestrate

    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
