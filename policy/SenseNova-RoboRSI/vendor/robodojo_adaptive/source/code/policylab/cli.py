import argparse
import ast
import json
from pathlib import Path
import shutil
import sys

from .common import load_config, validate_paths, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run autonomous exploration and continual evaluation with your benchmark adapter.")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Write an editable experiment configuration; no installation or execution")
    init.add_argument("config", type=Path)
    init.add_argument("--adapter", required=True, help="/path/to/adapter.py:Adapter")
    init.add_argument("--environment-dir", required=True)
    init.add_argument("--training-set", required=True)
    init.add_argument("--workspace", required=True, help="Persistent agent workspace directory")
    init.add_argument("--benchmark-python", default=sys.executable)
    for name in ("check", "run"):
        cmd = commands.add_parser(name, help="Check configuration without launching models/environments" if name == "check" else "Start a real experiment")
        cmd.add_argument("config", type=Path)
        if name == "check":
            cmd.add_argument("--adapter-only", action="store_true", help="Do not check the local Codex installation/login")
        else:
            cmd.add_argument("--output", type=Path, default=Path("runs"))
    inspect = commands.add_parser("inspect", help="Print the saved run summary")
    inspect.add_argument("run", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            if args.config.exists():
                raise ValueError("Configuration already exists; edit it directly or choose a new path")
            file, cls = args.adapter.rsplit(":", 1)
            cfg = {"adapter": str(Path(file).resolve()) + ":" + cls,
                   "environment_dir": str(Path(args.environment_dir).resolve()),
                   "training_set": str(Path(args.training_set).resolve()),
                   "agent_workspace": str(Path(args.workspace).resolve()),
                   "benchmark_python": str(Path(args.benchmark_python).absolute()),
                   "environment_revision": "record-your-benchmark-version",
                   "train_episodes": [],
                   "test_episodes": [{"id": "test-1", "task": "your-task", "seed": 1001}],
                   "options": {}, "public_instructions": "", "model": "gpt-6-astra",
                   "reasoning_effort": "max", "codex_binary": "codex", "agent_python": "/usr/bin/python3",
                   "explore_seconds": 3600, "episode_seconds": 900, "request_seconds": 600,
                   "terminal_feedback": False}
            write_json(args.config, cfg)
            print(f"Created {args.config}. Edit task/episode definitions and public instructions before running.")
            return 0
        if args.command == "inspect":
            print((args.run / "summary.json").read_text())
            return 0
        cfg = load_config(args.config)
        receipt = validate_paths(cfg)
        ast.parse(Path(cfg["adapter"].rsplit(":", 1)[0]).read_text())
        if args.command == "run" or not args.adapter_only:
            from .codex import auth_file, native_binary
            native_binary(cfg["codex_binary"])
            auth_file(cfg)
            if not shutil.which("bwrap"):
                raise ValueError("Install bubblewrap before running the Codex launcher")
        if args.command == "check":
            print(json.dumps({"configuration": "valid", **receipt,
                              "scope": "Paths, syntax and configuration only. No adapter import, simulator, model or login request was executed."}, indent=2))
            return 0
        from .runner import run_experiment
        run, summary = run_experiment(cfg, args.output)
        print(json.dumps({"run": str(run), **summary}, indent=2))
        return 0 if summary["status"] == "completed" else 1
    except (ValueError, KeyError, OSError) as exc:
        print(f"PolicyLab: {exc}", file=sys.stderr)
        return 2
