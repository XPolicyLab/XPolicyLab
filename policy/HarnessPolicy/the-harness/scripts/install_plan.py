#!/usr/bin/env python3
"""Print future installation steps. This script never executes them."""

import argparse
from pathlib import Path
import shlex
import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", choices=["robodojo", "robolab"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = tomllib.loads((root / "third_party/sources.toml").read_text())
    source = sources[args.environment]
    target = source["path"]
    revision = source["revision"] or "<SET_FULL_COMMIT_SHA_IN_sources.toml>"
    commands = [
        ["git", "clone", "--no-checkout", source["url"], target],
        ["git", "-C", target, "checkout", "--detach", revision],
    ]
    print("PLAN ONLY — no commands are executed. Run future setup on a Linux NVIDIA GPU host.")
    print(f"Repository root: {root}")
    for command in commands:
        print(shlex.join(command))
    if args.environment == "robodojo":
        print(f"git -C {target} submodule update --init --recursive")
        for rel, sha in (source.get("nested") or {}).items():
            if rel == "assets":
                print(f"# Asset checkout {sha} is separate from the RoboDojo git tree.")
                continue
            print(f"git -C {target}/{rel} checkout --detach {sha}")
        print(f"cd {target}")
        print("bash scripts/install.sh --install")
        print("# Assets are a separate, explicit download: bash scripts/init_assets.sh")
        print("# The nested SHAs above are the ones used by the stack_bowls transport check.")
        print("# Upstream install.sh may advance nested submodules with --remote.")
        print("# Re-validate before replacing the recorded nested revisions.")
    else:
        print(f"cd {target}")
        print("uv venv --python 3.11")
        print("uv sync --extra isaac50  # Choose the stack for your experiment; isaac51 is an alternative.")
    print("No environment, assets, credentials or model calls were created by this command.")


if __name__ == "__main__":
    main()
