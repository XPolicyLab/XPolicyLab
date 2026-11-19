#!/usr/bin/env python3
"""Copy the release into RoboDojo's native policy directory."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
POLICY_NAME = "JoyAI_VL"


def install(robodojo_repo: Path) -> Path:
    repo = robodojo_repo.expanduser().resolve()
    if not (repo / "XPolicyLab/setup_policy_server.py").is_file():
        raise ValueError("Expected a RoboDojo checkout with XPolicyLab/setup_policy_server.py")
    destination = repo / "XPolicyLab/policy" / POLICY_NAME
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Policy already exists; use a fresh checkout: {destination}")
    source = ROOT
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(
        "config.env", "__pycache__", "*.pyc", ".cache", "runs", ".venv"))
    # Docker's agent user must be able to traverse and read the mounted code.
    for path in [destination, *destination.rglob("*")]:
        path.chmod(0o755 if path.is_dir() or path.suffix == ".sh" else 0o644)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodojo-repo", required=True, type=Path)
    args = parser.parse_args()
    try:
        destination = install(args.robodojo_repo)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Installed {destination}")


if __name__ == "__main__":
    main()
