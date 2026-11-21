#!/usr/bin/env python3
"""Forward a task and arbitrary seed to RoboDojo's native evaluation entry."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex


def command(args: argparse.Namespace) -> tuple[Path, list[str]]:
    policy = args.robodojo_repo.expanduser().resolve() / "XPolicyLab/policy/JoyAI_VL"
    if not (policy / "eval.sh").is_file():
        raise ValueError("Install the policy first with tools/install_policy.py")
    if not (policy / "config.env").is_file():
        raise ValueError("Configure the installed policy first with tools/configure.py")
    return policy, ["bash", str(policy / "eval.sh"), "RoboDojo", args.task,
                    args.label, "arx_x5", "joint", str(args.seed), str(args.policy_gpu),
                    str(args.env_gpu), "unused", str(args.eval_env.expanduser().resolve())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodojo-repo", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--eval-env", required=True, type=Path,
                        help="RoboDojo Isaac environment directory (or its Python executable)")
    parser.add_argument("--policy-gpu", type=int, default=0)
    parser.add_argument("--env-gpu", type=int, default=0)
    parser.add_argument("--label", default="JoyAI-VL", help="Result label, not the API model ID")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running an episode")
    args = parser.parse_args()
    try:
        policy, argv = command(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(shlex.join(argv), flush=True)
    if not args.dry_run:
        os.chdir(policy)
        # Inherit the evaluator's layout selection; do not inject local seed/layout overrides.
        os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
