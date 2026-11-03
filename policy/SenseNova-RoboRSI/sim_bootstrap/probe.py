"""Run the portable native namespace probe without importing Isaac/Kit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _roborsi_sim_namespace import BootstrapError, check_sim_namespace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[4],
        help="native RoboDojo root (default: parent of the XPolicyLab checkout)",
    )
    parser.add_argument("--allow-missing-workspace", action="store_true")
    args = parser.parse_args()
    try:
        result = check_sim_namespace(
            args.workspace, allow_missing_workspace=args.allow_missing_workspace
        )
    except BootstrapError as exc:
        print(f"[ROBORSI sim bootstrap] {exc}", file=sys.stderr)
        return 78
    if result["status"] == "skipped":
        print(f"[ROBORSI sim bootstrap] WARNING: {result['reason']}", file=sys.stderr)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
