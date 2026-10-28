#!/usr/bin/env python3
"""Import checks for the two Python environments OEH_Review runs in.

  check_deps.py policy   the policy server's Python (OpenWAM policy env): XPolicyLab + vendored OpenWAM + torch,
                         websockets, numpy, Pillow, pyyaml, and the vendored runtime (PYTHONPATH=runtime/)
  check_deps.py eval     the evaluation client's Python (RoboDojo env): numpy, Pillow, pyyaml, the vendored runtime
"""
from __future__ import annotations

import importlib
import sys

POLICY = ("numpy", "PIL", "yaml", "websockets", "torch", "XPolicyLab", "openwam", "XPolicyLab.policy.OpenWAM.model",
          "oeh_sdk", "oeh_policy", "oeh_policy.model", "oeh_eval.direct_bridge")
EVAL = ("numpy", "PIL", "yaml", "oeh_sdk", "oeh_policy.deploy", "oeh_eval.direct_bridge")


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"policy", "eval"}:
        print(__doc__, file=sys.stderr)
        return 2
    modules = POLICY if sys.argv[1] == "policy" else EVAL
    missing = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - report every failing import, whatever its cause
            missing.append(f"{name}: {type(exc).__name__}: {exc}")
    if missing:
        print(f"[check_deps] {sys.argv[1]} environment {sys.executable} cannot import:", file=sys.stderr)
        for line in missing:
            print("  " + line, file=sys.stderr)
        return 1
    print(f"[check_deps] {sys.argv[1]} environment {sys.executable}: {len(modules)} imports ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
