from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys


HERE = pathlib.Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path)
    args = parser.parse_args()

    payload = json.loads((HERE / "experts.json").read_text(encoding="utf-8"))
    root = args.root or pathlib.Path(
        os.environ.get("CSU_CHECKPOINT_ROOT", payload["bundle"]["root"])
    )
    root = root.expanduser()
    if not root.is_absolute():
        root = HERE / root
    root = root.resolve()

    errors: list[str] = []
    resolved: dict[str, str] = {}
    for name, spec in payload["experts"].items():
        path = (root / spec["checkpoint_relpath"]).resolve()
        resolved[name] = str(path)
        if not path.is_dir():
            errors.append(f"{name}: missing directory {path}")

    xiaomi_state = pathlib.Path(resolved["xiaomi"]) / "model_states.pt"
    if xiaomi_state.is_file() and xiaomi_state.stat().st_size != 10226684853:
        errors.append(
            f"xiaomi: model_states.pt size {xiaomi_state.stat().st_size} != 10226684853"
        )

    report = {
        "bundle_root": str(root),
        "experts": resolved,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
