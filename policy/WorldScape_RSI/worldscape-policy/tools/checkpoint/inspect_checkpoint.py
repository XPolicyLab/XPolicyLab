#!/usr/bin/env python3
"""Inspect and validate a native WorldScape checkpoint bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from worldscape_policy.checkpoint.loader import validate_native_checkpoint_artifacts


def inspect_checkpoint(path: Path) -> dict[str, object]:
    if not (path / "checkpoint_manifest.json").is_file():
        raise FileNotFoundError(
            f"{path} is not a native checkpoint bundle; "
            "expected checkpoint_manifest.json"
        )
    manifest = validate_native_checkpoint_artifacts(path)
    return {"kind": "native", "path": str(path), "manifest": manifest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    rendered = json.dumps(inspect_checkpoint(args.checkpoint), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
