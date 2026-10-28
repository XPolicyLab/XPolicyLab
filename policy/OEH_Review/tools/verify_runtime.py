#!/usr/bin/env python3
"""Check every file of the adapter against runtime/MANIFEST.json (the generated artifact's sha256 list).

Exit 0 when every listed file is present and unchanged and no unlisted file was added under runtime/; otherwise
print each difference and exit 1. The per-machine directories (state/, runs/, checkpoints/, runtime/kernel/target/)
are not part of the manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

POLICY_DIR = Path(__file__).resolve().parents[1]
MANIFEST = POLICY_DIR / "runtime" / "MANIFEST.json"
SKIP_NAMES = {"__pycache__", ".pytest_cache", ".DS_Store", "target", ".mypy_cache", ".ruff_cache"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if not MANIFEST.is_file():
        print(f"[verify_runtime] no manifest at {MANIFEST}: this is not a generated OEH_Review adapter", file=sys.stderr)
        return 1
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    problems = []
    for rel, expected in manifest["files"].items():
        path = POLICY_DIR / rel
        if not path.is_file():
            problems.append(f"missing: {rel}")
        elif sha256(path) != expected:
            problems.append(f"changed: {rel}")
    listed = set(manifest["files"])
    for path in sorted((POLICY_DIR / "runtime").rglob("*")):
        rel = path.relative_to(POLICY_DIR).as_posix()
        if not path.is_file() or rel == "runtime/MANIFEST.json" or rel.startswith("runtime/kernel/target/"):
            continue
        if any(part in SKIP_NAMES for part in path.relative_to(POLICY_DIR).parts) or rel.endswith((".pyc", ".pyo")):
            continue
        if rel not in listed:
            problems.append(f"unlisted: {rel}")
    source = manifest.get("source", {})
    if problems:
        print(f"[verify_runtime] {len(problems)} difference(s) from the generated adapter "
              f"(source commit {source.get('commit', '?')[:12]}):", file=sys.stderr)
        for line in problems:
            print("  " + line, file=sys.stderr)
        return 1
    print(f"[verify_runtime] {len(listed)} files match runtime/MANIFEST.json (OEH {source.get('commit', '?')[:12]}"
          f"{', dirty' if source.get('dirty') else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
