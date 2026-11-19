#!/usr/bin/env python3
"""Check release provenance, checksums and source syntax without a GPU."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRS = {".git", "__pycache__", ".cache", ".venv", ".pytest_cache", "runs", "logs"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release_files(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if rel.as_posix().startswith("agent/build/") or any(part in IGNORED_DIRS for part in rel.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Unexpected symlink: {rel}")
        if not path.is_file() or path.suffix in (".pyc", ".pyo"):
            continue
        if rel.as_posix() == "metadata/release_manifest.json":
            continue
        if path.name in {"config.env", ".env"} or path.name.endswith(".local.json"):
            raise ValueError(f"Private configuration present in release tree: {rel}")
        result[rel.as_posix()] = digest(path)
    return result


def verify_source(root: Path) -> dict:
    manifest = json.loads((root / "metadata/variant_manifest.json").read_text())
    unchanged = 0
    for rel, original in manifest["source_files"].items():
        path = root / rel
        if rel in manifest["omitted_files"]:
            if path.exists():
                raise ValueError(f"Omitted source file is present: {rel}")
            continue
        expected = manifest["modified_files"].get(rel, {}).get("sha256", original)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Undocumented variant change: {rel}")
        unchanged += rel not in manifest["modified_files"]
    for rel in ("tasks", "agent/api/skills", "agent/recipes", "roboshell/recipe.py", "results"):
        if (root / rel).exists():
            raise ValueError(f"Task assets or prior results present: {rel}")
    for manual in (root / "agent/build").glob("AGENTS_*.md"):
        if manual.name != "AGENTS_generic.md":
            raise ValueError(f"Task manual present: {manual.name}")
    settings = json.loads((root / "configs/model_api.example.json").read_text())
    if settings["skills"] is not False:
        raise ValueError("This variant requires skills=false")
    reference = json.loads((root / "metadata/source_manifest.json").read_text())["reference_service"]
    if digest(root / reference["path"]) != reference["sha256"]:
        raise ValueError("Shared service bytes changed")
    return {"variant": "joyai-vl-agent", "unchanged_policy_files": unchanged,
            "documented_variant_edits": len(manifest["modified_files"]),
            "omitted_source_files": len(manifest["omitted_files"]),
            "shared_service": "verified"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true",
                        help="Refresh packaging checksums after validating the variant source relationship")
    args = parser.parse_args()
    try:
        report = verify_source(ROOT)
        files = release_files(ROOT)
        for rel in files:
            path = ROOT / rel
            if path.suffix == ".py":
                ast.parse(path.read_bytes(), filename=rel)
            elif path.suffix == ".sh":
                subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True)
        manifest_path = ROOT / "metadata/release_manifest.json"
        if args.write_manifest:
            manifest_path.write_text(json.dumps({"algorithm": "sha256", "files": files}, indent=2) + "\n")
        expected = json.loads(manifest_path.read_text())["files"]
        if files != expected:
            changed = sorted(rel for rel in files.keys() | expected.keys()
                             if files.get(rel) != expected.get(rel))
            raise ValueError(f"Release checksum mismatch: {', '.join(changed)}")
        report["checksummed_release_files"] = len(files)
        report["syntax"] = "Python and shell passed"
    except (ValueError, SyntaxError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Verification failed: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
