"""Download or verify a versioned model bundle using its public manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


ROOT = Path(__file__).resolve().parent


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def check(path, row):
    return path.is_file() and path.stat().st_size == row["bytes"] and digest(path) == row["sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--manifest", type=Path, default=ROOT / "weights_manifest.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = manifest["files"]
    if not rows:
        raise RuntimeError("The release manifest has no files.")
    for row in rows:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Invalid release path.")
        if not isinstance(row.get("bytes"), int) or row["bytes"] <= 0:
            raise RuntimeError("Invalid file size in weights_manifest.json.")
        value = row.get("sha256")
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise RuntimeError("Invalid SHA256 in weights_manifest.json.")
        if not args.verify_only and args.source_dir is None:
            if not str(row.get("url") or "").startswith("https://"):
                raise RuntimeError("Asset URL is not configured in weights_manifest.json.")
    for row in rows:
        destination = args.output_dir / row["path"]
        if destination.exists():
            if not check(destination, row):
                raise RuntimeError(f"Existing file differs from the manifest: {row['path']}")
            print(f"verified {row['path']}")
            continue
        if args.verify_only:
            raise FileNotFoundError(f"Missing bundle file: {row['path']}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        if temporary.exists():
            raise RuntimeError(f"An incomplete download needs inspection: {temporary}")
        if args.source_dir is not None:
            with (args.source_dir / row["path"]).open("rb") as source, temporary.open("xb") as target:
                shutil.copyfileobj(source, target, 8 * 1024 * 1024)
        else:
            with urllib.request.urlopen(row["url"], timeout=120) as source, temporary.open("xb") as target:
                shutil.copyfileobj(source, target, 8 * 1024 * 1024)
        if not check(temporary, row):
            raise RuntimeError(f"Downloaded bytes fail verification: {row['path']}")
        temporary.replace(destination)
        print(f"verified {row['path']}")


if __name__ == "__main__":
    main()
