#!/usr/bin/env python3
"""Verify the CSU-AI-0 Orbax checkpoint against its checked-in SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path


POLICY_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = POLICY_DIR / "checkpoints" / "CSU-AI-0"
DEFAULT_MANIFEST = POLICY_DIR / "CSU-AI-0.sha256"


@dataclass(frozen=True)
class ManifestEntry:
    sha256: str
    relative_path: Path


def load_manifest(path: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    seen: set[Path] = set()

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            digest, manifest_path = line.split(maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"Malformed manifest line {line_number}: {raw_line!r}") from exc

        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"Invalid SHA-256 on manifest line {line_number}")

        full_path = Path(manifest_path)
        if full_path.is_absolute() or ".." in full_path.parts:
            raise ValueError(f"Unsafe path on manifest line {line_number}: {manifest_path!r}")
        if not full_path.parts or full_path.parts[0] != "CSU-AI-0":
            raise ValueError(f"Manifest line {line_number} must start with CSU-AI-0/")

        relative_path = Path(*full_path.parts[1:])
        if not relative_path.parts or relative_path in seen:
            raise ValueError(f"Duplicate or empty path on manifest line {line_number}: {manifest_path!r}")
        seen.add(relative_path)
        entries.append(ManifestEntry(digest, relative_path))

    if not entries:
        raise ValueError(f"Manifest is empty: {path}")
    return entries


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(checkpoint_dir: Path, manifest_path: Path, *, reject_extra: bool = True) -> None:
    checkpoint_dir = checkpoint_dir.resolve()
    entries = load_manifest(manifest_path)
    expected = {entry.relative_path for entry in entries}

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    actual = {
        path.relative_to(checkpoint_dir)
        for path in checkpoint_dir.rglob("*")
        if path.is_file()
    }
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise RuntimeError("Missing checkpoint files: " + ", ".join(map(str, missing)))
    if reject_extra and extra:
        raise RuntimeError("Unexpected checkpoint files: " + ", ".join(map(str, extra)))

    for index, entry in enumerate(entries, 1):
        file_path = checkpoint_dir / entry.relative_path
        actual_digest = sha256_file(file_path)
        if actual_digest != entry.sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {entry.relative_path}: "
                f"expected {entry.sha256}, got {actual_digest}"
            )
        print(f"[{index:02d}/{len(entries):02d}] OK {entry.relative_path}")

    print(f"Verified {len(entries)} files in {checkpoint_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", nargs="?", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--allow-extra-files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_checkpoint(args.checkpoint_dir, args.manifest, reject_extra=not args.allow_extra_files)


if __name__ == "__main__":
    main()
