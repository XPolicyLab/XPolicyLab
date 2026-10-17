"""Unpack a pinned native Codex release without installing its Python wrapper."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


def unpack_cli_archive(
    metadata: dict[str, Any], archive: Path, destination: Path
) -> Path:
    """Authenticate the complete artifact and return its standalone package root."""
    cli = metadata["cli"]
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != cli["archive_sha256"]:
        raise ValueError("Codex native installation artifact SHA-256 mismatch")
    destination.mkdir()
    if cli.get("archive_format", "tar.gz") == "wheel":
        prefix = "codex_cli_bin/"
        seen = set()
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                if not member.filename.startswith(prefix) or member.is_dir():
                    continue
                relative = PurePosixPath(member.filename[len(prefix) :])
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or relative.as_posix() in {".", ""}
                    or relative.as_posix() in seen
                ):
                    raise ValueError("Unsafe native CLI wheel payload path")
                mode = member.external_attr >> 16
                if stat.S_IFMT(mode) not in {0, stat.S_IFREG}:
                    raise ValueError("Native CLI wheel payload must be regular files")
                seen.add(relative.as_posix())
                target = destination.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with package.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(0o755 if mode & 0o111 else 0o644)
        root = destination
    elif cli.get("archive_format", "tar.gz") == "tar.gz":
        with tarfile.open(archive, "r:gz") as package:
            package.extractall(destination, filter="data")
        manifests = list(destination.rglob("codex-package.json"))
        if len(manifests) != 1:
            raise ValueError("Codex artifact must contain one package manifest")
        root = manifests[0].parent
    else:
        raise ValueError("Unsupported pinned native CLI artifact format")
    try:
        package = json.loads((root / "codex-package.json").read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("Codex native package manifest is missing or invalid") from exc
    if any(
        package.get(key) != value
        for key, value in {
            "version": cli["version"],
            "target": cli["target"],
            "entrypoint": "bin/codex",
        }.items()
    ):
        raise ValueError("Codex native package identity differs from the fixed release")
    return root
