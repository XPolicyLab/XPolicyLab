"""Pinned external-source loading shared by training-only integrations."""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceContract:
    path: str
    package: str
    expected_commit: str | None = None
    tracked_file: str | None = None
    tracked_file_sha256: str | None = None
    clean_paths: tuple[str, ...] = ()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_contract(contract: SourceContract) -> dict[str, str]:
    """Fail closed when an external source tree is not the reviewed version."""

    root = Path(contract.path).expanduser().resolve()
    if not (root / contract.package).is_dir():
        raise FileNotFoundError(f"Expected package {contract.package!r} under {root}")

    observed: dict[str, str] = {"path": str(root)}
    if contract.expected_commit:
        try:
            commit = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"Could not verify git commit for {root}") from exc
        if commit != contract.expected_commit:
            raise RuntimeError(f"Source commit mismatch for {root}: expected {contract.expected_commit}, got {commit}")
        observed["commit"] = commit

    if contract.clean_paths:
        result = subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", *contract.clean_paths],
            check=False,
        )
        if result.returncode == 1:
            raise RuntimeError(f"Reviewed source files have uncommitted changes under {root}: {contract.clean_paths}")
        if result.returncode != 0:
            raise RuntimeError(f"Could not check reviewed source files under {root}")
        observed["clean_paths"] = ",".join(contract.clean_paths)

    if contract.tracked_file:
        tracked_path = root / contract.tracked_file
        if not tracked_path.is_file():
            raise FileNotFoundError(f"Tracked source file does not exist: {tracked_path}")
        digest = _sha256(tracked_path)
        if contract.tracked_file_sha256 and digest != contract.tracked_file_sha256:
            raise RuntimeError(
                f"Tracked file checksum mismatch for {tracked_path}: "
                f"expected {contract.tracked_file_sha256}, got {digest}"
            )
        observed["tracked_file"] = str(tracked_path)
        observed["tracked_file_sha256"] = digest

    return observed


def import_external_package(contract: SourceContract) -> Any:
    """Import a package only from its verified source root."""

    verify_source_contract(contract)
    root = Path(contract.path).expanduser().resolve()

    def belongs_to_root(module: Any) -> tuple[bool, str]:
        locations = []
        module_file = getattr(module, "__file__", None)
        if module_file:
            locations.append(Path(module_file).resolve())
        module_path = getattr(module, "__path__", None)
        if module_path:
            locations.extend(Path(value).resolve() for value in module_path)
        return any(location.is_relative_to(root) for location in locations), ",".join(map(str, locations))

    existing = sys.modules.get(contract.package)
    if existing is not None:
        valid, locations = belongs_to_root(existing)
        if not valid:
            raise RuntimeError(f"Package {contract.package!r} is imported from {locations}, expected {root}")
        return existing
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    imported = importlib.import_module(contract.package)
    valid, locations = belongs_to_root(imported)
    if not valid:
        raise RuntimeError(f"Imported {contract.package!r} from {locations}, expected a package under {root}")
    return imported
