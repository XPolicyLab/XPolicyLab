# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Build deterministic sealed packages from explicitly selected file trees."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from roborsi.components.icl.package import (
    PACKAGE_TYPE,
    ContextPackage,
    _relative_path,
    _sha256,
)


def _portable_path(value: Any, field: str) -> str:
    relative = _relative_path(value, field)
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{index}" for prefix in ("COM", "LPT") for index in range(1, 10)
    }
    if not relative.parts or PureWindowsPath(value).drive:
        raise ValueError(f"{field} must be a portable relative path")
    for part in relative.parts:
        if (
            any(ord(char) < 32 or char in '<>:"|?*' for char in part)
            or part.endswith((" ", "."))
            or part.split(".")[0].upper() in reserved
        ):
            raise ValueError(f"{field} contains a nonportable file name")
    return relative.as_posix()


def _json_object(value: Mapping[str, Any] | None, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")

    def plain(item: Any) -> Any:
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise ValueError(f"{field} keys must be text")
            return {key: plain(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [plain(child) for child in item]
        return item

    try:
        return json.loads(json.dumps(plain(value), allow_nan=False))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{field} must contain finite JSON values") from exc


@dataclass(frozen=True)
class ContextExports:
    """Explicit context declarations, with paths relative to the material root.

    Args:
        memory: Pairs of display titles and text file paths.
        skills: Skill files to preload.
        skill_paths: Directories available for on-demand skill discovery.
        initial_context: Typed text/image file declarations from the v1 format.
    """

    memory: tuple[tuple[str, str], ...] = ()
    skills: tuple[str, ...] = ()
    skill_paths: tuple[str, ...] = ()
    initial_context: tuple[Mapping[str, str], ...] = ()

    def __post_init__(self) -> None:
        document = _exports_document(self)
        object.__setattr__(
            self,
            "memory",
            tuple((entry["title"], entry["path"]) for entry in document["memory"]),
        )
        object.__setattr__(self, "skills", tuple(document["skills"]))
        object.__setattr__(self, "skill_paths", tuple(document["skill_paths"]))
        object.__setattr__(self, "initial_context", tuple(document["initial_context"]))


def _exports_document(exports: ContextExports) -> dict[str, Any]:
    if not isinstance(exports, ContextExports):
        raise ValueError("exports must be ContextExports")
    for field in ("memory", "skills", "skill_paths", "initial_context"):
        if not isinstance(getattr(exports, field), (tuple, list)):
            raise ValueError(f"exports.{field} must be a sequence")
    memory = []
    for entry in exports.memory:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError("memory entries need title and path")
        title, path = entry
        if not isinstance(title, str) or not title.strip():
            raise ValueError("memory title must be nonempty text")
        memory.append({"title": title, "path": _portable_path(path, "memory path")})
    initial = []
    for entry in exports.initial_context:
        if not isinstance(entry, Mapping) or set(entry) - {
            "type",
            "path",
            "media_type",
        }:
            raise ValueError("invalid initial_context entry")
        if entry.get("type") not in {"text", "image"}:
            raise ValueError("initial context type must be text or image")
        selected = {
            "type": entry["type"],
            "path": _portable_path(entry.get("path"), "initial context path"),
        }
        if "media_type" in entry:
            media_type = entry["media_type"]
            if not isinstance(media_type, str) or not media_type.strip():
                raise ValueError("initial context media_type must be nonempty text")
            selected["media_type"] = media_type
        initial.append(selected)
    return {
        "memory": memory,
        "skills": [_portable_path(path, "skill") for path in exports.skills],
        "skill_paths": [
            _portable_path(path, "skill path") for path in exports.skill_paths
        ],
        "initial_context": initial,
    }


def _checked_absolute(path: Path) -> Path:
    selected = Path(os.path.abspath(path))
    for part in reversed((selected, *selected.parents)):
        if part.is_symlink() or getattr(part, "is_junction", lambda: False)():
            raise ValueError("material paths cannot traverse links")
    return selected


def _prepare_paths(source: Path, destination: Path) -> tuple[Path, Path]:
    source, destination = _checked_absolute(source), _checked_absolute(destination)
    if not source.is_dir():
        raise ValueError("source must be a real directory")
    if os.path.lexists(destination):
        raise ValueError("destination must not exist")
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError("source and destination must not be nested")
    return source, destination


def _tree_entries(source: Path) -> list[tuple[Path, bool]]:
    entries: list[tuple[Path, bool]] = []
    seen: set[str] = set()

    def visit(directory: Path) -> None:
        for path in sorted(directory.iterdir()):
            relative = _portable_path(
                path.relative_to(source).as_posix(), "source path"
            )
            key = unicodedata.normalize("NFC", relative).casefold()
            if key in seen:
                raise ValueError(f"portable file name collision: {relative}")
            seen.add(key)
            info = path.lstat()
            if (
                stat.S_ISLNK(info.st_mode)
                or getattr(path, "is_junction", lambda: False)()
            ):
                raise ValueError(f"source cannot contain links: {relative}")
            if stat.S_ISDIR(info.st_mode):
                entries.append((path, True))
                visit(path)
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                entries.append((path, False))
            else:
                raise ValueError(f"source needs independent regular files: {relative}")

    visit(source)
    return entries


def _copy_tree(source: Path, destination: Path) -> dict[str, str]:
    entries = _tree_entries(source)
    if not any(not directory for _, directory in entries):
        raise ValueError("source file tree cannot be empty")
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for path, directory in entries:
        relative = path.relative_to(source)
        target = destination / relative
        if directory:
            target.mkdir()
        else:
            # Recheck after the inventory walk; copying never intentionally follows links.
            _checked_absolute(path)
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError(f"source file changed type: {relative}")
            shutil.copyfile(path, target, follow_symlinks=False)
            if target.is_symlink():
                raise ValueError(f"source file became a link: {relative}")
            files[relative.as_posix()] = _sha256(target)
    return files


def _publish_directory(staging: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        raise ValueError("destination must not exist")
    os.rename(staging, destination)


def build_package(
    source: Path,
    destination: Path,
    *,
    package_id: str,
    exports: ContextExports | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ContextPackage:
    """Copy a complete file tree into a verified, deterministic v1 package.

    Args:
        source: Existing self-contained material directory.
        destination: New package directory outside the source tree.
        package_id: One portable path component identifying the package.
        exports: Explicit resources exposed as context.
        metadata: Finite JSON provenance supplied by the caller.

    Returns:
        The published, verified package.

    Raises:
        ValueError: Inputs, declarations, file types, or paths are invalid.
    """
    source, destination = _prepare_paths(source, destination)
    identifier = _portable_path(package_id, "package_id")
    if "/" in identifier:
        raise ValueError("package_id must be one safe path component")
    document = {
        "schema_version": 1,
        "package_type": PACKAGE_TYPE,
        "package_id": identifier,
        **_exports_document(exports if exports is not None else ContextExports()),
        "metadata": _json_object(metadata, "metadata"),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / "package"
        document["files"] = _copy_tree(source, staging / "payload")
        (staging / "manifest.json").write_text(
            json.dumps(
                document,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        ContextPackage.load(staging)
        _publish_directory(staging, destination)
    return ContextPackage.load(destination)
