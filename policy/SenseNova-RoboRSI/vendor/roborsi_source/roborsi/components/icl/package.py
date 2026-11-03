# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Fail-closed, relocatable context packages for benchmark Agents."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

PACKAGE_TYPE = "rpent.icl.context"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: Any, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{field} must be a nonempty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"{field} must be a normalized relative path")
    return path


@dataclass(frozen=True, slots=True)
class ContextEntry:
    """One typed initial context file."""

    type: str
    path: Path
    media_type: str | None = None


@dataclass(frozen=True, slots=True)
class ContextPackage:
    """Verified context resources and their immutable identity."""

    root: Path
    package_id: str
    manifest_sha256: str
    memory: tuple[tuple[str, Path], ...]
    skills: tuple[Path, ...]
    skill_paths: tuple[Path, ...]
    initial_context: tuple[ContextEntry, ...]
    metadata: dict[str, Any]

    @classmethod
    def load(cls, root: str | Path) -> "ContextPackage":
        """Verify an entire sealed package before exposing any payload path."""
        selected = Path(os.path.abspath(root))
        if selected.is_symlink() or not selected.is_dir():
            raise ValueError("context package root must be a real directory")
        top = {entry.name: entry for entry in selected.iterdir()}
        if set(top) != {"manifest.json", "payload"}:
            raise ValueError(
                "context package root needs only manifest.json and payload"
            )
        manifest_path, payload = top["manifest.json"], top["payload"]
        if manifest_path.is_symlink() or payload.is_symlink() or not payload.is_dir():
            raise ValueError("context package manifest and payload cannot be links")
        try:
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid context package manifest") from exc
        if not isinstance(document, dict):
            raise ValueError("context package manifest must be an object")
        if (
            document.get("schema_version") != 1
            or document.get("package_type") != PACKAGE_TYPE
        ):
            raise ValueError("unsupported context package schema or type")
        package_id = document.get("package_id")
        if not isinstance(package_id, str) or not package_id.strip():
            raise ValueError("context package_id must be nonempty text")
        package_id_path = _relative_path(package_id, "context package_id")
        if len(package_id_path.parts) != 1 or package_id in {".", ".."}:
            raise ValueError("context package_id must be one safe path component")
        declared = document.get("files")
        if not isinstance(declared, dict) or not declared:
            raise ValueError("context package files must be a nonempty mapping")
        expected: dict[str, str] = {}
        for raw, digest in declared.items():
            path = _relative_path(raw, "files path")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError("context package file hashes must be SHA-256 hex")
            expected[path.as_posix()] = digest
        actual = {}
        for path in sorted(payload.rglob("*")):
            relative = path.relative_to(payload).as_posix()
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(f"context package cannot contain symlinks: {relative}")
            if path.is_dir():
                continue
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"context package needs regular files: {relative}")
            if info.st_nlink != 1:
                raise ValueError(
                    f"context package cannot contain hardlinks: {relative}"
                )
            actual[relative] = _sha256(path)
        if actual != expected:
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            changed = sorted(
                name
                for name in set(actual) & set(expected)
                if actual[name] != expected[name]
            )
            raise ValueError(
                f"context package file inventory differs: missing={missing}, "
                f"extra={extra}, changed={changed}"
            )

        def file(value: Any, field: str) -> Path:
            relative = _relative_path(value, field).as_posix()
            if relative not in actual:
                raise ValueError(f"{field} must name a declared file")
            return payload / relative

        memory = document.get("memory", [])
        if not isinstance(memory, list):
            raise ValueError("context package memory must be a list")
        memories = []
        for entry in memory:
            if not isinstance(entry, dict) or set(entry) != {"title", "path"}:
                raise ValueError("memory entries need only title and path")
            if not isinstance(entry["title"], str) or not entry["title"].strip():
                raise ValueError("memory title must be nonempty text")
            memories.append((entry["title"], file(entry["path"], "memory path")))

        skills = document.get("skills", [])
        skill_paths = document.get("skill_paths", [])
        if not isinstance(skills, list) or not isinstance(skill_paths, list):
            raise ValueError("skills and skill_paths must be lists")
        selected_skills = tuple(file(value, "skill") for value in skills)
        selected_dirs = []
        for value in skill_paths:
            relative = _relative_path(value, "skill path")
            directory = payload.joinpath(*relative.parts)
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("skill_paths must name declared payload directories")
            prefix = relative.as_posix().rstrip("/") + "/"
            if not any(name.startswith(prefix) for name in actual):
                raise ValueError("skill_paths directories cannot be empty")
            selected_dirs.append(directory)

        initial = document.get("initial_context", [])
        if not isinstance(initial, list):
            raise ValueError("initial_context must be a list")
        initial_context = []
        for entry in initial:
            if not isinstance(entry, dict) or set(entry) - {
                "type",
                "path",
                "media_type",
            }:
                raise ValueError("invalid initial_context entry")
            kind = entry.get("type")
            if kind not in {"text", "image"}:
                raise ValueError("initial context type must be text or image")
            media_type = entry.get("media_type")
            if media_type is not None and (
                not isinstance(media_type, str) or not media_type.strip()
            ):
                raise ValueError("initial context media_type must be nonempty text")
            initial_context.append(
                ContextEntry(
                    kind, file(entry.get("path"), "initial context path"), media_type
                )
            )
        metadata = document.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("context package metadata must be an object")
        return cls(
            root=selected,
            package_id=package_id_path.as_posix(),
            manifest_sha256=_sha256(manifest_path),
            memory=tuple(memories),
            skills=selected_skills,
            skill_paths=tuple(selected_dirs),
            initial_context=tuple(initial_context),
            metadata=json.loads(json.dumps(metadata, allow_nan=False)),
        )


def load_package_map(
    path: str | Path, package_root: str | Path
) -> dict[str, ContextPackage]:
    """Load task-to-package mappings whose values stay inside package_root."""
    mapping_path = Path(os.path.abspath(path))
    root = Path(os.path.abspath(package_root))
    if mapping_path.is_symlink() or not mapping_path.is_file():
        raise ValueError("context package map must be a regular non-link file")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("context package root must be a real directory")
    document = json.loads(mapping_path.read_text(encoding="utf-8"))
    if (
        not isinstance(document, dict)
        or set(document) != {"schema_version", "packages"}
        or document["schema_version"] != 1
        or not isinstance(document["packages"], dict)
        or not document["packages"]
    ):
        raise ValueError("context package map requires schema_version 1 and packages")
    result = {}
    for key, value in document["packages"].items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("context package map keys must be nonempty text")
        relative = _relative_path(value, "context package map value")
        selected = root.joinpath(*relative.parts)
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ValueError("context package map cannot traverse links")
        result[key] = ContextPackage.load(selected)
    return result
