# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Independent writable copies and explicit snapshots of ICL materials."""

from __future__ import annotations

import json
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roborsi.components.icl.build import (
    ContextExports,
    _checked_absolute,
    _copy_tree,
    _json_object,
    _prepare_paths,
    _publish_directory,
    build_package,
)
from roborsi.components.icl.package import ContextPackage, _sha256


@dataclass(frozen=True)
class IclWorkspace:
    """Caller-owned mutable materials, their exports, and source identity."""

    root: Path
    source_package_id: str
    source_manifest_sha256: str
    exports: ContextExports

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_package_id, str)
            or not self.source_package_id.strip()
        ):
            raise ValueError("workspace source_package_id must be nonempty text")
        digest = self.source_manifest_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("workspace source_manifest_sha256 must be SHA-256 hex")
        if not isinstance(self.exports, ContextExports):
            raise ValueError("workspace exports must be ContextExports")
        object.__setattr__(self, "root", Path(self.root))


def _package_exports(package: ContextPackage) -> ContextExports:
    payload = package.root / "payload"

    def relative(path: Path) -> str:
        return path.relative_to(payload).as_posix()

    return ContextExports(
        memory=tuple((title, relative(path)) for title, path in package.memory),
        skills=tuple(relative(path) for path in package.skills),
        skill_paths=tuple(relative(path) for path in package.skill_paths),
        initial_context=tuple(
            {
                "type": entry.type,
                "path": relative(entry.path),
                **(
                    {"media_type": entry.media_type}
                    if entry.media_type is not None
                    else {}
                ),
            }
            for entry in package.initial_context
        ),
    )


def _copy_mismatch_details(
    source: Path,
    staging: Path,
    expected: dict[str, str],
    copied: dict[str, str],
) -> str:
    """Preserve bounded evidence before failed staging directories are removed."""
    missing = sorted(set(expected) - set(copied))
    extra = sorted(set(copied) - set(expected))
    changed = sorted(
        name for name in set(expected) & set(copied)
        if expected[name] != copied[name]
    )

    def current_file(path: Path) -> dict[str, Any]:
        try:
            _checked_absolute(path)
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                return {"error": "not an independent regular file"}
            return {"bytes": info.st_size, "sha256": _sha256(path)}
        except (OSError, ValueError) as exc:
            # Diagnostic reads cannot replace the original integrity failure.
            return {"error": type(exc).__name__}

    details = []
    for kind, names in (("missing", missing), ("extra", extra), ("changed", changed)):
        for name in names:
            if len(details) == 20:
                break
            details.append({
                "kind": kind,
                "path": name,
                "expected_sha256": expected.get(name),
                "copied_sha256": copied.get(name),
                "source_current": current_file(source / name),
                "target_current": current_file(staging / name),
            })
    return json.dumps({
        "missing": len(missing), "extra": len(extra), "changed": len(changed),
        "omitted": len(missing) + len(extra) + len(changed) - len(details),
        "details": details,
    }, sort_keys=True)


def materialize_package(package: ContextPackage, destination: Path) -> IclWorkspace:
    """Reverify and copy every payload file into a new independent directory.

    Args:
        package: Previously verified package with the expected manifest identity.
        destination: New workspace directory outside the package tree.

    Returns:
        A writable copy with relative exports and the original package identity.
    """
    source, destination = _prepare_paths(package.root, destination)
    verified = ContextPackage.load(source)
    if (
        verified.manifest_sha256 != package.manifest_sha256
        or verified.package_id != package.package_id
    ):
        raise ValueError("source package identity changed")
    exports = _package_exports(verified)
    expected = json.loads(
        (verified.root / "manifest.json").read_text(encoding="utf-8")
    )["files"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / "workspace"
        copied = _copy_tree(verified.root / "payload", staging)
        if copied != expected:
            raise ValueError(
                "source package changed during materialization; "
                + _copy_mismatch_details(verified.root / "payload", staging, expected, copied)
            )
        _publish_directory(staging, destination)
    return IclWorkspace(
        destination, verified.package_id, verified.manifest_sha256, exports
    )


def snapshot_workspace(
    workspace: IclWorkspace,
    destination: Path,
    *,
    package_id: str,
    exports: ContextExports | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ContextPackage:
    """Seal current workspace contents with an explicit parent manifest identity.

    The caller must stop writes before snapshotting. This does not provide an
    atomic snapshot of concurrent file-system mutations.

    Args:
        workspace: Mutable source directory and provenance.
        destination: New package directory outside the workspace.
        package_id: Identity for the new package.
        exports: Updated declarations, or the workspace declarations by default.
        metadata: Additional provenance; a conflicting parent identity is rejected.

    Returns:
        A new verified package without changing its workspace or parent package.
    """
    document = _json_object(metadata, "metadata")
    icl = document.setdefault("icl", {})
    if not isinstance(icl, dict):
        raise ValueError("metadata.icl must be an object")
    parent = icl.get("parent_manifest_sha256", workspace.source_manifest_sha256)
    if parent != workspace.source_manifest_sha256:
        raise ValueError(
            "metadata.icl.parent_manifest_sha256 conflicts with workspace source"
        )
    icl["parent_manifest_sha256"] = workspace.source_manifest_sha256
    return build_package(
        workspace.root,
        destination,
        package_id=package_id,
        exports=workspace.exports if exports is None else exports,
        metadata=document,
    )
