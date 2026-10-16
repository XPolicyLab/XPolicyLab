# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Import one self-contained PolicyLab task directory into the sealed format."""

from __future__ import annotations

import json
import shutil
import stat
import tempfile
from pathlib import Path, PureWindowsPath
from typing import Any

from roborsi.components.icl.build import (
    ContextExports,
    _checked_absolute,
    _copy_tree,
    _json_object,
    _prepare_paths,
    _tree_entries,
    build_package,
)
from roborsi.components.icl.package import ContextPackage

_SOURCE_ATTACHMENTS = "__policylab_source__"


def _regular_file(path: Path) -> None:
    _checked_absolute(path)
    try:
        info = path.lstat()
    except OSError as exc:
        raise ValueError(f"PolicyLab input file is missing: {path.name}") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(
            f"PolicyLab input must be an independent regular file: {path.name}"
        )


def _has_historical_path(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith("/") or bool(PureWindowsPath(value).drive)
    if isinstance(value, dict):
        return any(_has_historical_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_historical_path(item) for item in value)
    return False


def import_policylab_workspace(
    source: Path,
    destination: Path,
    *,
    package_id: str,
    exports: ContextExports | None = None,
) -> ContextPackage:
    """Import a single task's workspace and preserve its original provenance.

    Args:
        source: Task directory containing workspace/ and provenance.json.
        destination: New sealed package directory outside the source tree.
        package_id: Identity of the imported package.
        exports: Declarations relative to the original workspace root.

    Returns:
        A verified package containing materials and original source attachments.

    Raises:
        ValueError: The source is incomplete, linked, or has invalid declarations.
    """
    source, destination = _prepare_paths(source, destination)
    workspace = _checked_absolute(source / "workspace")
    if not workspace.is_dir():
        raise ValueError("PolicyLab source must be one task with a workspace directory")
    if any(
        path.name.casefold() == _SOURCE_ATTACHMENTS.casefold()
        for path in workspace.iterdir()
    ):
        raise ValueError(
            f"workspace already occupies reserved {_SOURCE_ATTACHMENTS} directory"
        )
    # Validate the whole tree before making a staging copy.
    _tree_entries(workspace)
    provenance_path = source / "provenance.json"
    _regular_file(provenance_path)
    try:
        raw_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if not isinstance(raw_provenance, dict):
            raise ValueError("PolicyLab provenance must be an object")
        provenance = _json_object(raw_provenance, "PolicyLab provenance")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "PolicyLab provenance.json must contain a JSON object"
        ) from exc

    attachments = [provenance_path]
    for path in sorted(source.iterdir()):
        if path.name.lower() in {"readme", "readme.md", "readme.txt", "readme.rst"}:
            _regular_file(path)
            attachments.append(path)
    if _has_historical_path(provenance):
        from roborsi.utils.logging import get_logger

        get_logger("components.icl").warning(
            "PolicyLab provenance contains historical absolute locations; "
            "they remain source text. External and program dependencies are unverified."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-import-", dir=destination.parent
    ) as temporary:
        material = Path(temporary) / "material"
        _copy_tree(workspace, material)
        attachment_root = material / _SOURCE_ATTACHMENTS
        attachment_root.mkdir()
        for path in attachments:
            shutil.copyfile(path, attachment_root / path.name, follow_symlinks=False)
        return build_package(
            material,
            destination,
            package_id=package_id,
            exports=exports,
            metadata={
                "icl": {"importer": "policylab", "original_provenance": provenance}
            },
        )
