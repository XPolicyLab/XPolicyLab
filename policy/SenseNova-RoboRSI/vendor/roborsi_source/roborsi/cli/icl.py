# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Command-line adapters for verified ICL package and workspace operations."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from roborsi.components.icl.build import (
    ContextExports,
    _checked_absolute,
    _exports_document,
    _json_object,
    build_package,
)
from roborsi.components.icl.importers import import_policylab_workspace
from roborsi.components.icl.package import ContextPackage
from roborsi.components.icl.workspace import (
    IclWorkspace,
    _package_exports,
    materialize_package,
    snapshot_workspace,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roborsi-icl",
        description="Build, verify, copy, and snapshot ICL materials.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build", help="Build from a JSON spec with relative paths."
    )
    build.add_argument("--spec", required=True, type=Path)
    fetch = commands.add_parser(
        "fetch", help="Fetch a pinned local, HTTP ZIP, or HF package."
    )
    fetch.add_argument(
        "--source-json", required=True, type=Path, help="Source descriptor JSON file."
    )
    fetch.add_argument("--cache-dir", required=True, type=Path)
    fetch.add_argument("--offline", action="store_true")
    fetch.add_argument("--max-download-bytes", type=int, default=1024 * 1024 * 1024)
    fetch.add_argument("--max-unpacked-bytes", type=int, default=4 * 1024 * 1024 * 1024)
    for name in ("verify", "inspect"):
        command = commands.add_parser(
            name, help=f"{name.capitalize()} a complete sealed package."
        )
        command.add_argument("package", type=Path)
    materialize = commands.add_parser(
        "materialize", help="Create an independent writable copy."
    )
    materialize.add_argument("package", type=Path)
    materialize.add_argument("destination", type=Path)
    materialize.add_argument(
        "--workspace-record",
        type=Path,
        required=True,
        help="New JSON receipt outside the workspace.",
    )
    snapshot = commands.add_parser(
        "snapshot", help="Seal a stopped workspace into a new package."
    )
    snapshot.add_argument("--workspace-record", type=Path, required=True)
    snapshot.add_argument("destination", type=Path)
    snapshot.add_argument("--package-id", required=True)
    snapshot.add_argument(
        "--exports", type=Path, help="Optional replacement exports JSON file."
    )
    snapshot.add_argument("--metadata", type=Path, help="Optional metadata JSON file.")
    importer = commands.add_parser(
        "import-policylab", help="Import one PolicyLab task directory."
    )
    importer.add_argument("source", type=Path)
    importer.add_argument("destination", type=Path)
    importer.add_argument("--package-id", required=True)
    importer.add_argument(
        "--exports", type=Path, help="Exports JSON file relative to workspace contents."
    )
    return parser


def _read_object(path: Path) -> dict[str, Any]:
    path = _checked_absolute(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("JSON input must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON input must be an object")
    return _json_object(value, "JSON input")


def _fields(
    document: dict[str, Any], required: set[str], optional: set[str] | None = None
) -> None:
    if required - set(document) or set(document) - required - (optional or set()):
        raise ValueError(
            f"JSON fields must include {sorted(required)} with optional {sorted(optional or set())}"
        )


def _path(value: Any, base: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty path")
    path = Path(value)
    return path if path.is_absolute() else base / path


def _exports(document: Any) -> ContextExports:
    if not isinstance(document, dict):
        raise ValueError("exports must be an object")
    _fields(document, set(), {"memory", "skills", "skill_paths", "initial_context"})
    for name in document:
        if not isinstance(document[name], list):
            raise ValueError(f"exports.{name} must be a list")
    memories = []
    for entry in document.get("memory", []):
        if not isinstance(entry, dict):
            raise ValueError("memory entries must be objects")
        _fields(entry, {"title", "path"})
        memories.append((entry["title"], entry["path"]))
    return ContextExports(
        memory=tuple(memories),
        skills=tuple(document.get("skills", [])),
        skill_paths=tuple(document.get("skill_paths", [])),
        initial_context=tuple(document.get("initial_context", [])),
    )


def _identity(package: ContextPackage) -> dict[str, Any]:
    return {
        "root": str(package.root),
        "package_id": package.package_id,
        "manifest_sha256": package.manifest_sha256,
    }


def _record_path(path: Path, workspace_root: Path, *, new: bool = False) -> Path:
    selected = _checked_absolute(path)
    root = _checked_absolute(workspace_root)
    if selected == root or root in selected.parents:
        raise ValueError("workspace record must be outside the workspace")
    if new and os.path.lexists(selected):
        raise ValueError("workspace record must not already exist")
    return selected


def _write_record(path: Path, workspace: IclWorkspace) -> None:
    try:
        root = os.path.relpath(workspace.root, path.parent)
    except ValueError:
        root = str(workspace.root)
    document = {
        "schema_version": 1,
        "root": root,
        "source_package_id": workspace.source_package_id,
        "source_manifest_sha256": workspace.source_manifest_sha256,
        "exports": _exports_document(workspace.exports),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(document, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n"
        )


def _read_workspace(path: Path) -> IclWorkspace:
    path = _checked_absolute(path)
    document = _read_object(path)
    _fields(
        document,
        {
            "schema_version",
            "root",
            "source_package_id",
            "source_manifest_sha256",
            "exports",
        },
    )
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("workspace record schema_version must be 1")
    root = _checked_absolute(_path(document["root"], path.parent, "workspace root"))
    _record_path(path, root)
    if not root.is_dir():
        raise ValueError("workspace root must be a real directory")
    return IclWorkspace(
        root=root,
        source_package_id=document["source_package_id"],
        source_manifest_sha256=document["source_manifest_sha256"],
        exports=_exports(document["exports"]),
    )


def _fetch(args: argparse.Namespace) -> ContextPackage:
    from roborsi.components.icl.sources import (
        HfPackageSource,
        HttpArchiveSource,
        LocalPackageSource,
        fetch_package,
    )

    path = _checked_absolute(args.source_json)
    document = _read_object(path)
    kind = document.get("type")
    if kind == "local":
        _fields(document, {"type", "path"})
        source = LocalPackageSource(
            _path(document["path"], path.parent, "local source")
        )
    elif kind == "http":
        _fields(document, {"type", "url", "archive_sha256"}, {"manifest_sha256"})
        source = HttpArchiveSource(
            **{key: value for key, value in document.items() if key != "type"}
        )
    elif kind == "hf":
        _fields(
            document,
            {"type", "repo_id", "revision", "subpath"},
            {"repo_type", "manifest_sha256"},
        )
        source = HfPackageSource(
            **{key: value for key, value in document.items() if key != "type"}
        )
    else:
        raise ValueError("source type must be local, http, or hf")
    return fetch_package(
        source,
        args.cache_dir,
        offline=args.offline,
        max_download_bytes=args.max_download_bytes,
        max_unpacked_bytes=args.max_unpacked_bytes,
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "build":
        spec_path = _checked_absolute(args.spec)
        document = _read_object(spec_path)
        _fields(
            document, {"source", "destination", "package_id"}, {"exports", "metadata"}
        )
        return _identity(
            build_package(
                _path(document["source"], spec_path.parent, "source"),
                _path(document["destination"], spec_path.parent, "destination"),
                package_id=document["package_id"],
                exports=_exports(document.get("exports", {})),
                metadata=document.get("metadata"),
            )
        )
    if args.command == "fetch":
        return _identity(_fetch(args))
    if args.command in {"verify", "inspect"}:
        package = ContextPackage.load(args.package)
        result = _identity(package)
        if args.command == "inspect":
            document = _read_object(package.root / "manifest.json")
            result.update(
                file_count=len(document["files"]),
                total_bytes=sum(
                    (package.root / "payload" / name).stat().st_size
                    for name in document["files"]
                ),
                exports=_exports_document(_package_exports(package)),
                metadata=package.metadata,
            )
        return result
    if args.command == "materialize":
        package = ContextPackage.load(args.package)
        record = _record_path(args.workspace_record, args.destination, new=True)
        if package.root in record.parents:
            raise ValueError("workspace record must be outside the source package")
        workspace = materialize_package(package, args.destination)
        try:
            _write_record(record, workspace)
        except (OSError, ValueError) as exc:
            raise OSError(
                f"workspace was created at {workspace.root}, but its record could not be written: {exc}"
            ) from exc
        return {
            "root": str(workspace.root),
            "source_package_id": workspace.source_package_id,
            "source_manifest_sha256": workspace.source_manifest_sha256,
            "workspace_record": str(record),
        }
    exports = _exports(_read_object(args.exports)) if args.exports else None
    if args.command == "snapshot":
        return _identity(
            snapshot_workspace(
                _read_workspace(args.workspace_record),
                args.destination,
                package_id=args.package_id,
                exports=exports,
                metadata=_read_object(args.metadata) if args.metadata else None,
            )
        )
    return _identity(
        import_policylab_workspace(
            args.source,
            args.destination,
            package_id=args.package_id,
            exports=exports,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one ICL file operation and emit JSON on success.

    Args:
        argv: Command arguments, or process arguments when omitted.

    Returns:
        Zero on success, a nonzero argument or operation error code on failure.
    """
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    try:
        result = _run(args)
        sys.stdout.write(
            json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n"
        )
    except (
        OSError,
        ValueError,
        TypeError,
        UnicodeError,
        RuntimeError,
        ImportError,
    ) as exc:
        from roborsi.utils.logging import get_logger

        get_logger("cli.icl").error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
