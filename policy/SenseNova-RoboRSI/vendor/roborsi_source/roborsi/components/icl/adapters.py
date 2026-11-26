# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Explicit Agent projections and read-only tools for ICL references."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Literal

from .package import ContextPackage
from .workspace import IclWorkspace

if TYPE_CHECKING:
    from pydantic_ai import BinaryContent

    from roborsi.data_convert import TextDocument
    from roborsi.embodied_agent import LocalToolSpec


@dataclass(frozen=True)
class IclContext:
    """Caller-selected context for the existing EmbodiedAgent entry points."""

    memory: tuple[TextDocument, ...] = ()
    skills: tuple[Path, ...] = ()
    skill_paths: tuple[Path, ...] = ()
    initial_context: tuple[str | BinaryContent, ...] = ()


class _InputError(ValueError):
    """A caller can select a different path or read range."""


def _image_type(data: bytes, declared: str | None = None) -> str:
    # Detect the container from bytes; extensions are not authoritative.
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        actual = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        actual = "image/jpeg"
    elif len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        actual = "image/webp"
    else:
        raise _InputError("image must contain PNG, JPEG or WebP data")
    if declared is not None and actual != declared:
        raise _InputError("image media_type does not match its bytes")
    return actual


class _Reader:
    def __init__(self, resource: ContextPackage | IclWorkspace) -> None:
        self.expected: dict[str, str] | None = None
        if isinstance(resource, ContextPackage):
            verified = ContextPackage.load(resource.root)
            if verified.manifest_sha256 != resource.manifest_sha256:
                raise ValueError("ICL package manifest integrity changed")
            self.root = verified.root / "payload"
            self.expected = json.loads(
                (verified.root / "manifest.json").read_text(encoding="utf-8")
            )["files"]
        elif isinstance(resource, IclWorkspace):
            self.root = Path(os.path.abspath(resource.root))
        else:
            raise TypeError("resource must be a ContextPackage or IclWorkspace")
        self.path(".", directory=True)

    def path(self, value: str, *, directory: bool = False) -> Path:
        if not isinstance(value, str) or not value or "\\" in value:
            raise _InputError("path must be a relative POSIX path")
        if "\x00" in value:
            raise _InputError("path cannot contain NUL characters")
        relative = PurePosixPath(value)
        if (
            relative.is_absolute()
            or PureWindowsPath(value).drive
            or ".." in relative.parts
            or relative.as_posix() != value
        ):
            raise _InputError("path must stay inside the reference root")
        target = self.root.joinpath(*relative.parts)
        try:
            for item in (*reversed(target.parents), target):
                if item.is_symlink() or getattr(item, "is_junction", lambda: False)():
                    raise _InputError("reference paths cannot traverse links")
            if not target.resolve(strict=True).is_relative_to(
                self.root.resolve(strict=True)
            ):
                raise _InputError("path must stay inside the reference root")
            info = target.lstat()
            if directory:
                if not stat.S_ISDIR(info.st_mode):
                    raise _InputError("path must name a directory")
            elif not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise _InputError("path must name a regular unlinked file")
        except OSError as exc:
            raise _InputError("reference path is unavailable") from exc
        return target

    def open(self, value: str):
        target = self.path(value)
        descriptor = os.open(
            target,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        stream = os.fdopen(descriptor, "rb")
        try:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise _InputError("path must name a regular unlinked file")
            self.path(value)
            if self.expected is not None:
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                if self.expected.get(value) != digest.hexdigest():
                    raise ValueError("ICL package file integrity hash mismatch")
                stream.seek(0)
            return stream
        except BaseException:
            stream.close()
            raise

    def bytes(self, value: str, *, maximum: int | None = None) -> bytes:
        with self.open(value) as stream:
            data = stream.read() if maximum is None else stream.read(maximum + 1)
        if maximum is not None and len(data) > maximum:
            raise _InputError("file exceeds max_read_bytes")
        return data


def context_from(
    resource: ContextPackage | IclWorkspace,
    *,
    planner: str,
    skill_mode: Literal["declared", "preload", "on_demand", "none"] = "declared",
) -> IclContext:
    """Project declared exports with explicit backend skill-loading behavior.

    Args:
        resource: Verified package or caller-owned writable copy.
        planner: Existing planner name: api, codex, codex_cli or claude_code.
        skill_mode: Preserve declarations, preload, use API on-demand skills,
            or omit skills. Memory and initial context remain selected.

    Returns:
        Context values for EmbodiedAgent.run/start_episode or evaluation inputs.
    """
    from roborsi.data_convert import TextDocument
    from roborsi.runtime.skills import SkillCatalog, load_skill

    if planner not in {"api", "codex", "codex_cli", "claude_code"}:
        raise ValueError("unsupported planner")
    if skill_mode not in {"declared", "preload", "on_demand", "none"}:
        raise ValueError("unsupported skill_mode")
    reader = _Reader(resource)
    if isinstance(resource, ContextPackage):
        memories = tuple(
            (title, path.relative_to(reader.root).as_posix())
            for title, path in resource.memory
        )
        skills = tuple(
            path.relative_to(reader.root).as_posix() for path in resource.skills
        )
        directories = tuple(
            path.relative_to(reader.root).as_posix() for path in resource.skill_paths
        )
        initial = tuple(
            {
                "type": entry.type,
                "path": entry.path.relative_to(reader.root).as_posix(),
                **({"media_type": entry.media_type} if entry.media_type else {}),
            }
            for entry in resource.initial_context
        )
    else:
        memories, skills, directories, initial = (
            resource.exports.memory,
            resource.exports.skills,
            resource.exports.skill_paths,
            resource.exports.initial_context,
        )
    memory = tuple(
        TextDocument(title, reader.bytes(path).decode("utf-8"), str(reader.path(path)))
        for title, path in memories
    )
    parts = []
    for entry in initial:
        data = reader.bytes(entry["path"])
        if entry["type"] == "text":
            parts.append(data.decode("utf-8"))
        elif entry["type"] == "image":
            from pydantic_ai import BinaryContent

            parts.append(
                BinaryContent(
                    data=data, media_type=_image_type(data, entry.get("media_type"))
                )
            )
        else:
            raise ValueError("initial context type must be text or image")
    if skill_mode == "none":
        return IclContext(memory=memory, initial_context=tuple(parts))
    if skill_mode == "on_demand":
        if planner != "api":
            raise ValueError("on_demand skills require the api planner")
        if skills and not directories:
            raise ValueError("on_demand requires declared skill_paths; adjust exports")
        skills = ()
    elif planner != "api" and directories and skill_mode == "declared":
        backend = "Codex CLI" if planner == "codex_cli" else "SDK"
        raise ValueError(
            f"{backend} skill_paths require explicit skill_mode='preload' or 'none'"
        )
    skill_files = [reader.path(path) for path in dict.fromkeys(skills)]
    skill_dirs = [
        reader.path(path, directory=True) for path in dict.fromkeys(directories)
    ]
    for path in skill_files:
        reader.bytes(path.relative_to(reader.root).as_posix()).decode("utf-8")
        load_skill(path)
    if skill_mode == "preload":
        discovered = []

        def discover(directory: Path) -> None:
            reader.path(directory.relative_to(reader.root).as_posix(), directory=True)
            for item in sorted(directory.iterdir()):
                relative = item.relative_to(reader.root).as_posix()
                if item.is_dir():
                    reader.path(relative, directory=True)
                    discover(item)
                else:
                    reader.path(relative)
                    if item.name == "SKILL.md":
                        discovered.append(item.parent)

        for directory in skill_dirs:
            before = len(discovered)
            discover(directory)
            if len(discovered) == before:
                raise ValueError("declared skill collection contains no SKILL.md")
        skill_dirs = sorted(set(discovered))
    # On-demand modes retain the runtime's explicitly selected directories.
    for directory in skill_dirs:
        reader.bytes(
            (directory / "SKILL.md").relative_to(reader.root).as_posix()
        ).decode("utf-8")
    catalog = SkillCatalog(skill_dirs)
    if skill_mode == "preload":
        skill_files = list(
            dict.fromkeys(
                [
                    *skill_files,
                    *(
                        entry.directory / "SKILL.md"
                        for entry in sorted(
                            catalog.entries.values(),
                            key=lambda entry: str(entry.directory),
                        )
                    ),
                ]
            )
        )
        skill_dirs = []
    return IclContext(memory, tuple(skill_files), tuple(skill_dirs), tuple(parts))


def make_icl_tools(
    resource: ContextPackage | IclWorkspace,
    *,
    name_prefix: str = "icl",
    max_read_bytes: int = 1024 * 1024,
) -> tuple[LocalToolSpec, ...]:
    """Create bounded list/text/image readers without owning or modifying files.

    Args:
        resource: Verified package or writable reference copy.
        name_prefix: Tool name prefix; choose a distinct prefix for each package.
        max_read_bytes: Maximum returned UTF-8 or binary image bytes per read.

    Returns:
        Local tools in list, read_text, read_image order.
    """
    from jsonschema import Draft202012Validator

    from roborsi.embodied_agent import LocalToolSpec
    from roborsi.tools.toolkit import ToolResult

    if type(max_read_bytes) is not int or max_read_bytes < 1:
        raise ValueError("max_read_bytes must be a positive integer")
    if (
        not isinstance(name_prefix, str)
        or not name_prefix
        or not name_prefix.replace("_", "").isalnum()
    ):
        raise ValueError("invalid tool name prefix")
    reader = _Reader(resource)
    path_schema = {"type": "string", "minLength": 1}
    integer = {"type": "integer", "minimum": 0}
    limit_schema = {"type": "integer", "minimum": 1}

    def listing(args):
        value = args.get("path", ".")
        directory = reader.path(value, directory=True)
        entries = []
        for item in sorted(directory.iterdir(), key=lambda item: item.name):
            relative = item.relative_to(reader.root).as_posix()
            reader.path(relative, directory=item.is_dir())
            entries.append(
                {
                    "path": relative,
                    "type": "directory" if item.is_dir() else "file",
                }
            )
        offset, limit = args.get("offset", 0), min(args.get("limit", 100), 100)
        selected = []
        for entry in entries[offset : offset + limit]:
            candidate = [*selected, entry]
            if len(json.dumps(candidate).encode("utf-8")) > 48000:
                break
            selected = candidate
        end = offset + len(selected)
        return {
            "entries": selected,
            "next_offset": end if end < len(entries) else None,
            "truncated": end < len(entries),
        }

    def read_text(args):
        value, offset = args["path"], args.get("offset", 0)
        limit = min(
            args.get("limit", max_read_bytes),
            max_read_bytes,
            ToolResult.MAX_TEXT_BYTES_IN_RESULT,
        )
        try:
            with (
                reader.open(value) as raw,
                io.TextIOWrapper(raw, encoding="utf-8", newline="") as stream,
            ):
                remaining = offset
                while remaining:
                    chunk = stream.read(min(remaining, 8192))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                text = stream.read(limit)
                more = bool(stream.read(1))
        except UnicodeError as exc:
            raise _InputError("file must contain UTF-8 text") from exc
        bounded = text.encode("utf-8")[:max_read_bytes].decode("utf-8", errors="ignore")
        if text and not bounded:
            raise _InputError("max_read_bytes cannot fit the next Unicode character")

        def page(count):
            truncated = more or count < len(text)
            return {
                "path": value,
                "text": bounded[:count],
                "next_offset": offset + count if truncated else None,
                "truncated": truncated,
            }

        def fits(result):
            # Match ToolResult's central JSON encoding, including Unicode escapes.
            serialized = json.dumps(result, indent=2, default=str).encode("utf-8")
            return len(serialized) <= ToolResult.MAX_TEXT_BYTES_IN_RESULT

        result = page(len(bounded))
        if fits(result):
            return result
        low, high = 0, len(bounded)
        while low < high:
            middle = (low + high + 1) // 2
            if fits(page(middle)):
                low = middle
            else:
                high = middle - 1
        if not low:
            raise _InputError(
                "tool result byte budget cannot fit the next text character"
            )
        return page(low)

    def read_image(args):
        data = reader.bytes(args["path"], maximum=max_read_bytes)
        mime = _image_type(data)
        result = ToolResult(
            f"{name_prefix}_read_image",
            {"path": args["path"], "media_type": mime, "size_bytes": len(data)},
        )
        result.content_blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": base64.b64encode(data).decode("ascii"),
                },
            }
        )
        return result

    tools = []
    for suffix, operation, properties, required in (
        (
            "list",
            listing,
            {"path": path_schema, "offset": integer, "limit": limit_schema},
            [],
        ),
        (
            "read_text",
            read_text,
            {"path": path_schema, "offset": integer, "limit": limit_schema},
            ["path"],
        ),
        ("read_image", read_image, {"path": path_schema}, ["path"]),
    ):
        name = f"{name_prefix}_{suffix}"
        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
        validator = Draft202012Validator(schema)

        def handler(args, *, _operation=operation, _name=name, _validator=validator):
            if next(_validator.iter_errors(args), None) is not None:
                return ToolResult(
                    _name,
                    {"error": "invalid reference read arguments", "retryable": True},
                )
            # JSON Schema integers include integral floats; Python read/slice do not.
            normalized = dict(args)
            for field in ("offset", "limit"):
                if field in normalized:
                    normalized[field] = int(normalized[field])
            try:
                result = _operation(normalized)
            except _InputError as exc:
                return ToolResult(_name, {"error": str(exc), "retryable": True})
            return (
                result if isinstance(result, ToolResult) else ToolResult(_name, result)
            )

        tools.append(
            LocalToolSpec(
                name,
                f"Read ICL references: {suffix}. Does not execute programs or advance the environment.",
                schema,
                handler,
                propagate_errors=True,
            )
        )
    return tuple(tools)
