# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verify published original execution pins without inventing a reference.

Receipt schema 1 contains source_ref, cli, bubblewrap, code_mode_host,
codex_resources, agent_environment and controller_environment. The separate
controller proof contains baseline_interpreter_sha256 and
framework_dependency_lock(path/sha256). Binary entries contain sha256. Tree
Optional path_relocations explicitly maps original absolute roots to deployed
roots for symbolic-link targets only. No file contents are rewritten. Tree
entries contain complete files rows and optional exclude_cache limited to
__pycache__ and *.pyc. File rows contain path/bytes/sha256; symbolic-link rows
contain path/symlink. agent_environment includes interpreter_sha256,
dependency_lock(path/sha256), runtime_tree and base_tree.
Relative model lock paths stay within the original agent runtime. An explicit
absolute lock may identify a private host provenance file outside that runtime;
it is verified separately and never added to the original tree or model mounts.

This proves the declared CLI/resource/interpreter/dependency scope. System
libraries, container/OS identity, native GPU stack and services still require
separate deployment attestation and real acceptance.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

from roborsi.planner.codex_cli_config import CodexCliConfig

from .preparation import DeploymentPaths

SHA = re.compile(r"[0-9a-f]{64}\Z")
CACHE = frozenset({"__pycache__", "*.pyc"})
SECRET_NAMES = frozenset(
    {
        "auth.json",
        "api_keys.yaml",
        "api_keys.yml",
        ".env",
        "credentials.json",
        "tokens.json",
        "token.json",
        "secrets.json",
    }
)


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _pin(value: Any) -> str:
    if not isinstance(value, str) or SHA.fullmatch(value) is None:
        raise ValueError("Execution identity requires published SHA256 pins")
    return value


def _binary(path: Path, entry: Any) -> str:
    if not isinstance(entry, dict) or set(entry) != {"sha256"}:
        raise ValueError("Binary identity requires only a published SHA256 pin")
    expected = _pin(entry["sha256"])
    if not path.is_file() or _sha(path) != expected:
        raise ValueError("Execution binary differs from its published original pin")
    return expected


def _relative(value: Any, root: Path) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Invalid execution inventory path")
    path = Path(value)
    if ".." in path.parts:
        raise ValueError("Execution inventory path traverses its root")
    if path.is_absolute():
        if not path.is_relative_to(root):
            raise ValueError("Execution inventory path is outside its root")
        path = path.relative_to(root)
    if path == Path("."):
        raise ValueError("Execution inventory must identify a file")
    if any(part.lower() in SECRET_NAMES for part in path.parts):
        raise ValueError("Authentication files cannot be execution identity inputs")
    return path


def _dependency_lock_path(value: Any, runtime: Path) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Invalid dependency lock path")
    path = Path(value)
    if path.is_absolute():
        if path.as_posix() != value or ".." in path.parts:
            raise ValueError("Dependency lock requires a canonical absolute path")
        if any(part.lower() in SECRET_NAMES for part in path.parts):
            raise ValueError("Authentication files cannot be dependency lock inputs")
        return path
    return runtime / _relative(value, runtime)


def _dependency_lock_sha(path: Path, expected: Any) -> str:
    expected = _pin(expected)
    if not path.is_absolute():
        raise ValueError("Dependency lock must have an absolute deployment path")
    if any(part.lower() in SECRET_NAMES for part in path.parts):
        raise ValueError("Authentication files cannot be dependency lock inputs")
    # Walk directory descriptors so a replaced ancestor cannot redirect the
    # read between path validation and opening the final provenance file.
    directory_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    reader_fd = None
    try:
        for part in path.parts[1:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        reader_fd = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
        before = os.fstat(reader_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Dependency lock must be a regular file")
        digest = hashlib.sha256()
        while block := os.read(reader_fd, 1024 * 1024):
            digest.update(block)
        after = os.fstat(reader_fd)
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("Dependency lock changed during verification")
        if digest.hexdigest() != expected:
            raise ValueError("Dependency lock differs from its published original pin")
    except OSError as exc:
        raise ValueError(
            "Dependency lock must not have linked or invalid ancestors"
        ) from exc
    finally:
        if reader_fd is not None:
            os.close(reader_fd)
        os.close(directory_fd)
    return expected


def _excluded(path: Path, patterns: list[str]) -> bool:
    return any(
        any(fnmatch.fnmatch(part, pattern) for part in path.parts)
        for pattern in patterns
    )


def _relocations(value: Any, actual_roots: tuple[Path, ...]) -> list[tuple[Path, Path]]:
    if not isinstance(value, list):
        raise ValueError("Published path_relocations must be a list")
    pairs = []
    for row in value:
        if not isinstance(row, dict) or set(row) != {"original", "deployment"}:
            raise ValueError("Invalid published path relocation")
        original, deployed = row["original"], row["deployment"]
        if any(
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or ".." in Path(path).parts
            or "\x00" in path
            for path in (original, deployed)
        ):
            raise ValueError("Path relocations require explicit absolute roots")
        source = Path(os.path.normpath(original))
        target = Path(os.path.normpath(deployed))
        if source == Path("/") or target not in actual_roots:
            raise ValueError(
                "Path relocation must target an authenticated deployment root"
            )
        if any(
            source.is_relative_to(a)
            or a.is_relative_to(source)
            or target.is_relative_to(b)
            or b.is_relative_to(target)
            for a, b in pairs
        ):
            raise ValueError("Ambiguous overlapping published path relocations")
        pairs.append((source, target))
    return pairs


def _mapped_link(value: str, mappings: list[tuple[Path, Path]]) -> str:
    path = Path(value)
    if path.is_absolute():
        for original, deployed in mappings:
            if path.is_relative_to(original):
                return (deployed / path.relative_to(original)).as_posix()
    return value


def _tree(
    root: Path,
    entry: Any,
    *,
    allowed_link_roots: tuple[Path, ...],
    mappings: list[tuple[Path, Path]] | None = None,
) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Execution tree must be an existing materialized directory")
    if (
        not isinstance(entry, dict)
        or set(entry) - {"files", "exclude_cache"}
        or "files" not in entry
    ):
        raise ValueError("Execution tree requires a complete published file inventory")
    excluded = entry.get("exclude_cache", [])
    if not isinstance(excluded, list) or any(
        pattern not in CACHE for pattern in excluded
    ):
        raise ValueError(
            "Only explicitly declared Python cache exclusions are permitted"
        )
    if not isinstance(entry["files"], list) or not entry["files"]:
        raise ValueError("Execution tree inventory is empty")
    mappings = mappings or []
    expected: dict[str, dict[str, Any]] = {}
    for row in entry["files"]:
        if not isinstance(row, dict) or "path" not in row:
            raise ValueError("Malformed published execution inventory")
        path = _relative(
            _mapped_link(row["path"], mappings)
            if isinstance(row["path"], str)
            else row["path"],
            root,
        )
        name = path.as_posix()
        if name in expected or _excluded(path, excluded):
            raise ValueError("Duplicate or excluded published execution path")
        expected[name] = row
    actual = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if _excluded(relative, excluded):
            continue
        if path.is_symlink() or path.is_file():
            actual.add(relative.as_posix())
        elif not path.is_dir():
            raise ValueError("Execution tree contains a special filesystem node")
    if actual != set(expected):
        raise ValueError("Execution tree inventory differs from its published original")
    for name, row in expected.items():
        path = root / name
        if "symlink" in row:
            if set(row) != {"path", "symlink"} or not isinstance(row["symlink"], str):
                raise ValueError("Invalid published symbolic-link identity")
            if not path.is_symlink() or path.readlink().as_posix() != _mapped_link(
                row["symlink"], mappings
            ):
                raise ValueError("Execution symbolic link differs from original")
            target = path.resolve(strict=True)
            if not any(
                target.is_relative_to(allowed) for allowed in allowed_link_roots
            ):
                raise ValueError("Execution symbolic link escapes authenticated roots")
        else:
            if (
                set(row) != {"path", "bytes", "sha256"}
                or type(row["bytes"]) is not int
                or row["bytes"] < 0
            ):
                raise ValueError("Invalid published file identity")
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != row["bytes"]
                or _sha(path) != _pin(row["sha256"])
            ):
                raise ValueError("Execution tree content differs from original")
    return {
        "files": len(expected),
        "excluded_cache": excluded,
        "scope": "complete declared tree",
    }


def verify_execution_identity(
    cli_config: CodexCliConfig,
    deployment: DeploymentPaths,
    bubblewrap: Path,
    receipt: Path,
    *,
    controller_python: Path | None = None,
) -> dict[str, Any]:
    """Verify original CLI, launch resources and agent environment before start.

    Args:
        cli_config: Explicit CLI version and original executable SHA pin.
        deployment: Actual deployment paths, preserving the virtualenv symlink.
        bubblewrap: Actual original sandbox executable.
        receipt: Published original-reference pins; never generated here from
            current installation bytes.
        controller_python: Actual original controller interpreter, distinct
            from the model-facing agent virtualenv interpreter.

    Returns:
        Audited nonsecret scopes and reference identity, suitable for receipts.

    Raises:
        ValueError: Missing pins, different files, incomplete inventories or
            links outside authenticated roots.
    """
    if receipt.is_symlink() or not receipt.is_file():
        raise ValueError("Missing published original execution identity receipt")
    receipt_before = receipt.stat()
    receipt_bytes = receipt.read_bytes()
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    pins = json.loads(receipt_bytes.decode("utf-8"))
    required = {
        "schema_version",
        "source_ref",
        "cli",
        "bubblewrap",
        "code_mode_host",
        "codex_resources",
        "agent_environment",
        "controller_environment",
    }
    if (
        not isinstance(pins, dict)
        or (set(pins) - {"path_relocations"}) != required
        or type(pins["schema_version"]) is not int
        or pins["schema_version"] != 1
    ):
        raise ValueError("Invalid original execution identity receipt schema")
    if not isinstance(pins["source_ref"], str) or not pins["source_ref"].strip():
        raise ValueError("Original execution source_ref must be explicit")
    binary = deployment.codex_binary.resolve(strict=True)
    if Path(cli_config.executable).resolve(strict=True) != binary:
        raise ValueError("CLI identity and actual launcher payload differ")
    cli_sha = _binary(binary, pins["cli"])
    if cli_config.expected_sha256 != cli_sha:
        raise ValueError("CLI preset pin differs from published original identity")
    bwrap_sha = _binary(bubblewrap.resolve(strict=True), pins["bubblewrap"])
    host_sha = _binary(binary.with_name("codex-code-mode-host"), pins["code_mode_host"])
    resources = binary.parent.parent / "codex-resources"
    python = deployment.agent_python.absolute()
    interpreter = python.resolve(strict=True)
    runtime = python.parent.parent
    base = interpreter.parent.parent
    mappings = _relocations(
        pins.get("path_relocations", []), (resources, runtime, base)
    )
    resource_identity = _tree(
        resources,
        pins["codex_resources"],
        allowed_link_roots=(resources,),
        mappings=mappings,
    )
    environment = pins["agent_environment"]
    if not isinstance(environment, dict) or set(environment) != {
        "interpreter_sha256",
        "dependency_lock",
        "runtime_tree",
        "base_tree",
    }:
        raise ValueError(
            "Agent identity requires interpreter, lock and both complete trees"
        )
    interpreter_sha = _binary(
        interpreter, {"sha256": environment["interpreter_sha256"]}
    )
    lock = environment["dependency_lock"]
    if not isinstance(lock, dict) or set(lock) != {"path", "sha256"}:
        raise ValueError("Missing original dependency lock identity")
    _dependency_lock_path(lock["path"], runtime)
    lock_path = _dependency_lock_path(
        _mapped_link(lock["path"], mappings)
        if isinstance(lock["path"], str)
        else lock["path"],
        runtime,
    )
    lock_sha = _dependency_lock_sha(lock_path, lock["sha256"])
    runtime_identity = _tree(
        runtime,
        environment["runtime_tree"],
        allowed_link_roots=(runtime, base),
        mappings=mappings,
    )
    base_identity = _tree(
        base,
        environment["base_tree"],
        allowed_link_roots=(runtime, base),
        mappings=mappings,
    )
    controller = pins["controller_environment"]
    if (
        controller_python is None
        or not isinstance(controller, dict)
        or set(controller)
        != {"baseline_interpreter_sha256", "framework_dependency_lock"}
    ):
        raise ValueError(
            "Separate original controller interpreter and backend dependency proof are required"
        )
    controller_sha = _binary(
        controller_python.resolve(strict=True),
        {"sha256": controller["baseline_interpreter_sha256"]},
    )
    framework_lock = controller["framework_dependency_lock"]
    if (
        not isinstance(framework_lock, dict)
        or set(framework_lock) != {"path", "sha256"}
        or not isinstance(framework_lock["path"], str)
    ):
        raise ValueError("Missing explicit backend framework dependency lock")
    framework_path = Path(_mapped_link(framework_lock["path"], mappings))
    if not framework_path.is_absolute():
        raise ValueError("Backend dependency lock must have an explicit absolute path")
    _relative(str(framework_path), framework_path.parent)
    if framework_path.is_symlink():
        raise ValueError("Backend dependency lock must not be linked")
    framework_sha = _binary(framework_path, {"sha256": framework_lock["sha256"]})
    receipt_after = receipt.stat()
    if receipt.is_symlink() or (
        receipt_before.st_dev,
        receipt_before.st_ino,
        receipt_before.st_size,
        receipt_before.st_mtime_ns,
        receipt_before.st_ctime_ns,
    ) != (
        receipt_after.st_dev,
        receipt_after.st_ino,
        receipt_after.st_size,
        receipt_after.st_mtime_ns,
        receipt_after.st_ctime_ns,
    ):
        raise ValueError("Published identity receipt changed during verification")
    return {
        "schema_version": 1,
        "source_ref": pins["source_ref"],
        "receipt_sha256": receipt_sha,
        "cli_sha256": cli_sha,
        "bubblewrap_sha256": bwrap_sha,
        "code_mode_host_sha256": host_sha,
        "codex_resources": resource_identity,
        "interpreter_sha256": interpreter_sha,
        "dependency_lock_sha256": lock_sha,
        "agent_runtime": runtime_identity,
        "agent_base": base_identity,
        "agent_python_path_preserved": str(python),
        "path_relocations": [
            {"original": str(a), "deployment": str(b)} for a, b in mappings
        ],
        "controller_interpreter_sha256": controller_sha,
        "backend_framework_dependency_lock_sha256": framework_sha,
        "controller_dependency_scope": "original interpreter identity plus separately declared new backend lock; not the model agent virtualenv",
        "remaining_scope": ".pth/editable external imports, controller stdlib and installed backend dependencies, dynamic system libraries, container/OS/GPU/native-service attestation and real acceptance",
    }
