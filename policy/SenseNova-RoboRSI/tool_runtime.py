"""Verify the independent, publicly rebuilt B-compatible model-tool runtime."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from email.parser import BytesParser
from pathlib import Path, PurePosixPath


class ToolRuntimeError(ValueError):
    """The fixed model-tool build or its installed payload changed."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(value: str) -> Path:
    if not isinstance(value, str):
        raise ToolRuntimeError("Tool metadata requires relative public paths")
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or any(part in {"..", "."} for part in path.parts)
        or "\\" in value
        or any(character in value for character in "\0\r\n")
        or path.as_posix() != value
    ):
        raise ToolRuntimeError("Tool metadata contains an unsafe relative path")
    return Path(path)


def _packages(rows) -> dict[str, str]:
    result = {}
    if not isinstance(rows, list):
        raise ToolRuntimeError("Tool distribution declaration must be a list")
    for row in rows:
        if isinstance(row, dict):
            name, version = row.get("name"), row.get("version")
        elif isinstance(row, (list, tuple)) and len(row) == 2:
            name, version = row
        else:
            raise ToolRuntimeError("Tool distribution identity is malformed")
        if not isinstance(name, str) or not name or not isinstance(version, str):
            raise ToolRuntimeError("Tool distribution identity is malformed")
        name = re.sub(r"[-_.]+", "-", name).lower()
        if name in result:
            raise ToolRuntimeError("Duplicate tool distribution metadata")
        result[name] = version
    return result


def _metadata_sha(policy_dir: Path) -> str:
    return json.loads((Path(policy_dir) / "submission.json").read_text())[
        "tool_runtime"
    ]["manifest_sha256"]


def load_tool_metadata(policy_dir: Path) -> dict:
    """Authenticate every build input before downloading or executing it."""
    policy = Path(policy_dir)
    try:
        declaration = json.loads((policy / "submission.json").read_text())[
            "tool_runtime"
        ]
        path = policy / _relative(declaration["manifest_path"])
        if path.is_symlink() or _sha(path) != declaration["manifest_sha256"]:
            raise ValueError
        metadata = json.loads(path.read_text())
        if (
            metadata.get("schema_version") != 2
            or metadata.get("python_version") != "3.11.13"
        ):
            raise ValueError
        if len(_packages(metadata["distributions"])) != 82:
            raise ToolRuntimeError(
                "The model-tool runtime requires its 82 fixed distribution versions"
            )
        for relative, digest in metadata["resources"].items():
            resource = policy / _relative(relative)
            if resource.is_symlink() or _sha(resource) != digest:
                raise ToolRuntimeError("A pinned model-tool build resource changed")
        for name in ("runtime", "base", "receipt", "inventory"):
            _relative(metadata["paths"][name])
        return metadata
    except ToolRuntimeError:
        raise
    except (OSError, KeyError, TypeError, ValueError, AttributeError):
        raise ToolRuntimeError(
            "Pinned model-tool build inputs are missing or invalid"
        ) from None


def _root(policy_dir: Path, relative: str) -> Path:
    path = Path(policy_dir) / _relative(relative)
    if path.is_symlink():
        raise ToolRuntimeError("A model-tool payload root must be a real directory")
    try:
        return path.parent.resolve(strict=True) / path.name
    except OSError:
        raise ToolRuntimeError(
            "The model-tool runtime is missing; run bash install.sh"
        ) from None


def _inventory(roots: dict[str, Path]) -> list[dict]:
    """Record actual installed files; bytecode is generated and is never trusted."""
    rows = []
    for root, source in roots.items():
        for directory, names, files in os.walk(source, followlinks=False):
            names[:] = sorted(name for name in names if name != "__pycache__")
            links = [name for name in names if (Path(directory) / name).is_symlink()]
            for name in sorted(files + links):
                path = Path(directory) / name
                if path.suffix in {".pyc", ".pyo"}:
                    continue
                row = {"root": root, "path": path.relative_to(source).as_posix()}
                if path.is_symlink():
                    row["symlink"] = os.readlink(path)
                elif path.is_file():
                    row.update(bytes=path.stat().st_size, sha256=_sha(path))
                else:
                    raise ToolRuntimeError("Unsupported model-tool payload file")
                rows.append(row)
    return rows


def _critical(root: str, path: str) -> bool:
    return (
        root == "base"
        and path in {"bin/python3.11", "lib/libpython3.11.so.1.0"}
        or path == "pyvenv.cfg"
        or path.endswith(("/METADATA", "/RECORD", ".pth"))
        or "/site-packages/openpi_client/" in path
    )


def validate_tool_install(policy_dir: Path, *, full: bool = False) -> dict:
    """Verify the fixed interpreter, all 82 versions and original OpenPI sources."""
    policy = Path(policy_dir)
    metadata = load_tool_metadata(policy)
    roots = {
        name: _root(policy, metadata["paths"][name]) for name in ("runtime", "base")
    }
    deployment = roots["runtime"].parent
    if (
        roots["base"].parent != deployment
        or deployment.stat().st_uid != os.getuid()
        or deployment.stat().st_mode & 0o077
    ):
        raise ToolRuntimeError(
            "The model-tool deployment must be private to its installing user"
        )
    configured = os.environ.get("ROBORSI_TOOL_ROOT")
    if configured and Path(configured).expanduser().resolve() != deployment:
        raise ToolRuntimeError(
            "ROBORSI_TOOL_ROOT differs from the installed tool deployment"
        )
    try:
        receipt_path = policy / metadata["paths"]["receipt"]
        inventory_path = policy / metadata["paths"]["inventory"]
        if receipt_path.is_symlink() or inventory_path.is_symlink():
            raise ValueError
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt.get("schema_version") != 2
            or receipt.get("metadata_sha256") != _metadata_sha(policy)
            or receipt.get("python_version") != "3.11.13"
            or receipt.get("installation_mode") != "public-source-build"
            or receipt.get("inventory_sha256") != _sha(inventory_path)
        ):
            raise ValueError
        rows = json.loads(inventory_path.read_text())
    except (OSError, ValueError, TypeError):
        raise ToolRuntimeError(
            "Model-tool build receipt is missing or invalid; run bash install.sh"
        ) from None
    expected = set()
    hashed = 0
    for row in rows:
        root, relative = row["root"], _relative(row["path"])
        if root not in roots or (root, relative.as_posix()) in expected:
            raise ToolRuntimeError("Invalid installed model-tool inventory")
        expected.add((root, relative.as_posix()))
        path = roots[root] / relative
        if "symlink" in row:
            if (
                not path.is_symlink()
                or os.readlink(path) != row["symlink"]
                or not path.exists()
                or not any(
                    path.resolve().is_relative_to(item) for item in roots.values()
                )
            ):
                raise ToolRuntimeError("A model-tool link changed or escapes its roots")
        else:
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != row["bytes"]
                or path.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            ):
                raise ToolRuntimeError(
                    "A model-tool payload file is missing, changed or writable"
                )
            if full or _critical(root, relative.as_posix()):
                if _sha(path) != row["sha256"]:
                    raise ToolRuntimeError("A model-tool payload SHA-256 differs")
                hashed += 1
    actual = set()
    for root, source in roots.items():
        for directory, names, files in os.walk(source, followlinks=False):
            names[:] = [name for name in names if name != "__pycache__"]
            links = [name for name in names if (Path(directory) / name).is_symlink()]
            for name in files + links:
                path = Path(directory) / name
                if path.suffix not in {".pyc", ".pyo"}:
                    actual.add((root, path.relative_to(source).as_posix()))
    if actual != expected:
        raise ToolRuntimeError(
            "The model-tool inventory has missing or additional files"
        )
    site = roots["runtime"] / "lib/python3.11/site-packages"
    distributions = []
    for path in site.glob("*.dist-info/METADATA"):
        headers = BytesParser().parsebytes(path.read_bytes(), headersonly=True)
        distributions.append(
            {"name": headers.get("Name"), "version": headers.get("Version")}
        )
    if _packages(distributions) != _packages(metadata["distributions"]):
        raise ToolRuntimeError(
            "Model tools do not contain exactly the 82 fixed distribution versions"
        )
    python = roots["runtime"] / "bin/python"
    if (
        python.resolve(strict=True) != roots["base"] / "bin/python3.11"
        or _sha(python) != metadata["python"]["binary_sha256"]
    ):
        raise ToolRuntimeError("The tool interpreter differs from fixed Python 3.11.13")
    pins = json.loads((policy / "tools/openpi-client-files.json").read_text())
    for name, digest in pins.items():
        path = site / "openpi_client" / _relative(name)
        if path.is_symlink() or _sha(path) != digest:
            raise ToolRuntimeError("An installed original OpenPI client source changed")
    return {
        **receipt,
        "distribution_count": len(distributions),
        "inventory_entries_checked": len(rows),
        "payload_files_hashed": hashed,
        "full_payload_verified": full,
        "source_payloads_verified": True,
        "agent_python": str(python),
    }


def validate_tool_payload(policy_dir: Path, *, full: bool = True) -> dict:
    return validate_tool_install(policy_dir, full=full)


def resolve_tool_python(policy_dir: Path) -> Path:
    """Return the venv spelling; resolving its leaf would lose /runtime."""
    validate_tool_install(policy_dir)
    metadata = load_tool_metadata(policy_dir)
    return _root(policy_dir, metadata["paths"]["runtime"]) / "bin/python"
