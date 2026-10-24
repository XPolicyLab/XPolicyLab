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

"""Benchmark discovery and immutable manifest locks without native imports."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator, SchemaError, ValidationError

from roborsi.evaluation.messages import Task
from roborsi.evaluation.protocol import RunProvenance, TaskIdentity

_BENCHMARK_ID = re.compile(r"[a-z][a-z0-9_-]*\Z")


def benchmark_manifest_schema() -> dict[str, Any]:
    """Return a fresh copy of the RoboRSI benchmark manifest schema."""
    resource = files("roborsi.schemas").joinpath("roborsi_benchmark_manifest.v1.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {value}")


def _file_sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"asset is not a regular file: {path}")
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def directory_sha256(directory: str | Path) -> str:
    """Hash a directory tree, including relative names and empty directories.

    Symlinks and special files are rejected rather than followed. File content
    is streamed, so large trees do not need an in-memory archive.
    """
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"asset directory is unavailable: {root}")
    checksum = hashlib.sha256(b"RoboRSI-directory-v1\0")

    def visit(parent: Path, parts: tuple[str, ...]) -> None:
        with os.scandir(parent) as entries:
            children = sorted(entries, key=lambda item: item.name)
        for entry in children:
            path = Path(entry.path)
            if entry.is_symlink():
                raise ValueError(f"asset directory contains a symlink: {path}")
            relative = "/".join((*parts, entry.name)).encode("utf-8", "surrogateescape")
            if entry.is_dir(follow_symlinks=False):
                kind = b"D"
            elif entry.is_file(follow_symlinks=False):
                kind = b"F"
            else:
                raise ValueError(f"asset directory contains a special file: {path}")
            checksum.update(kind)
            checksum.update(len(relative).to_bytes(8, "big"))
            checksum.update(relative)
            if kind == b"D":
                visit(path, (*parts, entry.name))
            else:
                checksum.update(bytes.fromhex(_file_sha256(path)))

    visit(root, ())
    return checksum.hexdigest()


def snapshot_directory(source: str | Path, snapshot_root: str | Path) -> Path:
    """Copy one verified content-addressed tree for reuse by many cases.

    The returned directory is made read-only to avoid accidental changes.
    Registry locks still rehash it before each container startup.
    """
    if Path(source).is_symlink():
        raise ValueError("asset snapshot source cannot be a symlink")
    original = Path(source).resolve()
    snapshots = Path(snapshot_root).resolve()
    if snapshots == original or snapshots.is_relative_to(original):
        raise ValueError("snapshot root cannot be inside the source directory")
    digest = directory_sha256(original)
    snapshots.mkdir(parents=True, exist_ok=True)
    destination = snapshots / digest
    if destination.exists():
        if directory_sha256(destination) != digest:
            raise ValueError(f"existing asset snapshot changed: {destination}")
        return destination
    temporary = snapshots / f".snapshot-{uuid.uuid4().hex}"
    try:
        shutil.copytree(original, temporary, symlinks=True)
        if directory_sha256(temporary) != digest:
            raise ValueError("asset directory changed while creating snapshot")
        for path in sorted(temporary.rglob("*"), reverse=True):
            if path.is_symlink():
                raise ValueError("asset directory changed while sealing snapshot")
            path.chmod(path.stat().st_mode & ~0o222)
        temporary.chmod(temporary.stat().st_mode & ~0o222)
        try:
            temporary.rename(destination)
        except OSError:
            if not destination.exists():
                raise
            if directory_sha256(destination) != digest:
                raise ValueError(f"existing asset snapshot changed: {destination}")
        if directory_sha256(destination) != digest:
            raise ValueError("asset snapshot changed before publication")
        return destination
    finally:
        if temporary.exists():
            for parent, _, _ in os.walk(temporary):
                path = Path(parent)
                path.chmod(path.stat().st_mode | 0o700)
            shutil.rmtree(temporary)


@dataclass(frozen=True, slots=True)
class RegisteredTask:
    """One task/seed resolved from a locked benchmark manifest."""

    identity: TaskIdentity
    instruction: str
    max_steps: int
    native: dict[str, Any]
    capabilities: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BenchmarkManifestLock:
    """A pinned task manifest; asset content must be checked by its executor."""

    manifest_path: Path
    manifest_sha256: str
    document_json: str

    def as_dict(self) -> dict[str, Any]:
        """Return a detached manifest document."""
        return json.loads(self.document_json)

    @property
    def benchmark(self) -> str:
        return self.as_dict()["benchmark"]

    def provenance(self, agent_version: str | None = None) -> RunProvenance:
        """Build the version fields stored alongside a result."""
        document = self.as_dict()
        return RunProvenance(
            benchmark_version=document["benchmark_version"],
            benchmark_commit=document["benchmark_commit"],
            docker_image_digest=document["image_digest"],
            assets_version=document["assets_version"],
            adapter_version=document["adapter_version"],
            agent_version=agent_version,
        )

    @property
    def image_reference(self) -> str:
        """Return the pullable image reference pinned to its digest."""
        document = self.as_dict()
        return f"{document['image_repository']}@{document['image_digest']}"

    def list_tasks(self) -> tuple[TaskIdentity, ...]:
        """List all declared native task/seed cells."""
        document = self.as_dict()
        return tuple(
            TaskIdentity(document["benchmark"], task["task_id"], seed)
            for task in document["tasks"]
            for seed in task["seeds"]
        )

    def task(self, task_id: str, seed: int) -> RegisteredTask:
        """Resolve a declared task/seed or reject an unknown case."""
        identity = TaskIdentity(self.benchmark, task_id, seed)
        document = self.as_dict()
        for task in document["tasks"]:
            if task["task_id"] == task_id and seed in task["seeds"]:
                return RegisteredTask(
                    identity=identity,
                    instruction=task["instructions_by_seed"][str(seed)]
                    if "instructions_by_seed" in task
                    else task["instruction"],
                    max_steps=task["max_steps"],
                    native=task["native_by_seed"][str(seed)]
                    if "native_by_seed" in task
                    else task["native"],
                    capabilities=document["capabilities"],
                )
        raise KeyError(f"undeclared benchmark task/seed: {task_id}/{seed}")

    def task_message(self, task_id: str, seed: int) -> Task:
        """Build the public Agent/Adapter task without native imports."""
        registered = self.task(task_id, seed)
        return Task(
            benchmark=registered.identity.benchmark,
            task_id=registered.identity.task_id,
            seed=seed,
            instruction=registered.instruction,
            max_steps=registered.max_steps,
            native=registered.native,
            capabilities=registered.capabilities,
        )

    def verify(self) -> None:
        """Reject a changed manifest without accessing remote asset paths."""
        if _file_sha256(self.manifest_path) != self.manifest_sha256:
            raise ValueError(f"benchmark manifest changed: {self.manifest_path}")


@dataclass(frozen=True, slots=True)
class BenchmarkLock(BenchmarkManifestLock):
    """A content-pinned manifest with verified files and directory snapshots."""

    def verify(self) -> None:
        """Reject a changed manifest or unavailable/changed local asset content."""
        super(BenchmarkLock, self).verify()
        document = self.as_dict()
        root = self.manifest_path.parent.resolve()
        for asset in document["assets"]:
            asset_path = asset["path"]
            relative = PurePosixPath(asset_path)
            if (
                relative.is_absolute()
                or "\\" in asset_path
                or any(part in {"", ".", ".."} for part in asset_path.split("/"))
            ):
                raise ValueError(
                    f"asset path escapes benchmark directory: {asset_path}"
                )
            source = root
            for part in relative.parts:
                source /= part
                if source.is_symlink():
                    raise ValueError(f"asset path contains a symlink: {asset_path}")
            path = source.resolve()
            if not path.is_relative_to(root):
                raise ValueError(
                    f"asset path escapes benchmark directory: {asset['path']}"
                )
            kind = asset.get("kind", "file")
            if kind == "directory":
                if not path.is_dir() or path.is_symlink():
                    raise FileNotFoundError(f"benchmark asset is missing: {path}")
                actual = directory_sha256(path)
            else:
                if not path.is_file() or path.is_symlink():
                    raise FileNotFoundError(f"benchmark asset is missing: {path}")
                actual = _file_sha256(path)
            if actual != asset["sha256"]:
                raise ValueError(f"benchmark asset digest mismatch: {path}")


class BenchmarkRegistry:
    """Discover versioned benchmark manifests under one or more roots."""

    _lock_type = BenchmarkLock

    def __init__(self, roots: list[str | Path] | tuple[str | Path, ...]) -> None:
        self.roots = tuple(Path(root).resolve() for root in roots)
        if not self.roots:
            raise ValueError("at least one benchmark registry root is required")

    def discover(self) -> tuple[str, ...]:
        """List manifest directory names without importing benchmark code."""
        names: set[str] = set()
        for root in self.roots:
            if not root.is_dir():
                raise FileNotFoundError(f"benchmark registry root is missing: {root}")
            for child in root.iterdir():
                if child.is_dir() and (child / "manifest.json").is_file():
                    if not _BENCHMARK_ID.fullmatch(child.name):
                        raise ValueError(f"invalid benchmark directory: {child.name}")
                    if child.name in names:
                        raise ValueError(f"duplicate benchmark manifest: {child.name}")
                    names.add(child.name)
        return tuple(sorted(names))

    def lock(self, benchmark: str) -> BenchmarkLock:
        """Validate and pin a manifest and its local asset archives."""
        if not isinstance(benchmark, str) or not _BENCHMARK_ID.fullmatch(benchmark):
            raise ValueError("benchmark must be a path-safe identifier")
        matches = [
            root / benchmark / "manifest.json"
            for root in self.roots
            if (root / benchmark / "manifest.json").is_file()
        ]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one manifest for {benchmark}")
        path = matches[0]
        raw = path.read_bytes()
        try:
            document = json.loads(raw, parse_constant=_reject_constant)
        except (UnicodeError, ValueError) as exc:
            raise ValueError(f"invalid benchmark manifest JSON: {path}") from exc
        try:
            Draft202012Validator(benchmark_manifest_schema()).validate(document)
        except ValidationError as exc:
            raise ValueError(
                f"invalid benchmark manifest {path} at {exc.json_path}: {exc.message}"
            ) from exc
        try:
            Draft202012Validator.check_schema(document["capabilities"]["action_schema"])
        except SchemaError as exc:
            raise ValueError(
                f"invalid action_schema in benchmark manifest: {path}"
            ) from exc
        if document["benchmark"] != benchmark:
            raise ValueError(f"manifest benchmark does not match directory: {path}")
        task_ids = [task["task_id"] for task in document["tasks"]]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"duplicate benchmark task_id: {path}")
        for task in document["tasks"]:
            for field in ("instructions_by_seed", "native_by_seed"):
                values = task.get(field)
                if values is not None and set(values) != {
                    str(seed) for seed in task["seeds"]
                }:
                    raise ValueError(f"{field} must cover exactly the declared seeds")
        asset_names = [asset["name"] for asset in document["assets"]]
        if len(asset_names) != len(set(asset_names)):
            raise ValueError(f"duplicate benchmark asset name: {path}")
        lock = self._lock_type(
            manifest_path=path,
            manifest_sha256=hashlib.sha256(raw).hexdigest(),
            document_json=json.dumps(document, allow_nan=False, sort_keys=True),
        )
        lock.verify()
        return lock
