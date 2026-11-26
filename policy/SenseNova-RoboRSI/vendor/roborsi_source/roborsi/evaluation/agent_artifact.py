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

"""Content-pinned Agent source snapshots for independent RoboRSI evaluations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from roborsi.benchmarks.registry import directory_sha256, snapshot_directory
from roborsi.evaluation.agent_environment import AgentEnvironmentStore
from roborsi.evaluation.agent_process import AgentProcess, PythonAgentLauncher
from roborsi.evaluation.deadline import CaseDeadline
from roborsi.evaluation.messages import AgentRuntimeRecord
from roborsi.evaluation.result import write_json_atomic
from roborsi.utils.config import get_repo_root

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_FACTORY = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*\Z")
_COMPONENTS = frozenset({"prompt", "skill", "tool", "control"})


def _relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("component path must be relative POSIX text")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"invalid component path: {value}")
    return value


def _entries(root: Path, *, sealed: bool = False) -> dict[str, str]:
    entries = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"Agent source contains a symlink: {path}")
        mode = stat.S_IMODE(path.stat().st_mode)
        if sealed and mode & 0o222:
            raise ValueError(f"Agent artifact changed: writable path {path}")
        retained_mode = format(mode & ~0o222, "04o")
        if path.is_dir():
            entries[relative] = f"directory:{retained_mode}"
        elif path.is_file():
            checksum = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(block)
            entries[relative] = f"file:{checksum.hexdigest()}:{retained_mode}"
        else:
            raise ValueError(f"Agent source contains a special file: {path}")
    return entries


def _identifier(document: dict[str, Any]) -> str:
    encoded = json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {value}")


def _entries_digest(entries: dict[str, str]) -> str:
    return _identifier(entries)


def _has_factory(root: Path, factory: str) -> bool:
    module = factory.partition(":")[0].replace(".", "/")
    return (root / f"{module}.py").is_file() or (
        root / module / "__init__.py"
    ).is_file()


class AgentArtifact:
    """A verified source tree and its parent/change metadata."""

    def __init__(
        self, store: AgentArtifactStore, version_id: str, document: dict[str, Any]
    ) -> None:
        self.store = store
        self.version_id = version_id
        self.document = document

    @property
    def source_dir(self) -> Path:
        return (
            self.store.root
            / "trees"
            / self.document["entries_sha256"]
            / self.document["tree_sha256"]
        )

    @property
    def parent_version(self) -> str | None:
        return self.document["parent_version"]

    @property
    def environment_version(self) -> str | None:
        """Return the dependency bundle ID, or None for a source-only version."""
        return self.document.get("environment_version")

    @property
    def changes(self) -> tuple[dict[str, str], ...]:
        return tuple(dict(change) for change in self.document["changes"])

    def verify(self) -> None:
        """Rehash the source before starting another independent Agent process."""
        if directory_sha256(self.source_dir) != self.document["tree_sha256"]:
            raise ValueError(f"Agent artifact changed: {self.version_id}")
        if (
            _entries_digest(_entries(self.source_dir, sealed=True))
            != self.document["entries_sha256"]
        ):
            raise ValueError(f"Agent artifact changed: {self.version_id}")
        if _identifier(self.document) != self.version_id:
            raise ValueError(f"Agent artifact metadata changed: {self.version_id}")
        if not _has_factory(self.source_dir, self.document["factory"]):
            raise ValueError("Agent factory module is missing from source tree")
        for paths in self.document["components"].values():
            for path in paths:
                if not (self.source_dir / path).exists():
                    raise FileNotFoundError(f"Agent component is missing: {path}")

    def launcher(
        self,
        *,
        python_executable: str = sys.executable,
        startup_timeout_s: float = 30.0,
        credential_env: Sequence[str] = (),
    ) -> ArtifactAgentLauncher:
        """Launch this version using an environment with its dependencies."""
        self.verify()
        return ArtifactAgentLauncher(
            self,
            python_executable=python_executable,
            startup_timeout_s=startup_timeout_s,
            credential_env=credential_env,
        )

    def checkout(self, destination: str | Path) -> Path:
        """Copy verified source into a new writable directory for an improvement."""
        self.verify()
        target = Path(destination).resolve()
        if target.is_relative_to(self.source_dir.resolve()):
            raise ValueError("Agent checkout must be outside the saved source tree")
        shutil.copytree(self.source_dir, target)
        for path in (target, *target.rglob("*")):
            path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)
        self.verify()
        return target


class AgentArtifactStore:
    """Save immutable Agent trees and content-addressed version manifests."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.environments = AgentEnvironmentStore(self.root / "environments")

    def load(self, version_id: str) -> AgentArtifact:
        """Load a version and reject changed metadata or source bytes."""
        if not isinstance(version_id, str) or not _DIGEST.fullmatch(version_id):
            raise ValueError("Agent artifact version must be a SHA-256 digest")
        path = self.root / "versions" / f"{version_id}.json"
        document = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
        if not isinstance(document, dict) or _identifier(document) != version_id:
            raise ValueError(f"Agent artifact metadata changed: {version_id}")
        fields = {
            "schema_version",
            "tree_sha256",
            "entries_sha256",
            "factory",
            "parent_version",
            "components",
            "changes",
        }
        if document.get("schema_version") == 2:
            fields.add("environment_version")
        if (
            set(document) != fields
            or type(document["schema_version"]) is not int
            or document["schema_version"] not in {1, 2}
        ):
            raise ValueError("unsupported Agent artifact manifest")
        if document["schema_version"] == 2:
            self.environments.load(document["environment_version"])
        if not isinstance(document["tree_sha256"], str) or not _DIGEST.fullmatch(
            document["tree_sha256"]
        ):
            raise ValueError("invalid Agent artifact tree digest")
        if not isinstance(document["entries_sha256"], str) or not _DIGEST.fullmatch(
            document["entries_sha256"]
        ):
            raise ValueError("invalid Agent artifact entries digest")
        if not isinstance(document["factory"], str) or not _FACTORY.fullmatch(
            document["factory"]
        ):
            raise ValueError("invalid Agent artifact factory")
        parent = document["parent_version"]
        if parent is not None and (
            not isinstance(parent, str) or not _DIGEST.fullmatch(parent)
        ):
            raise ValueError("invalid Agent artifact parent version")
        components = document["components"]
        if (
            not isinstance(components, dict)
            or not components
            or set(components) - _COMPONENTS
        ):
            raise ValueError("invalid Agent artifact components")
        for paths in components.values():
            if not isinstance(paths, list) or not paths:
                raise ValueError("invalid Agent artifact component paths")
            selected = [_relative_path(path) for path in paths]
            if len(set(selected)) != len(selected):
                raise ValueError("duplicate Agent artifact component path")
        changes = document["changes"]
        if not isinstance(changes, list) or any(
            not isinstance(change, dict)
            or set(change) != {"path", "status"}
            or change["status"] not in {"added", "modified", "deleted"}
            or not isinstance(change["path"], str)
            for change in changes
        ):
            raise ValueError("invalid Agent artifact change list")
        for change in changes:
            _relative_path(change["path"])
        artifact = AgentArtifact(self, version_id, document)
        artifact.verify()
        return artifact

    def create(
        self,
        source: str | Path,
        factory: str,
        *,
        components: Mapping[str, Sequence[str]],
        parent_version: str | None = None,
        environment_version: str | None = None,
    ) -> AgentArtifact:
        """Snapshot source, derive changes, and publish one reproducible version."""
        if not isinstance(factory, str) or not _FACTORY.fullmatch(factory):
            raise ValueError("Agent factory must be module:function")
        if not isinstance(components, Mapping) or set(components) - _COMPONENTS:
            raise ValueError("unknown Agent component kind")
        if not components:
            raise ValueError("Agent artifact needs at least one component")
        parent = self.load(parent_version) if parent_version is not None else None
        if parent is not None and environment_version is None:
            environment_version = parent.document.get("environment_version")
        if environment_version is not None:
            self.environments.load(environment_version)
        source_entries = _entries(Path(source))
        entries_sha256 = _entries_digest(source_entries)
        snapshot = snapshot_directory(source, self.root / "trees" / entries_sha256)
        snapshot_entries = _entries(snapshot, sealed=True)
        if source_entries != snapshot_entries:
            raise ValueError("Agent source modes differ from an existing snapshot")
        if not _has_factory(snapshot, factory):
            raise ValueError("Agent factory module is missing from source tree")
        normalized = {}
        for kind, paths in components.items():
            if isinstance(paths, str) or not isinstance(paths, Sequence) or not paths:
                raise ValueError(f"{kind} component paths must be a non-empty list")
            selected = tuple(_relative_path(path) for path in paths)
            if len(set(selected)) != len(selected):
                raise ValueError(f"duplicate {kind} component path")
            for path in selected:
                if not (snapshot / path).exists():
                    raise FileNotFoundError(f"Agent component is missing: {path}")
            normalized[kind] = list(selected)
        before = _entries(parent.source_dir, sealed=True) if parent is not None else {}
        after = snapshot_entries
        changes = [
            {
                "path": path,
                "status": "added"
                if path not in before
                else "deleted"
                if path not in after
                else "modified",
            }
            for path in sorted(set(before) | set(after))
            if before.get(path) != after.get(path)
        ]
        document = {
            "schema_version": 1,
            "tree_sha256": snapshot.name,
            "entries_sha256": entries_sha256,
            "factory": factory,
            "parent_version": parent_version,
            "components": normalized,
            "changes": changes,
        }
        if environment_version is not None:
            document.update(schema_version=2, environment_version=environment_version)
        version_id = _identifier(document)
        directory = self.root / "versions"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{version_id}.json"
        temporary = directory / f".{uuid.uuid4().hex}.json"
        try:
            write_json_atomic(temporary, document)
            try:
                os.link(temporary, destination)
            except FileExistsError:
                self.load(version_id)
        finally:
            temporary.unlink(missing_ok=True)
        return self.load(version_id)


class ArtifactAgentLauncher:
    """Verify the artifact before spawning each case-scoped Python Agent."""

    def __init__(
        self,
        artifact: AgentArtifact,
        *,
        python_executable: str,
        startup_timeout_s: float,
        credential_env: Sequence[str] = (),
    ) -> None:
        self._artifact = artifact
        self._version_id = artifact.version_id
        self._python_executable = python_executable
        self._startup_timeout_s = startup_timeout_s
        if isinstance(credential_env, str) or any(
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
            or name.startswith(("PYTHON", "LD_", "DYLD_", "ROBORSI_", "HEART_"))
            or name in {"PATH", "HOME", "TMPDIR"}
            for name in credential_env
        ):
            raise ValueError("credential_env must contain credential variable names")
        self._credential_env = tuple(credential_env)
        if (
            self._credential_env
            and artifact.document.get("environment_version") is None
        ):
            raise ValueError("credential_env requires a pinned Agent environment")
        source = str(artifact.source_dir)
        paths = [source, str(get_repo_root())]
        inherited = os.environ.get("PYTHONPATH")
        if inherited:
            paths.append(inherited)
        self._launcher = PythonAgentLauncher(
            artifact.document["factory"],
            python_executable=python_executable,
            env_overrides={
                "PYTHONPATH": os.pathsep.join(paths),
                "PYTHONDONTWRITEBYTECODE": "1",
                "ROBORSI_AGENT_ARTIFACT_VERSION": self._version_id,
                "ROBORSI_AGENT_ARTIFACT_ROOT": source,
            },
            startup_timeout_s=startup_timeout_s,
            cwd=source,
        )

    @property
    def agent_version(self) -> str:
        return self._version_id

    def verify_version(self, requested: str) -> None:
        """Refuse a caller's mismatched version before claiming a case."""
        if requested != self._version_id:
            raise ValueError("declared Agent version differs from pinned artifact")
        self._artifact.store.load(self._version_id)

    def start(
        self,
        case_dir: Path,
        *,
        deadline: CaseDeadline | None = None,
        media_dir: Path | None = None,
    ) -> AgentProcess:
        self.verify_version(self._version_id)
        environment_version = self._artifact.document.get("environment_version")
        if environment_version is not None:
            bundle = self._artifact.store.environments.load(environment_version)
            prepared = bundle.prepare(
                case_dir / "python",
                python_executable=self._python_executable,
                deadline=deadline,
            )
            runtime = {
                "schema_version": 1,
                "agent_version": self._version_id,
                "environment_version": environment_version,
                "python": bundle.document["python"],
                "installed_sha256": prepared.digest,
                "verified_after_close": False,
            }
            write_json_atomic(case_dir / "runtime.json", runtime)

            def verify_runtime() -> AgentRuntimeRecord:
                self._artifact.verify()
                prepared.verify()
                record = AgentRuntimeRecord.from_dict(
                    {**runtime, "verified_after_close": True}
                )
                write_json_atomic(case_dir / "runtime.json", record.as_dict())
                return record

            temporary = case_dir / "tmp"
            temporary.mkdir(exist_ok=True)
            environment = {
                "PATH": f"{prepared.root / 'bin'}{os.pathsep}{os.defpath}",
                "HOME": str(case_dir.resolve()),
                "TMPDIR": str(temporary.resolve()),
                "LANG": "C.UTF-8",
                "ROBORSI_AGENT_ARTIFACT_VERSION": self._version_id,
                "ROBORSI_AGENT_ARTIFACT_ROOT": str(self._artifact.source_dir),
                **{
                    name: os.environ[name]
                    for name in self._credential_env
                    if name in os.environ
                },
            }
            return PythonAgentLauncher(
                self._artifact.document["factory"],
                python_executable=prepared.python_executable,
                env_overrides=environment,
                startup_timeout_s=self._startup_timeout_s,
                cwd=self._artifact.source_dir,
                isolated=True,
                source_root=self._artifact.source_dir,
                verify_runtime=verify_runtime,
            ).start(case_dir, deadline=deadline, media_dir=media_dir)
        return self._launcher.start(case_dir, deadline=deadline, media_dir=media_dir)
