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

"""Store hashed wheel bundles and rebuild an independent Python for each case."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
import zipfile
from email.parser import Parser
from pathlib import Path
from typing import Any

from roborsi.benchmarks.registry import directory_sha256, snapshot_directory
from roborsi.evaluation.deadline import CaseDeadline
from roborsi.evaluation.python_runtime import file_sha256
from roborsi.evaluation.result import write_json_atomic

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*\Z")


def _identifier(document: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            document, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _command(
    args: list[str], deadline: CaseDeadline | None, *, timeout_s: float
) -> str:
    """Run setup commands without inherited Python/pip settings or credentials."""
    if deadline is not None:
        deadline.check()
    # --isolated alone still reads global and virtualenv pip configuration.
    environment = {
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "PIP_CONFIG_FILE": os.devnull,
    }
    with subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
        start_new_session=True,
    ) as process:
        end = time.monotonic() + timeout_s
        try:
            while True:
                remaining = end - time.monotonic()
                if deadline is not None:
                    remaining = min(remaining, deadline.remaining())
                if remaining <= 0:
                    raise TimeoutError("Agent environment setup timed out")
                try:
                    output, _ = process.communicate(timeout=min(0.2, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise
        if process.returncode:
            raise ValueError(f"Agent environment command failed: {output.strip()}")
        return output


def _python_identity(executable: str, deadline: CaseDeadline | None = None) -> dict:
    script = Path(__file__).with_name("python_runtime.py")
    return json.loads(
        _command([executable, "-I", "-B", "-S", str(script)], deadline, timeout_s=60)
    )


def _installed_digest(root: Path) -> str:
    """Include executable modes and internal links as well as installed bytes."""
    entries = {}
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(
                    f"Agent environment link escapes its directory: {name}"
                )
            entries[name] = ["link", os.readlink(path)]
        elif path.is_file():
            entries[name] = ["file", path.stat().st_mode & 0o777, file_sha256(path)]
        elif path.is_dir():
            entries[name] = ["directory", path.stat().st_mode & 0o777]
        else:
            raise ValueError(f"Agent environment contains a special file: {name}")
    return _identifier(entries)


def _read_wheels(root: Path) -> list[dict[str, str]]:
    """Validate wheel metadata and hash every file in the bundle."""
    wheels, names = [], set()
    for path in sorted(root.iterdir()):
        if path.is_symlink() or not path.is_file() or path.suffix != ".whl":
            raise ValueError("Agent environment input must contain only wheel files")
        with zipfile.ZipFile(path) as archive:
            metadata = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata) != 1:
                raise ValueError(f"wheel metadata is ambiguous: {path.name}")
            package = Parser().parsestr(archive.read(metadata[0]).decode())
        name, version = package.get("Name", ""), package.get("Version", "")
        if not _NAME.fullmatch(name) or not _VERSION.fullmatch(version):
            raise ValueError(f"invalid wheel package name or version: {path.name}")
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if normalized in names:
            raise ValueError(f"duplicate wheel package: {name}")
        names.add(normalized)
        wheels.append(
            {
                "file": path.name,
                "name": name,
                "version": version,
                "sha256": file_sha256(path),
            }
        )
    if not wheels:
        raise ValueError("Agent environment needs at least one wheel")
    return wheels


class PreparedAgentEnvironment:
    """A fresh installation whose files are checked again after the case closes."""

    def __init__(
        self, bundle: AgentEnvironment, root: Path, digest: str, base_python: str
    ) -> None:
        self.bundle = bundle
        self.root = root
        self.digest = digest
        self.base_python = base_python

    @property
    def python_executable(self) -> str:
        return str(self.root / "bin" / "python")

    def verify(self) -> None:
        """Refuse changes to installed files or the underlying Python runtime."""
        self.bundle.verify(self.base_python)
        if _installed_digest(self.root) != self.digest:
            raise ValueError("installed Agent environment changed during evaluation")


class AgentEnvironment:
    """An immutable, platform-specific Python identity and complete wheel set."""

    def __init__(
        self, store: AgentEnvironmentStore, version_id: str, document: dict
    ) -> None:
        self.store, self.version_id, self.document = store, version_id, document

    @property
    def wheel_dir(self) -> Path:
        return self.store.root / "trees" / self.document["tree_sha256"]

    def verify(
        self,
        python_executable: str | None = None,
        *,
        deadline: CaseDeadline | None = None,
    ) -> None:
        """Check stored bytes and optionally require an identical base Python."""
        if _identifier(self.document) != self.version_id:
            raise ValueError("Agent environment metadata changed")
        if directory_sha256(self.wheel_dir) != self.document["tree_sha256"]:
            raise ValueError("Agent environment wheels changed")
        if _read_wheels(self.wheel_dir) != self.document["wheels"]:
            raise ValueError(
                "Agent environment wheel metadata differs from stored files"
            )
        if (
            python_executable is not None
            and _python_identity(python_executable, deadline) != self.document["python"]
        ):
            raise ValueError("Agent environment requires a different base Python")

    def prepare(
        self,
        directory: Path,
        *,
        python_executable: str = sys.executable,
        deadline: CaseDeadline | None = None,
    ) -> PreparedAgentEnvironment:
        """Install pinned wheels offline into a new per-case virtual environment."""
        if os.name != "posix":
            raise ValueError("locked Agent environments currently require POSIX")
        self.verify(python_executable, deadline=deadline)
        directory = directory.resolve()
        directory.mkdir(exist_ok=False)
        requirements = directory / "requirements.lock"
        requirements.write_text(
            "".join(
                f"{wheel['name']}=={wheel['version']} --hash=sha256:{wheel['sha256']}\n"
                for wheel in self.document["wheels"]
            )
        )
        _command(
            [
                python_executable,
                "-I",
                "-B",
                "-S",
                "-m",
                "venv",
                "--copies",
                str(directory),
            ],
            deadline,
            timeout_s=120,
        )
        executable = str(directory / "bin" / "python")
        _command(
            [
                executable,
                "-I",
                "-B",
                "-m",
                "pip",
                "--isolated",
                "--no-cache-dir",
                "install",
                "--no-index",
                "--no-deps",
                "--only-binary=:all:",
                "--require-hashes",
                "--no-compile",
                "--find-links",
                str(self.wheel_dir),
                "-r",
                str(requirements),
            ],
            deadline,
            timeout_s=600,
        )
        _command(
            [executable, "-I", "-B", "-m", "pip", "--isolated", "check"],
            deadline,
            timeout_s=60,
        )
        digest = _installed_digest(directory)
        return PreparedAgentEnvironment(self, directory, digest, python_executable)


class AgentEnvironmentStore:
    """Import a wheel directory once; reuse it without sharing writable installs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def create(
        self, wheel_dir: str | Path, *, python_executable: str = sys.executable
    ) -> AgentEnvironment:
        """Save all wheels, their hashes and the Python required to rebuild them."""
        root = Path(wheel_dir)
        wheels = _read_wheels(root)
        snapshot = snapshot_directory(root, self.root / "trees")
        # Detect a source that changed between metadata inspection and snapshot.
        if any(
            file_sha256(snapshot / item["file"]) != item["sha256"] for item in wheels
        ):
            raise ValueError("Agent environment wheels changed while importing")
        document = {
            "schema_version": 1,
            "python": _python_identity(python_executable),
            "tree_sha256": snapshot.name,
            "wheels": wheels,
        }
        version_id = _identifier(document)
        versions = self.root / "versions"
        versions.mkdir(parents=True, exist_ok=True)
        temporary = versions / f".{uuid.uuid4().hex}.json"
        try:
            write_json_atomic(temporary, document)
            try:
                os.link(temporary, versions / f"{version_id}.json")
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
        return self.load(version_id)

    def load(self, version_id: str) -> AgentEnvironment:
        """Load a content-addressed bundle and reject altered wheel files."""
        if not isinstance(version_id, str) or not _DIGEST.fullmatch(version_id):
            raise ValueError("Agent environment version must be a SHA-256 digest")
        document = json.loads(
            (self.root / "versions" / f"{version_id}.json").read_text()
        )
        if not isinstance(document, dict) or _identifier(document) != version_id:
            raise ValueError("Agent environment metadata changed")
        if (
            set(document) != {"schema_version", "python", "tree_sha256", "wheels"}
            or type(document["schema_version"]) is not int
            or document["schema_version"] != 1
        ):
            raise ValueError("unsupported Agent environment manifest")
        if not isinstance(document["tree_sha256"], str) or not _DIGEST.fullmatch(
            document["tree_sha256"]
        ):
            raise ValueError("invalid Agent environment tree digest")
        bundle = AgentEnvironment(self, version_id, document)
        bundle.verify()
        return bundle
