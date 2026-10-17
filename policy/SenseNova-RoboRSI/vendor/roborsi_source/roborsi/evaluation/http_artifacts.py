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

"""Publish Benchmark artifacts over HTTP and archive checked downloads."""

from __future__ import annotations

import hashlib
import os
import stat
import threading
import uuid
from pathlib import Path

from roborsi.evaluation.messages import ArtifactInfo, _case_relative_ref

MAX_ARTIFACT_BYTES = 256 * 1024 * 1024


def artifact_path(root: Path, ref: str, *, create: bool = False) -> Path:
    """Resolve an archive name without following symlinks or special files."""
    _case_relative_ref(ref)
    if ref.split("/")[0] not in {"media", "native", "trace"}:
        raise ValueError("invalid artifact archive name")
    if root.is_symlink():
        raise ValueError("artifact root must not be a symlink")
    if create:
        root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    path = root
    for part in ref.split("/")[:-1]:
        path = path / part
        if create:
            path.mkdir(exist_ok=True)
        if path.is_symlink() or not path.is_dir():
            raise ValueError("invalid artifact directory")
    path = path / ref.split("/")[-1]
    if path.is_symlink():
        raise ValueError("artifact symlinks are not allowed")
    return path


def read_artifact(root: Path, ref: str) -> bytes:
    path = artifact_path(root, ref)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_ARTIFACT_BYTES:
            raise ValueError("artifact is not a bounded regular file")
        data = stream.read(MAX_ARTIFACT_BYTES + 1)
    if len(data) > MAX_ARTIFACT_BYTES:
        raise ValueError("artifact exceeds size limit")
    return data


def save_artifact(root: Path, info: ArtifactInfo, data: bytes) -> None:
    """Save verified bytes once; failed transfers never become published files."""
    if len(data) != info.size_bytes or hashlib.sha256(data).hexdigest() != info.sha256:
        raise ValueError("downloaded artifact size or SHA-256 differs")
    path = artifact_path(root, info.ref, create=True)
    if path.exists():
        if read_artifact(root, info.ref) != data:
            raise ValueError("artifact archive name already contains different bytes")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            os.chmod(temporary, 0o600)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ArtifactPublisher:
    """Expose only explicitly published artifacts belonging to one episode."""

    def __init__(self, root: Path, episode_id: str) -> None:
        self.root, self.episode_id = root.resolve(), episode_id
        self._files: dict[str, ArtifactInfo] = {}
        self._lock = threading.RLock()

    def publish(
        self,
        ref: str,
        kind: str,
        mime_type: str,
        *,
        step_index: int | None = None,
        sha256: str | None = None,
        size_bytes: int | None = None,
    ) -> ArtifactInfo:
        data = read_artifact(self.root, ref)
        digest = hashlib.sha256(data).hexdigest()
        if (sha256 is not None and digest != sha256) or (
            size_bytes is not None and len(data) != size_bytes
        ):
            raise ValueError("Benchmark artifact differs from its result reference")
        identifier = hashlib.sha256(f"{self.episode_id}\0{ref}".encode()).hexdigest()
        info = ArtifactInfo(
            artifact_id=identifier,
            episode_id=self.episode_id,
            ref=ref,
            kind=kind,
            mime_type=mime_type,
            size_bytes=len(data),
            sha256=digest,
            step_index=step_index,
        )
        with self._lock:
            previous = self._files.get(identifier)
            if previous is not None and previous != info:
                raise ValueError("published artifact content changed")
            self._files[identifier] = info
        return info

    def list(self) -> list[ArtifactInfo]:
        with self._lock:
            return list(self._files.values())

    def get(self, identifier: str) -> tuple[ArtifactInfo, bytes]:
        with self._lock:
            info = self._files[identifier]
        data = read_artifact(self.root, info.ref)
        if (
            len(data) != info.size_bytes
            or hashlib.sha256(data).hexdigest() != info.sha256
        ):
            raise ValueError("published artifact content changed")
        return info, data
