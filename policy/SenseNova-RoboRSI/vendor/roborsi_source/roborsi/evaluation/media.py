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

"""Case-scoped binary observations shared by Adapter and Agent processes."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import uuid
from pathlib import Path, PurePosixPath

from roborsi.evaluation.messages import MediaRef, Observation

MAX_MEDIA_BYTES = 16 * 1024 * 1024
MAX_OBSERVATION_MEDIA_BYTES = 64 * 1024 * 1024
MAX_OBSERVATION_MEDIA_FILES = 8
_FILE_NAME = re.compile(r"[0-9a-f]{32}\.bin\Z")
_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})


class CaseMediaChannel:
    """Write immutable media files and verify one observation's references.

    ``media_dir`` is the same case-scoped directory in both runtimes. Files
    remain with the case evidence; temporary files are removed on write errors.
    """

    def __init__(self, media_dir: str | Path) -> None:
        self.media_dir = Path(media_dir)
        if self.media_dir.is_symlink():
            raise ValueError("media directory must not be a symlink")
        self.media_dir.mkdir(parents=True, exist_ok=True)
        if not self.media_dir.is_dir():
            raise ValueError("media directory must be a directory")
        self._root = self.media_dir.resolve()

    @staticmethod
    def _group(episode_id: str, step_index: int) -> tuple[str, str]:
        if not episode_id or step_index < 0:
            raise ValueError("media needs an episode ID and nonnegative step")
        return hashlib.sha256(
            episode_id.encode("utf-8")
        ).hexdigest(), f"{step_index:08d}"

    def write(
        self,
        episode_id: str,
        step_index: int,
        payload: bytes,
        mime_type: str,
    ) -> MediaRef:
        """Atomically publish one bounded camera image and return its reference."""
        group, step = self._group(episode_id, step_index)
        if mime_type not in _MIME_TYPES:
            raise ValueError("unsupported media MIME type")
        if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_MEDIA_BYTES:
            raise ValueError("media payload must be nonempty bounded bytes")
        group_directory = self.media_dir / group
        group_directory.mkdir(exist_ok=True)
        if group_directory.is_symlink() or not group_directory.resolve().is_relative_to(
            self._root
        ):
            raise ValueError("media path escapes case directory")
        directory = group_directory / step
        directory.mkdir(exist_ok=True)
        if directory.is_symlink() or not directory.resolve().is_relative_to(self._root):
            raise ValueError("media path escapes case directory")
        name = f"{uuid.uuid4().hex}.bin"
        destination = directory / name
        temporary = directory / f".{name}.tmp"
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return MediaRef(
            ref=f"media/{group}/{step}/{name}",
            mime_type=mime_type,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )

    def read(self, reference: MediaRef, episode_id: str, step_index: int) -> bytes:
        """Read a reference only for its own episode and observation step."""
        group, step = self._group(episode_id, step_index)
        parts = PurePosixPath(reference.ref).parts
        if (
            len(parts) != 4
            or parts[:3] != ("media", group, step)
            or not _FILE_NAME.fullmatch(parts[3])
        ):
            raise ValueError("media reference belongs to a different observation")
        if reference.mime_type not in _MIME_TYPES:
            raise ValueError("unsupported media MIME type")
        if not 0 < reference.size_bytes <= MAX_MEDIA_BYTES:
            raise ValueError("media reference exceeds size limit")
        parent = self.media_dir / group / step
        path = parent / parts[3]
        if (
            parent.is_symlink()
            or parent.parent.is_symlink()
            or path.is_symlink()
            or not path.resolve().is_relative_to(self._root)
        ):
            raise ValueError("media reference escapes case directory")
        flags = (
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise ValueError("media reference is unavailable") from exc
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("media reference must be a regular file")
            data = stream.read(MAX_MEDIA_BYTES + 1)
        if len(data) != reference.size_bytes:
            raise ValueError("media reference size mismatch")
        if hashlib.sha256(data).hexdigest() != reference.sha256:
            raise ValueError("media reference digest mismatch")
        return data

    def verify_observation(self, observation: Observation) -> None:
        """Reject missing, corrupt, oversized, repeated, or cross-step media."""
        if len(observation.media) > MAX_OBSERVATION_MEDIA_FILES:
            raise ValueError("observation has too many media files")
        if (
            sum(item.size_bytes for item in observation.media)
            > MAX_OBSERVATION_MEDIA_BYTES
        ):
            raise ValueError("observation media exceeds total size limit")
        seen: set[str] = set()
        for reference in observation.media:
            if reference.ref in seen:
                raise ValueError("observation repeats a media reference")
            seen.add(reference.ref)
            self.read(reference, observation.episode_id, observation.step_index)
