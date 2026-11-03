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

"""Authenticate chunked native records for a future own-episode resume.

This module never reads author experiment outputs. It restores only explicitly
collected artifacts below one attempt. The current publisher
uses complete or partial_technical_interruption; interrupted status is never
rewritten as complete. Original controller config, actions and
transitions are never replaced by native-service configuration or snapshots.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from roborsi.evaluation.messages import ArtifactInfo

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


def _safe(root: Path, relative: str, *, required: bool = True) -> Path:
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise ValueError("Invalid native artifact path")
    part = Path(relative)
    if (
        part.is_absolute()
        or ".." in part.parts
        or part == Path(".")
        or part.as_posix() != relative
        or "\\" in relative
    ):
        raise ValueError("Native artifact path escapes its own attempt")
    if any(name.lower() in SECRET_NAMES for name in part.parts):
        raise ValueError("Authentication files are not native artifacts")
    path = root / part
    current = path
    while current != root:
        if current.is_symlink():
            raise ValueError("Native artifact path traverses a symlink")
        current = current.parent
    if any(parent.is_symlink() for parent in (root, *root.parents)):
        raise ValueError("Native attempt root or ancestor must not be linked")
    if required and not path.is_file():
        raise ValueError("Missing collected native artifact")
    return path


def _hash(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Native artifacts must be regular files")
        count = 0
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            count += len(block)
            digest.update(block)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("Native artifact changed during verification")
    return count, digest.hexdigest()


def verify_native_manifest(attempt_dir: Path, episode_id: str) -> dict[str, Any]:
    """Check every declared chunk and complete-file SHA for one framework episode.

    Args:
        attempt_dir: Attempt containing native/icl-reconstruction.json and chunks.
        episode_id: Exact framework episode bound by the controller sidecar.

    Returns:
        Authenticated manifest. A partial snapshot remains explicitly partial.

    Raises:
        ValueError: Missing, foreign, linked, changed or incomplete chunk data.
    """
    attempt = attempt_dir.absolute()
    manifest_path = _safe(attempt, "native/icl-reconstruction.json")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(document, dict)
        or type(document.get("schema_version")) is not int
        or document.get("schema_version") != 1
        or document.get("status") not in {"complete", "partial_technical_interruption"}
    ):
        raise ValueError("Invalid native reconstruction manifest")
    if not isinstance(document.get("files"), list) or not isinstance(
        document.get("exclusions"), list
    ):
        raise ValueError("Native reconstruction manifest lacks inventory")
    seen = set()
    for row in document["files"]:
        if not isinstance(row, dict) or set(row) != {
            "path",
            "size_bytes",
            "sha256",
            "chunks",
        }:
            raise ValueError("Invalid native reconstruction row")
        if not isinstance(row["path"], str) or Path(row["path"]).is_absolute():
            raise ValueError("Native source path must be relative")
        target = _safe(attempt, "native/icl-original/" + row["path"], required=False)
        if (
            row["path"] in seen
            or type(row["size_bytes"]) is not int
            or row["size_bytes"] < 0
        ):
            raise ValueError("Duplicate or invalid native source size")
        seen.add(row["path"])
        if not isinstance(row["chunks"], list) or not row["chunks"]:
            raise ValueError("Native source has no authenticated chunks")
        combined = hashlib.sha256()
        size = 0
        chunk_refs = set()
        for chunk_data in row["chunks"]:
            chunk = ArtifactInfo.from_dict(chunk_data)
            if chunk.ref in chunk_refs:
                raise ValueError("Native source repeats the same chunk reference")
            chunk_refs.add(chunk.ref)
            if chunk.episode_id != episode_id:
                raise ValueError("Native chunk belongs to another framework episode")
            if not chunk.ref.startswith(("native/icl-original/", "native/icl-chunks/")):
                raise ValueError(
                    "Native chunk reference is outside original artifact namespace"
                )
            path = _safe(attempt, chunk.ref)
            actual_size, actual_sha = _hash(path)
            if actual_size != chunk.size_bytes or actual_sha != chunk.sha256:
                raise ValueError(
                    "Collected native chunk differs from published metadata"
                )
            # Stream again into the complete-file digest, without trusting an
            # already assembled file or a provider-supplied status flag.
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("Native chunk changed to a special node")
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    combined.update(block)
                    size += len(block)
        if size != row["size_bytes"] or combined.hexdigest() != row["sha256"]:
            raise ValueError(
                "Complete native source hash differs from collected chunks"
            )
        if target.exists() and _hash(target) != (row["size_bytes"], row["sha256"]):
            raise ValueError(
                "Existing reconstructed native file differs; do not overwrite"
            )
    return document


def restore_native_artifacts(attempt_dir: Path, episode_id: str) -> Path:
    """Reassemble verified files idempotently without overwriting existing data.

    This is a future own-run mechanism. Caller must hold its case lifecycle
    lock after native shutdown. The input manifest never authorizes a destination
    outside native/icl-original or any change to original host configuration.

    Args:
        attempt_dir: Closed attempt with its authenticated chunk manifest.
        episode_id: Exact controller-bound framework episode.

    Returns:
        Native original snapshot directory below the same attempt.
    """
    attempt = attempt_dir.absolute()
    document = verify_native_manifest(attempt, episode_id)
    destination = _safe(attempt, "native/icl-original", required=False)
    destination.mkdir(parents=True, exist_ok=True)
    for row in document["files"]:
        target = _safe(attempt, "native/icl-original/" + row["path"], required=False)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".native-restore-", dir=target.parent
        )
        try:
            digest = hashlib.sha256()
            size = 0
            with os.fdopen(descriptor, "wb") as writer:
                for chunk_data in row["chunks"]:
                    chunk = ArtifactInfo.from_dict(chunk_data)
                    path = _safe(attempt, chunk.ref)
                    reader_fd = os.open(
                        path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                    )
                    with os.fdopen(reader_fd, "rb") as reader:
                        if not stat.S_ISREG(os.fstat(reader.fileno()).st_mode):
                            raise ValueError("Native chunk changed to a special node")
                        for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                            writer.write(block)
                            digest.update(block)
                            size += len(block)
                writer.flush()
                os.fsync(writer.fileno())
            if size != row["size_bytes"] or digest.hexdigest() != row["sha256"]:
                raise ValueError("Native chunks changed before reconstruction")
            try:
                os.link(temporary, target)
            except FileExistsError:
                if _hash(target) != (row["size_bytes"], row["sha256"]):
                    raise ValueError(
                        "Native destination changed; do not overwrite"
                    ) from None
        finally:
            os.unlink(temporary)
    return destination


def copy_native_resume_records(
    attempt_dir: Path,
    episode_id: str,
    source_run: Path,
    *,
    experiment_root: Path,
    case_dir: Path,
) -> dict[str, Any]:
    """Fill missing own-run camera/physical records without altering host state.

    Only observations and episodes/<episode>/native files are copied. Complete
    raw native config, closure and scores stay in the attempt snapshot. The
    original host config/actions/transitions/workspace are never rewritten.
    Caller holds the shared lifecycle lock; controller and native are stopped.
    """
    from .resume import _owned
    from .runtime_records import _assert_stopped, assert_runtime_owner

    runtime = experiment_root.resolve()
    case = case_dir.resolve()
    assert_runtime_owner(runtime, case)
    _assert_stopped(runtime)
    source = _owned(source_run, runtime)
    if not source.is_relative_to(runtime / "runs"):
        raise ValueError("Native resume target is outside its own original runs")
    attempt = attempt_dir.absolute()
    if attempt != case and not attempt.is_relative_to(case / "attempts"):
        raise ValueError("Native attempt belongs to another case")
    binding = json.loads(
        _owned(source / "roborsi-controller.json", runtime).read_text()
    )
    if (
        binding.get("source_run") != str(source)
        or binding.get("case_dir") != str(case)
        or binding.get("attempt_dir") != str(attempt)
        or binding.get("framework_episode_id") != episode_id
    ):
        raise ValueError("Native snapshot does not match its controller attempt")
    snapshot = restore_native_artifacts(attempt, episode_id)
    native = json.loads(
        _safe(attempt, "native/icl-original/worker-close.json").read_text()
    )
    if (
        native.get("native_closed") is not True
        or native.get("episode_id") != episode_id
        or native.get("native_config_sha256") != binding.get("native_config_sha256")
    ):
        raise ValueError("Native worker closure is not verified for this own attempt")
    document = verify_native_manifest(attempt, episode_id)
    copied = []
    for row in document["files"]:
        relative = Path(row["path"])
        parts = relative.parts
        needed = parts[0] == "observations" or (
            len(parts) >= 4 and parts[0] == "episodes" and parts[2] == "native"
        )
        if not needed:
            continue
        target = _owned(source / relative, runtime, must_exist=False)
        if target.exists():
            if _hash(target) != (row["size_bytes"], row["sha256"]):
                raise ValueError(
                    "Original host observation/record conflicts with native snapshot"
                )
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".native-copy-", dir=target.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as writer:
                reader_fd = os.open(
                    snapshot / relative, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                )
                with os.fdopen(reader_fd, "rb") as reader:
                    if not stat.S_ISREG(os.fstat(reader.fileno()).st_mode):
                        raise ValueError("Native snapshot changed to a special node")
                    for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                        writer.write(block)
                writer.flush()
                os.fsync(writer.fileno())
            if _hash(Path(temporary)) != (row["size_bytes"], row["sha256"]):
                raise ValueError("Native snapshot changed during own-run copy")
            try:
                os.link(temporary, target)
            except FileExistsError:
                if _hash(target) != (row["size_bytes"], row["sha256"]):
                    raise ValueError(
                        "Original host record changed; do not overwrite"
                    ) from None
            copied.append(relative.as_posix())
        finally:
            os.unlink(temporary)
    return {
        "source_run": str(source),
        "attempt_dir": str(attempt),
        "copied": copied,
        "native_snapshot_status": document["status"],
        "host_configuration_preserved": True,
        "author_experiment_loaded": False,
    }
