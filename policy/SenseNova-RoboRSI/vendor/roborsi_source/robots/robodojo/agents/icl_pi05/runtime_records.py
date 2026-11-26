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

"""Own short private runtimes and publish stopped-episode record snapshots.

Original runtime paths stay intact for resume. No model input, original source
or live result is modified by this artifact helper.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

OWNER_FILE = "RUNTIME_OWNER.json"
EXCLUDED_NAMES = frozenset(
    {
        "auth.json",
        "api_keys.yaml",
        "api_keys.yml",
        ".env",
        "credentials.json",
        "credentials",
        "tokens.json",
        "token.json",
        "secrets.json",
        ".lifecycle.lock",
    }
)


def _write(path: Path, value: Any) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def create_runtime(case_dir: Path) -> Path:
    """Create a private short runtime, bound to one existing evaluation case.

    Args:
        case_dir: Existing case directory managed by the evaluation Runner.

    Returns:
        Owned 0700 runtime under /tmp, retained until explicit archive cleanup.
    """
    if case_dir.is_symlink() or not case_dir.is_dir():
        raise ValueError("Runtime requires an existing case directory")
    case = case_dir.resolve()
    index = case / "icl"
    if index.is_symlink():
        raise ValueError("Case ICL directory must not be a symlink")
    index.mkdir(exist_ok=True, mode=0o700)
    reference = index / "runtime-reference.json"
    if reference.exists() or reference.is_symlink():
        raise ValueError("Case already has a runtime reference")
    runtime = Path(tempfile.mkdtemp(prefix="roborsi-icl-case-", dir="/tmp"))
    runtime.chmod(0o700)
    owner = {
        "runtime": str(runtime),
        "case_dir": str(case),
        "uid": os.getuid(),
        "generation": uuid.uuid4().hex,
        "device": runtime.stat().st_dev,
        "inode": runtime.stat().st_ino,
    }
    _write(runtime / OWNER_FILE, owner)
    _write(reference, owner)
    return runtime


def assert_runtime_owner(runtime: Path, case_dir: Path) -> dict[str, Any]:
    """Validate private runtime identity against the case-side reference."""
    if runtime.is_symlink() or case_dir.is_symlink():
        raise ValueError("Runtime and case paths must not be symlinks")
    runtime = runtime.resolve()
    case = case_dir.resolve()
    if runtime.parent != Path("/tmp") or not runtime.name.startswith(
        "roborsi-icl-case-"
    ):
        raise ValueError("Not an owned short ICL runtime")
    owner_path = runtime / OWNER_FILE
    reference_path = case / "icl/runtime-reference.json"
    if (
        owner_path.is_symlink()
        or reference_path.is_symlink()
        or reference_path.parent.is_symlink()
    ):
        raise ValueError("Runtime ownership records must not be symlinks")
    owner = _read(owner_path)
    if owner != _read(reference_path):
        raise ValueError("Runtime ownership reference differs")
    metadata = runtime.stat()
    if (
        owner.get("runtime") != str(runtime)
        or owner.get("case_dir") != str(case)
        or owner.get("uid") != os.getuid()
        or metadata.st_uid != os.getuid()
        or owner.get("device") != metadata.st_dev
        or owner.get("inode") != metadata.st_ino
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ValueError("Runtime identity or private permissions differ")
    return owner


@contextmanager
def runtime_lock(runtime: Path, case_dir: Path) -> Iterator[None]:
    """Hold one nonblocking lifecycle lock for start, resume or publication.

    The lifecycle owner holds this across live execution. Publication obtains
    it after close. Resume must use the same lock, not an independent sentinel.
    """
    assert_runtime_owner(runtime, case_dir)
    descriptor = os.open(
        runtime / ".lifecycle.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("ICL runtime lifecycle is already active") from exc
        yield
    finally:
        os.close(descriptor)


def _process(pid: int) -> dict[str, Any] | None:
    # Never inspect a numeric host PID when /proc belongs to another namespace.
    # A sandbox may expose host procfs while Popen returns a container-local PID.
    try:
        own_proc_pid = int(Path("/proc/self/stat").read_text().split(" ", 1)[0])
    except (OSError, ValueError) as exc:
        raise ValueError("Cannot verify procfs process namespace") from exc
    if own_proc_pid != os.getpid():
        raise ValueError("Procfs and process PID namespaces differ")
    path = Path("/proc") / str(pid) / "stat"
    try:
        text = path.read_text()
        tail = text[text.rfind(")") + 2 :].split()
        return {
            "pid": pid,
            "uid": path.stat().st_uid,
            "starttime": tail[19],
            "state": tail[0],
        }
    except FileNotFoundError:
        return None


def record_process(runtime: Path, case_dir: Path, role: str, pid: int) -> None:
    """Register a local controller/native child; lifecycle owner calls under lock."""
    assert_runtime_owner(runtime, case_dir)
    if role not in {"controller", "native"} or type(pid) is not int or pid <= 0:
        raise ValueError("Invalid ICL child process identity")
    process = _process(pid)
    if process is None or process["uid"] != os.getuid():
        raise ValueError("ICL process is missing or belongs to another user")
    path = runtime / "PROCESS_RECEIPTS.json"
    if path.is_symlink():
        raise ValueError("Process receipt must not be linked")
    rows = _read(path) if path.exists() else []
    rows.append({"role": role, **process})
    _write(path, rows)


def _assert_stopped(runtime: Path) -> None:
    path = runtime / "PROCESS_RECEIPTS.json"
    if path.is_symlink():
        raise ValueError("Process receipt must not be linked")
    for recorded in _read(path) if path.exists() else []:
        current = _process(recorded["pid"])
        if (
            current is not None
            and current["starttime"] == recorded["starttime"]
            and current["state"] != "Z"
        ):
            raise ValueError("Registered ICL process is still running")


def _copy_regular(
    source: Path, destination: Path, rows: list[dict[str, Any]], relative: str
) -> None:
    before = source.stat(follow_symlinks=False)
    digest = hashlib.sha256()
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as reader, destination.open("xb") as writer:
        opened = os.fstat(reader.fileno())
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise ValueError("Runtime file changed before copying")
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            writer.write(chunk)
        after = os.fstat(reader.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("Runtime file changed while copying")
    shutil.copystat(source, destination, follow_symlinks=False)
    rows.append(
        {
            "path": relative,
            "kind": "file",
            "bytes": before.st_size,
            "sha256": digest.hexdigest(),
            "mode": stat.S_IMODE(before.st_mode),
        }
    )


def snapshot_runtime(runtime: Path, case_dir: Path) -> Path:
    """Publish stopped private records with SHA inventory; never delete runtime.

    Auth files and IPC nodes are explicitly excluded. Model-created links are
    saved without traversal. The original runs/<id>/workspace link is the only
    link materialized, and only when its resolved target is within this runtime.
    Existing records are preserved as previous numbered snapshots.

    Args:
        runtime: Owned short runtime returned by create_runtime.
        case_dir: Its evaluation case containing icl/runtime-reference.json.

    Returns:
        Published self-contained record directory under case/icl/records.
    """
    owner = assert_runtime_owner(runtime, case_dir)
    runtime = runtime.resolve()
    archive_parent = case_dir.resolve() / "icl"
    with runtime_lock(runtime, case_dir):
        _assert_stopped(runtime)
        staging = Path(tempfile.mkdtemp(prefix=".records-stage-", dir=archive_parent))
        staging.chmod(0o700)
        rows: list[dict[str, Any]] = []
        exclusions: list[dict[str, str]] = []

        def copy_tree(source: Path, target: Path, prefix: Path) -> None:
            for entry in sorted(source.iterdir()):
                relative = (prefix / entry.name).as_posix()
                if entry.name in EXCLUDED_NAMES:
                    exclusions.append(
                        {"path": relative, "reason": "authentication or lifecycle file"}
                    )
                    continue
                metadata = entry.lstat()
                destination = target / entry.name
                if stat.S_ISLNK(metadata.st_mode):
                    link = os.readlink(entry)
                    parts = (prefix / entry.name).parts
                    if (
                        len(parts) == 3
                        and parts[0] == "runs"
                        and parts[2] == "workspace"
                    ):
                        resolved = entry.resolve()
                        if (
                            not resolved.is_relative_to(runtime)
                            or not resolved.is_dir()
                            or entry.parent.is_relative_to(resolved)
                        ):
                            raise ValueError(
                                "Original workspace link escapes owned runtime"
                            )
                        destination.mkdir(mode=0o700)
                        copy_tree(resolved, destination, prefix / entry.name)
                        rows.append(
                            {
                                "path": relative,
                                "kind": "materialized_workspace",
                                "original_target": link,
                            }
                        )
                    else:
                        destination.symlink_to(link)
                        rows.append(
                            {"path": relative, "kind": "symlink", "target": link}
                        )
                elif stat.S_ISDIR(metadata.st_mode):
                    destination.mkdir(mode=stat.S_IMODE(metadata.st_mode))
                    copy_tree(entry, destination, prefix / entry.name)
                    shutil.copystat(entry, destination, follow_symlinks=False)
                    rows.append(
                        {
                            "path": relative,
                            "kind": "directory",
                            "mode": stat.S_IMODE(metadata.st_mode),
                        }
                    )
                elif stat.S_ISREG(metadata.st_mode):
                    _copy_regular(entry, destination, rows, relative)
                else:
                    exclusions.append(
                        {"path": relative, "reason": "IPC or special filesystem node"}
                    )

        try:
            copy_tree(runtime, staging, Path())
            _assert_stopped(runtime)
            _write(
                staging / "ARTIFACT_MANIFEST.json",
                {
                    "owner": owner,
                    "files": rows,
                    "exclusions": exclusions,
                    "original_runtime_retained": True,
                },
            )
            destination = archive_parent / "records"
            if destination.is_symlink():
                raise ValueError("Records destination must not be a symlink")
            if destination.exists():
                previous = archive_parent / f"records-previous-{uuid.uuid4().hex}"
                destination.rename(previous)
            staging.rename(destination)
            return destination
        except BaseException:
            # This is only our unpublished staging copy, never the live runtime.
            shutil.rmtree(staging)
            raise
