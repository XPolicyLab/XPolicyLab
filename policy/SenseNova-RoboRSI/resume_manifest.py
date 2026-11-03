"""Legacy checkpoint-receipt inspection (not an executable resume entry).

The live Model rejects ROBORSI_RESUME_MANIFEST. Use hybrid_resume.py and
resume_eval.py for physical same-layout recorded-action replay and validation.
These schema helpers remain available only for historical receipt inspection.

The normal policy path always starts a fresh episode.  A supervisor that wants
to restart a stopped process may opt in by creating a manifest with
``write_manifest`` and passing it to ``load_manifest`` in the new process.
This module deliberately does not perform a simulator reset or replay.  It
only validates that all state needed by the native evaluator and controller is
present and belongs to the same case.  The supervisor must create the
``native_restore_receipt`` *after* the official evaluator has restored the
native checkpoint; loading a manifest without that receipt is rejected.

The checks are intentionally strict: a malformed, incomplete, or foreign
manifest is rejected instead of silently starting from an incorrect step.
Paths are authenticated relative to the manifest's case root and no symlink
may occur anywhere in the retained state tree.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_ACTION_BYTES = 64 * 1024 * 1024
MAX_WORKSPACE_FILES = 100_000
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_TASK = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")


class ResumeManifestError(ValueError):
    """The retained state cannot be trusted for a same-episode resume."""


@dataclass(frozen=True)
class ResumeState:
    """Validated state handoff returned by :func:`load_manifest`."""

    manifest: Path
    case_root: Path
    task: str
    seed: int
    episode_id: str
    control_step: int
    step_limit: int
    source_trajectory: Path
    source_workspace: Path
    action_records: tuple[dict[str, Any], ...]
    session_id: str | None
    codex_home: Path | None
    elapsed_seconds: float
    native_checkpoint: Path
    native_restore_receipt: Path

    @property
    def steps_remaining(self) -> int:
        return self.step_limit - self.control_step


def _regular(path: Path, *, label: str) -> Path:
    """Resolve an existing regular path while rejecting symlink traversal."""

    path = Path(path)
    if not path.is_absolute():
        raise ResumeManifestError(f"{label} must be absolute")
    current = path
    while True:
        if current.is_symlink():
            raise ResumeManifestError(f"{label} cannot traverse symlinks")
        parent = current.parent
        if parent == current:
            break
        current = parent
    if not path.exists():
        raise ResumeManifestError(f"{label} is missing")
    return path


def _owned(path: Path, root: Path, *, label: str, must_exist: bool = True) -> Path:
    """Require a path to stay below ``root`` and contain no symlink component."""

    path = Path(path)
    root = Path(root)
    if not path.is_absolute():
        raise ResumeManifestError(f"{label} must be absolute")
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ResumeManifestError(f"{label} escapes the case root") from exc
    if not relative.parts:
        raise ResumeManifestError(f"{label} must not be the case root")
    current = root
    if current.is_symlink():
        raise ResumeManifestError("case root cannot be a symlink")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ResumeManifestError(f"{label} cannot traverse symlinks")
    if must_exist and not path.exists():
        raise ResumeManifestError(f"{label} is missing")
    return path


def _json_file(path: Path, *, label: str, max_bytes: int = MAX_MANIFEST_BYTES) -> Any:
    path = _regular(path, label=label)
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise ResumeManifestError(f"{label} must be a bounded regular file")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ResumeManifestError(f"{label} is not valid JSON") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_float(value: Any, *, label: str, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResumeManifestError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (maximum is not None and value > maximum):
        raise ResumeManifestError(f"{label} is outside its allowed range")
    return value


def _read_actions(path: Path, *, checkpoint: int) -> tuple[dict[str, Any], ...]:
    path = _regular(path, label="action record")
    if not path.is_file() or path.stat().st_size > MAX_ACTION_BYTES:
        raise ResumeManifestError("action record must be a bounded regular file")
    try:
        rows = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ResumeManifestError("action record cannot be read") from exc
    records: list[dict[str, Any]] = []
    expected = 1
    for line in rows:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ResumeManifestError("action record contains invalid JSON") from exc
        if not isinstance(row, dict) or row.get("status") != "completed":
            continue
        step = row.get("control_step")
        if type(step) is not int or step != expected:
            raise ResumeManifestError("completed action history is not contiguous")
        if "action" not in row or "applied_action" not in row:
            raise ResumeManifestError("completed action is missing applied_action")
        records.append(row)
        expected += 1
        if len(records) > checkpoint:
            raise ResumeManifestError("action history exceeds native checkpoint")
    if len(records) != checkpoint:
        raise ResumeManifestError("action history does not reach native checkpoint")
    return tuple(records)


def _check_workspace(path: Path) -> None:
    path = _regular(path, label="workspace")
    if not path.is_dir():
        raise ResumeManifestError("workspace must be a directory")
    count = 0
    for current, dirs, files in os.walk(path, topdown=True, followlinks=False):
        current_path = Path(current)
        if current_path.is_symlink():
            raise ResumeManifestError("workspace cannot contain symlink directories")
        for name in (*dirs, *files):
            item = current_path / name
            if item.is_symlink():
                raise ResumeManifestError("workspace cannot contain symlinks")
            count += 1
            if count > MAX_WORKSPACE_FILES:
                raise ResumeManifestError("workspace has too many files")


def _check_tree(path: Path, *, label: str) -> None:
    """Reject symlinked entries in a retained evidence directory."""

    path = _regular(path, label=label)
    if not path.is_dir():
        raise ResumeManifestError(f"{label} must be a directory")
    count = 0
    for current, dirs, files in os.walk(path, topdown=True, followlinks=False):
        current_path = Path(current)
        if current_path.is_symlink():
            raise ResumeManifestError(f"{label} cannot contain symlink directories")
        if any((current_path / name).is_symlink() for name in (*dirs, *files)):
            raise ResumeManifestError(f"{label} cannot contain symlinks")
        count += len(dirs) + len(files)
        if count > MAX_WORKSPACE_FILES:
            raise ResumeManifestError(f"{label} has too many files")


def _path_value(mapping: dict[str, Any], key: str, *, label: str) -> Path:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ResumeManifestError(f"{label} path is missing")
    return Path(value)


def _digest_matches(path: Path, expected: Any, *, label: str) -> None:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ResumeManifestError(f"{label} hash is invalid")
    if _sha256(path) != expected:
        raise ResumeManifestError(f"{label} hash does not match")


def load_manifest(
    path: str | os.PathLike[str],
    *,
    expected_task: str | None = None,
    expected_seed: int | None = None,
    expected_step_limit: int | None = None,
) -> ResumeState:
    """Load and validate a supervisor-produced same-episode resume manifest.

    ``expected_*`` values should come from the new evaluator invocation.  They
    are mandatory for callers that resume a live task; omitting them is useful
    only for offline inspection.  No state is modified by this function.
    """

    manifest = _regular(Path(path).absolute(), label="resume manifest")
    if not manifest.is_file() or manifest.stat().st_size > MAX_MANIFEST_BYTES:
        raise ResumeManifestError("resume manifest must be a bounded regular file")
    document = _json_file(manifest, label="resume manifest")
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise ResumeManifestError("unsupported resume manifest schema")
    case_root = _regular(Path(document.get("case_root", "")), label="case root")
    if not case_root.is_dir():
        raise ResumeManifestError("case root must be a directory")
    if manifest.parent != case_root and not manifest.is_relative_to(case_root):
        raise ResumeManifestError("manifest must be inside its case root")
    task = document.get("task")
    if not isinstance(task, str) or not _TASK.fullmatch(task):
        raise ResumeManifestError("manifest task is invalid")
    seed = document.get("seed")
    if type(seed) is not int or seed < 0:
        raise ResumeManifestError("manifest seed is invalid")
    episode_id = document.get("episode_id")
    if not isinstance(episode_id, str) or not _SAFE_ID.fullmatch(episode_id):
        raise ResumeManifestError("manifest episode_id is invalid")
    if expected_task is not None and task != expected_task:
        raise ResumeManifestError("resume task differs from requested task")
    if expected_seed is not None and seed != expected_seed:
        raise ResumeManifestError("resume seed differs from requested seed")
    limit = document.get("step_limit")
    step = document.get("control_step")
    if type(limit) is not int or limit <= 0 or type(step) is not int or not 0 < step <= limit:
        raise ResumeManifestError("native checkpoint step is invalid")
    if expected_step_limit is not None and limit != expected_step_limit:
        raise ResumeManifestError("resume step limit differs from requested limit")

    paths = document.get("paths")
    if not isinstance(paths, dict):
        raise ResumeManifestError("manifest paths are missing")
    trajectory = _owned(_path_value(paths, "trajectory", label="trajectory"), case_root, label="trajectory")
    workspace = _owned(_path_value(paths, "workspace", label="workspace"), case_root, label="workspace")
    actions = _owned(_path_value(paths, "actions", label="action record"), case_root, label="action record")
    checkpoint = _owned(_path_value(paths, "native_checkpoint", label="native checkpoint"), case_root, label="native checkpoint")
    restore_receipt = _owned(
        _path_value(paths, "native_restore_receipt", label="native restore receipt"),
        case_root,
        label="native restore receipt",
    )
    if not trajectory.is_dir() or not checkpoint.is_file():
        raise ResumeManifestError("trajectory/checkpoint types are invalid")
    restore = _json_file(restore_receipt, label="native restore receipt")
    if (
        not isinstance(restore, dict)
        or restore.get("verified") is not True
        or restore.get("task") != task
        or type(restore.get("seed")) is not int
        or restore.get("seed") != seed
        or restore.get("episode_id") != episode_id
        or type(restore.get("control_step")) is not int
        or restore.get("control_step") != step
    ):
        raise ResumeManifestError("native restore receipt does not match checkpoint")
    _check_tree(trajectory, label="trajectory")
    _check_workspace(workspace)
    records = _read_actions(actions, checkpoint=step)
    _digest_matches(actions, document.get("action_sha256"), label="action record")
    _digest_matches(checkpoint, document.get("native_checkpoint_sha256"), label="native checkpoint")
    _digest_matches(
        restore_receipt,
        document.get("native_restore_receipt_sha256"),
        label="native restore receipt",
    )
    session_id = document.get("session_id")
    if session_id is not None and (not isinstance(session_id, str) or not _SAFE_ID.fullmatch(session_id)):
        raise ResumeManifestError("session_id is invalid")
    codex_home = None
    if paths.get("codex_home") is not None:
        codex_home = _owned(_path_value(paths, "codex_home", label="Codex home"), case_root, label="Codex home")
        _check_workspace(codex_home)
        if any(item.name == "auth.json" for item in codex_home.rglob("*")):
            raise ResumeManifestError("Codex home must not contain auth.json")
    if session_id is not None and codex_home is None:
        raise ResumeManifestError("session_id requires its retained Codex home")
    elapsed = _safe_float(document.get("elapsed_seconds", 0), label="elapsed_seconds")
    if elapsed > _safe_float(document.get("episode_seconds", elapsed), label="episode_seconds"):
        raise ResumeManifestError("elapsed_seconds exceeds episode_seconds")
    return ResumeState(
        manifest=manifest,
        case_root=case_root,
        task=task,
        seed=seed,
        episode_id=episode_id,
        control_step=step,
        step_limit=limit,
        source_trajectory=trajectory,
        source_workspace=workspace,
        action_records=records,
        session_id=session_id,
        codex_home=codex_home,
        elapsed_seconds=elapsed,
        native_checkpoint=checkpoint,
        native_restore_receipt=restore_receipt,
    )


def write_manifest(path: str | os.PathLike[str], document: dict[str, Any]) -> Path:
    """Atomically write a bounded manifest after basic schema validation.

    Full filesystem and action validation happens on the subsequent
    ``load_manifest`` call, allowing callers to finish writing a checkpoint
    before publishing the manifest.  The destination must be a regular file
    below an existing case root and is never followed through a symlink.
    """

    destination = Path(path).absolute()
    if destination.exists() and destination.is_symlink():
        raise ResumeManifestError("manifest destination cannot be a symlink")
    if destination.parent.exists():
        _regular(destination.parent, label="manifest destination parent")
    else:
        current = destination.parent
        missing = []
        while not current.exists():
            missing.append(current)
            parent = current.parent
            if parent == current:
                raise ResumeManifestError("manifest destination parent is unavailable")
            current = parent
        _regular(current, label="manifest destination parent")
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise ResumeManifestError("unsupported resume manifest schema")
    payload = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if len(payload.encode("utf-8")) > MAX_MANIFEST_BYTES:
        raise ResumeManifestError("resume manifest exceeds size limit")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


__all__ = ["ResumeManifestError", "ResumeState", "SCHEMA_VERSION", "load_manifest", "write_manifest"]
