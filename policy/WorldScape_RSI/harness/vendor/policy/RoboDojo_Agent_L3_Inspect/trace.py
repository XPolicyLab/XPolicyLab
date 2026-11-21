"""Structured trace helpers for RoboDojo L3 Inspect episodes."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "l3-inspect-trace/v1"

_CAMERA_NAMES = {
    "cam_head": "head",
    "cam_left_wrist": "left_wrist",
    "cam_right_wrist": "right_wrist",
}

#: What a source file has to be named to count towards the revision hash.
_SOURCE_SUFFIXES = frozenset({".py", ".yml", ".sh"})


@lru_cache(maxsize=None)
def source_revision(*extra_dirs: Path) -> dict[str, str | None]:
    """Which code produced this episode, for a transcript that outlives it.

    Two answers, because neither is enough alone. The commit is what a reader
    can go and look at; the content hash is what is actually true, since the
    runs worth keeping are made on a checkout with uncommitted work in it.

    ``extra_dirs`` is how a specialised adapter adds its own package: this
    module's directory is always hashed, so the base adapter is covered
    whichever surface is running.

    Cached because the tree cannot change under a running episode, while the
    transcript is rewritten on every step.
    """
    here = Path(__file__).resolve()
    checkout = here.parents[2]
    directories = [here.parent, *(Path(d).resolve() for d in extra_dirs)]
    return {
        "commit": _git_commit(checkout),
        "adapter_sha1": _source_sha1(checkout, directories),
    }


def _source_sha1(checkout: Path, directories: list[Path]) -> str | None:
    """Hash the adapter sources, keyed by path relative to the checkout.

    Relative, so two machines running the same code agree. The sweep's own arm
    fingerprint does not: it hashes ``sha1sum`` output, which carries absolute
    paths, so it answers "is this the same checkout" rather than "is this the
    same code" and the two numbers are not comparable.
    """
    paths = sorted(
        {
            path
            for directory in directories
            for path in directory.rglob("*")
            if path.suffix in _SOURCE_SUFFIXES and path.is_file()
        }
    )
    if not paths:
        return None
    digest = hashlib.sha1()
    for path in paths:
        try:
            body = path.read_bytes()
        except OSError:
            continue
        digest.update(str(path.relative_to(checkout)).encode("utf-8"))
        digest.update(hashlib.sha1(body).digest())
    return digest.hexdigest()


def _git_commit(checkout: Path) -> str | None:
    """The checked-out commit, read out of ``.git`` rather than by running git.

    An adapter runs inside the simulator's environment, which is not promised a
    git binary, and a subprocess per transcript write would not be free either.
    """
    git = checkout / ".git"
    if git.is_file():
        # A linked worktree: .git is a pointer file, and the refs it names live
        # in the repository it was linked from, not beside its own HEAD.
        pointer = git.read_text(encoding="utf-8").partition("gitdir:")[2].strip()
        if not pointer:
            return None
        git = Path(pointer)
    head = _read_text(git / "HEAD")
    if head is None:
        return None
    if not head.startswith("ref:"):
        return head or None
    ref = head.partition("ref:")[2].strip()
    common = _read_text(git / "commondir")
    bases = [git]
    if common:
        bases.append((git / common).resolve())
    for base in bases:
        loose = _read_text(base / ref)
        if loose:
            return loose
    for base in bases:
        packed = _read_text(base / "packed-refs") or ""
        for line in packed.splitlines():
            commit, _, name = line.partition(" ")
            if name.strip() == ref:
                return commit
    return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def video_frame_counts(task_env: Any, env_idx: int = 0) -> dict[str, int]:
    """Read encoded frame counts without causing a new observation."""
    writers_by_env = getattr(task_env, "video_writers", {})
    writers = (
        writers_by_env.get(env_idx, {})
        if isinstance(writers_by_env, Mapping)
        else {}
    )
    return {
        _CAMERA_NAMES.get(str(camera), str(camera).removeprefix("cam_")): int(
            writer.n_frames
        )
        for camera, writer in writers.items()
        if hasattr(writer, "n_frames")
    }


def env_step(task_env: Any, env_idx: int = 0) -> int | None:
    """Read RoboDojo's executed-action counter for one environment."""
    counts = getattr(task_env, "take_action_cnt", None)
    if counts is None:
        return None
    try:
        if isinstance(counts, (list, tuple, np.ndarray)):
            return int(np.asarray(counts)[env_idx])
        return int(counts)
    except (IndexError, TypeError, ValueError):
        return None


def frame_ranges(
    frame_start: Mapping[str, int],
    frame_end: Mapping[str, int],
    *,
    include_frame: bool = False,
) -> dict[str, dict[str, int | None]]:
    """Build half-open per-camera ranges from two writer snapshots."""
    ranges: dict[str, dict[str, int | None]] = {}
    for camera in sorted(set(frame_start) | set(frame_end)):
        start = int(frame_start.get(camera, 0))
        end = max(start, int(frame_end.get(camera, start)))
        bounds: dict[str, int | None] = {"start": start, "end": end}
        if include_frame:
            bounds["frame"] = end - 1 if end > start else None
        ranges[camera] = bounds
    return ranges


def named_state(labels: tuple[str, ...], state: Mapping[str, Any]) -> dict[str, float]:
    """Serialize the measured joint state in model-facing label order."""
    values: list[float] = []
    for channel, size in (
        ("left_arm_joint_state", 6),
        ("left_ee_joint_state", 1),
        ("right_arm_joint_state", 6),
        ("right_ee_joint_state", 1),
    ):
        channel_values = np.asarray(state.get(channel, np.zeros(size))).reshape(-1)
        if channel_values.size < size:
            channel_values = np.pad(channel_values, (0, size - channel_values.size))
        values.extend(float(value) for value in channel_values[:size])
    return dict(zip(labels, values, strict=True))


def measured_flange_poses(state: Mapping[str, Any]) -> dict[str, list[float]]:
    """Record the world-frame flange poses a Cartesian rollout was driven from.

    Without these the trace cannot say where either hand actually was, which
    leaves a Cartesian episode un-auditable from the log alone.
    """
    poses: dict[str, list[float]] = {}
    for arm in ("left", "right"):
        raw = np.asarray(state.get(f"{arm}_ee_pose", []), dtype=np.float64).reshape(-1)
        if raw.size < 7 or not np.isfinite(raw[:7]).all():
            continue
        poses[f"{arm}_ee_pose"] = [float(value) for value in raw[:7]]
    return poses
