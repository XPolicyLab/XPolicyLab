"""Phase permissions of a workspace.

practice: skills/ (except skills/shared/), lessons/ and scratch/ are writable.
test:     only scratch/ is writable; skills/ and lessons/ are also made read-only on disk.
Always read-only: instruction.md and skills/shared/.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Union

from knowin_agent.workspace.files import set_read_only, set_writable

PHASES = ("practice", "test")
WRITABLE = {"practice": ("skills", "lessons", "scratch"), "test": ("scratch",)}


def check_phase(phase: str) -> str:
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, not {phase!r}")
    return phase


def relative_path(workspace: Union[str, Path], rel_path: Union[str, Path]) -> str:
    """Normalise a path given relative to (or absolute inside) the workspace; reject escapes."""
    root = Path(workspace).resolve()
    raw = str(rel_path)
    if "\\" in raw or "\0" in raw:
        raise PermissionError(f"invalid path: {raw!r}")
    path = Path(raw)
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(root)
        except ValueError:
            raise PermissionError(f"path escapes the workspace: {raw}") from None
    parts = PurePosixPath(path.as_posix()).parts
    if not parts or any(p in ("..", "") for p in parts):
        raise PermissionError(f"path escapes the workspace: {raw}")
    parts = tuple(p for p in parts if p != ".")
    if not parts:
        raise PermissionError(f"not a file path: {raw}")
    resolved = (root / Path(*parts)).resolve()
    if resolved != root and root not in resolved.parents:
        raise PermissionError(f"path escapes the workspace through a link: {raw}")
    return PurePosixPath(*parts).as_posix()


def is_writable(rel: str, phase: str) -> bool:
    parts = PurePosixPath(rel).parts
    if len(parts) < 2:
        return False
    if parts[0] == "skills" and parts[1] == "shared":
        return False
    return parts[0] in WRITABLE[check_phase(phase)]


def assert_writable(workspace: Union[str, Path], rel_path: Union[str, Path], phase: str) -> Path:
    """Raise PermissionError unless the agent may write rel_path in this phase; return the absolute path."""
    rel = relative_path(workspace, rel_path)
    if not is_writable(rel, phase):
        allowed = ", ".join(f"{d}/" for d in WRITABLE[check_phase(phase)])
        raise PermissionError(f"{rel} is read-only in {phase} (writable: {allowed}; never skills/shared/)")
    return Path(workspace) / rel


def apply_permissions(workspace: Union[str, Path], phase: str) -> None:
    """Mirror the phase rules on disk (best effort against a coding agent writing files directly)."""
    root = Path(workspace)
    check_phase(phase)
    for name in ("skills", "lessons", "scratch"):
        (root / name).mkdir(parents=True, exist_ok=True)
    if (root / "instruction.md").exists():
        set_read_only(root / "instruction.md")
    if phase == "test":
        set_read_only(root / "skills")
        set_read_only(root / "lessons")
    else:
        for name in ("skills", "lessons"):
            set_writable(root / name)
            if (root / name / "shared").exists() and name == "skills":
                set_read_only(root / "skills" / "shared")
    set_writable(root / "scratch")
