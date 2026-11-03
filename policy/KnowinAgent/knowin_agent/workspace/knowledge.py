"""Stage the unversioned packaged library, or an explicitly configured external training registry.

The preview ships no learned library. Explicit external registries remain supported.
Training registries retain the existing <root>/<version>/library layout, outside the installed package.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional, Union

from knowin_agent.workspace.files import (copy_tree, read_json, regular_files, remove_tree, set_read_only, sha256_file,
                                  tree_hashes, write_json)

LIBRARY_MAX_FILES = 40
LIBRARY_MAX_BYTES = 400_000
LIBRARY_SUFFIXES = (".py", ".md")
VERSIONS_FILE = "VERSIONS.json"
VERSIONS_SCHEMA = "knowin_agent-shared-library-versions/1"
SHARED_REL = "skills/shared"
PACKAGED = "packaged"


def packaged_root() -> Path:
    return Path(__file__).resolve().parents[1] / "library" / "shared"


def is_packaged_root(root: Path) -> bool:
    return Path(root).resolve() == packaged_root().resolve()


def library_root(root: Optional[Union[str, Path]] = None) -> Path:
    """`root`, else $KNOWIN_AGENT_SHARED_LIBRARY_ROOT, else the library shipped in the package."""
    if root:
        return Path(root)
    env = os.environ.get("KNOWIN_AGENT_SHARED_LIBRARY_ROOT")
    if env:
        return Path(env)
    return packaged_root()


def library_dir(version: Union[str, Path], root: Optional[Union[str, Path]] = None) -> Path:
    """The flat file directory of a version name, a version directory, or a library directory."""
    path = Path(version)
    if str(version) == PACKAGED:
        return packaged_root()
    if path.is_absolute() or len(path.parts) > 1:
        return path / "library" if (path / "library").is_dir() else path
    return library_root(root) / str(version) / "library"


def check_library(directory: Path) -> list[Path]:
    """Validate a library directory against the staging limits; return its files."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"shared library not found: {directory}")
    cache = directory / "__pycache__"
    if cache.exists() or cache.is_symlink():
        if cache.is_symlink() or not cache.is_dir() or any(
                p.is_symlink() or not p.is_file() or p.suffix != ".pyc" for p in cache.iterdir()):
            raise ValueError("shared-library __pycache__ may contain only regular .pyc files")
    entries = sorted(p for p in directory.iterdir() if p.name != "__pycache__")
    if any(p.is_symlink() or not p.is_file() for p in entries):
        raise ValueError("a shared library is one flat level of regular files (no links, no subfolders)")
    bad = [p.name for p in entries if p.suffix not in LIBRARY_SUFFIXES]
    if bad:
        raise ValueError(f"a shared library may contain only .py and .md files: {bad}")
    if not entries:
        raise ValueError("shared library is empty")
    size = sum(p.stat().st_size for p in entries)
    if len(entries) > LIBRARY_MAX_FILES or size > LIBRARY_MAX_BYTES:
        raise ValueError(f"shared library too large: {len(entries)} files / {size} bytes "
                         f"(limits {LIBRARY_MAX_FILES} files / {LIBRARY_MAX_BYTES} bytes)")
    return entries


def stage_library(workspace_root: Path, version: Union[str, Path], *,
                  root: Optional[Union[str, Path]] = None) -> dict:
    """Copy a library version into workspace/skills/shared/ and make it read-only."""
    source = library_dir(version, root)
    check_library(source)
    if is_packaged_root(source):
        expected = load_versions(packaged_root())["versions"][PACKAGED]["files"]
        if tree_hashes(source, exclude=("__pycache__",)) != expected:
            raise ValueError("packaged shared-library files do not match shared_manifest.json")
    target = Path(workspace_root) / SHARED_REL
    if target.exists():
        raise ValueError(f"{SHARED_REL} already exists in this workspace")
    files = copy_tree(source, target, exclude=("__pycache__",))
    set_read_only(target)
    name = str(version) if not Path(str(version)).is_absolute() else source.parent.name
    return {"version": name, "source": str(source), "files": files}


def seed_knowledge(workspace_root: Path, frozen_dir: Path) -> dict:
    """Replace workspace skills/ (except skills/shared/) and lessons/ with a frozen knowledge copy.

    The seed is the agent's own knowledge to keep revising, so the copies are writable.
    """
    frozen_dir = Path(frozen_dir)
    receipt = {}
    for component in ("skills", "lessons"):
        source = frozen_dir / component
        if not source.is_dir():
            raise ValueError(f"seed knowledge lacks {component}/: {frozen_dir}")
        regular_files(source)
        target = Path(workspace_root) / component
        if target.exists():
            for child in sorted(target.iterdir()):
                if component == "skills" and child.name == "shared":
                    continue
                remove_tree(child) if child.is_dir() else child.unlink()
        exclude = ("shared",) if component == "skills" else ()
        for rel, digest in copy_tree(source, target, exclude=exclude).items():
            receipt[f"{component}/{rel}"] = digest
    return {"source": str(frozen_dir), "files": receipt}


def restore_shared(workspace_root: Path, frozen_dir: Path) -> Optional[dict]:
    """Stage the exact skills/shared/ copy a frozen knowledge dir was trained with, read-only."""
    source = Path(frozen_dir) / SHARED_REL
    if not source.is_dir():
        manifest = read_json(Path(frozen_dir) / "manifest.json", {}) or {}
        library = manifest.get("shared_library") or {}
        if library.get("version") == PACKAGED:
            entry = load_versions(packaged_root())["versions"].get(PACKAGED)
            if entry is None:
                raise ValueError("preview has no packaged shared library; provide self-contained frozen knowledge")
            if library.get("files") != entry["files"]:
                raise ValueError("frozen knowledge references a different packaged shared library")
            return {**stage_library(workspace_root, PACKAGED), "from_frozen": True}
        return None
    target = Path(workspace_root) / SHARED_REL
    if target.exists():
        raise ValueError(f"{SHARED_REL} already exists in this workspace")
    files = copy_tree(source, target)
    set_read_only(target)
    manifest = read_json(Path(frozen_dir) / "manifest.json", {}) or {}
    version = (manifest.get("shared_library") or {}).get("version")
    return {"version": version, "source": str(source), "files": files, "from_frozen": True}


# ---------------- VERSIONS.json ----------------
def load_versions(root: Optional[Union[str, Path]] = None) -> dict:
    if is_packaged_root(library_root(root)):
        manifest = read_json(packaged_root().parent / "shared_manifest.json", None)
        if manifest is None:
            return {"schema": VERSIONS_SCHEMA, "versions": {}}
        return {"schema": VERSIONS_SCHEMA, "versions": {
            PACKAGED: {"status": "published", "files": manifest["files"], "digest": manifest["digest"]}
        }}
    data = read_json(library_root(root) / VERSIONS_FILE, None)
    return data if data else {"schema": VERSIONS_SCHEMA, "versions": {}}


def save_versions(index: dict, root: Optional[Union[str, Path]] = None) -> Path:
    if is_packaged_root(library_root(root)):
        raise ValueError("the packaged shared library is immutable; choose an external training registry")
    path = library_root(root) / VERSIONS_FILE
    write_json(path, index)
    return path


def version_entry(version_dir: Path, status: str, **meta) -> dict:
    if status not in ("published", "candidate"):
        raise ValueError("status must be 'published' or 'candidate'")
    version_dir = Path(version_dir)
    entry = {"status": status, **meta, "files": tree_hashes(version_dir / "library", exclude=("__pycache__",))}
    if (version_dir / "tests").is_dir():
        entry["tests"] = tree_hashes(version_dir / "tests", exclude=("__pycache__",))
    if (version_dir / "CHANGES.md").is_file():
        entry["changes_sha256"] = sha256_file(version_dir / "CHANGES.md")
    return entry


def register_version(name: str, status: str, *, root: Optional[Union[str, Path]] = None, **meta) -> dict:
    """Add or refresh one version's entry (hashes recomputed from disk)."""
    base = library_root(root)
    if is_packaged_root(base):
        raise ValueError("the packaged shared library is immutable; choose an external training registry")
    index = load_versions(base)
    previous = index["versions"].get(name) or {}
    entry = version_entry(base / name, status, **{**{k: v for k, v in previous.items()
                                                     if k not in ("status", "files", "tests", "changes_sha256")},
                                                  **meta})
    entry.setdefault("registered_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    index["versions"][name] = entry
    save_versions(index, base)
    return entry


def verify_versions(root: Optional[Union[str, Path]] = None) -> dict[str, list[str]]:
    """Library files whose bytes differ from VERSIONS.json, per version (empty dict = all match)."""
    base = library_root(root)
    problems = {}
    for name, entry in load_versions(base)["versions"].items():
        directory = library_dir(name, base)
        actual = tree_hashes(directory, exclude=("__pycache__",)) if directory.is_dir() else {}
        expected = entry.get("files") or {}
        diff = sorted(k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k))
        if diff:
            problems[name] = diff
    return problems


def version_number(name: str) -> Optional[int]:
    match = re.fullmatch(r"v(\d+)", str(name))
    return int(match.group(1)) if match else None


def published_versions(root: Optional[Union[str, Path]] = None) -> list[str]:
    if is_packaged_root(library_root(root)):
        return [PACKAGED] if PACKAGED in load_versions(root)["versions"] else []
    versions = load_versions(root)["versions"]
    names = [n for n, e in versions.items() if e.get("status") == "published" and version_number(n) is not None]
    return sorted(names, key=version_number)


def latest_published(root: Optional[Union[str, Path]] = None) -> Optional[str]:
    names = published_versions(root)
    return names[-1] if names else None
