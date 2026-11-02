"""Freeze skills + lessons into frozen/{skills,lessons} with a content manifest, and verify it.

manifest.json: schema, source run, shared-library version and per-file sha256 of the staged copy, the
sha256 of every frozen file, and one digest over all of them. Tests call verify() before and after each
episode: frozen knowledge that changed (edited, added or removed files) is rejected.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Optional, Union

from knowin_agent.workspace.files import (copy_tree, read_json, regular_files, set_read_only, tree_hashes,
                                  write_json)
from knowin_agent.workspace.knowledge import SHARED_REL

MANIFEST = "manifest.json"
SCHEMA = "knowin_agent-frozen-knowledge/1"
COMPONENTS = ("skills", "lessons")


class FrozenChanged(RuntimeError):
    def __init__(self, frozen_dir: Path, diff: dict):
        self.frozen_dir, self.diff = Path(frozen_dir), diff
        parts = [f"{k}: {', '.join(v[:5])}{' …' if len(v) > 5 else ''}" for k, v in diff.items() if v]
        super().__init__(f"frozen knowledge changed in {frozen_dir} ({'; '.join(parts)})")


def digest(files: dict[str, str]) -> str:
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(f"{rel}\0{files[rel]}\n".encode())
    return h.hexdigest()


def frozen_files(frozen_dir: Union[str, Path]) -> dict[str, str]:
    frozen_dir = Path(frozen_dir)
    files = {}
    for component in COMPONENTS:
        files.update(tree_hashes(frozen_dir / component, prefix=f"{component}/", exclude=("__pycache__",)))
    return files


def freeze_dirs(frozen_dir: Union[str, Path], *, skills: Optional[Path], lessons: Optional[Path],
                shared: Optional[Path] = None, source_run: str, library_version: Optional[str] = None,
                extra: Optional[dict] = None) -> Path:
    """Copy the given directories into frozen_dir and write the manifest; return the manifest path.

    `skills/shared/` inside `skills` is never copied from there: the staged copy comes from `shared`.
    """
    frozen_dir = Path(frozen_dir)
    if frozen_dir.exists() and any(frozen_dir.iterdir()):
        raise ValueError(f"frozen dir is not empty: {frozen_dir}")
    for component, source in (("skills", skills), ("lessons", lessons)):
        (frozen_dir / component).mkdir(parents=True, exist_ok=True)
        if source is not None and Path(source).is_dir():
            exclude = ("shared", "__pycache__") if component == "skills" else ("__pycache__",)
            regular_files(source)
            copy_tree(source, frozen_dir / component, exclude=exclude)
    shared_files = copy_tree(shared, frozen_dir / SHARED_REL) if shared is not None and Path(shared).is_dir() else {}
    files = frozen_files(frozen_dir)
    manifest = {
        "schema": SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_run": source_run,
        "shared_library": {"version": library_version, "files": shared_files} if shared_files else None,
        "files": files,
        "digest": digest(files),
        **(extra or {}),
    }
    path = frozen_dir / MANIFEST
    write_json(path, manifest)
    set_read_only(frozen_dir)
    return path


def freeze(workspace, frozen_dir: Union[str, Path], *, source_run: str,
           library_version: Optional[str] = None, extra: Optional[dict] = None) -> Path:
    """Freeze a workspace (a Workspace or its root path) after practice."""
    root = Path(getattr(workspace, "root", workspace))
    receipt = getattr(workspace, "receipt", None) or read_json(root.with_name(root.name + ".json"), {}) or {}
    library = receipt.get("shared_library") or {}
    version = library_version or library.get("version")
    shared = root / SHARED_REL
    payload = dict(extra or {})
    if library.get("files") is not None and shared.is_dir():
        payload["shared_library_intact"] = tree_hashes(shared) == library["files"]
    return freeze_dirs(frozen_dir, skills=root / "skills", lessons=root / "lessons",
                       shared=shared if shared.is_dir() else None, source_run=source_run,
                       library_version=version, extra=payload)


def load_manifest(frozen_dir: Union[str, Path]) -> dict:
    manifest = read_json(Path(frozen_dir) / MANIFEST)
    if not manifest or manifest.get("schema") != SCHEMA:
        raise ValueError(f"not a frozen knowledge dir (missing or foreign {MANIFEST}): {frozen_dir}")
    return manifest


def verify(frozen_dir: Union[str, Path]) -> dict:
    """Return the manifest if every frozen file matches it; raise FrozenChanged otherwise."""
    manifest = load_manifest(frozen_dir)
    expected = manifest.get("files") or {}
    actual = frozen_files(frozen_dir)
    diff = {"changed": sorted(k for k in expected if k in actual and actual[k] != expected[k]),
            "missing": sorted(k for k in expected if k not in actual),
            "added": sorted(k for k in actual if k not in expected)}
    if any(diff.values()) or digest(expected) != manifest.get("digest"):
        diff.setdefault("manifest", [])
        if digest(expected) != manifest.get("digest"):
            diff["manifest"] = ["digest mismatch"]
        raise FrozenChanged(frozen_dir, diff)
    return manifest
