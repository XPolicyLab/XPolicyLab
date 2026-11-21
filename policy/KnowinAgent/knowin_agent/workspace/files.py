"""File-tree helpers shared by staging, freezing and review: hashing, link-free copies, read-only marking."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Iterable, Optional


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def regular_files(root: Path, *, exclude: Iterable[str] = ()) -> list[Path]:
    """Every regular file under root (sorted). Symlinks anywhere are an error, never followed."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError(f"symlinked tree is not allowed: {root}")
    if not root.is_dir():
        return []
    skip = {Path(p).as_posix() for p in exclude}
    out = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"symlinks are not allowed: {path}")
        if any(rel == s or rel.startswith(s + "/") for s in skip):
            continue
        if path.is_file():
            out.append(path)
    return out


def tree_hashes(root: Path, *, prefix: str = "", exclude: Iterable[str] = ()) -> dict[str, str]:
    root = Path(root)
    return {prefix + p.relative_to(root).as_posix(): sha256_file(p) for p in regular_files(root, exclude=exclude)}


def tree_size(root: Path) -> tuple[int, int]:
    files = regular_files(root)
    return len(files), sum(p.stat().st_size for p in files)


def copy_tree(src: Path, dst: Path, *, exclude: Iterable[str] = (), max_files: Optional[int] = None) -> dict[str, str]:
    """Copy regular files (bytes only, no permissions) and return their hashes keyed by relative path."""
    src, dst = Path(src), Path(dst)
    receipt = {}
    for path in regular_files(src, exclude=exclude):
        if max_files is not None and len(receipt) >= max_files:
            break
        rel = path.relative_to(src)
        data = path.read_bytes()
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        receipt[rel.as_posix()] = sha256_bytes(data)
    return receipt


def set_read_only(root: Path) -> None:
    """Strip write bits from every file and directory under root (root included)."""
    root = Path(root)
    if not root.exists():
        return
    paths = [root] + list(root.rglob("*")) if root.is_dir() else [root]
    for path in sorted(paths, key=lambda p: len(p.parts), reverse=True):
        if path.is_symlink():
            continue
        mode = path.stat().st_mode
        path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def set_writable(root: Path) -> None:
    root = Path(root)
    if not root.exists():
        return
    root.chmod(root.stat().st_mode | stat.S_IWUSR)
    if root.is_dir():
        for path in root.rglob("*"):
            if not path.is_symlink():
                path.chmod(path.stat().st_mode | stat.S_IWUSR)


def remove_tree(root: Path) -> None:
    root = Path(root)
    if root.exists():
        set_writable(root)
        shutil.rmtree(root)


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
