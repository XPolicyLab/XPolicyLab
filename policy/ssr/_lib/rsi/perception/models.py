from __future__ import annotations

from .. import ASSETS

import hashlib
import json
import os
from pathlib import Path

REGISTRY = ASSETS / "models.json"
_checked: dict[str, Path] = {}
_root: Path | None = None


class ModelError(RuntimeError):
    pass


def set_root(p: os.PathLike | str | None) -> None:
    global _root
    _root = Path(p) if p is not None else None
    _checked.clear()


def root() -> Path:
    if _root is not None:
        return _root
    raise ModelError(
        "no checkpoint directory: the weights belong in policy/ssr/checkpoints/<ckpt_name>/ (Git LFS: git lfs pull)"
    )


def verify_all() -> dict[str, Path]:
    return {name: path(name) for name in entries()}


def entries() -> dict:
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def path(name: str) -> Path:
    if name in _checked:
        return _checked[name]
    e = entries().get(name)
    if not e:
        raise ModelError(f"model {name!r} is not registered in {REGISTRY}")
    p = root() / name / str(e["version"]) / e["file"]
    if not p.is_file():
        raise ModelError(f"model {name!r} version {e['version']}: {p} is missing")
    got = sha256(p)
    if got != e["sha256"]:
        raise ModelError(f"model {name!r} version {e['version']}: sha256 {got} != registered {e['sha256']}")
    _checked[name] = p
    return p
