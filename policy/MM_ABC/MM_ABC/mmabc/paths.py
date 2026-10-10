"""Resolve configs relative to this source tree and assets relative to the adapter.

MMABC_HOME overrides the asset root. Absolute paths are preserved."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HOME = Path(os.environ.get("MMABC_HOME", REPO_ROOT.parent)).resolve()


def home_path(path: str | os.PathLike) -> Path:
    """Resolve an asset path (pretrained weights, data, checkpoints) against HOME."""
    p = Path(os.path.expanduser(str(path)))
    return p if p.is_absolute() else HOME / p


def repo_path(path: str | os.PathLike) -> Path:
    """Resolve a config path against the source tree."""
    p = Path(os.path.expanduser(str(path)))
    return p if p.is_absolute() else REPO_ROOT / p


def relative_to_home(path: str | os.PathLike) -> str:
    """Write a path under HOME relative to it, so generated configs stay portable."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(HOME))
    except ValueError:
        return str(p)
