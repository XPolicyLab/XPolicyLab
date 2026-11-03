"""RoboDojo's prerecorded evaluation layouts addressed by one integer seed.

A harness seed is ``bank * 1_000_000 + ordinal``. ``bank`` is the native eval
seed (the layout directory 0, 1 or 2) and ``ordinal`` is the native layout id:
the index into that directory's ``<task>_<n>.json`` files sorted by numeric n,
which is how the native SeedManager numbers them.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Union

SEED_STRIDE = 1_000_000
BANKS = (0, 1, 2)
ROOT_ENV = "KNOWIN_AGENT_ROBODOJO_ROOT"
LAYOUT_SUBDIR = Path("Assets", "Eval_Layout", "RoboDojo", "arx_x5")

_TASK_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")

PathLike = Union[str, os.PathLike]


def resolve_root(root: Optional[PathLike] = None) -> Path:
    """The RoboDojo checkout: the explicit argument, else $KNOWIN_AGENT_ROBODOJO_ROOT."""
    value = root if root is not None else os.environ.get(ROOT_ENV)
    if not value:
        raise RuntimeError(f"RoboDojo root not given; pass robodojo_root or set {ROOT_ENV}")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"RoboDojo root {path} is not a directory")
    return path


def _check_task(task: str) -> None:
    if not isinstance(task, str) or not _TASK_NAME.fullmatch(task):
        raise ValueError(f"invalid RoboDojo task name {task!r}")


def layout_files(task: str, bank: int, root: Optional[PathLike] = None) -> list[Path]:
    """Layout files of one bank in native ordinal order."""
    _check_task(task)
    if bank not in BANKS:
        raise ValueError(f"layout bank must be one of {BANKS}, got {bank!r}")
    directory = resolve_root(root) / LAYOUT_SUBDIR / str(bank)
    if not directory.is_dir():
        return []
    pattern = re.compile(rf"{re.escape(task)}_(\d+)\.json")
    numbered = []
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    return [path for _, path in sorted(numbered, key=lambda item: item[0])]


def encode_seed(bank: int, ordinal: int) -> int:
    if bank not in BANKS or not isinstance(ordinal, int) or not 0 <= ordinal < SEED_STRIDE:
        raise ValueError(f"invalid layout identity bank={bank!r} ordinal={ordinal!r}")
    return bank * SEED_STRIDE + ordinal


def decode_scene(task: str, seed: int, root: Optional[PathLike] = None) -> tuple[int, int, Path]:
    """(bank, ordinal, layout path) for a harness seed; raises if the layout does not exist."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    bank, ordinal = divmod(seed, SEED_STRIDE)
    if bank not in BANKS:
        raise ValueError(f"seed {seed} decodes to bank {bank}; banks are {BANKS}")
    files = layout_files(task, bank, root)
    if ordinal >= len(files):
        raise ValueError(f"no {task} layout at bank {bank} ordinal {ordinal} ({len(files)} available)")
    return bank, ordinal, files[ordinal]


def available_scenes(task: str, root: Optional[PathLike] = None) -> list[int]:
    """Every harness seed with a layout for this task, bank by bank."""
    return [encode_seed(bank, n) for bank in BANKS for n in range(len(layout_files(task, bank, root)))]
