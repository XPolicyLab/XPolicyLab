"""Host-only identities for RoboDojo's finite, prerecorded layout banks.

A harness seed encodes ``bank * 1_000_000 + ordinal``. Ordinals follow the
native SeedManager's numeric filename ordering, not the filename suffix.
Only enumerate the Test pool after the harness has been frozen.
"""

from __future__ import annotations

import re

from simulators.vendor import REPO

ROOT = REPO / "vendor" / "robodojo"
STRIDE = 1_000_000


def layout_files(task_name, bank):
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", task_name):
        raise ValueError("RoboDojo requires a native task name")
    directory = ROOT / "Assets/Eval_Layout/RoboDojo/arx_x5" / str(bank)
    pattern = re.compile(rf"{re.escape(task_name)}_(\d+)\.json")
    paths = [p for p in directory.glob(f"{task_name}_*.json") if pattern.fullmatch(p.name)]
    return sorted(paths, key=lambda p: int(p.stem.rsplit("_", 1)[1]))


def decode_scene(task_name, seed):
    if type(seed) is not int or seed < 0:
        raise ValueError("RoboDojo requires a nonnegative encoded layout identity")
    bank, ordinal = divmod(seed, STRIDE)
    if bank not in (0, 1, 2):
        raise ValueError("RoboDojo layout bank must be 0, 1, or 2")
    paths = layout_files(task_name, bank)
    if ordinal >= len(paths):
        raise ValueError(f"No RoboDojo {task_name} layout at bank={bank}, ordinal={ordinal}")
    return bank, ordinal, paths[ordinal]


def available_scenes(task_name):
    return [
        bank * STRIDE + ordinal
        for bank in (0, 1, 2)
        for ordinal in range(len(layout_files(task_name, bank)))
    ]
