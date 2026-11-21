"""Fixed scene splits per task: explore / validate from layout bank 0 (disjoint), test-dev from banks 1-2.

Validation and version selection must call ``assert_not_test_dev`` (or ``selection_scenes``): any seed in
the task's test-dev split, or from banks 1-2 at all, is refused.

Generate with ``python -m knowin_agent.experiments.splits --root <RoboDojo> --out splits.json`` (or
``--bank-sizes 20,10,10`` without a checkout).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

from knowin_agent.backends.robodojo.layouts import (BANKS, LAYOUT_SUBDIR, SEED_STRIDE, available_scenes, encode_seed,
                                            resolve_root)

EXPLORE, VALIDATE, TEST_DEV = "explore", "validate", "test-dev"
SPLITS = (EXPLORE, VALIDATE, TEST_DEV)
SELECTION_SPLITS = (EXPLORE, VALIDATE)
DEFAULT_COUNTS = {EXPLORE: 6, VALIDATE: 3, TEST_DEV: 3}
DEFAULT_RNG_SEED = 20261028
SCHEMA = "knowin_agent.splits/1"
SPLITS_ENV = "KNOWIN_AGENT_SPLITS"
DEFAULT_PATH = Path(__file__).with_name("splits.json")
RULE = ("explore and validate come from layout bank 0 and are disjoint; test-dev comes from banks 1-2; "
        "validation and version selection never use test-dev or banks 1-2")

SplitsArg = Union[None, str, os.PathLike, Mapping]


class SplitViolation(RuntimeError):
    pass


def bank_of(seed: int) -> int:
    return int(seed) // SEED_STRIDE


def bank_sizes_from_root(tasks: Sequence[str], root) -> dict[str, list[int]]:
    """Number of layouts per bank for each task, from a RoboDojo checkout."""
    sizes = {}
    for task in tasks:
        counts = [0] * len(BANKS)
        for seed in available_scenes(task, root):
            counts[bank_of(seed)] += 1
        sizes[task] = counts
    return sizes


def discover_tasks(root) -> list[str]:
    """Tasks with at least one bank-0 layout in a RoboDojo checkout (``<task>_<n>.json``)."""
    directory = resolve_root(root) / LAYOUT_SUBDIR / str(BANKS[0])
    pattern = re.compile(r"([A-Za-z][A-Za-z0-9_]*)_\d+\.json")
    names = {m.group(1) for p in directory.iterdir() if (m := pattern.fullmatch(p.name))} if directory.is_dir() else set()
    return sorted(names)


def _split_task(task: str, sizes: Sequence[int], counts: Mapping[str, int], rng_seed: int) -> dict[str, list[int]]:
    rng = random.Random(f"{rng_seed}:{task}")
    bank0 = list(range(sizes[0]))
    rng.shuffle(bank0)
    need = counts[EXPLORE] + counts[VALIDATE]
    if len(bank0) < need:
        raise ValueError(f"{task}: bank 0 has {len(bank0)} layouts, explore + validate need {need}")
    later = []
    for bank in BANKS[1:]:
        ordinals = list(range(sizes[bank]))
        rng.shuffle(ordinals)
        later.append([encode_seed(bank, n) for n in ordinals])
    interleaved = [group[i] for i in range(max(len(g) for g in later)) for group in later if i < len(group)]
    if len(interleaved) < counts[TEST_DEV]:
        raise ValueError(f"{task}: banks 1-2 have {len(interleaved)} layouts, test-dev needs {counts[TEST_DEV]}")
    seeds0 = [encode_seed(0, n) for n in bank0]
    return {EXPLORE: sorted(seeds0[:counts[EXPLORE]]),
            VALIDATE: sorted(seeds0[counts[EXPLORE]:need]),
            TEST_DEV: sorted(interleaved[:counts[TEST_DEV]])}


def build_splits(tasks: Optional[Sequence[str]] = None, *, root=None,
                 bank_sizes: Union[None, Sequence[int], Mapping[str, Sequence[int]]] = None,
                 counts: Optional[Mapping[str, int]] = None, rng_seed: int = DEFAULT_RNG_SEED) -> dict:
    """Deterministic splits. Layout counts come from ``root`` (a RoboDojo checkout) or ``bank_sizes``
    (one ``(n0, n1, n2)`` for every task, or per task)."""
    counts = {**DEFAULT_COUNTS, **(counts or {})}
    if set(counts) != set(SPLITS) or any(not isinstance(v, int) or v < 1 for v in counts.values()):
        raise ValueError(f"counts must give a positive integer for each of {SPLITS}, got {counts}")
    if (root is None) == (bank_sizes is None):
        raise ValueError("give exactly one of root (RoboDojo checkout) or bank_sizes")
    if root is not None:
        tasks = list(tasks) if tasks else discover_tasks(root)
        sizes = bank_sizes_from_root(tasks, root)
        source = "robodojo_root"
    else:
        if isinstance(bank_sizes, Mapping):
            tasks = list(tasks) if tasks else sorted(bank_sizes)
            sizes = {t: list(bank_sizes[t]) for t in tasks}
        else:
            if not tasks:
                from knowin_agent.knowledge import task_names

                tasks = list(task_names())
            sizes = {t: list(bank_sizes) for t in tasks}
        source = "bank_sizes"
    if not tasks:
        raise ValueError("no tasks to split")
    for task, value in sizes.items():
        if len(value) != len(BANKS) or any(not isinstance(n, int) or n < 0 for n in value):
            raise ValueError(f"{task}: bank sizes must be {len(BANKS)} non-negative integers, got {value}")
    data = {"schema": SCHEMA, "rule": RULE, "seed_encoding": f"bank * {SEED_STRIDE} + ordinal",
            "rng_seed": rng_seed, "counts": counts, "source": source, "bank_sizes": sizes,
            "tasks": {t: _split_task(t, sizes[t], counts, rng_seed) for t in sorted(tasks)}}
    check_splits(data)
    return data


def check_splits(data: Mapping) -> None:
    """Raise SplitViolation unless every task's splits are disjoint and respect the bank rule."""
    if data.get("schema") != SCHEMA:
        raise SplitViolation(f"not a {SCHEMA} document")
    for task, splits in data["tasks"].items():
        if set(splits) != set(SPLITS):
            raise SplitViolation(f"{task}: splits must be exactly {SPLITS}")
        seen: dict[int, str] = {}
        for name in SPLITS:
            for seed in splits[name]:
                if seed in seen:
                    raise SplitViolation(f"{task}: seed {seed} is in both {seen[seed]} and {name}")
                seen[seed] = name
                bank = bank_of(seed)
                if (name == TEST_DEV) != (bank != 0) or bank not in BANKS:
                    raise SplitViolation(f"{task}: seed {seed} (bank {bank}) cannot be in {name}")


def save_splits(data: Mapping, path: Union[str, os.PathLike]) -> Path:
    check_splits(data)
    path = Path(path)
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return path


def load_splits(splits: SplitsArg = None) -> dict:
    """A splits document from a mapping, a path, $KNOWIN_AGENT_SPLITS, or the packaged splits.json (checked)."""
    if isinstance(splits, Mapping):
        data = dict(splits)
    else:
        path = Path(splits or os.environ.get(SPLITS_ENV) or DEFAULT_PATH)
        if not path.is_file():
            raise SplitViolation(f"splits file {path} not found; generate it with python -m knowin_agent.experiments.splits")
        data = json.loads(path.read_text(encoding="utf-8"))
    check_splits(data)
    return data


def _task_splits(task: str, splits: SplitsArg) -> dict:
    data = load_splits(splits)
    if task not in data["tasks"]:
        raise SplitViolation(f"task {task!r} has no splits")
    return data["tasks"][task]


def scenes_for(split: str, task: str, splits: SplitsArg = None) -> list[int]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    return list(_task_splits(task, splits)[split])


def split_of(seed: int, task: str, splits: SplitsArg = None) -> Optional[str]:
    for name, seeds in _task_splits(task, splits).items():
        if seed in seeds:
            return name
    return None


def assert_not_test_dev(seed: int, task: str, splits: SplitsArg = None) -> None:
    """Refuse a seed for validation / version selection: test-dev scenes and banks 1-2 are off limits."""
    if seed in _task_splits(task, splits)[TEST_DEV]:
        raise SplitViolation(f"{task} seed {seed} is a test-dev scene; validation and selection may not use it")
    if bank_of(seed) != 0:
        raise SplitViolation(f"{task} seed {seed} is from bank {bank_of(seed)}, reserved for test-dev")


def selection_scenes(task: str, splits: SplitsArg = None, split: str = VALIDATE) -> list[int]:
    """Scenes for validation or version selection (explore or validate only), each checked by the guard."""
    if split not in SELECTION_SPLITS:
        raise SplitViolation(f"selection may only use {SELECTION_SPLITS}, not {split!r}")
    data = load_splits(splits)
    seeds = scenes_for(split, task, data)
    for seed in seeds:
        assert_not_test_dev(seed, task, data)
    return seeds


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m knowin_agent.experiments.splits", description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", help="RoboDojo checkout (layout banks under Assets/Eval_Layout)")
    source.add_argument("--bank-sizes", help="layouts per bank for every task, e.g. 20,10,10")
    parser.add_argument("--tasks", help="comma-separated tasks (default: discovered from the root, else all 42)")
    parser.add_argument("--explore", type=int, default=DEFAULT_COUNTS[EXPLORE])
    parser.add_argument("--validate", type=int, default=DEFAULT_COUNTS[VALIDATE])
    parser.add_argument("--test-dev", type=int, default=DEFAULT_COUNTS[TEST_DEV])
    parser.add_argument("--rng-seed", type=int, default=DEFAULT_RNG_SEED)
    parser.add_argument("--out", default=str(DEFAULT_PATH))
    args = parser.parse_args(argv)
    tasks = [t for t in (args.tasks or "").split(",") if t]
    sizes = [int(n) for n in args.bank_sizes.split(",")] if args.bank_sizes else None
    data = build_splits(tasks or None, root=args.root, bank_sizes=sizes, rng_seed=args.rng_seed,
                        counts={EXPLORE: args.explore, VALIDATE: args.validate, TEST_DEV: args.test_dev})
    path = save_splits(data, args.out)
    print(f"wrote {path}: {len(data['tasks'])} tasks, counts {data['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
