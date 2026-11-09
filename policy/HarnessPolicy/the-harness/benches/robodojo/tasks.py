"""Pure RoboDojo task catalog and per-task config resolution.

This module must stay importable without Isaac, torch, numpy, or YAML. The
general worker in ``general.py`` uses it before it launches Kit, so a wrong
task name or layout id fails without touching the simulator.
"""

import difflib
from pathlib import Path
import re


BENCHMARK = "RoboDojo"
DEFAULT_TASK = "stack_bowls"
DEFAULT_SCENE = "default"
DEFAULT_ROBOT = "dual_x5"
# Every robot config whose policy-controlled arms are the dual X5. The
# competition config adds a scripted Franka support arm that the policy does not drive.
ROBOT_CONFIGS = ("dual_x5", "dual_x5_and_franka_competition")
SCENE_CONFIGS = ("default", "conveyor")
LAYOUT_PACK_SEEDS = (0, 1, 2)


def task_catalog(checkout):
    """Task names with a config file under ``task/RoboDojo/config``.

    Names starting with ``_`` (``_task.yml``) are shared settings, not tasks.
    """
    directory = Path(checkout) / "task" / BENCHMARK / "config"
    if not directory.is_dir():
        raise RuntimeError(f"RoboDojo task config directory not found at {directory}")
    return sorted(path.stem for path in directory.glob("*.yml") if not path.stem.startswith("_"))


def validate_task(name, catalog):
    if not isinstance(name, str) or not name:
        raise ValueError("task must be a non-empty string")
    if name not in catalog:
        close = difflib.get_close_matches(name, catalog, n=3)
        hint = f" Did you mean {', '.join(close)}?" if close else ""
        raise ValueError(f"Unknown RoboDojo task {name!r}.{hint}")
    return name


def task_settings(task, index):
    """Resolve the ``_task.yml`` overrides that ``process_config`` applies.

    ``index`` is the parsed ``_task.yml``: ``{"common": {...}, "tasks": {name: {...}}}``.
    Upstream replaces ``env_cfg.scene`` and ``env_cfg.robot`` but leaves
    ``eval_cfg.config`` untouched, so the effective names are only recoverable here.
    """
    if not isinstance(index, dict):
        raise ValueError("RoboDojo task index must be a mapping")
    common = index.get("common") or {}
    tasks = index.get("tasks") or {}
    specific = tasks.get(task) or {}

    def setting(key, default):
        return specific.get(key, common.get(key, default))

    scene = str(setting("scene_config", DEFAULT_SCENE))
    robot = str(setting("robot_config", DEFAULT_ROBOT))
    validate_robot_config(robot)
    if scene not in SCENE_CONFIGS:
        raise ValueError(f"Unsupported RoboDojo scene config {scene!r}")
    return {
        "scene_config": scene,
        "robot_config": robot,
        "data_source": str(setting("data_source", "datagen")),
    }


def validate_robot_config(name):
    if name not in ROBOT_CONFIGS:
        raise ValueError(f"Unsupported RoboDojo robot config {name!r}; expected one of {list(ROBOT_CONFIGS)}")
    return name


def validate_layout_pack_seed(value):
    if type(value) is not int or value not in LAYOUT_PACK_SEEDS:
        raise ValueError(f"layout_pack_seed must be one of {list(LAYOUT_PACK_SEEDS)}")
    return value


def layout_dir(checkout, env_config, layout_pack_seed):
    return Path(checkout) / "Assets" / "Eval_Layout" / BENCHMARK / env_config / str(layout_pack_seed)


def layout_count(directory, task):
    """Number of ``<task>_<n>.json`` layouts, matching ``SeedManager.init_eval``."""
    directory = Path(directory)
    if not directory.is_dir():
        raise RuntimeError(f"RoboDojo layout directory not found at {directory}")
    pattern = re.compile(rf"{re.escape(task)}_\d+\.json")
    return sum(1 for path in directory.iterdir() if pattern.fullmatch(path.name))


def validate_layout(seed, count):
    """The harness seed is the layout id inside one eval-seed directory."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(count) is not int or count < 1:
        raise ValueError("Task has no evaluation layouts")
    if seed >= count:
        raise ValueError(f"seed {seed} is outside the {count} layouts for this task")
    return seed
