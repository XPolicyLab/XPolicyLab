"""Load and check PrismWAM training recipes.

A recipe is a YAML file under ``configs/``. It may list parent files under
``defaults``; parents are merged first (in order) and the recipe's own keys win.
``key=value`` overrides from the command line are applied last::

    cfg = load_config("configs/robotwin/stage1_depth.yaml", ["paths.data_root=/data"])
"""

from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig, OmegaConf

# Keys that only organise the YAML files; they are not training options.
COMPOSITION_KEYS = ("defaults",)

# Fields that name files or directories the recipe reads.
_ASSET_KEYS = {
    "root_dir",
    "checkpoint_path",
    "tokenizer_path",
    "text_encoder_path",
    "text_embedding_cache_dir",
    "pretrained_norm_stats",
    "task_instruction_dir",
}
_ASSET_SUFFIXES = (
    "_checkpoint_path",
    "_index_path",
    "_stats_cache_path",
    "_pretrained_path",
)


def _compose(path: Path, chain: tuple[Path, ...] = ()) -> DictConfig:
    path = path.resolve()
    if path in chain:
        raise ValueError(f"Config inheritance cycle: {' -> '.join(map(str, chain))}")
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"{path} must contain a mapping.")
    parents = cfg.pop("defaults", None) or []
    if isinstance(parents, str):
        parents = [parents]
    merged = OmegaConf.create()
    for parent in parents:
        merged = OmegaConf.merge(merged, _compose(path.parent / parent, (*chain, path)))
    return OmegaConf.merge(merged, cfg)


def load_config(
    path: str | Path, overrides: Iterable[str] = (), *, resolve: bool = True
) -> DictConfig:
    """Compose a recipe with its parents and apply ``key=value`` overrides."""
    cfg = _compose(Path(path).expanduser())
    overrides = list(overrides)
    if any("=" not in item or not item.split("=", 1)[0] for item in overrides):
        raise ValueError(f"Overrides must use key=value: {overrides}")
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    if resolve:
        OmegaConf.resolve(cfg)
    return cfg


def gradient_accumulation_steps(cfg, world_size: int) -> int:
    """Accumulation that keeps ``global_batch_size`` on ``world_size`` GPUs."""
    batch = int(cfg.batch_size)
    world_size = int(world_size)
    if batch <= 0 or world_size <= 0:
        raise ValueError("batch_size and the number of GPUs must be positive.")
    global_batch = cfg.get("global_batch_size")
    if global_batch is None:
        return int(cfg.get("gradient_accumulation_steps", 1))
    per_step = batch * world_size
    if int(global_batch) % per_step:
        raise ValueError(
            f"global_batch_size={global_batch} is not a multiple of batch_size={batch} "
            f"x {world_size} GPUs. Pick a batch_size that divides "
            f"{int(global_batch) // world_size}, or change the number of GPUs."
        )
    return int(global_batch) // per_step


def find_missing_assets(cfg) -> list[dict[str, str]]:
    """Files and directories named by the recipe that do not exist yet."""
    missing = []

    def walk(value, key=""):
        if isinstance(value, dict):
            for name, item in value.items():
                walk(item, name)
        elif isinstance(value, list):
            for item in value:
                walk(item, key)
        elif (
            isinstance(value, str)
            and value
            and (key in _ASSET_KEYS or key.endswith(_ASSET_SUFFIXES))
            and not Path(value).expanduser().exists()
        ):
            missing.append({"field": key, "path": value})

    walk(OmegaConf.to_container(cfg, resolve=True))
    return missing


def check_recipe(cfg, *, world_size: int | None = None, assets: bool = False) -> dict:
    """Validate a resolved recipe without loading weights or allocating GPUs.

    Every ``_target_`` must be a PrismWAM constructor that accepts the configured
    keyword arguments. With ``assets=True``, every referenced path must exist;
    run it again for Stage 2 once Stage 1 has produced its checkpoint.
    """
    raw = OmegaConf.to_container(cfg, resolve=True)
    if raw.get("max_steps", 0) <= 0 or raw.get("batch_size", 0) <= 0:
        raise ValueError("max_steps and batch_size must be positive.")
    targets = []

    def walk(value):
        if isinstance(value, dict):
            target = value.get("_target_")
            if target is not None:
                if not isinstance(target, str) or not target.startswith("prismwam."):
                    raise ValueError(f"Unexpected _target_: {target}")
                module_name, name = target.rsplit(".", 1)
                constructor = getattr(importlib.import_module(module_name), name)
                kwargs = {k: v for k, v in value.items() if not k.startswith("_")}
                inspect.signature(constructor).bind_partial(**kwargs)
                targets.append(target)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(raw)
    if raw["model"].get("share_rgb_depth_blocks") is False:
        raise ValueError("PrismWAM shares the transformer blocks of RGB and depth.")
    world = int(world_size or os.environ.get("WORLD_SIZE", 1))
    accumulation = gradient_accumulation_steps(cfg, world)
    report = {
        "targets": sorted(set(targets)),
        "world_size": world,
        "batch_size": raw["batch_size"],
        "gradient_accumulation_steps": accumulation,
        "global_batch_size": raw["batch_size"] * accumulation * world,
        "max_steps": raw["max_steps"],
        "output_dir": raw["output_dir"],
    }
    if assets:
        report["missing_assets"] = find_missing_assets(cfg)
    return report
