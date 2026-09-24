# Copyright (C) 2026 Xiaomi Corporation.
import os
from typing import Any

from lightning import seed_everything
from lightning.pytorch.strategies import DeepSpeedStrategy
from mmengine import Config
from omegaconf import DictConfig, OmegaConf
from transformers.utils import logging

logger = logging.get_logger(__name__)


def helper(cfg: DictConfig) -> Config:
    cfg = Config(OmegaConf.to_container(cfg, resolve=True))
    process_save_cfg(cfg)
    seed_everything(cfg.trainer.pop("seed", 42) + int(os.environ.get("RANK", 0)), workers=True)
    strategy_helper(cfg)
    cfg.model.params.optimizer = cfg.trainer.pop("optimizer")
    cfg.model.params.scheduler = cfg.trainer.pop("scheduler")

    return cfg


def strategy_helper(cfg: Config) -> Config:
    strategy_type: str = cfg.trainer.strategy.type
    if strategy_type == "deepspeed":
        strategy = DeepSpeedStrategy(**cfg.trainer.strategy.params)
    else:
        raise TypeError("Unsupported strategy.")
    cfg.trainer.strategy = strategy
    return cfg


def fill_num_nodes(cfg: Config) -> None:
    if cfg.trainer.num_nodes > 0:
        assert cfg.trainer.num_nodes <= int(os.environ.get("MLP_WORKER_NUM", 1)), (
            "Number of nodes exceeds available workers"
        )
    else:
        cfg.trainer.num_nodes = int(os.environ.get("MLP_WORKER_NUM", 1))


def process_save_cfg(cfg: Config) -> None:
    root_dir = os.path.join(
        cfg.trainer.default_root_dir,
        f"project_{cfg.trainer.project}",
        cfg.trainer.exp_name,
    )
    cfg.trainer.default_root_dir = root_dir

    fill_num_nodes(cfg)

    os.environ["_max_steps"] = str(cfg.trainer.max_steps)

    os.makedirs(root_dir, exist_ok=True)
    cfg.dump(os.path.join(root_dir, "config.yaml"))
    cfg.dump(os.path.join(root_dir, "config.py"))
    cfg.dump(os.path.join("./assets", "config.py"))
