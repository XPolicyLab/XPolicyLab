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
    """Process and finalize the training configuration.

    Resolves the OmegaConf config, sets the random seed (offset by RANK for
    multi-process determinism), configures the training strategy, and injects
    optimizer/scheduler configs into the model params.

    Args:
        cfg: Raw Hydra/OmegaConf configuration object.

    Returns:
        Fully resolved ``mmengine.Config`` ready for trainer construction.
    """
    # Resolve config
    cfg = Config(OmegaConf.to_container(cfg, resolve=True))
    process_save_cfg(cfg)

    # Set random seed for reproducibility (offset by RANK so each process differs)
    seed_everything(cfg.trainer.pop("seed", 42) + int(os.environ.get("RANK", 0)), workers=True)

    # Configure training strategy
    strategy_helper(cfg)

    # Inject optimizer & scheduler into model params so BaseRunner can access them
    cfg.model.params.optimizer = cfg.trainer.pop("optimizer")
    cfg.model.params.scheduler = cfg.trainer.pop("scheduler")

    return cfg


def strategy_helper(cfg: Config) -> Config:
    """Instantiate and attach the distributed training strategy.

    Currently only DeepSpeed is supported.

    Args:
        cfg: The configuration object to modify in-place.

    Returns:
        The configuration object with ``cfg.trainer.strategy`` replaced by
        an instantiated strategy object.
    """
    strategy_type: str = cfg.trainer.strategy.type
    if strategy_type == "deepspeed":
        strategy = DeepSpeedStrategy(**cfg.trainer.strategy.params)
    else:
        raise TypeError("Unsupported strategy.")
    cfg.trainer.strategy = strategy
    return cfg


def fill_num_nodes(cfg: Config) -> None:
    """Ensure ``cfg.trainer.num_nodes`` is a positive integer.

    If the configured value is <= 0, it is auto-detected from the
    ``MLP_WORKER_NUM`` environment variable.

    Args:
        cfg: The configuration object to modify in-place.
    """
    if cfg.trainer.num_nodes > 0:
        assert cfg.trainer.num_nodes <= int(os.environ.get("MLP_WORKER_NUM", 1)), (
            "Number of nodes exceeds available workers"
        )
    else:
        cfg.trainer.num_nodes = int(os.environ.get("MLP_WORKER_NUM", 1))


def process_save_cfg(cfg: Config) -> None:
    """Create the output directory, persist the resolved config, and expose
    ``max_steps`` via an environment variable for downstream data code.

    The config is dumped in three locations:
      - ``<root_dir>/config.yaml`` — human-readable YAML
      - ``<root_dir>/config.py``   — Python-readable mmengine format
      - ``./assets/config.py``     — convenience copy for deployment scripts

    Args:
        cfg: The configuration object to modify and save.
    """
    root_dir = os.path.join(
        cfg.trainer.default_root_dir,
        f"project_{cfg.trainer.project}",
        cfg.trainer.exp_name,
    )
    cfg.trainer.default_root_dir = root_dir

    fill_num_nodes(cfg)

    # Expose max_steps so dataset code can read it if needed
    os.environ["_max_steps"] = str(cfg.trainer.max_steps)

    os.makedirs(root_dir, exist_ok=True)
    cfg.dump(os.path.join(root_dir, "config.yaml"))
    cfg.dump(os.path.join(root_dir, "config.py"))
    cfg.dump(os.path.join("./assets", "config.py"))
