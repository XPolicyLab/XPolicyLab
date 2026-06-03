# Copyright (C) 2026 Xiaomi Corporation.
"""Training entry point for the MiBot framework.

Uses Hydra for config management and PyTorch Lightning for training.
Configures the model, data module, trainer, and launches training.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import hydra
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import WandbLogger

from mmengine import Config, DATASETS
from omegaconf import DictConfig
from mibot.models import MIMODEL

from mibot.utils.cfg_utils import helper

import mibot.data  # noqa: F401 - trigger dataset registration

warnings.filterwarnings("ignore")


def prepare(cfg: Dict[str, Any]) -> Tuple[Config, LightningDataModule, LightningModule, List[Any]]:
    """Prepare the training environment: resolve config, build modules and trainer.

    Steps:
        1. Resolve the OmegaConf config and set seeds / strategy.
        2. Build the data module from the registry.
        3. Build the model runner from the registry.
        4. Configure callbacks (ModelSummary, ModelCheckpoint) and logger (Wandb).

    Args:
        cfg: Raw Hydra/OmegaConf configuration.

    Returns:
        Tuple of (resolved_config, datamodule, model, logger_list).
    """
    ######################### Process config ###########################
    cfg: Config = helper(cfg)

    ########################## Build modules ###########################
    # dataset
    datamodule: LightningDataModule = DATASETS.build(cfg.data)

    # model
    model: LightningModule = MIMODEL.build(cfg.model)

    ########################## Build trainer ###########################
    # callbacks
    cfg.trainer["callbacks"] = [
        ModelSummary(max_depth=2),
        ModelCheckpoint(
            save_top_k=-1,
            save_last=True,
            every_n_train_steps=cfg.trainer.pop("save_interval", 10000),
            dirpath=cfg.trainer.default_root_dir,
            enable_version_counter=False,
        ),
    ]

    # logger
    logger: List[Any] = [
        WandbLogger(
            project=cfg.trainer.pop("project"),
            name=cfg.trainer.pop("exp_name"),
            config=cfg,
        )
    ]

    logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

    return (cfg, datamodule, model, logger)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point: prepare and launch training."""
    cfg, datamodule, model, logger = prepare(cfg)
    ckpt_path: Optional[str] = cfg.trainer.pop("ckpt_path", None)
    trainer = Trainer(**cfg.trainer, logger=logger)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
