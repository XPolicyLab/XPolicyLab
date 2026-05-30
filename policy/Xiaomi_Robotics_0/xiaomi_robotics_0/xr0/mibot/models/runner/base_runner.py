# Copyright (C) 2026 Xiaomi Corporation.
from importlib import import_module
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
from lightning import LightningModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from transformers.utils import logging

from mibot.models import MIMODEL

logger = logging.get_logger(__name__)


@MIMODEL.register_module()
class BaseRunner(LightningModule):
    def __init__(self, params: Dict[str, Any]):
        """
        Initializes the BaseRunner module.

        Args:
            params (Dict[str, Any]): A dictionary containing configurations for the model,
                                    optimizer, scheduler, and optionally pretrained weights.
        """
        super().__init__()
        self._pretrained: Optional[str] = params.get("pretrained")
        self._model: Dict[str, Any] = params.get("model")
        self._optimizer: Dict[str, Any] = params.get("optimizer")
        self._scheduler: Optional[Dict[str, Any]] = params.get("scheduler")

    def configure_model(self) -> None:
        """
        Builds the model architecture and loads pretrained weights if provided.
        """
        self.model = MIMODEL.build(self._model)

        if self._pretrained:
            ckpt = torch.load(self._pretrained, map_location="cpu")
            info = self.load_state_dict(ckpt["module"], strict=False)
            logger.info(f"Loaded pretrained model from {self._pretrained}: {info}")

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            Dict[str, Any]: A dictionary containing the optimizer and scheduler configuration.
        """
        optimizer = self.build_optimizer(self._optimizer, self.named_parameters())
        scheduler = self.build_scheduler(self._scheduler, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @staticmethod
    def build_optimizer(cfg: Dict[str, Any], parameters: Iterator[Tuple[str, torch.Tensor]]) -> Optimizer:
        """
        Builds an optimizer with weight decay applied conditionally based on parameter names.

        Args:
            cfg (Dict[str, Any]): Optimizer configuration dictionary, including 'type' and 'params'.
            parameters (Iterator[Tuple[str, torch.Tensor]]): Model parameters with their names.

        Returns:
            Optimizer: Constructed optimizer instance.
        """
        module_params = cfg.get("params", {})
        parameters = list(parameters)

        no_decay = [
            "bias",
            "norm",
            "Norm",
            "ln",
            "Ln",
            "rotary_emb",
            "adaln",
        ]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in parameters if not any(nd in n.lower() for nd in no_decay) and p.requires_grad],
                "weight_decay": module_params.get("weight_decay", 0.1),
            },
            {
                "params": [p for n, p in parameters if any(nd in n.lower() for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

        module_type = cfg["type"]
        module_name, class_name = module_type.rsplit(".", 1)
        module = import_module(module_name)
        optimizer_class = getattr(module, class_name)

        return optimizer_class(optimizer_grouped_parameters, **module_params)

    @staticmethod
    def build_scheduler(cfg: Optional[Dict[str, Any]], optimizer: Optimizer) -> Optional[LRScheduler]:
        """
        Builds a learning rate scheduler from the given configuration.

        Args:
            cfg (Optional[Dict[str, Any]]): Scheduler configuration dictionary.
            optimizer (Optimizer): The optimizer to be used with the scheduler.

        Returns:
            Optional[LRScheduler]: Constructed learning rate scheduler, or None if no config is given.
        """
        if cfg is None:
            return None

        module_type = cfg["type"]
        module_params = cfg.get("params", {})
        module_name, class_name = module_type.rsplit(".", 1)
        module = import_module(module_name)
        scheduler_class = getattr(module, class_name)

        return scheduler_class(optimizer, **module_params)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Performs a single training step.

        Args:
            batch (Dict[str, Any]): A dictionary containing input tensors for the model.
            batch_idx (int): Index of the current batch.

        Returns:
            Dict[str, torch.Tensor]: Loss dictionary for the current training step.
        """
        self.log("train/token", batch["input_ids"].shape[1])
        loss_dict = self.model(batch, return_loss=True)
        for loss_name, loss_value in loss_dict.items():
            self.log(f"train/{loss_name}", loss_value.detach())
        self.log("lr", self.optimizers().param_groups[0]["lr"])
        return loss_dict

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        Performs a single validation step.

        Args:
            batch (Dict[str, Any]): A dictionary containing input tensors for the model.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int, optional): Index of the dataloader, useful when using multiple val datasets.

        Returns:
            Dict[str, torch.Tensor]: Loss dictionary for the current validation step.
        """
        loss_dict = self.model(batch, return_loss=True)
        for loss_name, loss_value in loss_dict.items():
            self.log(f"val/{loss_name}", loss_value.detach(), sync_dist=True)
        return loss_dict
